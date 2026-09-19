"""Build the region training sample by STREAMING the silver dataset.

The full set is 5.25M rows x 699 features. Dense that is ~17 GB, on a 16 GB
machine with other work running, so nothing here ever holds more than one
50k-row chunk in dense form: each chunk is converted to CSR and the dense
buffer is reused. Peak RSS is reported by the caller through /usr/bin/time -l.

Sampling policy (from the brief):
  - masked rows only (m == 1); the unmasked twin is the tag baseline's input,
    never the model's;
  - `unlabelled` excluded entirely;
  - every row of every rare class kept;
  - content_block downsampled to at most CONTENT_BLOCK_RATIO x the largest
    other class;
  - total capped at MAX_TRAIN_ROWS.

Selection is deterministic: a row is kept when a hash of (pageId, nodeId) falls
under the class's rate, so two runs pick the same rows and a rerun after a
crash is not a different experiment.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp

CHUNK_ROWS = 50_000
MAX_TRAIN_ROWS = 600_000
CONTENT_BLOCK_RATIO = 3.0

# The 10 product classes. article_footer folds into article_header as
# `article_chrome`: both are a section's own chrome, the distinction does not
# change any fix, and article_footer has 1,194 training nodes on its own.
REGION_CLASSES = [
    "site_header",
    "footer",
    "navigation",
    "sidebar",
    "main_content",
    "content_block",
    "article_chrome",
    "form",
    "search",
    "dialog",
]
CLASS_INDEX = {name: i for i, name in enumerate(REGION_CLASSES)}
COLLAPSE = {"article_header": "article_chrome", "article_footer": "article_chrome"}


def collapse(label: str) -> str | None:
    """Product class for a silver label, or None when the row is not trained on."""
    if label == "unlabelled":
        return None
    return COLLAPSE.get(label, label)


def keep_hash(page_id: str, node_id: int) -> float:
    """Deterministic uniform in [0, 1) for one node."""
    digest = hashlib.sha256(f"{page_id}:{node_id}".encode()).digest()
    return int.from_bytes(digest[:6], "big") / float(1 << 48)


def rows(path: Path):
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line:
                yield json.loads(line)


def count_labels(path: Path) -> Counter:
    counts: Counter = Counter()
    for row in rows(path):
        if row["m"] != 1:
            continue
        label = collapse(row["l"])
        if label is not None:
            counts[label] += 1
    return counts


def sampling_rates(counts: Counter) -> dict[str, float]:
    """One rate per class. Everything except content_block is kept whole."""
    others = {k: v for k, v in counts.items() if k != "content_block"}
    largest_other = max(others.values()) if others else 0
    rates = dict.fromkeys(counts, 1.0)
    target = int(CONTENT_BLOCK_RATIO * largest_other)
    have = counts.get("content_block", 0)
    if have > target > 0:
        rates["content_block"] = target / have

    # Global cap, applied to content_block first and only then across the board,
    # so the cap never eats a rare class before it touches the abundant one.
    projected = sum(counts[name] * rates[name] for name in counts)
    if projected > MAX_TRAIN_ROWS:
        surplus = projected - MAX_TRAIN_ROWS
        cb = counts.get("content_block", 0) * rates.get("content_block", 0)
        if cb > 0:
            take = min(surplus, cb * 0.95)
            rates["content_block"] *= (cb - take) / cb
            surplus -= take
        if surplus > 0:
            remaining = sum(counts[n] * rates[n] for n in counts)
            scale = max(0.0, (remaining - surplus) / remaining)
            for name in rates:
                if name != "content_block":
                    rates[name] *= scale
    return rates


class ChunkedBuilder:
    """Accumulates rows as CSR, one reusable dense chunk at a time."""

    def __init__(self, n_numeric: int, n_tokens: int) -> None:
        self.width = n_numeric + n_tokens
        self.n_numeric = n_numeric
        self.buffer = np.zeros((CHUNK_ROWS, self.width), dtype=np.float32)
        self.fill = 0
        self.blocks: list[sp.csr_matrix] = []
        self.labels: list[int] = []
        self.groups: list[str] = []
        self.refs: list[tuple[str, int]] = []

    def add(self, features: list[float], tokens: list[int], label: int, domain: str,
            page_id: str, node_id: int) -> None:
        row = self.buffer[self.fill]
        row[: self.n_numeric] = features
        for bucket in tokens:
            row[self.n_numeric + bucket] = 1.0
        self.labels.append(label)
        self.groups.append(domain)
        self.refs.append((page_id, node_id))
        self.fill += 1
        if self.fill == CHUNK_ROWS:
            self.flush()

    def flush(self) -> None:
        if self.fill == 0:
            return
        self.blocks.append(sp.csr_matrix(self.buffer[: self.fill]))
        self.buffer[: self.fill] = 0.0
        self.fill = 0

    def finish(self):
        self.flush()
        matrix = sp.vstack(self.blocks, format="csr") if self.blocks else sp.csr_matrix(
            (0, self.width), dtype=np.float32
        )
        self.blocks = []
        return matrix, np.asarray(self.labels, dtype=np.int8), self.groups, self.refs


def build_split(path: Path, rates: dict[str, float], n_numeric: int, n_tokens: int):
    builder = ChunkedBuilder(n_numeric, n_tokens)
    kept: Counter = Counter()
    for row in rows(path):
        if row["m"] != 1:
            continue
        label = collapse(row["l"])
        if label is None:
            continue
        rate = rates.get(label, 1.0)
        if rate < 1.0 and keep_hash(row["p"], row["n"]) >= rate:
            continue
        kept[label] += 1
        builder.add(row["f"], row["k"], CLASS_INDEX[label], row["d"], row["p"], row["n"])
    return (*builder.finish(), kept)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    dataset = data_root / "silver-v1"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    schema = json.loads((dataset / "feature-schema.json").read_text())
    n_numeric = schema["nodeFeatureCount"]
    n_tokens = schema["tokenBuckets"]

    print("counting train labels", flush=True)
    counts = count_labels(dataset / "nodes-train.jsonl.gz")
    rates = sampling_rates(counts)
    print(json.dumps({"counts": dict(counts), "rates": rates}, indent=2), flush=True)

    manifest: dict[str, object] = {
        "featurizerVersion": schema["featurizerVersion"],
        "classes": REGION_CLASSES,
        "trainLabelCounts": dict(counts),
        "samplingRates": rates,
        "maxTrainRows": MAX_TRAIN_ROWS,
        "contentBlockRatio": CONTENT_BLOCK_RATIO,
        "splits": {},
    }

    for split in ("train", "val", "holdout"):
        # Only the training split is subsampled. Val and holdout are kept whole,
        # so every reported number is over the real class distribution.
        split_rates = rates if split == "train" else {}
        matrix, labels, groups, refs, kept = build_split(
            dataset / f"nodes-{split}.jsonl.gz", split_rates, n_numeric, n_tokens
        )
        sp.save_npz(out / f"X-{split}.npz", matrix)
        np.save(out / f"y-{split}.npy", labels)
        (out / f"refs-{split}.jsonl").write_text(
            "".join(
                json.dumps({"p": p, "n": n, "d": d}) + "\n"
                for (p, n), d in zip(refs, groups, strict=True)
            )
        )
        by_class = Counter(REGION_CLASSES[int(i)] for i in labels)
        manifest["splits"][split] = {
            "rows": int(matrix.shape[0]),
            "nnz": int(matrix.nnz),
            "domains": len(set(groups)),
            "byClass": dict(by_class),
        }
        print(f"{split}: {matrix.shape[0]} rows, {matrix.nnz} nnz, {len(set(groups))} domains",
              flush=True)
        del matrix, labels, groups, refs, kept

    (out / "sample-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
