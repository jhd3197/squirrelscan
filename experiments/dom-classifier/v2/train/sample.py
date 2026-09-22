"""Build the region training sample by STREAMING the silver dataset.

The full set is 5.25M rows x 699 features. Dense that is ~17 GB, on a 16 GB
machine with other work running, so nothing here ever holds more than one
50k-row chunk in dense form: each chunk is converted to CSR and the dense
buffer is reused. Peak RSS is reported by the caller through /usr/bin/time -l.

Sampling policy:
  - BOTH the masked and unmasked row of each node, which is 50/50 by
    construction because the builder emits exactly one of each. Training on
    masked rows alone was a train/serve mismatch: the model never saw a
    landmark tag or role, while the runtime feeds it true, unmasked features.
    That is what produced <nav class="main-navigation"> -> content_block at
    0.74 on a stored site. The model has to learn BOTH "the tag says nav, so
    navigation" and the structural signal for when the tag is absent;
  - `unlabelled` excluded entirely;
  - every row of every rare class kept;
  - content_block downsampled to at most CONTENT_BLOCK_RATIO x the largest
    other class;
  - total capped at --max-train-rows (see DEFAULT_MAX_TRAIN_ROWS).

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

#: Default node budget for the training sample, unchanged from the silver-v1
#: run so a rerun of that reproduces it. `--max-train-rows` overrides it.
#:
#: This is the number that decides whether a bigger corpus buys anything. Every
#: class except content_block is kept whole until the GLOBAL cap binds, and then
#: the cap scales the rare classes down by whatever factor brings the total back
#: under it. Run the arithmetic on the silver-v1 class counts at 1x and at 2.7x
#: and the kept non-content_block rows come out at 570,593 and 570,594 — the
#: same number, footer 196,025 both times. A corpus 2.7x the size therefore
#: delivers a more diverse DRAW (more domains behind the same row count) and not
#: one extra row of supervision, unless this moves with it.
DEFAULT_MAX_TRAIN_ROWS = 600_000
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
    """Counts NODES, not rows: the masked and unmasked rows of one node are the
    same node, so the sampling rates are computed from the unmasked half."""
    counts: Counter = Counter()
    for row in rows(path):
        if row["m"] != 0:
            continue
        label = collapse(row["l"])
        if label is not None:
            counts[label] += 1
    return counts


def sampling_rates(
    counts: Counter, max_train_rows: int = DEFAULT_MAX_TRAIN_ROWS
) -> dict[str, float]:
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
    if projected > max_train_rows:
        surplus = projected - max_train_rows
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
        self.masked: list[int] = []
        self.weights: list[float] = []

    def add(self, features: list[float], tokens: list[int], label: int, domain: str,
            page_id: str, node_id: int, masked: int, weight: float) -> None:
        row = self.buffer[self.fill]
        row[: self.n_numeric] = features
        for bucket in tokens:
            row[self.n_numeric + bucket] = 1.0
        self.labels.append(label)
        self.groups.append(domain)
        self.refs.append((page_id, node_id))
        self.masked.append(masked)
        self.weights.append(weight)
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
        return (matrix, np.asarray(self.labels, dtype=np.int8), self.groups, self.refs,
                np.asarray(self.masked, dtype=np.int8),
                np.asarray(self.weights, dtype=np.float32))


def build_split(path: Path, rates: dict[str, float], n_numeric: int, n_tokens: int,
                jev: dict[tuple[str, int], tuple[str, float]] | None = None):
    """Rows for one split, from the silver labels and optionally from Jev.

    Jev is a THIRD label source, not a correction of the second. Three cases,
    and the difference between them is the whole point:

    * **silver only** - the node sits in a landmark. Unchanged.
    * **Jev only** - the node is on a landmark-less page, so silver said
      `unlabelled` and the row did not exist at all. This is the case the
      corpus expansion was for: it is pure addition, and it is the only way
      `footer` and `navigation` ever get a training row on a div-soup theme.
    * **both, disagreeing** - BOTH rows are emitted, each with its own label
      and its own weight. Dropping either one would be a silent decision about
      which annotator is right on a population nobody has adjudicated; keeping
      both lets the trees see the conflict and return a probability that
      reflects it. The rate is counted and reported.

    A Jev row's weight is the model's own confidence, so a 0.76 answer counts
    for less than a 0.99 one.
    """
    builder = ChunkedBuilder(n_numeric, n_tokens)
    kept: Counter = Counter()
    sources: Counter = Counter()
    agree = disagree = 0
    for row in rows(path):
        silver = collapse(row["l"])
        jev_label: str | None = None
        jev_weight = 1.0
        if jev is not None:
            found = jev.get((row["p"], row["n"]))
            if found is not None:
                jev_label, jev_weight = found

        if silver is None and jev_label is None:
            continue
        # The keep decision is a hash of (page, node), so a node's masked and
        # unmasked rows are kept or dropped TOGETHER and the mix stays 50/50.
        # Downsampling is driven by the SILVER label, because that is what the
        # rates were computed from. A Jev-only node has no silver label and is
        # never downsampled: those rows are the scarce ones this whole exercise
        # exists to produce, and there are at most ~40k of them against a 600k
        # cap.
        if silver is not None:
            rate = rates.get(silver, 1.0)
            if rate < 1.0 and keep_hash(row["p"], row["n"]) >= rate:
                continue

        if silver is not None:
            if row["m"] == 0:
                kept[silver] += 1
                sources["silver" if jev_label is None else
                        ("both-agree" if jev_label == silver else "both-disagree")] += 1
                if jev_label is not None:
                    if jev_label == silver:
                        agree += 1
                    else:
                        disagree += 1
            builder.add(row["f"], row["k"], CLASS_INDEX[silver], row["d"], row["p"], row["n"],
                        int(row["m"]), float(row.get("w", 1.0)))

        # The Jev row is emitted when it ADDS something: a label where silver
        # had none, or a different label from silver's.
        if jev_label is not None and jev_label != silver:
            if row["m"] == 0:
                kept[jev_label] += 1
                if silver is None:
                    sources["jev-only"] += 1
            builder.add(row["f"], row["k"], CLASS_INDEX[jev_label], row["d"], row["p"], row["n"],
                        int(row["m"]), float(jev_weight))
    return (*builder.finish(), kept, sources, agree, disagree)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset-version", default="silver-v1",
                        help="subdirectory of --data-root holding nodes-*.jsonl.gz")
    parser.add_argument("--max-train-rows", type=int, default=DEFAULT_MAX_TRAIN_ROWS,
                        help="node budget for the training sample; see DEFAULT_MAX_TRAIN_ROWS")
    parser.add_argument("--jev-labels", default=None,
                        help="run-divsoup-v7.ts output; adds Jev as a third label source")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    dataset = data_root / args.dataset_version
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    schema = json.loads((dataset / "feature-schema.json").read_text())
    n_numeric = schema["nodeFeatureCount"]
    n_tokens = schema["tokenBuckets"]

    # Jev labels are loaded once and applied to EVERY split the sample covers.
    # They are drawn from train pages only (see corpus/node-sample.ts), so val
    # and holdout are untouched in practice; looking them up per split rather
    # than assuming that keeps the assumption checkable.
    jev: dict[tuple[str, int], tuple[str, float]] | None = None
    if args.jev_labels:
        from jev_labels import load_jev_region_labels

        jev = load_jev_region_labels(Path(args.jev_labels))
        print(json.dumps({"jevNodeLabels": len(jev)}), flush=True)

    print("counting train labels", flush=True)
    counts = count_labels(dataset / "nodes-train.jsonl.gz")
    rates = sampling_rates(counts, args.max_train_rows)
    print(json.dumps({"counts": dict(counts), "rates": rates}, indent=2), flush=True)

    manifest: dict[str, object] = {
        "featurizerVersion": schema["featurizerVersion"],
        "datasetVersion": args.dataset_version,
        "classes": REGION_CLASSES,
        "trainLabelCounts": dict(counts),
        "samplingRates": rates,
        "maxTrainRows": args.max_train_rows,
        "contentBlockRatio": CONTENT_BLOCK_RATIO,
        "jevLabelFile": args.jev_labels,
        "jevNodeLabels": len(jev) if jev is not None else 0,
        "splits": {},
    }

    for split in ("train", "val", "holdout"):
        # Only the training split is subsampled. Val and holdout are kept whole,
        # so every reported number is over the real class distribution.
        split_rates = rates if split == "train" else {}
        matrix, labels, groups, refs, masked, weights, kept, sources, agree, disagree = build_split(
            dataset / f"nodes-{split}.jsonl.gz", split_rates, n_numeric, n_tokens, jev
        )
        sp.save_npz(out / f"X-{split}.npz", matrix)
        np.save(out / f"y-{split}.npy", labels)
        np.save(out / f"m-{split}.npy", masked)
        np.save(out / f"w-{split}.npy", weights)
        (out / f"refs-{split}.jsonl").write_text(
            "".join(
                json.dumps({"p": p, "n": n, "d": d}) + "\n"
                for (p, n), d in zip(refs, groups, strict=True)
            )
        )
        by_class = Counter(REGION_CLASSES[int(i)] for i in labels)
        manifest["splits"][split] = {
            "rows": int(matrix.shape[0]),
            "maskedRows": int((masked == 1).sum()),
            "unmaskedRows": int((masked == 0).sum()),
            "inheritedRows": int((weights < 1.0).sum()),
            "nnz": int(matrix.nnz),
            "domains": len(set(groups)),
            "byClass": dict(by_class),
            "labelSources": dict(sources),
            # Over the nodes BOTH sources labelled. A node only one of them
            # labelled is not a disagreement, and counting it as agreement
            # would make the rate look better the more div-soup data is added.
            "silverJevOverlap": agree + disagree,
            "silverJevDisagreements": disagree,
            "silverJevDisagreementRate": (
                round(disagree / (agree + disagree), 4) if (agree + disagree) else None
            ),
        }
        print(f"{split}: {matrix.shape[0]} rows, {matrix.nnz} nnz, {len(set(groups))} domains, "
              f"jev-only={sources.get('jev-only', 0)}, disagree={disagree}/{agree + disagree}",
              flush=True)
        del matrix, labels, groups, refs, masked, weights, kept

    (out / "sample-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
