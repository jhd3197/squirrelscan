"""Build the section-purpose training sample by streaming the silver node rows.

Labels, per node (unmasked rows only: the runtime sees true features):
  1. Jev section-purpose-v2 (weight = Jev's confidence) - the only source on
     val and holdout, so every reported number is against Jev;
  2. a free label (corpus/section-labels.ts), TRAIN ONLY, and only for a
     (class, source) whose precision against Jev - measured on the val and
     holdout nodes both labelled - is >= --free-min-precision; weight = that
     precision;
  3. `none`: up to --negatives-per-page random unlabelled candidates on each
     Jev-labelled page (weight 0.5). The shortlist Jev saw includes every node
     whose tokens hint at a purpose, so an unshortlisted node is very likely
     `none`.
The free-label precision table is written out: it is the noise report.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp

SOURCES = ("md", "ld", "list", "net", "sem")


def jsonl(path: Path):
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def rank(page: str, node: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{page}:{node}".encode()).digest()[:6], "big")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="silver-v3 dir")
    parser.add_argument("--section-dir", required=True, help="section-v2 dir (node-labels, jev-*)")
    parser.add_argument("--classes", required=True, help="section-classes-v2.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--free-min-precision", type=float, default=0.85)
    parser.add_argument("--min-overlap", type=int, default=20)
    parser.add_argument("--negatives-per-page", type=int, default=15)
    args = parser.parse_args()

    section = Path(args.section_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    classes = [c["name"] for c in json.loads(Path(args.classes).read_text())["classes"]]
    index = {c: i for i, c in enumerate(classes)}

    jev: dict[tuple[str, int], dict] = {}
    jev_pages: set[str] = set()
    for split in ("holdout", "val", "train"):
        path = section / f"jev-{split}.jsonl"
        if path.exists():
            for row in jsonl(path):
                if row.get("section") in index:
                    jev[(row["pageId"], row["nodeId"])] = row
                    jev_pages.add(row["pageId"])
    free: dict[tuple[str, int], dict] = {}
    for row in jsonl(section / "node-labels.jsonl"):
        if "n" in row:
            free[(row["p"], row["n"])] = row

    # Noise report: each free (class, source) against Jev, on val + holdout.
    hits: Counter = Counter()
    seen: Counter = Counter()
    for key, row in free.items():
        j = jev.get(key)
        if j is None or j["split"] == "train":
            continue
        for source in SOURCES:
            label = row.get(source)
            if label:
                seen[(label, source)] += 1
                hits[(label, source)] += j["section"] == label
    precision = {f"{c}|{s}": {"n": seen[(c, s)], "precision": round(hits[(c, s)] / seen[(c, s)], 3)}
                 for (c, s) in seen}
    usable = {(c, s) for (c, s) in seen
              if seen[(c, s)] >= args.min_overlap
              and hits[(c, s)] / seen[(c, s)] >= args.free_min_precision}
    (out / "free-label-precision.json").write_text(json.dumps(
        {"minPrecision": args.free_min_precision, "minOverlap": args.min_overlap,
         "table": precision, "usedForTraining": sorted(f"{c}|{s}" for c, s in usable)},
        indent=2) + "\n")

    schema = json.loads((Path(args.dataset) / "feature-schema.json").read_text())
    n_numeric, n_tokens = schema["nodeFeatureCount"], schema["tokenBuckets"]
    width = n_numeric + n_tokens
    for split in ("train", "val", "holdout"):
        data, indices, indptr = [], [], [0]
        labels, weights, meta = [], [], []
        negatives_by_page: dict[str, list] = defaultdict(list)
        counts: Counter = Counter()
        for row in jsonl(Path(args.dataset) / f"nodes-{split}.jsonl.gz"):
            if row["m"] != 0:
                continue
            key = (row["p"], row["n"])
            label = weight = source = None
            if key in jev:
                label, weight, source = jev[key]["section"], jev[key].get("confidence") or 1.0, "jev"
            elif split == "train" and key in free:
                best = None
                for s in SOURCES:
                    c = free[key].get(s)
                    if c and (c, s) in usable:
                        p = hits[(c, s)] / seen[(c, s)]
                        if best is None or p > best[1]:
                            best = (c, p, s)
                if best:
                    label, weight, source = best[0], best[1], f"free-{best[2]}"
            if label is None:
                if row["p"] in jev_pages and key not in free:
                    negatives_by_page[row["p"]].append(row)
                continue
            vec = np.zeros(width, dtype=np.float32)
            vec[:n_numeric] = row["f"]
            for b in row["k"]:
                vec[n_numeric + b] = 1.0
            nz = np.flatnonzero(vec)
            data.extend(vec[nz]); indices.extend(nz); indptr.append(len(indices))
            labels.append(index[label]); weights.append(float(weight))
            ad = jev[key].get("adToken") if key in jev else free.get(key, {}).get("adToken")
            meta.append({"p": row["p"], "n": row["n"], "d": row["d"], "src": source, "adToken": ad})
            counts[(label, source)] += 1
        for page, rows in negatives_by_page.items():
            rows.sort(key=lambda r: rank(page, r["n"]))
            for row in rows[: args.negatives_per_page]:
                vec = np.zeros(width, dtype=np.float32)
                vec[:n_numeric] = row["f"]
                for b in row["k"]:
                    vec[n_numeric + b] = 1.0
                nz = np.flatnonzero(vec)
                data.extend(vec[nz]); indices.extend(nz); indptr.append(len(indices))
                labels.append(index["none"]); weights.append(0.5)
                meta.append({"p": page, "n": row["n"], "d": row["d"], "src": "negative",
                             "adToken": None})
                counts[("none", "negative")] += 1
        matrix = sp.csr_matrix((np.asarray(data, np.float32), np.asarray(indices, np.int32),
                                np.asarray(indptr, np.int64)), shape=(len(labels), width))
        sp.save_npz(out / f"X-{split}.npz", matrix)
        np.save(out / f"y-{split}.npy", np.asarray(labels, np.int8))
        np.save(out / f"w-{split}.npy", np.asarray(weights, np.float32))
        with (out / f"meta-{split}.jsonl").open("w") as handle:
            for m in meta:
                handle.write(json.dumps(m) + "\n")
        print(json.dumps({"split": split, "rows": len(labels),
                          "byClassSource": {f"{c}|{s}": n for (c, s), n in sorted(counts.items())}}),
              flush=True)
    (out / "classes.json").write_text(json.dumps({"classes": classes}) + "\n")


if __name__ == "__main__":
    main()
