"""Per-row fold bucket and page index for a region or section sample (#2307).

cv_compare.py needs, for every train/val row, the domain-grouped fold and the
page it came from, but the samples carry those only in refs-*.jsonl (region,
~250 MB) or meta-*.jsonl (section). This writes two small arrays per split so
only they, not the refs, travel to the training box:

  bucket-{split}.npy  uint32  int(sha256(domainHash)[0:8], 16), the fold
                              function of train_pagetype_v8 --cv-folds (% K)
  page-{split}.npy    int32   index into pages.json (page ids, shared by splits)

Holdout is never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument(
        "--out",
        default="",
        help="where the arrays go (default: --sample); "
        "a sample owned by an earlier run stays untouched",
    )
    args = parser.parse_args()
    sample = Path(args.sample)
    out = Path(args.out) if args.out else sample
    out.mkdir(parents=True, exist_ok=True)
    pages: dict[str, int] = {}
    bucket_of: dict[str, int] = {}
    for split in ("train", "val"):
        source = sample / f"refs-{split}.jsonl"
        if not source.exists():
            source = sample / f"meta-{split}.jsonl"
        buckets, page_index = [], []
        with source.open() as handle:
            for line in handle:
                row = json.loads(line)
                d = row["d"]
                if d not in bucket_of:
                    bucket_of[d] = int(hashlib.sha256(d.encode()).hexdigest()[:8], 16)
                buckets.append(bucket_of[d])
                page_index.append(pages.setdefault(row["p"], len(pages)))
        expected = len(np.load(sample / f"y-{split}.npy"))
        if len(buckets) != expected:
            raise SystemExit(f"{source.name}: {len(buckets)} rows, y-{split}.npy has {expected}")
        np.save(out / f"bucket-{split}.npy", np.asarray(buckets, np.uint32))
        np.save(out / f"page-{split}.npy", np.asarray(page_index, np.int32))
        print(json.dumps({"split": split, "rows": len(buckets)}))
    (out / "pages.json").write_text(json.dumps(list(pages)) + "\n")
    print(json.dumps({"pages": len(pages), "domains": len(bucket_of)}))


if __name__ == "__main__":
    main()
