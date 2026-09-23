"""Jev page-type v8 work queues, cheapest first.

A page needs a v8 Jev call only when no existing label already answers v8:
  - its v7 label is one of the classes v8 keeps 1:1 (legal, contact, about,
    homepage, pricing, docs): the v7 label is reused, no call;
  - otherwise (article, other, product_*, listing_index, or no v7 label at all)
    it is queued.

Queues, in the order the runner should take them:
  queue-noise.jsonl    a stratified sample of schema-labelled pages, to measure
                       how often the free schema label agrees with Jev v8;
  queue-holdout.jsonl, queue-val.jsonl, queue-train.jsonl.
Noise-sample pages are left out of the split queues (their v8 label is written
once, to noise.jsonl, and merged back into its split at training time).
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from schema_labels_v8 import V7_REUSED


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="silver-v3 dir (page-index.jsonl.gz)")
    parser.add_argument("--v7", required=True, help="pagetype-jev-v7 dir")
    parser.add_argument("--schema-labels", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--noise-sample", type=int, default=500)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pages = {}
    with gzip.open(Path(args.dataset) / "page-index.jsonl.gz", "rt") as handle:
        for line in handle:
            row = json.loads(line)
            pages[row["p"]] = row
    v7: dict[str, str] = {}
    for split in ("train", "val", "holdout"):
        for line in (Path(args.v7) / f"{split}.jsonl").read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                v7[row["pageId"]] = row["pageType"]
    schema = {}
    for line in Path(args.schema_labels).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            schema[row["pageId"]] = row["pageType"]

    # Stratified noise sample: equal share per schema class, capped by what exists.
    rng = random.Random(20260923)
    by_class: dict[str, list[str]] = defaultdict(list)
    for page_id, label in sorted(schema.items()):
        if page_id in pages:
            by_class[label].append(page_id)
    per_class = max(1, args.noise_sample // max(len(by_class), 1))
    noise: list[str] = []
    for label in sorted(by_class):
        members = by_class[label]
        rng.shuffle(members)
        noise.extend(members[:per_class])
    noise_set = set(noise)

    def row_for(page_id: str) -> str:
        page = pages[page_id]
        return json.dumps({"pageId": page_id, "contentHash": page_id, "url": page["u"],
                           "split": page["s"]})

    (out / "queue-noise.jsonl").write_text("".join(row_for(p) + "\n" for p in noise))
    summary: dict = {"pages": len(pages), "noiseSample": len(noise),
                     "noiseByClass": dict(Counter(schema[p] for p in noise)),
                     "reusedV7": Counter(), "queued": Counter(), "queuedWhy": Counter()}
    queues: dict[str, list[str]] = defaultdict(list)
    for page_id in sorted(pages):
        split = pages[page_id]["s"]
        label = v7.get(page_id)
        if label in V7_REUSED:
            summary["reusedV7"][split] += 1
            continue
        if page_id in noise_set:
            continue
        queues[split].append(row_for(page_id))
        summary["queued"][split] += 1
        summary["queuedWhy"]["no v7 label" if label is None else f"v7 {label}"] += 1
    for split in ("holdout", "val", "train"):
        (out / f"queue-{split}.jsonl").write_text("".join(r + "\n" for r in queues[split]))
    summary = {k: dict(v) if isinstance(v, Counter) else v for k, v in summary.items()}
    (out / "queue-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
