#!/usr/bin/env python3
"""Draw the deterministic reviewer subset for the div-soup evaluation labels.

Stratified on **tag, split and domain**: facts about the page. The sample file
also carries the model's `predicted` class and its confidence, and neither is
read here. Sampling on a model's own output would make the subset a function of
the thing being evaluated, which is the same mistake as labelling with it in
view, only quieter.

A per-domain cap keeps one site from supplying the audit, and order is a hash of
the node identity, so the subset is reproducible from the inputs alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DATA_ROOT = Path(os.environ["DOM_CLASSIFIER_DATA_ROOT"])


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def order_key(row: dict[str, Any]) -> str:
    return hashlib.sha256(f"{row['pageId']}\0{row['nodeId']}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, default=600)
    parser.add_argument("--max-per-domain", type=int, default=12)
    args = parser.parse_args()
    for path in (args.packets, args.output):
        if not str(path.resolve()).startswith(f"{DATA_ROOT.resolve()}/"):
            raise ValueError("paths must be under the private data root")

    packets = read_jsonl(args.packets)
    for row in packets:
        if "predicted" in row or "confidence" in row:
            raise ValueError("packet carries a model prediction; refusing to sample from it")

    by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(packets, key=order_key):
        by_tag[row["stratum"]].append(row)

    # Proportional to the tag's share, with a floor so a rare tag is not lost
    # and a ceiling so `div` does not become the whole audit.
    total = len(packets)
    chosen: list[dict[str, Any]] = []
    per_domain: Counter[str] = Counter()
    for tag, rows in sorted(by_tag.items(), key=lambda item: (-len(item[1]), item[0])):
        share = len(rows) / total
        quota = max(min(len(rows), 8), round(args.target * min(share, 0.45)))
        taken = 0
        for row in rows:
            if taken >= quota:
                break
            domain = row.get("domain") or ""
            if per_domain[domain] >= args.max_per_domain:
                continue
            per_domain[domain] += 1
            chosen.append(row)
            taken += 1
    # Trim or extend to the target in the same deterministic order.
    chosen.sort(key=order_key)
    if len(chosen) > args.target:
        chosen = chosen[: args.target]
    elif len(chosen) < args.target:
        seen = {(row["pageId"], row["nodeId"]) for row in chosen}
        for row in sorted(packets, key=order_key):
            if len(chosen) >= args.target:
                break
            if (row["pageId"], row["nodeId"]) in seen:
                continue
            if per_domain[row.get("domain") or ""] >= args.max_per_domain:
                continue
            per_domain[row.get("domain") or ""] += 1
            chosen.append(row)
        chosen.sort(key=order_key)

    args.output.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in chosen))
    args.output.chmod(0o600)
    print(
        json.dumps(
            {
                "packets": total,
                "sampled": len(chosen),
                "tags": dict(sorted(Counter(row["stratum"] for row in chosen).items())),
                "splits": dict(sorted(Counter(row["split"] for row in chosen).items())),
                "domains": len({row.get("domain") for row in chosen}),
                "sha256": "sha256:" + hashlib.sha256(args.output.read_bytes()).hexdigest(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
