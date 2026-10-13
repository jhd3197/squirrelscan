# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Score a human adjudication file (answers.json from adjudicate-v1.html) against the two text judges (#2307).

    uv run --script score_human.py --answers answers.json --items items-private.json --haiku h.jsonl [--haiku h2.jsonl ...] --opus o.jsonl [...]

Reports: how often the human sides with Haiku, with Opus, with neither, on the DISPUTED items; the control items
(judges agreed) as a sanity check on the human/judge baseline; and which class pairs the human resolves which way.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

MERGE = {"blog_post": "editorial", "article": "editorial", "product_listing": "listing", "listing_index": "listing",
         "editorial (article/blog)": "editorial", "product_listing": "listing", "listing_index": "listing"}  # fmt: skip


def norm(label: str) -> str:
    return MERGE.get(label, label)


def load(paths: list[Path]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if "error" not in row:
                    out[row["itemId"]] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--haiku", type=Path, action="append", required=True)
    parser.add_argument("--opus", type=Path, action="append", required=True)
    args = parser.parse_args()
    answers = json.loads(args.answers.read_text())
    items = {i["id"]: i for i in json.loads(args.items.read_text())}
    haiku, opus = load(args.haiku), load(args.opus)
    side: Counter = Counter()
    resolved: dict[tuple[str, str], Counter] = defaultdict(Counter)
    control = Counter()
    for item_id, label in answers.items():
        if label == "skip" or item_id not in items or item_id not in haiku or item_id not in opus:
            continue
        human, h, o = norm(label), norm(haiku[item_id]["label"]), norm(opus[item_id]["label"])
        if items[item_id]["_control"]:
            control["agrees with both judges" if human == h == o else "differs from the agreed label"] += 1
            continue
        key = "haiku" if human == h else "opus" if human == o else "neither"
        side[key] += 1
        resolved[tuple(sorted((h, o)))][human] += 1
    total = sum(side.values())
    print(f"disputed items answered: {total}")
    for k, v in side.items():
        print(f"  human sides with {k}: {v} ({v / max(total, 1):.0%})")
    print("controls (judges agreed):", dict(control))
    print("how the human resolves each disputed pair (pair -> human label counts):")
    for pair, counts in sorted(resolved.items(), key=lambda x: -sum(x[1].values())):
        print(f"  {pair}: {dict(counts.most_common())}")


if __name__ == "__main__":
    main()
