# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Calibration-gold cohort for the gold lane (gold-cal-v1, #2307).

Calibration gold is TRAIN/VAL only and separate from gold-v1 (test gold). It
measures every label source's accuracy for the label model, so it must not
lean toward any one source:

- half the items are BLIND (no suggestion; the reviewer picks the class),
  chosen per stratum by a seeded hash, so blindness is balanced over class,
  candidate/rest, platform and language;
- a shown item shows ONE source's label, chosen by a seeded coin between Jev
  and the Sonnet adjudicator where Sonnet answered, else Jev. Which source was
  shown goes in the PRIVATE key only, never in the item the UI reads.

Writes:
  <out>/gold-cal-v1-items.jsonl   what the gold lane builds (location, capture,
                                  blind/shown, fixes to offer, stratum, weight)
  <out>/gold-cal-v1-key.jsonl     private: shown source and every source's label

    uv run --script handoff.py --items cohort.jsonl --render <run> --votes votes.jsonl \
        --sonnet <answers.jsonl> --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

SEED = "gold-cal-v1"
PAGE_NEAR = {
    "homepage": ["other", "about", "product_listing"],
    "about": ["contact", "homepage", "other"],
    "contact": ["about", "docs", "other"],
    "legal": ["docs", "other", "article"],
    "pricing": ["product_detail", "product_listing", "other"],
    "blog_post": ["article", "docs", "listing_index"],
    "article": ["blog_post", "docs", "other"],
    "docs": ["article", "blog_post", "listing_index"],
    "listing_index": ["product_listing", "blog_post", "other"],
    "product_detail": ["product_listing", "pricing", "other"],
    "product_listing": ["listing_index", "product_detail", "other"],
    "other": ["homepage", "docs", "product_detail"],
}


def jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rank(key: str) -> str:
    return hashlib.sha256(f"{SEED}:{key}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--render", type=Path, required=True)
    parser.add_argument("--votes", type=Path, required=True)
    parser.add_argument("--sonnet", type=Path, required=True, help="Sonnet adjudicator answers (pilot items)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    items = jsonl(args.items)
    renders = {r["pageId"]: r for r in jsonl(args.render / "renders.jsonl") if not r.get("error")}
    sonnet = {a["itemId"]: a for a in jsonl(args.sonnet) if not a.get("error")}
    votes: dict[str, dict] = defaultdict(dict)
    for v in jsonl(args.votes):
        votes[v["key"]][v["source"]] = v["label"]

    # Blind half per stratum: the first half of each stratum in seeded hash order.
    blind: set[str] = set()
    by_stratum: dict[tuple, list[dict]] = defaultdict(list)
    for item in items:
        by_stratum[(item["axis"], item["stratum"])].append(item)
    for (axis, stratum), rows in sorted(by_stratum.items()):
        rows = sorted(rows, key=lambda r: rank(f"blind:{r['itemId']}"))
        # Odd strata alternate which side gets the extra item, so the total stays near half.
        extra = int(rank(f"odd:{axis}:{stratum}"), 16) % 2
        blind.update(r["itemId"] for r in rows[: (len(rows) + extra) // 2])

    out_items, key_rows = [], []
    for item in items:
        k = item["pageId"] if item["axis"] == "page" else f"{item['pageId']}#{item['nodeId']}"
        source_votes = votes.get(k, {})
        if item["axis"] == "page":
            jev = next((source_votes[s] for s in ("jev81", "jev8", "jev7") if s in source_votes), item["given"])
        else:
            jev = source_votes.get("jev", item["given"])
        son = sonnet.get(item["itemId"], {}).get("label")
        is_blind = item["itemId"] in blind
        shown_source = shown = None
        if not is_blind:
            shown_source = "sonnet" if son and int(rank(f"source:{item['itemId']}"), 16) % 2 else "jev"
            shown = son if shown_source == "sonnet" else jev
        offered = [jev, son, item.get("oofPred"), source_votes.get("clef"), item["given"]]
        if item["axis"] == "page":
            offered += PAGE_NEAR.get(shown or jev, [])
        else:
            offered += ["none"]
        # A blind item offers no source-ordered quick picks: the reviewer sees the full class list.
        alternatives = [] if is_blind else [c for c in dict.fromkeys(offered) if c and c != shown][:4]
        render = renders[item["pageId"]]
        located = render.get("located", {}).get(str(item.get("nodeId"))) or {}
        out_items.append(
            {
                "itemId": item["itemId"],
                "axis": item["axis"],
                "question": "pageTypeV8" if item["axis"] == "page" else "sectionPurposeV3",
                "pageId": item["pageId"],
                "nodeId": item.get("nodeId"),
                "url": item["url"],
                "split": item["split"],
                "platform": item["platform"],
                "lang": item["lang"],
                "corpus": item["corpus"],
                "blind": is_blind,
                "shown": shown,
                "alternatives": alternatives,
                "stratum": item["stratum"],
                "weight": item["weight"],
                "capture": {
                    "run": args.render.name,
                    "captureId": render["captureId"],
                    "captureNodeId": located.get("nodeId"),
                    "rect": located.get("rect"),
                    "scriptMode": render["scriptMode"],
                    "scriptReason": render.get("scriptReason"),
                },
            }
        )
        key_rows.append(
            {
                "itemId": item["itemId"],
                "blind": is_blind,
                "shownSource": shown_source,
                "shown": shown,
                "jev": jev,
                "sonnet": son,
                "trainingLabel": item["given"],
                "oofPred": item.get("oofPred"),
                "clef": source_votes.get("clef"),
            }
        )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "gold-cal-v1-items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out_items))
    (args.out / "gold-cal-v1-key.jsonl").write_text("".join(json.dumps(r) + "\n" for r in key_rows))
    print(
        json.dumps(
            {
                "items": Counter(r["axis"] for r in out_items),
                "blind": Counter(f"{r['axis']}:{r['blind']}" for r in out_items),
                "shownSource": Counter(f"{r['itemId'].split('-')[1]}:{r['shownSource']}" for r in key_rows),
                "missingSectionNode": sum(
                    1 for r in out_items if r["axis"] == "section" and not r["capture"]["captureNodeId"]
                ),
            },
            default=dict,
        )
    )


if __name__ == "__main__":
    main()
