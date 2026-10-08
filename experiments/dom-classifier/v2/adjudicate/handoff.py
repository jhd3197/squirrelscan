# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Calibration-gold item list for the gold lane (gold-cal-v1, #2307).

One row per calibration item: where it is (pageId, nodeId, url), what to show
(suggested = the adjudicator's answer) and the fixes to offer (alternatives:
the adjudicator's runner-up, the training label, the OOF model, Clef), plus the
stratum and inclusion weight the label model needs later. The render it was
judged on is named so the gold lane can reuse the same capture.

Calibration gold is TRAIN/VAL only and separate from gold-v1 (test gold).

    uv run --script handoff.py --items cohort.jsonl --answers <arm answers.jsonl> --render <run> \
        --votes votes.jsonl --out gold-cal-v1-items.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True, help="the chosen adjudicator arm's answers")
    parser.add_argument("--render", type=Path, required=True)
    parser.add_argument("--votes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    answers = {a["itemId"]: a for a in jsonl(args.answers) if not a.get("error")}
    renders = {r["pageId"]: r for r in jsonl(args.render / "renders.jsonl") if not r.get("error")}
    votes: dict[str, dict] = {}
    for v in jsonl(args.votes):
        votes.setdefault(v["key"], {})[v["source"]] = v["label"]
    rows, missing = [], Counter()
    for item in jsonl(args.items):
        answer = answers.get(item["itemId"])
        if not answer:
            missing["no_answer"] += 1
            continue
        key = item["pageId"] if item["axis"] == "page" else f"{item['pageId']}#{item['nodeId']}"
        source = votes.get(key, {})
        offered = [answer.get("runnerUp"), item["given"], item.get("oofPred"), source.get("clef")]
        alternatives = [c for c in dict.fromkeys(offered) if c and c != answer["label"]]
        render = renders[item["pageId"]]
        rows.append(
            {
                "itemId": item["itemId"],
                "axis": item["axis"],
                "pageId": item["pageId"],
                "nodeId": item.get("nodeId"),
                "url": item["url"],
                "split": item["split"],
                "platform": item["platform"],
                "lang": item["lang"],
                "suggested": answer["label"],
                "suggestedBy": answer["arm"],
                "alternatives": alternatives,
                "stratum": item["stratum"],
                "weight": item["weight"],
                "pilot": item["pilot"],
                "render": {
                    "run": args.render.name,
                    "captureId": render["captureId"],
                    "scriptMode": render["scriptMode"],
                    "rect": (render.get("located", {}).get(str(item.get("nodeId"))) or {}).get("rect"),
                },
            }
        )
    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps({"items": Counter(r["axis"] for r in rows), "missing": missing}, default=dict))


if __name__ == "__main__":
    main()
