# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Per-class precision/recall of section heads against text-judge labels on sampled corpus nodes (#2307).

    uv run --script section_eval.py --pred predictions.jsonl --answers answers-cli.jsonl [--answers more.jsonl ...] \
        [--opus opus-answers.jsonl] --out report.json [--goal-precision 0.9 --min-answers 10]

Truth: the judge's `label` (a section purpose, `none` for no section / an item OF a section), Opus over Haiku where both
answered. A prediction fires when the artifact's top class is not `none` (thresholds were zeroed in
predict-section-nodes.ts); for each class the best threshold is the one with the highest recall at precision >= goal
and at least --min-answers answers. The sample is per-page random nodes with a landmark share: not the natural node
distribution, so read recall as "of sampled nodes", and precision as the quantity that matters for a finding.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

BARS = [round(0.3 + 0.05 * i, 2) for i in range(14)]


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--answers", type=Path, action="append", required=True)
    parser.add_argument("--opus", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--goal-precision", type=float, default=0.9)
    parser.add_argument("--min-answers", type=int, default=10)
    parser.add_argument("--item-classes", nargs="*", default=["comments"],
                        help="classes whose ITEMS also count as the class (the runtime rule marks every node of a thread)")
    args = parser.parse_args()

    truth: dict[str, str] = {}
    item_of: dict[str, str] = {}
    for path in [*args.answers, *args.opus]:
        for row in jsonl(path):
            if not row.get("error") and row.get("label"):
                truth[row["itemId"]] = row["label"]
                item_of[row["itemId"]] = row.get("itemOf") or ""

    def is_class(item_id: str, cls: str) -> bool:
        """The comments rule marks every node inside a thread, so an ITEM of a comment thread (label none, itemOf comments)
        counts for `comments`; every other class needs the whole section."""
        return truth[item_id] == cls or (cls in args.item_classes and item_of.get(item_id) == cls)

    by_artifact: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in jsonl(args.pred):
        if row["itemId"] in truth:
            by_artifact[row["artifact"]][row["itemId"]] = row
    counts = Counter(truth.values())
    report: dict = {"labelled": len(truth), "teacherByClass": dict(counts.most_common())}
    for artifact, rows in by_artifact.items():
        classes = sorted({c for c in counts if c != "none"})
        per_class = {}
        for cls in classes:
            support = sum(1 for i in rows if is_class(i, cls))
            curve = []
            best = None
            for bar in BARS:
                fired = [i for i, r in rows.items() if r["label"] == cls and (r["confidence"] or 0) >= bar]
                if len(fired) < args.min_answers:
                    continue
                hit = sum(1 for i in fired if is_class(i, cls))
                point = {"bar": bar, "answered": len(fired), "precision": round(hit / len(fired), 3), "recall": round(hit / max(support, 1), 3)}
                curve.append(point)
                if point["precision"] >= args.goal_precision and (best is None or point["recall"] > best["recall"]):
                    best = point
            at_ship = [i for i, r in rows.items() if r["label"] == cls]
            ship_hit = sum(1 for i in at_ship if is_class(i, cls))
            per_class[cls] = {
                "support": support,
                "firedAnyThreshold": len(at_ship),
                "precisionAnyThreshold": round(ship_hit / len(at_ship), 3) if at_ship else None,
                "recallAnyThreshold": round(ship_hit / support, 3) if support else None,
                "bestAtGoal": best,
            }
        fired_all = [i for i, r in rows.items() if r["label"]]
        none_truth = [i for i in rows if truth[i] == "none" and not item_of.get(i)]
        report[artifact] = {
            "nodes": len(rows),
            "firedShare": round(len(fired_all) / len(rows), 3),
            "precisionOverall": round(sum(1 for i in fired_all if is_class(i, rows[i]["label"])) / max(len(fired_all), 1), 3),
            "noneKept": round(sum(1 for i in none_truth if rows[i]["label"] is None) / max(len(none_truth), 1), 3),
            "perClass": per_class,
        }
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"labelled {len(truth)}; teacher classes: {dict(counts.most_common(12))}")
    for artifact in by_artifact:
        a = report[artifact]
        print(f"== {artifact}: nodes {a['nodes']}, fired {a['firedShare']}, precision when fired {a['precisionOverall']}, none kept {a['noneKept']}")
        for cls, v in sorted(a["perClass"].items(), key=lambda x: -x[1]["support"]):
            if v["support"] >= 8 or v["firedAnyThreshold"] >= 8:
                print(f"   {cls:18s} support {v['support']:4d}  fired {v['firedAnyThreshold']:4d}  precision {v['precisionAnyThreshold']}  recall {v['recallAnyThreshold']}  best@{args.goal_precision}: {v['bestAtGoal']}")


if __name__ == "__main__":
    main()
