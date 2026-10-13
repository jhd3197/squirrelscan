# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Per-class thresholds for a section head from labelled sampled nodes, checked on held-out pages (#2307).

    uv run --script calibrate_section_head.py --artifact artifact.json --name r1 --pred predictions.jsonl \
        --answers haiku.jsonl --opus opus.jsonl --out artifact-cal.json --report report.json

Thresholds come from predict-section-nodes.ts output (the head's top class and probability with thresholds off):
a class gets the lowest bar whose precision >= goal with >= --min-answers answers; a class that never gets one abstains
(threshold 1.01). Pages are split in two by hash: thresholds are tuned on half A and the precision/recall on half B is
what the report calls held-out; the written artifact uses thresholds tuned on ALL labelled nodes. The comments rule is
not a head class and is unaffected. Writes the artifact with the section head's classThresholds replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

BARS = [round(0.3 + 0.01 * i, 2) for i in range(70)]


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--name", required=True, help="artifact name used in the predictions file")
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--answers", type=Path, action="append", required=True)
    parser.add_argument("--opus", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--goal-precision", type=float, default=0.9)
    parser.add_argument("--min-answers", type=int, default=10)
    args = parser.parse_args()

    truth: dict[str, str] = {}
    for path in [*args.answers, *args.opus]:
        for row in jsonl(path):
            if not row.get("error") and row.get("label"):
                truth[row["itemId"]] = row["label"]
    rows = [r for r in jsonl(args.pred) if r["artifact"] == args.name and r["itemId"] in truth and r["source"] != "rules"]
    page_of = lambda r: r["itemId"].split("-")[1]  # noqa: E731
    half = lambda r: int(hashlib.sha256(page_of(r).encode()).hexdigest()[:4], 16) % 2  # noqa: E731
    classes = sorted({r["label"] for r in rows if r["label"]})

    def thresholds(sample: list[dict]) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for cls in classes:
            chosen = None
            for bar in BARS:
                fired = [r for r in sample if r["label"] == cls and (r["confidence"] or 0) >= bar]
                if len(fired) < args.min_answers:
                    break
                if sum(1 for r in fired if truth[r["itemId"]] == cls) / len(fired) >= args.goal_precision:
                    chosen = bar
                    break
            out[cls] = chosen
        return out

    def score(sample: list[dict], th: dict[str, float | None]) -> dict:
        per = {}
        for cls in classes:
            if th[cls] is None:
                continue
            fired = [r for r in sample if r["label"] == cls and (r["confidence"] or 0) >= th[cls]]
            support = sum(1 for r in sample if truth[r["itemId"]] == cls)
            hit = sum(1 for r in fired if truth[r["itemId"]] == cls)
            per[cls] = {"threshold": th[cls], "answered": len(fired), "precision": round(hit / len(fired), 3) if fired else None,
                        "recall": round(hit / support, 3) if support else None, "support": support}
        fired_all = [r for r in sample if r["label"] and th.get(r["label"]) is not None and (r["confidence"] or 0) >= th[r["label"]]]
        overall = sum(1 for r in fired_all if truth[r["itemId"]] == r["label"]) / max(len(fired_all), 1)
        return {"perClass": per, "firedNodes": len(fired_all), "precisionOverall": round(overall, 3)}

    a = [r for r in rows if half(r) == 0]
    b = [r for r in rows if half(r) == 1]
    th_a, th_b = thresholds(a), thresholds(b)
    th_all = thresholds(rows)
    report = {"classes": classes, "tunedOnA_scoredOnB": score(b, th_a), "tunedOnB_scoredOnA": score(a, th_b), "thresholdsAll": th_all,
              "allNodesScoredWithAllThresholds": score(rows, th_all), "abstaining": [c for c in classes if th_all[c] is None]}
    args.report.write_text(json.dumps(report, indent=1) + "\n")

    artifact = json.loads(args.artifact.read_text())
    head = artifact["section"]
    head["classThresholds"] = [th_all.get(c) if th_all.get(c) is not None else 1.01 for c in head["classes"]]
    head["platformClassThresholds"] = {}
    args.out.write_text(json.dumps(artifact))
    print("thresholds (all nodes):", {c: th_all[c] for c in classes})
    print("abstaining:", report["abstaining"])
    for key in ("tunedOnA_scoredOnB", "tunedOnB_scoredOnA"):
        r = report[key]
        print(f"{key}: fired {r['firedNodes']} nodes, precision {r['precisionOverall']}; " + "; ".join(f"{c} {v['precision']}/{v['recall']} (n={v['answered']})" for c, v in r["perClass"].items()))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
