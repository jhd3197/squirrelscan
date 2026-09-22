"""Score two artifacts against the embargoed div-soup evaluation labels.

This is the number the exercise is for: region accuracy on LANDMARK-LESS pages,
where the silver labels the region head was trained on do not exist.

Read it once. The labels are embargoed for a reason — a modelling lane that has
seen them can no longer be graded by them, whatever anybody intends — so this
script is the only thing that opens that directory, it opens it at the very
end, and it reports before and after in the same run so there is no temptation
to iterate against it.

The Jev labels are mapped onto the ten region classes with `train/jev_labels.py`,
the SAME mapping the training rows were built with. Scoring against a different
definition of `footer` than the one the model was taught would measure the gap
between two conventions rather than the model.

  python divsoup_score.py --labels <embargo>/jev-v7.jsonl \
    --before before-predictions.jsonl --after after-predictions.jsonl \
    --out divsoup-eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from jev_labels import load_jev_region_labels  # noqa: E402

#: Chrome is what a landmark-less page fails to declare, so it is reported
#: separately from the overall number: an accuracy dominated by content_block
#: would hide the whole point.
CHROME = {"site_header", "footer", "navigation", "sidebar"}


def load_predictions(path: Path) -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[(row["pageId"], int(row["nodeId"]))] = row
    return out


def score(truth: dict, predictions: dict) -> dict:
    """Accuracy over the nodes BOTH the labels and this artifact cover.

    `unknown` is a prediction, not a missing value: the runtime returns it when
    the winning class is under its per-class threshold, and a consumer sees it.
    Counting it as "not attempted" would let a model that abstains on
    everything score 100%. It is counted as wrong, and reported separately as
    the abstention rate so the two can be read apart.
    """
    total = correct = abstained = 0
    chrome_total = chrome_correct = chrome_abstained = 0
    confusion: dict[str, Counter] = defaultdict(Counter)
    covered_by_class: Counter = Counter()
    for key, (gold, _confidence) in truth.items():
        row = predictions.get(key)
        if row is None:
            continue
        got = row["label"]
        total += 1
        covered_by_class[gold] += 1
        confusion[gold][got] += 1
        if got == "unknown":
            abstained += 1
        elif got == gold:
            correct += 1
        if gold in CHROME:
            chrome_total += 1
            if got == "unknown":
                chrome_abstained += 1
            elif got == gold:
                chrome_correct += 1
    per_class = {}
    for gold, counts in confusion.items():
        seen = sum(counts.values())
        per_class[gold] = {
            "support": seen,
            "correct": counts.get(gold, 0),
            "recall": round(counts.get(gold, 0) / seen, 4) if seen else None,
            "abstained": counts.get("unknown", 0),
            "topConfusion": counts.most_common(3),
        }
    return {
        "nodesScored": total,
        "accuracy": round(correct / total, 4) if total else None,
        "abstentionRate": round(abstained / total, 4) if total else None,
        "chrome": {
            "nodesScored": chrome_total,
            "accuracy": round(chrome_correct / chrome_total, 4) if chrome_total else None,
            "abstentionRate": (
                round(chrome_abstained / chrome_total, 4) if chrome_total else None
            ),
        },
        "perClass": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="the embargoed jev-v7.jsonl")
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    truth = load_jev_region_labels(Path(args.labels))
    before = load_predictions(Path(args.before))
    after = load_predictions(Path(args.after))

    report = {
        "labelSource": "Jev v7 (dom-suggestions-v7), embargoed div-soup eval set",
        "labelMapping": "train/jev_labels.py, the same mapping the training rows use",
        "mappedLabels": len(truth),
        "note": (
            "Both annotator and model are machines; this is agreement with a "
            "policy-v7 reading of each node, not human ground truth."
        ),
        "before": score(truth, before),
        "after": score(truth, after),
    }
    delta_all = None
    if report["before"]["accuracy"] is not None and report["after"]["accuracy"] is not None:
        delta_all = round(report["after"]["accuracy"] - report["before"]["accuracy"], 4)
    delta_chrome = None
    if (
        report["before"]["chrome"]["accuracy"] is not None
        and report["after"]["chrome"]["accuracy"] is not None
    ):
        delta_chrome = round(
            report["after"]["chrome"]["accuracy"] - report["before"]["chrome"]["accuracy"], 4
        )
    report["delta"] = {"accuracy": delta_all, "chromeAccuracy": delta_chrome}
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("mappedLabels", "delta")}, indent=2))
    for side in ("before", "after"):
        s = report[side]
        print(
            f"{side}: n={s['nodesScored']} acc={s['accuracy']} "
            f"abstain={s['abstentionRate']} | chrome n={s['chrome']['nodesScored']} "
            f"acc={s['chrome']['accuracy']} abstain={s['chrome']['abstentionRate']}"
        )


if __name__ == "__main__":
    sys.exit(main())
