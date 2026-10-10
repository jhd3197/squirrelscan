# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Two-teacher cross-check on calibration gold (#2307): Opus vs a second model, plus a blind cohort.

For every item that both teachers answered: Opus | second model | Jev | Clef | Sonnet |
human (where the labeller already holds an answer). Agreement per axis and per class.
The items where the teachers disagree (label, or the item-of purpose of a section)
become a BLIND labeller cohort: no suggestion is shown, so the human's pick settles
which teacher was right. A few items the teachers agreed on are mixed in as
controls, so the cohort is not all hard cases.

Writes into --out (private):
  table.jsonl / table.tsv             the per-item comparison
  agreement.json                      agreement per axis and class
  cohort/items.jsonl                  the cohort in the gold manifest shape
  cohort/model-suggestions.jsonl      the labeller sidecar (blind rows carry only their axis)
  cohort/private/key.jsonl            both teachers' answers and the outside sources

    uv run --script teacher_check.py --data <dom-classifier-data> --set gold-cal-v1 \
        --second <answers-codex.jsonl> --out <dom-classifier-data>/teacher-check-v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from teacher_queue import AXIS_NAME, human_answers, jsonl

SEED = "teacher-check-v1"


def rank(key: str) -> str:
    return hashlib.sha256(f"{SEED}:{key}".encode()).hexdigest()


def answer_key(a: dict | None) -> str | None:
    """A teacher's answer as one comparable token: the label, or none~item-of~<purpose>."""
    if not a:
        return None
    return f"none~item~{a['itemOf']}" if a.get("itemOf") else a["label"]


def agreement(pairs) -> dict:
    pairs = [(a, b) for a, b in pairs if a and b]
    return {"n": len(pairs), "agree": round(sum(a == b for a, b in pairs) / len(pairs), 3) if pairs else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--set", default="gold-cal-v1")
    parser.add_argument("--first", default="cli-opus-medium", help="Opus arm dir under <set>/teacher/answers")
    parser.add_argument("--second", type=Path, required=True, help="the second model's answers file")
    parser.add_argument("--controls", type=float, default=0.15, help="agreed controls as a share of the cohort")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    teacher = args.data / args.set / "teacher"
    items = {r["itemId"]: r for r in jsonl(teacher / "items.jsonl")}
    sources = {r["itemId"]: r for r in jsonl(teacher / "sources.jsonl")}
    first = {
        r["itemId"]: r for r in jsonl(teacher / "answers" / args.first / "answers-cli.jsonl") if not r.get("error")
    }
    second = {r["itemId"]: r for r in jsonl(args.second) if not r.get("error")}
    human = human_answers(args.data)
    table = []
    for i in sorted(set(first) | set(second)):
        item, src = items[i], sources.get(i, {})
        h = human.get(i)
        table.append(
            {
                "itemId": i,
                "axis": item["axis"],
                "platform": item["platform"],
                "stratum": item.get("stratum"),
                "opus": answer_key(first.get(i)),
                "second": answer_key(second.get(i)),
                "jev": src.get("jev"),
                "clef": src.get("clef"),
                "sonnet": src.get("sonnet"),
                "human": h and (f"none~item~{h['itemOf']}" if h.get("itemOf") else h["label"]),
                "opusConf": first.get(i, {}).get("confidence"),
                "secondConf": second.get(i, {}).get("confidence"),
            }
        )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "table.jsonl").write_text("".join(json.dumps(r) + "\n" for r in table))
    cols = ["itemId", "axis", "platform", "stratum", "opus", "second", "jev", "clef", "sonnet", "human"]
    (args.out / "table.tsv").write_text(
        "\t".join(cols) + "\n" + "".join("\t".join(str(r[c] or "") for c in cols) + "\n" for r in table)
    )

    def plain(token: str | None) -> str | None:
        """Outside sources have no item-of: compare a teacher's item answer as none."""
        return "none" if token and token.startswith("none~item~") else token

    report = {}
    both = [r for r in table if r["opus"] and r["second"]]
    for axis in ("page", "section"):
        rows = [r for r in both if r["axis"] == axis]
        by_class = defaultdict(list)
        for r in rows:
            by_class[plain(r["opus"])].append(r)
        report[axis] = {
            "opusVsSecond": agreement([(r["opus"], r["second"]) for r in rows]),
            "opusVsSecondIgnoringItemOf": agreement([(plain(r["opus"]), plain(r["second"])) for r in rows]),
            "byOpusClass": {c: agreement([(r["opus"], r["second"]) for r in v]) for c, v in sorted(by_class.items())},
            "opusVsJev": agreement([(plain(r["opus"]), r["jev"]) for r in rows]),
            "secondVsJev": agreement([(plain(r["second"]), r["jev"]) for r in table if r["axis"] == axis]),
            "opusVsClef": agreement([(plain(r["opus"]), r["clef"]) for r in rows]),
            "secondVsClef": agreement([(plain(r["second"]), r["clef"]) for r in table if r["axis"] == axis]),
            "opusVsSonnet": agreement([(plain(r["opus"]), r["sonnet"]) for r in rows]),
            "secondVsSonnet": agreement([(plain(r["second"]), r["sonnet"]) for r in table if r["axis"] == axis]),
            "opusVsHuman": agreement([(r["opus"], r["human"]) for r in rows]),
            "secondVsHuman": agreement([(r["second"], r["human"]) for r in table if r["axis"] == axis]),
            "topDisagreements": dict(
                Counter(f"{r['opus']} | {r['second']}" for r in rows if r["opus"] != r["second"]).most_common(12)
            ),
        }
    report["counts"] = {"opus": len(first), "second": len(second), "both": len(both),
                        "disagree": sum(1 for r in both if r["opus"] != r["second"])}  # fmt: skip
    (args.out / "agreement.json").write_text(json.dumps(report, indent=1))

    # The blind cohort: every disagreement the human has not already answered, plus agreed controls.
    answered = set(human)
    disagree = [r for r in both if r["opus"] != r["second"] and r["itemId"] not in answered]
    agreed = sorted([r for r in both if r["opus"] == r["second"] and r["itemId"] not in answered],
                    key=lambda r: rank(f"control:{r['itemId']}"))  # fmt: skip
    n_controls = max(3, round(args.controls * len(disagree)))
    # Controls spread over axes in proportion to the disagreements.
    controls = []
    for axis in ("page", "section"):
        share = round(n_controls * sum(1 for r in disagree if r["axis"] == axis) / max(len(disagree), 1))
        controls += [r for r in agreed if r["axis"] == axis][:share]
    cohort = sorted(disagree + controls, key=lambda r: rank(f"order:{r['itemId']}"))
    created = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    out_items, sidecar, key = [], [], []
    for position, r in enumerate(cohort):
        item = items[r["itemId"]]
        capture = json.loads((args.data / args.set / "captures" / f"{item['pageId']}.json").read_text())
        h = rank(f"suggestion:{r['itemId']}")
        sid = f"msug_{h[:8]}-{h[8:12]}-4{h[13:16]}-8{h[17:20]}-{h[20:32]}"
        axis = AXIS_NAME[item["axis"]]
        out_items.append(
            {
                "suggestionId": sid, "position": position, "axis": axis, "blind": True, "shown": None,
                "alternatives": [], "capturePageId": item["pageId"], "captureNodeId": item["nodeId"],
                "captureDir": f"{args.set}/captures", "stratum": "disagree" if r in disagree else "control",
                "sourceCohort": args.set, "sourceSuggestionId": r["itemId"], "slicePageId": item.get("slicePageId"),
                "featurizerNode": item.get("featurizerNode"), "url": item["url"], "platform": item["platform"],
                "lang": item.get("lang"),
            }
        )  # fmt: skip
        sidecar.append(
            {
                "schemaVersion": 1, "id": sid, "pageId": item["pageId"], "nodeId": item["nodeId"],
                "captureHash": capture["captureHash"], "provider": "teacher-check", "modelId": "teacher-check-v1",
                "modelRevision": "v1", "promptRevision": first[r["itemId"]].get("promptRevision"),
                "snapshotHash": "sha256:" + hashlib.sha256(f"{sid}:{axis}:blind".encode()).hexdigest(),
                "rawAnswers": None, "provisional": True, "mappedLabels": {}, "blindAxis": axis, "createdAt": created,
            }
        )  # fmt: skip
        key.append({"suggestionId": sid, "sourceSuggestionId": r["itemId"], "stratum": out_items[-1]["stratum"],
                    **{k: r[k] for k in ("opus", "second", "jev", "clef", "sonnet")}})  # fmt: skip
    cohort_dir = args.out / "cohort"
    (cohort_dir / "private").mkdir(parents=True, exist_ok=True)
    (cohort_dir / "items.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out_items))
    (cohort_dir / "model-suggestions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in sidecar))
    (cohort_dir / "private" / "key.jsonl").write_text("".join(json.dumps(x) + "\n" for x in key))
    print(json.dumps({**report["counts"], "cohort": len(out_items), "disagreeInCohort": len(disagree),
                      "controls": len(controls), "alreadyAnswered": len(answered & set(first) & set(second)),
                      "page": {k: report["page"][k] for k in ("opusVsSecond", "opusVsSecondIgnoringItemOf", "opusVsJev", "secondVsJev")},
                      "section": {k: report["section"][k] for k in ("opusVsSecond", "opusVsSecondIgnoringItemOf", "opusVsJev", "secondVsJev")}},
                     indent=1))  # fmt: skip


if __name__ == "__main__":
    main()
