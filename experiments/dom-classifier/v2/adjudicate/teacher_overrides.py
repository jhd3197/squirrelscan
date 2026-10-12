# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Teacher answers -> training-label overrides for the retrain (#2307).

Reads the suspect-label list (`overnight_list.py` items) and the teacher
answers on it (GPT via adjudicate_codex.py, Opus via adjudicate_cli.py) and
writes what `retrain-all.sh` takes:

  pagetype-teacher.jsonl   one row per accepted page item:
                           {pageId, pageType, confidence, split, source, itemId}
                           (the shape of a Jev row for `--jev-override`, i.e. PT_OVERRIDES)
  section-teacher.jsonl    one row per accepted section item:
                           {pageId, nodeId, section, confidence, split, source, itemId}
                           (append after the Jev section labels; the last row for a node wins)
  drop-pages-teacher.txt   page ids whose given label the teachers dispute but
                           do not settle (PT_DROP, appended to the existing list)
  report.json              counts by axis, source and flip

Acceptance (set from 300 items both teachers answered, 2026-10-13: GPT at
confidence >= 0.9 agreed with Opus on 95%, at 0.8-0.9 on 82%):
  both answered, same label     accept, confidence = the lower of the two
  both answered, labels differ  none (disputed)
  GPT only                      accept at confidence >= GPT_MIN, else none
  Opus only                     accept at confidence >= OPUS_MIN, else none
An item that is not accepted and whose teacher answer differs from the given
label goes to the drop list (page axis only): the given label is suspect and
nothing replaces it.

Guard: an item whose split is not train or val, or whose domain hashes into a
sealed domain or a bucket >= 90 (split rule: sha256(domain)[0:8] % 100), is
excluded and counted, never written (a sealed domain can sit in the train bucket).

    uv run --script teacher_overrides.py --items items-image.jsonl [--items items-text.jsonl ...] \
        --gpt <answers-codex.jsonl> --opus <answers-cli.jsonl> [--opus <more>] \
        --sealed <sealed-domains.jsonl> --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

GPT_MIN = 0.9
OPUS_MIN = 0.8


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def answers(paths: list[Path]) -> dict[str, dict]:
    """Valid answers by itemId; a later file or row replaces an earlier one."""
    out: dict[str, dict] = {}
    for path in paths:
        if not path.exists():
            continue
        for row in jsonl(path):
            if row.get("label") and not row.get("error"):
                out[row["itemId"]] = row
    return out


def domain_bucket(domain: str) -> tuple[str, int]:
    digest = hashlib.sha256(domain.encode()).hexdigest()
    return digest[:16], int(digest[:8], 16) % 100


def decide(gpt: dict | None, opus: dict | None) -> tuple[str | None, float | None, str]:
    """(label, confidence, source) for an item, label None when nothing is accepted."""
    if gpt and opus:
        if gpt["label"] == opus["label"]:
            return gpt["label"], min(gpt["confidence"], opus["confidence"]), "gpt+opus"
        return None, None, "disputed"
    if gpt:
        ok = gpt["confidence"] >= GPT_MIN
        return (gpt["label"], gpt["confidence"], "gpt") if ok else (None, None, "gpt-low")
    if opus:
        ok = opus["confidence"] >= OPUS_MIN
        return (opus["label"], opus["confidence"], "opus") if ok else (None, None, "opus-low")
    return None, None, "unanswered"


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--items", type=Path, action="append", required=True)
    parser.add_argument("--gpt", type=Path, action="append", default=[])
    parser.add_argument("--opus", type=Path, action="append", default=[])
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--eval-out",
        type=Path,
        help="also write the accepted teacher labels on CALIBRATION-role sealed domains here "
        "(an evaluation set training never sees)",
    )
    args = parser.parse_args()

    sealed_rows = [json.loads(line) for line in args.sealed.read_text().splitlines() if line.strip()]
    sealed = {r["d"] for r in sealed_rows}
    roles = {r["d"]: r["role"] for r in sealed_rows}
    if len(sealed) < 1000:
        raise SystemExit(f"{len(sealed)} sealed hashes, expected ~1069 (key `d`)")
    gpt = answers(args.gpt)
    opus = answers(args.opus)
    items: dict[str, dict] = {}
    for path in args.items:
        for row in jsonl(path):
            items[row["itemId"]] = row

    args.out.mkdir(parents=True, exist_ok=True)
    page_rows: list[dict] = []
    section_rows: list[dict] = []
    drops: set[str] = set()
    eval_rows: list[dict] = []
    counts: Counter = Counter()
    flips: Counter = Counter()
    for item_id, item in sorted(items.items()):
        axis = item["axis"]
        digest, bucket = domain_bucket(item["domain"])
        if item["split"] not in ("train", "val") or bucket >= 90 or digest in sealed:
            # Labelled, but never a training label: a sealed domain (calibration or test) can sit in
            # the train bucket, and a holdout page is evaluation data.
            counts[(axis, "excluded:sealed" if digest in sealed else "excluded:holdout")] += 1
            if digest in sealed and roles[digest] == "calibration" and args.eval_out:
                label, confidence, source = decide(gpt.get(item_id), opus.get(item_id))
                if label is not None:
                    eval_rows.append(
                        {
                            "itemId": item_id,
                            "axis": axis,
                            "pageId": item["pageId"],
                            "nodeId": item.get("nodeId"),
                            "label": label,
                            "confidence": round(confidence, 4),
                            "given": item["given"],
                            "platform": item.get("platform"),
                            "url": item["url"],
                            "source": f"teacher:{source}",
                        }
                    )
            continue
        label, confidence, source = decide(gpt.get(item_id), opus.get(item_id))
        counts[(axis, source)] += 1
        if label is None:
            teacher = gpt.get(item_id) or opus.get(item_id)
            if teacher and teacher["label"] != item["given"] and axis == "page":
                drops.add(item["pageId"])
            continue
        if label != item["given"]:
            flips[(axis, f"{item['given']}->{label}")] += 1
        row = {"pageId": item["pageId"], "confidence": round(confidence, 4), "split": item["split"],
               "source": f"teacher:{source}", "itemId": item_id}  # fmt: skip
        if axis == "page":
            page_rows.append({**row, "pageType": label})
        else:
            section_rows.append({**row, "nodeId": item["nodeId"], "section": label})

    (args.out / "pagetype-teacher.jsonl").write_text("".join(json.dumps(r) + "\n" for r in page_rows))
    (args.out / "section-teacher.jsonl").write_text("".join(json.dumps(r) + "\n" for r in section_rows))
    if args.eval_out:
        args.eval_out.write_text("".join(json.dumps(r) + "\n" for r in eval_rows))
    (args.out / "drop-pages-teacher.txt").write_text("".join(f"{p}\n" for p in sorted(drops)))
    report = {
        "items": len(items),
        "gptAnswers": len(gpt),
        "opusAnswers": len(opus),
        "acceptedPage": len(page_rows),
        "acceptedSection": len(section_rows),
        "droppedPages": len(drops),
        "bySource": {f"{a}:{s}": n for (a, s), n in sorted(counts.items())},
        "flips": {f"{a}:{f}": n for (a, f), n in flips.most_common(40)},
        "flippedShare": round(sum(flips.values()) / max(len(page_rows) + len(section_rows), 1), 3),
        "thresholds": {"gptMin": GPT_MIN, "opusMin": OPUS_MIN},
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("flips", "bySource")}))


if __name__ == "__main__":
    main()
