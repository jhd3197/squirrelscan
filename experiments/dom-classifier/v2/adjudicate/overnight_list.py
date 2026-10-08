# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3"]
# ///
"""The capped overnight adjudication list (#2307): ~3k pages + ~3k sections.

Pages come from three candidate lists: the OOF model confidently contradicting
the label, Jev/Clef disagreements, and the policy v8.2 relabel candidates. Each
gets a priority in [0.5, 1.1]: the OOF confidence, half of Clef's confidence
plus one half, or one half plus 0.15 per independent v8.2 signal. A page on
several lists takes its best score plus 0.1. Sections come from the Jev rows the
OOF model contradicts, by confidence.

Stratified so every class and platform is represented: a (training label x
platform) cell gets its share of the cap in proportion to the square root of its
size, at least min(cell size, --floor), and within a cell the highest priority
goes first. Calibration-cohort items (they are gold) and gold-v1 pages never
enter. Sections are oversampled (--section-oversample) because about a third of
section targets render hidden and drop out after the render.

    uv run --script overnight_list.py --cc <2026-10-08-cc> --cl <cl dir> --cohort cohort.jsonl \
        --gold queue-pages.jsonl --out items.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from confident_learning import jsonl, load_pages


def item_id(axis: str, key: str) -> str:
    return f"ovn-{axis}-{hashlib.sha256(f'overnight-v1:{key}'.encode()).hexdigest()[:12]}"


def allocate(cells: dict[tuple, list[dict]], cap: int, floor: int) -> list[dict]:
    weights = {c: math.sqrt(len(rows)) for c, rows in cells.items()}
    total = sum(weights.values())
    quota = {c: max(min(len(rows), floor), round(cap * weights[c] / total)) for c, rows in cells.items()}
    quota = {c: min(q, len(cells[c])) for c, q in quota.items()}
    # Trim the largest cells first if the floors pushed the total over the cap.
    while sum(quota.values()) > cap:
        biggest = max(quota, key=lambda c: quota[c] - min(len(cells[c]), floor))
        quota[biggest] -= 1
    chosen = []
    for c, rows in cells.items():
        chosen += sorted(rows, key=lambda r: -r["priority"])[: quota[c]]
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", type=Path, required=True)
    parser.add_argument("--cl", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--gold", type=Path, nargs="+", required=True)
    parser.add_argument("--pages", type=int, default=3000)
    parser.add_argument("--v82-pages", type=int, default=1000, help="share of --pages reserved for v8.2 candidates")
    parser.add_argument("--sections", type=int, default=3000)
    parser.add_argument("--section-oversample", type=float, default=1.4)
    parser.add_argument("--floor", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pages = load_pages(args.cc)
    cohort = jsonl(args.cohort)
    taken_pages = {i["pageId"] for i in cohort if i["axis"] == "page"}
    taken_nodes = {f"{i['pageId']}#{i['nodeId']}" for i in cohort if i["axis"] == "section"}
    gold_pages = {r["pageId"] for path in args.gold for r in jsonl(path)}

    best: dict[str, dict] = {}

    def offer(page_id: str, given: str, score: float, source: str, suggested: str | None) -> None:
        if page_id not in pages or page_id in taken_pages or page_id in gold_pages:
            return
        row = best.get(page_id)
        if row is None:
            best[page_id] = {"given": given, "priority": score, "sources": [source], "suggested": suggested}
        else:
            row["priority"] = max(row["priority"], score) + 0.1
            row["sources"].append(source)

    for r in jsonl(args.cl / "pagetype-candidates.jsonl"):
        if "oof" in r["sources"]:
            offer(r["pageId"], r["given"], r["conf"], "oof", r["suggested"])
        if "jev_clef" in r["sources"]:
            offer(r["pageId"], r["given"], 0.5 + 0.5 * (r.get("clefConf") or 0), "jev_clef", r.get("clef"))
    # Policy v8.2 candidates get their own share of the cap: their labels are stale by definition,
    # and on OOF confidence alone they would rank below the CL list.
    v82 = {r["pageId"]: r for r in jsonl(args.cl / "pagetype-v82-candidates.jsonl")}
    for page_id, r in v82.items():
        if page_id in best:
            best[page_id]["sources"].append("policy_v82")
    page_cells: dict[tuple, list[dict]] = defaultdict(list)
    for page_id, row in best.items():
        page_cells[(row["given"], pages[page_id]["platform"])].append({"pageId": page_id, **row})
    page_rows = allocate(page_cells, args.pages - args.v82_pages, args.floor)
    v82_cells: dict[tuple, list[dict]] = defaultdict(list)
    for page_id, r in v82.items():
        if page_id in best or page_id in taken_pages or page_id in gold_pages or page_id not in pages:
            continue
        score = min(1.0, 0.5 + 0.15 * len(r["signals"])) + 0.01 * r["jevProductMass"]
        v82_cells[(r["given"], pages[page_id]["platform"])].append(
            {"pageId": page_id, "given": r["given"], "priority": score, "sources": ["policy_v82"],
             "suggested": r["suggested"]}
        )  # fmt: skip
    page_rows += allocate(v82_cells, args.v82_pages, args.floor)
    page_cells.update({k: v for k, v in v82_cells.items() if k not in page_cells})

    section_cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in jsonl(args.cl / "section-candidates.jsonl"):
        key = f"{r['pageId']}#{r['nodeId']}"
        if r["src"] != "jev" or key in taken_nodes or r["pageId"] in gold_pages or r["pageId"] not in pages:
            continue
        section_cells[(r["given"], r["platform"])].append(
            {"pageId": r["pageId"], "nodeId": r["nodeId"], "given": r["given"], "priority": r["conf"],
             "sources": ["oof"], "suggested": r["suggested"]}
        )  # fmt: skip
    section_rows = allocate(section_cells, round(args.sections * args.section_oversample), args.floor)

    out = []
    for axis, rows in (("page", page_rows), ("section", section_rows)):
        for r in rows:
            meta = pages[r["pageId"]]
            key = r["pageId"] if axis == "page" else f"{r['pageId']}#{r['nodeId']}"
            out.append(
                {
                    "itemId": item_id(axis, key),
                    "axis": axis,
                    "pageId": r["pageId"],
                    **({"nodeId": r["nodeId"]} if axis == "section" else {}),
                    "given": r["given"],
                    "suggestedBySource": r["suggested"],
                    "sources": r["sources"],
                    "priority": round(r["priority"], 4),
                    "stratum": f"{r['given']}|{meta['platform']}",
                    **{k: meta[k] for k in ("url", "split", "corpus", "platform", "lang", "domain")},
                }
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r) + "\n" for r in out))
    print(
        json.dumps(
            {
                "pages": len(page_rows),
                "sections": len(section_rows),
                "renderPages": len({r["pageId"] for r in out}),
                "pageBySource": Counter(s for r in page_rows for s in r["sources"]),
                "pageByClass": Counter(r["given"] for r in page_rows),
                "sectionByClass": Counter(r["given"] for r in section_rows),
                "byPlatform": Counter(f"{r['axis']}:{r['platform']}" for r in out),
                "emptyCells": {
                    "page": len(set(page_cells) - {(r["given"], pages[r["pageId"]]["platform"]) for r in page_rows}),
                    "section": len(
                        set(section_cells) - {(r["given"], pages[r["pageId"]]["platform"]) for r in section_rows}
                    ),
                },
            },
            default=dict,
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
