# /// script
# requires-python = ">=3.12"
# dependencies = ["pillow==11.3.0"]
# ///
"""Teacher inputs for a gold set, cut locally from its labeller captures (#2307).

The Opus teacher labels the gold sets (gold-v1, gold-cal-v1, gold-v1b) in place of
the human. Its inputs come only from what the set already holds: the capture PNG
and node boxes (`captures/page_*.{png,json}`) and the item manifest. No browser,
no Modal: the views are crops of the existing screenshot (views.py, the same cut
the Modal renderer makes), so the teacher sees the pixels the human sees.

Writes, under <set>/teacher/ (private):
  items.jsonl     one row per gold item, in adjudicate_cli's item shape; nodeId is
                  the capture node id (the outline's box)
  sources.jsonl   other sources' labels for LATER comparison (Jev, Clef, Sonnet,
                  training label); never read by the prompt
  views/<capturePageId>/{page,overview,n<node>,n<node>-overview}.jpg

    uv run --script teacher_prep.py --set <dom-classifier-data>/gold-v1
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from views import compose_views

AXES = {"pageTypeV8": "page", "sectionPurposeV3": "section"}


def jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def label_of(value) -> str | None:
    """A manifest source field is either a label or a {label, raw, ...} record."""
    if isinstance(value, dict):
        return value.get("label")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", type=Path, required=True, help="a gold set directory")
    parser.add_argument("--skip-views", action="store_true")
    args = parser.parse_args()
    name = args.set.name
    out = args.set / "teacher"
    (out / "views").mkdir(parents=True, exist_ok=True)
    manifest = jsonl(args.set / "private" / "manifest.jsonl")
    items, sources = [], []
    for row in manifest:
        axis = AXES[row["axis"]]
        srcs = row.get("sources") or {}
        items.append(
            {
                "itemId": row["suggestionId"],
                "set": name,
                "axis": axis,
                "pageId": row["capturePageId"],
                "nodeId": row.get("captureNodeId") if axis == "section" else None,
                "url": row["url"],
                "platform": row.get("platform", "custom"),
                "lang": row.get("lang"),
                "slicePageId": row.get("slicePageId"),
                "featurizerNode": row.get("featurizerNode"),
                "stratum": row.get("stratum"),
            }
        )
        sources.append(
            {
                "itemId": row["suggestionId"],
                "set": name,
                "axis": axis,
                "jev": label_of(row.get("jev")) or srcs.get("jev"),
                "clef": label_of(row.get("clef")) or srcs.get("clef"),
                "sonnet": srcs.get("sonnet"),
                "trainingLabel": srcs.get("trainingLabel"),
                "oofPred": srcs.get("oofPred"),
            }
        )
    (out / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in items))
    (out / "sources.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sources))
    if args.skip_views:
        print(json.dumps({"set": name, "items": len(items)}))
        return
    targets: dict[str, set[str]] = defaultdict(set)
    for item in items:
        nodes = targets[item["pageId"]]  # every page gets its page views, targets or not
        if item["nodeId"]:
            nodes.add(item["nodeId"])
    counts = Counter()
    for page_id, nodes in sorted(targets.items()):
        capture = args.set / "captures" / f"{page_id}.json"
        png = args.set / "captures" / f"{page_id}.png"
        if not capture.exists() or not png.exists():
            counts["missing_capture"] += 1
            continue
        page = json.loads(capture.read_text())
        rects = {n["id"]: n["rect"] for n in page["nodes"] if n["id"] in nodes}
        located = {n: {"visible": n in rects, "rect": rects.get(n)} for n in nodes}
        counts["missing_node"] += sum(1 for n in nodes if n not in rects)
        views = compose_views(png.read_bytes(), located)
        page_dir = out / "views" / page_id
        page_dir.mkdir(exist_ok=True)
        for view, data in views.items():
            (page_dir / f"{view}.jpg").write_bytes(data)
        counts["pages"] += 1
        counts["views"] += len(views)
    print(json.dumps({"set": name, "items": len(items), "byAxis": Counter(i["axis"] for i in items), **counts}))


if __name__ == "__main__":
    main()
