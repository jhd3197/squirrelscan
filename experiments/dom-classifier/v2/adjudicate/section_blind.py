# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Blind check of a section head on FRESH labelled nodes (#2307): precision/recall per class at thresholds fixed beforehand.

    uv run --script section_blind.py --nodes blind/nodes-features.jsonl --classes <..>.classes.json --answers haiku.jsonl --opus opus.jsonl \
        --artifact stack --thresholds export-stack/section-head.meta.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=Path, required=True)
    ap.add_argument("--classes", type=Path, required=True)
    ap.add_argument("--answers", type=Path, action="append", required=True)
    ap.add_argument("--opus", type=Path, action="append", default=[])
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--thresholds", type=Path, required=True)
    a = ap.parse_args()
    classes = json.loads(a.classes.read_text())["classes"][a.artifact]
    thr = dict(zip(*[json.loads(a.thresholds.read_text())[k] for k in ("classes", "thresholds")]))
    truth: dict[str, str] = {}
    for p in [*a.answers, *a.opus]:
        for r in jsonl(p):
            if not r.get("error") and r.get("label"):
                truth[r["itemId"]] = "comments" if r.get("itemOf") == "comments" else r["label"]
    rows = [r for r in jsonl(a.nodes) if r["itemId"] in truth]
    print(f"{len(rows)} labelled nodes")
    print(f"{'class':18s} {'support':>7s} {'fired':>5s} {'prec':>5s} {'recall':>6s}  thr")
    for cls, t in thr.items():
        if cls == "none" or t > 1:
            continue
        j = classes.index(cls)
        fired = [r for r in rows if max(range(len(classes)), key=lambda i: r["p"][a.artifact][i]) == j and r["p"][a.artifact][j] >= t]
        sup = sum(1 for r in rows if truth[r["itemId"]] == cls)
        hit = sum(1 for r in fired if truth[r["itemId"]] == cls)
        print(f"{cls:18s} {sup:7d} {len(fired):5d} {hit / len(fired) if fired else float('nan'):5.2f} {hit / sup if sup else float('nan'):6.2f}  {t}")


if __name__ == "__main__":
    main()
