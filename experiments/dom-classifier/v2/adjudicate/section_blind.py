# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Blind check of a section head on FRESH labelled nodes (#2307): precision/recall per class at thresholds fixed beforehand.

    uv run --script section_blind.py --nodes blind/nodes-features.jsonl --classes <..>.classes.json --answers haiku.jsonl --opus opus.jsonl \
        --artifact stack --thresholds export-stack/section-head.meta.json [--items-as-section] \
        [--domains pool-domains.json --train-pages pages.jsonl --train-pages loop/r1/nat/pages.jsonl ...]

With --domains and --train-pages (page files of `{"p": id}` rows) the table prints three times: all nodes, nodes whose domain
has a page in the training pages (seen), and nodes whose domain has none (unseen). Unseen is the number that counts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def table(rows: list[dict], truth: dict[str, str], classes: list[str], thr: dict[str, float], artifact: str) -> None:
    print(f"{len(rows)} labelled nodes")
    print(f"{'class':18s} {'support':>7s} {'fired':>5s} {'prec':>5s} {'recall':>6s}  thr")
    for cls, t in thr.items():
        if cls == "none" or t > 1:
            continue
        j = classes.index(cls)
        fired = [r for r in rows if max(range(len(classes)), key=lambda i: r["p"][artifact][i]) == j and r["p"][artifact][j] >= t]
        sup = sum(1 for r in rows if truth[r["itemId"]] == cls)
        hit = sum(1 for r in fired if truth[r["itemId"]] == cls)
        print(f"{cls:18s} {sup:7d} {len(fired):5d} {hit / len(fired) if fired else float('nan'):5.2f} {hit / sup if sup else float('nan'):6.2f}  {t}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=Path, required=True)
    ap.add_argument("--classes", type=Path, required=True)
    ap.add_argument("--answers", type=Path, action="append", required=True)
    ap.add_argument("--opus", type=Path, action="append", default=[])
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--thresholds", type=Path, required=True)
    ap.add_argument("--items-as-section", action="store_true",
                    help="an item of ANY class (label none, itemOf set) counts as that class; default: comments items only")
    ap.add_argument("--domains", type=Path, help="JSON {page id: domain hash}")
    ap.add_argument("--train-pages", type=Path, action="append", default=[], help="pages.jsonl of the training pages (repeatable)")
    a = ap.parse_args()
    classes = json.loads(a.classes.read_text())["classes"][a.artifact]
    thr = dict(zip(*[json.loads(a.thresholds.read_text())[k] for k in ("classes", "thresholds")]))
    truth: dict[str, str] = {}
    for p in [*a.answers, *a.opus]:
        for r in jsonl(p):
            if not r.get("error") and r.get("label"):
                item = r.get("itemOf")
                truth[r["itemId"]] = item if item and (a.items_as_section or item == "comments") else r["label"]
    rows = [r for r in jsonl(a.nodes) if r["itemId"] in truth]
    table(rows, truth, classes, thr, a.artifact)
    if a.domains and a.train_pages:
        dom: dict[str, str] = json.loads(a.domains.read_text())
        seen = {dom[p] for f in a.train_pages for p in (json.loads(x)["p"] for x in f.read_text().splitlines() if x.strip()) if p in dom}
        in_seen = [r for r in rows if dom.get(r["pageId"], r["pageId"]) in seen]
        unseen = [r for r in rows if dom.get(r["pageId"], r["pageId"]) not in seen]
        print(f"-- seen domains ({len(seen)} training domains)")
        table(in_seen, truth, classes, thr, a.artifact)
        print("-- unseen domains")
        table(unseen, truth, classes, thr, a.artifact)


if __name__ == "__main__":
    main()
