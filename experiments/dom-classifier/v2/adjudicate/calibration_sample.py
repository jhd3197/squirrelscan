# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3"]
# ///
"""Calibration-gold candidates from TRAIN/VAL pages (#2307), separate from test gold.

Draws page items and section items stratified by the training label, with the
confident-learning candidates oversampled to half of every class, and spread
over platform x language cells by round-robin. Each item records its stratum
size and inclusion weight, so per-source accuracy measured on the reviewed
sample can be reweighted to the pool (the label model needs that: a class's
candidate half is a far smaller share of the pool than of the sample).

Items are oversampled (~20%) because some will not render usably; the final
cohort is cut after rendering. 40% of each stratum is marked `pilot`.
Only train/val pages are eligible, a page needs stored HTML, and no page of
gold-v1 (test gold) can be drawn.

    uv run --script calibration_sample.py --cc <2026-10-08-cc> --cl <cl dir> \
        --stores a.db:b.db --gold <gold-v1/labels/queue-pages.jsonl> --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from confident_learning import jsonl, load_pages

SEED = "gold-cal-v1"
# Provisional: the pilot is re-cut from the usable renders, same share per stratum.
PILOT_SHARE = 0.4
PAGE_CLASSES = [
    "homepage",
    "product_detail",
    "product_listing",
    "article",
    "blog_post",
    "about",
    "contact",
    "docs",
    "legal",
    "pricing",
    "listing_index",
    "other",
]


def rank(key: str) -> str:
    return hashlib.sha256(f"{SEED}:{key}".encode()).hexdigest()


def stored_prefixes(stores: list[str]) -> set[str]:
    out: set[str] = set()
    for path in stores:
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        out.update(row[0] for row in db.execute("select substr(hash, 1, 16) from content"))
        db.close()
    return out


def spread(
    pool: list[dict], n: int, per_domain: Counter, domain_cap: int, per_page: Counter | None = None
) -> list[dict]:
    """Round-robin over platform x language cells, hash order within a cell."""
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for item in sorted(pool, key=lambda r: rank(r["key"])):
        cells[(item["platform"], item["lang"])].append(item)
    order = sorted(cells, key=lambda c: rank(f"cell:{c}"))
    chosen: list[dict] = []
    cursor = dict.fromkeys(order, 0)
    while len(chosen) < n and any(cursor[c] < len(cells[c]) for c in order):
        for c in order:
            while cursor[c] < len(cells[c]):
                item = cells[c][cursor[c]]
                cursor[c] += 1
                if per_domain[item["domain"]] >= domain_cap:
                    continue
                if per_page is not None and per_page[item["pageId"]] >= 1:
                    continue
                per_domain[item["domain"]] += 1
                if per_page is not None:
                    per_page[item["pageId"]] += 1
                chosen.append(item)
                break
            if len(chosen) >= n:
                break
    return chosen


def draw(universe: list[dict], classes: list[str], quota, domain_cap: int, per_page: Counter | None, axis: str):
    per_domain: Counter = Counter()
    items = []
    for cls in classes:
        half = quota(cls) // 2
        cand_pool = [r for r in universe if r["given"] == cls and r["cand"]]
        rest_pool = [r for r in universe if r["given"] == cls and not r["cand"]]
        cand = spread(cand_pool, half, per_domain, domain_cap, per_page)
        # A thin candidate half is topped up from the rest of the class.
        rest = spread(rest_pool, quota(cls) - len(cand), per_domain, domain_cap, per_page)
        for name, got, pool in (("cand", cand, cand_pool), ("rest", rest, rest_pool)):
            for i, item in enumerate(got):
                items.append(
                    {
                        **item,
                        "axis": axis,
                        "stratum": f"{cls}|{name}",
                        "stratumSize": len(pool),
                        "stratumSampled": len(got),
                        "weight": round(len(pool) / len(got), 3),
                        "pilot": i < round(PILOT_SHARE * len(got)),
                    }
                )
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", type=Path, required=True)
    parser.add_argument("--cl", type=Path, required=True)
    parser.add_argument("--stores", required=True)
    parser.add_argument("--gold", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--page-per-class", type=int, default=30)
    parser.add_argument("--section-per-class", type=int, default=18)
    parser.add_argument("--section-none", type=int, default=40)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pages = load_pages(args.cc)
    gold = {r["pageId"] for path in args.gold for r in jsonl(path)}
    stored = stored_prefixes(args.stores.split(":"))
    # html-lang "unk" is 0.3% of the pool; as its own round-robin cell it would take a tenth of the sample.
    eligible = {p for p in pages if p in stored and p not in gold and pages[p]["lang"] != "unk"}

    def meta(p: str) -> dict:
        return {k: pages[p][k] for k in ("split", "corpus", "platform", "lang", "domain", "url")}

    cand_pages = {r["pageId"]: r for r in jsonl(args.cl / "pagetype-candidates.jsonl")}
    page_universe = []
    for r in jsonl(args.cl / "pagetype-pool.jsonl.gz"):
        p = r["pageId"]
        if p not in eligible:
            continue
        c = cand_pages.get(p)
        page_universe.append(
            {
                "key": p,
                "pageId": p,
                "given": r["given"],
                "cand": c is not None,
                "suggested": c["suggested"] if c else None,
                "sources": c["sources"] if c else [],
                "oofPred": r["pred"],
                "oofConf": r["conf"],
                **meta(p),
            }
        )
    page_items = draw(page_universe, PAGE_CLASSES, lambda cls: args.page_per_class, 1, None, "page")

    section_classes = json.loads((args.cc / "cv" / "section" / "cv-report.json").read_text())["classes"]
    cand_nodes = {
        (r["pageId"], r["nodeId"]): r for r in jsonl(args.cl / "section-candidates.jsonl") if r["src"] == "jev"
    }
    section_universe = []
    for r in jsonl(args.cl / "section-pool.jsonl.gz"):
        p = r["pageId"]
        if p not in eligible:
            continue
        c = cand_nodes.get((p, r["nodeId"]))
        section_universe.append(
            {
                "key": f"{p}#{r['nodeId']}",
                "pageId": p,
                "nodeId": r["nodeId"],
                "given": r["given"],
                "cand": c is not None,
                "suggested": c["suggested"] if c else None,
                "oofPred": r["pred"],
                "oofConf": r["conf"],
                **meta(p),
            }
        )
    section_items = draw(
        section_universe,
        section_classes,
        lambda cls: args.section_none if cls == "none" else args.section_per_class,
        2,
        Counter(),
        "section",
    )
    items = page_items + section_items
    for item in items:
        item["itemId"] = f"cal-{item['axis']}-{rank(item['key'])[:12]}"
        item.pop("key")
    with (args.out / "calibration-candidates.jsonl").open("w") as handle:
        for item in items:
            handle.write(json.dumps(item) + "\n")
    summary = {
        "eligiblePages": len(eligible),
        "poolPages": len(pages),
        "storedPoolPages": sum(1 for p in pages if p in stored),
        "goldExcluded": sum(1 for p in pages if p in gold),
        "pageItems": len(page_items),
        "sectionItems": len(section_items),
        "pilot": Counter(f"{i['axis']}:{i['pilot']}" for i in items),
        "pageByStratum": Counter(i["stratum"] for i in page_items),
        "sectionByStratum": Counter(i["stratum"] for i in section_items),
        "platform": Counter(f"{i['axis']}:{i['platform']}" for i in items),
        "lang": Counter(f"{i['axis']}:{i['lang']}" for i in items),
        "corpus": Counter(f"{i['axis']}:{i['corpus']}" for i in items),
        "renderPages": len({i["pageId"] for i in items}),
    }
    (args.out / "calibration-summary.json").write_text(json.dumps(summary, indent=1, default=dict))
    print(json.dumps(summary, indent=1, default=dict))


if __name__ == "__main__":
    main()
