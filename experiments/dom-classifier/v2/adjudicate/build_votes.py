# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3"]
# ///
"""Every label source's vote per train/val item, in the trainers' label space (#2307).

Input to label_model.py. One row per (item, source):
  {key, axis, source, label, conf?}
key is the pageId (page type) or "pageId#nodeId" (section). Labels are mapped
the way the training pipeline maps them, so sources compare like for like:
page type through policy v8.2's URL rules (policy_rules.py) and the 12-class fold; sections
through the S9 fold and the promo/ad markup rule.

Page-type sources: jev81 (Common Crawl pages), jev8 / jev7 (older pages; v7 only
for the classes v8 reuses 1:1), clef (low-confidence pool pages), url (the
URL-only rule of url_rule_baseline.py), schema (declared schema.org main
entity, schema_labels_v8.py), and adj:<arm> for each adjudicator run given.
Section sources: jev (the Jev-derived training label, v3 with v2
table_of_contents), sem (ARIA / semantic markup), schema (JSON-LD or microdata
on the node), adnet (EasyList / ad, cookie and chat network markup), adj:<arm>.

Reads train/val rows only (every mixed file is filtered by split first).

    uv run --script build_votes.py --cc <2026-10-08-cc> --old <2026-09-22> --cl <cl dir> \
        [--adjudicator arm=<answers.jsonl> ...] [--items cohort.jsonl] --out votes.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from confident_learning import jsonl, load_pages
from policy_rules import page_rules_v82

sys.path.insert(0, str(Path(__file__).parent.parent / "train"))
from schema_labels_v8 import TEMPLATE_PRONE, V7_REUSED, page_label  # noqa: E402
from url_rule_baseline import rule as url_rule  # noqa: E402

SECTION_FOLD = {"recipe_card", "event_details", "chat_widget"}
POOL = ("train", "val")


def section_rules(label: str, ad_marked: bool) -> str:
    fixed = "none" if label in SECTION_FOLD else label
    if fixed in ("ad", "promo_banner"):
        fixed = "ad" if ad_marked else "promo_banner"
    return fixed


def cc_schema_labels(cc: Path) -> dict[str, str]:
    """schema_labels_v8's rule over the Common Crawl pages' declared types (train/val only)."""
    rows = [r for r in jsonl(cc / "section-cc" / "schema-types.jsonl") if r["s"] in POOL]
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_domain[row["d"]].append(row)
    site_wide = {}
    for domain, pages in by_domain.items():
        presence: Counter = Counter()
        for page in pages:
            presence.update(set(page["ld"]) | set(page["md"]))
        site_wide[domain] = (
            set()
            if len(pages) < 4
            else {t for t, n in presence.items() if t in TEMPLATE_PRONE and n / len(pages) >= 0.6}
        )
    out = {}
    for row in rows:
        label, _ = page_label(row, site_wide[row["d"]])
        if label:
            out[row["p"]] = label
    return out


def page_votes(args, pages: dict, keep: set | None):
    def emit(page_id, source, label, conf=None):
        if keep is not None and page_id not in keep:
            return None
        if page_id not in pages or label is None:
            return None
        meta = pages[page_id]
        row = {
            "key": page_id,
            "axis": "page",
            "source": source,
            "label": page_rules_v82(label, meta["url"], meta["platform"]),
        }
        if conf is not None:
            row["conf"] = conf
        return row

    rows = []
    seen_jev = set()
    for split in POOL:
        for r in jsonl(args.cc / "labels" / "pagetype-v81" / f"{split}.jsonl"):
            rows.append(emit(r["pageId"], "jev81", r["pageType"], r.get("confidence")))
            seen_jev.add(r["pageId"])
    for split in (*POOL, "noise"):
        path = args.old / "pagetype-jev-v8" / f"{split}.jsonl"
        for r in jsonl(path) if path.exists() else []:
            if r["pageId"] not in seen_jev and r.get("split", split) in POOL:
                rows.append(emit(r["pageId"], "jev8", r["pageType"], r.get("confidence")))
    for split in POOL:
        for r in jsonl(args.old / "pagetype-jev-v7" / f"{split}.jsonl"):
            if r.get("pageType") in V7_REUSED:
                rows.append(emit(r["pageId"], "jev7", r["pageType"], r.get("confidence")))
    for r in jsonl(args.cc / "labels" / "clef-v81" / "pool.jsonl"):
        rows.append(emit(r["pageId"], "clef", r["choice"], r.get("confidence")))
    for page_id, meta in pages.items():
        rows.append(emit(page_id, "url", url_rule(meta["url"])))
    for r in jsonl(args.old / "pagetype-v8" / "schema-labels-v8.jsonl"):
        if r["split"] in POOL:
            rows.append(emit(r["pageId"], "schema", r["pageType"]))
    for page_id, label in cc_schema_labels(args.cc).items():
        rows.append(emit(page_id, "schema", label))
    return [r for r in rows if r]


def section_votes(args, pages: dict, keep: set | None):
    ad_marked = {f"{r['pageId']}#{r['nodeId']}" for r in jsonl(args.cc / "section-v4" / "ad-markers.jsonl")}
    jev_conf = {}
    for split in POOL:
        for r in jsonl(args.cc / "section-v4" / f"jev-{split}.jsonl"):
            jev_conf[f"{r['pageId']}#{r['nodeId']}"] = r.get("confidence")
    rows = []
    with gzip.open(args.cl / "section-pool.jsonl.gz", "rt") as handle:
        for line in handle:
            r = json.loads(line)
            key = f"{r['pageId']}#{r['nodeId']}"
            if keep is None or key in keep:
                rows.append(
                    {"key": key, "axis": "section", "source": "jev", "label": r["given"], "conf": jev_conf.get(key)}
                )
    for r in jsonl(args.cc / "section-v4" / "node-labels.jsonl"):
        if r.get("s") not in POOL or "n" not in r:
            continue
        key = f"{r['p']}#{r['n']}"
        if keep is not None and key not in keep:
            continue
        marked = key in ad_marked
        if r.get("sem"):
            rows.append({"key": key, "axis": "section", "source": "sem", "label": section_rules(r["sem"], marked)})
        if r.get("ld") or r.get("md"):
            rows.append(
                {
                    "key": key,
                    "axis": "section",
                    "source": "schema",
                    "label": section_rules(r.get("ld") or r["md"], marked),
                }
            )
        if r.get("list") or r.get("net"):
            rows.append(
                {
                    "key": key,
                    "axis": "section",
                    "source": "adnet",
                    "label": section_rules(r.get("list") or r["net"], marked),
                }
            )
    return rows


def adjudicator_votes(specs: list[str], items: dict):
    rows = []
    for spec in specs:
        arm, path = spec.split("=", 1)
        for a in jsonl(Path(path)):
            item = items.get(a["itemId"])
            if not item or a.get("error"):
                continue
            key = item["pageId"] if item["axis"] == "page" else f"{item['pageId']}#{item['nodeId']}"
            rows.append(
                {"key": key, "axis": item["axis"], "source": f"adj:{arm}", "label": a["label"], "conf": a["confidence"]}
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", type=Path, required=True)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--cl", type=Path, required=True)
    parser.add_argument("--items", type=Path, help="restrict to these items (e.g. the calibration cohort)")
    parser.add_argument("--adjudicator", action="append", default=[], help="arm=answers.jsonl (needs --items)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pages = load_pages(args.cc)
    items = {i["itemId"]: i for i in jsonl(args.items)} if args.items else {}
    keep_pages = {i["pageId"] for i in items.values() if i["axis"] == "page"} if items else None
    keep_nodes = {f"{i['pageId']}#{i['nodeId']}" for i in items.values() if i["axis"] == "section"} if items else None
    rows = (
        page_votes(args, pages, keep_pages)
        + section_votes(args, pages, keep_nodes)
        + adjudicator_votes(args.adjudicator, items)
    )
    with args.out.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(
        json.dumps(
            {"rows": len(rows), "bySource": dict(Counter(f"{r['axis']}:{r['source']}" for r in rows).most_common())},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
