# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Our page-type labels vs what the site's own schema declares (#2307).

Schema is a second opinion, never a label by itself. For every page that has both,
this counts agree / schema-missing / conflict per class, and writes the conflicts
as a review queue (ids and classes only, no urls or text).

  uv run --script schema_vs_labels.py --schema <extract_schema out>/schema.jsonl \
      --labels teacher=<pagetype-teacher.jsonl> --labels jev=<pagetype-v81/train.jsonl> \
      --sealed <sealed-domains.jsonl> --out <dir>

Join key: sha256[:16] of the page bytes (corpus `sha`) = the labels' `pageId`/`contentHash` prefix.
Sealed (calibration/test) domains and bucket >= 90 are excluded: this is analysis for
training decisions, so it never reads evaluation data.
Schema class comes from train/schema_labels_v8.py (page_label), so its type map and
noise filters (site-wide types, template-prone types per domain, listings) apply.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("schema_labels_v8", HERE.parent / "train" / "schema_labels_v8.py")
assert SPEC and SPEC.loader
V8 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V8)

#: label classes that schema can speak about; others (other, legal, ...) cannot be confirmed by it
SCHEMA_CLASSES = set(V8.TYPE_TO_V8.values()) | {"product_listing", "listing_index", "homepage"}


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def given_label(row: dict) -> str | None:
    return row.get("pageType")


#: Schema cannot tell these apart, so they agree within a family: Article vs BlogPosting is a
#: site convention, ItemList says "listing" not what the items are.
FAMILY = {"blog_post": "editorial", "article": "editorial", "product_listing": "listing", "listing_index": "listing"}
#: LocalBusiness describes the site's entity, so it is compatible with any site-level page
#: (it still counts as a conflict for a page labelled article, product, etc.).
SITE_LEVEL = {"homepage", "about", "contact", "legal", "other", "local_business", "listing"}


def agrees(given: str, schema: str) -> bool:
    fg, fs = FAMILY.get(given, given), FAMILY.get(schema, schema)
    return fg == fs or (fs == "local_business" and fg in SITE_LEVEL)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--labels", action="append", required=True, help="name=path (jsonl with pageId|contentHash + pageType)")
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sealed = {r["d"] for r in load(args.sealed)}
    if len(sealed) < 1000:
        raise SystemExit("sealed list too small")

    rows = [r for r in load(args.schema) if r.get("ok") and r["d"] not in sealed]
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_domain[row["d"]].append(row)
    site_wide: dict[str, set[str]] = {}
    for domain, pages in by_domain.items():
        presence: Counter = Counter()
        if len(pages) >= 4:
            for page in pages:
                presence.update(set(page["ld"]) | set(page["md"]))
        site_wide[domain] = {t for t, n in presence.items() if t in V8.TEMPLATE_PRONE and n / len(pages) >= 0.6}
    schema_of: dict[str, tuple[str | None, str, dict]] = {}
    reasons: Counter = Counter()
    for row in rows:
        label, reason = V8.page_label(row, site_wide[row["d"]])
        reasons[reason] += 1
        schema_of[row["sha"][:16]] = (label, reason, {"og": row.get("og"), "ld": sorted(row["ld"])[:6]})

    args.out.mkdir(parents=True, exist_ok=True)
    report: dict = {"schemaPages": len(rows), "schemaReasons": dict(reasons)}
    queue = args.out / "conflicts.jsonl"
    with queue.open("w") as sink:
        for spec in args.labels:
            name, _, path = spec.partition("=")
            matrix: Counter = Counter()
            per_class: dict[str, Counter] = defaultdict(Counter)
            joined = 0
            for row in load(Path(path)):
                key = (row.get("contentHash") or row.get("pageId") or "")[:16]
                if key not in schema_of or not given_label(row):
                    continue
                joined += 1
                given = given_label(row)
                label, reason, detail = schema_of[key]
                if label is None:
                    outcome = "schema-silent"  # no main entity: says nothing either way
                elif agrees(given, label):
                    outcome = "agree"
                else:
                    outcome = "conflict"
                    sink.write(json.dumps({"set": name, "pageId": key, "given": given, "schema": label, "reason": reason, **detail}) + "\n")
                per_class[given][outcome] += 1
                matrix[(given, label or "-")] += 1
            report[name] = {
                "joined": joined,
                "perGivenClass": {c: dict(o) for c, o in sorted(per_class.items())},
                "agreeRateWhereSchemaSpeaks": {
                    c: round(o["agree"] / (o["agree"] + o["conflict"]), 3)
                    for c, o in sorted(per_class.items())
                    if o["agree"] + o["conflict"] >= 10
                },
                "topConflicts": {f"{g}->{s}": n for (g, s), n in matrix.most_common(60) if g != s and s != "-"}
                                and dict(list({f"{g}->{s}": n for (g, s), n in matrix.most_common(200) if g != s and s != "-"}.items())[:25]),
            }
    (args.out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: (v if not isinstance(v, dict) or k.startswith("schema") else v["joined"]) for k, v in report.items() if k != "schemaReasons"}))


if __name__ == "__main__":
    main()
