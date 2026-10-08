# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3"]
# ///
"""Relabel candidates created by page-type policy v8.2, train/val pool (#2307).

v8.2 widens product_detail / product_listing from purchasable products to
products, plans and service offerings, and `other` stops covering landing and
service pages about one offering. Labels made under v8 / v8.1 are stale there:

  other or listing_index, about ONE product/plan/service  -> product_detail
  listing_index that is a services/solutions overview      -> product_listing

No rule can decide "about one offering", so this ranks pages for the visual
adjudicator by independent signals that some source already leans that way:
Jev's own probability mass on the product classes, Clef's vote, the OOF model's
prediction, and offering-shaped URL paths. A page needs at least one signal.

    uv run --script v82_candidates.py --cc <2026-10-08-cc> --old <2026-09-22> --cl <cl dir> --out <file>
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

from confident_learning import jsonl, load_pages
from policy_rules import segments

OFFERING = {
    "product", "products", "solution", "solutions", "service", "services", "feature", "features", "platform",
    "plans", "leistungen", "dienstleistungen", "leistung", "angebot", "angebote", "servicios", "servicio",
    "soluciones", "servizi", "soluzioni", "uslugi", "prestations", "offres", "offre", "diensten", "oplossingen",
    "produkte", "produkt", "produits", "produit", "productos", "producto", "prodotti", "integrations", "tools",
}  # fmt: skip
PRODUCT = ("product_detail", "product_listing")
STALE = ("other", "listing_index")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", type=Path, required=True)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--cl", type=Path, required=True)
    parser.add_argument("--min-jev-mass", type=float, default=0.15)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pages = load_pages(args.cc)
    jev: dict[str, dict] = {}
    for split in ("train", "val"):
        for r in jsonl(args.cc / "labels" / "pagetype-v81" / f"{split}.jsonl"):
            jev[r["pageId"]] = r.get("probabilities") or {}
        for r in jsonl(args.old / "pagetype-jev-v8" / f"{split}.jsonl"):
            jev.setdefault(r["pageId"], r.get("probabilities") or {})
    clef = {r["pageId"]: r["choice"] for r in jsonl(args.cc / "labels" / "clef-v81" / "pool.jsonl")}
    rows = []
    with gzip.open(args.cl / "pagetype-pool.jsonl.gz", "rt") as handle:
        for line in handle:
            r = json.loads(line)
            page_id = r["pageId"]
            meta = pages.get(page_id)
            if not meta or r["given"] not in STALE:
                continue
            segs = segments(meta["url"])
            probs = jev.get(page_id, {})
            mass = {c: probs.get(c, 0.0) for c in PRODUCT}
            signals = []
            if sum(mass.values()) >= args.min_jev_mass:
                signals.append("jev_mass")
            if clef.get(page_id) in PRODUCT:
                signals.append(f"clef:{clef[page_id]}")
            if r.get("pred") in PRODUCT:
                signals.append(f"oof:{r['pred']}")
            overview = bool(segs) and segs[-1] in OFFERING
            under = bool(OFFERING & set(segs[:-1]))
            if overview:
                signals.append("url:overview")
            if under:
                signals.append("url:offering_path")
            if not signals:
                continue
            # Proposed class: an offering overview (or an index) leans to the listing, one offering to the detail.
            listing = overview or r["given"] == "listing_index" and mass["product_listing"] >= mass["product_detail"]
            rows.append(
                {
                    "pageId": page_id,
                    **{k: meta[k] for k in ("split", "corpus", "platform", "lang", "domain")},
                    "given": r["given"],
                    "suggested": "product_listing" if listing else "product_detail",
                    "signals": signals,
                    "jevProductMass": round(sum(mass.values()), 3),
                    "sources": ["policy_v82"],
                }
            )
    rows.sort(key=lambda x: (-len(x["signals"]), -x["jevProductMass"]))
    for rank, row in enumerate(rows):
        row["rank"] = rank
    args.out.write_text("".join(json.dumps(x) + "\n" for x in rows))
    print(
        json.dumps(
            {
                "candidates": len(rows),
                "bySplit": Counter(x["split"] for x in rows),
                "byChange": Counter(f"{x['given']}->{x['suggested']}" for x in rows),
                "bySignalCount": Counter(len(x["signals"]) for x in rows),
                "bySignal": Counter(s.split(":")[0] for x in rows for s in x["signals"]),
                "byPlatform": Counter(x["platform"] for x in rows).most_common(8),
                "trainWith2PlusSignals": sum(1 for x in rows if x["split"] == "train" and len(x["signals"]) >= 2),
            },
            default=dict,
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
