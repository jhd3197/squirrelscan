# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3"]
# ///
"""Candidate deterministic page-type policy rules, counted on TRAIN pages (#2307).

Drafted from the visual adjudicator's pilot disagreements with labels that the
OOF model also held. NOT applied: calibration gold confirms each rule first,
then the trainer applies it through --policy-rules. Each rule reports how many
train pages it would relabel (by current label and platform), and, on pilot
items where it fires, how often the Sonnet adjudicator already picked the
rule's class.

    uv run --script rule_candidates.py --cc <2026-10-08-cc> --cl <cl dir> --cohort cohort.jsonl \
        --sonnet answers.jsonl --out rules.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

from confident_learning import jsonl, load_pages

# ISO 639-1 codes (plus fil), optionally with a region: "de", "en-us", "pt_br", "zh-hant".
ISO639_1 = set(
    "aa ab af ak am an ar as av ay az ba be bg bh bi bm bn bo br bs ca ce ch co cr cs cu cv cy da de dv dz ee el "
    "en eo es et eu fa ff fi fj fo fr fy ga gd gl gn gu gv ha he hi ho hr ht hu hy hz ia id ie ig ii ik io is it iu "
    "ja jv ka kg ki kj kk kl km kn ko kr ks ku kv kw ky la lb lg li ln lo lt lu lv mg mh mi mk ml mn mr ms mt my na "
    "nb nd ne ng nl nn no nr nv ny oc oj om or os pa pi pl ps pt qu rm rn ro ru rw sa sc sd se sg si sk sl sm sn so "
    "sq sr ss st su sv sw ta te tg th ti tk tl tn to tr ts tt tw ty ug uk ur uz ve vi vo wa wo xh yi yo za zh zu fil".split()
)
LOCALE = re.compile(r"^([a-z]{2,3})([-_]([a-z]{2}|hans|hant))?$")
NEWS = {
    "news",
    "press",
    "presse",
    "newsroom",
    "actualites",
    "actualite",
    "noticias",
    "nachrichten",
    "aktuelles",
    "notizie",
    "nieuws",
    "novedades",
    "comunicados",
    "media-releases",
}
SUPPORT = {
    "support",
    "help",
    "hilfe",
    "aide",
    "ayuda",
    "soporte",
    "assistance",
    "helpcenter",
    "help-center",
    "kundenservice",
}
SERVICES = {
    "services",
    "service",
    "leistungen",
    "dienstleistungen",
    "servicios",
    "servizi",
    "uslugi",
    "prestations",
    "diensten",
    "solutions",
}
LOCALE_EXCLUDE = {
    "api",
    "app",
    "faq",
    "art",
    "pro",
    "tag",
    "web",
    "dev",
    "job",
    "shop",
    "blog",
    "news",
    "docs",
    "about",
}


def segments(url: str) -> list[str]:
    return [unquote(s).lower() for s in urlsplit(url).path.split("/") if s]


def rules(url: str, platform: str, label: str, feats: dict) -> list[tuple[str, str]]:
    """(rule id, proposed class) for every rule that fires and would change the label."""
    segs = segments(url)
    query = urlsplit(url).query
    out = []
    if platform == "wix" and len(segs) >= 2 and segs[0] == "post" and label != "blog_post":
        out.append(("R1 wix /post/<slug> is a blog entry", "blog_post"))
    locale = LOCALE.match(segs[0]) if len(segs) == 1 else None
    if locale and locale.group(1) in ISO639_1 and not query and segs[0] not in LOCALE_EXCLUDE and label != "homepage":
        out.append(("R2 locale root /xx/ or /xx-yy/ is the homepage", "homepage"))
    if label == "blog_post" and "blog" not in " ".join(segs) and NEWS & set(segs[:-1]):
        out.append(("R3 news/press path without any blog signal is an article (rule 1)", "article"))
    if (
        label == "contact"
        and SUPPORT & set(segs)
        and not ({"contact", "kontakt", "contacto", "contact-us"} & set(segs))
    ):
        out.append(("R4 support/help path labelled contact: rule 7 says docs (review, not auto-fix)", "docs"))
    if (
        label == "product_detail"
        and feats.get("priceHitCount", 0) == 0
        and feats.get("cartHitCount", 0) == 0
        and feats.get("urlSlug_product", 0) == 0
        and feats.get("schema") != "product_detail"
    ):
        out.append(("R5 product_detail: no price, no cart text, no /product/ slug, no Product schema", "other"))
    if label in ("product_detail", "product_listing") and SERVICES & set(segs[:-1] or segs):
        out.append(("R6 product label under a services path is a service page", "other"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", type=Path, required=True)
    parser.add_argument("--cl", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--sonnet", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pages = load_pages(args.cc)
    schema = json.loads((args.cc / "silver-v4" / "feature-schema.json").read_text())
    names = schema["pageFeatures"]
    wanted = {n: names.index(n) for n in ("priceHitCount", "cartHitCount", "urlSlug_product")}
    feats: dict[str, dict] = {}
    with gzip.open(args.cc / "silver-v4" / "pages-train.jsonl.gz", "rt") as handle:
        for line in handle:
            row = json.loads(line)
            feats[row["p"]] = {n: row["f"][i] for n, i in wanted.items()}
    # Declared schema main entity (the free page-type source), train only.
    old = args.cc.parent / "2026-09-22" / "pagetype-v8" / "schema-labels-v8.jsonl"
    for r in jsonl(old):
        if r["pageId"] in feats:
            feats[r["pageId"]]["schema"] = r["pageType"]
    from build_votes import cc_schema_labels

    for page_id, label in cc_schema_labels(args.cc).items():
        if page_id in feats:
            feats[page_id]["schema"] = label
    labels = {}
    with gzip.open(args.cl / "pagetype-pool.jsonl.gz", "rt") as handle:
        for line in handle:
            row = json.loads(line)
            labels[row["pageId"]] = row
    hits: dict[str, Counter] = {}
    by_platform: dict[str, Counter] = {}
    oof_agrees: dict[str, Counter] = {}
    for page_id, row in labels.items():
        meta = pages.get(page_id)
        if not meta or meta["split"] != "train" or page_id not in feats:
            continue
        for rule_id, proposed in rules(meta["url"], meta["platform"], row["given"], feats[page_id]):
            hits.setdefault(rule_id, Counter())[f"{row['given']}->{proposed}"] += 1
            by_platform.setdefault(rule_id, Counter())[meta["platform"]] += 1
            oof_agrees.setdefault(rule_id, Counter())[
                "oof=proposed"
                if row.get("pred") == proposed
                else "oof=label"
                if row.get("pred") == row["given"]
                else "oof=other"
            ] += 1
    sonnet = {a["itemId"]: a for a in jsonl(args.sonnet) if not a.get("error")}
    pilot: dict[str, Counter] = {}
    for item in jsonl(args.cohort):
        if item["axis"] != "page" or item["itemId"] not in sonnet:
            continue
        f = feats.get(item["pageId"], {"priceHitCount": 1, "cartHitCount": 1})
        for rule_id, proposed in rules(item["url"], item["platform"], item["given"], f):
            pilot.setdefault(rule_id, Counter())[
                "sonnet=proposed"
                if sonnet[item["itemId"]]["label"] == proposed
                else "sonnet=label"
                if sonnet[item["itemId"]]["label"] == item["given"]
                else "sonnet=other"
            ] += 1
    report = {
        rule_id: {
            "trainPages": sum(c.values()),
            "byChange": dict(c.most_common()),
            "byPlatform": dict(by_platform[rule_id].most_common()),
            "oofModel": dict(oof_agrees[rule_id]),
            "pilotSonnet": dict(pilot.get(rule_id, {})),
        }
        for rule_id, c in sorted(hits.items())
    }
    args.out.write_text(json.dumps(report, indent=1))
    for rule_id, r in report.items():
        print(
            f"{rule_id}: {r['trainPages']} train pages {r['byChange']} | OOF {r['oofModel']} | pilot {r['pilotSonnet']} | {dict(list(r['byPlatform'].items())[:4])}"
        )


if __name__ == "__main__":
    main()
