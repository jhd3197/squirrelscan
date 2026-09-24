"""URL-only page-type rule (no model), and where the page-type head beats it.

The rule reads nothing but the URL: root, path keywords in several languages,
and path shape (dates, product ids). It is the baseline every page-type claim
has to beat: a class the rule already gets right ships as a rule.

Input: the trainer's per-page holdout predictions (`--dump`ed by
train_pagetype_v8 --score-holdout) and the dataset's page index for urls.
Output: per class, rule precision/coverage, model precision/coverage at its
threshold, and the model on the pages the rule does NOT fire on.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

# (class, pattern over "/"-joined lower-cased path segments), first match wins.
RULES: list[tuple[str, re.Pattern]] = [
    ("legal", re.compile(r"(^|/)(privacy|privacy-policy|terms|terms-of-service|terms-and-conditions|tos|legal|"
                         r"impressum|imprint|datenschutz|cookies?|cookie-policy|agb|disclaimer|gdpr|"
                         r"accessibility|mentions-legales|aviso-legal|privacidad|note-legali)(/|$|\.)")),
    ("pricing", re.compile(r"(^|/)(pricing|prices|plans|preise|tarifs?|precios|prezzi)(/|$|\.)")),
    ("contact", re.compile(r"(^|/)(contact|contact-us|contactus|kontakt|contacto|contatti|contactez-nous|"
                           r"get-in-touch)(/|$|\.)")),
    ("about", re.compile(r"(^|/)(about|about-us|aboutus|team|our-team|company|who-we-are|our-story|mission|"
                         r"uber-uns|ueber-uns|a-propos|qui-sommes-nous|chi-siamo|sobre|sobre-nosotros|"
                         r"quienes-somos|nosotros|over-ons|om-oss)(/|$|\.)")),
    ("docs", re.compile(r"(^|/)(docs?|documentation|guides?|help|kb|knowledge-?base|api|reference|"
                        r"manual|tutorials?|support/articles)(/|$|\.)")),
    ("blog_post", re.compile(r"(^|/)(blog|posts?)/[^/]+")),
    ("listing_index", re.compile(r"(^|/)(blog|news|posts|articles|tag|tags|category|categories|archive|"
                                 r"events|jobs|careers)/?$|(^|/)page/\d+/?$|(^|/)(tag|author)/[^/]+/?$")),
    ("article", re.compile(r"(^|/)(news|articles?|press|stories|insights)/[^/]+|(^|/)(19|20)\d\d/\d\d/")),
    ("product_listing", re.compile(r"(^|/)(collections?|shop|store|catalog|categories|category|c)"
                                   r"(/[^/]+)?/?$")),
    ("product_detail", re.compile(r"(^|/)(products?|p|item|items|dp|artikel|produit|producto|prodotto)/[^/]+")),
]


def rule(url: str) -> str | None:
    parts = urlsplit(url)
    path = parts.path.lower().rstrip("/")
    if not path.strip("/") and not parts.query:
        return "homepage"
    for cls, pattern in RULES:
        if pattern.search(path.lstrip("/")):
            return cls
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="holdout-predictions.jsonl")
    parser.add_argument("--page-index", required=True, help="<dataset>/page-index.jsonl.gz")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    urls = {}
    with gzip.open(args.page_index, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            urls[row["p"]] = row["u"]
    rows = [json.loads(line) for line in Path(args.predictions).read_text().splitlines() if line]
    classes = sorted({r["label"] for r in rows} | {c for c, _ in RULES} | {"homepage"})
    table = {}
    for cls in classes:
        support = sum(r["label"] == cls for r in rows)
        rule_fires = [r for r in rows if rule(urls.get(r["pageId"], "")) == cls]
        model_fires = [r for r in rows if r["fires"] and r["pred"] == cls]
        rule_silent = [r for r in rows if rule(urls.get(r["pageId"], "")) is None]
        rest_support = sum(r["label"] == cls for r in rule_silent)
        rest_fires = [r for r in rule_silent if r["fires"] and r["pred"] == cls]

        def pc(fired, sup):
            n = len(fired)
            ok = sum(r["label"] == cls for r in fired)
            return {"fires": n, "precision": round(ok / n, 4) if n else None,
                    "coverage": round(ok / sup, 4) if sup else None}

        table[cls] = {"support": support, "urlRule": pc(rule_fires, support),
                      "model": pc(model_fires, support),
                      "modelWhereRuleSilent": {"support": rest_support, **pc(rest_fires, rest_support)}}
    Path(args.out).write_text(json.dumps(table, indent=2) + "\n")
    print(f"{'class':16s} {'sup':>5s} | rule P / cov | model P / cov | model where rule silent: sup P / cov")
    for cls, v in table.items():
        u, m, r = v["urlRule"], v["model"], v["modelWhereRuleSilent"]
        print(f"{cls:16s} {v['support']:5d} | {u['precision']} / {u['coverage']} | "
              f"{m['precision']} / {m['coverage']} | {r['support']} {r['precision']} / {r['coverage']}")


if __name__ == "__main__":
    main()
