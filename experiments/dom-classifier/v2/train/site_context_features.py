"""Per-page cross-page features from `corpus/site-context.ts` output (#2307 v8.1).

Inbound anchors: for each dataset page, the internal links on OTHER pages of the
same site that point at it, counted by region (nav / header / footer / aside /
body) and by what their anchor text says (about, contact, blog, ...), in several
languages. Template: how many of the site's dataset pages share this page's
structural skeleton.

Only pages inside the dataset can contribute anchors (up to 25 per domain), so
these are a lower bound on what a full crawl sees at audit time.

Output: one row per page, {"p": pageId, "f": [floats]}, plus the feature names.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

CATEGORIES = {
    "about": r"\babout\b|über uns|ueber uns|a propos|à propos|chi siamo|sobre|quienes somos|who we are|our story|team",
    "contact": r"contact|kontakt|contacto|contatti|get in touch|reach us",
    "blog": r"\bblog\b|articles?\b|insights|stories|journal",
    "news": r"\bnews\b|press|newsroom|aktuelles|actualit|noticias",
    "docs": r"\bdocs?\b|documentation|guide|help|support|tutorial|api|reference|knowledge",
    "pricing": r"pricing|prices|plans\b|preise|tarif|precios",
    "legal": r"privacy|terms|legal|impressum|datenschutz|cookie|imprint|conditions|agb|disclaimer",
    "home": r"^home$|startseite|accueil|inicio|^start$",
    "shop": r"\bshop\b|store|products?|collections?|catalog|buy",
    "careers": r"career|jobs|karriere|work with us|hiring",
    "faq": r"\bfaq|questions",
}
REGIONS = ["nav", "header", "footer", "aside", "body"]
COMPILED = {k: re.compile(v) for k, v in CATEGORIES.items()}


def names() -> list[str]:
    out = [f"inbound_{r}" for r in ["total", *REGIONS]]
    out += [f"inbound_anchor_{c}" for c in CATEGORIES]
    out += [f"inbound_chrome_{c}" for c in CATEGORIES]
    out += ["template_cluster_size", "template_cluster_share", "outbound_internal"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", required=True, help="site-context.jsonl")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    pages = [json.loads(line) for line in Path(args.context).read_text().splitlines() if line]
    by_url: dict[tuple[str, str], str] = {}
    domain_pages: Counter = Counter()
    clusters: Counter = Counter()
    for page in pages:
        if page.get("url"):
            by_url[(page["d"], page["url"])] = page["p"]
        domain_pages[page["d"]] += 1
        clusters[(page["d"], page["skeleton"])] += 1
    inbound: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for page in pages:
        for link in page["links"]:
            target = by_url.get((page["d"], link["t"]))
            if target and target != page["p"]:
                inbound[target].append((link["a"], link["r"]))

    with Path(args.out).open("w") as handle:
        for page in pages:
            links = inbound.get(page["p"], [])
            regions = Counter(r for _, r in links)
            f = [math.log1p(len(links))] + [math.log1p(regions[r]) for r in REGIONS]
            total = max(len(links), 1)
            anchor = Counter()
            chrome = Counter()
            for text, region in links:
                for cat, rx in COMPILED.items():
                    if rx.search(text):
                        anchor[cat] += 1
                        if region in ("nav", "header", "footer"):
                            chrome[cat] += 1
            f += [anchor[c] / total for c in CATEGORIES]
            f += [1.0 if chrome[c] else 0.0 for c in CATEGORIES]
            size = clusters[(page["d"], page["skeleton"])]
            f += [math.log1p(size), size / domain_pages[page["d"]], math.log1p(len(page["links"]))]
            handle.write(json.dumps({"p": page["p"], "f": [round(v, 5) for v in f]}) + "\n")
    Path(args.out).with_suffix(".names.json").write_text(json.dumps(names()) + "\n")
    print(json.dumps({"pages": len(pages), "withInbound": sum(1 for p in pages if inbound.get(p["p"])),
                      "features": len(names())}))


if __name__ == "__main__":
    main()
