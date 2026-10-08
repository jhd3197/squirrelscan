# /// script
# requires-python = ">=3.12"
# dependencies = ["duckdb==1.4.3", "httpx==0.28.1"]
# ///
"""Common Crawl diversity extension for the DOM classifier corpus (#2307).

The squirrelscan-sourced corpus is dev-site skewed (Next.js and custom
landing pages, English, .app/.io/.ai). Customers run WordPress, WooCommerce and
Shopify, more often in German or Spanish. This samples NEW registrable domains
from Common Crawl by platform, language and vertical, and writes their HTML into
a dedicated squirrel-shaped store (content-store.db + projects/<x>/project.db),
so build-dataset.ts, the Jev runners and site-context.ts read it unchanged.

No S3 credentials: the columnar index is read over https with column
projection, one row group at a time, and each page is ONE http range request
for its WARC record. At most --concurrency (<= 8) requests are in flight, with
backoff on 429/503.

Platform is not in the index, so candidate domains are pre-screened by URL
shape (a /collections/ + /products/ site is probably Shopify), their homepage
is fetched and fingerprinted, and quotas are filled from the fingerprints.
Schema markup is never read for selection here; URL shape picks the page mix.

Subcommands (all paths private; nothing here names a site):
  meta         row-group url_surtkey ranges of every index part (footers only)
  plan         pick row groups per TLD
  index        per-domain aggregates + candidate urls per planned row group
  candidates   merge, split/exclusion via cc-domains.ts, homepage sample
  fetch        WARC range fetch of a queue into the store (resumable)
  fingerprint  platform + language of fetched homepages
  select       quota fill, url mix per domain -> page queue
  finalize     projects/cc/project.db rows for every selected, fetched page
  tag-pages    platform + language for every page of a dataset (old and new)
"""

from __future__ import annotations

import argparse
import codecs
import csv
import gzip
import hashlib
import json
import re
import sqlite3
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CC_BASE = "https://data.commoncrawl.org/"
USER_AGENT = "squirrelscan-dom-classifier-research/1.0 (+https://squirrelscan.com)"
V2_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# URL shape: which page type a path probably is. Ordered: first match wins.
# Multilingual on purpose; the point of this corpus is the non-English web.
# ---------------------------------------------------------------------------

SHAPES: list[tuple[str, str]] = [
    (
        "recipe",
        r"/(recipes?|rezepte?|recettes?|recetas?|ricette|receitas?|recepty|przepisy|recepten)(/|$)",
    ),
    (
        "event",
        r"/(events?|veranstaltungen|veranstaltung|evenements?|eventos?|eventi|agenda|termine|"
        r"calendar|kalender|whats-on|concerts?|tickets)(/|$)",
    ),
    (
        "job",
        r"/(jobs?|careers?|karriere|stellenangebote|stellen|vacatures|vacancies|vacancy|empleos?|"
        r"offres-d-emploi|emplois?|lavora-con-noi|vagas|oferty-pracy|praca|job-openings|positions)(/|$)",
    ),
    (
        "product",
        r"/(products?|produkte?|produits?|productos?|prodotti|produtos|produkty|items?|p|dp|"
        r"artikel|product-page|shop/p|store/p)/[^/]+",
    ),
    (
        "listing",
        r"/(collections?|product-category|categor(y|ies)|kategorien?|categories|categorie|"
        r"categoria|categorias|catalog|catalogue|katalog|shop|store|boutique|tienda|negozio|loja|"
        r"sklep|winkel|brands?|marken)(/|$)",
    ),
    (
        "article",
        r"/(blogs?|news|post|posts|articles?|stories|story|magazine|journal|aktuelles|"
        r"nachrichten|actualites|actualidad|noticias|notizie|nieuws|aktualnosci|wiadomosci|"
        r"press|presse|insights)(/|$)|/20[0-2][0-9]/[01]?[0-9]/",
    ),
    (
        "about",
        r"(about|ueber-uns|uber-uns|about-us|a-propos|qui-sommes-nous|quienes-somos|nosotros|"
        r"sobre|chi-siamo|over-ons|o-nas|o-firmie|unternehmen|company|our-team|team|our-story|"
        r"historia|geschichte)",
    ),
    ("contact", r"(contact|kontakt|contacto|contato|contatti|contactez|impressum-kontakt)"),
    (
        "legal",
        r"(privacy|terms|impressum|datenschutz|legal|cookie|agb|mentions-legales|aviso-legal|"
        r"privacidad|condiciones|imprint|disclaimer|rodo|polityka|privacy-policy|"
        r"informativa|termini)",
    ),
    ("pricing", r"(pricing|plans|preise|tarifs|precios|prezzi|cennik|tarieven|planos)(/|$)"),
    ("search", r"/(search|suche|recherche|buscar|cerca|szukaj|zoeken)(/|$)"),
    (
        "local",
        r"/(locations?|store-locator|standorte?|filialen|menu|speisekarte|carta|services?|"
        r"leistungen|servicios|servizi|uslugi|diensten|our-services|opening-hours)(/|$)",
    ),
    (
        "docs",
        r"/(docs?|documentation|help|support|faq|faqs|hilfe|aide|ayuda|kb|knowledge-base|"
        r"guides?|manual)(/|$)",
    ),
]

#: Domain-level platform hints from the url list alone. A hint only decides
#: which homepages are worth fetching; the fingerprint decides the platform.
HINTS: list[tuple[str, str]] = [
    (
        "shopify",
        r"^/(collections|products)/[^/]+|^/collections/[^/]+/products/|^/pages/[^/]+$|"
        r"^/blogs/[^/]+/[^/]+$",
    ),
    ("woo", r"^/product/[^/]+/?$|/product-category/|^/(cart|checkout|my-account)/?$"),
    ("wp", r"/(category|tag|author)/[^/]+/?$|^/20[0-2][0-9]/[01][0-9]/|/wp-|^/feed/?$"),
    ("wix", r"^/post/[^/]+$|^/product-page/|^/blank(-[0-9]+)?$|^/copy-of-|^/general-[0-9]+$"),
    ("squarespace", r"^/(shop|store)/p/|^/s/[^/]+|^/blog/[^/]+$|^/(shop|store)/?$"),
    ("webflow", r"^/(post|blog-posts?|projects?|work)/[^/]+$"),
    ("drupal", r"^/node/[0-9]+|^/taxonomy/term/[0-9]+|^/sites/default/"),
    ("magento", r"^/catalog/product/|^/customer/account|^/checkout/cart"),
    ("bigcommerce", r"^/(cart|login|account)\.php|^/brands/"),
    ("ghost", r"^/(tag|author)/[^/]+/$"),
    ("hubspot", r"^/hubfs/|^/hs-|/hs-search-results"),
]
HOST_HINTS = {
    "shopify": ".myshopify.com",
    "wix": ".wixsite.com",
    "squarespace": ".squarespace.com",
    "webflow": ".webflow.io",
}

# ---------------------------------------------------------------------------
# Fingerprints: platform and language from the HTML itself. Used for new
# homepages AND for every page of the old corpus, so both sides of the
# comparison are tagged by one function.
# ---------------------------------------------------------------------------

#: Lower-cased substrings per platform, tested with `in` (memchr speed: the
#: same function tags all ~90k corpus pages). First matching platform wins,
#: except that WooCommerce needs WordPress too.
PLATFORM_MARKERS: list[tuple[str, tuple[str, ...]]] = [
    ("shopify", ("cdn.shopify.com", "shopify.theme", "shopify-section", "myshopify.com")),
    ("woocommerce", ("woocommerce",)),
    ("wordpress", ("/wp-content/", "/wp-includes/")),
    ("wix", ("static.wixstatic.com", "wix-bolt")),
    ("squarespace", ("static1.squarespace.com", "squarespace-cdn.com", "<!-- this is squarespace")),
    ("webflow", ("data-wf-page", "data-wf-site", "webflow.js", "webflow.com/css")),
    ("drupal", ("drupal.settings", "drupal-settings-json", "/sites/default/files/")),
    ("magento", ("magento_", "mage/cookies", "data-mage-init")),
    ("bigcommerce", (".bigcommerce.com/", "data-stencil", "stencil-utils")),
    ("ghost", ("ghost-portal", "ghost-search")),
    ("hubspot", ("hs-sites.com", "hubspotusercontent")),
    ("joomla", ("/media/jui/",)),
    ("framer", ("framerusercontent.com", "data-framer-")),
    ("nextjs", ("__next_data__", "/_next/static/")),
    ("nuxt", ("__nuxt__", "/_nuxt/")),
    ("astro", ("astro-island", "/_astro/")),
    ("gatsby", ("___gatsby",)),
]
#: <meta name="generator"> values, checked in the first 64 KB only.
GENERATOR = re.compile(r"<meta[^>]+name=[\"']?generator[^>]*>", re.I)
GENERATOR_NAMES = {
    "wordpress": "wordpress",
    "wix.com": "wix",
    "webflow": "webflow",
    "drupal": "drupal",
    "ghost": "ghost",
    "hubspot": "hubspot",
    "joomla": "joomla",
    "squarespace": "squarespace",
}


#: Report buckets. Everything not named is "custom".
PLATFORM_BUCKETS = {
    "shopify": "shopify",
    "woocommerce": "woocommerce",
    "wordpress": "wordpress",
    "wix": "wix",
    "squarespace": "squarespace",
    "webflow": "webflow",
    "nextjs": "nextjs",
}


def platform_of(html: str) -> str:
    head = html[:300_000].lower()
    hits = {name for name, markers in PLATFORM_MARKERS if any(m in head for m in markers)}
    for tag in GENERATOR.findall(head[:65_536]):
        for needle, name in GENERATOR_NAMES.items():
            if needle in tag:
                hits.add(name)
    if "shopify" in hits:
        return "shopify"
    if "woocommerce" in hits and "wordpress" in hits:
        return "woocommerce"
    for name, _ in PLATFORM_MARKERS:
        if name in hits and name != "woocommerce":
            return name
    return "custom"


LANG_ATTR = re.compile(r"<html[^>]*?\slang\s*=\s*[\"']?([a-zA-Z]{2,3})", re.I)
STRIP = re.compile(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>|<[^>]+>", re.I)
STOPWORDS = {
    "en": "the and of to in is for that with you this are on your our from",
    "de": "der die und das ist nicht mit den sie ein eine auf für von zu",
    "fr": "le la les et des est pour une dans que vous sur avec pas du",
    "es": "el la los las y de que en para con una por del su más",
    "pt": "o a os as e de que em para com uma do da não seu",
    "it": "il la di che e per una con non sono del della gli le",
    "nl": "de het een en van is dat op voor met niet zijn je",
    "pl": "i w na z do nie się jest że to o jak dla oraz",
}
STOP_SETS = {lang: set(words.split()) for lang, words in STOPWORDS.items()}


def lang_of(html: str) -> str:
    """Two-letter language: the <html lang> attribute when present, else the
    script (CJK, Cyrillic) or a stopword vote over visible text, else `unk`."""
    match = LANG_ATTR.search(html[:20_000])
    if match:
        return match.group(1).lower()[:2]
    text = STRIP.sub(" ", html[:200_000])
    kana = len(re.findall(r"[぀-ヿ]", text))
    han = len(re.findall(r"[一-鿿]", text))
    hangul = len(re.findall(r"[가-힯]", text))
    cyr = len(re.findall(r"[Ѐ-ӿ]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if kana > 20:
        return "ja"
    if hangul > 20:
        return "ko"
    if han > 50 and han > latin / 4:
        return "zh"
    if cyr > 50 and cyr > latin:
        return "ru"
    words = re.findall(r"[a-zà-ÿąćęłńóśźż]+", text.lower())[:3000]
    votes = Counter(
        {lang: sum(1 for w in words if w in stops) for lang, stops in STOP_SETS.items()}
    )
    best, n = votes.most_common(1)[0] if votes else ("unk", 0)
    return best if n >= 8 else "unk"


def lang_bucket(lang: str) -> str:
    return "en" if lang == "en" else ("unk" if lang == "unk" else "non-en")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def jsonl_read(path: Path):
    if not path.exists():
        return
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for line in handle:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def stable(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:12], 16)


def log(path: Path, message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with path.open("a") as handle:
        handle.write(line + "\n")


def duck():
    import duckdb

    con = duckdb.connect()
    con.execute(
        "SET memory_limit='1500MB'; SET threads=4; SET http_retries=6; "
        "SET http_retry_wait_ms=2000; SET http_retry_backoff=2;"
    )
    return con


# ---------------------------------------------------------------------------
# meta / plan / index
# ---------------------------------------------------------------------------


def cmd_meta(args) -> None:
    out = Path(args.out)
    paths = Path(args.parts).read_text().split()
    urls = [p if p.startswith("http") else CC_BASE + p for p in paths if "subset=warc" in p]
    con = duck()
    rows = []
    for i in range(0, len(urls), 20):
        batch = urls[i : i + 20]
        listed = ",".join(f"'{u}'" for u in batch)
        rows += con.execute(
            "SELECT file_name, row_group_id, row_group_num_rows, stats_min_value, stats_max_value "
            f"FROM parquet_metadata([{listed}]) WHERE path_in_schema='url_surtkey'"
        ).fetchall()
        print(f"meta {i + len(batch)}/{len(urls)} files, {len(rows)} row groups", flush=True)
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "file_name",
                "row_group_id",
                "row_group_num_rows",
                "stats_min_value",
                "stats_max_value",
            ]
        )
        writer.writerows(rows)


def surt_tld(surt: str) -> str:
    return surt.split(",")[0].split(")")[0]


def cmd_plan(args) -> None:
    """Spread N row groups over each TLD's range; a row group spanning two TLDs
    counts for the first. Deterministic: evenly spaced picks."""
    plan = json.loads(args.tld_plan)
    rows = list(csv.DictReader(open(args.meta)))
    by_tld: dict[str, list[dict]] = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["stats_min_value"]):
        by_tld[surt_tld(row["stats_min_value"])].append(row)
    picked = []
    for tld, n in plan.items():
        groups = by_tld.get(tld, [])
        if not groups:
            print(f"plan: no row groups start in {tld}", file=sys.stderr)
            continue
        step = len(groups) / min(n, len(groups))
        for k in range(min(n, len(groups))):
            row = groups[int(k * step + step / 2)]
            picked.append(
                {
                    "tld": tld,
                    "file": row["file_name"],
                    "rg": int(row["row_group_id"]),
                    "rows": int(row["row_group_num_rows"]),
                    "lo": row["stats_min_value"],
                    "hi": row["stats_max_value"],
                }
            )
    Path(args.out).write_text("\n".join(json.dumps(p) for p in picked) + "\n")
    print(
        json.dumps(
            {
                "rowGroups": len(picked),
                "rows": sum(p["rows"] for p in picked),
                "perTld": Counter(p["tld"] for p in picked),
            }
        )
    )


def shape_sql() -> str:
    cases = " ".join(f"WHEN regexp_matches(path, '{rx}') THEN '{name}'" for name, rx in SHAPES)
    return f"CASE WHEN path = '/' AND q IS NULL THEN 'root' {cases} ELSE 'other' END"


def hint_sql() -> str:
    cols = []
    for name, rx in HINTS:
        host = HOST_HINTS.get(name)
        cond = f"regexp_matches(path, '{rx}')"
        if host:
            cond += f" OR ends_with(pdom, '{host}')"
        cols.append(f"sum(CASE WHEN {cond} THEN 1 ELSE 0 END) AS h_{name}")
    return ", ".join(cols)


def cmd_index(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logp = out / "index.log"
    con = duck()
    plan = [json.loads(line) for line in Path(args.plan).read_text().splitlines() if line]
    for item in plan:
        tag = f"{Path(item['file']).name.split('-')[1]}-rg{item['rg']}"
        dom_path, url_path = out / f"domains-{tag}.parquet", out / f"urls-{tag}.parquet"
        if dom_path.exists() and url_path.exists():
            continue
        started = time.time()
        # url_surtkey bounds select the row group (its min/max stats prune
        # every other one); the filters drop non-200, non-HTML and truncated
        # records, whose bodies are not whole pages.
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE rg AS
            SELECT url, url_host_name AS host,
                   coalesce(url_host_private_domain, url_host_registered_domain) AS pdom,
                   url_host_tld AS tld, lower(url_path) AS path, url_query AS q,
                   split_part(coalesce(content_languages, ''), ',', 1) AS lang,
                   content_charset AS charset, warc_filename AS wf,
                   warc_record_offset AS wo, warc_record_length AS wl
            FROM read_parquet('{item["file"]}')
            WHERE url_surtkey >= '{item["lo"].replace("'", "''")}'
              AND url_surtkey <= '{item["hi"].replace("'", "''")}'
              AND fetch_status = 200 AND content_mime_detected = 'text/html'
              AND content_truncated IS NULL AND pdom IS NOT NULL
        """)
        con.execute(f"CREATE OR REPLACE TEMP TABLE rgs AS SELECT *, {shape_sql()} AS shape FROM rg")
        shapes = ["root"] + [name for name, _ in SHAPES] + ["other"]
        shape_cols = ", ".join(
            f"sum(CASE WHEN shape='{s}' THEN 1 ELSE 0 END) AS s_{s}" for s in shapes
        )
        con.execute(f"""
            COPY (
              SELECT pdom, any_value(tld) AS tld, count(*) AS pages,
                     count(DISTINCT host) AS hosts, mode(lang) AS lang,
                     avg(CASE WHEN lang = 'eng' THEN 1.0 ELSE 0.0 END) AS eng_share,
                     sum(CASE WHEN q IS NOT NULL THEN 1 ELSE 0 END) AS with_query,
                     {shape_cols}, {hint_sql()}
              FROM rgs GROUP BY pdom
            ) TO '{dom_path}' (FORMAT parquet)
        """)
        # Candidate urls: a few per (domain, shape), picked by a hash of the
        # url (not crawl order), only for domains with enough pages to give a
        # page mix. Query strings are kept only for the search shape.
        con.execute(f"""
            COPY (
              WITH counted AS (SELECT pdom, count(*) AS n FROM rgs GROUP BY pdom),
              ranked AS (
                SELECT r.*, row_number() OVER (PARTITION BY r.pdom, r.shape ORDER BY hash(r.url)) AS k
                FROM rgs r JOIN counted c USING (pdom)
                WHERE c.n >= {args.min_pages} AND (r.q IS NULL OR r.shape = 'search')
                  AND length(r.url) < 300
              )
              SELECT pdom, host, url, shape, lang, charset, wf, wo, wl FROM ranked
              WHERE k <= CASE WHEN shape IN ('root') THEN 3 WHEN shape IN ('product', 'listing',
                                'article', 'other') THEN 6 ELSE 3 END
            ) TO '{url_path}' (FORMAT parquet)
        """)
        n_dom = con.execute(f"SELECT count(*) FROM '{dom_path}'").fetchone()[0]
        n_url = con.execute(f"SELECT count(*) FROM '{url_path}'").fetchone()[0]
        log(
            logp,
            f"index {tag} tld={item['tld']} rows={item['rows']} domains={n_dom} "
            f"candidateUrls={n_url} s={time.time() - started:.0f}",
        )


# ---------------------------------------------------------------------------
# candidates: merge index output, split + exclusion, homepage sample
# ---------------------------------------------------------------------------


def domain_info(hosts: list[str], exclusions: str, work: Path) -> dict[str, dict]:
    """host -> {domain, domainHash, split, excluded} via cc-domains.ts, so the
    split rule lives in exactly one TypeScript function."""
    host_file, out_file = work / "hosts.txt", work / "hosts-domains.jsonl"
    host_file.write_text("\n".join(hosts) + "\n")
    subprocess.run(
        [
            "bun",
            "run",
            "corpus/cc-domains.ts",
            "--in",
            str(host_file),
            "--exclusions",
            exclusions,
            "--out",
            str(out_file),
        ],
        cwd=V2_DIR,
        check=True,
    )
    return {row["input"]: row for row in jsonl_read(out_file)}


def cmd_candidates(args) -> None:
    index = Path(args.index)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    con = duck()
    shapes = ["root"] + [name for name, _ in SHAPES] + ["other"]
    sums = ", ".join(f"sum(s_{s}) AS s_{s}" for s in shapes)
    hints = ", ".join(f"sum(h_{name}) AS h_{name}" for name, _ in HINTS)
    # A domain cut by a row-group boundary appears twice: sum it back together.
    con.execute(f"""
        CREATE TABLE dom AS SELECT pdom, any_value(tld) AS tld, sum(pages) AS pages,
               max(hosts) AS hosts, mode(lang) AS lang, avg(eng_share) AS eng_share, {sums}, {hints}
        FROM read_parquet('{index}/domains-*.parquet') GROUP BY pdom
    """)
    con.execute(
        f"CREATE TABLE urls AS SELECT DISTINCT * FROM read_parquet('{index}/urls-*.parquet')"
    )
    # Only domains that can give a page mix AND have a root page to fingerprint.
    roots = con.execute("""
        SELECT u.pdom, u.host, u.url, u.lang, u.charset, u.wf, u.wo, u.wl,
               row_number() OVER (PARTITION BY u.pdom ORDER BY
                 (u.host = u.pdom OR u.host = 'www.' || u.pdom) DESC, length(u.url), u.url) AS k
        FROM urls u WHERE u.shape = 'root'
    """).fetchall()
    root_of = {}
    for pdom, host, url, lang, charset, wf, wo, wl, k in roots:
        if k == 1:
            root_of[pdom] = {
                "host": host,
                "url": url,
                "lang": lang,
                "charset": charset,
                "wf": wf,
                "wo": int(wo),
                "wl": int(wl),
            }
    cols = [d[0] for d in con.execute("SELECT * FROM dom LIMIT 0").description]
    doms = [
        dict(zip(cols, row, strict=True))
        for row in con.execute(f"SELECT * FROM dom WHERE pages >= {args.min_pages}").fetchall()
    ]
    doms = [d for d in doms if d["pdom"] in root_of]
    info = domain_info(sorted({d["pdom"] for d in doms}), args.exclusions, out)
    kept, dropped = [], Counter()
    for d in doms:
        meta = info.get(d["pdom"])
        if not meta or meta["excluded"]:
            dropped["excluded"] += 1
            continue
        d.update(
            {"domain": meta["domain"], "domainHash": meta["domainHash"], "split": meta["split"]}
        )
        d["root"] = root_of[d["pdom"]]
        d["strata"] = strata_of(d)
        kept.append(d)
    # Two private suffixes can map onto one registrable domain: keep the bigger.
    by_domain: dict[str, dict] = {}
    for d in kept:
        if d["domain"] not in by_domain or d["pages"] > by_domain[d["domain"]]["pages"]:
            by_domain[d["domain"]] = d
    dropped["same_registrable"] = len(kept) - len(by_domain)
    kept = list(by_domain.values())
    with (out / "domains.jsonl").open("w") as handle:
        for d in kept:
            handle.write(json.dumps(d, default=float) + "\n")
    con.execute(
        "COPY (SELECT * FROM urls WHERE pdom IN (SELECT unnest($1))) TO "
        f"'{out}/urls.parquet' (FORMAT parquet)",
        [[d["pdom"] for d in kept]],
    )
    strata = Counter(s for d in kept for s in d["strata"])
    langs = Counter(d["lang"] or "none" for d in kept)
    print(
        json.dumps(
            {
                "domains": len(kept),
                "dropped": dropped,
                "strata": strata,
                "splits": Counter(d["split"] for d in kept),
                "langTop": langs.most_common(20),
            },
            default=int,
        )
    )


GOV_EDU = re.compile(
    r"(^|\.)(gov|gob|gouv|gv|govt|go|edu|ac|mil|bund|admin|gc)\.[a-z.]+$|"
    r"\.(gov|edu|mil)$|(^|\.)(uni|univ|universit)[a-z-]*\."
)


def strata_of(d: dict) -> list[str]:
    """Sampling strata for the homepage fetch. Overlapping on purpose."""
    pages = max(d["pages"], 1)
    out = []
    hint_share = {name: d[f"h_{name}"] / pages for name, _ in HINTS}
    if hint_share["shopify"] >= 0.3 or d["pdom"].endswith(".myshopify.com"):
        out.append("p:shopify")
    if hint_share["woo"] >= 0.05 or (d["h_woo"] >= 2 and hint_share["wp"] > 0):
        out.append("p:woo")
    if hint_share["wp"] >= 0.15:
        out.append("p:wordpress")
    if hint_share["wix"] >= 0.2 or d["pdom"].endswith(".wixsite.com"):
        out.append("p:wix")
    if hint_share["squarespace"] >= 0.2 or d["pdom"].endswith(".squarespace.com"):
        out.append("p:squarespace")
    if hint_share["webflow"] >= 0.2 or d["pdom"].endswith(".webflow.io"):
        out.append("p:webflow")
    for name in ("drupal", "magento", "bigcommerce", "ghost", "hubspot"):
        if d[f"h_{name}"] >= 2:
            out.append(f"p:{name}")
    if GOV_EDU.search(d["pdom"]):
        out.append("v:gov_edu")
    if d["s_article"] >= max(5, 0.3 * pages):
        out.append("v:news")
    if d["s_recipe"] >= 2:
        out.append("v:recipe")
    if d["s_event"] >= 2:
        out.append("v:event")
    if d["s_job"] >= 2:
        out.append("v:job")
    if d["s_product"] >= 3 or d["s_listing"] >= 3:
        out.append("v:ecommerce")
    if pages <= 40 and (d["s_contact"] + d["s_local"]) >= 1 and d["s_product"] == 0:
        out.append("v:local")
    lang = (d["lang"] or "").lower()
    if lang and lang != "eng":
        out.append(f"l:{lang}")
    return out or ["random"]


def cmd_homepages(args) -> None:
    """Homepage fetch queue: per stratum, up to --per-stratum domains (stable
    hash order), plus a uniform random draw for the mixed/custom share."""
    work = Path(args.work)
    doms = list(jsonl_read(work / "domains.jsonl"))
    quota = json.loads(args.quota)
    chosen: dict[str, dict] = {}
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for d in doms:
        for s in d["strata"]:
            by_stratum[s].append(d)
    report = {}
    # Within a stratum, English and non-English domains are drawn separately
    # (--en-share of n from English), so the language mix of the fetched
    # homepages follows the corpus target, not the TLD mix of the row groups.
    for stratum, n in quota.items():
        pool = sorted(by_stratum.get(stratum, []), key=lambda d: stable("hp:" + d["domain"]))
        english = [d for d in pool if (d["lang"] or "") == "eng"]
        other = [d for d in pool if (d["lang"] or "") != "eng"]
        want_en = int(round(n * args.en_share))
        added = Counter()
        for group, cap, tag in (
            (english, want_en, "en"),
            (other, n - want_en, "nonen"),
            (english + other, n, "spill"),
        ):
            for d in group:
                if sum(added.values()) >= n or added[tag] >= cap:
                    break
                if d["domain"] not in chosen:
                    chosen[d["domain"]] = d
                    added[tag] += 1
        report[stratum] = {"pool": len(pool), "added": dict(added)}
    rest = sorted(doms, key=lambda d: stable("rand:" + d["domain"]))
    added = 0
    for d in rest:
        if added >= args.random:
            break
        if d["domain"] not in chosen:
            chosen[d["domain"]] = d
            added += 1
    report["random"] = {"pool": len(doms), "added": added}
    with (work / "homepage-queue.jsonl").open("w") as handle:
        for d in chosen.values():
            root = d["root"]
            handle.write(
                json.dumps(
                    {
                        "key": f"home:{d['domain']}",
                        "domain": d["domain"],
                        "url": root["url"],
                        "charset": root["charset"],
                        "wf": root["wf"],
                        "wo": root["wo"],
                        "wl": root["wl"],
                    }
                )
                + "\n"
            )
    print(json.dumps({"queued": len(chosen), "perStratum": report}))


# ---------------------------------------------------------------------------
# fetch: WARC record -> decoded HTML -> content store
# ---------------------------------------------------------------------------

STORE_SCHEMA = """
CREATE TABLE IF NOT EXISTS content (
  hash TEXT PRIMARY KEY,
  content BLOB NOT NULL,
  content_type TEXT NOT NULL,
  original_size INTEGER NOT NULL,
  compressed_size INTEGER NOT NULL,
  created_at INTEGER NOT NULL,
  last_accessed INTEGER NOT NULL,
  access_count INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_content_last_accessed ON content(last_accessed);
CREATE INDEX IF NOT EXISTS idx_content_type ON content(content_type);
CREATE INDEX IF NOT EXISTS idx_content_sizes ON content(compressed_size, original_size, last_accessed);
"""

META_CHARSET = re.compile(rb"<meta[^>]+charset\s*=\s*[\"']?\s*([a-zA-Z0-9_.:-]+)", re.I)


def lookup_codec(name: str | None) -> str | None:
    if not name:
        return None
    try:
        return codecs.lookup(name.strip().strip("\"'").lower()).name
    except LookupError:
        return None


def parse_record(raw: bytes) -> tuple[dict, bytes]:
    """One gzip member holding one WARC response record -> (http headers, body)."""
    data = gzip.decompress(raw)
    warc_end = data.index(b"\r\n\r\n")
    warc_headers = data[:warc_end].decode("latin-1")
    length = int(re.search(r"(?im)^Content-Length:\s*(\d+)", warc_headers).group(1))
    block = data[warc_end + 4 : warc_end + 4 + length]
    http_end = block.find(b"\r\n\r\n")
    if http_end < 0:
        raise ValueError("no http header end")
    header_lines = block[:http_end].decode("latin-1").split("\r\n")
    status = header_lines[0]
    headers: dict[str, str] = {"_status": status}
    for line in header_lines[1:]:
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
    body = block[http_end + 4 :]
    # CC decodes content- and transfer-encoding and renames the header to
    # x-crawler-*; a header still named plainly means the bytes are encoded.
    if headers.get("transfer-encoding", "").lower() == "chunked":
        body = dechunk(body)
    if headers.get("content-encoding", "").lower() in ("gzip", "x-gzip"):
        body = gzip.decompress(body)
    return headers, body


def dechunk(body: bytes) -> bytes:
    out, i = bytearray(), 0
    while i < len(body):
        end = body.find(b"\r\n", i)
        if end < 0:
            break
        size = int(body[i:end].split(b";")[0] or b"0", 16)
        if size == 0:
            break
        out += body[end + 2 : end + 2 + size]
        i = end + 2 + size + 2
    return bytes(out)


def decode_html(body: bytes, content_type: str, cc_charset: str | None) -> tuple[str, str]:
    """Bytes -> text with the declared charset (http header, then meta, then the
    index's detected charset, then utf-8). The store holds the decoded text as
    UTF-8, exactly as the squirrel crawler stores a fetched page."""
    candidates = []
    match = re.search(r"charset=([^\s;]+)", content_type or "", re.I)
    if match:
        candidates.append(match.group(1))
    meta = META_CHARSET.search(body[:8192])
    if meta:
        candidates.append(meta.group(1).decode("latin-1"))
    candidates += [cc_charset, "utf-8"]
    for name in candidates:
        codec = lookup_codec(name)
        if not codec:
            continue
        if codec in ("latin-1", "iso8859-1", "ascii"):
            codec = "cp1252"  # the WHATWG mapping of every one of these labels
        try:
            return body.decode(codec), codec
        except UnicodeDecodeError:
            continue
    return body.decode("utf-8", errors="replace"), "utf-8-replace"


class Store:
    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(root / "content-store.db", check_same_thread=False)
        self.db.executescript("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;" + STORE_SCHEMA)
        self.lock = threading.Lock()

    def put(self, html: str) -> tuple[str, int]:
        raw = html.encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        packed = gzip.compress(raw, compresslevel=6)
        now = int(time.time() * 1000)
        with self.lock:
            self.db.execute(
                "INSERT OR IGNORE INTO content (hash, content, content_type, original_size, "
                "compressed_size, created_at, last_accessed, access_count) "
                "VALUES (?, ?, 'text/html', ?, ?, ?, ?, 1)",
                (digest, packed, len(raw), len(packed), now, now),
            )
            self.db.commit()
        return digest, len(raw)


class Fetcher:
    """Range requests with a shared politeness gate: <= `concurrency` in
    flight, request STARTS spaced to <= `rate` per second across the pool, and
    every throttle answer pushes the whole pool back, not just one worker.

    data.commoncrawl.org answers an IP that goes too fast with CloudFront 403s
    (2026-10-09: blocked after ~2,200 requests in ~3 minutes at ~13/s), not
    429/503, so 403 is a throttle here too: pause, then retry the same row."""

    def __init__(self, concurrency: int, rate: float):
        import httpx

        self.client = httpx.Client(
            http2=False,
            timeout=60,
            headers={"User-Agent": USER_AGENT},
            limits=httpx.Limits(max_connections=concurrency),
        )
        self.pause_until = 0.0
        self.next_start = 0.0
        self.interval = 1.0 / rate
        self.lock = threading.Lock()
        self.throttled = 0

    def _gate(self) -> None:
        with self.lock:
            start = max(time.time(), self.next_start, self.pause_until)
            self.next_start = start + self.interval
        wait = start - time.time()
        if wait > 0:
            time.sleep(wait)

    def get(self, wf: str, wo: int, wl: int) -> bytes:
        delay = 30.0
        for attempt in range(10):
            self._gate()
            try:
                response = self.client.get(
                    CC_BASE + wf, headers={"Range": f"bytes={wo}-{wo + wl - 1}"}
                )
            except Exception:
                if attempt == 9:
                    raise
                time.sleep(min(delay, 120))
                continue
            if response.status_code in (403, 429, 500, 502, 503, 504):
                with self.lock:
                    self.throttled += 1
                    self.pause_until = max(self.pause_until, time.time() + delay)
                delay = min(delay * 2, 600)
                continue
            if response.status_code != 206:
                raise RuntimeError(f"http {response.status_code}")
            return response.content
        raise RuntimeError("gave up after backoff")


def cmd_fetch(args) -> None:
    queue = list(jsonl_read(Path(args.queue)))
    ledger_path = Path(args.ledger)
    # Only successes are final: a failed row is retried on the next run (the
    # ledger keeps every attempt; readers take the ok row).
    done = {row["key"] for row in jsonl_read(ledger_path) if row.get("ok")}
    todo = [row for row in queue if row["key"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    store = Store(Path(args.store))
    fetcher = Fetcher(args.concurrency, args.rate)
    ledger_lock = threading.Lock()
    tally: Counter = Counter()
    started = time.time()

    def one(row: dict) -> None:
        result = {"key": row["key"], "domain": row.get("domain"), "url": row["url"]}
        try:
            raw = fetcher.get(row["wf"], int(row["wo"]), int(row["wl"]))
            headers, body = parse_record(raw)
            if not headers["_status"].split(" ")[1:2] == ["200"]:
                raise RuntimeError(f"status {headers['_status'][:40]}")
            html, codec = decode_html(body, headers.get("content-type", ""), row.get("charset"))
            if len(html.strip()) == 0:
                raise RuntimeError("empty body")
            digest, size = store.put(html)
            result.update({"ok": True, "hash": digest, "bytes": size, "codec": codec})
            tally["ok"] += 1
        except Exception as error:  # noqa: BLE001 - every failure is recorded by message
            result.update({"ok": False, "error": str(error)[:200]})
            tally["failed"] += 1
        with ledger_lock:
            with ledger_path.open("a") as handle:
                handle.write(json.dumps(result) + "\n")
            n = tally["ok"] + tally["failed"]
            if n % 250 == 0:
                rate = n / max(time.time() - started, 1)
                print(
                    f"fetch {n}/{len(todo)} ok={tally['ok']} failed={tally['failed']} "
                    f"throttled={fetcher.throttled} {rate:.1f}/s",
                    flush=True,
                )

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(one, todo))
    print(
        json.dumps(
            {
                "fetched": tally,
                "throttled": fetcher.throttled,
                "seconds": round(time.time() - started),
            }
        )
    )


# ---------------------------------------------------------------------------
# fingerprint / select / finalize
# ---------------------------------------------------------------------------


def read_html(db: sqlite3.Connection, digest: str) -> str | None:
    row = db.execute("SELECT content FROM content WHERE hash = ?", (digest,)).fetchone()
    return gzip.decompress(row[0]).decode("utf-8") if row else None


def cmd_fingerprint(args) -> None:
    db = sqlite3.connect(f"file:{args.store}/content-store.db?mode=ro", uri=True)
    out = Path(args.out)
    tally: Counter = Counter()
    with out.open("w") as handle:
        for row in jsonl_read(Path(args.ledger)):
            if not row.get("ok"):
                continue
            html = read_html(db, row["hash"])
            if html is None:
                continue
            platform, lang = platform_of(html), lang_of(html)
            tally[platform] += 1
            handle.write(
                json.dumps(
                    {
                        "domain": row["domain"],
                        "hash": row["hash"],
                        "platform": platform,
                        "lang": lang,
                    }
                )
                + "\n"
            )
    print(json.dumps({"platforms": tally}))


#: Page-type shape priority for a domain's url mix (root is always first).
SHAPE_PRIORITY = [
    "product",
    "listing",
    "article",
    "about",
    "contact",
    "legal",
    "pricing",
    "event",
    "recipe",
    "job",
    "local",
    "search",
    "docs",
    "other",
]


def cmd_select(args) -> None:
    work = Path(args.work)
    doms = {d["domain"]: d for d in jsonl_read(work / "domains.jsonl")}
    prints = {r["domain"]: r for r in jsonl_read(Path(args.fingerprints))}
    home_ledger = {r["domain"]: r for r in jsonl_read(Path(args.home_ledger)) if r.get("ok")}
    quota = json.loads(args.quota)  # platform bucket -> share of --domains
    platform_min = json.loads(args.platform_min)  # named platform inside `other` -> domains
    vertical_min = json.loads(args.vertical_min)
    target = args.domains
    pool = []
    for domain, fp in prints.items():
        d = doms.get(domain)
        if not d or domain not in home_ledger:
            continue
        platform = fp["platform"]
        bucket = platform if platform in quota else "other"
        cc_lang = (d["lang"] or "").lower()
        nonen = cc_lang not in ("", "eng") or fp["lang"] not in ("en", "unk")
        verticals = [s[2:] for s in d["strata"] if s.startswith("v:")]
        pool.append(
            {
                "domain": domain,
                "bucket": bucket,
                "platform": platform,
                "nonen": nonen,
                "lang": fp["lang"],
                "ccLang": cc_lang,
                "verticals": verticals,
                "split": d["split"],
                "pages": d["pages"],
            }
        )
    pool.sort(key=lambda p: stable("sel:" + p["domain"]))
    want = {b: int(round(share * target)) for b, share in quota.items()}
    # Language is a quota INSIDE each platform bucket, so the language mix of a
    # platform's candidate pool (Shopify skews English, Wix less so) cannot
    # decide the corpus language mix, nor the other way round.
    want_cell = {}
    for b, n in want.items():
        want_cell[(b, True)] = int(round(n * args.nonen_share))
        want_cell[(b, False)] = n - want_cell[(b, True)]
    chosen: list[dict] = []
    taken: set[str] = set()
    have: Counter = Counter()

    def take(p: dict) -> None:
        chosen.append(p)
        taken.add(p["domain"])
        have[(p["bucket"], p["nonen"])] += 1

    def room(p: dict, slack: float = 1.0) -> bool:
        return have[(p["bucket"], p["nonen"])] < want_cell.get((p["bucket"], p["nonen"]), 0) * slack

    # Pass 1: vertical minimums, so the rare page types (recipe, event, job)
    # have pages at all; a little over the cell quota is allowed for them.
    for vertical, n in vertical_min.items():
        got = sum(1 for c in chosen if vertical in c["verticals"])
        for p in pool:
            if got >= n:
                break
            if p["domain"] not in taken and vertical in p["verticals"] and room(p, 1.1):
                take(p)
                got += 1
    # Pass 2: named platforms inside `other` (Drupal, Magento, ...).
    for platform, n in platform_min.items():
        got = sum(1 for c in chosen if c["platform"] == platform)
        for p in pool:
            if got >= n:
                break
            if p["domain"] not in taken and p["platform"] == platform and room(p, 1.1):
                take(p)
                got += 1
    # Pass 3: every (platform, language) cell in hash order.
    for p in pool:
        if p["domain"] not in taken and room(p):
            take(p)
    # Pass 4: a cell its pool could not fill borrows from the same platform's
    # other language, so the platform share holds before the language share.
    for p in pool:
        b = p["bucket"]
        if p["domain"] in taken:
            continue
        if have[(b, True)] + have[(b, False)] < want.get(b, 0):
            take(p)
    # Url mix per chosen domain.
    import duckdb

    con = duckdb.connect()
    urls: dict[str, list[dict]] = defaultdict(list)
    cols = ["pdom", "host", "url", "shape", "lang", "charset", "wf", "wo", "wl"]
    pdom_of = {d["domain"]: d["pdom"] for d in doms.values()}
    wanted_pdoms = [pdom_of[c["domain"]] for c in chosen]
    for row in con.execute(
        f"SELECT {', '.join(cols)} FROM '{work}/urls.parquet' WHERE pdom IN (SELECT unnest($1))",
        [wanted_pdoms],
    ).fetchall():
        urls[row[0]].append(dict(zip(cols, row, strict=True)))
    queue = []
    per_domain = Counter()
    for c in chosen:
        d = doms[c["domain"]]
        home = home_ledger[c["domain"]]
        rows = [u for u in urls[d["pdom"]] if u["shape"] != "root" and u["url"] != home["url"]]
        rows.sort(key=lambda u: stable("url:" + u["url"]))
        picks: list[dict] = []
        seen = {home["url"]}
        ecommerce = c["platform"] in ("shopify", "woocommerce", "magento", "bigcommerce")
        extra = {"product": 2, "listing": 2} if ecommerce else {"article": 2}
        for shape in SHAPE_PRIORITY:
            for u in [u for u in rows if u["shape"] == shape][: extra.get(shape, 1)]:
                if u["url"] not in seen and len(picks) < args.pages_per_domain - 1:
                    picks.append(u)
                    seen.add(u["url"])
        for u in rows:  # fill from whatever is left
            if len(picks) >= args.pages_per_domain - 1:
                break
            if u["url"] not in seen:
                picks.append(u)
                seen.add(u["url"])
        for u in picks:
            queue.append(
                {
                    "key": f"page:{u['url']}",
                    "domain": c["domain"],
                    "url": u["url"],
                    "shape": u["shape"],
                    "charset": u["charset"],
                    "wf": u["wf"],
                    "wo": int(u["wo"]),
                    "wl": int(u["wl"]),
                }
            )
        per_domain[c["domain"]] = len(picks) + 1
    with (work / "selected-domains.jsonl").open("w") as handle:
        for c in chosen:
            handle.write(json.dumps(c) + "\n")
    with (work / "page-queue.jsonl").open("w") as handle:
        for row in queue:
            handle.write(json.dumps(row) + "\n")
    print(
        json.dumps(
            {
                "domains": len(chosen),
                "pagesPlanned": sum(per_domain.values()),
                "perCell": {f"{b}/{'nonen' if n else 'en'}": v for (b, n), v in have.items()},
                "want": want,
                "nonEnglish": sum(1 for c in chosen if c["nonen"]),
                "verticals": Counter(v for c in chosen for v in c["verticals"]),
                "splits": Counter(c["split"] for c in chosen),
                "shapes": Counter(r["shape"] for r in queue),
                "poolSize": len(pool),
                "poolPerBucket": Counter(p["bucket"] for p in pool),
            },
            default=int,
        )
    )


PAGES_DDL = """
CREATE TABLE IF NOT EXISTS pages (
  crawl_id TEXT NOT NULL, url TEXT NOT NULL, normalized_url TEXT NOT NULL, final_url TEXT,
  depth INTEGER NOT NULL, parent_url TEXT, redirect_chain TEXT, status INTEGER NOT NULL,
  content_type TEXT, size_bytes INTEGER NOT NULL, load_time_ms INTEGER NOT NULL, ttfb INTEGER,
  download_time INTEGER, fetched_at INTEGER NOT NULL, etag TEXT, last_modified TEXT,
  content_hash TEXT NOT NULL, html TEXT, parsed_data TEXT, headers TEXT NOT NULL,
  security_headers TEXT NOT NULL, request_headers TEXT, fetcher_id TEXT, fallback_reason TEXT,
  source_hash TEXT, html_hash TEXT, PRIMARY KEY (crawl_id, normalized_url)
);
"""


def cmd_finalize(args) -> None:
    """The url index build-dataset.ts reads: one pages row per selected, fetched
    page (homepage + url mix). Rebuilt from scratch every run."""
    selected = {c["domain"] for c in jsonl_read(Path(args.selected))}
    rows = []
    for ledger in args.ledgers:
        for r in jsonl_read(Path(ledger)):
            if r.get("ok") and r.get("domain") in selected:
                rows.append(r)
    project = Path(args.store) / "projects" / args.project
    project.mkdir(parents=True, exist_ok=True)
    db_path = project / "project.db"
    if db_path.exists():
        db_path.unlink()
    db = sqlite3.connect(db_path)
    db.executescript(PAGES_DDL)
    now = int(time.time() * 1000)
    seen = set()
    for r in rows:
        if r["url"] in seen:
            continue
        seen.add(r["url"])
        db.execute(
            "INSERT OR IGNORE INTO pages (crawl_id, url, normalized_url, depth, status, "
            "content_type, size_bytes, load_time_ms, fetched_at, content_hash, headers, "
            "security_headers, fetcher_id, html_hash) VALUES (?, ?, ?, 0, 200, 'text/html', "
            "?, 0, ?, ?, '{}', '{}', 'commoncrawl', ?)",
            (args.crawl_id, r["url"], r["url"], r["bytes"], now, r["hash"], r["hash"]),
        )
    db.commit()
    n = db.execute("SELECT count(*), count(DISTINCT content_hash) FROM pages").fetchone()
    print(json.dumps({"pages": n[0], "distinctHashes": n[1], "domains": len(selected)}))


def cmd_tag_pages(args) -> None:
    """platform + language per page of a dataset, read from its stores (read
    only), for the per-platform / per-language report slices."""
    stores = [sqlite3.connect(f"file:{p}?mode=ro", uri=True) for p in args.stores.split(":") if p]
    out = Path(args.out)
    done = {r["p"] for r in jsonl_read(out)}
    tally: Counter = Counter()
    with out.open("a") as handle:
        for row in jsonl_read(Path(args.dataset) / "page-index.jsonl.gz"):
            if row["p"] in done:
                continue
            html = None
            for db in stores:
                hit = db.execute(
                    "SELECT content FROM content WHERE hash >= ? AND hash < ? LIMIT 2",
                    (row["p"], row["p"] + "g"),
                ).fetchall()
                if len(hit) == 1:
                    html = gzip.decompress(hit[0][0]).decode("utf-8", errors="replace")
                    break
            if html is None:
                tally["missing"] += 1
                handle.write(json.dumps({"p": row["p"], "platform": None, "lang": None}) + "\n")
                continue
            platform, lang = platform_of(html), lang_of(html)
            tally[platform] += 1
            handle.write(json.dumps({"p": row["p"], "platform": platform, "lang": lang}) + "\n")
            if sum(tally.values()) % 5000 == 0:
                print(f"tag {sum(tally.values())}", flush=True)
    print(json.dumps({"tagged": tally}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("meta")
    p.add_argument("--parts", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--meta", required=True)
    p.add_argument("--tld-plan", required=True, help='json {"com": 12, "de": 5, ...}')
    p.add_argument("--out", required=True)
    p = sub.add_parser("index")
    p.add_argument("--plan", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-pages", type=int, default=4)
    p = sub.add_parser("candidates")
    p.add_argument("--index", required=True)
    p.add_argument("--exclusions", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-pages", type=int, default=4)
    p = sub.add_parser("homepages")
    p.add_argument("--work", required=True)
    p.add_argument("--quota", required=True, help='json {"p:shopify": 1500, ...}')
    p.add_argument("--random", type=int, default=1500)
    p.add_argument("--en-share", type=float, default=0.62)
    p = sub.add_parser("fetch")
    p.add_argument("--queue", required=True)
    p.add_argument("--ledger", required=True)
    p.add_argument("--store", required=True)
    p.add_argument("--concurrency", type=int, default=4, choices=range(1, 9))
    p.add_argument("--rate", type=float, default=4.0, help="request starts per second, pool-wide")
    p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("fingerprint")
    p.add_argument("--ledger", required=True)
    p.add_argument("--store", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("select")
    p.add_argument("--work", required=True)
    p.add_argument("--fingerprints", required=True)
    p.add_argument("--home-ledger", required=True)
    p.add_argument("--quota", required=True, help="json platform bucket -> share")
    p.add_argument("--vertical-min", default="{}")
    p.add_argument("--platform-min", default="{}")
    p.add_argument("--domains", type=int, default=3000)
    p.add_argument("--nonen-share", type=float, default=0.3)
    p.add_argument("--pages-per-domain", type=int, default=8)
    p = sub.add_parser("finalize")
    p.add_argument("--selected", required=True)
    p.add_argument("--ledgers", nargs="+", required=True)
    p.add_argument("--store", required=True)
    p.add_argument("--project", default="cc")
    p.add_argument("--crawl-id", default="commoncrawl")
    p = sub.add_parser("tag-pages")
    p.add_argument("--dataset", required=True)
    p.add_argument("--stores", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    {
        "meta": cmd_meta,
        "plan": cmd_plan,
        "index": cmd_index,
        "candidates": cmd_candidates,
        "homepages": cmd_homepages,
        "fetch": cmd_fetch,
        "fingerprint": cmd_fingerprint,
        "select": cmd_select,
        "finalize": cmd_finalize,
        "tag-pages": cmd_tag_pages,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
