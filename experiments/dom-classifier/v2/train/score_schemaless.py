# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "lightgbm"]
# ///
"""Pick the pages for the "declares no schema" precision check (#2307).

Trains the skeleton-only head on every schema-labelled page (see skeleton_vs_schema.py) and
scores the pages that declare NO article, blog or product type anywhere (JSON-LD, nested or
microdata). Those are the pages a squirrel finding would fire on, so a teacher has to judge
a sample of them before the finding means anything.

Writes `sample.jsonl` (ids and model scores only, no text):
  article    predicted article >= 0.9
  product    predicted product_detail >= 0.9
  negative   article and product both < 0.1 (to estimate what the head misses)
At most 2 pages per domain, content-duplicates dropped, sealed domains and cloud-source
pages excluded.

  uv run --script train/score_schemaless.py --schema <schema.jsonl> --skeletons <dir> \
      --manifest <page-corpus/pages/manifest.jsonl> --sealed <sealed.jsonl> --out <dir> \
      [--n-article 150 --n-product 100 --n-negative 100]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("schema_labels_v8", HERE / "schema_labels_v8.py")
assert SPEC and SPEC.loader
V8 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V8)

MERGE = {"blog_post": "article", "product_listing": "listing", "listing_index": "listing"}
MIN_CLASS = 150
WATCHED = {t for t, c in V8.TYPE_TO_V8.items() if c in ("article", "blog_post", "product_detail")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--skeletons", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-article", type=int, default=150)
    parser.add_argument("--n-product", type=int, default=100)
    parser.add_argument("--n-negative", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--n-random", type=int, default=0,
                        help="also write random.jsonl: this many schema-less pages, equal per article-score decile, <= 2 per domain")
    parser.add_argument("--uniform-n", type=int, default=0,
                        help="also write uniform.jsonl: this many pages drawn uniformly from the pool (<= 2 per domain), tagged random-dN by decile")
    parser.add_argument("--exclude", type=Path, action="append", default=[],
                        help="sample files whose pages are already judged (never drawn again)")
    args = parser.parse_args()
    sealed = {json.loads(line)["d"] for line in args.sealed.read_text().splitlines() if line.strip()}
    source = {}
    for line in args.manifest.read_text().splitlines():
        if line.strip():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:  # DG may be appending to the manifest: skip a half-written last line
                continue
            source[((row.get("raw") or {}).get("sha256") or "")[:16]] = row["source"]

    rows = [json.loads(line) for line in args.schema.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r.get("ok") and r["d"] not in sealed]
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
    labelled: dict[str, str] = {}
    schemaless: dict[str, str] = {}
    for row in rows:
        key = row["sha"][:16]
        label, _ = V8.page_label(row, site_wide[row["d"]])
        if label:
            labelled[key] = MERGE.get(label, label)
        declared = set(row["ld"]) | set(row["ldNested"]) | set(row["md"])
        if not declared & WATCHED:
            schemaless[key] = row["d"]

    skeleton: dict[str, np.ndarray] = {}
    paths = sorted(args.skeletons.glob("shard-*/skeletons.jsonl"))
    for path in paths:
        for line in path.open():
            r = json.loads(line)
            skeleton[r["p"]] = np.asarray(r["f"], dtype=np.float32)
    names = json.loads((paths[0].parent / "features.json").read_text())["order"]

    ids = [p for p in labelled if p in skeleton]
    counts = Counter(labelled[p] for p in ids)
    classes = sorted(c for c, n in counts.items() if n >= MIN_CLASS)
    ids = [p for p in ids if labelled[p] in classes]
    x = np.vstack([skeleton[p] for p in ids])
    y = np.asarray([classes.index(labelled[p]) for p in ids])
    params = {"objective": "multiclass", "num_class": len(classes), "learning_rate": 0.05, "num_leaves": 31,
              "min_data_in_leaf": 20, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
              "num_threads": 2, "verbose": -1, "seed": 7}
    booster = lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=95)

    pool = [p for p in schemaless if p in skeleton and source.get(p) != "cloud"]
    probs = booster.predict(np.vstack([skeleton[p] for p in pool]))
    art, prod = classes.index("article"), classes.index("product_detail")
    scored = [(p, probs[i]) for i, p in enumerate(pool)]
    rng = random.Random(args.seed)
    rng.shuffle(scored)

    def take(predicate, n: int, kind: str) -> list[dict]:
        taken: list[dict] = []
        per_domain: Counter = Counter()
        for p, pr in scored:
            if len(taken) >= n:
                break
            domain = schemaless[p]
            if not predicate(pr) or per_domain[domain] >= 2:
                continue
            per_domain[domain] += 1
            taken.append({"p": p, "kind": kind, "pArticle": round(float(pr[art]), 3), "pProduct": round(float(pr[prod]), 3),
                          "top": classes[int(pr.argmax())], "d": domain})
        return taken

    sample = (
        take(lambda pr: pr[art] >= 0.9, args.n_article, "article")
        + take(lambda pr: pr[prod] >= 0.9, args.n_product, "product")
        + take(lambda pr: pr[art] < 0.1 and pr[prod] < 0.1, args.n_negative, "negative")
    )
    args.out.mkdir(parents=True, exist_ok=True)
    if args.n_random:
        excluded = {json.loads(line)["p"] for f in args.exclude for line in f.read_text().splitlines() if line.strip()}
        deciles: dict[int, list] = defaultdict(list)
        for p, pr in scored:
            if p not in excluded:
                deciles[min(int(pr[art] * 10), 9)].append((p, pr))
        per = args.n_random // 10
        drawn: list[dict] = []
        used: Counter = Counter()
        for d in range(10):
            got = 0
            for p, pr in deciles[d]:
                if got >= per:
                    break
                if used[schemaless[p]] >= 2:
                    continue
                used[schemaless[p]] += 1
                got += 1
                drawn.append({"p": p, "kind": f"random-d{d}", "pArticle": round(float(pr[art]), 3),
                              "pProduct": round(float(pr[prod]), 3), "top": classes[int(pr.argmax())], "d": schemaless[p]})
        (args.out / "random.jsonl").write_text("".join(json.dumps(r) + "\n" for r in drawn))
        print(f"random sample {len(drawn)}: per decile {Counter(r['kind'] for r in drawn)}")
    (args.out / "sample.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sample))
    flagged_art = int(sum(1 for _, pr in scored if pr[art] >= 0.9))
    flagged_prod = int(sum(1 for _, pr in scored if pr[prod] >= 0.9))
    if args.uniform_n:
        excluded_u = {json.loads(line)["p"] for f in args.exclude for line in f.read_text().splitlines() if line.strip()}
        used_u: Counter = Counter()
        uniform: list[dict] = []
        for p, pr in scored:  # `scored` is already shuffled
            if len(uniform) >= args.uniform_n:
                break
            if p in excluded_u or used_u[schemaless[p]] >= 2:
                continue
            used_u[schemaless[p]] += 1
            uniform.append({"p": p, "kind": f"random-d{min(int(pr[art] * 10), 9)}", "pArticle": round(float(pr[art]), 3),
                            "pProduct": round(float(pr[prod]), 3), "top": classes[int(pr.argmax())], "d": schemaless[p]})
        (args.out / "uniform.jsonl").write_text("".join(json.dumps(r) + "\n" for r in uniform))
        print(f"uniform sample {len(uniform)}: by decile {dict(sorted(Counter(r['kind'] for r in uniform).items()))}")
    decile_pool = Counter(min(int(pr[art] * 10), 9) for _, pr in scored)
    summary = {"decilePool": {str(d): decile_pool[d] for d in range(10)}, "trainedOn": len(ids), "classes": classes, "schemalessPool": len(pool), "flaggedArticle": flagged_art,
               "flaggedProduct": flagged_prod, "sample": dict(Counter(r["kind"] for r in sample)), "features": len(names)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
