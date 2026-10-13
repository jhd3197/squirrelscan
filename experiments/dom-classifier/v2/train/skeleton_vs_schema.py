# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "lightgbm"]
# ///
"""Can the skeleton alone predict the type a site declares in its schema? (#2307)

Labels: the main-entity class from train/schema_labels_v8.py (site-wide types and
template-prone types removed), taken from extract_schema.py rows. Inputs: the numeric
skeleton features ONLY (runtime/skeleton.ts), which contain no schema, og:type, URL or page
words. Domain-grouped 5-fold CV; same-content pages (equal sha) are one row.

A high precision here means the model recognises the page without reading its schema, so
"looks like an article but declares none" is a trustworthy finding.

  uv run --script train/skeleton_vs_schema.py --schema <schema.jsonl> \
      --skeletons <dir with shard-*/skeletons.jsonl> --sealed <sealed-domains.jsonl> --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("schema_labels_v8", HERE / "schema_labels_v8.py")
assert SPEC and SPEC.loader
V8 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V8)

#: classes schema cannot tell apart are merged
MERGE = {"blog_post": "article", "product_listing": "listing", "listing_index": "listing"}
MIN_CLASS = 150
BARS = (0.8, 0.9, 0.95)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--skeletons", type=Path, required=True)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=300)
    args = parser.parse_args()
    sealed = {json.loads(line)["d"] for line in args.sealed.read_text().splitlines() if line.strip()}

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
    label_of: dict[str, tuple[str, str]] = {}
    for row in rows:
        label, _ = V8.page_label(row, site_wide[row["d"]])
        if label:
            label_of[row["sha"][:16]] = (MERGE.get(label, label), row["d"])

    skeleton: dict[str, np.ndarray] = {}
    paths = sorted(args.skeletons.glob("shard-*/skeletons.jsonl"))
    order: list[str] = []
    for path in paths:
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                skeleton[row["p"]] = np.asarray(row["f"], dtype=np.float32)
                order.append(row["p"])
    names = json.loads((paths[0].parent / "features.json").read_text())["order"]

    ids = [p for p in order if p in label_of]
    counts = Counter(label_of[p][0] for p in ids)
    classes = sorted(c for c, n in counts.items() if n >= MIN_CLASS)
    ids = [p for p in ids if label_of[p][0] in classes]
    x = np.vstack([skeleton[p] for p in ids])
    y = np.asarray([classes.index(label_of[p][0]) for p in ids])
    groups = [label_of[p][1] for p in ids]
    print(f"{len(ids)} pages, classes {dict((c, int((y == i).sum())) for i, c in enumerate(classes))}", flush=True)

    folds = 5
    fold = np.asarray([int(hashlib.sha256(g.encode()).hexdigest()[:8], 16) % folds for g in groups])
    oof = np.zeros((len(ids), len(classes)))
    importance = np.zeros(x.shape[1])
    params = {"objective": "multiclass", "num_class": len(classes), "learning_rate": 0.05, "num_leaves": 31,
              "min_data_in_leaf": 20, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
              "num_threads": 2, "verbose": -1, "seed": 7}
    for k in range(folds):
        test = fold == k
        train = lgb.Dataset(x[~test], label=y[~test])
        valid = lgb.Dataset(x[test], label=y[test], reference=train)
        booster = lgb.train(params, train, num_boost_round=args.rounds, valid_sets=[valid],
                            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
        oof[test] = booster.predict(x[test], num_iteration=booster.best_iteration)
        importance += booster.feature_importance("gain")
        print(f"fold {k} best_iteration {booster.best_iteration}", flush=True)

    pred, conf = oof.argmax(axis=1), oof.max(axis=1)
    report: dict = {"pages": len(ids), "classes": classes, "byClass": {c: int((y == i).sum()) for i, c in enumerate(classes)},
                    "accuracy": round(float((pred == y).mean()), 4),
                    "majorityBaseline": round(float(np.bincount(y).max() / len(y)), 4), "atConfidence": {}}
    for bar in BARS:
        answered = conf >= bar
        per_class = {}
        for i, name in enumerate(classes):
            said = answered & (pred == i)
            n = int(said.sum())
            per_class[name] = {"answered": n, "precision": round(float((y[said] == i).mean()), 3) if n else None,
                               "recall": round(float((said & (y == i)).sum() / max((y == i).sum(), 1)), 3)}
        report["atConfidence"][str(bar)] = {"coverage": round(float(answered.mean()), 3),
                                            "precision": round(float((pred[answered] == y[answered]).mean()), 3) if answered.any() else None,
                                            "perClass": per_class}
    top = np.argsort(-importance)[:15]
    report["topFeatures"] = {names[i]: round(float(importance[i] / importance.sum()), 3) for i in top}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "skeleton_vs_schema.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("pages", "accuracy", "majorityBaseline")}))
    for bar, block in report["atConfidence"].items():
        print(bar, "coverage", block["coverage"], "precision", block["precision"])
        for name, m in block["perClass"].items():
            print(f"   {name:14s} answered={m['answered']:5d} precision={m['precision']} recall={m['recall']}")
    print("top features:", report["topFeatures"])


if __name__ == "__main__":
    main()
