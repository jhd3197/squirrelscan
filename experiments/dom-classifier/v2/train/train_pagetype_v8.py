"""Page type v8: "this page should carry schema.org type X", trained to abstain.

Labels, one per page, first source that has one:
  1. Jev v8 (noise sample + split files), weight = Jev's confidence;
  2. Jev v7, only for the classes v8 keeps 1:1 (schema_labels_v8.V7_REUSED);
  3. the page's own schema (schema_labels_v8.py), only for classes whose free
     label agreed with Jev v8 on the noise sample at >= --free-min-agreement,
     weight = that agreement, and never on holdout (holdout is Jev only).
The featurizer never reads schema markup, so the schema-labelled pages are
trained with their schema hidden by construction.

Splits are the dataset's (domain-grouped, same holdout domains as silver-v3).
Train on train, early-stop and pick the config on val, pick per-class
thresholds on val for precision >= --target-precision, report holdout once.

Also reported on holdout:
  - schemaHidden: pages whose declared schema agrees with Jev v8, i.e. "correct
    schema", scored with that schema hidden (it always is);
  - vsV7: the v7 head on the classes both taxonomies share.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from schema_labels_v8 import V7_REUSED

N_JOBS = int(os.environ.get("DOM_CLASSIFIER_JOBS", "4"))
V8_CLASSES = [
    "homepage", "product_detail", "product_listing", "article", "blog_post", "faq", "recipe",
    "event", "job_posting", "local_business", "about", "contact", "docs", "legal", "pricing",
    "search_results", "listing_index", "other",
]
V7_CLASSES = [
    "article", "product_detail", "product_listing", "docs", "homepage", "pricing", "contact",
    "about", "legal", "listing_index", "other",
]


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_features(dataset: Path, n_numeric: int, n_tokens: int) -> dict[str, dict]:
    out: dict[str, dict] = {}
    width = n_numeric + n_tokens
    for split in ("train", "val", "holdout"):
        with gzip.open(dataset / f"pages-{split}.jsonl.gz", "rt") as handle:
            for line in handle:
                row = json.loads(line)
                vector = np.zeros(width, dtype=np.float32)
                vector[:n_numeric] = row["f"]
                for bucket in row["k"]:
                    vector[n_numeric + bucket] = 1.0
                out[row["p"]] = {"x": vector, "d": row["d"], "s": split}
    return out


def per_class(y: np.ndarray, pred: np.ndarray, fires: np.ndarray, classes: list[str]) -> dict:
    """Argmax F1 per class, plus precision/coverage when only `fires` rows answer."""
    out, f1s = {}, []
    for i, name in enumerate(classes):
        support = int((y == i).sum())
        tp = int(((pred == i) & (y == i)).sum())
        fp = int(((pred == i) & (y != i)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        fired = fires & (pred == i)
        n_fired = int(fired.sum())
        out[name] = {
            "support": support, "f1": round(f1, 4), "precision": round(precision, 4),
            "recall": round(recall, 4),
            "fires": n_fired,
            "precisionAtThreshold": round(float((y[fired] == i).mean()), 4) if n_fired else None,
            "coverageAtThreshold": round(int((y[fired] == i).sum()) / support, 4) if support else None,
        }
        if support:
            f1s.append(f1)
    return {"perClass": out, "macroF1": round(float(np.mean(f1s)), 4) if f1s else 0.0,
            "accuracy": round(float((pred == y).mean()), 4) if len(y) else 0.0,
            "coverage": round(float(fires.mean()), 4) if len(y) else 0.0,
            "accuracyWhenFiring": round(float((pred[fires] == y[fires]).mean()), 4)
            if fires.any() else None, "n": int(len(y))}


def pick_thresholds(probs: np.ndarray, y: np.ndarray, classes: list[str], target: float,
                    min_fires: int) -> list[float]:
    top = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    out = []
    for i in range(len(classes)):
        chosen = 1.01
        for t in np.arange(0.3, 1.0, 0.01):
            fires = (top == i) & (conf >= t)
            if int(fires.sum()) < min_fires:
                break
            if float((y[fires] == i).mean()) >= target:
                chosen = round(float(t), 2)
                break
        out.append(chosen)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", default="silver-v3")
    parser.add_argument("--jev-v8", required=True)
    parser.add_argument("--jev-v7", required=True)
    parser.add_argument("--schema-labels", required=True)
    parser.add_argument("--v7-model", default="", help="v7 page-type head to compare against")
    parser.add_argument("--merge", default="", help="rare:parent,... applied to every source")
    parser.add_argument("--free-min-agreement", type=float, default=0.7)
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument("--min-fires", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import lightgbm as lgb

    root = Path(args.data_root)
    dataset = root / args.dataset
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merge = dict(pair.split(":") for pair in args.merge.split(",") if pair)
    classes = [c for c in V8_CLASSES if c not in merge]
    index = {c: i for i, c in enumerate(classes)}
    fold = lambda label: merge.get(label, label)  # noqa: E731

    schema = {r["pageId"]: r["pageType"] for r in jsonl(Path(args.schema_labels))}
    jev_dir = Path(args.jev_v8)
    jev8: dict[str, dict] = {}
    for name in ("noise", "holdout", "val", "train"):
        for row in jsonl(jev_dir / f"{name}.jsonl"):
            jev8[row["pageId"]] = row
    # Free-label agreement per class, measured on the noise sample BEFORE merging.
    noise = jsonl(jev_dir / "noise.jsonl")
    agree: Counter = Counter()
    seen: Counter = Counter()
    for row in noise:
        label = schema.get(row["pageId"])
        if label is None:
            continue
        seen[label] += 1
        agree[label] += row["pageType"] == label
    agreement = {c: round(agree[c] / seen[c], 3) for c in seen}
    free_ok = {c for c, a in agreement.items() if a >= args.free_min_agreement}
    v7 = {}
    for split in ("train", "val", "holdout"):
        for row in jsonl(Path(args.jev_v7) / f"{split}.jsonl"):
            v7[row["pageId"]] = row

    schema_features = json.loads((dataset / "feature-schema.json").read_text())
    features = load_features(dataset, schema_features["pageFeatureCount"],
                             schema_features["tokenBuckets"])
    rows = {"train": ([], [], [], []), "val": ([], [], [], []), "holdout": ([], [], [], [])}
    sources: dict[str, Counter] = {s: Counter() for s in rows}
    counts: dict[str, Counter] = {s: Counter() for s in rows}
    for page_id, feat in features.items():
        split = feat["s"]
        label = weight = source = None
        if page_id in jev8 and jev8[page_id]["pageType"] in V8_CLASSES:
            label, weight, source = jev8[page_id]["pageType"], jev8[page_id].get("confidence"), "jev-v8"
        elif page_id in v7 and v7[page_id]["pageType"] in V7_REUSED:
            label, weight, source = v7[page_id]["pageType"], v7[page_id].get("confidence"), "jev-v7"
        elif split != "holdout" and schema.get(page_id) in free_ok:
            label, weight, source = schema[page_id], agreement[schema[page_id]], "schema"
        if label is None:
            continue
        label = fold(label)
        x, y, w, ids = rows[split]
        x.append(feat["x"])
        y.append(index[label])
        w.append(float(weight if weight is not None else 1.0))
        ids.append(page_id)
        sources[split][source] += 1
        counts[split][label] += 1

    def matrix(split):
        x, y, w, ids = rows[split]
        return sp.csr_matrix(np.vstack(x)), np.asarray(y), np.asarray(w), ids

    xt, yt, wt, _ = matrix("train")
    xv, yv, wv, _ = matrix("val")
    xh, yh, _, ids_h = matrix("holdout")
    print(json.dumps({"classes": classes, "counts": {s: dict(c) for s, c in counts.items()},
                      "sources": {s: dict(c) for s, c in sources.items()},
                      "freeAgreement": agreement, "freeClassesUsed": sorted(free_ok)}), flush=True)

    base = {"objective": "multiclass", "num_class": len(classes), "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 1, "max_bin": 63, "num_threads": N_JOBS,
            "verbose": -1, "seed": 20260923}
    best = None
    grid = []
    for leaves, lr in ((31, 0.05), (63, 0.05), (127, 0.05)):
        params = {**base, "num_leaves": leaves, "learning_rate": lr, "min_data_in_leaf": 20}
        booster = lgb.train(params, lgb.Dataset(xt, label=yt, weight=wt), num_boost_round=args.rounds,
                            valid_sets=[lgb.Dataset(xv, label=yv, weight=wv)],
                            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        vp = booster.predict(xv, num_iteration=booster.best_iteration)
        val_f1 = per_class(yv, vp.argmax(1), np.ones(len(yv), bool), classes)["macroF1"]
        grid.append({"leaves": leaves, "lr": lr, "iters": booster.best_iteration, "valMacroF1": val_f1})
        print(json.dumps(grid[-1]), flush=True)
        if best is None or val_f1 > best[0]:
            best = (val_f1, booster, vp)
    _, booster, vp = best
    thresholds = pick_thresholds(vp, yv, classes, args.target_precision, args.min_fires)
    thr = np.asarray(thresholds)

    def score(x, y):
        probs = booster.predict(x, num_iteration=booster.best_iteration)
        pred = probs.argmax(1)
        return per_class(y, pred, probs.max(1) >= thr[pred], classes)

    report = {
        "classes": classes, "merge": merge,
        "labelProvenance": "page type v8 (dom-suggestions-v8): Jev v8 + Jev v7 (1:1 classes) "
                           "+ declared schema (agreement-qualified classes, schema hidden)",
        "counts": {s: dict(c) for s, c in counts.items()},
        "sources": {s: dict(c) for s, c in sources.items()},
        "freeAgreement": agreement, "freeClassesUsed": sorted(free_ok),
        "grid": grid, "iterations": int(booster.best_iteration),
        "thresholds": dict(zip(classes, thresholds, strict=True)),
        "targetPrecision": args.target_precision,
        "val": score(xv, yv), "holdout": score(xh, yh),
    }
    # "Correct schema" holdout pages: declared schema agrees with Jev v8.
    keep = np.asarray([schema.get(p) is not None and fold(schema[p]) == classes[y]
                       for p, y in zip(ids_h, yh, strict=True)])
    if keep.any():
        report["schemaHidden"] = score(xh[keep], yh[keep])
    if args.v7_model:
        v7_booster = lgb.Booster(model_file=args.v7_model)
        v7_pred = v7_booster.predict(xh).argmax(1)
        shared = [c for c in V7_CLASSES if c in index]
        v7_as_v8 = np.asarray([index.get(fold(V7_CLASSES[i]), -1) for i in v7_pred])
        v8_pred = booster.predict(xh, num_iteration=booster.best_iteration).argmax(1)
        both = {}
        for name in shared:
            i = index[name]
            for tag, pred in (("v7", v7_as_v8), ("v8", v8_pred)):
                tp = int(((pred == i) & (yh == i)).sum())
                fp = int(((pred == i) & (yh != i)).sum())
                support = int((yh == i).sum())
                p = tp / (tp + fp) if tp + fp else 0.0
                r = tp / support if support else 0.0
                both.setdefault(name, {})[tag] = {
                    "precision": round(p, 4), "recall": round(r, 4),
                    "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}
        report["vsV7"] = both
    booster.save_model(str(out / "pagetype-model.txt"), num_iteration=booster.best_iteration)
    (out / "class-thresholds.json").write_text(json.dumps({"classes": classes,
                                                            "thresholds": thresholds}) + "\n")
    (out / "pagetype-jev-report.json").write_text(json.dumps(report, indent=2) + "\n")
    h = report["holdout"]
    print(json.dumps({"holdoutMacroF1": h["macroF1"], "holdoutAccuracy": h["accuracy"],
                      "coverage": h["coverage"], "accuracyWhenFiring": h["accuracyWhenFiring"]}))


if __name__ == "__main__":
    main()
