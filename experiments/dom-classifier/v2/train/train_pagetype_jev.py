"""Train the 11-class page-type head on Jev v7 labels.

The silver page-type labels are demoted: schema-derived classes agreed with Jev
only 27% of the time on holdout, and 40% of schema-less pages belong to classes
(`legal`, `pricing`) that no schema.org type maps to, so silver could not
express them at all.

Protocol:

  pool        Jev-labelled train (1,606) + val (2,458) = 4,064 pages. The old
              silver `val` split is absorbed into the training pool because it
              is labelled and holdout is not: there is no separate val set left,
              so model selection is domain-grouped 5-fold CV INSIDE the pool.
  holdout     2,688 Jev-labelled pages, read ONCE, at the end.

Grouping is by registrable-domain hash at every level: the folds, the learning
curve subsamples, and the frozen train/holdout split itself. A page's domain
never appears on both sides of anything.

`homepage` is trained as a class because Jev labels it, but the runtime answers
homepage from the URL before the model is consulted, so its holdout numbers
here describe the head in isolation rather than what ships.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp

N_JOBS = 4

# Jev v7's label set, in a fixed order the artifact depends on.
CLASSES = [
    "article",
    "listing_index",
    "docs",
    "product_detail",
    "about",
    "contact",
    "legal",
    "pricing",
    "homepage",
    "product_listing",
    "other",
]
CLASS_INDEX = {name: i for i, name in enumerate(CLASSES)}

# Silver's classes, for the optional prior. Silver can never say legal/pricing.
SILVER_TO_CLASS = {
    "article": "article",
    "listing_index": "listing_index",
    "docs": "docs",
    "product_detail": "product_detail",
    "about": "about",
    "contact": "contact",
    "homepage": "homepage",
    "product_listing": "product_listing",
    "other": "other",
}


def load_jev(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["pageId"]] = row
    return out


def load_features(path: Path, n_numeric: int, n_tokens: int) -> dict[str, dict]:
    out = {}
    width = n_numeric + n_tokens
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            vector = np.zeros(width, dtype=np.float32)
            vector[:n_numeric] = row["f"]
            for bucket in row["k"]:
                vector[n_numeric + bucket] = 1.0
            out[row["p"]] = {"x": vector, "d": row["d"], "silver": row["l"]}
    return out


def assemble(jev: dict, features: dict):
    xs, ys, groups, weights, silver, ids = [], [], [], [], [], []
    for page_id, label in jev.items():
        found = features.get(page_id)
        if found is None or label["pageType"] not in CLASS_INDEX:
            continue
        xs.append(found["x"])
        ys.append(CLASS_INDEX[label["pageType"]])
        groups.append(found["d"])
        weights.append(float(label.get("confidence", 1.0)))
        silver.append(found["silver"])
        ids.append(page_id)
    return (
        sp.csr_matrix(np.vstack(xs)),
        np.asarray(ys, dtype=np.int64),
        groups,
        np.asarray(weights, dtype=np.float64),
        silver,
        ids,
    )


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out, f1s = {}, []
    for index, name in enumerate(CLASSES):
        tp = int(np.sum((y_pred == index) & (y_true == index)))
        fp = int(np.sum((y_pred == index) & (y_true != index)))
        fn = int(np.sum((y_pred != index) & (y_true == index)))
        support = int(np.sum(y_true == index))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[name] = {"precision": round(precision, 4), "recall": round(recall, 4),
                     "f1": round(f1, 4), "support": support}
        if support:
            f1s.append(f1)
    return {"perClass": out, "macroF1": round(float(np.mean(f1s)), 4) if f1s else 0.0,
            "accuracy": round(float(np.mean(y_pred == y_true)), 4), "n": int(len(y_true))}


def fold_of(domain_hash: str, folds: int) -> int:
    return int(hashlib.sha256(domain_hash.encode()).hexdigest()[:8], 16) % folds


def train_booster(x, y, w, x_val, y_val, w_val, config: dict, rounds: int):
    """With a validation set, early-stops on it. Without one (x_val is None),
    trains exactly `rounds` and stops — which is what the final fit does, so it
    can use the WHOLE pool instead of holding a slice back for a stopping rule
    that is not domain-grouped and would leak sites across the boundary."""
    import lightgbm as lgb

    train_set = lgb.Dataset(x, label=y, weight=w)
    valid = None if x_val is None else lgb.Dataset(x_val, label=y_val, weight=w_val,
                                                   reference=train_set)
    params = {
        "objective": "multiclass",
        "num_class": len(CLASSES),
        "learning_rate": float(config["learningRate"]),
        "num_leaves": int(config["numLeaves"]),
        "min_data_in_leaf": int(config["minDataInLeaf"]),
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "max_bin": 63,
        "num_threads": N_JOBS,
        "verbose": -1,
        "seed": 20260920,
    }
    if valid is None:
        return lgb.train(params, train_set, num_boost_round=rounds,
                         callbacks=[lgb.log_evaluation(0)])
    return lgb.train(params, train_set, num_boost_round=rounds, valid_sets=[valid],
                     callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)])


def blend_silver(probs: np.ndarray, silver: list[str], alpha: float) -> np.ndarray:
    """Optional prior. Silver is NEVER a feature: it is mixed into the output
    afterwards, for pages where schema said something at all."""
    if alpha <= 0:
        return probs
    out = probs.copy()
    for row, label in enumerate(silver):
        mapped = SILVER_TO_CLASS.get(label)
        if mapped is None:
            continue
        prior = np.zeros(len(CLASSES))
        prior[CLASS_INDEX[mapped]] = 1.0
        out[row] = (1 - alpha) * out[row] + alpha * prior
    return out


def cross_validate(x, y, groups, w, silver, config: dict, folds: int, rounds: int,
                   alpha: float, use_weights: bool):
    assignments = np.asarray([fold_of(g, folds) for g in groups])
    oof = np.zeros((x.shape[0], len(CLASSES)))
    iterations = []
    for fold in range(folds):
        test = assignments == fold
        train = ~test
        if test.sum() == 0 or train.sum() == 0:
            continue
        weights = w[train] if use_weights else np.ones(int(train.sum()))
        val_weights = w[test] if use_weights else np.ones(int(test.sum()))
        booster = train_booster(x[train], y[train], weights, x[test], y[test], val_weights,
                                config, rounds)
        oof[test] = booster.predict(x[test], num_iteration=booster.best_iteration)
        iterations.append(int(booster.best_iteration))
    blended = blend_silver(oof, silver, alpha)
    return per_class_metrics(y, blended.argmax(axis=1)), iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--jev", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rounds", type=int, default=400)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    root = Path(args.data_root)
    jev_dir = Path(args.jev)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    schema = json.loads((root / "silver-v1" / "feature-schema.json").read_text())
    n_numeric, n_tokens = schema["pageFeatureCount"], schema["tokenBuckets"]

    features: dict[str, dict] = {}
    for split in ("train", "val", "holdout"):
        features.update(load_features(root / "silver-v1" / f"pages-{split}.jsonl.gz",
                                      n_numeric, n_tokens))

    pool_jev = {**load_jev(jev_dir / "train.jsonl"), **load_jev(jev_dir / "val.jsonl")}
    hold_jev = load_jev(jev_dir / "holdout.jsonl")
    x, y, groups, w, silver, _ = assemble(pool_jev, features)
    xh, yh, _, _, silver_h, ids_h = assemble(hold_jev, features)
    print(f"pool {x.shape[0]} pages / {len(set(groups))} domains, holdout {xh.shape[0]}",
          flush=True)

    report: dict = {
        "classes": CLASSES,
        "labelProvenance": "Jev v7 (dom-suggestions-v7), 11 classes",
        "protocol": "domain-grouped 5-fold CV inside train+val; holdout read once",
        "pool": {"pages": int(x.shape[0]), "domains": len(set(groups)),
                 "byClass": dict(Counter(CLASSES[i] for i in y).most_common())},
        "holdout": {"pages": int(xh.shape[0]),
                    "byClass": dict(Counter(CLASSES[i] for i in yh).most_common())},
        "cv": [],
    }

    best = None
    for leaves, min_data in ((15, 20), (31, 10), (63, 10)):
        for use_weights in (False, True):
            config = {"numLeaves": leaves, "minDataInLeaf": min_data, "learningRate": 0.08}
            metrics, iterations = cross_validate(x, y, groups, w, silver, config, args.folds,
                                                 args.rounds, 0.0, use_weights)
            entry = {**config, "softTargetWeights": use_weights,
                     "cvMacroF1": metrics["macroF1"], "cvAccuracy": metrics["accuracy"],
                     "iterations": iterations}
            report["cv"].append(entry)
            print(f"cv leaves {leaves} soft={use_weights} -> macroF1 {metrics['macroF1']} "
                  f"acc {metrics['accuracy']} iters {iterations}", flush=True)
            if best is None or metrics["macroF1"] > best["cvMacroF1"]:
                best = entry

    # The silver prior, chosen on CV like everything else.
    alpha_rows = []
    best_alpha = 0.0
    best_alpha_score = -1.0
    for alpha in (0.0, 0.15, 0.3):
        config = {k: best[k] for k in ("numLeaves", "minDataInLeaf", "learningRate")}
        metrics, _ = cross_validate(x, y, groups, w, silver, config, args.folds, args.rounds,
                                    alpha, best["softTargetWeights"])
        alpha_rows.append({"alpha": alpha, "cvMacroF1": metrics["macroF1"]})
        print(f"cv silver prior alpha {alpha} -> macroF1 {metrics['macroF1']}", flush=True)
        if metrics["macroF1"] > best_alpha_score:
            best_alpha, best_alpha_score = alpha, metrics["macroF1"]
    report["silverPrior"] = {"tried": alpha_rows, "chosen": best_alpha,
                             "helped": best_alpha > 0.0}

    # Learning curve, by DOMAIN so a subsample is a smaller corpus rather than a
    # thinner slice of the same sites.
    domains = sorted(set(groups))
    ordered = sorted(domains, key=lambda d: hashlib.sha256(d.encode()).hexdigest())
    curve = []
    for target in (1000, 2000, 4000):
        keep: set[str] = set()
        count = 0
        for domain in ordered:
            if count >= target:
                break
            keep.add(domain)
            count += sum(1 for g in groups if g == domain)
        mask = np.asarray([g in keep for g in groups])
        if mask.sum() < 200:
            continue
        config = {k: best[k] for k in ("numLeaves", "minDataInLeaf", "learningRate")}
        kept_groups = [g for g, m in zip(groups, mask, strict=True) if m]
        kept_silver = [s for s, m in zip(silver, mask, strict=True) if m]
        metrics, _ = cross_validate(x[mask], y[mask], kept_groups, w[mask], kept_silver,
                                    config, args.folds, args.rounds, best_alpha,
                                    best["softTargetWeights"])
        curve.append({"pages": int(mask.sum()), "domains": len(keep),
                      "cvMacroF1": metrics["macroF1"]})
        print(f"learning curve {int(mask.sum())} pages -> cvMacroF1 {metrics['macroF1']}",
              flush=True)
    report["learningCurve"] = curve

    # Final model: the whole pool, at the median CV best iteration.
    iterations = int(np.median(best["iterations"])) if best["iterations"] else 100
    config = {k: best[k] for k in ("numLeaves", "minDataInLeaf", "learningRate")}
    weights = w if best["softTargetWeights"] else np.ones(x.shape[0])
    booster = train_booster(x, y, weights, None, None, None, config, iterations)
    booster.save_model(str(out / "pagetype-model.txt"))

    probs = blend_silver(booster.predict(xh), silver_h, best_alpha)
    pred = probs.argmax(axis=1)
    present = np.asarray([s != "unlabelled" for s in silver_h])
    holdout_metrics = per_class_metrics(yh, pred)
    report["shipped"] = {
        "config": config, "softTargetWeights": best["softTargetWeights"],
        "silverPriorAlpha": best_alpha, "iterations": int(booster.num_trees() / len(CLASSES)),
        **holdout_metrics,
        "schemaPresent": per_class_metrics(yh[present], pred[present]),
        "schemaLess": per_class_metrics(yh[~present], pred[~present]),
    }

    # The product metric: article on pages schema could not label at all.
    article = CLASS_INDEX["article"]
    for threshold in (0.5, 0.7, 0.9):
        picked = (~present) & (probs[:, article] >= threshold)
        hits = int(np.sum(yh[picked] == article))
        report["shipped"][f"articleOnSchemaLessAt{threshold}"] = {
            "n": int(picked.sum()),
            "precision": round(hits / max(int(picked.sum()), 1), 4),
            "recall": round(hits / max(int(np.sum((~present) & (yh == article))), 1), 4),
            "whatTheyReallyAre": dict(
                Counter(CLASSES[i] for i in yh[picked]).most_common()
            ),
        }

    (out / "pagetype-jev-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"HOLDOUT macroF1 {holdout_metrics['macroF1']} acc {holdout_metrics['accuracy']} "
          f"(schema-present {report['shipped']['schemaPresent']['accuracy']}, "
          f"schema-less {report['shipped']['schemaLess']['accuracy']})", flush=True)


if __name__ == "__main__":
    main()
