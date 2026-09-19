"""Train and evaluate the page-type head.

Two exclusions, both deliberate:

  - `homepage` is not a model class. Its silver label was derived partly from
    the URL path being "/", and the URL shape IS a feature, so a model trained
    on it would be learning a rule the runtime can state exactly. It ships as
    a rule in rules-fallback.ts instead. Every homepage row is `leak: true`,
    so dropping the leaky rows and dropping homepage are the same cut.
  - `product_listing` has 14 training pages. That is not a class, it is a
    rumour; it is reported as unsupported rather than trained and quietly
    scored at 0.

The page rows are small (21k train), so this loads them directly rather than
going through a streaming sampler.
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

N_JOBS = 4
MIN_SUPPORT = 50

PAGE_CLASSES = ["article", "listing_index", "docs", "product_detail", "about", "contact", "other"]
CLASS_INDEX = {name: i for i, name in enumerate(PAGE_CLASSES)}
EXCLUDED_LABELS = {"homepage", "unlabelled", "product_listing"}


def load_split(path: Path, n_numeric: int, n_tokens: int, keep_unlabelled: bool = False):
    features: list[np.ndarray] = []
    labels: list[int] = []
    page_ids: list[str] = []
    domains: list[str] = []
    raw_labels: list[str] = []
    width = n_numeric + n_tokens
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if not line:
                continue
            row = json.loads(line)
            label = row["l"]
            if row.get("leak"):
                # Reaches here only for homepage; kept explicit so a future
                # leak-prone label is excluded by the same rule.
                if not keep_unlabelled:
                    continue
            if keep_unlabelled:
                if label != "unlabelled":
                    continue
            elif label in EXCLUDED_LABELS:
                continue
            vector = np.zeros(width, dtype=np.float32)
            vector[:n_numeric] = row["f"]
            for bucket in row["k"]:
                vector[n_numeric + bucket] = 1.0
            features.append(vector)
            labels.append(-1 if keep_unlabelled else CLASS_INDEX[label])
            page_ids.append(row["p"])
            domains.append(row["d"])
            raw_labels.append(label)
    matrix = sp.csr_matrix(np.vstack(features)) if features else sp.csr_matrix((0, width))
    return matrix, np.asarray(labels, dtype=np.int64), page_ids, domains, raw_labels


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict:
    out = {}
    f1s = []
    for index, name in enumerate(classes):
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
    return {"perClass": out, "macroF1": round(float(np.mean(f1s)), 4),
            "accuracy": round(float(np.mean(y_pred == y_true)), 4),
            "classesWithSupport": len(f1s)}


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> list[list[int]]:
    matrix = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=True):
        matrix[t, p] += 1
    return matrix.tolist()


def calibration(probs: np.ndarray, y_true: np.ndarray, bins: int = 10) -> list[dict]:
    top = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = pred == y_true
    table = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (top >= lo) & (top < hi if i < bins - 1 else top <= hi)
        n = int(mask.sum())
        if n == 0:
            table.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": 0})
        else:
            table.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": n,
                          "meanConfidence": round(float(top[mask].mean()), 4),
                          "accuracy": round(float(correct[mask].mean()), 4)})
    return table


def recall_at_precision(probs: np.ndarray, y_true: np.ndarray, target_class: int,
                        min_precision: float) -> dict:
    """The product question: how much of the article class can be recovered
    while staying at or above `min_precision`, using the class probability as
    the decision score."""
    score = probs[:, target_class]
    positive = y_true == target_class
    order = np.argsort(-score)
    tp = np.cumsum(positive[order])
    fp = np.cumsum(~positive[order])
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int(positive.sum()), 1)
    eligible = np.flatnonzero(precision >= min_precision)
    if eligible.size == 0:
        return {"minPrecision": min_precision, "achievable": False,
                "bestPrecision": round(float(precision.max()), 4)}
    best = eligible[int(np.argmax(recall[eligible]))]
    return {"minPrecision": min_precision, "achievable": True,
            "recall": round(float(recall[best]), 4),
            "precision": round(float(precision[best]), 4),
            "threshold": round(float(score[order][best]), 4),
            "positives": int(positive.sum())}


def train(x_train, y_train, x_val, y_val, rounds: int, config: dict):
    import lightgbm as lgb

    counts = np.bincount(y_train, minlength=len(PAGE_CLASSES)).astype(np.float64)
    counts[counts == 0] = 1.0
    weight_map = (counts.sum() / (len(PAGE_CLASSES) * counts)) ** float(config["weightPower"])
    train_set = lgb.Dataset(x_train, label=y_train, weight=weight_map[y_train])
    val_set = lgb.Dataset(x_val, label=y_val, weight=weight_map[y_val], reference=train_set)
    params = {
        "objective": "multiclass",
        "num_class": len(PAGE_CLASSES),
        "learning_rate": float(config["learningRate"]),
        "num_leaves": int(config["numLeaves"]),
        "min_data_in_leaf": int(config["minDataInLeaf"]),
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "max_bin": 63,
        "num_threads": N_JOBS,
        "verbose": -1,
        "seed": 20260919,
    }
    started = time.time()
    booster = lgb.train(params, train_set, num_boost_round=rounds, valid_sets=[val_set],
                        callbacks=[lgb.early_stopping(40, verbose=False),
                                   lgb.log_evaluation(0)])
    return booster, round(time.time() - started, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rounds", type=int, default=400)
    parser.add_argument("--budget-bytes", type=int, default=600_000)
    args = parser.parse_args()

    dataset = Path(args.data_root) / "silver-v1"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    schema = json.loads((dataset / "feature-schema.json").read_text())
    n_numeric = schema["pageFeatureCount"]
    n_tokens = schema["tokenBuckets"]

    splits = {}
    for split in ("train", "val", "holdout"):
        splits[split] = load_split(dataset / f"pages-{split}.jsonl.gz", n_numeric, n_tokens)
        print(f"{split}: {splits[split][0].shape[0]} labelled pages", flush=True)

    x_train, y_train = splits["train"][0], splits["train"][1]
    x_val, y_val = splits["val"][0], splits["val"][1]
    x_hold, y_hold = splits["holdout"][0], splits["holdout"][1]

    support = np.bincount(y_train, minlength=len(PAGE_CLASSES))
    unsupported = [PAGE_CLASSES[i] for i, n in enumerate(support) if n < MIN_SUPPORT]

    report: dict[str, object] = {
        "featurizerVersion": schema["featurizerVersion"],
        "classes": PAGE_CLASSES,
        "excludedLabels": sorted(EXCLUDED_LABELS),
        "excludedReason": {
            "homepage": "URL-derived silver label; URL shape is a feature. Shipped as a rule.",
            "product_listing": "14 training pages. Reported as unsupported, not trained.",
            "unlabelled": "no structured data; the inference target, not a class",
        },
        "unsupportedClasses": unsupported,
        "rows": {k: int(v[0].shape[0]) for k, v in splits.items()},
        "trainSupport": {PAGE_CLASSES[i]: int(n) for i, n in enumerate(support)},
        "labelProvenance": "silver: JSON-LD @type / microdata itemtype / og:type",
        "selection": [],
    }

    best_config, best_val = None, -1.0
    for leaves, min_data in ((15, 20), (31, 20), (63, 10)):
        for power in (0.0, 0.5):
            config = {"numLeaves": leaves, "minDataInLeaf": min_data,
                      "learningRate": 0.08, "weightPower": power}
            booster, seconds = train(x_train, y_train, x_val, y_val, args.rounds, config)
            val_macro = per_class_metrics(
                y_val, booster.predict(x_val, num_iteration=booster.best_iteration).argmax(axis=1),
                PAGE_CLASSES,
            )["macroF1"]
            report["selection"].append({**config, "valMacroF1": val_macro,
                                        "bestIteration": int(booster.best_iteration),
                                        "seconds": seconds})
            print(f"sweep leaves {leaves} power {power} -> val macroF1 {val_macro} "
                  f"iters {booster.best_iteration}", flush=True)
            if val_macro > best_val:
                best_config, best_val = config, val_macro
            del booster

    print(f"chosen {best_config} (val macroF1 {best_val})", flush=True)
    booster, seconds = train(x_train, y_train, x_val, y_val, args.rounds, best_config)
    trees = booster.dump_model()["tree_info"]
    n_class = len(PAGE_CLASSES)

    curve = []
    for k in range(5, booster.best_iteration + 1, 5):
        leaves_k = sum(t["num_leaves"] for t in trees[: k * n_class])
        bytes_k = 18 * leaves_k - 14 * (k * n_class)
        val_macro = per_class_metrics(
            y_val, booster.predict(x_val, num_iteration=k).argmax(axis=1), PAGE_CLASSES
        )["macroF1"]
        curve.append({"iterations": k, "leaves": int(leaves_k), "bytes": int(bytes_k),
                      "valMacroF1": val_macro})
    # The cut is early stopping's own best iteration, truncated only if it
    # does not fit the budget.
    #
    # NOT argmax of val macro-F1 over the curve: that was tried and it picked
    # 10 iterations (18 KB, nowhere near the 600 KB budget) because macro-F1
    # over 926 val pages and 7 classes is noisy enough to peak almost anywhere.
    # A 10-round model at lr 0.08 has probabilities squashed toward uniform, so
    # NO page anywhere in the corpus reached p >= 0.9 for article and the whole
    # product metric read zero. Early stopping ranks on weighted logloss, which
    # is about the probabilities, which is what the threshold consumes.
    chosen_k = int(booster.best_iteration)
    over_budget = [c for c in curve if c["iterations"] <= chosen_k
                   and c["bytes"] > args.budget_bytes]
    if over_budget:
        fitting = [c for c in curve if c["bytes"] <= args.budget_bytes]
        chosen_k = max(fitting, key=lambda c: c["iterations"])["iterations"] if fitting else 5

    probs = booster.predict(x_hold, num_iteration=chosen_k)
    pred = probs.argmax(axis=1)
    report["shipped"] = {
        "config": best_config,
        "iterations": chosen_k,
        "bytesEstimate": next(c["bytes"] for c in curve if c["iterations"] == chosen_k),
        **per_class_metrics(y_hold, pred, PAGE_CLASSES),
        "confusion": confusion(y_hold, pred, n_class),
        "calibration": calibration(probs, y_hold),
        "articleRecallAtPrecision": recall_at_precision(
            probs, y_hold, CLASS_INDEX["article"], 0.9
        ),
    }
    report["sizeCurve"] = curve

    # Baselines on the same holdout rows.
    majority = np.full(len(y_hold), int(np.bincount(y_train).argmax()), dtype=np.int64)
    report["baselines"] = {
        "majority_class": per_class_metrics(y_hold, majority, PAGE_CLASSES),
    }

    # The product question: how many of the schema-less pages does this call an
    # article with high confidence? Those are the "article without Article
    # schema" findings the rule would fire on.
    unl = load_split(dataset / "pages-holdout.jsonl.gz", n_numeric, n_tokens,
                     keep_unlabelled=True)
    unl_train = load_split(dataset / "pages-train.jsonl.gz", n_numeric, n_tokens,
                           keep_unlabelled=True)
    unl_val = load_split(dataset / "pages-val.jsonl.gz", n_numeric, n_tokens,
                         keep_unlabelled=True)
    scan = {}
    for name, bundle in (("holdout", unl), ("train", unl_train), ("val", unl_val)):
        matrix, _, ids, _, _ = bundle
        if matrix.shape[0] == 0:
            continue
        p = booster.predict(matrix, num_iteration=chosen_k)
        article = p[:, CLASS_INDEX["article"]]
        high = article >= 0.9
        # Also at the operating point that actually holds precision >= 0.9 on
        # holdout, which is the threshold a rule would ship with.
        operating = report["shipped"]["articleRecallAtPrecision"].get("threshold")
        scan[name] = {
            "unlabelledPages": int(matrix.shape[0]),
            "articleAtP90": int(high.sum()),
            "share": round(float(high.mean()), 4),
            "operatingThreshold": operating,
            "articleAtOperatingThreshold": int((article >= operating).sum())
            if operating is not None
            else None,
        }
        if name == "holdout":
            order = np.argsort(-article)
            cut = max(int(high.sum()), int((article >= (operating or 1.1)).sum()))
            (out / "article-candidates-holdout.jsonl").write_text(
                "".join(
                    json.dumps({"p": ids[int(i)], "articleProb": round(float(article[int(i)]), 4)})
                    + "\n"
                    for i in order[:cut]
                )
            )
    report["unlabelledScan"] = scan
    print(json.dumps(scan), flush=True)

    booster.save_model(str(out / "pagetype-model.txt"), num_iteration=chosen_k)
    (out / "pagetype-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"shipped: {chosen_k} iters, macroF1 {report['shipped']['macroF1']}, "
          f"~{report['shipped']['bytesEstimate']} bytes", flush=True)


if __name__ == "__main__":
    main()
