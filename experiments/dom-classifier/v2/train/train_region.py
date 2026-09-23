"""Train and evaluate the region head.

Everything is measured on the HOLDOUT split, which is frozen by registrable
domain in split-manifest.json and is never re-split here.

What the numbers are, and are not: the labels are SILVER. They are read off the
page's own landmark markup, not from a human. A macro-F1 against them says
"the model reproduces what semantic HTML would have said", which is exactly
what is wanted for a div-soup page that has no semantic HTML -- but it is not
human-verified truth, and the fully-token-hidden column is the honest end of
the range, not the headline.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

#: The dataset directory under --data-root. The 2026-09-19 build is `silver-v1`;
#: a larger corpus is built beside it rather than over it, and every script in
#: this directory has to agree on which one is being read.
DATASET = os.environ.get("DOM_CLASSIFIER_DATASET_VERSION", "silver-v1")


N_JOBS = 4

REGION_CLASSES = [
    "site_header",
    "footer",
    "navigation",
    "sidebar",
    "main_content",
    "content_block",
    "article_chrome",
    "form",
    "search",
    "dialog",
]

# Column layout of the 699-wide matrix, derived from feature-schema.json rather
# than hardcoded: scalars+categoricals, then the curated vocabulary twice (self
# and parent), then the hashed token buckets.
def layout(schema: dict) -> dict[str, tuple[int, int]]:
    n_numeric = schema["nodeFeatureCount"]
    n_vocab = len(schema["curatedVocab"])
    scalar_end = n_numeric - 2 * n_vocab
    return {
        "scalars": (0, scalar_end),
        "curated": (scalar_end, n_numeric),
        "tokens": (n_numeric, n_numeric + schema["tokenBuckets"]),
    }


ABLATIONS = {
    "full": ("scalars", "curated", "tokens"),
    "no_tokens": ("scalars", "curated"),
    "no_curated_vocab": ("scalars", "tokens"),
    "tokens_only": ("tokens",),
}


def select(matrix, blocks: dict[str, tuple[int, int]], names: tuple[str, ...]):
    """Column subset, as a mask so the feature indices stay comparable."""
    keep = np.zeros(matrix.shape[1], dtype=bool)
    for name in names:
        start, end = blocks[name]
        keep[start:end] = True
    return matrix[:, keep], np.flatnonzero(keep)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {}
    f1s = []
    for index, name in enumerate(REGION_CLASSES):
        tp = int(np.sum((y_pred == index) & (y_true == index)))
        fp = int(np.sum((y_pred == index) & (y_true != index)))
        fn = int(np.sum((y_pred != index) & (y_true == index)))
        support = int(np.sum(y_true == index))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        # A class with no support does not get to drag the macro average: it is
        # absent from the holdout, so its F1 of 0 would be measuring nothing.
        if support:
            f1s.append(f1)
    return {
        "perClass": out,
        "macroF1": round(float(np.mean(f1s)), 4),
        "accuracy": round(float(np.mean(y_pred == y_true)), 4),
        "classesWithSupport": len(f1s),
    }


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    size = len(REGION_CLASSES)
    matrix = np.zeros((size, size), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=True):
        matrix[t, p] += 1
    return matrix.tolist()


def calibration(probs: np.ndarray, y_true: np.ndarray, bins: int = 10) -> list[dict]:
    """Reliability of the top-1 confidence: the runtime thresholds on this."""
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
            continue
        table.append(
            {
                "bin": f"{lo:.1f}-{hi:.1f}",
                "n": n,
                "meanConfidence": round(float(top[mask].mean()), 4),
                "accuracy": round(float(correct[mask].mean()), 4),
            }
        )
    return table


def coverage_at_threshold(probs: np.ndarray, y_true: np.ndarray) -> list[dict]:
    """What the rules fallback costs at each candidate threshold."""
    top = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    rows = []
    for threshold in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        mask = top >= threshold
        n = int(mask.sum())
        rows.append(
            {
                "threshold": threshold,
                "coverage": round(n / len(y_true), 4),
                "accuracyOnCovered": round(float(np.mean(pred[mask] == y_true[mask])), 4)
                if n
                else 0.0,
                "macroF1OnCovered": per_class_metrics(y_true[mask], pred[mask])["macroF1"]
                if n
                else 0.0,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

# Curated words that name a region, in precedence order. This IS the "class/id
# keyword rule" G2 requires the model to beat.
KEYWORD_RULES = [
    ("dialog", ["modal", "dialog", "overlay", "popup", "cookie", "consent"]),
    ("search", ["search"]),
    ("navigation", ["nav", "navbar", "menu", "breadcrumb", "pagination", "toc"]),
    ("footer", ["footer", "copyright", "legal"]),
    ("site_header", ["header", "masthead", "logo", "banner"]),
    ("sidebar", ["sidebar", "aside", "widget"]),
    ("form", ["form", "newsletter", "subscribe", "login", "account"]),
    ("main_content", ["main", "content"]),
    ("article_chrome", ["byline", "author"]),
]


def keyword_baseline(matrix, schema: dict, blocks: dict) -> np.ndarray:
    """Predict from the curated class/id vocabulary flags alone."""
    vocab = schema["curatedVocab"]
    start, end = blocks["curated"]
    n_vocab = len(vocab)
    # self_* occupies the first half of the curated block, parent_* the second.
    dense = np.asarray(matrix[:, start : start + n_vocab].todense())
    index = {word: i for i, word in enumerate(vocab)}
    pred = np.full(dense.shape[0], REGION_CLASSES.index("content_block"), dtype=np.int64)
    assigned = np.zeros(dense.shape[0], dtype=bool)
    for label, words in KEYWORD_RULES:
        columns = [index[w] for w in words if w in index]
        if not columns:
            continue
        hit = dense[:, columns].max(axis=1) > 0
        pred[hit & ~assigned] = REGION_CLASSES.index(label)
        assigned |= hit
    return pred


def position_baseline(matrix, schema: dict) -> np.ndarray:
    """First/last element child of body, and depth, and nothing else."""
    names = schema["nodeFeatures"]
    first = names.index("isFirstBodyChild")
    last = names.index("isLastBodyChild")
    body_child = names.index("isBodyChild")
    link_density = names.index("linkDensity")
    dense = np.asarray(matrix[:, [first, last, body_child, link_density]].todense())
    pred = np.full(dense.shape[0], REGION_CLASSES.index("content_block"), dtype=np.int64)
    pred[dense[:, 3] > 0.7] = REGION_CLASSES.index("navigation")
    pred[dense[:, 2] > 0] = REGION_CLASSES.index("main_content")
    pred[dense[:, 0] > 0] = REGION_CLASSES.index("site_header")
    pred[dense[:, 1] > 0] = REGION_CLASSES.index("footer")
    return pred


def majority_baseline(y_train: np.ndarray, n: int) -> np.ndarray:
    return np.full(n, int(np.bincount(y_train).argmax()), dtype=np.int64)


# ---------------------------------------------------------------------------


def class_weights(y: np.ndarray, power: float) -> np.ndarray:
    """Inverse-frequency weights raised to `power`.

    power = 0 is no weighting at all, which is the right default here: the
    training sample ALREADY rebalances by downsampling content_block from 89%
    of nodes to 50%. Stacking full inverse-frequency weights on top of that
    rebalances twice, and the first run showed exactly what that costs --
    sidebar recall 0.82 at precision 0.18, article_chrome recall 0.73 at
    precision 0.11. The exponent is chosen on VAL macro-F1, never on holdout.
    """
    counts = np.bincount(y, minlength=len(REGION_CLASSES)).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = (counts.sum() / (len(REGION_CLASSES) * counts)) ** power
    return weights[y]


DEFAULT_CONFIG = {
    "weightPower": 0.25,
    "learningRate": 0.08,
    "numLeaves": 31,
    "minDataInLeaf": 40,
    "featureFraction": 0.8,
}


def train_one(x_train, y_train, x_val, y_val, rounds: int, config: dict,
              w_train=None, w_val=None):
    """`w_train`/`w_val` are the PER-ROW weights, multiplied into the class
    weights rather than replacing them:

    * 1 for a node that declares its own region;
    * 0.5 for one that inherits it from the landmark it sits in;
    * the annotator's own confidence, anywhere in (0, 1], for a row whose label
      came from Jev. A 0.38 answer therefore contributes 0.38 of a row, which
      is how a label source with a real but uncertain reading of a node earns
      less than a landmark tag without being thrown away.
    """
    import lightgbm as lgb

    power = float(config["weightPower"])
    tw = class_weights(y_train, power)
    vw = class_weights(y_val, power)
    if w_train is not None:
        tw = tw * w_train
    if w_val is not None:
        vw = vw * w_val
    train_set = lgb.Dataset(x_train, label=y_train, weight=tw)
    # The val set carries the SAME weighting, so early stopping optimises the
    # objective the model is being trained for rather than a different one.
    val_set = lgb.Dataset(x_val, label=y_val, weight=vw, reference=train_set)
    params = {
        "objective": "multiclass",
        "num_class": len(REGION_CLASSES),
        "learning_rate": float(config["learningRate"]),
        "num_leaves": int(config["numLeaves"]),
        "min_data_in_leaf": int(config["minDataInLeaf"]),
        "feature_fraction": float(config["featureFraction"]),
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "max_bin": 63,
        "num_threads": N_JOBS,
        "verbose": -1,
        "seed": 20260919,
    }
    started = time.time()
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=rounds,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )
    return booster, round(time.time() - started, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rounds", type=int, default=400)
    parser.add_argument("--region-budget-bytes", type=int, default=1_250_000)
    parser.add_argument("--leaves", type=int, default=31)
    parser.add_argument("--feature-sets", default="full,no_tokens,no_curated_vocab,tokens_only",
                        help="comma-separated feature sets for the val grid")
    parser.add_argument("--leaf-configs", default="31:40,63:20,127:20",
                        help="comma-separated numLeaves:minDataInLeaf pairs for the val grid")
    parser.add_argument("--skip-ablations", action="store_true",
                        help="skip the holdout feature-set ablations (reporting only)")
    args = parser.parse_args()

    sample = Path(args.sample)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    schema = json.loads((Path(args.data_root) / DATASET / "feature-schema.json").read_text())
    blocks = layout(schema)

    x_train = sp.load_npz(sample / "X-train.npz").tocsr()
    y_train = np.load(sample / "y-train.npy").astype(np.int64)
    w_train = np.load(sample / "w-train.npy").astype(np.float64)
    x_val = sp.load_npz(sample / "X-val.npz").tocsr()
    y_val = np.load(sample / "y-val.npy").astype(np.int64)
    w_val = np.load(sample / "w-val.npy").astype(np.float64)
    x_hold = sp.load_npz(sample / "X-holdout.npz").tocsr()
    y_hold = np.load(sample / "y-holdout.npy").astype(np.int64)
    m_hold = np.load(sample / "m-holdout.npy").astype(np.int64)
    print(f"train {x_train.shape} val {x_val.shape} holdout {x_hold.shape}", flush=True)

    report: dict[str, object] = {
        "featurizerVersion": schema["featurizerVersion"],
        "classes": REGION_CLASSES,
        "labelProvenance": "silver: landmark tags + ARIA roles, masked at training time",
        "rows": {
            "train": int(x_train.shape[0]),
            "val": int(x_val.shape[0]),
            "holdout": int(x_hold.shape[0]),
        },
        "baselines": {},
        "ablations": {},
    }

    # Baselines first: they need no model and they bound what a model must beat.
    for name, pred in (
        ("class_id_keyword_rule", keyword_baseline(x_hold, schema, blocks)),
        ("position_heuristic", position_baseline(x_hold, schema)),
        ("majority_class", majority_baseline(y_train, x_hold.shape[0])),
    ):
        report["baselines"][name] = per_class_metrics(y_hold, pred)
        print(f"baseline {name}: macroF1 {report['baselines'][name]['macroF1']}", flush=True)

    # Model selection, entirely on VAL. Holdout is not read until the chosen
    # config is fixed: choosing it on holdout would be choosing the test answer.
    #
    # The grid is small on purpose. The number that has to move is macro-F1,
    # and macro-F1 here is held down by two classes (article_chrome, sidebar)
    # that a hyperparameter cannot conjure evidence for, so a 60-point grid
    # would mostly be reading noise off a 156-row class.
    # All four feature sets are in the grid, not just two. The first pass swept
    # only `full` and `no_tokens`, and the holdout ablation then showed
    # no_curated_vocab ahead -- which is exactly the number a selection rule
    # must not be allowed to read. Widening the VAL grid is the fix; the
    # decision rule is still argmax of val macro-F1 and holdout is not consulted.
    # Narrowing the grid is a cost knob, not a selection one: whatever is in it,
    # the winner is still argmax of val macro-F1.
    leaf_configs = [tuple(int(v) for v in pair.split(":")) for pair in args.leaf_configs.split(",")]
    grid = []
    for feature_set in args.feature_sets.split(","):
        for leaves, min_data in leaf_configs:
            grid.append({**DEFAULT_CONFIG, "featureSet": feature_set,
                         "numLeaves": leaves, "minDataInLeaf": min_data,
                         "learningRate": 0.08})

    cached: dict[str, tuple] = {}
    def features_for(name):
        if name not in cached:
            cached[name] = (select(x_train, blocks, ABLATIONS[name])[0],
                            select(x_val, blocks, ABLATIONS[name])[0])
        return cached[name]

    sweep = []
    best_config, best_val = None, -1.0
    for config in grid:
        xt, xv = features_for(config["featureSet"])
        booster, seconds = train_one(xt, y_train, xv, y_val, args.rounds, config,
                                     w_train, w_val)
        val_pred = booster.predict(xv, num_iteration=booster.best_iteration).argmax(axis=1)
        val_macro = per_class_metrics(y_val, val_pred)["macroF1"]
        leaves_total = sum(t["num_leaves"] for t in booster.dump_model()["tree_info"])
        sweep.append({**config, "valMacroF1": val_macro,
                      "bestIteration": int(booster.best_iteration),
                      "totalLeaves": int(leaves_total), "seconds": seconds})
        print(f"sweep {config['featureSet']:9s} leaves {config['numLeaves']:3d} "
              f"lr {config['learningRate']} -> val macroF1 {val_macro} "
              f"iters {booster.best_iteration} leavesTotal {leaves_total}", flush=True)
        if val_macro > best_val:
            best_config, best_val = config, val_macro
        del booster, val_pred
    cached.clear()
    report["selection"] = {"featureSets": args.feature_sets, "leafConfigs": args.leaf_configs,
                           "grid": sweep, "chosen": best_config, "chosenOn": "val macro-F1",
                           "chosenValMacroF1": best_val}
    print(f"chosen {best_config} (val macroF1 {best_val})", flush=True)

    for name, names in ABLATIONS.items():
        if args.skip_ablations and name != best_config["featureSet"]:
            continue
        xt, columns = select(x_train, blocks, names)
        xv, _ = select(x_val, blocks, names)
        xh, _ = select(x_hold, blocks, names)
        booster, seconds = train_one(xt, y_train, xv, y_val, args.rounds, best_config,
                                     w_train, w_val)
        probs = booster.predict(xh, num_iteration=booster.best_iteration)
        pred = probs.argmax(axis=1)
        metrics = per_class_metrics(y_hold, pred)
        entry = {
            "features": list(names),
            "featureCount": int(len(columns)),
            "bestIteration": int(booster.best_iteration),
            "trainSeconds": seconds,
            **metrics,
        }
        if name == best_config["featureSet"]:
            report["chosenFeatureSet"] = name
        report["ablations"][name] = entry
        print(f"ablation {name}: macroF1 {metrics['macroF1']} acc {metrics['accuracy']} "
              f"iters {booster.best_iteration} {seconds}s", flush=True)
        del xt, xv, xh, booster, probs, pred

    # Size/accuracy curve. The artifact budget is 2 MB for BOTH heads, and the
    # chosen config's full tree set is already over that on its own, so the
    # shipped model is the same model READ AT FEWER ITERATIONS. One training
    # run, several sizes, and the cut is chosen on val like everything else.
    xt, _ = select(x_train, blocks, ABLATIONS[best_config["featureSet"]])
    xv, _ = select(x_val, blocks, ABLATIONS[best_config["featureSet"]])
    xh, _ = select(x_hold, blocks, ABLATIONS[best_config["featureSet"]])
    booster, _ = train_one(xt, y_train, xv, y_val, args.rounds, best_config, w_train, w_val)
    trees = booster.dump_model()["tree_info"]
    n_class = len(REGION_CLASSES)
    curve = []
    chosen_k = None
    for k in range(10, booster.best_iteration + 1, 5):
        leaves_k = sum(t["num_leaves"] for t in trees[: k * n_class])
        # 14 bytes per internal node (uint16 feature, float32 threshold, two
        # int32 children) + 4 bytes per leaf value; a tree with L leaves has
        # L-1 internal nodes.
        bytes_k = 18 * leaves_k - 14 * (k * n_class)
        val_macro = per_class_metrics(
            y_val, booster.predict(xv, num_iteration=k).argmax(axis=1)
        )["macroF1"]
        curve.append({"iterations": k, "leaves": int(leaves_k), "bytes": int(bytes_k),
                      "valMacroF1": val_macro})
        if bytes_k <= args.region_budget_bytes:
            chosen_k = k
    # Best val macro-F1 among the cuts that fit, not simply the largest that fits.
    fitting = [c for c in curve if c["bytes"] <= args.region_budget_bytes]
    if fitting:
        chosen_k = max(fitting, key=lambda c: c["valMacroF1"])["iterations"]
    chosen_k = chosen_k or booster.best_iteration
    holdout_at = {}
    for target in (500_000, 1_000_000, 2_000_000):
        fit = [c for c in curve if c["bytes"] <= target]
        if not fit:
            continue
        k = max(fit, key=lambda c: c["valMacroF1"])["iterations"]
        pred = booster.predict(xh, num_iteration=k).argmax(axis=1)
        holdout_at[f"{target // 1000}KB"] = {
            "iterations": k,
            "bytes": next(c["bytes"] for c in curve if c["iterations"] == k),
            **per_class_metrics(y_hold, pred),
        }
    probs = booster.predict(xh, num_iteration=chosen_k)
    pred = probs.argmax(axis=1)
    # Three readings of the same holdout, because the training mix now contains
    # both variants and they answer different questions: unmasked is "can it
    # read a landmark tag when there is one" (it should be near perfect), masked
    # is the headline, fully-hidden lives in region-maskall-report.json.
    unmasked = m_hold == 0
    report["holdoutByVariant"] = {
        "unmasked": per_class_metrics(y_hold[unmasked], pred[unmasked]),
        "masked": per_class_metrics(y_hold[~unmasked], pred[~unmasked]),
    }
    report["shipped"] = {
        "config": best_config,
        "iterations": chosen_k,
        "bytesEstimate": next(c["bytes"] for c in curve if c["iterations"] == chosen_k),
        **per_class_metrics(y_hold, pred),
        "confusion": confusion(y_hold, pred),
        "calibration": calibration(probs, y_hold),
        "coverageAtThreshold": coverage_at_threshold(probs, y_hold),
        # The same table on VAL, which is what export_model.py picks the global
        # threshold from: a threshold read off holdout would be tuned on the
        # split that grades it.
        "coverageAtThresholdVal": coverage_at_threshold(
            booster.predict(xv, num_iteration=chosen_k), y_val
        ),
    }
    report["sizeCurve"] = curve
    report["holdoutAtBudget"] = holdout_at
    booster.save_model(str(out / "region-model.txt"), num_iteration=chosen_k)
    np.save(out / "columns-model.npy", select(x_train, blocks,
                                              ABLATIONS[best_config["featureSet"]])[1])
    print(f"shipped: {chosen_k} iters, macroF1 {report['shipped']['macroF1']}, "
          f"~{report['shipped']['bytesEstimate']} bytes", flush=True)

    (out / "region-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", out / "region-report.json", flush=True)


if __name__ == "__main__":
    main()
