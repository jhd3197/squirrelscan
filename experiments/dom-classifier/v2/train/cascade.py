"""Does a cheap content_block gate in front of the region head pay for itself?

content_block is ~89% of holdout nodes, so most of the 550-tree evaluation is
spent confirming the majority class. The cascade puts a small binary model
(content_block vs rest) in front: nodes it confidently calls content_block skip
the full head entirely.

The gate costs its own trees on EVERY node, so it only wins if it can skip a
large majority. Both sides of that trade are reported as deterministic tree-node
visit counts rather than wall-clock, because this machine is not quiet enough
for the latter to mean anything.

The threshold is chosen on VAL. Holdout is read once, to answer the one question
the lead set: does macro-F1 drop by less than 0.005?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))
from sample import CLASS_INDEX, REGION_CLASSES  # noqa: E402
from train_region import ABLATIONS, layout, per_class_metrics, select  # noqa: E402

CONTENT_BLOCK = CLASS_INDEX["content_block"]
N_JOBS = 4


def mean_depth(booster) -> float:
    """Average root-to-leaf path length, for the visit accounting."""
    total_leaves, total_depth = 0, 0.0

    def walk(node: dict, depth: int) -> None:
        nonlocal total_leaves, total_depth
        if "leaf_value" in node:
            total_leaves += 1
            total_depth += depth
            return
        walk(node["left_child"], depth + 1)
        walk(node["right_child"], depth + 1)

    for tree in booster.dump_model()["tree_info"]:
        walk(tree["tree_structure"], 1)
    return total_depth / max(total_leaves, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--gate-rounds", type=int, default=40)
    args = parser.parse_args()

    import lightgbm as lgb

    sample = Path(args.sample)
    models = Path(args.models)
    schema = json.loads((Path(args.data_root) / "silver-v1" / "feature-schema.json").read_text())
    blocks = layout(schema)
    report = json.loads((models / "region" / "region-report.json").read_text())
    feature_set = report["chosenFeatureSet"]
    iterations = report["shipped"]["iterations"]

    x_train = sp.load_npz(sample / "X-train.npz").tocsr()
    y_train = np.load(sample / "y-train.npy").astype(np.int64)
    x_val = sp.load_npz(sample / "X-val.npz").tocsr()
    y_val = np.load(sample / "y-val.npy").astype(np.int64)
    x_hold = sp.load_npz(sample / "X-holdout.npz").tocsr()
    y_hold = np.load(sample / "y-holdout.npy").astype(np.int64)
    xt, _ = select(x_train, blocks, ABLATIONS[feature_set])
    xv, _ = select(x_val, blocks, ABLATIONS[feature_set])
    xh, _ = select(x_hold, blocks, ABLATIONS[feature_set])

    gate_train = (y_train == CONTENT_BLOCK).astype(np.int64)
    gate_val = (y_val == CONTENT_BLOCK).astype(np.int64)
    gate = lgb.train(
        {
            "objective": "binary",
            "learning_rate": 0.1,
            "num_leaves": 31,
            "min_data_in_leaf": 40,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "max_bin": 63,
            "num_threads": N_JOBS,
            "verbose": -1,
            "seed": 20260920,
        },
        lgb.Dataset(xt, label=gate_train),
        num_boost_round=args.gate_rounds,
        valid_sets=[lgb.Dataset(xv, label=gate_val)],
        callbacks=[lgb.log_evaluation(0)],
    )

    full = lgb.Booster(model_file=str(models / "region" / "region-model.txt"))
    full_depth = mean_depth(full)
    gate_depth = mean_depth(gate)
    full_trees = iterations * len(REGION_CLASSES)
    gate_trees = args.gate_rounds

    baseline_hold = per_class_metrics(
        y_hold, full.predict(xh, num_iteration=iterations).argmax(axis=1)
    )

    def evaluate(x, y, threshold: float) -> dict:
        gate_p = gate.predict(x)
        skipped = gate_p >= threshold
        pred = np.full(len(y), CONTENT_BLOCK, dtype=np.int64)
        if (~skipped).sum() > 0:
            pred[~skipped] = full.predict(x[~skipped], num_iteration=iterations).argmax(axis=1)
        metrics = per_class_metrics(y, pred)
        # Visits per node: the gate always, the full head only when not skipped.
        visits = gate_trees * gate_depth + float((~skipped).mean()) * full_trees * full_depth
        return {
            "threshold": threshold,
            "skipShare": round(float(skipped.mean()), 4),
            "macroF1": metrics["macroF1"],
            "accuracy": metrics["accuracy"],
            "visitsPerNode": round(visits, 1),
        }

    baseline_visits = full_trees * full_depth
    rows = [evaluate(xv, y_val, t) for t in (0.5, 0.7, 0.9, 0.95, 0.99)]
    for row in rows:
        print(f"val threshold {row['threshold']}: skip {row['skipShare']} "
              f"macroF1 {row['macroF1']} visits/node {row['visitsPerNode']} "
              f"({row['visitsPerNode'] / baseline_visits:.2f}x)", flush=True)

    val_baseline = per_class_metrics(
        y_val, full.predict(xv, num_iteration=iterations).argmax(axis=1)
    )
    # Cheapest threshold whose VAL macro-F1 is within the 0.005 the lead set.
    eligible = [r for r in rows if val_baseline["macroF1"] - r["macroF1"] < 0.005]
    chosen = min(eligible, key=lambda r: r["visitsPerNode"]) if eligible else None
    out = {
        "gateRounds": args.gate_rounds,
        "gateTrees": gate_trees,
        "fullTrees": full_trees,
        "meanDepth": {"gate": round(gate_depth, 2), "full": round(full_depth, 2)},
        "baselineVisitsPerNode": round(baseline_visits, 1),
        "valBaselineMacroF1": val_baseline["macroF1"],
        "val": rows,
        "chosenOnVal": chosen,
    }
    if chosen is not None:
        holdout = evaluate(xh, y_hold, chosen["threshold"])
        out["holdout"] = {
            "baselineMacroF1": baseline_hold["macroF1"],
            "cascadeMacroF1": holdout["macroF1"],
            "delta": round(holdout["macroF1"] - baseline_hold["macroF1"], 4),
            "skipShare": holdout["skipShare"],
            "visitsPerNode": holdout["visitsPerNode"],
            "visitsRatio": round(holdout["visitsPerNode"] / baseline_visits, 3),
            "keep": bool(baseline_hold["macroF1"] - holdout["macroF1"] < 0.005),
        }
        print(json.dumps(out["holdout"], indent=2), flush=True)
    else:
        print("no threshold stayed within 0.005 on val; cascade rejected", flush=True)

    (models / "region" / "cascade-report.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
