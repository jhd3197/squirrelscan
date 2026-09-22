"""Per-class confidence thresholds, chosen on VAL for precision >= 0.95.

A single global threshold was the wrong instrument. The rule that picked it
("lowest value whose covered accuracy clears 0.98") landed on 0.95 once the
training population included inherited nodes, which left 31.6% coverage: two
thirds of nodes got no name at all, including the ones the model is good at.

The population every number here is measured on is NON-DECLARED nodes -- the
ones without their own landmark tag or ARIA role. Those are the only nodes the
runtime asks the model about, because markup wins for the rest.

A class that cannot reach 0.95 precision at any threshold, or that has fewer
than MIN_SUPPORT val predictions, abstains entirely: its threshold is set above
1 so it can never fire. Silence is the correct output for a class this model
cannot call reliably.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))
from sample import REGION_CLASSES  # noqa: E402
from train_region import ABLATIONS, layout, select  # noqa: E402

#: The dataset directory under --data-root. The 2026-09-19 build is `silver-v1`;
#: a larger corpus is built beside it rather than over it, and every script in
#: this directory has to agree on which one is being read.
DATASET = os.environ.get("DOM_CLASSIFIER_DATASET_VERSION", "silver-v1")


TARGET_PRECISION = 0.95
MIN_SUPPORT = 50
FLOOR = 0.5
NEVER = 1.01

# Classes a node can hold by declaring its own landmark. A node labelled one of
# these at full weight declared it; the runtime answers those from the markup.
DECLARED_CLASSES = {
    "site_header", "footer", "navigation", "sidebar", "main_content",
    "form", "search", "dialog", "article_chrome",
}


def non_declared_mask(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """True for the nodes the model actually serves."""
    declared_ids = {REGION_CLASSES.index(c) for c in DECLARED_CLASSES}
    declared = (w >= 1.0) & np.isin(y, list(declared_ids))
    return ~declared


def sweep(probs: np.ndarray, y: np.ndarray, index: int) -> dict:
    """Lowest threshold at or above FLOOR reaching TARGET_PRECISION."""
    score = probs[:, index]
    top = probs.argmax(axis=1)
    best = None
    for threshold in np.arange(FLOOR, 1.0, 0.01):
        fires = (top == index) & (score >= threshold)
        n = int(fires.sum())
        if n < MIN_SUPPORT:
            continue
        precision = float(np.mean(y[fires] == index))
        if precision >= TARGET_PRECISION:
            recall = float(np.sum(y[fires] == index) / max(int(np.sum(y == index)), 1))
            best = {"threshold": round(float(threshold), 2),
                    "valPrecision": round(precision, 4), "valFires": n,
                    "valRecall": round(recall, 4)}
            break
    if best is None:
        fires = (top == index) & (score >= FLOOR)
        n = int(fires.sum())
        why = "cannot reach 0.95 precision" if n >= MIN_SUPPORT else "under 50 val support"
        at_floor = round(float(np.mean(y[fires] == index)), 4) if n else None
        return {"threshold": NEVER, "abstains": True, "valFires": n,
                "bestPrecisionAtFloor": at_floor, "reason": why}
    return {**best, "abstains": False}


def measure(probs: np.ndarray, y: np.ndarray, thresholds: list[float]) -> dict:
    top = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    fires = conf >= np.asarray(thresholds)[top]
    out = {}
    for index, name in enumerate(REGION_CLASSES):
        selected = fires & (top == index)
        n = int(selected.sum())
        support = int(np.sum(y == index))
        out[name] = {
            "fires": n,
            "precision": round(float(np.mean(y[selected] == index)), 4) if n else None,
            "recall": round(n and float(np.sum(y[selected] == index) / max(support, 1)) or 0.0, 4),
            "support": support,
        }
    accuracy = round(float(np.mean(top[fires] == y[fires])), 4) if fires.any() else 0.0
    return {"perClass": out, "coverage": round(float(fires.mean()), 4),
            "accuracyWhenFiring": accuracy}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--models", required=True)
    args = parser.parse_args()

    import lightgbm as lgb

    schema = json.loads((Path(args.data_root) / DATASET / "feature-schema.json").read_text())
    blocks = layout(schema)
    models = Path(args.models)
    report = json.loads((models / "region" / "region-report.json").read_text())
    feature_set = report["chosenFeatureSet"]
    iterations = report["shipped"]["iterations"]
    sample = Path(args.sample)
    booster = lgb.Booster(model_file=str(models / "region" / "region-model.txt"))

    loaded = {}
    for split in ("val", "holdout"):
        x = sp.load_npz(sample / f"X-{split}.npz").tocsr()
        y = np.load(sample / f"y-{split}.npy").astype(np.int64)
        w = np.load(sample / f"w-{split}.npy")
        xs, _ = select(x, blocks, ABLATIONS[feature_set])
        keep = non_declared_mask(y, w)
        loaded[split] = (booster.predict(xs[keep], num_iteration=iterations), y[keep])
        print(f"{split}: {int(keep.sum())} non-declared rows of {len(y)}", flush=True)

    val_probs, val_y = loaded["val"]
    chosen = []
    thresholds = []
    for index, name in enumerate(REGION_CLASSES):
        entry = {"class": name, **sweep(val_probs, val_y, index)}
        chosen.append(entry)
        thresholds.append(entry["threshold"])
        detail = "ABSTAINS" if entry["abstains"] else (
            f"valP={entry['valPrecision']:.3f} valR={entry['valRecall']:.3f} n={entry['valFires']}"
        )
        print(f"  {name:16s} t={entry['threshold']:.2f} {detail}", flush=True)

    hold_probs, hold_y = loaded["holdout"]
    out = {
        "population": "non-declared nodes only (the ones the runtime asks the model about)",
        "targetPrecision": TARGET_PRECISION,
        "minValSupport": MIN_SUPPORT,
        "floor": FLOOR,
        "classes": REGION_CLASSES,
        "thresholds": thresholds,
        "chosenOnVal": chosen,
        "val": measure(val_probs, val_y, thresholds),
        "holdout": measure(hold_probs, hold_y, thresholds),
    }
    (models / "region" / "class-thresholds.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"valCoverage": out["val"]["coverage"],
                      "holdoutCoverage": out["holdout"]["coverage"],
                      "holdoutAccuracyWhenFiring": out["holdout"]["accuracyWhenFiring"]}, indent=2))


if __name__ == "__main__":
    main()
