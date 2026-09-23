"""Score a region head and a page-type head on ONE dataset, under one protocol.

Why this exists: the 2026-09-19 baselines were measured with two leaks (a
global threshold read off the holdout coverage table, and page-type labels
split by label file rather than by the dataset). Comparing a new model with
those numbers compares protocols, not models. This scores ANY pair of heads,
old or new, on the same sample and the same Jev holdout, so the columns of the
comparison table differ only in the model.

Region, on the sample's val and holdout matrices:
  - holdout macro-F1 / accuracy / per class, overall and by variant
    (masked / unmasked);
  - coverage and accuracy-when-firing on NON-DECLARED nodes, at two threshold
    sets: `asShipped` (the thresholds stored in the artifact, when given) and
    `valChosen` (per-class, re-picked on THIS val by class_thresholds.sweep).
Page type, on the Jev labels whose page the DATASET puts in holdout:
  agreement (accuracy), macro-F1 and per class.

Holdout is never used to choose anything here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))
from class_thresholds import measure, non_declared_mask, sweep  # noqa: E402
from sample import REGION_CLASSES  # noqa: E402
from train_pagetype_jev import (  # noqa: E402
    CLASSES as PAGE_CLASSES,
    assemble,
    load_features,
    load_jev,
)
from train_pagetype_jev import per_class_metrics as page_metrics  # noqa: E402
from train_region import per_class_metrics  # noqa: E402


def region_scores(args) -> dict:
    import lightgbm as lgb

    sample = Path(args.sample)
    booster = lgb.Booster(model_file=args.region_model)
    columns = np.load(args.columns)
    out: dict = {}
    probs_by_split = {}
    for split in ("val", "holdout"):
        x = sp.load_npz(sample / f"X-{split}.npz").tocsr()[:, columns]
        y = np.load(sample / f"y-{split}.npy").astype(np.int64)
        w = np.load(sample / f"w-{split}.npy")
        m = np.load(sample / f"m-{split}.npy")
        probs = booster.predict(x)
        del x
        probs_by_split[split] = (probs, y, w, m)
    probs, y, w, m = probs_by_split["holdout"]
    pred = probs.argmax(axis=1)
    out["holdout"] = {
        **per_class_metrics(y, pred),
        "unmasked": per_class_metrics(y[m == 0], pred[m == 0]),
        "masked": per_class_metrics(y[m == 1], pred[m == 1]),
    }
    vp, vy, vw, _ = probs_by_split["val"]
    vkeep = non_declared_mask(vy, vw)
    hkeep = non_declared_mask(y, w)
    chosen = [sweep(vp[vkeep], vy[vkeep], index) for index in range(len(REGION_CLASSES))]
    val_thresholds = [c["threshold"] for c in chosen]
    out["thresholds"] = {
        "population": "non-declared nodes",
        "valChosen": {
            "thresholds": dict(zip(REGION_CLASSES, val_thresholds, strict=True)),
            "val": measure(vp[vkeep], vy[vkeep], val_thresholds),
            "holdout": measure(probs[hkeep], y[hkeep], val_thresholds),
        },
    }
    if args.artifact:
        region = json.loads(Path(args.artifact).read_text())["region"]
        shipped = region.get("classThresholds") or [region["confidenceThreshold"]] * len(
            REGION_CLASSES
        )
        out["thresholds"]["asShipped"] = {
            "thresholds": dict(zip(REGION_CLASSES, shipped, strict=True)),
            "globalConfidenceThreshold": region.get("confidenceThreshold"),
            "holdout": measure(probs[hkeep], y[hkeep], shipped),
        }
    return out


def page_scores(args) -> dict:
    import lightgbm as lgb

    root = Path(args.data_root)
    schema = json.loads((root / args.dataset / "feature-schema.json").read_text())
    features: dict = {}
    for split in ("train", "val", "holdout"):
        features.update(load_features(root / args.dataset / f"pages-{split}.jsonl.gz",
                                      schema["pageFeatureCount"], schema["tokenBuckets"], split))
    jev_dir = Path(args.jev)
    labels = {**load_jev(jev_dir / "train.jsonl"), **load_jev(jev_dir / "val.jsonl"),
              **load_jev(jev_dir / "holdout.jsonl")}
    xh, yh, _, _, _, _ = assemble(labels, features, frozenset({"holdout"}))
    booster = lgb.Booster(model_file=args.pagetype_model)
    pred = booster.predict(xh).argmax(axis=1)
    return {"classes": PAGE_CLASSES, **page_metrics(yh, pred)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", required=True, help="e.g. silver-v3")
    parser.add_argument("--sample", required=True, help="region sample dir (X/y/w/m val+holdout)")
    parser.add_argument("--region-model", required=True)
    parser.add_argument("--columns", required=True)
    parser.add_argument("--artifact", default="", help="artifact whose shipped thresholds to score")
    parser.add_argument("--pagetype-model", default="")
    parser.add_argument("--jev", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report = {"dataset": args.dataset, "regionModel": args.region_model,
              "region": region_scores(args)}
    if args.pagetype_model:
        report["pageType"] = page_scores(args)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    r = report["region"]
    print(json.dumps({"holdoutMacroF1": r["holdout"]["macroF1"],
                      "holdoutAccuracy": r["holdout"]["accuracy"],
                      "pageTypeAccuracy": report.get("pageType", {}).get("accuracy")}), flush=True)


if __name__ == "__main__":
    main()
