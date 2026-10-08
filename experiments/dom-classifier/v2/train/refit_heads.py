"""Refit a region or section head on a whole CV pool, for export (#2307).

cv_compare.py measures; this writes the model that would ship from the same
config: one fit on train + val of the sample, fixed iterations, and the
thresholds cv_compare picked on that arm's out-of-fold rows. Holdout is never
read. Output matches what export_model.py expects:

  region:  <out>/region/{region-model.txt, columns-model.npy, region-report.json,
                         class-thresholds.json}
  section: <out>/{section-model.txt, class-thresholds.json}

  python refit_heads.py region --sample S --schema feature-schema.json \
      --config '{...}' --iterations 160 --cv <cv_compare out> --out DIR
  python refit_heads.py section --sample S --config '{...}' --cv <cv out> --out DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from cv_compare import N_JOBS, pool


def coverage_table(pred: np.ndarray, conf: np.ndarray, y: np.ndarray) -> list[dict]:
    """train_region.coverage_at_threshold, from (argmax, max prob) pairs."""
    rows = []
    for threshold in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        mask = conf >= threshold
        n = int(mask.sum())
        rows.append(
            {
                "threshold": threshold,
                "coverage": round(n / len(y), 4),
                "accuracyOnCovered": round(float(np.mean(pred[mask] == y[mask])), 4) if n else 0.0,
            }
        )
    return rows


def refit_region(args) -> None:
    import lightgbm as lgb
    from sample import REGION_CLASSES
    from train_region import ABLATIONS, class_weights, layout, select

    config = json.loads(args.config)
    blocks = layout(json.loads(Path(args.schema).read_text()))
    x, y, w, _, _, _, _ = pool(Path(args.sample), False, Path(args.groups) if args.groups else None)
    x, columns = select(x, blocks, ABLATIONS[config["featureSet"]])
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
    booster = lgb.train(
        params,
        lgb.Dataset(x, label=y, weight=class_weights(y, float(config["weightPower"])) * w),
        num_boost_round=args.iterations,
    )
    cv = Path(args.cv)
    report = json.loads((cv / "cv-report.json").read_text())
    rows, oof = np.load(cv / "rows.npz"), np.load(cv / f"oof-{args.arm}.npz")
    keep = rows["evaluate"]
    out = Path(args.out) / "region"
    out.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out / "region-model.txt"))
    np.save(out / "columns-model.npy", columns)
    (out / "region-report.json").write_text(
        json.dumps(
            {
                "classes": REGION_CLASSES,
                "chosenFeatureSet": config["featureSet"],
                "refitOn": "train+val pool of the sample (no holdout)",
                "shipped": {
                    "config": config,
                    "iterations": args.iterations,
                    # export_model reads the global fallback threshold off this
                    # table; here it is the OOF table, not a single val split.
                    "coverageAtThresholdVal": coverage_table(
                        oof["pred"][keep].astype(np.int64),
                        oof["conf"][keep].astype(np.float32),
                        rows["y"][keep].astype(np.int64),
                    ),
                },
            },
            indent=2,
        )
        + "\n"
    )
    thresholds = report["arms"][args.arm]["thresholds"]
    (out / "class-thresholds.json").write_text(
        json.dumps(
            {
                "classes": REGION_CLASSES,
                "thresholds": thresholds,
                "chosenOn": f"cv_compare OOF, arm {args.arm}",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"head": "region", "rows": int(x.shape[0]), "thresholds": thresholds}))


def refit_section(args) -> None:
    import lightgbm as lgb
    from train_section import label_transform

    cfg = json.loads(args.config)
    sample = Path(args.sample)
    all_classes = json.loads((sample / "classes.json").read_text())["classes"]
    classes = [c for c in all_classes if c not in set(cfg.get("dropClasses", []))]
    remap = np.asarray(
        [classes.index(c) if c in classes else classes.index("none") for c in all_classes]
    )
    apply, _ = label_transform(
        all_classes,
        cfg.get("labels", ""),
        cfg.get("tightestLabels", ""),
        cfg.get("tightestClasses", "table_of_contents"),
        cfg.get("adMarkers", ""),
    )
    drop_rows = set()
    for path in cfg.get("dropRows", []):
        for line in Path(path).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                drop_rows.add((r["pageId"], r["nodeId"]))
    x, y, w, _, _, meta, split = pool(sample, True, Path(args.groups) if args.groups else None)
    y = remap[apply(y, meta)]
    keep = np.asarray(
        [
            (m["p"], m["n"]) not in drop_rows and (s == 0 or m["src"] == "jev")
            for m, s in zip(meta, split, strict=True)
        ]
    )
    x, y, w = x[keep], y[keep], w[keep].copy()
    meta = [m for m, k in zip(meta, keep, strict=True) if k]
    w[np.asarray([m["src"] == "negative" for m in meta])] = float(cfg["negativeWeight"])
    if cfg.get("balancePower", 0) > 0:
        counts = np.bincount(y, minlength=len(classes)).astype(np.float64)
        median = float(np.median(counts[counts > 0]))
        w = (
            w
            * np.where(counts > 0, (median / np.maximum(counts, 1)) ** cfg["balancePower"], 0.0)[y]
        )
    params = {
        "objective": "multiclass",
        "num_class": len(classes),
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "max_bin": 63,
        "num_threads": N_JOBS,
        "verbose": -1,
        "seed": 20260924,
        "num_leaves": int(cfg["leaves"]),
        "learning_rate": 0.05,
        "min_data_in_leaf": 10,
    }
    booster = lgb.train(
        params, lgb.Dataset(x, label=y, weight=w), num_boost_round=int(cfg["rounds"])
    )
    report = json.loads((Path(args.cv) / "cv-report.json").read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out / "section-model.txt"))
    thresholds = report["arms"][args.arm]["thresholds"]
    (out / "class-thresholds.json").write_text(
        json.dumps(
            {
                "classes": classes,
                "thresholds": thresholds,
                "chosenOn": f"cv_compare OOF, arm {args.arm}",
            }
        )
        + "\n"
    )
    print(json.dumps({"head": "section", "rows": int(x.shape[0]), "thresholds": thresholds}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("head", choices=("region", "section"))
    parser.add_argument("--sample", required=True)
    parser.add_argument("--groups", default="")
    parser.add_argument("--schema", default="")
    parser.add_argument("--config", required=True)
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--cv", required=True, help="cv_compare output dir (thresholds, OOF)")
    parser.add_argument("--arm", default="new")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    refit_region(args) if args.head == "region" else refit_section(args)


if __name__ == "__main__":
    main()
