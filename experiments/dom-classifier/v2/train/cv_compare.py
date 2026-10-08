"""Old corpus vs old+new corpus, measured by domain-grouped CV only (#2307).

Two arms, one evaluation set. The NEW sample (old + new pages) is cut into K
folds by a hash of each row's domain (the same rule as train_pagetype_v8's
--cv-folds). For every fold f:
  - arm `old` fits on the OLD sample's rows outside fold f (the corpus the
    shipped head was trained on, sampled by the shipped rule);
  - arm `new` fits on the NEW sample's rows outside fold f;
  - both predict the NEW sample's rows inside fold f.
Every evaluated row is therefore scored by models that never saw its domain,
and both arms are scored on exactly the same rows. Holdout files are never
opened: this script reads only X-/y-/w- train and val.

Heads:
  region   the A-full config (feature set, leaves, learning rate, weight power)
           with a FIXED iteration count (the shipped one); thresholds per arm
           on its own OOF non-declared rows, by class_thresholds.sweep.
  section  the S9 config (leaves, rounds, negative weight, balance power, label
           transforms of train_section); thresholds on OOF Jev rows.

Per-row inputs the samples do not carry are passed as small arrays:
  <sample>/bucket-{train,val}.npy  uint32 sha256(domainHash)[0:8] (fold = % K)
  <sample>/page-{train,val}.npy    int32 index into <sample>/pages.json
Output: oof-<arm>.npz (pred, conf per evaluated row), rows.npz (y, w, page,
fold, eval mask), thresholds and pooled metrics in cv-report.json. Slicing by
platform / language happens offline (cv_slices.py) from the page index.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

N_JOBS = int(os.environ.get("DOM_CLASSIFIER_JOBS", "4"))


def load(sample: Path, split: str, with_meta: bool, groups: Path):
    x = sp.load_npz(sample / f"X-{split}.npz").tocsr()
    y = np.load(sample / f"y-{split}.npy").astype(np.int64)
    w = np.load(sample / f"w-{split}.npy").astype(np.float64)
    bucket = np.load(groups / f"bucket-{split}.npy").astype(np.int64)
    page = np.load(groups / f"page-{split}.npy").astype(np.int64)
    meta = None
    if with_meta:
        meta = [
            json.loads(line) for line in (sample / f"meta-{split}.jsonl").read_text().splitlines()
        ]
    return x, y, w, bucket, page, meta


def pool(sample: Path, with_meta: bool, groups: Path | None = None):
    """Train + val of one sample as one CV pool. `groups` holds the
    sample_groups.py arrays when they live outside the sample directory."""
    parts = [load(sample, split, with_meta, groups or sample) for split in ("train", "val")]
    x = sp.vstack([p[0] for p in parts], format="csr")
    y = np.concatenate([p[1] for p in parts])
    w = np.concatenate([p[2] for p in parts])
    bucket = np.concatenate([p[3] for p in parts])
    page = np.concatenate([p[4] for p in parts])
    meta = (parts[0][5] or []) + (parts[1][5] or []) if with_meta else None
    split = np.concatenate(
        [np.zeros(len(parts[0][1]), np.int8), np.ones(len(parts[1][1]), np.int8)]
    )
    return x, y, w, bucket, page, meta, split


# ---------------------------------------------------------------------------
# region
# ---------------------------------------------------------------------------


def run_region(args, out: Path) -> dict:
    import lightgbm as lgb
    from class_thresholds import non_declared_mask, sweep
    from sample import REGION_CLASSES
    from train_region import ABLATIONS, class_weights, layout, select

    config = json.loads(args.config)
    schema = json.loads(Path(args.schema).read_text())
    blocks = layout(schema)
    names = ABLATIONS[config["featureSet"]]
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

    def weights(y, w):
        return class_weights(y, float(config["weightPower"])) * w

    ex, ey, ew, eb, epage, _, esplit = pool(Path(args.new_sample), False, new_groups(args))
    ex = select(ex, blocks, names)[0]
    efold = eb % args.folds
    evaluate = non_declared_mask(ey, ew)
    oof = {
        arm: (np.full(len(ey), -1, np.int8), np.zeros(len(ey), np.float16))
        for arm in ("old", "new")
    }
    timings = []
    for arm in args.arms.split(","):
        if arm == "new":
            tx, ty, tw, tfold = ex, ey, ew, efold
        else:
            ox, oy, ow, ob, _, _, _ = pool(Path(args.old_sample), False, old_groups(args))
            tx, ty, tw, tfold = select(ox, blocks, names)[0], oy, ow, ob % args.folds
        for f in range(args.folds):
            started = time.time()
            fit = tfold != f
            booster = lgb.train(
                params,
                lgb.Dataset(tx[fit], label=ty[fit], weight=weights(ty[fit], tw[fit])),
                num_boost_round=args.iterations,
            )
            test = efold == f
            probs = booster.predict(ex[test])
            oof[arm][0][test] = probs.argmax(1)
            oof[arm][1][test] = probs.max(1)
            timings.append(
                {
                    "arm": arm,
                    "fold": f,
                    "fitRows": int(fit.sum()),
                    "seconds": round(time.time() - started),
                }
            )
            print(json.dumps(timings[-1]), flush=True)
            del booster, probs
        if arm == "old":
            del tx, ty, tw, tfold
    np.savez_compressed(
        out / "rows.npz",
        y=ey.astype(np.int8),
        w=ew.astype(np.float32),
        page=epage,
        fold=efold.astype(np.int8),
        split=esplit,
        evaluate=evaluate,
    )
    report = {
        "head": "region",
        "classes": REGION_CLASSES,
        "config": config,
        "iterations": args.iterations,
        "folds": args.folds,
        "timings": timings,
        "population": "non-declared nodes (runtime population)",
        "arms": {},
    }
    for arm in args.arms.split(","):
        pred, conf = oof[arm]
        np.savez_compressed(out / f"oof-{arm}.npz", pred=pred, conf=conf)
        # sweep() reads a class's own probability; at the argmax that IS conf.
        probs_like = np.zeros((int(evaluate.sum()), len(REGION_CLASSES)), np.float32)
        probs_like[np.arange(len(probs_like)), pred[evaluate].astype(np.int64)] = conf[evaluate]
        chosen = [sweep(probs_like, ey[evaluate], i) for i in range(len(REGION_CLASSES))]
        report["arms"][arm] = {"thresholds": [c["threshold"] for c in chosen], "chosen": chosen}
    return report


# ---------------------------------------------------------------------------
# section
# ---------------------------------------------------------------------------


def run_section(args, out: Path) -> dict:
    import lightgbm as lgb
    from train_section import label_transform, pick_thresholds

    cfg = json.loads(args.config)
    new_sample, old_sample = Path(args.new_sample), Path(args.old_sample)
    all_classes = json.loads((new_sample / "classes.json").read_text())["classes"]
    dropped = set(cfg.get("dropClasses", []))
    classes = [c for c in all_classes if c not in dropped]
    remap = np.asarray(
        [classes.index(c) if c in classes else classes.index("none") for c in all_classes]
    )
    apply, rule_counts = label_transform(
        all_classes,
        cfg.get("labels", ""),
        cfg.get("tightestLabels", ""),
        cfg.get("tightestClasses", "table_of_contents"),
        cfg.get("adMarkers", ""),
    )
    drop_rows: set = set()
    for path in cfg.get("dropRows", []):
        for line in Path(path).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                drop_rows.add((row["pageId"], row["nodeId"]))
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

    def prepared(sample: Path, groups: Path | None):
        x, y, w, bucket, page, meta, split = pool(sample, True, groups)
        y = remap[apply(y, meta)]
        keep = np.asarray([(m["p"], m["n"]) not in drop_rows for m in meta])
        # Val contributes its Jev rows only, as in train_section.
        keep &= np.asarray([s == 0 or m["src"] == "jev" for m, s in zip(meta, split, strict=True)])
        x, y, w, bucket, page, split = (
            x[keep],
            y[keep],
            w[keep].copy(),
            bucket[keep],
            page[keep],
            split[keep],
        )
        meta = [m for m, k in zip(meta, keep, strict=True) if k]
        w[np.asarray([m["src"] == "negative" for m in meta])] = float(cfg["negativeWeight"])
        if cfg.get("balancePower", 0) > 0:
            counts = np.bincount(y, minlength=len(classes)).astype(np.float64)
            median = float(np.median(counts[counts > 0]))
            factor = np.where(
                counts > 0, (median / np.maximum(counts, 1)) ** cfg["balancePower"], 0.0
            )
            w = w * factor[y]
        return x, y, w, bucket, page, meta, split

    ex, ey, ew, eb, epage, emeta, esplit = prepared(new_sample, new_groups(args))
    efold = eb % args.folds
    judged = np.asarray([m["src"] == "jev" for m in emeta])
    oof = {}
    timings = []
    for arm in args.arms.split(","):
        if arm == "new":
            tx, ty, tw, tfold = ex, ey, ew, efold
        else:
            tx, ty, tw, tb, _, _, _ = prepared(old_sample, old_groups(args))
            tfold = tb % args.folds
        pred = np.full(len(ey), -1, np.int8)
        conf = np.zeros(len(ey), np.float16)
        for f in range(args.folds):
            started = time.time()
            fit = tfold != f
            booster = lgb.train(
                params,
                lgb.Dataset(tx[fit], label=ty[fit], weight=tw[fit]),
                num_boost_round=int(cfg["rounds"]),
            )
            test = efold == f
            probs = booster.predict(ex[test])
            pred[test] = probs.argmax(1)
            conf[test] = probs.max(1)
            timings.append(
                {
                    "arm": arm,
                    "fold": f,
                    "fitRows": int(fit.sum()),
                    "seconds": round(time.time() - started),
                }
            )
            print(json.dumps(timings[-1]), flush=True)
        oof[arm] = (pred, conf)
    np.savez_compressed(
        out / "rows.npz",
        y=ey.astype(np.int8),
        w=ew.astype(np.float32),
        page=epage,
        fold=efold.astype(np.int8),
        split=esplit,
        evaluate=judged,
        node=np.asarray([m["n"] for m in emeta], np.int32),
    )
    report = {
        "head": "section",
        "classes": classes,
        "config": cfg,
        "folds": args.folds,
        "timings": timings,
        "promoAdRule": dict(rule_counts),
        "population": "Jev-labelled rows (train shortlist + val)",
        "arms": {},
    }
    for arm, (pred, conf) in oof.items():
        np.savez_compressed(out / f"oof-{arm}.npz", pred=pred, conf=conf)
        probs_like = np.zeros((int(judged.sum()), len(classes)), np.float32)
        probs_like[np.arange(len(probs_like)), pred[judged].astype(np.int64)] = conf[judged]
        thresholds = pick_thresholds(
            probs_like, ey[judged], len(classes), args.target_precision, args.min_fires
        )
        thresholds[classes.index("none")] = 0.0
        report["arms"][arm] = {"thresholds": thresholds}
    return report


def old_groups(args) -> Path | None:
    return Path(args.old_groups) if args.old_groups else None


def new_groups(args) -> Path | None:
    return Path(args.new_groups) if args.new_groups else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", choices=("region", "section"), required=True)
    parser.add_argument("--old-sample", required=True)
    parser.add_argument("--new-sample", required=True)
    parser.add_argument("--old-groups", default="", help="sample_groups.py output for --old-sample")
    parser.add_argument("--new-groups", default="", help="sample_groups.py output for --new-sample")
    parser.add_argument("--config", required=True, help="JSON head config (see module doc)")
    parser.add_argument("--schema", default="", help="feature-schema.json (region)")
    parser.add_argument("--iterations", type=int, default=0, help="region: fixed boosting rounds")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--arms", default="old,new")
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument("--min-fires", type=int, default=15)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = run_region(args, out) if args.head == "region" else run_section(args, out)
    (out / "cv-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"head": args.head, "arms": list(report["arms"])}), flush=True)


if __name__ == "__main__":
    main()
