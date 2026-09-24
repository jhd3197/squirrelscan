"""Train the section-purpose head (section-purpose-v2) and report holdout once.

Input: the `section_sample.py` directory. Config picked on val macro-F1 over the
classes val supports; per-class thresholds on val for precision >=
--target-precision (a class that cannot reach it, or has too few val fires,
abstains: threshold 1.01). Holdout is scored once, against Jev labels only.

Also reported:
  - adGeneralization: holdout recall on Jev `ad` nodes WITHOUT a known ad-network
    token (adToken false) vs with one, at the ad threshold - whether the head
    learned ads or memorised "adsbygoogle";
  - per class precision@threshold, coverage (recall at threshold), argmax F1.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import scipy.sparse as sp

N_JOBS = int(os.environ.get("DOM_CLASSIFIER_JOBS", "4"))


def load(sample: Path, split: str):
    x = sp.load_npz(sample / f"X-{split}.npz").tocsr()
    y = np.load(sample / f"y-{split}.npy").astype(np.int64)
    w = np.load(sample / f"w-{split}.npy").astype(np.float64)
    meta = [json.loads(line) for line in (sample / f"meta-{split}.jsonl").read_text().splitlines()]
    return x, y, w, meta


def pick_thresholds(probs, y, n_class, target, min_fires):
    top, conf = probs.argmax(1), probs.max(1)
    out = []
    for i in range(n_class):
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


def score(probs, y, thresholds, classes):
    top, conf = probs.argmax(1), probs.max(1)
    fires = conf >= np.asarray(thresholds)[top]
    out, f1s = {}, []
    for i, name in enumerate(classes):
        support = int((y == i).sum())
        tp = int(((top == i) & (y == i)).sum())
        fp = int(((top == i) & (y != i)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / support if support else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        fired = fires & (top == i)
        n = int(fired.sum())
        out[name] = {"support": support, "f1": round(f1, 4), "precision": round(p, 4),
                     "recall": round(r, 4), "threshold": thresholds[i], "fires": n,
                     "precisionAtThreshold": round(float((y[fired] == i).mean()), 4) if n else None,
                     "coverageAtThreshold": round(int((y[fired] == i).sum()) / support, 4)
                     if support else None}
        if support and name != "none":
            f1s.append(f1)
    return {"perClass": out, "macroF1NonNone": round(float(np.mean(f1s)), 4) if f1s else 0.0,
            "n": int(len(y))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument("--min-fires", type=int, default=15)
    parser.add_argument("--rounds", type=int, default=1500)
    args = parser.parse_args()

    import lightgbm as lgb

    sample = Path(args.sample)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    classes = json.loads((sample / "classes.json").read_text())["classes"]
    xt, yt, wt, _ = load(sample, "train")
    xv, yv, wv, _ = load(sample, "val")
    xh, yh, _, mh = load(sample, "holdout")
    print(f"train {xt.shape} val {xv.shape} holdout {xh.shape}", flush=True)

    base = {"objective": "multiclass", "num_class": len(classes), "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 1, "max_bin": 63, "num_threads": N_JOBS,
            "verbose": -1, "seed": 20260924}
    best, grid = None, []
    for leaves in (63, 127):
        params = {**base, "num_leaves": leaves, "learning_rate": 0.05, "min_data_in_leaf": 10}
        booster = lgb.train(params, lgb.Dataset(xt, label=yt, weight=wt), num_boost_round=args.rounds,
                            valid_sets=[lgb.Dataset(xv, label=yv, weight=wv)],
                            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        vp = booster.predict(xv, num_iteration=booster.best_iteration)
        val = score(vp, yv, [0.0] * len(classes), classes)["macroF1NonNone"]
        grid.append({"leaves": leaves, "iters": booster.best_iteration, "valMacroF1NonNone": val})
        print(json.dumps(grid[-1]), flush=True)
        if best is None or val > best[0]:
            best = (val, booster, vp)
    _, booster, vp = best
    thresholds = pick_thresholds(vp, yv, len(classes), args.target_precision, args.min_fires)
    thresholds[classes.index("none")] = 0.0
    hp = booster.predict(xh, num_iteration=booster.best_iteration)
    report = {
        "classes": classes, "grid": grid, "iterations": int(booster.best_iteration),
        "thresholds": dict(zip(classes, thresholds, strict=True)),
        "targetPrecision": args.target_precision,
        "val": score(vp, yv, thresholds, classes), "holdout": score(hp, yh, thresholds, classes),
    }
    ad = classes.index("ad")
    top, conf = hp.argmax(1), hp.max(1)
    fires_ad = (top == ad) & (conf >= thresholds[ad])
    gen = {}
    for flag in (True, False):
        mask = np.asarray([m.get("adToken") is flag for m in mh]) & (yh == ad)
        n = int(mask.sum())
        gen["withNetworkToken" if flag else "withoutNetworkToken"] = {
            "adNodes": n,
            "recallAtThreshold": round(float(fires_ad[mask].mean()), 4) if n else None,
            "argmaxRecall": round(float((top[mask] == ad).mean()), 4) if n else None,
        }
    report["adGeneralization"] = gen
    booster.save_model(str(out / "section-model.txt"), num_iteration=booster.best_iteration)
    (out / "class-thresholds.json").write_text(
        json.dumps({"classes": classes, "thresholds": thresholds}) + "\n")
    (out / "section-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"holdoutMacroF1NonNone": report["holdout"]["macroF1NonNone"],
                      "adGeneralization": gen}), flush=True)


if __name__ == "__main__":
    main()
