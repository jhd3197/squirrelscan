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
    parser.add_argument("--rounds", type=int, default=800)
    parser.add_argument("--negative-weight", type=float, default=0.5,
                        help="weight of the random unlabelled `none` rows (they can hide a section)")
    parser.add_argument("--leaves", default="63,127", help="num_leaves grid")
    parser.add_argument("--drop-classes", default="",
                        help="comma list folded into `none` (no data to learn or evaluate them)")
    parser.add_argument("--cv-folds", type=int, default=0,
                        help="thresholds from domain-grouped OOF over train+val (needs `d` in "
                             "the sample meta); 0 = val split")
    parser.add_argument("--balance-power", type=float, default=0.0,
                        help="class-balance exponent: rows weighted by (median/class count)^p")
    args = parser.parse_args()

    import lightgbm as lgb

    sample = Path(args.sample)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    all_classes = json.loads((sample / "classes.json").read_text())["classes"]
    dropped = {c for c in args.drop_classes.split(",") if c}
    classes = [c for c in all_classes if c not in dropped]
    remap = np.asarray([classes.index(c) if c in classes else classes.index("none")
                        for c in all_classes])
    xt, yt, wt, mt = load(sample, "train")
    yt = remap[yt]
    wt = wt.copy()
    negative = np.asarray([m["src"] == "negative" for m in mt])
    wt[negative] = args.negative_weight
    if args.balance_power > 0:
        counts = np.bincount(yt, minlength=len(classes)).astype(np.float64)
        median = float(np.median(counts[counts > 0]))
        factor = np.where(counts > 0, (median / np.maximum(counts, 1)) ** args.balance_power, 0.0)
        wt = wt * factor[yt]
    # Val and holdout are scored on Jev-labelled rows ONLY. The random
    # unlabelled `none` rows the sample adds can hide a real section, and a
    # prediction on one would count as a false positive it may not be.
    xv, yv, wv, mv = load(sample, "val")
    keep_v = np.asarray([m["src"] == "jev" for m in mv])
    xv, yv, wv = xv[keep_v], remap[yv[keep_v]], wv[keep_v]
    mv = [m for m, k in zip(mv, keep_v, strict=True) if k]
    xh, yh, _, mh = load(sample, "holdout")
    keep_h = np.asarray([m["src"] == "jev" for m in mh])
    xh, yh = xh[keep_h], remap[yh[keep_h]]
    mh = [m for m, k in zip(mh, keep_h, strict=True) if k]
    print(f"train {xt.shape} val {xv.shape} holdout {xh.shape}", flush=True)

    base = {"objective": "multiclass", "num_class": len(classes), "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 1, "max_bin": 63, "num_threads": N_JOBS,
            "verbose": -1, "seed": 20260924}
    # No early stopping on weighted log-loss: with ~40k `none` rows it stops
    # before the rare classes are learned. One long run per config, read at
    # several iteration counts, and the count is picked on val macro-F1.
    best, grid = None, []
    for leaves in [int(v) for v in args.leaves.split(",")]:
        params = {**base, "num_leaves": leaves, "learning_rate": 0.05, "min_data_in_leaf": 10}
        booster = lgb.train(params, lgb.Dataset(xt, label=yt, weight=wt), num_boost_round=args.rounds)
        for k in range(100, args.rounds + 1, 100):
            vp = booster.predict(xv, num_iteration=k)
            val = score(vp, yv, [0.0] * len(classes), classes)["macroF1NonNone"]
            grid.append({"leaves": leaves, "iters": k, "valMacroF1NonNone": val})
            print(json.dumps(grid[-1]), flush=True)
            if best is None or val > best[0]:
                best = (val, booster, vp, k)
    _, booster, vp, chosen_k = best
    booster.best_iteration = chosen_k
    sel_probs, sel_y = vp, yv
    if args.cv_folds > 0:
        # Domain-grouped out-of-fold probabilities over train + val (val's Jev
        # rows, train's rows as sampled), for thresholds that do not lean on one
        # split's domains. The shipped model stays the one fit on train.
        import hashlib

        leaves_k = max(grid, key=lambda g: g["valMacroF1NonNone"])
        params = {**base, "num_leaves": leaves_k["leaves"], "learning_rate": 0.05,
                  "min_data_in_leaf": 10}
        xp = sp.vstack([xt, xv], format="csr")
        yp = np.concatenate([yt, yv])
        wp = np.concatenate([wt, wv])
        domains = [m["d"] for m in mt] + [m["d"] for m in mv]
        fold = np.asarray([int(hashlib.sha256(d.encode()).hexdigest()[:8], 16) % args.cv_folds
                           for d in domains])
        # Thresholds are judged on Jev rows only: a free label is itself only
        # ~0.85-0.98 precise, so counting it as truth caps the reachable
        # precision below the target and every class would abstain.
        judged = np.concatenate([np.asarray([m["src"] == "jev" for m in mt]),
                                 np.ones(len(yv), bool)])
        oof = np.zeros((xp.shape[0], len(classes)))
        for f in range(args.cv_folds):
            test = fold == f
            fb = lgb.train(params, lgb.Dataset(xp[~test], label=yp[~test], weight=wp[~test]),
                           num_boost_round=chosen_k)
            oof[test] = fb.predict(xp[test])
        sel_probs, sel_y = oof[judged], yp[judged]
    thresholds = pick_thresholds(sel_probs, sel_y, len(classes), args.target_precision,
                                 args.min_fires)
    thresholds[classes.index("none")] = 0.0
    hp = booster.predict(xh, num_iteration=booster.best_iteration)
    report = {
        "classes": classes, "grid": grid, "iterations": int(booster.best_iteration),
        "thresholds": dict(zip(classes, thresholds, strict=True)),
        "targetPrecision": args.target_precision,
        "thresholdsFrom": f"oof-{args.cv_folds}fold" if args.cv_folds else "val",
        "droppedClasses": sorted(dropped),
        # Why a class abstains: its precision and fire count on the selection
        # set at a few thresholds.
        "selectionCurve": {
            name: {f"{t:.2f}": [int(((sel_probs.argmax(1) == i) & (sel_probs.max(1) >= t)).sum()),
                                round(float((sel_y[(sel_probs.argmax(1) == i)
                                                   & (sel_probs.max(1) >= t)] == i).mean()), 3)
                                if ((sel_probs.argmax(1) == i) & (sel_probs.max(1) >= t)).any()
                                else None]
                   for t in (0.5, 0.7, 0.9, 0.95, 0.99)}
            for i, name in enumerate(classes)},
        "selectionRows": int(len(sel_y)),
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
