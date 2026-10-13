# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "lightgbm"]
# ///
"""Does the page skeleton lift the page-type head? (#2307)

Same pool, labels, folds and booster as train_pagetype_jev.py (domain-grouped 5-fold CV
inside the Jev-labelled train+val pool; the holdout is NOT read), three feature sets:

  A  existing page features (node-featurizer page vector + hashed class/id tokens)
  B  A + skeleton features (runtime/skeleton.ts: depth, repeat runs, flags, landmarks)
  C  skeleton features only

and reports per class: precision/recall/F1 on argmax, plus precision and coverage when
the head only answers above a confidence bar (the way squirrel would use it).

  DOM_CLASSIFIER_DATASET_VERSION=silver-cc uv run --script train/ablate_skeleton.py \
      --data-root /Volumes/BACKUP/dom-classifier-data/2026-10-08-cc \
      --jev <labels/pagetype-v81> --skeletons <skeletons.jsonl> --out <dir>

Schema, og:type and URL tokens are in none of the three sets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_pagetype_jev as base  # noqa: E402

CONFIG = {"learningRate": 0.05, "numLeaves": 31, "minDataInLeaf": 20}
BARS = (0.8, 0.9, 0.95)


def load_skeleton(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            out[row["p"]] = np.asarray(row["f"], dtype=np.float32)
    return out


def bar_report(oof: np.ndarray, y: np.ndarray) -> dict:
    confidence = oof.max(axis=1)
    pred = oof.argmax(axis=1)
    out: dict = {}
    for bar in BARS:
        answered = confidence >= bar
        per_class = {}
        for index, name in enumerate(base.CLASSES):
            said = answered & (pred == index)
            n = int(said.sum())
            per_class[name] = {
                "answered": n,
                "precision": round(float((y[said] == index).mean()), 3) if n else None,
                "recallOfClass": round(float((said & (y == index)).sum() / max((y == index).sum(), 1)), 3),
            }
        out[str(bar)] = {
            "coverage": round(float(answered.mean()), 3),
            "precision": round(float((pred[answered] == y[answered]).mean()), 3) if answered.any() else None,
            "perClass": per_class,
        }
    return out


def run_variant(name: str, x: sp.csr_matrix, y, groups, w, rounds: int, folds: int) -> dict:
    assignments = np.asarray([base.fold_of(g, folds) for g in groups])
    oof = np.zeros((x.shape[0], len(base.CLASSES)))
    iterations = []
    for fold in range(folds):
        test = assignments == fold
        train = ~test
        booster = base.train_booster(x[train], y[train], w[train], x[test], y[test], w[test], CONFIG, rounds)
        oof[test] = booster.predict(x[test], num_iteration=booster.best_iteration)
        iterations.append(int(booster.best_iteration))
        print(f"{name} fold {fold} best_iteration {iterations[-1]}", flush=True)
    return {
        "argmax": base.per_class_metrics(y, oof.argmax(axis=1)),
        "atConfidence": bar_report(oof, y),
        "iterations": iterations,
        "features": int(x.shape[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--jev", required=True)
    parser.add_argument("--skeletons", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=400)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    root = Path(args.data_root)
    args.out.mkdir(parents=True, exist_ok=True)

    schema = json.loads((root / base.DATASET / "feature-schema.json").read_text())
    n_numeric, n_tokens = schema["pageFeatureCount"], schema["tokenBuckets"]
    features: dict[str, dict] = {}
    for split in ("train", "val"):  # the holdout split is never read here
        features.update(base.load_features(root / base.DATASET / f"pages-{split}.jsonl.gz", n_numeric, n_tokens, split))
    skeleton = load_skeleton(args.skeletons)
    jev = {**base.load_jev(Path(args.jev) / "train.jsonl"), **base.load_jev(Path(args.jev) / "val.jsonl")}

    # keep only pages that have both feature sets, so A, B and C see the same rows
    keep = {p: f for p, f in features.items() if p in skeleton}
    for p in keep:
        keep[p] = {**keep[p], "x": keep[p]["x"], "sk": skeleton[p]}
    x_a, y, groups, w, _, ids = base.assemble(jev, keep, frozenset({"train", "val"}))
    sk = sp.csr_matrix(np.vstack([skeleton[p] for p in ids]))
    x_b = sp.hstack([x_a, sk]).tocsr()
    print(f"pool {x_a.shape[0]} pages / {len(set(groups))} domains (of {len(features)} featured, {len(skeleton)} skeletons)", flush=True)

    report = {
        "pool": {"pages": int(x_a.shape[0]), "domains": len(set(groups)),
                 "byClass": {c: int((y == i).sum()) for i, c in enumerate(base.CLASSES)}},
        "protocol": "domain-grouped 5-fold CV, holdout not read, same rows for A/B/C",
    }
    for name, x in (("A_existing", x_a), ("B_existing_plus_skeleton", x_b), ("C_skeleton_only", sk)):
        report[name] = run_variant(name, x, y, groups, w, args.rounds, args.folds)
        (args.out / "ablation.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: {"macroF1": v["argmax"]["macroF1"], "acc": v["argmax"]["accuracy"],
                          "at0.9": {"cov": v["atConfidence"]["0.9"]["coverage"], "prec": v["atConfidence"]["0.9"]["precision"]}}
                      for k, v in report.items() if k.startswith(("A_", "B_", "C_"))}, indent=1))


if __name__ == "__main__":
    main()
