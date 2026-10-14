# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "lightgbm"]
# ///
"""Train the natural-node section stage and pack it as a drop-in section head (#2307).

Same input as the shipped section head ([197 numeric, 512 token buckets]), so the runtime needs no change: the output
`section-head.json` is a HeadJson that replaces `section` in an artifact. Per-class thresholds come from page-grouped
out-of-fold predictions (lowest bar with precision >= goal and >= --min-answers answers); a class with none abstains (1.01).
Parity: float32 reference evaluator vs LightGBM on real rows.

    uv run --script export_section_stack.py --nodes nodes-features.jsonl --answers haiku.jsonl --opus opus.jsonl --out dir
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import lightgbm as lgb
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_model as em  # noqa: E402

BARS = [round(0.3 + 0.01 * i, 2) for i in range(70)]


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--answers", type=Path, action="append", required=True)
    parser.add_argument("--opus", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--goal-precision", type=float, default=0.9)
    parser.add_argument("--min-answers", type=int, default=10)
    parser.add_argument("--min-class", type=int, default=15)
    parser.add_argument("--natural-only-thresholds", action="store_true",
                        help="choose thresholds on out-of-fold rows whose itemId does not start with secb- (mined, class-enriched)")
    args = parser.parse_args()

    truth: dict[str, str] = {}
    item_of: dict[str, str] = {}
    for path in [*args.answers, *args.opus]:
        for row in jsonl(path):
            if not row.get("error") and row.get("label"):
                truth[row["itemId"]] = row["label"]
                item_of[row["itemId"]] = row.get("itemOf") or ""
    rows = [r for r in jsonl(args.nodes) if r["itemId"] in truth]
    label_of = {r["itemId"]: ("comments" if item_of[r["itemId"]] == "comments" else truth[r["itemId"]]) for r in rows}
    counts = Counter(label_of.values())
    classes = sorted(c for c, n in counts.items() if n >= args.min_class and c != "none") + ["none"]
    y = np.asarray([classes.index(label_of[r["itemId"]]) if label_of[r["itemId"]] in classes else len(classes) - 1 for r in rows])
    groups = np.asarray([int(hashlib.sha256(r["pageId"].encode()).hexdigest()[:8], 16) % args.folds for r in rows])
    n_dense, buckets = len(rows[0]["f"]), 512
    tok = sp.lil_matrix((len(rows), buckets), dtype=np.float32)
    for i, r in enumerate(rows):
        tok[i, r["k"]] = 1.0
    x = sp.hstack([sp.csr_matrix(np.asarray([r["f"] for r in rows], dtype=np.float32)), tok.tocsr()]).tocsr()

    params = {"objective": "multiclass", "num_class": len(classes), "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 8,
              "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1, "num_threads": 2, "verbose": -1, "seed": 5,
              "lambda_l2": 1.0}
    oof = np.zeros((len(rows), len(classes)))
    for k in range(args.folds):
        test = groups == k
        oof[test] = lgb.train(params, lgb.Dataset(x[~test], label=y[~test]), num_boost_round=args.rounds).predict(x[test])
    pred, conf = oof.argmax(axis=1), oof.max(axis=1)
    nat = np.asarray([not r["itemId"].startswith("secb-") for r in rows]) if args.natural_only_thresholds else np.ones(len(rows), bool)
    thresholds, report = [], {}
    for i, cls in enumerate(classes):
        if cls == "none":
            thresholds.append(1.01)
            continue
        chosen = None
        for bar in BARS:
            fired = (pred == i) & (conf >= bar) & nat
            if fired.sum() < args.min_answers:
                break
            prec = float((y[fired] == i).mean())
            if prec >= args.goal_precision:
                chosen = {"bar": bar, "answered": int(fired.sum()), "precision": round(prec, 3),
                          "recall": round(float(((y == i) & fired).sum() / max(((y == i) & nat).sum(), 1)), 3)}
                break  # lowest bar holding the goal = most recall
        thresholds.append(chosen["bar"] if chosen else 1.01)
        report[cls] = {"support": int(((y == i) & nat).sum()), "supportAll": int((y == i).sum()), "threshold": chosen["bar"] if chosen else None, "oof": chosen}

    final = lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=args.rounds)
    args.out.mkdir(parents=True, exist_ok=True)
    model_path = args.out / "section-stack.txt"
    final.save_model(str(model_path))
    booster, head_json, reference = em.head(model_path, None, classes, threshold=1.01, class_thresholds=thresholds)
    dense_x = x[:2000].toarray().astype(np.float32)
    lgb_probs, ref_probs = booster.predict(dense_x), em.evaluate(reference, dense_x)
    diff = np.abs(lgb_probs - ref_probs)
    agree = float((lgb_probs.argmax(axis=1) == ref_probs.argmax(axis=1)).mean())
    parity = {"rows": len(dense_x), "maxAbsProbabilityDiff": float(diff.max()), "argmaxAgreement": agree}
    print(json.dumps(parity))
    if agree < 0.999 or diff.max() > 1e-3:
        raise SystemExit("parity failed")
    (args.out / "section-head.json").write_text(json.dumps(head_json) + "\n")
    (args.out / "section-head.meta.json").write_text(json.dumps({"classes": classes, "thresholds": thresholds, "perClass": report,
        "nodes": len(rows), "goalPrecision": args.goal_precision, "nDense": n_dense, "parity": parity}, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
