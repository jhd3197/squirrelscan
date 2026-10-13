# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "lightgbm"]
# ///
"""Second-stage section model trained on labelled NATURAL nodes (#2307).

Inputs per node: the shipped head's probability vector, the 197 dense node features and the 512 hashed tokens (the same
vector the head reads). Labels: the text judge's section (Opus over Haiku); an ITEM of a section is none, except
`comments` (the rule marks every node of a thread). Page-grouped K-fold: the held-out predictions are what is scored.
Compared with the head alone (its top class + probability, thresholds off). For each class: best recall at
precision >= goal with >= --min-answers held-out answers.

    uv run --script section_stack.py --nodes nodes-features.jsonl --classes nodes-features.jsonl.classes.json --artifact artifact3 \
        --answers haiku.jsonl --opus opus.jsonl --out report.json [--rounds 200 --folds 5 --no-head]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import lightgbm as lgb
import numpy as np
import scipy.sparse as sp

BARS = [round(0.3 + 0.01 * i, 2) for i in range(70)]


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--artifact", default="artifact3")
    parser.add_argument("--answers", type=Path, action="append", required=True)
    parser.add_argument("--opus", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--goal-precision", type=float, default=0.9)
    parser.add_argument("--min-answers", type=int, default=10)
    parser.add_argument("--min-class", type=int, default=15, help="classes with fewer labelled nodes fold into none")
    parser.add_argument("--no-head", action="store_true", help="leave the shipped head's probabilities out of the inputs")
    args = parser.parse_args()

    meta = json.loads(args.classes.read_text())
    head_classes = meta["classes"][args.artifact]
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
    dense = np.asarray([r["f"] for r in rows], dtype=np.float32)
    head = np.asarray([r["p"][args.artifact] for r in rows], dtype=np.float32)
    tok = sp.lil_matrix((len(rows), buckets), dtype=np.float32)
    for i, r in enumerate(rows):
        tok[i, r["k"]] = 1.0
    parts = [sp.csr_matrix(dense), tok.tocsr()] + ([] if args.no_head else [sp.csr_matrix(head)])
    x = sp.hstack(parts).tocsr()
    print(f"{len(rows)} nodes, classes {dict((c, int((y == i).sum())) for i, c in enumerate(classes))}", flush=True)

    params = {"objective": "multiclass", "num_class": len(classes), "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 8,
              "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1, "num_threads": 2, "verbose": -1, "seed": 5,
              "lambda_l2": 1.0}
    oof = np.zeros((len(rows), len(classes)))
    for k in range(args.folds):
        test = groups == k
        booster = lgb.train(params, lgb.Dataset(x[~test], label=y[~test]), num_boost_round=args.rounds)
        oof[test] = booster.predict(x[test])
    pred, conf = oof.argmax(axis=1), oof.max(axis=1)

    def is_class(i: int, cls_idx: int) -> bool:
        return int(y[i]) == cls_idx

    report: dict = {"nodes": len(rows), "classes": classes, "labelled": {c: int((y == i).sum()) for i, c in enumerate(classes)}, "stack": {}, "head": {}}
    head_idx = {c: j for j, c in enumerate(head_classes)}
    for i, cls in enumerate(classes[:-1]):
        support = int((y == i).sum())
        for name, labels, confs in (("stack", pred, conf),):
            best, curve = None, []
            for bar in BARS:
                fired = (labels == i) & (confs >= bar)
                if fired.sum() < args.min_answers:
                    break
                prec = float((y[fired] == i).mean())
                rec = float(((y == i) & fired).sum() / max(support, 1))
                if prec >= args.goal_precision and (best is None or rec > best["recall"]):
                    best = {"bar": bar, "answered": int(fired.sum()), "precision": round(prec, 3), "recall": round(rec, 3)}
            report["stack"][cls] = {"support": support, "bestAtGoal": best}
        if cls in head_idx:
            j = head_idx[cls]
            best = None
            for bar in BARS:
                fired = (head.argmax(axis=1) == j) & (head[:, j] >= bar)
                if fired.sum() < args.min_answers:
                    break
                prec = float((y[fired] == i).mean())
                rec = float(((y == i) & fired).sum() / max(support, 1))
                if prec >= args.goal_precision and (best is None or rec > best["recall"]):
                    best = {"bar": bar, "answered": int(fired.sum()), "precision": round(prec, 3), "recall": round(rec, 3)}
            fired_any = head.argmax(axis=1) == j
            report["head"][cls] = {"support": support, "bestAtGoal": best, "precisionAnyThreshold": round(float((y[fired_any] == i).mean()), 3) if fired_any.any() else None}
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"{'class':18s} {'support':>7s}   head best@{args.goal_precision}      stack best@{args.goal_precision}")
    for cls in classes[:-1]:
        h = report["head"].get(cls, {}).get("bestAtGoal")
        s = report["stack"][cls]["bestAtGoal"]
        fmt = lambda b: f"{b['precision']:.2f}@{b['recall']:.2f} (n={b['answered']})" if b else "-"  # noqa: E731
        print(f"{cls:18s} {report['stack'][cls]['support']:7d}   {fmt(h):26s} {fmt(s)}")


if __name__ == "__main__":
    main()
