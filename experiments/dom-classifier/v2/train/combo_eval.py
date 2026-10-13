# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy"]
# ///
"""Keyword gate + model probability, scored on the loop_eval held-out predictions (#2307).

For a class with a title/heading keyword flag (contact, legal, about, docs), the fire rule is
    (flag AND p >= t_low) OR p >= t_high
which both exist at runtime (skeleton features + the head's probability). Reports, per class, the best recall at
precision >= goal for: the model alone on a fine threshold grid, and the gated rule over a (t_low, t_high) grid.
Weights are the natural-pool decile weights, as in loop_eval.py.

  uv run --script train/combo_eval.py --oof <iter dir>/oof.jsonl --skeletons <v5 dir> --samples ... --decile-pool ... --out report.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

GATES = {"contact": ["kwContact"], "legal": ["kwLegal"], "about": ["kwAbout"], "docs": ["kwDocs"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True, help="pool_classes.json or any json list of the oof class order")
    parser.add_argument("--skeletons", type=Path, required=True)
    parser.add_argument("--samples", type=Path, action="append", required=True)
    parser.add_argument("--decile-pool", type=Path, required=True)
    parser.add_argument("--goal-precision", type=float, default=0.90)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    classes = json.loads(args.classes.read_text())
    rows = [json.loads(line) for line in args.oof.read_text().splitlines() if line.strip()]
    kind = {}
    for f in args.samples:
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                kind[r["p"]] = r["kind"]
    pool = {int(k): v for k, v in json.loads(args.decile_pool.read_text())["decilePool"].items()}
    dec = [int(kind[r["p"]].split("-d")[1]) for r in rows]
    drawn = Counter(dec)
    w = np.asarray([pool[d] / drawn[d] for d in dec])
    w = w / w.mean()
    names = None
    feats: dict[str, list] = {}
    wanted = {r["p"] for r in rows}
    for path in sorted(args.skeletons.glob("shard-*/skeletons.jsonl")):
        if names is None:
            names = json.loads((path.parent / "features.json").read_text())["order"]
        for line in path.open():
            r = json.loads(line)
            if r["p"] in wanted:
                feats[r["p"]] = r["f"]
    ix = {n: i for i, n in enumerate(names)}
    probs = np.asarray([r["probs"] for r in rows])
    y = np.asarray([classes.index(r["true"]) if r["true"] in classes else -1 for r in rows])
    report = {}
    grid = [round(0.05 + 0.01 * i, 2) for i in range(91)]

    def score(fire: np.ndarray, cls: int) -> tuple[float, float, int]:
        if fire.sum() == 0:
            return 0.0, 0.0, 0
        tp = float(w[fire & (y == cls)].sum())
        return tp / float(w[fire].sum()), tp / max(float(w[y == cls].sum()), 1e-9), int(fire.sum())

    for cls, name in enumerate(classes):
        if (y == cls).sum() < 10:
            continue
        p = probs[:, cls]
        best = None
        for t in grid:
            prec, rec, n = score(p >= t, cls)
            if n >= 10 and prec >= args.goal_precision and (best is None or rec > best["recall"]):
                best = {"threshold": t, "precision": round(prec, 3), "recall": round(rec, 3), "answered": n}
        entry = {"support": int((y == cls).sum()), "modelOnlyFineGrid": best}
        if name in GATES:
            gate = np.asarray([any(feats[r["p"]][ix[g]] > 0 for g in GATES[name]) for r in rows])
            gbest = None
            for t_high in grid[::3]:
                for t_low in grid[::3]:
                    if t_low > t_high:
                        continue
                    fire = ((p >= t_low) & gate) | (p >= t_high)
                    prec, rec, n = score(fire, cls)
                    if n >= 10 and prec >= args.goal_precision and (gbest is None or rec > gbest["recall"]):
                        gbest = {"tLow": t_low, "tHigh": t_high, "precision": round(prec, 3), "recall": round(rec, 3), "answered": n}
            entry["gatedRule"] = gbest
        report[name] = entry
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    for name, e in report.items():
        print(f"{name:15s} n={e['support']:4d} model-only {e['modelOnlyFineGrid']}   gated {e.get('gatedRule', '-')}")


if __name__ == "__main__":
    main()
