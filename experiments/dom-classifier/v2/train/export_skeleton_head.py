# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "lightgbm"]
# ///
"""Pack the skeleton page-type head into the runtime's tree format (#2307).

Reads loop_eval.py --save-model output (model.txt + meta.json) and writes `skeleton-head.json`: a HeadJson
(the same packed-tree block as the region/page/section heads, via export_model.pack) plus the feature names
the runtime must fill, in order. Parity: the float32 reference evaluator (export_model.evaluate) against
LightGBM's own float64 predictions on real feature rows; the TypeScript walk is tested against the same
reference separately.

  uv run --script train/export_skeleton_head.py --model-dir <dir> --skeletons <shard dir> --out <dir> [--parity-rows 2000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_model as em  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--skeletons", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parity-rows", type=int, default=2000)
    args = parser.parse_args()
    meta = json.loads((args.model_dir / "meta.json").read_text())
    classes, names = meta["classes"], meta["featureNames"]
    if meta.get("tokens"):
        raise SystemExit("the tokens variant is not exportable: the runtime head reads skeleton features only")
    booster, head_json, reference = em.head(
        args.model_dir / "model.txt", None, classes, threshold=1.01, class_thresholds=meta["classThresholds"]
    )
    if booster.num_feature() != len(names):
        raise SystemExit(f"model has {booster.num_feature()} features, meta lists {len(names)}")

    rows = []
    for path in sorted(args.skeletons.glob("shard-*/skeletons.jsonl")):
        with path.open() as handle:
            for line in handle:
                rows.append(json.loads(line)["f"][: len(names)])
                if len(rows) >= args.parity_rows:
                    break
        if len(rows) >= args.parity_rows:
            break
    x = np.asarray(rows, dtype=np.float32)
    lgb_probs = booster.predict(x)
    ref_probs = em.evaluate(reference, x)
    diff = np.abs(lgb_probs - ref_probs)
    agree = float((lgb_probs.argmax(axis=1) == ref_probs.argmax(axis=1)).mean())
    parity = {"rows": len(rows), "maxAbsProbabilityDiff": float(diff.max()), "meanAbsDiff": float(diff.mean()), "argmaxAgreement": agree}
    print(json.dumps(parity))
    if agree < 0.999 or diff.max() > 1e-3:
        raise SystemExit("parity failed: the packed head disagrees with LightGBM beyond float32 rounding")

    args.out.mkdir(parents=True, exist_ok=True)
    out = {"format": "squirrel-skeleton-head", "headVersion": 1, "featureNames": names, "head": head_json,
           "evaluation": meta["evaluation"], "trainedOn": meta["trainedOn"], "goalPrecision": meta["goalPrecision"], "parity": parity}
    (args.out / "skeleton-head.json").write_text(json.dumps(out) + "\n")
    # fixture for the TypeScript parity test: rows and the reference probabilities (float32 evaluator)
    fixture = args.out / "parity-fixture.json"
    fixture.write_text(json.dumps({"names": names, "classes": classes, "rows": x[:200].tolist(), "probs": ref_probs[:200].tolist()}) + "\n")
    print(f"wrote {args.out / 'skeleton-head.json'} ({(args.out / 'skeleton-head.json').stat().st_size / 1024:.0f} KB), classes {classes}")


if __name__ == "__main__":
    main()
