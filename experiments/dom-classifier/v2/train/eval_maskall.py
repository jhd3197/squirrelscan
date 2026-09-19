"""Score the shipped region model on the FULLY token-hidden holdout.

The training rows hide a label word in a node's class/id half the time, so the
model learns that class="site-footer" is evidence without being able to lean on
it alone. This run hides them every time, which is the situation on a div-soup
page whose class names say nothing: it is the honest lower bound, and the
number to quote when someone asks what happens on a site with no semantics.

Reuses the metric functions from train_region.py so the two columns are
computed by the same code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))
from sample import CLASS_INDEX, REGION_CLASSES, collapse  # noqa: E402
from train_region import (  # noqa: E402
    ABLATIONS,
    coverage_at_threshold,
    layout,
    per_class_metrics,
    select,
)

CHUNK = 20_000


def load_rows(path: Path, n_numeric: int, n_tokens: int):
    width = n_numeric + n_tokens
    blocks: list[sp.csr_matrix] = []
    labels: list[int] = []
    buffer = np.zeros((CHUNK, width), dtype=np.float32)
    fill = 0
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            label = collapse(row["l"])
            if label is None:
                continue
            buffer[fill, :n_numeric] = row["f"]
            for bucket in row["k"]:
                buffer[fill, n_numeric + bucket] = 1.0
            labels.append(CLASS_INDEX[label])
            fill += 1
            if fill == CHUNK:
                blocks.append(sp.csr_matrix(buffer))
                buffer[:] = 0.0
                fill = 0
    if fill:
        blocks.append(sp.csr_matrix(buffer[:fill]))
    return sp.vstack(blocks, format="csr"), np.asarray(labels, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--models", required=True)
    args = parser.parse_args()

    import lightgbm as lgb

    root = Path(args.data_root)
    models = Path(args.models)
    schema = json.loads((root / "silver-v1" / "feature-schema.json").read_text())
    blocks = layout(schema)
    report = json.loads((models / "region" / "region-report.json").read_text())
    feature_set = report["chosenFeatureSet"]
    iterations = report["shipped"]["iterations"]

    x, y = load_rows(root / "models" / "holdout-maskall.jsonl",
                     schema["nodeFeatureCount"], schema["tokenBuckets"])
    print(f"fully-hidden holdout: {x.shape[0]} rows", flush=True)

    booster = lgb.Booster(model_file=str(models / "region" / "region-model.txt"))
    xs, _ = select(x, blocks, ABLATIONS[feature_set])
    probs = booster.predict(xs, num_iteration=iterations)
    pred = probs.argmax(axis=1)

    out = {
        "rows": int(x.shape[0]),
        "featureSet": feature_set,
        "iterations": iterations,
        "note": "every class/id token equal to a label word removed (mask probability 1.0)",
        **per_class_metrics(y, pred),
        "coverageAtThreshold": coverage_at_threshold(probs, y),
    }
    (models / "region" / "region-maskall-report.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"fully-hidden macroF1 {out['macroF1']} acc {out['accuracy']}", flush=True)
    for name in REGION_CLASSES:
        row = out["perClass"][name]
        print(f"  {name:16s} P {row['precision']:.3f} R {row['recall']:.3f} "
              f"F1 {row['f1']:.3f} n={row['support']}", flush=True)


if __name__ == "__main__":
    main()
