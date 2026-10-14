# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "lightgbm"]
# ///
"""Score exported node features with a section head's LightGBM model, without the TypeScript stack (#2307).

Reads `nodes-features.jsonl` rows ([197 numeric `f`, token buckets `k`]) and writes the same rows with `p.<name>` = the head's
class probabilities, plus the `<out>.classes.json` sidecar that `adjudicate/section_blind.py` reads. The model file is
`section-stack.txt` from export_section_stack.py, whose parity against the runtime evaluator is checked at export time, so
this gives the numbers of `bun run experiments/export-section-nodes.ts --artifact` in seconds.

    uv run --script score_section_head.py --model dir/section-stack.txt --meta dir/section-head.meta.json \
        --nodes blind/nodes-features.jsonl --out dir/blind-nodes.jsonl [--name new]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import scipy.sparse as sp

BUCKETS = 512


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", default="new")
    args = parser.parse_args()

    meta = json.loads(args.meta.read_text())
    rows = [json.loads(line) for line in args.nodes.read_text().splitlines() if line.strip()]
    tok = sp.lil_matrix((len(rows), BUCKETS), dtype=np.float32)
    for i, r in enumerate(rows):
        tok[i, r["k"]] = 1.0
    dense = sp.csr_matrix(np.asarray([r["f"] for r in rows], dtype=np.float32))
    x = sp.hstack([dense, tok.tocsr()]).tocsr()
    if len(rows[0]["f"]) != meta["nDense"]:
        raise SystemExit(f"dense width {len(rows[0]['f'])} != model {meta['nDense']}")
    probs = lgb.Booster(model_file=str(args.model)).predict(x)
    with args.out.open("w") as f:
        for r, p in zip(rows, probs, strict=True):
            f.write(json.dumps({**r, "p": {args.name: [round(float(v), 4) for v in p]}}) + "\n")
    sidecar = args.out.with_name(args.out.name + ".classes.json")
    sidecar.write_text(json.dumps({"classes": {args.name: meta["classes"]}, "featurizerVersion": meta.get("featurizerVersion")}) + "\n")
    print(json.dumps({"rows": len(rows), "classes": len(meta["classes"]), "out": str(args.out)}))


if __name__ == "__main__":
    main()
