# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Active-learning pick: unlabelled pages the current model is unsure about, per weak class (#2307).

Reads loop_eval.py --pool-out probabilities and keeps, for each wanted class, pages whose probability for it sits
in [lo, hi] (the model thinks it might be that class but cannot tell). At most 2 pages per domain, no page twice.
Output rows are TRAINING-ONLY samples (`kind: active-<class>`), never evaluation rows.

  uv run --script train/active_pick.py --pool pool.jsonl --classes pool_classes.json --out active.jsonl \
      [--want product_detail=350 --want about=300 --want contact=300 --want listing=200 --want legal=100]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--want", action="append", default=[])
    parser.add_argument("--lo", type=float, default=0.12)
    parser.add_argument("--hi", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=71)
    args = parser.parse_args()
    classes = json.loads(args.classes.read_text())
    want = {k: int(v) for k, _, v in (w.partition("=") for w in args.want)} or {
        "product_detail": 350, "about": 300, "contact": 300, "listing": 200, "legal": 100}  # fmt: skip
    pool = [json.loads(line) for line in args.pool.read_text().splitlines() if line.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    out, seen, per_domain = [], set(), Counter()
    for name, n in want.items():
        i = classes.index(name)
        got = 0
        for row in pool:
            if got >= n:
                break
            p = row["probs"][i]
            if not args.lo <= p <= args.hi or row["p"] in seen or per_domain[row["d"]] >= 2:
                continue
            seen.add(row["p"])
            per_domain[row["d"]] += 1
            got += 1
            out.append({"p": row["p"], "kind": f"active-{name}", "d": row["d"], "prob": p})
    args.out.write_text("".join(json.dumps(r) + "\n" for r in out))
    print(json.dumps({"drawn": dict(Counter(r["kind"] for r in out)), "total": len(out)}))


if __name__ == "__main__":
    main()
