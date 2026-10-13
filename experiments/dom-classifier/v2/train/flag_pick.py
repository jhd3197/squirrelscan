# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Round-3 training-only picks by title/heading keyword flags (#2307).

The model's uncertain band holds few real contact/legal/about pages (the keyword flags already make it confident
on most). To get more POSITIVES for those rare classes, draw schema-less pages whose title or first heading
carries the class keyword (skeleton features kwContact/kwLegal/kwAbout/kwPricing/kwDocs), plus price/buy pages for
product_detail. The teacher still decides every label; these rows are TRAINING-ONLY (they shift prevalence on purpose).

  uv run --script train/flag_pick.py --skeletons <v5 dir> --schema <schema.jsonl> --sealed <sealed> --manifest <manifest> \
      --exclude <sample files...> --out flag.jsonl [--want contact=450 ...]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("schema_labels_v8", HERE / "schema_labels_v8.py")
assert SPEC and SPEC.loader
V8 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V8)
WATCHED = {t for t, c in V8.TYPE_TO_V8.items() if c in ("article", "blog_post", "product_detail")}
RULES = {
    "contact": lambda g: g("kwContact"),
    "legal": lambda g: g("kwLegal"),
    "about": lambda g: g("kwAbout"),
    "docs": lambda g: g("kwDocs"),
    "product_detail": lambda g: g("kwPricing") or g("price") or g("buy"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skeletons", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--want", action="append", default=[])
    parser.add_argument("--seed", type=int, default=97)
    args = parser.parse_args()
    want = {k: int(v) for k, _, v in (w.partition("=") for w in args.want)} or {
        "contact": 450, "legal": 450, "about": 400, "product_detail": 400, "docs": 250}  # fmt: skip
    sealed = {json.loads(line)["d"] for line in args.sealed.read_text().splitlines() if line.strip()}
    excluded = {json.loads(line)["p"] for f in args.exclude for line in f.read_text().splitlines() if line.strip()}
    source, root = {}, set()
    for line in args.manifest.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        sha = ((row.get("raw") or {}).get("sha256") or "")[:16]
        source[sha] = row["source"]
        if row.get("is_homepage"):
            root.add(sha)
    schemaless: dict[str, str] = {}
    for line in args.schema.read_text().splitlines():
        row = json.loads(line)
        if row.get("ok") and row["d"] not in sealed and not (set(row["ld"]) | set(row["ldNested"]) | set(row["md"])) & WATCHED:
            schemaless[row["sha"][:16]] = row["d"]
    paths = sorted(args.skeletons.glob("shard-*/skeletons.jsonl"))
    ix = {n: i for i, n in enumerate(json.loads((paths[0].parent / "features.json").read_text())["order"])}
    pools: dict[str, list[tuple[str, str]]] = {k: [] for k in want}
    for path in paths:
        for line in path.open():
            r = json.loads(line)
            p = r["p"]
            if p not in schemaless or p in excluded or p in root or source.get(p) == "cloud":
                continue
            g = lambda n: r["f"][ix[n]]  # noqa: E731
            for name in want:
                if RULES[name](g):
                    pools[name].append((p, schemaless[p]))
    rng = random.Random(args.seed)
    out, seen, per_domain = [], set(), Counter()
    for name, n in want.items():
        rng.shuffle(pools[name])
        got = 0
        for p, d in pools[name]:
            if got >= n:
                break
            if p in seen or per_domain[d] >= 2:
                continue
            seen.add(p)
            per_domain[d] += 1
            got += 1
            out.append({"p": p, "kind": f"flag-{name}", "d": d})
    args.out.write_text("".join(json.dumps(r) + "\n" for r in out))
    print(json.dumps({"pools": {k: len(v) for k, v in pools.items()}, "drawn": dict(Counter(r["kind"] for r in out)), "total": len(out)}))


if __name__ == "__main__":
    main()
