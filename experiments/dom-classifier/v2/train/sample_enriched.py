# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Training-only extra rows for the weak classes (#2307).

Draws schema-less pages whose skeleton features hint at a class the natural random sample has too few of
(product: price/buy; contact: tel/mailto/email text; legal: updated line; about/service: wpPage without
comments). Used ONLY as extra training rows, never for scoring: it shifts class prevalence on purpose, so
the natural random sample stays the evaluation set.

  uv run --script train/sample_enriched.py --skeletons <v2 dir> --schema <schema.jsonl> --sealed <sealed> \
      --manifest <manifest> --exclude <sample.jsonl> --exclude <random.jsonl> --out enriched.jsonl [--per 350]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("schema_labels_v8", HERE / "schema_labels_v8.py")
assert SPEC and SPEC.loader
V8 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V8)
WATCHED = {t for t, c in V8.TYPE_TO_V8.items() if c in ("article", "blog_post", "product_detail")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skeletons", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per", type=int, default=350)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()
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
        if not row.get("ok") or row["d"] in sealed:
            continue
        if not (set(row["ld"]) | set(row["ldNested"]) | set(row["md"])) & WATCHED:
            schemaless[row["sha"][:16]] = row["d"]
    paths = sorted(args.skeletons.glob("shard-*/skeletons.jsonl"))
    names = json.loads((paths[0].parent / "features.json").read_text())["order"]
    ix = {n: i for i, n in enumerate(names)}
    buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for path in paths:
        for line in path.open():
            r = json.loads(line)
            p, f = r["p"], r["f"]
            if p not in schemaless or p in excluded or p in root or source.get(p) == "cloud":
                continue
            g = lambda n: f[ix[n]]  # noqa: E731
            if g("buy") or g("price") or g("priceTerms"):
                buckets["product"].append((p, schemaless[p]))
            if g("tel") or g("mailto") or g("emailText") or g("phoneText"):
                buckets["contact"].append((p, schemaless[p]))
            if g("updated"):
                buckets["legal"].append((p, schemaless[p]))
            if g("wpPage") and not g("commentItems") and not g("wpAttachment"):
                buckets["about"].append((p, schemaless[p]))
    rng = random.Random(args.seed)
    out, seen = [], set()
    used: Counter = Counter()
    for kind, items in buckets.items():
        rng.shuffle(items)
        got = 0
        for p, d in items:
            if got >= args.per:
                break
            if p in seen or used[d] >= 2:
                continue
            seen.add(p)
            used[d] += 1
            got += 1
            out.append({"p": p, "kind": f"enriched-{kind}", "d": d})
    args.out.write_text("".join(json.dumps(r) + "\n" for r in out))
    print(json.dumps({"pools": {k: len(v) for k, v in buckets.items()}, "drawn": dict(Counter(r["kind"] for r in out)), "total": len(out)}))


if __name__ == "__main__":
    main()
