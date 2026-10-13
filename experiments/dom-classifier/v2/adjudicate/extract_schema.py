# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Declared page types from structured data, one row per stored page (#2307).

Reads the shared corpus manifest and every stored blob (read-only) and writes what the
site itself declared about the page: JSON-LD @type values, og:type, microdata itemtype
and RDFa typeof. This is a SECOND opinion beside our own labels, never a label by
itself: disagreements are the review queue and the "schema missing / schema wrong"
findings. Model inputs must have these signals stripped.

  uv run --script extract_schema.py --corpus /Volumes/BACKUP/page-corpus --out <dir> [--limit N]

Output `schema.jsonl` rows:
  {p, d, s, root, sha, ld: {type: topLevelCount}, ldNested: {...}, md: {type: n}, rdfa: {...},
   og: str|null, ldBlocks, ldErrors, ok}
The `p d s root ld ldNested md` keys are the schema-types.jsonl shape that
train/schema_labels_v8.py reads, so its mapping and noise filters apply unchanged.
Types are the last path segment of schema.org URLs ("https://schema.org/Article" -> "Article").
No page text is written.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path

MAX_BYTES = 8 * 1024 * 1024
LD = re.compile(rb"<script\b[^>]*\btype\s*=\s*[\"']?application/ld\+json[\"']?[^>]*>(.*?)</script>", re.I | re.S)
META = re.compile(rb"<meta\b[^>]*>", re.I)
ATTR = re.compile(rb"([a-zA-Z:_-]+)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s>]+))")
ITEMTYPE = re.compile(rb"\bitemtype\s*=\s*(?:\"([^\"]*)\"|'([^']*)')", re.I)
TYPEOF = re.compile(rb"\btypeof\s*=\s*(?:\"([^\"]*)\"|'([^']*)')", re.I)


def short(value: str) -> str:
    return value.strip().rstrip("/").rsplit("/", 1)[-1].rsplit(":", 1)[-1]


def types_of(node: dict) -> list[str]:
    kind = node.get("@type")
    if isinstance(kind, str):
        return [short(kind)]
    if isinstance(kind, list):
        return [short(k) for k in kind if isinstance(k, str)]
    return []


def walk(node: object, top: Counter, nested: Counter, depth: int = 0) -> None:
    """Top-level entities (roots, and the members of a top-level @graph) vs everything nested."""
    if depth > 8:
        return
    if isinstance(node, list):
        for item in node[:200]:
            walk(item, top, nested, depth)
    elif isinstance(node, dict):
        counter = top if depth == 0 else nested
        counter.update(types_of(node))
        for key, value in list(node.items())[:60]:
            if isinstance(value, (dict, list)):
                walk(value, top, nested, depth if key == "@graph" and depth == 0 else depth + 1)


def extract(raw: bytes) -> dict:
    top: Counter = Counter()
    nested: Counter = Counter()
    blocks = bad = 0
    for match in LD.finditer(raw):
        blocks += 1
        try:
            walk(json.loads(match.group(1).decode("utf-8", "replace").strip()), top, nested)
        except (ValueError, RecursionError):
            bad += 1
    og = None
    for tag in META.findall(raw):
        attrs = {m[0].lower(): (m[1] or m[2] or m[3]) for m in ATTR.findall(tag)}
        if attrs.get(b"property", b"").lower() == b"og:type" and attrs.get(b"content"):
            og = attrs[b"content"].decode("utf-8", "replace").strip().lower()[:40]
            break
    micro: Counter = Counter()
    for m in ITEMTYPE.finditer(raw):
        for v in (m.group(1) or m.group(2) or b"").split():
            micro[short(v.decode("utf-8", "replace"))] += 1
    rdfa: Counter = Counter()
    for m in TYPEOF.finditer(raw):
        for v in (m.group(1) or m.group(2) or b"").split():
            rdfa[short(v.decode("utf-8", "replace"))] += 1
    cap = lambda c: dict(c.most_common(40))  # noqa: E731
    return {"ld": cap(top), "ldNested": cap(nested), "md": cap(micro), "rdfa": cap(rdfa), "og": og,
            "ldBlocks": blocks, "ldErrors": bad}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    done = 0
    with (args.out / "schema.jsonl").open("w") as sink:
        for line in (args.corpus / "pages" / "manifest.jsonl").open():
            page = json.loads(line)
            raw_info = page.get("raw") or {}
            if raw_info.get("status") != "stored" or not raw_info.get("sha256"):
                counts["not_stored"] += 1
                continue
            if page["bucket"] >= 90:
                counts["skipped_bucket90"] += 1
                continue
            row = {"p": page["page_id"], "d": page["domain_hash"], "s": page["split"], "root": bool(page.get("is_homepage")), "sha": raw_info["sha256"]}
            try:
                with gzip.open(args.corpus / "pages" / "blobs" / f"{raw_info['sha256']}.html.gz", "rb") as handle:
                    body = handle.read(MAX_BYTES)
                row |= extract(body) | {"ok": True}
            except (OSError, EOFError) as error:
                row |= {"ok": False, "error": type(error).__name__}
            counts["ok" if row["ok"] else "error"] += 1
            if row.get("ld"):
                counts["has_jsonld"] += 1
            if row.get("og"):
                counts["has_og"] += 1
            sink.write(json.dumps(row) + "\n")
            done += 1
            if args.limit and done >= args.limit:
                break
    (args.out / "report.json").write_text(json.dumps(dict(counts), indent=1) + "\n")
    print(json.dumps(dict(counts)))


if __name__ == "__main__":
    main()
