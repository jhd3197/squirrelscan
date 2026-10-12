# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Copy a CC link-batch staging dir (modal_cc_batch.py) into the shared corpus (#2307).

    uv run --script place_cc_batch.py --staging <dir> --name cc-linkctx-32k [--dry-run]

Placement is the guard (page-corpus README):
  train run    pages  -> probe/<name>/{blobs, manifest.jsonl}
               labels -> labels/treetop/link-section-v1/<name>/{links,pages}.jsonl
  holdout run  pages  -> holdout/treetop/<name>/{blobs, manifest.jsonl}
               labels -> holdout/treetop/link-section-v1/<name>/{links,pages}.jsonl
With `--merge-headers [<backfill dir> ...]` it instead rewrites `headers.jsonl` beside
each side's manifest from the staging dir and the headers-only backfill runs.

A train-run capture whose page content (sha256) already sits anywhere under
holdout/ or gold/ goes to the holdout side instead, as `<name>-held`: the same bytes must never
be readable by training. Place the holdout run first, so its pages count.

Never overwrites: a destination manifest or label file that exists stops the
run. Take HEAVY.lock around it. Counts only on stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

CORPUS = Path("/Volumes/BACKUP/page-corpus")
SHA = re.compile(r"[0-9a-f]{64}")
SHA_FIELD = re.compile(r'"(?:page_sha256|sha256|content_sha256)":\s*"([0-9a-f]{64})"')


def protected_shas() -> set[str]:
    """Every page hash under holdout/ and gold/: blob names and manifest fields."""
    out: set[str] = set()
    for root in (CORPUS / "holdout", CORPUS / "gold"):
        for dirpath, _, files in os.walk(root):
            for name in files:
                if m := SHA.match(name):
                    out.add(m.group(0))
                elif name.endswith(".jsonl") and "locators" not in dirpath:
                    with open(os.path.join(dirpath, name)) as handle:
                        for line in handle:
                            out.update(SHA_FIELD.findall(line))
    return out


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def split_rows(staging: Path, route: str, protected: set[str]) -> dict[str, dict[str, list[dict]]]:
    """Rows of each file, by destination side."""
    manifest = read_jsonl(staging / "manifest.jsonl")
    held = {r["capture"] for r in manifest if route == "holdout" or (r["page_sha256"] or "") in protected}
    sides: dict[str, dict[str, list[dict]]] = {"train": {}, "holdout": {}}
    for name in ("manifest", "links", "pages", "headers"):
        if not (staging / f"{name}.jsonl").exists():
            continue
        for row in read_jsonl(staging / f"{name}.jsonl"):
            side = "holdout" if row["capture"] in held else "train"
            sides[side].setdefault(name, []).append(row)
    return sides


def destinations(side: str, name: str) -> tuple[Path, Path]:
    if side == "train":
        return CORPUS / "probe" / name, CORPUS / "labels/treetop/link-section-v1" / name
    return CORPUS / "holdout/treetop" / name, CORPUS / "holdout/treetop/link-section-v1" / name


def write_side(staging: Path, side: str, files: dict[str, list[dict]], name: str, dry_run: bool) -> Counter:
    """Copy one side's blobs and rows to their destination, refusing overwrites."""
    pages_dir, labels_dir = destinations(side, name)
    targets = {
        "manifest": pages_dir / "manifest.jsonl",
        "links": labels_dir / "links.jsonl",
        "pages": labels_dir / "pages.jsonl",
        "headers": pages_dir / "headers.jsonl",
    }
    for key, path in targets.items():
        if files.get(key) and path.exists():
            raise SystemExit(f"refusing to overwrite {path.relative_to(CORPUS)}")
    # A row can carry a hash with no stored page (a response that was not HTML).
    shas = {
        r["page_sha256"]
        for r in files.get("manifest", [])
        if r["page_sha256"] and (staging / "blobs" / f"{r['page_sha256']}.html.gz").exists()
    }
    counts = Counter({f"{side}:{key}": len(files.get(key, [])) for key in targets})
    counts[f"{side}:blobs"] = len(shas)
    if dry_run:
        return counts
    (pages_dir / "blobs").mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for sha in shas:
        target = pages_dir / "blobs" / f"{sha}.html.gz"
        if not target.exists():
            shutil.copyfile(staging / "blobs" / f"{sha}.html.gz", target)
            counts[f"{side}:blobs_copied"] += 1
    for key, path in targets.items():
        if files.get(key):
            path.write_text("".join(json.dumps(r) + "\n" for r in files[key]))
    return counts


def place(staging: Path, name: str, dry_run: bool) -> Counter:
    route = (staging / "route").read_text().strip()
    if route not in ("train", "holdout"):
        raise SystemExit(f"{staging}: no route marker")
    counts: Counter = Counter()
    for side, files in split_rows(staging, route, protected_shas()).items():
        if files:
            # Held-over captures of a train run get their own folder, so a holdout
            # run of the same name never collides with them.
            side_name = f"{name}-held" if side == "holdout" and route == "train" else name
            counts += write_side(staging, side, files, side_name, dry_run)
    return counts


def merge_headers(staging: Path, extra: list[Path], name: str, dry_run: bool) -> Counter:
    """(Re)write `headers.jsonl` beside each side's manifest from `staging` and the
    headers-only backfill dirs in `extra`. It is derived data, so it may be rewritten."""
    route = (staging / "route").read_text().strip()
    sides = split_rows(staging, route, protected_shas())
    side_of = {r["capture"]: side for side, files in sides.items() for r in files.get("manifest", [])}
    held_name = f"{name}-held" if route == "train" else name
    rows: dict[str, dict] = {}
    for directory in [staging, *extra]:
        path = directory / "headers.jsonl"
        for row in read_jsonl(path) if path.exists() else []:
            rows[row["capture"]] = row
    counts: Counter = Counter()
    out: dict[str, list[dict]] = {"train": [], "holdout": []}
    for capture, row in rows.items():
        if capture in side_of:
            out[side_of[capture]].append(row)
        else:
            counts["headers_for_unknown_capture"] += 1
    for side, side_rows in out.items():
        counts[f"{side}:headers"] = len(side_rows)
        counts[f"{side}:captures_without_headers"] = sum(1 for c, sd in side_of.items() if sd == side and c not in rows)
        if side_rows and not dry_run:
            pages_dir, _ = destinations(side, held_name if side == "holdout" else name)
            if not pages_dir.exists():
                raise SystemExit(f"{pages_dir.relative_to(CORPUS)} is not placed yet")
            (pages_dir / "headers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in side_rows))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--merge-headers",
        nargs="*",
        type=Path,
        default=None,
        help="instead of placing, rewrite headers.jsonl from --staging and these headers-only backfill dirs",
    )
    args = parser.parse_args()
    if args.merge_headers is not None:
        print(json.dumps(dict(merge_headers(args.staging, args.merge_headers, args.name, args.dry_run))))
    else:
        print(json.dumps(dict(place(args.staging, args.name, args.dry_run))))


if __name__ == "__main__":
    main()
