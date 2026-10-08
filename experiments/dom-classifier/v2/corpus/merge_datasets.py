"""Merge an extension dataset into a base dataset version (#2307).

silver-v4 = silver-v3 + the Common Crawl extension's TRAIN and VAL splits.
The extension's holdout-bucket pages are left out on purpose: they are the
stored test slice for a later one-time score and must never sit in a dataset a
trainer reads.

Both inputs must come from the same featurizer (feature-schema.json is compared
byte for byte) and must not disagree on any domain's split. Row files are
concatenated as gzip members (every reader here, Python gzip and Bun zlib,
reads multi-member files), so nothing is decompressed but the page index.

  python merge_datasets.py --base <root>/silver-v3 --extra <root>/silver-cc \
      --extra-splits train,val --out <root>/silver-v4
Also writes <out>/extension-pages.txt: the page ids that came from --extra.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path

SPLITS = ("train", "val", "holdout")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--extra", required=True)
    parser.add_argument("--extra-splits", default="train,val")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    base, extra, out = Path(args.base), Path(args.extra), Path(args.out)
    take = [s for s in args.extra_splits.split(",") if s]
    if "holdout" in take:
        raise SystemExit("the extension's holdout pages are a test slice: never merged")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)

    schema = (base / "feature-schema.json").read_bytes()
    if (extra / "feature-schema.json").read_bytes() != schema:
        raise SystemExit(
            "feature schemas differ: the two datasets were built by different featurizers"
        )
    (out / "feature-schema.json").write_bytes(schema)

    manifest = json.loads((base / "split-manifest.json").read_text())
    extra_manifest = json.loads((extra / "split-manifest.json").read_text())
    clashes = {
        d
        for d, s in extra_manifest["domains"].items()
        if d in manifest["domains"] and manifest["domains"][d] != s
    }
    if clashes:
        raise SystemExit(f"{len(clashes)} domains have a different split in the two manifests")
    overlap = sum(1 for d in extra_manifest["domains"] if d in manifest["domains"])
    for domain, split in extra_manifest["domains"].items():
        if split in take:
            manifest["domains"][domain] = split
    manifest.update(
        {"datasetVersion": out.name, "mergedFrom": [base.name, extra.name], "extraSplits": take}
    )
    (out / "split-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    for split in SPLITS:
        for kind in ("nodes", "pages"):
            name = f"{kind}-{split}.jsonl.gz"
            with (out / name).open("wb") as sink:
                with (base / name).open("rb") as source:
                    shutil.copyfileobj(source, sink, 1 << 20)
                if split in take:
                    with (extra / name).open("rb") as source:
                        shutil.copyfileobj(source, sink, 1 << 20)

    added: list[str] = []
    with gzip.open(out / "page-index.jsonl.gz", "wt") as sink:
        with gzip.open(base / "page-index.jsonl.gz", "rt") as source:
            shutil.copyfileobj(source, sink, 1 << 20)
        with gzip.open(extra / "page-index.jsonl.gz", "rt") as source:
            for line in source:
                row = json.loads(line)
                if row["s"] in take:
                    sink.write(line)
                    added.append(row["p"])
    (out / "extension-pages.txt").write_text("\n".join(added) + "\n")

    base_stats = json.loads((base / "stats.json").read_text())
    extra_stats = json.loads((extra / "stats.json").read_text())
    stats = {
        "datasetVersion": out.name,
        "featurizerVersion": base_stats["featurizerVersion"],
        "base": {
            "version": base.name,
            "pages": base_stats["pages"],
            "domains": base_stats["domains"],
        },
        "extra": {
            "version": extra.name,
            "pages": extra_stats["pages"],
            "domains": extra_stats["domains"],
            "splitsMerged": take,
        },
        "pages": {
            s: base_stats["pages"][s] + (extra_stats["pages"][s] if s in take else 0)
            for s in SPLITS
        },
        "domainsShared": overlap,
        "extensionPagesAdded": len(added),
    }
    if extra_stats["featurizerVersion"] != base_stats["featurizerVersion"]:
        raise SystemExit("featurizer versions differ")
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
