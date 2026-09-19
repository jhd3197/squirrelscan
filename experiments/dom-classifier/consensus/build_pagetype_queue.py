#!/usr/bin/env python3
"""Build the deterministic page queues for the Jev page-type labelling run.

Order is fixed by the brief: every holdout and val page first, because those are
the splits an evaluation reads, then a stratified train sample.

The train sample over-represents the pages the silver labels cannot describe:
pages with no schema source at all, and pages whose URL slug suggests a class
schema.org has no vocabulary for (`/pricing`, `/legal`, `/terms`, `/careers`,
…). **The slug is used only to choose which pages to label, never as a label**,
and it is not carried into the labelling packet: the annotator sees the URL path
like any other page and decides from the page. Sampling on a signal and
labelling from it are different things, and only the first is sound here.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

DATA_ROOT = Path(os.environ["DOM_CLASSIFIER_DATA_ROOT"])
SILVER = DATA_ROOT / "2026-09-19" / "classifier-v2" / "silver-v1"

# Slug families whose classes have no schema.org source, so the silver labels
# are blind to them. Sampling aid only.
RARE_SLUGS = {
    "pricing": re.compile(r"/(pricing|plans|price|subscribe|upgrade)(/|$|\?)", re.I),
    "legal": re.compile(r"/(legal|privacy|terms|tos|cookie|gdpr|dpa|policy|policies|imprint|accessibility)(/|$|\?)", re.I),
    "docs": re.compile(r"/(docs?|documentation|guide|guides|manual|reference|api|kb|help|support|faq)(/|$|\?)", re.I),
    "careers": re.compile(r"/(careers?|jobs?|hiring|work-with-us|join-us)(/|$|\?)", re.I),
    "contact": re.compile(r"/(contact|contact-us|get-in-touch|locations?)(/|$|\?)", re.I),
    "about": re.compile(r"/(about|about-us|company|team|who-we-are|our-story)(/|$|\?)", re.I),
}


def read_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def order_key(page_id: str) -> str:
    """Content-derived and stable; never crawl order, never `random`."""
    return hashlib.sha256(page_id.encode()).hexdigest()


def slug_family(url: str) -> str | None:
    for name, pattern in RARE_SLUGS.items():
        if pattern.search(url):
            return name
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-sample", type=int, default=12_000)
    parser.add_argument("--max-per-domain", type=int, default=8)
    args = parser.parse_args()
    if not str(args.output_dir.resolve()).startswith(f"{DATA_ROOT.resolve()}/"):
        raise ValueError("output must be under the private data root")

    index = {row["p"]: row for row in read_gz(SILVER / "page-index.jsonl.gz")}
    queues: dict[str, list[dict[str, Any]]] = {}
    schema_state: dict[str, dict[str, str]] = {}
    for split in ("holdout", "val", "train"):
        pages = read_gz(SILVER / f"pages-{split}.jsonl.gz")
        schema_state[split] = {row["p"]: row.get("ls") or "none" for row in pages}
        rows = []
        for page in pages:
            entry = index.get(page["p"])
            if not entry:
                continue
            rows.append(
                {
                    "pageId": page["p"],
                    # The pageId is a unique prefix of the store's content hash;
                    # the label row keys on the hash so it binds to exact bytes.
                    "contentHash": entry.get("h") or page["p"],
                    "url": entry["u"],
                    "split": split,
                    "schemaLabel": page.get("l"),
                    "schemaSource": page.get("ls"),
                    "divSoup": page.get("divSoup"),
                }
            )
        queues[split] = rows

    # Holdout and val in full.
    selected = {
        split: sorted(queues[split], key=lambda row: order_key(row["pageId"]))
        for split in ("holdout", "val")
    }

    # Train: stratified toward what the silver labels cannot see.
    train = sorted(queues["train"], key=lambda row: order_key(row["pageId"]))
    buckets: dict[str, list[dict[str, Any]]] = {"schemaless_rare_slug": [], "schemaless": [], "rare_slug": [], "labelled": []}
    for row in train:
        schemaless = (row["schemaSource"] or "none") == "none"
        rare = slug_family(row["url"]) is not None
        key = (
            "schemaless_rare_slug" if schemaless and rare
            else "schemaless" if schemaless
            else "rare_slug" if rare
            else "labelled"
        )
        buckets[key].append(row)
    # Quotas: the two schema-less buckets carry the sample, and a quarter stays
    # on schema-labelled pages so Jev-versus-schema agreement is measurable on
    # train as well as on the evaluation splits.
    quotas = {
        "schemaless_rare_slug": int(args.train_sample * 0.25),
        "schemaless": int(args.train_sample * 0.40),
        "rare_slug": int(args.train_sample * 0.10),
        "labelled": int(args.train_sample * 0.25),
    }
    per_domain: Counter[str] = Counter()
    chosen: list[dict[str, Any]] = []
    for name, quota in quotas.items():
        taken = 0
        for row in buckets[name]:
            if taken >= quota:
                break
            domain = index[row["pageId"]]["domain"]
            if per_domain[domain] >= args.max_per_domain:
                continue
            per_domain[domain] += 1
            chosen.append(row)
            taken += 1
    # Redistribute any shortfall in document order across the remaining pages.
    if len(chosen) < args.train_sample:
        seen = {row["pageId"] for row in chosen}
        for row in train:
            if len(chosen) >= args.train_sample:
                break
            if row["pageId"] in seen:
                continue
            domain = index[row["pageId"]]["domain"]
            if per_domain[domain] >= args.max_per_domain:
                continue
            per_domain[domain] += 1
            chosen.append(row)
    selected["train"] = sorted(chosen, key=lambda row: order_key(row["pageId"]))

    args.output_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    summary: dict[str, Any] = {"maxPerDomain": args.max_per_domain, "splits": {}}
    for split, rows in selected.items():
        path = args.output_dir / f"queue-{split}.jsonl"
        # The queue carries the URL because the labeller needs the path; the
        # emitted label rows do not.
        path.write_text(
            "".join(
                json.dumps(
                    {k: row[k] for k in ("pageId", "contentHash", "url", "split")}, sort_keys=True
                )
                + "\n"
                for row in rows
            )
        )
        path.chmod(0o600)
        meta = args.output_dir / f"queue-{split}.meta.jsonl"
        meta.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        meta.chmod(0o600)
        summary["splits"][split] = {
            "pages": len(rows),
            "schemaSource": dict(sorted(Counter(row["schemaSource"] or "none" for row in rows).items())),
            "schemaLabel": dict(sorted(Counter(row["schemaLabel"] for row in rows).items())),
            "divSoup": sum(1 for row in rows if row["divSoup"]),
            "domains": len({index[row["pageId"]]["domain"] for row in rows}),
        }
    (args.output_dir / "queue-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
