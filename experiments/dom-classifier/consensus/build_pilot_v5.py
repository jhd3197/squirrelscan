#!/usr/bin/env python3
"""Draw the deterministic, stratified G1 pilot sample for prompt revision v5.

The pool is every ``diverse-500-v1`` node packet outside the burned 75-page test
split that carries **both** v4 annotators, so a v4 -> v5 comparison is possible
on the same node without labelling anything new for v4. Strata oversample the
boundaries the v4 error analysis measured as broken: ``<footer>`` and ``<main>``
container-vs-section, ``<header>``, non-semantic ``<div>``, and media elements
that reached v4 as the literal ``[empty]`` token.

Selection is a pure function of the frozen inputs: a stratum is ordered by
sha256 of its identity, never by ``random``, so re-running reproduces the sample
byte for byte. Nothing here writes to the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

DATA_ROOT = Path(os.environ["DOM_CLASSIFIER_DATA_ROOT"])
CORPUS = DATA_ROOT / "2026-09-19" / "diverse-500-v1"
TRAINING = CORPUS / "training-data-v1"

# Quotas sum to 200. The first six strata are the measured failure boundaries;
# `other` keeps a small control group of tags the tag rule already answers, so a
# v5 regression on the easy cases cannot hide behind the hard ones.
QUOTAS: dict[str, int] = {
    "footer": 40,
    "main": 35,
    "header": 35,
    "div": 35,
    "empty_media": 25,
    "figure": 10,
    "iframe": 8,
    "other": 12,
}
# Where a short stratum's unfilled quota goes, in order.
OVERFLOW = ("div", "footer", "main", "header", "other", "empty_media")
MEDIA_TAGS = frozenset({"img", "picture", "video", "audio", "svg", "input", "embed", "object"})


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def packet_field(packet_input: str, name: str) -> str:
    """Read one ``key=value`` line out of a serialized packet input."""
    prefix = f"{name}="
    for line in packet_input.split("\n"):
        if line.startswith(prefix):
            return line[len(prefix) :]
    return ""


def stratum_of(row: dict[str, Any]) -> str:
    tag = packet_field(row["input"], "tag").strip().lower()
    body = packet_field(row["input"], "text").strip()
    if tag in MEDIA_TAGS and body in {"", "[empty]"}:
        return "empty_media"
    if tag in {"footer", "main", "header", "div", "figure", "iframe"}:
        return tag
    return "other"


def order_key(row: dict[str, Any]) -> tuple[str, str]:
    """A stable, content-derived order; ties break on identity, never position."""
    identity = f"{row['split']}\0{row['pageId']}\0{row.get('nodeId') or ''}"
    return (hashlib.sha256(identity.encode()).hexdigest(), identity)


def build_pool(exclude_split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jev_keys = {
        (row["split"], row["pageId"], row.get("nodeId"))
        for name in ("train-jev.jsonl", "validation-jev.jsonl", "test-jev.jsonl")
        for row in read_jsonl(TRAINING / name)
    }
    pool = [
        row
        for row in read_jsonl(CORPUS / "luna-complete-800.jsonl")
        if row["split"] != exclude_split
        and (row["split"], row["pageId"], row.get("nodeId")) in jev_keys
    ]
    nodes = [row for row in pool if row["recordType"] == "node"]
    pages = [row for row in pool if row["recordType"] == "page"]
    if not nodes or not pages:
        raise ValueError("pool is empty; check the corpus paths and the excluded split")
    return nodes, pages


def allocate(strata: dict[str, list[dict[str, Any]]], target: int) -> list[dict[str, Any]]:
    taken: dict[str, list[dict[str, Any]]] = {}
    shortfall = 0
    for name, quota in QUOTAS.items():
        available = strata.get(name, [])
        taken[name] = available[:quota]
        shortfall += quota - len(taken[name])
    for name in OVERFLOW:
        if shortfall <= 0:
            break
        available = strata.get(name, [])
        extra = available[len(taken.get(name, [])) : len(taken.get(name, [])) + shortfall]
        taken[name] = taken.get(name, []) + extra
        shortfall -= len(extra)
    selected = [row for name in QUOTAS for row in taken.get(name, [])]
    if len(selected) != target - shortfall:
        raise ValueError("allocation lost rows")
    return sorted(selected, key=order_key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-nodes", type=int, default=200)
    parser.add_argument("--exclude-split", default="test")
    args = parser.parse_args()
    if args.target_nodes != sum(QUOTAS.values()):
        raise ValueError("--target-nodes must equal the quota total")

    nodes, pages = build_pool(args.exclude_split)
    strata: dict[str, list[dict[str, Any]]] = {}
    for row in sorted(nodes, key=order_key):
        strata.setdefault(stratum_of(row), []).append(row)
    selected = allocate(strata, args.target_nodes)
    selected_pages = {row["pageId"] for row in selected}
    # Every page packet in the pool is kept: the pageType axis is measured per
    # page, and dropping pages that happen to contribute no sampled node would
    # bias the page sample toward pages with awkward nodes.
    page_rows = sorted(pages, key=order_key)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in selected:
            handle.write(
                json.dumps(
                    {
                        "recordType": "node",
                        "split": row["split"],
                        "pageId": row["pageId"],
                        "nodeId": row["nodeId"],
                        "captureHash": row["captureHash"],
                        "stratum": stratum_of(row),
                        "v4InputSha256": row["inputSha256"],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        for row in page_rows:
            handle.write(
                json.dumps(
                    {
                        "recordType": "page",
                        "split": row["split"],
                        "pageId": row["pageId"],
                        "nodeId": None,
                        "captureHash": row["captureHash"],
                        "stratum": "page",
                        "v4InputSha256": row["inputSha256"],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    print(
        json.dumps(
            {
                "poolNodes": len(nodes),
                "poolPages": len(pages),
                "selectedNodes": len(selected),
                "selectedPages": len(page_rows),
                "pagesContributingNodes": len(selected_pages),
                "strata": dict(sorted(Counter(stratum_of(row) for row in selected).items())),
                "selectionSha256": "sha256:"
                + hashlib.sha256(args.output.read_bytes()).hexdigest(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
