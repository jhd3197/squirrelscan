#!/usr/bin/env python3
"""Materialize the **v4** packets for the pilot records, unchanged.

The within-reviewer G1 design needs one reviewer to answer the same 200 nodes
and 125 pages twice: once under v4 and once under v5. The v5 packets are built
by ``run-jev-v5.ts``; this script recovers the v4 side by copying the frozen
``runtime-dom-text-v1`` / ``dom-page-text-v1`` inputs that the v4 annotators
actually saw, out of ``luna-complete-800.jsonl``.

It copies. It never rebuilds, re-cleans or re-serializes a v4 input: if the v4
packet bytes changed, the v4 leg of the delta would not be the v4 leg.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

DATA_ROOT = Path(os.environ["DOM_CLASSIFIER_DATA_ROOT"])
CORPUS = DATA_ROOT / "2026-09-19" / "diverse-500-v1"
TAXONOMY_REVISION = "dom-taxonomy-v2"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = read_jsonl(args.selection)
    strata = {
        (row["split"], row["pageId"], row.get("nodeId") or ""): row["stratum"] for row in selection
    }
    wanted = set(strata)
    rows: list[dict[str, Any]] = []
    for row in read_jsonl(CORPUS / "luna-complete-800.jsonl"):
        key = (row["split"], row["pageId"], row.get("nodeId") or "")
        if key not in wanted:
            continue
        if row["inputSha256"] != sha256_text(row["input"]):
            raise ValueError(f"v4 packet {key} does not bind its own input")
        rows.append(
            {
                # packetId and taxonomyRevision are what the v4 prompt builder
                # requires; every other field is copied verbatim.
                "packetId": row["packetId"],
                "taxonomyRevision": TAXONOMY_REVISION,
                "packetRevision": "dom-suggestions-v4",
                "recordType": row["recordType"],
                "split": row["split"],
                "pageId": row["pageId"],
                "nodeId": row.get("nodeId"),
                "captureHash": row["captureHash"],
                "input": row["input"],
                "inputSha256": row["inputSha256"],
                "stratum": strata[key],
                "attributesAvailable": False,
            }
        )
    missing = wanted - {(r["split"], r["pageId"], r.get("nodeId") or "") for r in rows}
    if missing:
        raise ValueError(f"{len(missing)} pilot records have no v4 packet")
    rows.sort(key=lambda row: (row["recordType"], row["pageId"], row.get("nodeId") or ""))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    args.output.chmod(0o600)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "nodes": sum(1 for row in rows if row["recordType"] == "node"),
                "pages": sum(1 for row in rows if row["recordType"] == "page"),
                "packetsSha256": "sha256:"
                + hashlib.sha256(args.output.read_bytes()).hexdigest(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
