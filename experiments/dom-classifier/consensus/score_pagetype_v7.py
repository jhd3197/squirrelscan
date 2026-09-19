#!/usr/bin/env python3
"""Summarise the Jev page-type labelling run.

Three questions, and they are not the same question:

``distribution``
    What Jev called the pages, split by whether a schema-derived silver label
    exists at all. The classes with no schema.org source — `pricing`, `legal`,
    `careers` and the rest — can only ever appear in the schema-less half, which
    is the point of the run.
``agreementWithSchema``
    Where both exist, how often Jev matches the silver label. This **calibrates
    the silver labels**; it does not score Jev. Silver is derived from JSON-LD,
    og:type and microdata, which are publisher assertions, and a publisher
    calling a landing page an `Article` does not make it one.
``auditAgreement``
    Jev against the independent reviewer on the audited pages. Both are models,
    so this is a label-noise ceiling, not accuracy.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DATA_ROOT = Path(os.environ["DOM_CLASSIFIER_DATA_ROOT"])


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:  # type: ignore[operator]
        return [json.loads(line) for line in handle if line.strip()]


def rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report: dict[str, Any] = {
        "disclaimer": "Jev and the reviewer are both models. Agreement is a label-noise ceiling, never accuracy. Agreement with the silver label calibrates that label; it does not score either side.",
        "promptRevision": None,
        "splits": {},
    }
    reviewer = {row["pageId"]: row for row in read_jsonl(args.dir / "reviewer-b-audit.jsonl")}
    audit_agreed = audit_total = 0
    audit_confusions: Counter[tuple[str, str]] = Counter()
    cost = {"inputTokens": 0, "pages": 0}

    for split in ("holdout", "val", "train"):
        labels = read_jsonl(args.dir / f"{split}.jsonl")
        if not labels:
            continue
        meta = {row["pageId"]: row for row in read_jsonl(args.dir / f"queue-{split}.meta.jsonl")}
        report["promptRevision"] = labels[0].get("promptRevision")
        cost["inputTokens"] += sum(int((row.get("usage") or {}).get("input_tokens") or 0) for row in labels)
        cost["pages"] += len(labels)

        schema_labelled = [row for row in labels if (meta.get(row["pageId"], {}).get("schemaSource") or "none") != "none"]
        schema_less = [row for row in labels if (meta.get(row["pageId"], {}).get("schemaSource") or "none") == "none"]

        agreed = total = 0
        confusions: Counter[tuple[str, str]] = Counter()
        per_class: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "agreed": 0})
        for row in schema_labelled:
            silver = meta[row["pageId"]].get("schemaLabel")
            if not silver or silver == "unlabelled":
                continue
            total += 1
            per_class[silver]["n"] += 1
            if silver == row["pageType"]:
                agreed += 1
                per_class[silver]["agreed"] += 1
            else:
                confusions[(silver, row["pageType"])] += 1

        confident = [row for row in labels if (row.get("confidence") or 0) >= 0.75]
        report["splits"][split] = {
            "pages": len(labels),
            "schemaLabelled": len(schema_labelled),
            "schemaLess": len(schema_less),
            "jevDistributionAll": dict(sorted(Counter(row["pageType"] for row in labels).items())),
            "jevDistributionSchemaLess": dict(
                sorted(Counter(row["pageType"] for row in schema_less).items())
            ),
            "jevDistributionSchemaLabelled": dict(
                sorted(Counter(row["pageType"] for row in schema_labelled).items())
            ),
            "agreementWithSchema": {
                "n": total,
                "agreed": agreed,
                "rate": rate(agreed, total),
                "perSilverClass": {
                    name: {**value, "rate": rate(value["agreed"], value["n"])}
                    for name, value in sorted(per_class.items())
                },
                "topConfusions": [
                    {"silver": silver, "jev": jev, "count": count}
                    for (silver, jev), count in confusions.most_common(8)
                ],
            },
            "confidence": {
                "atLeast0.75": rate(len(confident), len(labels)),
                "median": sorted(row.get("confidence") or 0 for row in labels)[len(labels) // 2],
            },
        }

        for row in labels:
            other = reviewer.get(row["pageId"])
            if not other:
                continue
            audit_total += 1
            if other.get("pageType") == row["pageType"]:
                audit_agreed += 1
            else:
                audit_confusions[(row["pageType"], other.get("pageType") or "none")] += 1

    if audit_total:
        report["auditAgreement"] = {
            "n": audit_total,
            "agreed": audit_agreed,
            "rate": rate(audit_agreed, audit_total),
            "reviewerModel": next((row.get("reviewerModel") for row in reviewer.values()), None),
            "topConfusions": [
                {"jev": jev, "reviewer": other, "count": count}
                for (jev, other), count in audit_confusions.most_common(8)
            ],
        }
    report["cost"] = {
        **cost,
        "usd": round(cost["inputTokens"] * 42 / 1_000_000_000, 4),
        "inputTokensPerPage": round(cost["inputTokens"] / cost["pages"]) if cost["pages"] else None,
        "pricing": "$0.042 per million input tokens; output free",
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output.chmod(0o600)
    print(json.dumps({k: report[k] for k in ("promptRevision", "cost")}, sort_keys=True))
    for split, value in report["splits"].items():
        print(
            json.dumps(
                {
                    "split": split,
                    "pages": value["pages"],
                    "schemaLess": value["schemaLess"],
                    "agreementWithSchema": value["agreementWithSchema"]["rate"],
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
