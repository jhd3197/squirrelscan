#!/usr/bin/env python3
"""Score the G1 label-quality pilot: prompt revision v4 against v5, same nodes.

Every number here is **agreement between two synthetic annotators**, never
measured human accuracy. Both are models. Their agreement bounds what any
student trained on these labels can score; it does not say either is right.

Four comparisons are reported, and they are not interchangeable:

``v4``
    Jev v4 against Luna v4. The baseline, recomputed on the pilot nodes rather
    than quoted from the test split.
``v5``
    Jev v5 against Luna v5, both under ``policy-v5.md``. This is the G1 number.
    Absent when the reviewer leg has not run.
``proxy``
    Jev v5 against Luna **v4**. One-sided: the policy codifies the ``<footer>``
    convention Luna v4 already used, so movement here mostly measures whether
    Jev adopted Luna's prior, not whether the two now agree. Never quote it as
    G1.
``shift``
    Jev v4 against Jev v5 on the same node: how often the teacher changed its
    own answer, and to what.

Multi-label axes are scored as exact set identity, the same rule the v4 error
analysis used, so the baseline column is comparable to the published 57.6%.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

DATA_ROOT = Path(os.environ["DOM_CLASSIFIER_DATA_ROOT"])
CORPUS = DATA_ROOT / "2026-09-19" / "diverse-500-v1"
LABELS_DIR = Path(__file__).resolve().parents[1] / "v2" / "labels"
NOUL_THRESHOLD = 0.75
# TypeSafe charges input tokens only; output is free. docs.typesafe.ai/models.md
USD_PER_INPUT_TOKEN = 42.0 / 1_000_000_000


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def identity(row: dict[str, Any]) -> tuple[str, str, str]:
    return (row["split"], row["pageId"], row.get("nodeId") or "")


def jev_v4_sets(row: dict[str, Any]) -> dict[str, Any]:
    targets = row["softTargets"]
    out: dict[str, Any] = {}
    if "componentType" in targets:
        out["componentType"] = targets["componentType"].get("choice")
    for axis in ("regions", "purposes", "pageTypes", "contentKinds"):
        if axis in targets:
            out[axis] = {
                item["label"]
                for item in targets[axis]
                if item.get("yesProbability", 0) >= NOUL_THRESHOLD
            }
    return out


def luna_v4_sets(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for axis, value in row.get("observedAxes", {}).items():
        if not value.get("complete"):
            continue
        labels = value.get("labels")
        if axis == "componentType":
            out[axis] = labels[0] if isinstance(labels, list) and labels else labels
        else:
            out[axis] = set(labels or [])
    return out


def v5_sets(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for axis, value in row.get("observedAxes", {}).items():
        if not value.get("complete"):
            continue
        labels = value.get("labels")
        if axis in {"componentType", "pageType"}:
            out[axis] = labels[0] if isinstance(labels, list) and labels else labels
        else:
            out[axis] = set(labels or [])
    return out


def agreement(
    left: dict[tuple[str, str, str], dict[str, Any]],
    right: dict[tuple[str, str, str], dict[str, Any]],
    axis: str,
) -> dict[str, Any]:
    matched = both = 0
    confusions: Counter[tuple[str, str]] = Counter()
    for key, left_row in left.items():
        right_row = right.get(key)
        if right_row is None or axis not in left_row or axis not in right_row:
            continue
        both += 1
        if left_row[axis] == right_row[axis]:
            matched += 1
        elif isinstance(left_row[axis], str) or left_row[axis] is None:
            confusions[(str(left_row[axis]), str(right_row[axis]))] += 1
        else:
            confusions[
                (",".join(sorted(left_row[axis])) or "(none)", ",".join(sorted(right_row[axis])) or "(none)")
            ] += 1
    return {
        "n": both,
        "agreed": matched,
        "rate": round(matched / both, 4) if both else None,
        "topConfusions": [
            {"left": a, "right": b, "count": c} for (a, b), c in confusions.most_common(8)
        ],
    }


def project_purposes(value: Any, retained: set[str]) -> Any:
    return {label for label in value if label in retained} if isinstance(value, set) else value


def project_page_types(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, set):
        return {mapping.get(label, "other") for label in value}
    return {mapping.get(value, "other")} if isinstance(value, str) else value


def abstention(
    rows: dict[tuple[str, str, str], dict[str, Any]],
    axis: str,
    token: str,
    eligible: set[tuple[str, str, str]],
) -> dict[str, Any]:
    """Rates over the records the axis *applies* to, never over every row.

    A node axis measured against a denominator that includes page rows reads as
    a 38% omission rate that is really "pages have no componentType". `eligible`
    is the record set the axis is defined for.
    """
    scoped = {key: row for key, row in rows.items() if key in eligible}
    present = [row for row in scoped.values() if axis in row]
    if not scoped:
        return {"n": 0, "eligible": 0, "abstainToken": token, "abstainRate": None, "emptySet": None, "omitted": None}
    unknown = sum(1 for row in present if row[axis] == token or row[axis] == {token})
    empty = sum(1 for row in present if isinstance(row[axis], set) and not row[axis])
    return {
        "n": len(present),
        "eligible": len(scoped),
        "abstainToken": token,
        "abstainRate": round(unknown / len(present), 4) if present else None,
        "emptySet": round(empty / len(present), 4) if present else None,
        "omitted": round(1 - len(present) / len(scoped), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True, help="label-pilot-v5 directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    purposes_doc = json.loads((LABELS_DIR / "purposes-v5.json").read_text())
    page_map = json.loads((LABELS_DIR / "page-type-map-v5.json").read_text())["map"]
    retained = set(purposes_doc["retained"])

    selection = read_jsonl(args.pilot / "pilot-selection.jsonl")
    wanted = {(row["split"], row["pageId"], row.get("nodeId") or "") for row in selection}
    strata = {(row["split"], row["pageId"], row.get("nodeId") or ""): row["stratum"] for row in selection}

    jev_v4 = {
        identity(row): jev_v4_sets(row)
        for name in ("train-jev.jsonl", "validation-jev.jsonl")
        for row in read_jsonl(CORPUS / "training-data-v1" / name)
        if identity(row) in wanted
    }
    luna_v4 = {
        identity(row): luna_v4_sets(row)
        for row in read_jsonl(CORPUS / "luna-complete-800.jsonl")
        if identity(row) in wanted
    }
    jev_v5_rows = read_jsonl(args.pilot / "jev-v5.jsonl")
    jev_v5 = {identity(row): v5_sets(row) for row in jev_v5_rows}
    luna_v5_rows = read_jsonl(args.pilot / "luna-v5.jsonl")
    luna_v5 = {identity(row): v5_sets(row) for row in luna_v5_rows}

    # v4 page-type sets and purpose sets projected into the v5 vocabulary, so the
    # effect of the definitions is separable from the effect of the reduction.
    def project(rows: dict[Any, dict[str, Any]], page_axis: str) -> dict[Any, dict[str, Any]]:
        out = {}
        for key, row in rows.items():
            projected = dict(row)
            if "purposes" in row:
                projected["purposes"] = project_purposes(row["purposes"], retained)
            if page_axis in row:
                projected["pageType"] = project_page_types(row[page_axis], page_map)
            out[key] = projected
        return out

    jev_v4_p = project(jev_v4, "pageTypes")
    luna_v4_p = project(luna_v4, "pageTypes")
    # A v5 single choice compared against a projected v4 set is scored as a set of
    # one, which is strict; membership is reported alongside it.
    def as_set(rows: dict[Any, dict[str, Any]]) -> dict[Any, dict[str, Any]]:
        out = {}
        for key, row in rows.items():
            projected = dict(row)
            if isinstance(row.get("pageType"), str):
                projected["pageType"] = {row["pageType"]}
            out[key] = projected
        return out

    report: dict[str, Any] = {
        "disclaimer": "Synthetic-annotator agreement, not measured human accuracy. Both annotators are models.",
        "pilot": {
            "nodes": sum(1 for row in selection if row["recordType"] == "node"),
            "pages": sum(1 for row in selection if row["recordType"] == "page"),
            "strata": dict(sorted(Counter(row["stratum"] for row in selection).items())),
        },
        "legs": {
            "jevV4": len(jev_v4),
            "lunaV4": len(luna_v4),
            "jevV5": len(jev_v5),
            "lunaV5": len(luna_v5),
        },
    }

    comparisons = {
        "v4": (jev_v4_p, luna_v4_p),
        "proxy_jevV5_vs_lunaV4": (as_set(jev_v5), luna_v4_p),
        "shift_jevV4_vs_jevV5": (jev_v4_p, as_set(jev_v5)),
    }
    if luna_v5:
        comparisons["v5"] = (as_set(jev_v5), as_set(luna_v5))
        comparisons["shift_lunaV4_vs_lunaV5"] = (luna_v4_p, as_set(luna_v5))
    report["agreement"] = {
        name: {axis: agreement(left, right, axis) for axis in ("componentType", "regions", "purposes", "pageType")}
        for name, (left, right) in comparisons.items()
    }
    # Raw-vocabulary v4 purposes, so the reduction's contribution is visible.
    report["agreement"]["v4"]["purposesRawVocabulary"] = agreement(jev_v4, luna_v4, "purposes")

    report["componentByStratum"] = {}
    for name, (left, right) in comparisons.items():
        per: dict[str, dict[str, int]] = {}
        for key, left_row in left.items():
            right_row = right.get(key)
            if not right_row or "componentType" not in left_row or "componentType" not in right_row:
                continue
            bucket = per.setdefault(strata.get(key, "unknown"), {"n": 0, "agreed": 0})
            bucket["n"] += 1
            bucket["agreed"] += int(left_row["componentType"] == right_row["componentType"])
        report["componentByStratum"][name] = {
            stratum: {**value, "rate": round(value["agreed"] / value["n"], 4)}
            for stratum, value in sorted(per.items())
            if value["n"]
        }

    node_keys = {
        (row["split"], row["pageId"], row.get("nodeId") or "")
        for row in selection
        if row["recordType"] == "node"
    }
    page_keys = {
        (row["split"], row["pageId"], "") for row in selection if row["recordType"] == "page"
    }
    legs = {"jevV4": jev_v4_p, "lunaV4": luna_v4_p, "jevV5": jev_v5}
    if luna_v5:
        legs["lunaV5"] = luna_v5
    report["abstention"] = {
        name: {
            "componentType": abstention(rows, "componentType", "unknown", node_keys),
            "regions": abstention(rows, "regions", "unknown", node_keys),
            "purposes": abstention(rows, "purposes", "unknown", node_keys),
            "pageType": abstention(rows, "pageType", "other", page_keys),
        }
        for name, rows in legs.items()
    }

    node_rows = [row for row in jev_v5_rows if row["recordType"] == "node"]
    input_tokens = sum(int((row.get("usage") or {}).get("input_tokens") or 0) for row in jev_v5_rows)
    latencies = sorted(int(row["latencyMs"]) for row in jev_v5_rows if row.get("latencyMs"))
    requests = len(latencies)
    judgments = len(jev_v5_rows)
    report["cost"] = {
        "jev": {
            "pricing": "$0.042 per million input tokens; output tokens free (docs.typesafe.ai/models.md)",
            "requests": requests,
            "rows": judgments,
            "nodes": len(node_rows),
            "inputTokens": input_tokens,
            "usd": round(input_tokens * USD_PER_INPUT_TOKEN, 4),
            "usdPer1000Nodes": round(
                input_tokens * USD_PER_INPUT_TOKEN / max(1, len(node_rows)) * 1000, 4
            ),
            "secondsPer1000Nodes": round(
                sum(latencies) / 1000 / max(1, len(node_rows)) * 1000, 1
            )
            if latencies
            else None,
            "medianRequestMs": latencies[len(latencies) // 2] if latencies else None,
            "p95RequestMs": latencies[int(len(latencies) * 0.95)] if latencies else None,
        }
    }
    if luna_v5_rows:
        luna_latency = sum(
            {row["lunaRun"]["batch"]: row["lunaRun"].get("latencyMs", 0) for row in luna_v5_rows}.values()
        )
        luna_nodes = sum(1 for row in luna_v5_rows if row["recordType"] == "node")
        report["cost"]["luna"] = {
            "pricing": "Codex CLI subscription; no per-token price is published, so cost is reported as wall-clock only",
            "rows": len(luna_v5_rows),
            "nodes": luna_nodes,
            "totalMs": luna_latency,
            "secondsPer1000Nodes": round(luna_latency / 1000 / max(1, luna_nodes) * 1000, 1),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["agreement"], indent=2, sort_keys=True))
    print(json.dumps(report["cost"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
