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


def positive_claims(row: dict[str, Any]) -> dict[str, Any]:
    """Positive multi-label claims, whether or not the axis was declared complete.

    An incomplete observation is a positive claim whose negatives the annotator
    would not decide. Dropping it loses the claim as well as the abstention,
    which is why the exact-set view alone understates how much the two agree.
    """
    out: dict[str, Any] = {}
    for axis, value in row.get("observedAxes", {}).items():
        if axis in {"componentType", "pageType"}:
            continue
        labels = value.get("labels")
        if isinstance(labels, list):
            out[axis] = {label for label in labels if label != "unknown"}
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


def multi_label_overlap(
    left: dict[tuple[str, str, str], dict[str, Any]],
    right: dict[tuple[str, str, str], dict[str, Any]],
    axis: str,
) -> dict[str, Any]:
    """Micro precision, recall and F1 over positive labels on a multi-label axis.

    Exact-set identity is the headline because it is what the v4 error analysis
    used, but it scores only the rows where **both** annotators declared the axis
    complete. Reviewer B declares `regions` complete on 87 of 200 nodes and
    `purposes` on 22, so exact-set agreement there is computed on a subset the
    annotator chose, which is not a random one. Micro F1 over positive claims
    uses every row where both said something, and it degrades gracefully instead
    of dropping the row.
    """
    matched = predicted = actual = rows = 0
    for key, left_row in left.items():
        right_row = right.get(key)
        if right_row is None or axis not in left_row or axis not in right_row:
            continue
        a, b = left_row[axis], right_row[axis]
        if not isinstance(a, set) or not isinstance(b, set):
            continue
        rows += 1
        matched += len(a & b)
        predicted += len(a)
        actual += len(b)
    precision = matched / predicted if predicted else None
    recall = matched / actual if actual else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall and (precision + recall)
        else None
    )
    return {
        "rows": rows,
        "microPrecision": round(precision, 4) if precision is not None else None,
        "microRecall": round(recall, 4) if recall is not None else None,
        "microF1": round(f1, 4) if f1 is not None else None,
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
    jev_v6_rows = read_jsonl(args.pilot / "jev-v6.jsonl")
    jev_v6 = {identity(row): v5_sets(row) for row in jev_v6_rows}
    luna_v5_rows = read_jsonl(args.pilot / "luna-v5.jsonl")
    luna_v5 = {identity(row): v5_sets(row) for row in luna_v5_rows}
    # Reviewer B answers the same records under both revisions. Its two legs are
    # the only pair that differs in nothing but the prompt, so they carry the gate.
    reviewer_b_v4_rows = read_jsonl(args.pilot / "reviewer-b-v4.jsonl")
    reviewer_b_v5_rows = read_jsonl(args.pilot / "reviewer-b-v5.jsonl")
    reviewer_b_v6_rows = read_jsonl(args.pilot / "reviewer-b-v6.jsonl")
    for rows, revision in (
        (reviewer_b_v4_rows, "v4"),
        (reviewer_b_v5_rows, "v5"),
        (reviewer_b_v6_rows, "v6"),
    ):
        models = {row.get("reviewerModel") for row in rows}
        revisions = {row.get("promptRevision") for row in rows}
        if rows and (len(models) != 1 or revisions != {f"dom-suggestions-{revision}"}):
            raise ValueError(f"reviewer B {revision} file mixes reviewers or revisions")
    reviewer_b_v4 = {identity(row): luna_v4_sets(row) for row in reviewer_b_v4_rows}
    reviewer_b_v5 = {identity(row): v5_sets(row) for row in reviewer_b_v5_rows}
    reviewer_b_v6 = {identity(row): v5_sets(row) for row in reviewer_b_v6_rows}

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
            "jevV6": len(jev_v6),
            "reviewerBv4": len(reviewer_b_v4),
            "reviewerBv5": len(reviewer_b_v5),
            "reviewerBv6": len(reviewer_b_v6),
        },
        "reviewers": {
            "A": {"model": "gpt-5.6-luna", "revisions": ["v4"] + (["v5"] if luna_v5 else [])},
            "B": {
                "model": next(
                    (row.get("reviewerModel") for row in reviewer_b_v5_rows + reviewer_b_v4_rows),
                    None,
                ),
                "revisions": [
                    name
                    for name, rows in (
                        ("v4", reviewer_b_v4),
                        ("v5", reviewer_b_v5),
                        ("v6", reviewer_b_v6),
                    )
                    if rows
                ],
            },
        },
    }

    reviewer_b_v4_p = project(reviewer_b_v4, "pageTypes")
    comparisons = {
        "v4_reviewerA": (jev_v4_p, luna_v4_p),
        "proxy_jevV5_vs_reviewerAv4": (as_set(jev_v5), luna_v4_p),
        "shift_jevV4_vs_jevV5": (jev_v4_p, as_set(jev_v5)),
    }
    if reviewer_b_v4:
        comparisons["v4_reviewerB"] = (jev_v4_p, reviewer_b_v4_p)
    if reviewer_b_v5:
        # The gate: same reviewer, same records, only the prompt revision differs.
        comparisons["v5_reviewerB"] = (as_set(jev_v5), as_set(reviewer_b_v5))
    if reviewer_b_v6:
        comparisons["v6_reviewerB"] = (as_set(jev_v6), as_set(reviewer_b_v6))
    if reviewer_b_v5 and reviewer_b_v6:
        comparisons["shift_reviewerB_v5_vs_v6"] = (as_set(reviewer_b_v5), as_set(reviewer_b_v6))
    if jev_v5 and jev_v6:
        comparisons["shift_jevV5_vs_jevV6"] = (as_set(jev_v5), as_set(jev_v6))
    if reviewer_b_v4 and reviewer_b_v5:
        comparisons["shift_reviewerB_v4_vs_v5"] = (reviewer_b_v4_p, as_set(reviewer_b_v5))
        comparisons["crossReviewer_v4"] = (luna_v4_p, reviewer_b_v4_p)
    if luna_v5:
        comparisons["v5_reviewerA"] = (as_set(jev_v5), as_set(luna_v5))
        comparisons["shift_reviewerA_v4_vs_v5"] = (luna_v4_p, as_set(luna_v5))
    report["agreement"] = {
        name: {axis: agreement(left, right, axis) for axis in ("componentType", "regions", "purposes", "pageType")}
        for name, (left, right) in comparisons.items()
    }
    # Jev answers every Noul, so its thresholded set is already a complete claim;
    # only the reviewers can leave an axis incomplete.
    claims = {
        "jevV4": jev_v4_p,
        "jevV5": jev_v5,
        "jevV6": jev_v6,
        "lunaV4": {
            identity(row): positive_claims(row)
            for row in read_jsonl(CORPUS / "luna-complete-800.jsonl")
            if identity(row) in wanted
        },
        "reviewerBv4": {identity(row): positive_claims(row) for row in reviewer_b_v4_rows},
        "reviewerBv5": {identity(row): positive_claims(row) for row in reviewer_b_v5_rows},
        "reviewerBv6": {identity(row): positive_claims(row) for row in reviewer_b_v6_rows},
    }
    claim_pairs = {
        "v4_reviewerB": ("jevV4", "reviewerBv4"),
        "v5_reviewerB": ("jevV5", "reviewerBv5"),
        "v6_reviewerB": ("jevV6", "reviewerBv6"),
        "v4_reviewerA": ("jevV4", "lunaV4"),
    }
    report["multiLabelOverlap"] = {
        "note": "Micro precision/recall/F1 over positive labels, including observations the annotator marked incomplete. Use alongside the exact-set rates, which drop those rows.",
        **{
            name: {
                axis: multi_label_overlap(claims[left], claims[right], axis)
                for axis in ("regions", "purposes")
            }
            for name, (left, right) in claim_pairs.items()
            if claims[left] and claims[right]
        },
    }
    # Raw-vocabulary v4 purposes, so the reduction's contribution is visible.
    report["agreement"]["v4_reviewerA"]["purposesRawVocabulary"] = agreement(
        jev_v4, luna_v4, "purposes"
    )
    # The gate, stated once so no reader has to assemble it from the table. The
    # newest revision with a reviewer leg is the one under test; v4 is always the
    # baseline, because that is the revision the 57.6% headline was measured on.
    latest = "v6" if reviewer_b_v6 else "v5" if reviewer_b_v5 else None
    if reviewer_b_v4 and latest:
        before = report["agreement"]["v4_reviewerB"]["componentType"]
        after = report["agreement"][f"{latest}_reviewerB"]["componentType"]
        report["g1"] = {
            "revision": latest,
            "definition": f"within-reviewer componentType agreement, reviewer B, v4 -> {latest}",
            "threshold": 0.75,
            "v4": before,
            "v5": after,
            "delta": round((after["rate"] or 0) - (before["rate"] or 0), 4),
            "passes": bool(after["rate"] is not None and after["rate"] >= 0.75),
            "caveat": "Synthetic agreement between two models. Reviewer B is Claude Sonnet; reviewer A (gpt-5.6-luna) could not be re-run, so the v5 leg is not confirmed by a second reviewer.",
        }

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
    if reviewer_b_v4:
        legs["reviewerBv4"] = reviewer_b_v4_p
    if reviewer_b_v5:
        legs["reviewerBv5"] = reviewer_b_v5
    if reviewer_b_v6:
        legs["reviewerBv6"] = reviewer_b_v6
    if jev_v6:
        legs["jevV6"] = jev_v6
    report["abstention"] = {
        name: {
            "componentType": abstention(rows, "componentType", "unknown", node_keys),
            "regions": abstention(rows, "regions", "unknown", node_keys),
            "purposes": abstention(rows, "purposes", "unknown", node_keys),
            "pageType": abstention(rows, "pageType", "other", page_keys),
        }
        for name, rows in legs.items()
    }

    # Cost is reported for the newest Jev leg present.
    jev_cost_rows = jev_v6_rows or jev_v5_rows
    node_rows = [row for row in jev_cost_rows if row["recordType"] == "node"]
    input_tokens = sum(int((row.get("usage") or {}).get("input_tokens") or 0) for row in jev_cost_rows)
    latencies = sorted(int(row["latencyMs"]) for row in jev_cost_rows if row.get("latencyMs"))
    requests = len(latencies)
    judgments = len(jev_cost_rows)
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

    for name, rows in (
        ("reviewerBv4", reviewer_b_v4_rows),
        ("reviewerBv5", reviewer_b_v5_rows),
        ("reviewerBv6", reviewer_b_v6_rows),
    ):
        if not rows:
            continue
        # One cost and one latency per batch, not per row: a 20-packet batch is
        # one call, and counting it twenty times would understate the unit cost
        # by the batch size.
        by_batch = {
            row["reviewerRun"]["batch"]: (
                row["reviewerRun"].get("costUsd", 0.0),
                row["reviewerRun"].get("latencyMs", 0),
            )
            for row in rows
            if row.get("reviewerRun")
        }
        cost = sum(value[0] for value in by_batch.values())
        latency = sum(value[1] for value in by_batch.values())
        nodes = sum(1 for row in rows if row["recordType"] == "node")
        report["cost"][name] = {
            "pricing": "Claude CLI, cost as reported by the CLI envelope (total_cost_usd)",
            "model": rows[0].get("reviewerModel"),
            "calls": len(by_batch),
            "rows": len(rows),
            "nodes": nodes,
            "usd": round(cost, 4),
            "usdPer1000Nodes": round(cost / max(1, nodes) * 1000, 2),
            "secondsPer1000Nodes": round(latency / 1000 / max(1, nodes) * 1000, 1),
            "medianCallMs": sorted(v[1] for v in by_batch.values())[len(by_batch) // 2]
            if by_batch
            else None,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output.chmod(0o600)
    print(json.dumps(report.get("g1", {"g1": "not available"}), indent=2, sort_keys=True))
    print(json.dumps(report["agreement"], indent=2, sort_keys=True))
    print(json.dumps(report["cost"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
