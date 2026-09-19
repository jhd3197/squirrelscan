#!/usr/bin/env python3
"""Write a human adjudication cohort from Jev-versus-reviewer disagreements.

Same shape as ``human-review-cohort-v1``: a ``cohort.json`` describing what was
selected and how, a ``queue.jsonl`` with one row per disagreement, and the Jev
sidecar rows. A human works the queue with **1** for the first annotator, **2**
for the second and **X** for neither.

Two rules this script enforces rather than documents:

* **It never reads a model prediction file.** When the source directory is a
  blind evaluation set, any predictions in it belong to whoever is being
  evaluated, and a labelling cohort that had seen them would not be an
  independent label. The file names it refused are recorded in ``cohort.json``,
  so the refusal is auditable rather than assumed.
* **It carries no page text beyond the packet the annotators already saw.** The
  packet is the privacy-scrubbed evidence; the HTML reference is carried as an
  identifier, never as content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_ROOT = Path(os.environ["DOM_CLASSIFIER_DATA_ROOT"])
# Anything that looks like a prediction from the model under evaluation. The
# match is on the name, and every skipped name is reported.
PREDICTION_NAME_HINTS = ("predict", "inference", "eval-out", "scores", "logits", "checkpoint")
AXES = ("componentType", "regions", "purposes", "pageType")
# Boundaries worth a human's time, in the order a reviewer should meet them.
BOUNDARIES = {
    frozenset({"layout_container", "content_section"}): "container_vs_section",
    frozenset({"layout_container", "navigation_menu"}): "chrome_vs_menu",
    frozenset({"card", "list"}): "item_vs_wrapper",
    frozenset({"image", "media"}): "still_vs_timebased",
    frozenset({"button", "link"}): "control_vs_link",
    frozenset({"article", "content_section"}): "body_vs_section",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def identity(row: dict[str, Any]) -> tuple[str, str]:
    return (row["pageId"], row.get("nodeId") or "")


def single(value: Any) -> Any:
    return value[0] if isinstance(value, list) and len(value) == 1 else value


def observed(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for axis, value in row.get("observedAxes", {}).items():
        if not value.get("complete"):
            continue
        out[axis] = single(value.get("labels"))
    return out


def boundary_of(left: Any, right: Any) -> str:
    if isinstance(left, str) and isinstance(right, str):
        return BOUNDARIES.get(frozenset({left, right}), "other")
    return "multi_label"


def refused_prediction_files(source: Path) -> list[str]:
    if not source.exists():
        return []
    return sorted(
        path.name
        for path in source.rglob("*")
        if path.is_file() and any(hint in path.name.lower() for hint in PREDICTION_NAME_HINTS)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jev", type=Path, required=True)
    parser.add_argument("--reviewer", type=Path, required=True)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="the corpus directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort-id", required=True)
    parser.add_argument("--purpose", default="Adjudicate Jev-versus-reviewer component disagreements.")
    args = parser.parse_args()

    for path in (args.jev, args.reviewer, args.packets, args.output):
        if not str(path.resolve()).startswith(f"{DATA_ROOT.resolve()}/"):
            raise ValueError(f"{path} is outside the private data root")

    jev_rows = read_jsonl(args.jev)
    reviewer_rows = read_jsonl(args.reviewer)
    packets = {identity(row): row for row in read_jsonl(args.packets)}
    if not jev_rows or not reviewer_rows:
        raise ValueError("both annotator files must be present")

    revisions = {row.get("promptRevision") for row in jev_rows} | {
        row.get("promptRevision") for row in reviewer_rows
    }
    if len(revisions) != 1:
        raise ValueError(f"annotators disagree on prompt revision: {sorted(revisions)}")
    revision = revisions.pop()

    jev = {identity(row): observed(row) for row in jev_rows}
    reviewer = {identity(row): observed(row) for row in reviewer_rows}
    reviewer_model = next((row.get("reviewerModel") for row in reviewer_rows), None)

    queue: list[dict[str, Any]] = []
    for key, left in jev.items():
        right = reviewer.get(key)
        packet = packets.get(key)
        if right is None or packet is None:
            continue
        disagreements = [axis for axis in AXES if axis in left and axis in right and left[axis] != right[axis]]
        if not disagreements:
            continue
        primary = "componentType" if "componentType" in disagreements else disagreements[0]
        queue.append(
            {
                "pageId": key[0],
                "nodeId": key[1] or None,
                "recordType": packet.get("recordType"),
                "captureHash": packet.get("captureHash"),
                "sourceRef": packet.get("sourceRef") or packet.get("htmlRef"),
                "stratum": packet.get("stratum"),
                "disagreementAxes": disagreements,
                "category": boundary_of(left.get(primary), right.get(primary)),
                "boundary": "_vs_".join(sorted(str(v) for v in (left.get(primary), right.get(primary)))),
                "guidanceRef": f"v2/labels/policy-{revision.rsplit('-', 1)[-1]}.md#1-componenttype",
                # The packet, verbatim: the only evidence either annotator saw.
                "packet": packet.get("input"),
                "inputSha256": packet.get("inputSha256"),
                "choices": {"1": "jev", "2": "reviewer", "X": "neither"},
                "jev": {axis: left.get(axis) for axis in AXES if axis in left},
                "reviewer": {axis: right.get(axis) for axis in AXES if axis in right},
            }
        )
    queue.sort(key=lambda row: (row["category"], row["pageId"], row["nodeId"] or ""))

    refused = refused_prediction_files(args.source)
    args.output.mkdir(parents=True, mode=0o700, exist_ok=True)
    queue_path = args.output / "queue.jsonl"
    queue_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in queue))
    queue_path.chmod(0o600)
    suggestions_path = args.output / "model-suggestions.jsonl"
    suggestions_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in jev_rows))
    suggestions_path.chmod(0o600)

    cohort = {
        "cohortId": args.cohort_id,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "format": "queue.jsonl; adjudicate each row with 1 (jev), 2 (reviewer) or X (neither)",
        "purpose": args.purpose,
        "promptRevision": revision,
        "policySha256": next((row.get("policySha256") for row in jev_rows), None),
        "taxonomyRevision": next((row.get("taxonomyRevision") for row in jev_rows), None),
        "annotators": {
            "jev": {"role": "teacher suggestion", "model": next((row.get("modelId") for row in jev_rows), None)},
            "reviewer": {"role": "independent blind review", "model": reviewer_model},
        },
        "labelDefinitions": f"v2/labels/policy-{revision.rsplit('-', 1)[-1]}.md",
        "selection": {
            "rule": "every record where the two annotators disagree on at least one axis",
            "jevRecords": len(jev),
            "reviewerRecords": len(reviewer),
            "comparableRecords": sum(1 for key in jev if key in reviewer),
            "disagreements": len(queue),
            "byCategory": dict(sorted(Counter(row["category"] for row in queue).items())),
            "byAxis": dict(
                sorted(Counter(axis for row in queue for axis in row["disagreementAxes"]).items())
            ),
        },
        "blindEvaluationSafety": {
            "modelPredictionsRead": False,
            "refusedFiles": refused,
            "note": "This cohort is built from annotator labels only. No prediction, checkpoint or score file in the source directory was opened, so the labels stay independent of the model being evaluated.",
        },
        "sourceHashes": {
            args.jev.name: sha256_file(args.jev),
            args.reviewer.name: sha256_file(args.reviewer),
            args.packets.name: sha256_file(args.packets),
        },
    }
    cohort_path = args.output / "cohort.json"
    cohort_path.write_text(json.dumps(cohort, indent=2, sort_keys=True) + "\n")
    cohort_path.chmod(0o600)
    print(json.dumps({"queue": len(queue), "output": str(args.output), **cohort["selection"]}, sort_keys=True))


if __name__ == "__main__":
    main()
