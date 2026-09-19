#!/usr/bin/env python3
"""Run the independent Luna reviewer under prompt revision ``dom-suggestions-v5``.

Differences from :mod:`luna_batch` (revision v4), each one measured rather than
guessed:

* The prompt embeds ``v2/labels/policy-v5.md`` **verbatim**, and every accepted
  row records ``policySha256``. The Jev adapter puts the same bytes in request
  state, so "both annotators were given the same definitions" is a checkable
  claim about a file hash, not a description of intent.
* ``pageType`` is one choice from eleven classes, replacing v4's multi-label
  battery over sixty-five.
* ``purposes`` offers the twelve values with support; the other fifteen are not
  in the schema and cannot be emitted.

The reviewer stays blind: a batch carries a derived ``packetKey``, the record
type and the packet text, never an id, hash, URL or the other annotator's view.
There is no heuristic fallback; a malformed or off-taxonomy batch is rejected
whole and no partial row is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL = "gpt-5.6-luna"
PROMPT_REVISION = "dom-suggestions-v5"
NODE_AXES = ("componentType", "regions", "purposes")
PAGE_AXES = ("pageType",)
LABELS_DIR = Path(__file__).resolve().parents[1] / "v2" / "labels"


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def load_vocabulary() -> dict[str, list[str]]:
    taxonomy = json.loads((LABELS_DIR.parents[1] / "taxonomy.json").read_text())
    page_types = json.loads((LABELS_DIR / "page-type-map-v5.json").read_text())
    purposes = json.loads((LABELS_DIR / "purposes-v5.json").read_text())
    return {
        "componentType": sorted(taxonomy["allowedLabels"]["componentTypes"]),
        "regions": sorted(taxonomy["allowedLabels"]["regions"]),
        "purposes": sorted(purposes["retained"]),
        "pageType": list(page_types["classes"]),
    }


def load_policy() -> tuple[str, str]:
    text = (LABELS_DIR / "policy-v5.md").read_text()
    if "## 1. componentType" not in text or "## 4. pageType" not in text:
        raise ValueError("policy-v5.md is missing a required section")
    return text, sha256_text(text)


def packet_key(packet: dict[str, Any]) -> str:
    """A short model-facing key; the full identity never leaves this process."""
    material = "\0".join(
        str(value)
        for value in (
            packet["recordType"],
            packet["split"],
            packet["pageId"],
            packet.get("nodeId") or "",
            packet["captureHash"],
            packet["inputSha256"],
        )
    )
    return "pkt_" + hashlib.sha256(material.encode()).hexdigest()[:12]


def validate_packet(packet: dict[str, Any]) -> None:
    for name in ("recordType", "split", "pageId", "captureHash", "input", "inputSha256"):
        if not isinstance(packet.get(name), str) or not packet[name]:
            raise ValueError(f"packet field {name} is required")
    if packet["recordType"] not in {"node", "page"}:
        raise ValueError("packet recordType must be node or page")
    if packet["recordType"] == "node" and not packet.get("nodeId"):
        raise ValueError("node packet needs nodeId")
    if packet["recordType"] == "page" and packet.get("nodeId") is not None:
        raise ValueError("page packet nodeId must be null")
    if packet["inputSha256"] != sha256_text(packet["input"]):
        raise ValueError("packet inputSha256 does not bind its input")


def blind(packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "packetKey": packet_key(packet),
        "recordType": packet["recordType"],
        "input": packet["input"],
    }


def response_schema(batch: list[dict[str, Any]], vocabulary: dict[str, list[str]]) -> dict[str, Any]:
    multi = {
        axis: {
            "type": "object",
            "additionalProperties": False,
            "required": ["axis", "complete", "labels"],
            "properties": {
                "axis": {"type": "string", "const": axis},
                "complete": {"type": "boolean"},
                "labels": {
                    "type": "array",
                    "items": {"type": "string", "enum": vocabulary[axis]},
                },
            },
        }
        for axis in ("regions", "purposes")
    }
    single = {
        axis: {
            "type": "object",
            "additionalProperties": False,
            "required": ["axis", "complete", "labels"],
            "properties": {
                "axis": {"type": "string", "const": axis},
                "complete": {"type": "boolean"},
                "labels": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 1,
                    "items": {"type": "string", "enum": vocabulary[axis]},
                },
            },
        }
        for axis in ("componentType", "pageType")
    }
    observation = {"anyOf": [multi["regions"], multi["purposes"], single["componentType"], single["pageType"]]}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["records"],
        "properties": {
            "records": {
                "type": "array",
                "minItems": len(batch),
                "maxItems": len(batch),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "packetKey",
                        "source",
                        "gold",
                        "model",
                        "labelingMethod",
                        "promptRevision",
                        "evidence",
                        "observations",
                    ],
                    "properties": {
                        "packetKey": {"type": "string", "enum": [packet_key(p) for p in batch]},
                        "source": {"type": "string", "const": "luna"},
                        "gold": {"type": "boolean", "const": False},
                        "model": {"type": "string", "const": MODEL},
                        "labelingMethod": {
                            "type": "string",
                            "const": "independent_reasoned_review",
                        },
                        "promptRevision": {"type": "string", "const": PROMPT_REVISION},
                        "evidence": {"type": "string", "minLength": 1, "maxLength": 600},
                        "observations": {"type": "array", "items": observation},
                    },
                },
            }
        },
    }


PREAMBLE = """You are Luna, an independent DOM-labeling reviewer. Return JSON only: an object with a `records` array, one record per blind packet, preserving each `packetKey` exactly. Do not use tools, browse, or infer visual details the packet does not carry. Emit no ids, hashes, URLs, selectors or source metadata beyond the requested fields. Set source exactly `luna`, gold exactly false, model exactly `{model}`, labelingMethod exactly `independent_reasoned_review`, promptRevision exactly `{revision}`, and an evidence string grounded only in that packet's input.

THE LABEL POLICY BELOW IS THE ONLY SOURCE OF LABEL MEANING. Apply it literally, including its precedence ladder and its `<footer>`, `<header>` and `<main>` conventions. Where the policy and your own intuition disagree, the policy wins.

--- BEGIN SHARED LABEL POLICY (revision {revision}, sha256 {policy_hash}) ---
{policy}
--- END SHARED LABEL POLICY ---

Packet format. A node packet is `{node_revision}` and carries tag, role, ancestor chain, position among siblings, rendered size in CSS pixels, direct child tags, a subtree tag summary, and bounded text. `text=[empty]` means the element renders no text; judge it from tag, role, size, position and subtree, and use `unknown` when that is not enough. These packets are built from an immutable capture that holds no element attributes, so `alt`, `title` and `src` are unavailable: do not assume an image has a caption you cannot see. A page packet is `{page_revision}` and carries title, URL path and bounded body evidence.

Observations. For a node packet use only the axes regions, purposes, componentType. For a page packet use only pageType. Each observation is `{{"axis": string, "complete": boolean, "labels": string[]}}`.
- `componentType` and `pageType` take exactly one label.
- `regions` and `purposes` are multi-label. Set complete true when the packet decides the whole allowed set, including deciding that none applies, in which case labels is empty. Set complete false only for a positive claim whose negatives you cannot decide; incomplete labels are positive claims and never imply a negative.
- Omit an axis entirely when the packet cannot support any judgment on it.
- `unknown`, where the vocabulary offers it, must stand alone as the only label for its axis.

Allowed labels, literal:
{vocabulary}

Blind packets:
{packets}
"""


def prompt_for(
    batch: list[dict[str, Any]], vocabulary: dict[str, list[str]], policy: str, policy_hash: str
) -> str:
    return PREAMBLE.format(
        model=MODEL,
        revision=PROMPT_REVISION,
        policy=policy,
        policy_hash=policy_hash,
        node_revision="runtime-dom-text-v2",
        page_revision="dom-page-text-v2",
        vocabulary=json.dumps(vocabulary, sort_keys=True),
        packets=json.dumps([blind(packet) for packet in batch], sort_keys=True),
    )


def validate_record(
    record: dict[str, Any], packet: dict[str, Any], vocabulary: dict[str, list[str]]
) -> dict[str, Any]:
    if record.get("packetKey") != packet_key(packet):
        raise ValueError("response packetKey does not match its blind packet")
    if (
        record.get("source") != "luna"
        or record.get("gold") is not False
        or record.get("model") != MODEL
        or record.get("labelingMethod") != "independent_reasoned_review"
        or record.get("promptRevision") != PROMPT_REVISION
    ):
        raise ValueError("response provenance is invalid")
    evidence = record.get("evidence")
    if not isinstance(evidence, str) or not evidence.strip() or len(evidence) > 600:
        raise ValueError("response needs bounded evidence")
    observations = record.get("observations")
    if not isinstance(observations, list):
        raise ValueError("observations must be an array")
    allowed_axes = NODE_AXES if packet["recordType"] == "node" else PAGE_AXES
    axes: dict[str, dict[str, Any]] = {}
    for observation in observations:
        if not isinstance(observation, dict) or set(observation) != {"axis", "complete", "labels"}:
            raise ValueError("each observation needs axis, complete and labels")
        axis = observation["axis"]
        if axis not in allowed_axes or axis in axes:
            raise ValueError(f"axis {axis} is duplicate or invalid for a {packet['recordType']}")
        labels = observation["labels"]
        if not isinstance(observation["complete"], bool) or not isinstance(labels, list):
            raise ValueError(f"{axis} observation is malformed")
        if len(labels) != len(set(labels)) or any(not isinstance(label, str) for label in labels):
            raise ValueError(f"{axis} labels must be unique strings")
        if any(label not in vocabulary[axis] for label in labels):
            raise ValueError(f"{axis} contains an off-taxonomy label")
        if "unknown" in labels and labels != ["unknown"]:
            raise ValueError(f"{axis} unknown must stand alone")
        if axis in {"componentType", "pageType"} and len(labels) != 1:
            raise ValueError(f"{axis} needs exactly one label")
        axes[axis] = {"complete": observation["complete"], "labels": labels}
    row = {key: value for key, value in record.items() if key not in {"packetKey", "observations"}}
    row["observedAxes"] = axes
    for key in (
        "recordType",
        "split",
        "pageId",
        "nodeId",
        "captureHash",
        "inputSha256",
        "input",
        "stratum",
        "policySha256",
    ):
        row[key] = packet.get(key)
    return row


def validated_batch(
    response_text: str,
    batch: list[dict[str, Any]],
    vocabulary: dict[str, list[str]],
    prompt: str,
    number: int,
    latency_ms: int,
) -> list[dict[str, Any]]:
    start = response_text.find("{")
    if start < 0:
        raise ValueError("response contains no JSON")
    payload = json.loads(response_text[start:])
    rows = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or len(rows) != len(batch):
        raise ValueError("response must contain exactly one record per packet")
    by_key = {row.get("packetKey"): row for row in rows if isinstance(row, dict)}
    if len(by_key) != len(rows) or set(by_key) != {packet_key(p) for p in batch}:
        raise ValueError("response packet keys do not match the batch")
    validated = []
    for packet in batch:
        row = validate_record(by_key[packet_key(packet)], packet, vocabulary)
        row["lunaRun"] = {
            "promptSha256": sha256_text(prompt),
            "responseSha256": sha256_text(response_text),
            "batch": number,
            "latencyMs": latency_ms,
            "recordedAt": datetime.now(timezone.utc).isoformat(),
        }
        validated.append(row)
    return validated


def run_batch(prompt: str, output_path: Path, schema_path: Path) -> tuple[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "codex",
        "exec",
        "-m",
        MODEL,
        "--ephemeral",
        "--ignore-user-config",
        "--skip-git-repo-check",
        "-s",
        "read-only",
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(output_path),
        "-",
    ]
    started = time.monotonic()
    completed = subprocess.run(
        command, input=prompt, text=True, cwd="/tmp", stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    latency_ms = int((time.monotonic() - started) * 1000)
    (output_path.parent / (output_path.stem + ".log")).write_text(completed.stdout)
    if completed.returncode != 0 or not output_path.exists():
        raise RuntimeError(f"Luna command failed with exit status {completed.returncode}")
    return output_path.read_text(), latency_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-batches", type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 40:
        raise ValueError("batch-size must be 1..40")

    vocabulary = load_vocabulary()
    policy, policy_hash = load_policy()
    packets = read_jsonl(args.packets)
    for packet in packets:
        validate_packet(packet)
        if packet.get("policySha256") not in (None, policy_hash):
            raise ValueError("packet was built against a different policy revision")
    if len({packet_key(packet) for packet in packets}) != len(packets):
        raise ValueError("derived packet keys must be unique")

    prior = read_jsonl(args.output) if args.output.exists() else []
    completed = {(row.get("pageId"), row.get("nodeId")) for row in prior}
    accepted = list(prior)
    rejected: list[dict[str, Any]] = []
    latencies: list[int] = []
    batches = 0
    for number, start in enumerate(range(0, len(packets), args.batch_size), 1):
        batch = [
            packet
            for packet in packets[start : start + args.batch_size]
            if (packet["pageId"], packet.get("nodeId")) not in completed
        ]
        if not batch:
            continue
        if args.max_batches and batches >= args.max_batches:
            break
        batches += 1
        prompt = prompt_for(batch, vocabulary, policy, policy_hash)
        batch_dir = args.runs_dir / f"batch-{number:04d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        (batch_dir / "prompt.txt").write_text(prompt)
        schema_path = batch_dir / "response-schema.json"
        schema_path.write_text(
            json.dumps(response_schema(batch, vocabulary), indent=2, sort_keys=True) + "\n"
        )
        response_text, latency_ms = run_batch(prompt, batch_dir / "response.json", schema_path)
        latencies.append(latency_ms)
        try:
            accepted.extend(
                validated_batch(response_text, batch, vocabulary, prompt, number, latency_ms)
            )
            completed.update((packet["pageId"], packet.get("nodeId")) for packet in batch)
        except Exception as error:  # noqa: BLE001 - recorded, never silently dropped
            rejected.append(
                {
                    "batch": number,
                    "reason": str(error),
                    "packets": len(batch),
                    "promptSha256": sha256_text(prompt),
                    "responseSha256": sha256_text(response_text),
                }
            )
        write_jsonl(args.output, accepted)
        write_jsonl(args.runs_dir / "rejected-batches.jsonl", rejected)
        print(
            json.dumps(
                {"batch": number, "accepted": len(accepted), "rejected": len(rejected),
                 "latencyMs": latency_ms},
                sort_keys=True,
            ),
            flush=True,
        )
    print(
        json.dumps(
            {
                "accepted": len(accepted),
                "rejectedBatches": len(rejected),
                "batchesRun": batches,
                "policySha256": policy_hash,
                "medianBatchLatencyMs": sorted(latencies)[len(latencies) // 2] if latencies else None,
                "totalLatencyMs": sum(latencies),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
