#!/usr/bin/env python3
"""Reviewer B: a second independent reviewer, run under v4 **and** v5.

Why a second reviewer exists. Reviewer A (``gpt-5.6-luna``, via the Codex CLI)
cannot be re-run while its account is over quota, so the only v5 number
available was Jev-v5 against Reviewer-A-**v4** — a cross-revision proxy that
measures whether Jev adopted the reviewer's prior, not whether the two agree.
Reviewer B answers the same 200 nodes and 125 pages **twice**, once under each
revision, so G1 can be judged on a within-reviewer delta:

    Jev-v4 vs B-v4   ->   Jev-v5 vs B-v5

Both legs then share a reviewer, a record set and a scoring rule; only the
prompt revision differs. That is the comparison the gate actually asks for.

Comparability is by construction, not by description. The prompts are the
reviewer-A modules' own ``prompt_for`` output, with one literal substituted: the
model id the response must declare. Every other byte of the v4 and v5
instructions is the text reviewer A was given. The substitution is asserted, so
a future prompt edit cannot silently skip it.

Reviewer B is blind in the same way reviewer A is: a batch carries a derived
packet key, the record type and the packet text. It never sees Jev's answers,
reviewer A's answers, its own other-revision answers, an id, a hash or a URL.
Rows from different reviewers or revisions are never written to one file.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REVIEWER = "B"
REVIEWER_MODEL = "claude-sonnet-5"
CLI_MODEL_ALIAS = "sonnet"
# Claude Code loads project context by default; the reviewer must not read the
# repository it is reviewing labels for, so tools are denied and the process
# runs from a directory with no CLAUDE.md.
DENIED_TOOLS = "Bash,Edit,Write,Read,WebFetch,WebSearch,Glob,Grep,Task,NotebookEdit,TodoWrite"
WORKING_DIR = "/tmp"


def _load(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"reviewer_b_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    path.chmod(0o600)


def retarget_model(text: str, original: str) -> str:
    """Point the response contract at reviewer B without touching anything else.

    The two revisions name the model differently: v5 interpolates the module
    constant, so moving the constant is enough, while v4 hardcodes the literal in
    its prompt string and needs the substitution. Both paths end at the same
    invariant, which is asserted rather than assumed: the prompt names reviewer
    B's model and no longer names reviewer A's.
    """
    result = text.replace(original, REVIEWER_MODEL) if original in text else text
    if REVIEWER_MODEL not in result:
        raise ValueError("prompt does not name reviewer B's model")
    if original in result:
        raise ValueError("prompt still names reviewer A's model")
    return result


def extract_json(raw: str) -> str:
    """The first complete JSON object in a reply, fences and prose tolerated."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.rstrip().endswith("```"):
            text = text.rstrip()[: -len("```")]
    start = text.find("{")
    if start < 0:
        raise ValueError("reviewer reply contains no JSON object")
    decoded, end = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(decoded, dict):
        raise ValueError("reviewer reply is not a JSON object")
    return text[start : start + end]


class Revision:
    """One prompt revision's packet source, instructions and validator."""

    def __init__(self, name: str) -> None:
        if name not in {"v4", "v5", "v6", "v7"}:
            raise ValueError("revision must be v4, v5, v6 or v7")
        self.name = name
        if name == "v4":
            self.module = _load("luna_batch")
            self.original_model = self.module.MODEL
            self.revision_id = "dom-suggestions-v4"
            self.taxonomy_revision, self.allowed = self.module.load_taxonomy(
                Path(__file__).parents[1] / "taxonomy.json"
            )
            self.policy_sha256 = None
        else:
            # v5 and v6 share one runner and differ only in their policy file.
            self.module = _load("luna_v5_batch")
            self.original_model = self.module.MODEL
            self.revision_id = self.module.prompt_revision(name)
            self.allowed = self.module.load_vocabulary()
            _, self.policy_sha256 = self.module.load_policy(name)
        # Reviewer B must declare its own model. Reviewer A's validators compare
        # against their module constant, so the constant moves with the prompt
        # and the two stay in lockstep.
        self.module.MODEL = REVIEWER_MODEL

    def key(self, packet: dict[str, Any]) -> str:
        if self.name == "v4":
            return self.module.compact_packet_key(packet)
        return self.module.packet_key(packet)

    def prompt(self, batch: list[dict[str, Any]]) -> str:
        if self.name == "v4":
            text = self.module.prompt_for(batch, self.taxonomy_revision, self.allowed)
        else:
            policy, policy_hash = self.module.load_policy(self.name)
            text = self.module.prompt_for(
                batch, self.allowed, policy, policy_hash, self.revision_id
            )
        return retarget_model(text, self.original_model)

    def validate(self, record: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
        if self.name == "v4":
            return self.module.validate_record(
                record, packet, self.taxonomy_revision, self.allowed
            )
        return self.module.validate_record(record, packet, self.allowed, self.revision_id)

    def validate_packet(self, packet: dict[str, Any]) -> None:
        if self.name == "v4":
            self.module.validate_packet(packet, self.taxonomy_revision)
        else:
            self.module.validate_packet(packet)


def truncate_packet(text: str, budget: int) -> tuple[str, bool]:
    """Bound a packet's text deterministically, keeping every structural line.

    Only the `text=` line is cut, and always at the same character count, so a
    truncated packet is reproducible. The structural lines — tag, role,
    ancestors, attributes — are what most judgments turn on and are never
    dropped. A truncated row is marked rather than silently shortened, and a row
    is truncated rather than dropped: a missing row biases the audit toward
    whatever is short.
    """
    lines = text.split("\n")
    for index, line in enumerate(lines):
        if line.startswith("text=") and len(line) > budget + 5:
            lines[index] = line[: budget + 5]
            return "\n".join(lines), True
    return text, False


def run_cli(prompt: str, log_path: Path, timeout_s: float) -> tuple[str, float, int]:
    command = [
        "claude",
        "-p",
        "--model",
        CLI_MODEL_ALIAS,
        "--output-format",
        "json",
        "--strict-mcp-config",
        "--disallowed-tools",
        DENIED_TOOLS,
    ]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            input=prompt,
            text=True,
            cwd=WORKING_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as error:
        # A hung call is the failure mode that cost this run an hour: the CLI
        # produced nothing and never returned. Bound it so the batch is recorded
        # as rejected and the rest of the queue continues.
        raise RuntimeError(f"reviewer call exceeded {timeout_s:.0f}s") from error
    latency_ms = int((time.monotonic() - started) * 1000)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(
            f"reviewer B exited {completed.returncode}: {completed.stderr.strip()[:300]}"
        )
    envelope = json.loads(completed.stdout)
    if envelope.get("is_error") or envelope.get("subtype") not in (None, "success"):
        raise RuntimeError(f"reviewer B returned an error envelope: {envelope.get('subtype')}")
    reply = envelope.get("result")
    if not isinstance(reply, str) or not reply.strip():
        raise RuntimeError("reviewer B returned no text")
    return reply, float(envelope.get("total_cost_usd") or 0.0), latency_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--revision", choices=("v4", "v5", "v6", "v7"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument(
        "--max-packet-chars",
        type=int,
        default=0,
        help="bound each packet's text= line; 0 leaves packets untouched",
    )
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 40:
        raise ValueError("batch-size must be 1..40")
    if not 1 <= args.concurrency <= 2:
        raise ValueError("concurrency must be 1 or 2")

    revision = Revision(args.revision)
    packets = read_jsonl(args.packets)
    truncated_ids: set[str] = set()
    if args.max_packet_chars:
        bounded = []
        for packet in packets:
            text, was_cut = truncate_packet(packet["input"], args.max_packet_chars)
            if was_cut:
                truncated_ids.add(f"{packet['pageId']}\0{packet.get('nodeId') or ''}")
                packet = {**packet, "input": text, "inputSha256": sha256_text(text)}
            bounded.append(packet)
        packets = bounded
    for packet in packets:
        revision.validate_packet(packet)
    if len({revision.key(packet) for packet in packets}) != len(packets):
        raise ValueError("derived packet keys must be unique")

    prior = read_jsonl(args.output) if args.output.exists() else []
    for row in prior:
        if row.get("reviewer") != REVIEWER or row.get("promptRevision") != revision.revision_id:
            raise ValueError("output already holds another reviewer or revision; use a new file")
    completed_ids = {(row.get("pageId"), row.get("nodeId")) for row in prior}

    batches: list[tuple[int, list[dict[str, Any]]]] = []
    for number, start in enumerate(range(0, len(packets), args.batch_size), 1):
        batch = [
            packet
            for packet in packets[start : start + args.batch_size]
            if (packet["pageId"], packet.get("nodeId")) not in completed_ids
        ]
        if batch:
            batches.append((number, batch))
    if args.max_batches:
        batches = batches[: args.max_batches]

    accepted = list(prior)
    rejected: list[dict[str, Any]] = []
    costs: list[float] = []
    latencies: list[int] = []
    lock = threading.Lock()

    def work(item: tuple[int, list[dict[str, Any]]]) -> None:
        number, batch = item
        prompt = revision.prompt(batch)
        batch_dir = args.runs_dir / f"batch-{number:04d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        (batch_dir / "prompt.txt").write_text(prompt)
        try:
            reply, cost, latency_ms = run_cli(
                prompt, batch_dir / "cli-envelope.json", args.timeout_s
            )
        except Exception as error:  # noqa: BLE001 - recorded, never silently dropped
            with lock:
                rejected.append({"batch": number, "reason": str(error), "packets": len(batch)})
            return
        (batch_dir / "response.txt").write_text(reply)
        try:
            payload = json.loads(extract_json(reply))
            rows = payload.get("records")
            if not isinstance(rows, list) or len(rows) != len(batch):
                raise ValueError("reply must contain exactly one record per packet")
            by_key = {row.get("packetKey"): row for row in rows if isinstance(row, dict)}
            if len(by_key) != len(rows) or set(by_key) != {revision.key(p) for p in batch}:
                raise ValueError("reply packet keys do not match the batch")
            validated = []
            for packet in batch:
                row = revision.validate(by_key[revision.key(packet)], packet)
                row = dict(row)
                row["reviewer"] = REVIEWER
                row["reviewerModel"] = REVIEWER_MODEL
                row["promptRevision"] = revision.revision_id
                row["packetRevision"] = packet.get("packetRevision", revision.revision_id)
                row["stratum"] = packet.get("stratum")
                row["truncated"] = (
                    f"{packet['pageId']}\0{packet.get('nodeId') or ''}" in truncated_ids
                )
                if revision.policy_sha256:
                    row["policySha256"] = revision.policy_sha256
                row["reviewerRun"] = {
                    "batch": number,
                    "promptSha256": sha256_text(prompt),
                    "responseSha256": sha256_text(reply),
                    "costUsd": cost,
                    "latencyMs": latency_ms,
                    "recordedAt": datetime.now(timezone.utc).isoformat(),
                }
                validated.append(row)
        except Exception as error:  # noqa: BLE001 - a bad sibling voids the batch
            with lock:
                rejected.append(
                    {
                        "batch": number,
                        "reason": str(error),
                        "packets": len(batch),
                        "responseSha256": sha256_text(reply),
                        "costUsd": cost,
                    }
                )
                costs.append(cost)
            return
        with lock:
            accepted.extend(validated)
            costs.append(cost)
            latencies.append(latency_ms)
            write_jsonl(args.output, accepted)
            write_jsonl(args.runs_dir / "rejected-batches.jsonl", rejected)
            print(
                json.dumps(
                    {"batch": number, "accepted": len(accepted), "rejected": len(rejected)},
                    sort_keys=True,
                ),
                flush=True,
            )

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(work, batches))
    write_jsonl(args.output, accepted)
    write_jsonl(args.runs_dir / "rejected-batches.jsonl", rejected)
    print(
        json.dumps(
            {
                "reviewer": REVIEWER,
                "reviewerModel": REVIEWER_MODEL,
                "promptRevision": revision.revision_id,
                "accepted": len(accepted),
                "rejectedBatches": len(rejected),
                "batchesRun": len(batches),
                "totalCostUsd": round(sum(costs), 4),
                "medianBatchLatencyMs": sorted(latencies)[len(latencies) // 2]
                if latencies
                else None,
                "totalLatencyMs": sum(latencies),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
