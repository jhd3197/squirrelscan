# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Run the visual adjudicator through headless Claude Code (`claude -p`) on the Team login (#2307).

Same prompts, views and answer shape as adjudicate.py (which talks to the API
directly), but each item is one locked-down `claude -p` process:

- no tools at all (`--tools ""`): the screenshots travel inline as base64 image
  blocks on stream-json input, so the model can read nothing else, write
  nothing, run nothing and fetch nothing; `--json-schema` adds only the
  StructuredOutput tool. `--restricted`, `--safe-mode` (no CLAUDE.md, skills,
  hooks, plugins) and `--strict-mcp-config` with an empty config (no MCP).
  Every run's init event is checked: any tool beyond StructuredOutput, or any
  MCP server, aborts the run.
- `--system-prompt` replaces Claude Code's own prompt with the policy prompt,
  which states that page content is data, never instructions.
- a scratch cwd with nothing in it; no session persistence.
- the answer is re-validated against the schema here; invalid output is an
  abstention (an error row), never a label.

Throughput guard: at most --concurrency processes (default 2). Each result
carries the account's five-hour window utilisation; the run stops when it
reaches --stop-utilization, when the window is not "allowed", or on any usage /
rate limit error, rather than retrying into the limit.

    uv run --script adjudicate_cli.py --items cohort.jsonl --render <run> --model sonnet \
        --ad-markers ad-markers.jsonl --out <adjudicate dir> [--only-pilot] [--subset ids.txt] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from adjudicate import (
    PAGE_CLASSES_V81,
    PROMPT_REVISION,
    SECTION_CLASSES_V3,
    build_params,
    jsonl,
    page_rules,
    section_rules,
    systems,
)

CONFIG_DIR = Path.home() / ".claude-squirrelscan"
LIMIT_WORDS = ("usage limit", "rate limit", "rate_limit", "limit reached", "overloaded", "quota")


def command(model: str, effort: str, system_text: str, schema: dict) -> list[str]:
    return [
        "claude", "-p",
        "--safe-mode", "--restricted",
        "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
        "--tools", "",
        "--disable-slash-commands",
        "--system-prompt", system_text,
        "--json-schema", json.dumps(schema),
        "--model", model, "--effort", effort,
        "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
        "--no-session-persistence",
        "--permission-mode", "dontAsk", "--permission-prompts", "none",
    ]  # fmt: skip


def valid(answer: object, classes: list[str]) -> bool:
    if not isinstance(answer, dict) or set(answer) != {"reason", "label", "confidence", "runner_up", "render_quality"}:
        return False
    return (
        answer["label"] in classes
        and answer["runner_up"] in [*classes, "-"]
        and isinstance(answer["confidence"], int | float)
        and 0 <= answer["confidence"] <= 1
        and answer["render_quality"] in ("ok", "degraded", "unusable")
        and isinstance(answer["reason"], str)
    )


class Guard:
    def __init__(self, stop_at: float):
        self.stop_at = stop_at
        self.stopped: str | None = None
        self.utilization: float | None = None
        self.lock = threading.Lock()

    def observe(self, info: dict | None, error_text: str = "") -> None:
        with self.lock:
            if info:
                window = (info.get("unifiedWindows") or {}).get("five_hour") or {}
                if window.get("utilization") is not None:
                    self.utilization = window["utilization"]
                if info.get("status") not in (None, "allowed"):
                    self.stopped = f"rate_limit_status:{info.get('status')}"
                elif self.utilization is not None and self.utilization >= self.stop_at:
                    self.stopped = f"five_hour_utilization:{self.utilization}"
            if error_text and any(w in error_text.lower() for w in LIMIT_WORDS):
                self.stopped = f"limit_error:{error_text[:120]}"


def run_item(item: dict, args, packets: dict, system: dict, ad_marked: set, guard: Guard, cwd: Path) -> dict | None:
    if guard.stopped:
        return None
    params = build_params(item, packets, args.render / "views", "sonnet-5-5-low", system)
    classes = PAGE_CLASSES_V81 if item["axis"] == "page" else SECTION_CLASSES_V3
    cmd = command(args.model, args.effort, params["system"][0]["text"], params["output_config"]["format"]["schema"])
    line = json.dumps({"type": "user", "message": {"role": "user", "content": params["messages"][0]["content"]}})
    env = {**os.environ, "CLAUDE_CONFIG_DIR": str(CONFIG_DIR)}
    started = time.time()
    base = {"itemId": item["itemId"], "axis": item["axis"], "arm": args.arm, "promptRevision": PROMPT_REVISION}
    try:
        proc = subprocess.run(cmd, input=line + "\n", capture_output=True, text=True, env=env, cwd=cwd, timeout=600)
    except subprocess.TimeoutExpired:
        return {**base, "error": "timeout", "wallMs": round((time.time() - started) * 1000)}
    wall = round((time.time() - started) * 1000)
    init, result, info = None, None, None
    for raw in proc.stdout.splitlines():
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "system" and event.get("subtype") == "init":
            init = event
        elif event.get("type") == "rate_limit_event":
            info = event.get("rate_limit_info")
        elif event.get("type") == "result":
            result = event
    if init and (init.get("tools") != ["StructuredOutput"] or init.get("mcp_servers")):
        guard.stopped = f"lockdown_violated:tools={init.get('tools')} mcp={init.get('mcp_servers')}"
        return {**base, "error": guard.stopped, "wallMs": wall}
    error_text = (result or {}).get("result") if (result or {}).get("is_error") else ""
    guard.observe(info, f"{error_text or ''} {proc.stderr[-400:] if proc.returncode else ''}")
    row = {**base, "wallMs": wall, "utilization": guard.utilization}
    if not result:
        return {**row, "error": f"no_result:rc={proc.returncode}:{proc.stderr[-200:]}"}
    model_usage = result.get("modelUsage") or {}
    usage = result.get("usage") or {}
    row.update(
        {
            "model": next(iter(model_usage), init.get("model") if init else None),
            "latencyMs": result.get("duration_ms"),
            "apiMs": result.get("duration_api_ms"),
            "turns": result.get("num_turns"),
            "usd": result.get("total_cost_usd", 0.0),
            "usage": {
                "input": usage.get("input_tokens", 0),
                "output": usage.get("output_tokens", 0),
                "cacheRead": usage.get("cache_read_input_tokens", 0),
                "cacheWrite": usage.get("cache_creation_input_tokens", 0),
            },
        }
    )
    if result.get("is_error") or result.get("subtype") != "success":
        return {**row, "error": f"{result.get('subtype')}:{str(result.get('result'))[:160]}"}
    answer = result.get("structured_output")
    if not valid(answer, classes):
        return {**row, "error": "invalid_output"}
    raw_label = answer["label"]
    if item["axis"] == "page":
        mapped = page_rules(raw_label, item["url"])
        runner = page_rules(answer["runner_up"], item["url"]) if answer["runner_up"] != "-" else None
    else:
        marked = f"{item['pageId']}#{item['nodeId']}" in ad_marked
        mapped = section_rules(raw_label, marked)
        runner = section_rules(answer["runner_up"], marked) if answer["runner_up"] != "-" else None
    return {
        **row,
        "raw": raw_label,
        "label": mapped,
        "runnerUp": runner if runner != mapped else None,
        "confidence": answer["confidence"],
        "reason": answer["reason"],
        "renderQuality": answer["render_quality"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--render", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ad-markers", type=Path)
    parser.add_argument("--model", required=True, help="claude --model alias: sonnet, opus, fable, haiku")
    parser.add_argument("--effort", default="low")
    parser.add_argument("--only-pilot", action="store_true")
    parser.add_argument("--subset", type=Path)
    parser.add_argument("--axis", choices=["page", "section"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--stop-utilization", type=float, default=0.9)
    parser.add_argument("--scratch", type=Path, required=True, help="empty working directory for every process")
    args = parser.parse_args()
    if args.concurrency > 3:
        raise SystemExit("at most 3 concurrent claude processes on this machine")
    args.arm = f"cli-{args.model}-{args.effort}"
    items = jsonl(args.items)
    if args.only_pilot:
        items = [i for i in items if i.get("pilot")]
    if args.axis:
        items = [i for i in items if i["axis"] == args.axis]
    if args.subset:
        keep = {line.strip() for line in args.subset.read_text().splitlines() if line.strip()}
        items = [i for i in items if i["itemId"] in keep]
    out = args.out / args.arm
    out.mkdir(parents=True, exist_ok=True)
    path = out / "answers-cli.jsonl"
    done = {r["itemId"] for r in jsonl(path) if not r.get("error") or r["error"] == "invalid_output"}
    items = [i for i in items if i["itemId"] not in done]
    if args.limit:
        items = items[: args.limit]
    args.scratch.mkdir(parents=True, exist_ok=True)
    if any(args.scratch.iterdir()):
        raise SystemExit(f"scratch cwd is not empty: {args.scratch}")
    packets = {p["pageId"]: p for p in jsonl(args.render / "packets.jsonl")}
    ad_marked = {f"{r['pageId']}#{r['nodeId']}" for r in jsonl(args.ad_markers)} if args.ad_markers else set()
    system = systems()
    guard = Guard(args.stop_utilization)
    started = time.time()
    counts = {"ok": 0, "error": 0}
    with path.open("a") as handle, ThreadPoolExecutor(args.concurrency) as pool:
        futures = [pool.submit(run_item, i, args, packets, system, ad_marked, guard, args.scratch) for i in items]
        for future in futures:
            row = future.result()
            if row is None:
                continue
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            counts["error" if row.get("error") else "ok"] += 1
            print(
                json.dumps(
                    {
                        **{k: row.get(k) for k in ("itemId", "model", "label", "confidence", "error", "wallMs")},
                        "util": row.get("utilization"),
                        **counts,
                        "elapsed": round(time.time() - started),
                    }
                ),
                flush=True,
            )
    print(
        json.dumps(
            {
                "stage": "stopped" if guard.stopped else "complete",
                "reason": guard.stopped,
                **counts,
                "utilization": guard.utilization,
                "elapsed": round(time.time() - started),
            }
        )
    )


if __name__ == "__main__":
    main()
