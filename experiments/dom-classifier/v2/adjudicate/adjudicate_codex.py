# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Second teacher for cross-checks: an OpenAI model through the Codex CLI (#2307).

Same views, same policy prompt and same JSON schema as adjudicate_cli.py, so a
disagreement is between models, not between inputs. Each item is one
`codex exec` process, locked down:

- the binary is called directly (`command codex` in a shell; an alias may add a
  full-access sandbox), with --sandbox read-only, --ephemeral, --ignore-user-config,
  --ignore-rules and an empty working directory;
- the policy prompt REPLACES Codex's base instructions (model_instructions_file);
  the page images go in with -i, the text packet is the prompt;
- shell, exec, browser, computer use, apps, plugins, image generation, skills and
  hooks are disabled by feature flag;
- the --json event stream is the guard: any item other than the agent's message
  or its reasoning (a command, a tool or MCP call, a web search, a file change)
  aborts the WHOLE run, because a read-only sandbox can still read the home dir.

The answer is validated against the schema; invalid output is an abstention.
The stream carries no model id; an id the account cannot use fails the call (a
400), so the requested id is what ran.

    uv run --script adjudicate_codex.py --items items.jsonl --render <dir with views/ and packets.jsonl> \
        --subset ids.txt --model gpt-6-astra --out <dir> --scratch <empty dir>
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
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
from adjudicate import schema as answer_schema
from adjudicate_cli import valid

DISABLED = ["shell_tool", "unified_exec", "apps", "browser_use", "browser_use_external", "computer_use",
            "image_generation", "in_app_browser", "plugins", "remote_plugin", "skill_search", "sleep_tool",
            "tool_suggest", "hooks", "personality"]  # fmt: skip
ALLOWED_ITEMS = {"agent_message", "reasoning"}
LIMIT_WORDS = ("usage limit", "rate limit", "rate_limit", "quota", "too many requests")


def codex_binary() -> str:
    path = shutil.which("codex")
    if not path:
        raise SystemExit("codex CLI not found")
    return path


class Abort:
    def __init__(self) -> None:
        self.reason: str | None = None
        self.lock = threading.Lock()

    def set(self, reason: str) -> None:
        with self.lock:
            self.reason = self.reason or reason


def run_item(item: dict, args, packets: dict, system: dict, files: dict, abort: Abort) -> dict | None:
    if abort.reason:
        return None
    params = build_params(item, packets, args.render / "views", "sonnet-5-5-low", system)
    axis = item["axis"]
    page_views = args.render / "views" / item["pageId"]
    if axis == "page":
        images = [page_views / "page.jpg", page_views / "overview.jpg"]
    else:
        images = [page_views / f"n{item['nodeId']}.jpg", page_views / f"n{item['nodeId']}-overview.jpg"]
    text = params["messages"][0]["content"][-1]["text"]
    cmd = [codex_binary(), "exec", "-m", args.model, "-c", f'model_reasoning_effort="{args.effort}"',
           "-c", f'model_instructions_file="{files[axis]["instructions"]}"',
           "--sandbox", "read-only", "--ephemeral", "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check",
           "-C", str(args.scratch), "--output-schema", str(files[axis]["schema"]), "--json"]  # fmt: skip
    for feature in DISABLED:
        cmd += ["--disable", feature]
    for image in images:
        cmd += ["-i", str(image)]
    # -i takes several values, so the prompt goes on stdin ("-"), never as a trailing argument.
    cmd.append("-")
    base = {"itemId": item["itemId"], "axis": axis, "arm": args.arm, "modality": "image", "model": args.model,
            "promptRevision": PROMPT_REVISION}  # fmt: skip
    started = time.time()
    try:
        proc = subprocess.run(cmd, input=text, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return {**base, "error": "timeout"}
    wall = round((time.time() - started) * 1000)
    message, usage, kinds, failure = None, {}, [], None
    for line in proc.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get("type")
        item_event = event.get("item") or {}
        if kind and kind.startswith("item.") and item_event.get("type"):
            kinds.append(item_event["type"])
            if item_event["type"] not in ALLOWED_ITEMS | {"error"}:
                abort.set(f"lockdown:{item_event['type']}")
            if item_event["type"] == "agent_message" and kind == "item.completed":
                message = item_event.get("text")
        if kind == "turn.completed":
            usage = event.get("usage") or {}
        if kind in ("turn.failed", "error"):
            failure = json.dumps(event)[:300]
    row = {**base, "wallMs": wall, "events": sorted(set(kinds)), "usage": {
        "input": usage.get("input_tokens", 0), "cachedInput": usage.get("cached_input_tokens", 0),
        "output": usage.get("output_tokens", 0), "reasoning": usage.get("reasoning_output_tokens", 0)}}  # fmt: skip
    if abort.reason:
        return {**row, "error": abort.reason}
    if failure or proc.returncode:
        text_err = (failure or proc.stderr[-300:]).lower()
        if any(w in text_err for w in LIMIT_WORDS):
            abort.set(f"limit_error:{text_err[:120]}")
        detail = failure or proc.stderr.strip().splitlines()[-1:] and proc.stderr.strip().splitlines()[-1]
        return {**row, "error": f"failed:rc={proc.returncode}:{str(detail)[:160]}"}
    try:
        answer = json.loads(message or "")
    except json.JSONDecodeError:
        return {**row, "error": "invalid_output"}
    classes = PAGE_CLASSES_V81 if axis == "page" else SECTION_CLASSES_V3
    if not valid(answer, classes, axis == "section"):
        return {**row, "error": "invalid_output"}
    raw = answer["label"]
    if axis == "page":
        mapped = page_rules(raw, item["url"], item.get("platform"))
        runner = (
            page_rules(answer["runner_up"], item["url"], item.get("platform")) if answer["runner_up"] != "-" else None
        )
        extra = {}
    else:
        purpose = section_rules(raw, None)
        scope = answer["scope"] if purpose != "none" else "whole"
        mapped = purpose if scope == "whole" else "none"
        runner = section_rules(answer["runner_up"], None) if answer["runner_up"] != "-" else None
        extra = {"scope": scope, "itemOf": purpose if scope == "item" else None}
    return {**row, "raw": raw, "label": mapped, **extra, "runnerUp": runner if runner != mapped else None,
            "confidence": answer["confidence"], "reason": answer["reason"], "renderQuality": answer["render_quality"]}  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--render", type=Path, required=True)
    parser.add_argument("--subset", type=Path)
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True, help="empty working directory for every process")
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()
    args.arm = f"codex-{args.model}-{args.effort}"
    if any(args.scratch.iterdir()):
        raise SystemExit(f"scratch dir is not empty: {args.scratch}")
    items = jsonl(args.items)
    if args.subset:
        keep = {line.strip() for line in args.subset.read_text().splitlines() if line.strip()}
        items = [i for i in items if i["itemId"] in keep]
    out = args.out / args.arm
    out.mkdir(parents=True, exist_ok=True)
    path = out / "answers-codex.jsonl"
    done = {r["itemId"] for r in jsonl(path) if not r.get("error") or r["error"] == "invalid_output"}
    items = [i for i in items if i["itemId"] not in done]
    packets = {p["pageId"]: p for p in jsonl(args.render / "packets.jsonl")}
    system = systems()
    tmp = Path(tempfile.mkdtemp(prefix="codex-teacher-"))
    files = {}
    for axis, classes in (("page", PAGE_CLASSES_V81), ("section", SECTION_CLASSES_V3)):
        (tmp / f"{axis}-schema.json").write_text(json.dumps(answer_schema(classes, axis == "section")))
        (tmp / f"{axis}-instructions.md").write_text(system[axis])
        files[axis] = {"schema": tmp / f"{axis}-schema.json", "instructions": tmp / f"{axis}-instructions.md"}
    abort = Abort()
    started = time.time()
    counts = {"ok": 0, "errors": 0}
    with path.open("a") as handle, ThreadPoolExecutor(args.concurrency) as pool:
        for row in pool.map(lambda i: run_item(i, args, packets, system, files, abort), items):
            if row is None:
                continue
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            counts["errors" if row.get("error") else "ok"] += 1
            print(json.dumps({**{k: row.get(k) for k in ("itemId", "label", "confidence", "error", "wallMs", "events")},
                              **counts, "elapsed": round(time.time() - started)}), flush=True)  # fmt: skip
    shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps({"stage": "aborted" if abort.reason else "complete", "reason": abort.reason, **counts,
                      "elapsed": round(time.time() - started)}))  # fmt: skip


if __name__ == "__main__":
    main()
