# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Haiku, then Opus on the uncertain items, for one packets file (#2307).

    uv run --script label_pipeline.py --packets p.jsonl --prefix <dir>/answers-x [--stop-utilization 0.8]

Writes `<prefix>-haiku.jsonl` and `<prefix>-opus.jsonl` (resumable: finished items are skipped) on the Max login via
judge_text.py. Opus sees every item Haiku answered under 0.8 confidence, or did not answer.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(packets: Path, out: Path, model: str, stop: float, effort: str) -> None:
    subprocess.run(
        ["uv", "run", "--script", str(HERE / "judge_text.py"), "--packets", str(packets), "--out", str(out),
         "--model", model, "--effort", effort, "--concurrency", "3", "--stop-utilization", str(stop)],
        check=True,
    )  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--stop-utilization", type=float, default=0.8)
    args = parser.parse_args()
    haiku = Path(f"{args.prefix}-haiku.jsonl")
    opus = Path(f"{args.prefix}-opus.jsonl")
    packets = {json.loads(line)["itemId"]: line for line in args.packets.read_text().splitlines() if line.strip()}
    run(args.packets, haiku, "haiku", args.stop_utilization, "low")
    answered = {}
    for line in haiku.read_text().splitlines():
        row = json.loads(line)
        if "error" not in row:
            answered[row["itemId"]] = row
    pick = [i for i in packets if i not in answered or answered[i]["confidence"] < 0.8]
    second = Path(f"{args.prefix}-opus-packets.jsonl")
    second.write_text("".join(packets[i] + "\n" for i in pick))
    print(f"haiku answered {len(answered)}/{len(packets)}; opus second look {len(pick)}", flush=True)
    run(second, opus, "opus", args.stop_utilization, "low")
    print("pipeline done", flush=True)


if __name__ == "__main__":
    main()
