"""Run a bundled Bun script on Modal instead of the shared Mac (#2307).

The probe and the bench are TypeScript. `bun build --target=bun` folds each into
one file with no node_modules, so the image is plain Bun and the bundle travels
as an argument: no image rebuild per code change. Data lives in the same
private Volume as phase 3, mounted at /data (never in an image layer):

    bun build --target=bun ../experiments/link-position.ts --outfile /tmp/probe.js
    modal volume put dom-classifier-2307 <local-inputs> /probe/<name>
    modal run modal_bun.py --bundle /tmp/probe.js \\
        --argv "--in /data/probe/<name> --artifact /data/... --out /data/probe-runs/<name>"

Per-page timings are this container's, not the Mac's: compare a run against a
baseline taken here, never against a local run.
"""

from __future__ import annotations

import shlex
import time
from pathlib import Path

import modal

VOLUME_NAME = "dom-classifier-2307"
BUN_VERSION = "1.3.14"  # the Mac's, so a bundle runs on the Bun that built it
CPUS = 4

image = modal.Image.from_registry(f"oven/bun:{BUN_VERSION}-debian", add_python="3.12")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App("dom-classifier-bun", image=image)


@app.function(cpu=CPUS, memory=8192, timeout=2 * 3600, volumes={"/data": volume})
def run(bundle: bytes, argv: str) -> dict:
    import os
    import subprocess

    job = Path("/root/job.js")
    job.write_bytes(bundle)
    started = time.time()
    result = subprocess.run(["bun", "run", str(job), *shlex.split(argv)],
                            cwd="/root", capture_output=True, text=True)
    volume.commit()
    return {"argv": argv, "returncode": result.returncode,
            "seconds": round(time.time() - started, 1), "cpus": os.cpu_count(),
            "stdout": result.stdout[-20000:], "stderr": result.stderr[-8000:]}


@app.local_entrypoint()
def main(bundle: str, argv: str) -> None:
    out = run.remote(Path(bundle).read_bytes(), argv)
    print(out["stdout"])
    if out["stderr"]:
        print("--- stderr ---")
        print(out["stderr"])
    print(f"=== returncode {out['returncode']} in {out['seconds']}s on {out['cpus']} cpus ===")
    if out["returncode"] != 0:
        raise SystemExit(out["returncode"])
