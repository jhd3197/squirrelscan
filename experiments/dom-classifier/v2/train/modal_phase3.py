"""Run the phase-3 training scripts on Modal instead of the shared Mac.

The scripts in this directory run unchanged: the image carries only this
directory's code and the pinned dependencies, and every byte of data lives in a
private Modal Volume mounted at /data (never in an image layer). A step is one
command line, run with the Volume as --data-root:

    modal run modal_phase3.py --argv "train_region.py --sample /data/models-v3/B/region-sample ..."

Upload the inputs first with `modal volume put <volume> <local> <remote>`; the
Volume layout mirrors the private data root, so paths read the same on both.
"""

from __future__ import annotations

import shlex
import time
from pathlib import Path

import modal

VOLUME_NAME = "dom-classifier-2307"
CPUS = 32

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    # Pinned to the classifier-v2 venv that trained the shipped artifact, so a
    # Modal-trained head is not also a new-dependency-trained head.
    .pip_install(
        "lightgbm==4.7.0",
        "numpy==2.5.3",
        "scipy==1.18.1",
        "scikit-learn==1.9.1",
        "joblib==1.6.0",
        "threadpoolctl==3.7.0",
    )
    .apt_install("libgomp1")
    .add_local_dir(Path(__file__).parent, "/root/train", ignore=["__pycache__", "*.pyc"])
)
# Page-type policy v8.2's rules live with the adjudicator (one function both
# sides import); train_pagetype_v8 --policy-version v8.2 reads it from here.
POLICY_RULES = Path(__file__).parent.parent / "adjudicate" / "policy_rules.py"
if POLICY_RULES.exists():
    image = image.add_local_file(POLICY_RULES, "/root/adjudicate/policy_rules.py")
app = modal.App("dom-classifier-phase3", image=image)


def _run(argv: str, dataset: str, cpus: int) -> dict:
    import os
    import subprocess

    env = {**os.environ, "DOM_CLASSIFIER_DATASET_VERSION": dataset,
           "DOM_CLASSIFIER_JOBS": str(cpus)}
    started = time.time()
    result = subprocess.run(["python", *shlex.split(argv)], cwd="/root/train", env=env,
                            capture_output=True, text=True)
    volume.commit()
    return {"argv": argv, "returncode": result.returncode,
            "seconds": round(time.time() - started, 1),
            "stdout": result.stdout[-20000:], "stderr": result.stderr[-8000:]}


@app.function(cpu=CPUS, memory=65536, timeout=4 * 3600, volumes={"/data": volume})
def run(argv: str, dataset: str) -> dict:
    return _run(argv, dataset, CPUS)


#: Staging copies and exports are single-threaded Python: a 32-core worker is
#: waste, and on 2026-10-09 it waited up to 20 minutes for capacity.
LIGHT_CPUS = 4


@app.function(cpu=LIGHT_CPUS, memory=16384, timeout=2 * 3600, volumes={"/data": volume})
def run_light(argv: str, dataset: str) -> dict:
    return _run(argv, dataset, LIGHT_CPUS)


@app.local_entrypoint()
def main(argv: str, dataset: str = "silver-v3", light: bool = False) -> None:
    out = (run_light if light else run).remote(argv, dataset)
    print(out["stdout"])
    if out["stderr"]:
        print("--- stderr ---")
        print(out["stderr"])
    print(f"=== returncode {out['returncode']} in {out['seconds']}s ===")
    if out["returncode"] != 0:
        raise SystemExit(out["returncode"])
