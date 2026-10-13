"""Skeleton extraction over the shared corpus on Modal (#2307).

The page-corpus volume already holds every stored page (read-only here), so nothing is
uploaded: N containers each take the pages whose id hashes to their shard, run
experiments/extract-skeletons.ts, and return the rows as a gzipped tarball. Nothing is
written to any volume. Sealed domains and bucket >= 90 are skipped by the extractor.

    bun build --target=bun ../experiments/extract-skeletons.ts --outfile /tmp/skel-batch.js
    modal run modal_skeleton.py --bundle /tmp/skel-batch.js --out <dir> [--shards 16] [--only 0,3]

Each shard lands in `<out>/shard-<i>/{skeletons.jsonl,features.json}`; a shard that already
has its features.json is skipped, so a rerun resumes. The rows hold page words (headings,
titles): PRIVATE, same rules as the corpus.
"""

from __future__ import annotations

import io
import subprocess
import tarfile
import tempfile
from pathlib import Path

import modal

BUN_VERSION = "1.3.14"  # the Mac's, so the bundle runs on the Bun that built it

image = modal.Image.from_registry(f"oven/bun:{BUN_VERSION}-debian", add_python="3.12")
corpus = modal.Volume.from_name("page-corpus").read_only()
app = modal.App("treetop-skeleton", image=image)


@app.function(cpu=2, memory=6144, timeout=3 * 3600, volumes={"/corpus": corpus})
def run_shard(bundle: bytes, shard: int, shards: int) -> bytes:
    work = Path(tempfile.mkdtemp())
    (work / "batch.js").write_bytes(bundle)
    out = work / "out"
    proc = subprocess.run(
        ["bun", "run", str(work / "batch.js"), "--shard", f"{shard}/{shards}", "--out", str(out), "--corpus", "/corpus"],
        capture_output=True,
        text=True,
    )
    print(f"shard {shard}/{shards} rc={proc.returncode} {proc.stdout.strip()[-300:]}")
    if proc.returncode != 0:
        raise RuntimeError(f"shard {shard} failed: {proc.stderr[-800:]}")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name in ("skeletons.jsonl", "features.json"):
            tar.add(out / name, arcname=name)
    return buffer.getvalue()


@app.local_entrypoint()
def main(bundle: str, out: str, shards: int = 16, only: str = "") -> None:
    data = Path(bundle).read_bytes()
    root = Path(out)
    wanted = [int(i) for i in only.split(",")] if only else list(range(shards))
    todo = [i for i in wanted if not (root / f"shard-{i}" / "features.json").exists()]
    print(f"{len(todo)} of {len(wanted)} shards to run ({shards} total)")
    for shard, blob in zip(todo, run_shard.map([data] * len(todo), todo, [shards] * len(todo), order_outputs=True), strict=True):
        target = root / f"shard-{shard}"
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
            tar.extractall(target, filter="data")
        print(f"shard {shard} written: {sum(1 for _ in (target / 'skeletons.jsonl').open())} rows", flush=True)
