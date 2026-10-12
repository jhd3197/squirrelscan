"""Run the CC link batch (experiments/cc-link-batch.ts) on Modal (#2307).

DomainGenius's linking pages exist only as Common Crawl WARC locators. One
Modal container at a time fetches them from data.commoncrawl.org (the batch
holds at most 8 requests in flight, about 4 attempts a second, and backs off
on 403/429/503), reads pages DG already stored from the page-corpus volume
mounted read-only, places the backlinks with the runtime, and returns each
chunk as a tarball. Nothing is written to any volume.

One run is one route. The entrypoint refuses locator rows whose recomputed
bucket, sealed flag or gold domain does not fit `--route`, the batch re-checks
every row from its URL, and the staging dir holds that route only, so it is
copied whole to the right place (train: probe/ + labels/, holdout: holdout/).

    bun build --target=bun ../experiments/cc-link-batch.ts --outfile /tmp/cc-batch.js
    modal run modal_cc_batch.py --bundle /tmp/cc-batch.js --artifact <classifier-v2.json> \\
        --locators <a.jsonl,b.jsonl> --route train --out <staging dir> [--sample 500] [--chunk 2000]

Rerunning into the same `--out` resumes: captures already in its manifest are
skipped, except those left unfetched or failed in transit.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import shutil
import tarfile
import time
from pathlib import Path

import modal

BUN_VERSION = "1.3.14"  # the Mac's, so a bundle runs on the Bun that built it
CORPUS = Path("/Volumes/BACKUP/page-corpus")

image = modal.Image.from_registry(f"oven/bun:{BUN_VERSION}-debian", add_python="3.12")
corpus = modal.Volume.from_name("page-corpus").read_only()
app = modal.App("treetop-cc-link-batch", image=image)

RETRY_STATUSES = ("not_fetched_throttled", "error", "warc_error: gave up")


def capture_key(row: dict) -> str:
    raw = f"{row['warc_filename']}|{row['offset']}|{row['length']}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def route_of(row: dict, sealed: set[str]) -> str:
    return "holdout" if row["bucket"] >= 90 or row["domain_hash"] in sealed else "train"


def preflight(rows: list[dict], route: str, sealed: set[str], gold: set[str]) -> None:
    """Refuse the whole run if any row does not belong to `route`."""
    bad: dict[str, int] = {}
    for row in rows:
        problem = None
        if row.get("capture") is not None and row["capture"] != capture_key(row):
            problem = "capture_mismatch"
        elif int(row["domain_hash"][:8], 16) % 100 != row["bucket"]:
            problem = "bucket_mismatch"
        elif bool(row.get("sealed")) != (row["domain_hash"] in sealed):
            problem = "sealed_flag_mismatch"
        elif route_of(row, sealed) != route:
            problem = "route_mismatch"
        elif route == "train" and row["domain_hash"] in gold:
            problem = "gold_domain"
        if problem:
            bad[problem] = bad.get(problem, 0) + 1
    if bad:
        raise SystemExit(f"preflight refused {sum(bad.values())} rows for route {route}: {bad}")


@app.function(
    cpu=2,
    memory=8192,
    timeout=3 * 3600,
    volumes={"/corpus": corpus},
    max_containers=1,
    # HTTPS_PROXY_URL for the batch's outbound CC requests (cc-warc.ts reads it).
    # Created once: modal secret create treetop-cc-proxy HTTPS_PROXY_URL=<url> (never commit the URL)
    secrets=[modal.Secret.from_name("treetop-cc-proxy")],
)
def run_chunk(
    bundle: bytes, artifact: bytes, guard: bytes, locators: bytes, concurrency: int, rate: float, headers_only: bool
) -> dict:
    import subprocess

    job = Path("/root/job")
    if job.exists():
        shutil.rmtree(job)
    (job / "out").mkdir(parents=True)
    for name, data in {
        "batch.js": bundle,
        "artifact.json": artifact,
        "guard.json": guard,
        "locators.jsonl": locators,
    }.items():
        (job / name).write_bytes(data)
    started = time.time()
    result = subprocess.run(
        [
            "bun",
            "run",
            str(job / "batch.js"),
            "--locators",
            str(job / "locators.jsonl"),
            "--artifact",
            str(job / "artifact.json"),
            "--guard",
            str(job / "guard.json"),
            "--out",
            str(job / "out"),
            "--corpus",
            "/corpus",
            "--concurrency",
            str(concurrency),
            "--rate",
            str(rate),
            *(["--headers-only"] if headers_only else []),
        ],
        cwd="/root",
        capture_output=True,
        text=True,
    )
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.add(job / "out", arcname="out")
    return {
        "returncode": result.returncode,
        "seconds": round(time.time() - started, 1),
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
        "tar": buffer.getvalue(),
    }


def merge_chunk(tar_bytes: bytes, out: Path) -> dict:
    """Append one chunk's rows to the staging dir and move its blobs in."""
    work = out / "_chunk"
    if work.exists():
        shutil.rmtree(work)
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        tar.extractall(work, filter="data")
    chunk = work / "out"
    (out / "blobs").mkdir(exist_ok=True)
    for blob in (chunk / "blobs").iterdir():
        target = out / "blobs" / blob.name
        if not target.exists():
            shutil.move(blob, target)
    for name in ("manifest.jsonl", "links.jsonl", "pages.jsonl", "headers.jsonl"):
        with (out / name).open("a") as handle:
            handle.write((chunk / name).read_text())
    stats = json.loads((chunk / "stats.json").read_text())
    with (out / "chunk-stats.jsonl").open("a") as handle:
        handle.write(json.dumps(stats) + "\n")
    shutil.rmtree(work)
    return stats


def load_guard_sets() -> tuple[set[str], set[str]]:
    sealed = {json.loads(line)["d"] for line in (CORPUS / "splits/sealed-domains.jsonl").open()}
    if len(sealed) < 1000:
        raise SystemExit(f"{len(sealed)} sealed hashes, expected ~1069 (key `d`)")
    gold = {line.strip() for line in (CORPUS / "gold-domain-hashes.txt").open() if line.strip()}
    return sealed, gold


def resume(out_dir: Path) -> set[str]:
    """Captures already done in `out_dir`; rows left unfetched or failed in
    transit are dropped from its manifest so they run again."""
    done: set[str] = set()
    manifest = out_dir / "manifest.jsonl"
    if not manifest.exists():
        return done
    kept = []
    for line in manifest.read_text().splitlines():
        row = json.loads(line)
        if row["status"].startswith(RETRY_STATUSES):
            continue
        done.add(row["capture"])
        kept.append(line)
    manifest.write_text("".join(f"{line}\n" for line in kept))
    return done


@app.local_entrypoint()
def main(
    bundle: str,
    artifact: str,
    locators: str,
    route: str,
    out: str,
    sample: int = 0,
    chunk: int = 2000,
    seed: int = 2307,
    concurrency: int = 8,
    rate: float = 4.0,
    only: str = "",
    headers_only: bool = False,
) -> None:
    if route not in ("train", "holdout"):
        raise SystemExit("--route must be train or holdout")
    sealed, gold = load_guard_sets()
    rows = [json.loads(line) for path in locators.split(",") for line in Path(path).open() if line.strip()]
    preflight(rows, route, sealed, gold)

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "route"
    if marker.exists() and marker.read_text().strip() != route:
        raise SystemExit(f"{out_dir} holds route {marker.read_text().strip()}, not {route}")
    marker.write_text(route + "\n")
    done = resume(out_dir)

    by_capture: dict[str, list[dict]] = {}
    for row in rows:
        by_capture.setdefault(capture_key(row), []).append(row)
    todo = sorted(key for key in by_capture if key not in done)
    if only:
        # A backfill: only the captures an earlier run's manifest lists.
        listed = {json.loads(line)["capture"] for line in Path(only).read_text().splitlines() if line.strip()}
        todo = [key for key in todo if key in listed]
    if sample:
        random.Random(seed).shuffle(todo)
        todo = sorted(todo[:sample])
    print(f"route {route}: {len(rows)} rows, {len(by_capture)} captures, {len(done)} done, {len(todo)} to run")

    guard = json.dumps({"route": route, "sealed": sorted(sealed), "gold": sorted(gold)}).encode()
    bundle_bytes = Path(bundle).read_bytes()
    artifact_bytes = Path(artifact).read_bytes()
    for start in range(0, len(todo), chunk):
        keys = todo[start : start + chunk]
        payload = "".join(json.dumps(row) + "\n" for key in keys for row in by_capture[key]).encode()
        result = run_chunk.remote(bundle_bytes, artifact_bytes, guard, payload, concurrency, rate, headers_only)
        if result["returncode"] != 0:
            print(result["stdout"], result["stderr"])
            raise SystemExit(f"chunk at {start} failed: returncode {result['returncode']}")
        stats = merge_chunk(result["tar"], out_dir)
        print(
            f"chunk {start // chunk + 1}: {len(keys)} captures in {result['seconds']}s "
            f"{json.dumps({k: stats[k] for k in ('statuses', 'throttles', 'tripped', 'fetchMs', 'classifyMs')})}"
        )
        if stats["tripped"]:
            raise SystemExit("throttled: the circuit breaker tripped; stopping before the next chunk")
