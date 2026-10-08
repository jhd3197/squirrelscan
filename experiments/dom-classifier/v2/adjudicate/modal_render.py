"""Replay-render stored pages on Modal and compose the adjudicator's views (#2307).

Screenshots for thousands of pages must not render on the shared 16 GB Mac. Each
container runs the labeller's hardened replay (`render-worker.ts`, which calls
`labeler/replay-capture.ts` and its egress guard unchanged) under Bun, then cuts
the images the adjudicator sees with Pillow:

  page.jpg              the top 2400 px of the page at 0.75 scale
  overview.jpg          the whole capture as a narrow thumbnail
  n<N>.jpg              a full-width band around section target N, outlined
  n<N>-overview.jpg     the thumbnail with target N outlined

Inputs come from prepare-render.ts (<run>/jobs.jsonl + jobs/<pageId>.html.gz) and
travel as call arguments; the full captures (PNG + manifest) are written to the
private Volume under /adjudicate/<run>/captures, never to an image layer or a
repo. Views and render rows come back to <run>/views and <run>/renders.jsonl.
Resumable: a page is skipped when renders.jsonl already holds a failure for it, or a render
that located every marker its job asks for (a page that gains a section target re-renders).

    modal run modal_render.py --run-dir <private>/render/cal-v1 [--limit N] [--chunk 12]
"""

from __future__ import annotations

import gzip
import io
import json
import time
from pathlib import Path

import modal

VOLUME_NAME = "dom-classifier-2307"
HERE = Path(__file__).parent
EXPERIMENT = HERE.parent.parent  # experiments/dom-classifier
APP_ROOT = "/root/app"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "unzip", "ca-certificates", "fonts-noto-core", "fonts-noto-cjk", "fonts-noto-color-emoji")
    .run_commands('curl -fsSL https://bun.sh/install | bash -s "bun-v1.3.14"')
    .env({"PATH": "/root/.bun/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"})
    # Same Playwright as the labeller's package.json, so the replay is the labeller's replay.
    .run_commands(
        f'mkdir -p {APP_ROOT} && cd {APP_ROOT} && echo \'{{"name":"render","private":true}}\' > package.json'
        " && bun add playwright@1.55.0 && bunx playwright install --with-deps chromium"
    )
    .pip_install("pillow==11.3.0")
    .add_local_dir(
        EXPERIMENT / "labeler",
        f"{APP_ROOT}/experiments/dom-classifier/labeler",
        ignore=["*.test.ts", "*.html", "*.css", "*.js", "*.md"],
    )
    .add_local_file(HERE / "render-worker.ts", f"{APP_ROOT}/experiments/dom-classifier/v2/adjudicate/render-worker.ts")
    .add_local_file(HERE / "replay-html.ts", f"{APP_ROOT}/experiments/dom-classifier/v2/adjudicate/replay-html.ts")
)
app = modal.App("dom-classifier-adjudicate-render", image=image)

PAGE_TOP = 2400
PAGE_SCALE = 0.75
OVERVIEW_WIDTH = 320
OVERVIEW_MAX_HEIGHT = 2400
BAND_CONTEXT = 250
BAND_MAX = 1700
BAND_SCALE = 0.8
OUTLINE = (0, 200, 83)


def _jpeg(img, quality: int = 82) -> bytes:
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, "JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def _outline(draw, box, width: int, label: bool, font=None):
    x0, y0, x1, y1 = box
    # A dark hairline outside the green keeps the outline visible on green pages.
    draw.rectangle((x0 - width - 1, y0 - width - 1, x1 + width + 1, y1 + width + 1), outline=(20, 20, 20), width=1)
    draw.rectangle((x0 - width, y0 - width, x1 + width, y1 + width), outline=OUTLINE, width=width)
    if label:
        tag_y = y0 - width - 26 if y0 - width - 26 >= 0 else y0 + 2
        draw.rectangle((x0 - width, tag_y, x0 - width + 84, tag_y + 24), fill=OUTLINE)
        draw.text((x0 - width + 8, tag_y + 4), "TARGET", fill=(255, 255, 255), font=font)


def compose_views(png: bytes, located: dict) -> dict[str, bytes]:
    from PIL import Image, ImageDraw, ImageFont

    Image.MAX_IMAGE_PIXELS = None
    shot = Image.open(io.BytesIO(png)).convert("RGB")
    width, height = shot.size
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    views: dict[str, bytes] = {}
    top = shot.crop((0, 0, width, min(height, PAGE_TOP)))
    views["page"] = _jpeg(top.resize((int(top.width * PAGE_SCALE), int(top.height * PAGE_SCALE)), Image.LANCZOS))
    scale = min(OVERVIEW_WIDTH / width, OVERVIEW_MAX_HEIGHT / height)
    overview = shot.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.LANCZOS)
    views["overview"] = _jpeg(overview, 75)
    for marker, at in located.items():
        rect = at.get("rect")
        if not (at.get("visible") and rect):
            continue
        x0, y0 = rect["x"], rect["y"]
        x1, y1 = x0 + rect["width"], y0 + rect["height"]
        top_y = max(0, y0 - BAND_CONTEXT)
        bottom_y = min(height, y1 + BAND_CONTEXT, top_y + BAND_MAX)
        band = shot.crop((0, top_y, width, bottom_y))
        draw = ImageDraw.Draw(band)
        _outline(draw, (x0, y0 - top_y, x1, min(y1, bottom_y) - top_y), 5, True, font)
        views[f"n{marker}"] = _jpeg(
            band.resize((int(band.width * BAND_SCALE), int(band.height * BAND_SCALE)), Image.LANCZOS)
        )
        thumb = overview.copy()
        tdraw = ImageDraw.Draw(thumb)
        _outline(tdraw, (x0 * scale, y0 * scale, x1 * scale, y1 * scale), 3, False)
        views[f"n{marker}-overview"] = _jpeg(thumb, 75)
    return views


@app.function(cpu=4, memory=8192, timeout=1800, volumes={"/data": volume}, max_containers=40)
def render_chunk(run: str, jobs: list[dict], html: dict[str, bytes]) -> list[dict]:
    import shutil
    import subprocess

    work = Path("/tmp/work")
    shutil.rmtree(work, ignore_errors=True)
    (work / "html").mkdir(parents=True)
    for page_id, data in html.items():
        (work / "html" / f"{page_id}.html").write_bytes(gzip.decompress(data))
    (work / "jobs.json").write_text(json.dumps(jobs))
    results_path = work / "results.jsonl"
    started = time.time()
    proc = subprocess.run(
        [
            "bun",
            "run",
            f"{APP_ROOT}/experiments/dom-classifier/v2/adjudicate/render-worker.ts",
            "--jobs",
            str(work / "jobs.json"),
            "--html",
            str(work / "html"),
            "--store",
            str(work / "store"),
            "--out",
            str(results_path),
            "--concurrency",
            "3",
        ],
        capture_output=True,
        text=True,
        timeout=1700,
    )
    rows = [json.loads(line) for line in results_path.read_text().splitlines()] if results_path.exists() else []
    seen = {row["pageId"] for row in rows}
    for job in jobs:
        if job["pageId"] not in seen:
            rows.append({"pageId": job["pageId"], "error": f"worker exited {proc.returncode}: {proc.stderr[-300:]}"})
    target = Path("/data/adjudicate") / run / "captures"
    target.mkdir(parents=True, exist_ok=True)
    for row in rows:
        row["containerSeconds"] = round(time.time() - started, 1)
        if row.get("error"):
            continue
        capture = work / "store" / "captures" / row["captureId"]
        png = capture.with_suffix(".png").read_bytes()
        shutil.copy(capture.with_suffix(".png"), target / f"{row['captureId']}.png")
        shutil.copy(capture.with_suffix(".json"), target / f"{row['captureId']}.json")
        try:
            row["views"] = compose_views(png, row.get("located", {}))
        except Exception as error:  # a broken capture is a failed row, not a failed chunk
            row["viewError"] = str(error)[:200]
    volume.commit()
    return rows


@app.local_entrypoint()
def main(run_dir: str, limit: int = 0, chunk: int = 12) -> None:
    root = Path(run_dir)
    run = root.name
    out = root / "renders.jsonl"
    # A page is done when a render (or a failure) exists that located every marker its job now asks for.
    located: dict[str, set] = {}
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                located.setdefault(row["pageId"], set()).update(
                    row.get("located", {}) if not row.get("error") else {"*"}
                )
    jobs = [json.loads(line) for line in (root / "jobs.jsonl").read_text().splitlines() if line.strip()]
    done = {
        job["pageId"]
        for job in jobs
        if job["pageId"] in located and ("*" in located[job["pageId"]] or set(job["markers"]) <= located[job["pageId"]])
    }
    jobs = [job for job in jobs if job["pageId"] not in done]
    if limit:
        jobs = jobs[:limit]
    chunks = [jobs[i : i + chunk] for i in range(0, len(jobs), chunk)]
    print(json.dumps({"pages": len(jobs), "chunks": len(chunks), "alreadyDone": len(done)}))
    args = [
        (run, part, {job["pageId"]: (root / "jobs" / f"{job['pageId']}.html.gz").read_bytes() for job in part})
        for part in chunks
    ]
    views_dir = root / "views"
    totals = {"ok": 0, "failed": 0, "views": 0}
    started = time.time()
    with out.open("a") as handle:
        for rows in render_chunk.starmap(args, order_outputs=False):
            for row in rows:
                views = row.pop("views", {})
                if views:
                    page_dir = views_dir / row["pageId"]
                    page_dir.mkdir(parents=True, exist_ok=True)
                    for name, data in views.items():
                        (page_dir / f"{name}.jpg").write_bytes(data)
                    row["views"] = sorted(views)
                totals["failed" if row.get("error") else "ok"] += 1
                totals["views"] += len(views)
                handle.write(json.dumps(row) + "\n")
            handle.flush()
            print(json.dumps({**totals, "elapsed": round(time.time() - started)}), flush=True)
    print(json.dumps({"stage": "complete", **totals, "elapsed": round(time.time() - started)}))
