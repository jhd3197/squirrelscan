"""Sentence embeddings of page text on Modal (#2307): an upper-bound probe for the text channel.

    modal run modal_embed.py --texts texts.jsonl --out embeddings.npz [--model intfloat/multilingual-e5-small]

Input rows {id, text}; output npz with `ids` and float16 `x` (n, 384). One container, CPU. Nothing is written to a
volume; the text goes up as an argument and the vectors come back. Page text: PRIVATE.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.5.1", "sentence-transformers==3.3.1", "numpy", extra_options="--extra-index-url https://download.pytorch.org/whl/cpu"
)
app = modal.App("treetop-embed", image=image)


@app.function(cpu=8, memory=16384, timeout=3600)
def embed(rows_json: str, model_name: str) -> bytes:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    rows = [json.loads(line) for line in rows_json.splitlines() if line.strip()]
    model = SentenceTransformer(model_name, device="cpu")
    model.max_seq_length = 256
    prefix = "query: " if "e5" in model_name else ""
    vectors = model.encode([prefix + r["text"] for r in rows], batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, ids=np.asarray([r["id"] for r in rows]), x=vectors.astype("float16"))
    return buffer.getvalue()


@app.local_entrypoint()
def main(texts: str, out: str, model: str = "intfloat/multilingual-e5-small") -> None:
    blob = embed.remote(Path(texts).read_text(), model)
    Path(out).write_bytes(blob)
    print(f"wrote {out}: {len(blob) / 1e6:.1f} MB")
