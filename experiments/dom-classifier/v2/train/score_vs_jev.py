"""Score the CURRENT (silver-trained) page-type head against Jev's labels.

This is the honest "before" number for the page-type rebuild. The current head
was trained on labels derived from JSON-LD/microdata/og:type; Jev labelled the
same holdout pages under the frozen v7 definitions. Two things are reported
separately, because they are two different questions:

  schema-present  pages whose silver label existed. This is the population the
                  head was trained and previously scored on.
  schema-less     pages silver could not label at all. This is the population
                  the product rule actually fires on, and the one where the
                  hand inspection put precision at 0.30 rather than 0.91.

Jev's labels carry 11 classes; the model has 7 and can never emit `legal`,
`pricing`, `homepage` or `product_listing`. Those are counted as misses against
Jev rather than quietly excluded, because "the model cannot say this" is the
finding.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp

#: The dataset directory under --data-root. The 2026-09-19 build is `silver-v1`;
#: a larger corpus is built beside it rather than over it, and every script in
#: this directory has to agree on which one is being read.
DATASET = os.environ.get("DOM_CLASSIFIER_DATASET_VERSION", "silver-v1")


MODEL_CLASSES = ["article", "listing_index", "docs", "product_detail", "about", "contact", "other"]
UNREACHABLE = ["legal", "pricing", "homepage", "product_listing"]


def load_pages(path: Path, n_numeric: int, n_tokens: int):
    """Every holdout page, labelled or not, with its silver label kept."""
    rows, ids, silver = [], [], []
    width = n_numeric + n_tokens
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            vector = np.zeros(width, dtype=np.float32)
            vector[:n_numeric] = row["f"]
            for bucket in row["k"]:
                vector[n_numeric + bucket] = 1.0
            rows.append(vector)
            ids.append(row["p"])
            silver.append(row["l"])
    return sp.csr_matrix(np.vstack(rows)), ids, silver


def metrics(pairs: list[tuple[str, str]], classes: list[str]) -> dict:
    """Per-class P/R/F1 over (truth, predicted) pairs, truth = Jev."""
    out, f1s = {}, []
    for name in classes:
        tp = sum(1 for t, p in pairs if t == name and p == name)
        fp = sum(1 for t, p in pairs if t != name and p == name)
        fn = sum(1 for t, p in pairs if t == name and p != name)
        support = sum(1 for t, _ in pairs if t == name)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[name] = {"precision": round(precision, 4), "recall": round(recall, 4),
                     "f1": round(f1, 4), "support": support,
                     "reachableByModel": name in MODEL_CLASSES}
        if support:
            f1s.append(f1)
    agree = sum(1 for t, p in pairs if t == p)
    return {"perClass": out, "macroF1": round(float(np.mean(f1s)), 4) if f1s else 0.0,
            "agreement": round(agree / len(pairs), 4) if pairs else 0.0, "n": len(pairs)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--jev", required=True)
    args = parser.parse_args()

    import lightgbm as lgb

    root = Path(args.data_root)
    schema = json.loads((root / DATASET / "feature-schema.json").read_text())
    report = json.loads((Path(args.models) / "pagetype" / "pagetype-report.json").read_text())

    x, ids, silver = load_pages(root / DATASET / "pages-holdout.jsonl.gz",
                                schema["pageFeatureCount"], schema["tokenBuckets"])
    booster = lgb.Booster(model_file=str(Path(args.models) / "pagetype" / "pagetype-model.txt"))
    probs = booster.predict(x, num_iteration=report["shipped"]["iterations"])
    predicted = [MODEL_CLASSES[i] for i in probs.argmax(axis=1)]
    top = probs.max(axis=1)

    jev = {}
    for line in (Path(args.jev) / "holdout.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        jev[row["pageId"]] = row["pageType"]

    present, absent, missing = [], [], 0
    for page_id, pred, sil in zip(ids, predicted, silver, strict=True):
        truth = jev.get(page_id)
        if truth is None:
            missing += 1
            continue
        (absent if sil == "unlabelled" else present).append((truth, pred))

    all_classes = MODEL_CLASSES + UNREACHABLE
    out = {
        "note": "truth = Jev v7 labels; the model is the CURRENT silver-trained head",
        "modelClasses": MODEL_CLASSES,
        "classesTheModelCannotEmit": UNREACHABLE,
        "pagesWithoutAJevLabel": missing,
        "schemaPresent": metrics(present, all_classes),
        "schemaLess": metrics(absent, all_classes),
        "overall": metrics(present + absent, all_classes),
        "jevLabelDistribution": {
            "schemaPresent": dict(Counter(t for t, _ in present).most_common()),
            "schemaLess": dict(Counter(t for t, _ in absent).most_common()),
        },
        "shareOfSchemaLessInUnreachableClasses": round(
            sum(1 for t, _ in absent if t in UNREACHABLE) / max(len(absent), 1), 4
        ),
    }

    # The product question again, now against Jev rather than silver: of the
    # schema-less pages the head calls `article` at high confidence, how many
    # does Jev agree with?
    for threshold in (0.7124, 0.9):
        picked = [
            (jev[i], p)
            for i, p, s, c in zip(ids, predicted, silver, top, strict=True)
            if s == "unlabelled" and p == "article" and c >= threshold and i in jev
        ]
        out[f"articleAtConfidence{threshold}"] = {
            "n": len(picked),
            "precisionAgainstJev": round(
                sum(1 for t, _ in picked if t == "article") / max(len(picked), 1), 4
            ),
            "whatTheyReallyAre": dict(Counter(t for t, _ in picked).most_common()),
        }

    path = Path(args.models) / "pagetype" / "vs-jev-holdout.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "jevLabelDistribution"}, indent=2))


if __name__ == "__main__":
    main()
