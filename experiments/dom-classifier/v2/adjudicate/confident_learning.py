# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3"]
# ///
"""Confident-learning pass over the domain-grouped OOF predictions (#2307).

Flags train/val pages and nodes whose training label the out-of-fold model
confidently contradicts, plus every Jev/Clef page-type disagreement, and writes
a ranked candidate list for the visual adjudicator. Only train/val inputs are
read: the CV pool (cv_compare.py / train_pagetype_v8 --cv-folds output), the
page index filtered to train/val, platform tags, and the Jev/Clef pool files.
No holdout file is opened.

The CV runs saved argmax `pred` and `conf` per row, not the full probability
vector, so the confident joint is computed from the argmax only:

- per-class threshold t_j = mean p_j(x) over rows labelled j. For a row
  labelled j and predicted j, p_j = conf; for a row labelled j and predicted
  k != j, p_j is unknown but lies in [0, 1 - conf]. The midpoint is used, and
  the candidate counts under both bounds are reported.
- a row labelled i is off-diagonal (a candidate) when pred = k != i and
  conf >= t_k. When conf >= 0.5 the argmax is the only class that can clear its
  threshold, so this matches the full-vector rule whenever every t_j > 0.5.
- rank: conf (how sure the contradiction is), then margin = conf - t_k.

    uv run --script confident_learning.py --cc <2026-10-08-cc> --out <dir>
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

PLATFORM_BUCKETS = ("shopify", "woocommerce", "wordpress", "wix", "squarespace", "webflow", "nextjs")
PAGE_FOLD = {"event", "search_results", "local_business", "job_posting", "faq", "recipe"}


def platform_bucket(platform: str | None) -> str:
    if platform is None:
        return "unknown"
    return platform if platform in PLATFORM_BUCKETS else "custom"


def lang_bucket(lang: str | None) -> str:
    if not lang or lang == "unk":
        return "unk"
    return "en" if lang == "en" else "non-en"


def jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def page_rules(label: str, url: str) -> str:
    """policy-v8's own rules (root is homepage, /blog/ parent is blog_post), then the 12-class fold."""
    parts = urlparse(url)
    segments = [s for s in parts.path.lower().split("/") if s]
    fixed = label
    if not segments and not parts.query:
        fixed = "homepage"
    elif "blog" in segments[:-1] and label in ("docs", "article", "other"):
        fixed = "blog_post"
    return "other" if fixed in PAGE_FOLD else fixed


def thresholds(y: np.ndarray, pred: np.ndarray, conf: np.ndarray, k: int):
    """Per-class mean self-confidence: (low, mid, high) bounds from argmax-only OOF."""
    out = np.zeros((k, 3), np.float64)
    for j in range(k):
        rows = y == j
        if not rows.any():
            out[j] = np.nan
            continue
        hit = rows & (pred == j)
        miss = rows & (pred != j)
        known = conf[hit].astype(np.float64).sum()
        upper = (1.0 - conf[miss].astype(np.float64)).sum()
        n = rows.sum()
        out[j] = [known / n, (known + upper / 2) / n, (known + upper) / n]
    return out


def flag(y, pred, conf, t):
    """Off-diagonal of the argmax confident joint, under threshold column t."""
    tk = t[np.clip(pred, 0, None)]
    return (pred >= 0) & (pred != y) & (conf.astype(np.float64) >= tk)


def matrix(rows, classes):
    counts = Counter((r["given"], r["suggested"]) for r in rows)
    return {f"{a}->{b}": n for (a, b), n in counts.most_common()}


def tally(rows, key):
    by = defaultdict(Counter)
    for r in rows:
        by[r[key]][r["given"]] += 1
    return {k: dict(v.most_common()) for k, v in sorted(by.items())}


def rate_table(flagged_rows, all_meta, key):
    total = Counter(m[key] for m in all_meta)
    hit = Counter(r[key] for r in flagged_rows)
    return {
        k: {"flagged": hit.get(k, 0), "rows": total[k], "rate": round(hit.get(k, 0) / total[k], 4)}
        for k in sorted(total)
    }


def load_pages(cc: Path):
    index = {}
    for row in jsonl(cc / "silver-v4" / "page-index.jsonl.gz"):
        if row["s"] in ("train", "val"):
            index[row["p"]] = row
    new_pages = set((cc / "silver-v4" / "extension-pages.txt").read_text().split())
    tags = {}
    for row in jsonl(cc / "tags" / "tags-all.jsonl"):
        if row["p"] in index:
            tags[row["p"]] = row
    meta = {}
    for p, row in index.items():
        tag = tags.get(p, {})
        meta[p] = {
            "split": row["s"],
            "domain": row["domain"],
            "url": row["u"],
            "corpus": "new" if p in new_pages else "old",
            "platform": platform_bucket(tag.get("platform")),
            "lang": lang_bucket(tag.get("lang")),
        }
    return meta


def pagetype(cc: Path, pages: dict, out: Path):
    report = json.loads((cc / "cv" / "pt-new" / "pagetype-jev-report.json").read_text())
    classes = report["classes"]
    cid = {c: i for i, c in enumerate(classes)}
    rows = [r for r in jsonl(cc / "cv" / "pt-new" / "selection-predictions.jsonl") if r["pageId"] in pages]
    y = np.asarray([cid[r["label"]] for r in rows])
    pred = np.asarray([cid[r["pred"]] for r in rows])
    conf = np.asarray([r["conf"] for r in rows], np.float64)
    t = thresholds(y, pred, conf, len(classes))
    flags = {name: flag(y, pred, conf, t[:, i]) for i, name in enumerate(("low", "mid", "high"))}
    # Jev/Clef disagreements on the pool (Clef voted only on low-confidence Jev pages).
    jev = {}
    for split in ("train", "val"):
        for r in jsonl(cc / "labels" / "pagetype-v81" / f"{split}.jsonl"):
            if r["pageId"] in pages:
                jev[r["pageId"]] = r
    disagreements = {}
    for r in jsonl(cc / "labels" / "clef-v81" / "pool.jsonl"):
        p = r["pageId"]
        if p not in pages or p not in jev:
            continue
        url = pages[p]["url"]
        a = page_rules(jev[p]["pageType"], url)
        b = page_rules(r["choice"], url)
        if a != b:
            disagreements[p] = {"jev": a, "jevConf": jev[p]["confidence"], "clef": b, "clefConf": r["confidence"]}
    candidates = {}
    for i, r in enumerate(rows):
        if not flags["mid"][i]:
            continue
        k = pred[i]
        candidates[r["pageId"]] = {
            "given": r["label"],
            "suggested": r["pred"],
            "conf": round(float(conf[i]), 4),
            "threshold": round(float(t[k, 1]), 4),
            "margin": round(float(conf[i] - t[k, 1]), 4),
            "fitExcluded": r.get("fitExcluded", False),
            "sources": ["oof"],
        }
    for p, d in disagreements.items():
        entry = candidates.setdefault(
            p,
            {"given": d["jev"], "suggested": d["clef"], "conf": None, "threshold": None, "margin": None, "sources": []},
        )
        entry["sources"].append("jev_clef")
        entry["jev"], entry["jevConf"], entry["clef"], entry["clefConf"] = (
            d["jev"],
            d["jevConf"],
            d["clef"],
            d["clefConf"],
        )
    out_rows = []
    for p, entry in candidates.items():
        m = pages[p]
        out_rows.append({"pageId": p, **{k: m[k] for k in ("split", "corpus", "platform", "lang", "domain")}, **entry})
    # Both sources first, then OOF by confidence, then Jev/Clef-only by Clef confidence.
    out_rows.sort(key=lambda r: (-len(r["sources"]), -(r["conf"] or 0), -(r.get("clefConf") or 0)))
    for rank, r in enumerate(out_rows):
        r["rank"] = rank
    with (out / "pagetype-candidates.jsonl").open("w") as handle:
        for r in out_rows:
            handle.write(json.dumps(r) + "\n")
    # Every pool page with its training label, for stratified sampling outside the candidates.
    in_oof = {r["pageId"] for r in rows}
    with gzip.open(out / "pagetype-pool.jsonl.gz", "wt") as handle:
        for r in rows:
            handle.write(
                json.dumps({"pageId": r["pageId"], "given": r["label"], "pred": r["pred"], "conf": r["conf"]}) + "\n"
            )
        for p, d in disagreements.items():
            if p not in in_oof:
                handle.write(
                    json.dumps({"pageId": p, "given": d["jev"], "pred": None, "conf": None, "dropped": True}) + "\n"
                )
    all_meta = [pages[r["pageId"]] for r in rows]
    oof_rows = [r for r in out_rows if "oof" in r["sources"]]
    summary = {
        "classes": classes,
        "poolRows": len(rows),
        "thresholds": {c: [round(float(v), 4) for v in t[i]] for i, c in enumerate(classes)},
        "oofCandidates": {name: int(f.sum()) for name, f in flags.items()},
        "jevClefDisagreements": len(disagreements),
        "jevClefNotInOof": len(set(disagreements) - {r["pageId"] for r in rows}),
        "overlap": sum(1 for r in out_rows if len(r["sources"]) == 2),
        "total": len(out_rows),
        "givenToSuggested": matrix(oof_rows, classes),
        "jevToClef": dict(Counter(f"{d['jev']}->{d['clef']}" for d in disagreements.values()).most_common()),
        "byPlatform": rate_table(oof_rows, all_meta, "platform"),
        "byLang": rate_table(oof_rows, all_meta, "lang"),
        "byCorpus": rate_table(oof_rows, all_meta, "corpus"),
        "byGivenClass": {
            c: {"flagged": int(((y == i) & flags["mid"]).sum()), "rows": int((y == i).sum())}
            for i, c in enumerate(classes)
        },
        "platformByGiven": tally(out_rows, "platform"),
    }
    return summary


def section_meta(cc: Path, rows_npz):
    """Rebuild cv_compare's kept meta rows (same filter) to recover src per row."""
    sample = cc / "section-sample-v4"
    meta = [json.loads(line) for line in (sample / "meta-train.jsonl").read_text().splitlines()]
    n_train = len(meta)
    meta += [json.loads(line) for line in (sample / "meta-val.jsonl").read_text().splitlines()]
    drop = {(r["pageId"], r["nodeId"]) for r in jsonl(cc / "section-v4" / "drop-rows.jsonl")}
    kept = [m for i, m in enumerate(meta) if (m["p"], m["n"]) not in drop and (i < n_train or m["src"] == "jev")]
    node = rows_npz["node"]
    if len(kept) != len(node) or any(m["n"] != int(n) for m, n in zip(kept, node, strict=True)):
        raise SystemExit(f"section meta does not align with rows.npz ({len(kept)} vs {len(node)})")
    jev = np.asarray([m["src"] == "jev" for m in kept])
    if not np.array_equal(jev, rows_npz["evaluate"]):
        raise SystemExit("section meta src=jev does not match rows.npz evaluate")
    return kept


def section(cc: Path, pages: dict, out: Path):
    report = json.loads((cc / "cv" / "section" / "cv-report.json").read_text())
    classes = report["classes"]
    # NpzFile decompresses on every key access: materialise each array once.
    rows_npz = dict(np.load(cc / "cv" / "section" / "rows.npz"))
    oof = dict(np.load(cc / "cv" / "section" / "oof-new.npz"))
    kept = section_meta(cc, rows_npz)
    page_ids = json.loads((cc / "section-sample-v4" / "pages.json").read_text())
    by_index = np.asarray([page_ids[i] for i in rows_npz["page"]])
    if any(m["p"] != p for m, p in zip(kept, by_index, strict=True)):
        raise SystemExit("section pages.json index does not match meta pageIds")
    y = rows_npz["y"].astype(np.int64)
    pred = oof["pred"].astype(np.int64)
    conf = oof["conf"].astype(np.float64)
    in_pool = np.asarray([p in pages for p in by_index])
    src = np.asarray([m["src"] for m in kept])
    t = thresholds(y[in_pool], pred[in_pool], conf[in_pool], len(classes))
    flags = {name: flag(y, pred, conf, t[:, i]) & in_pool for i, name in enumerate(("low", "mid", "high"))}
    out_rows = []
    for i in np.flatnonzero(flags["mid"]):
        p = by_index[i]
        m = pages[p]
        k = pred[i]
        out_rows.append(
            {
                "pageId": p,
                "nodeId": int(rows_npz["node"][i]),
                "src": src[i],
                **{key: m[key] for key in ("split", "corpus", "platform", "lang", "domain")},
                "given": classes[y[i]],
                "suggested": classes[k],
                "conf": round(float(conf[i]), 4),
                "threshold": round(float(t[k, 1]), 4),
                "margin": round(float(conf[i] - t[k, 1]), 4),
            }
        )
    # Jev rows first (the labels the head is scored on), then by confidence.
    out_rows.sort(key=lambda r: (r["src"] != "jev", -r["conf"], -r["margin"]))
    for rank, r in enumerate(out_rows):
        r["rank"] = rank
    with (out / "section-candidates.jsonl").open("w") as handle:
        for r in out_rows:
            handle.write(json.dumps(r) + "\n")
    pool_idx = np.flatnonzero(in_pool)
    with gzip.open(out / "section-pool.jsonl.gz", "wt") as handle:
        for i in pool_idx:
            if src[i] == "jev":
                handle.write(
                    json.dumps(
                        {
                            "pageId": by_index[i],
                            "nodeId": int(rows_npz["node"][i]),
                            "given": classes[y[i]],
                            "pred": classes[pred[i]],
                            "conf": round(float(conf[i]), 4),
                        }
                    )
                    + "\n"
                )
    all_meta = [{**pages[by_index[i]], "src": src[i]} for i in pool_idx]
    jev_rows = [r for r in out_rows if r["src"] == "jev"]
    jev_meta = [m for m in all_meta if m["src"] == "jev"]
    return {
        "classes": classes,
        "poolRows": int(in_pool.sum()),
        "rowsBySrc": dict(Counter(src[in_pool]).most_common()),
        "thresholds": {c: [round(float(v), 4) for v in t[i]] for i, c in enumerate(classes)},
        "candidates": {name: int(f.sum()) for name, f in flags.items()},
        "candidatesBySrc": dict(Counter(r["src"] for r in out_rows).most_common()),
        "jevGivenToSuggested": matrix(jev_rows, classes),
        "otherSrcGivenToSuggested": matrix([r for r in out_rows if r["src"] != "jev"], classes),
        "jevByPlatform": rate_table(jev_rows, jev_meta, "platform"),
        "jevByLang": rate_table(jev_rows, jev_meta, "lang"),
        "jevByCorpus": rate_table(jev_rows, jev_meta, "corpus"),
        "jevByGivenClass": {
            c: {
                "flagged": sum(1 for r in jev_rows if r["given"] == c),
                "rows": sum(1 for i in pool_idx if src[i] == "jev" and y[i] == j),
            }
            for j, c in enumerate(classes)
        },
    }


def region(cc: Path, pages: dict):
    """Counts only: rows.npz carries no node id for region, so no node-level list."""
    report = json.loads((cc / "cv" / "region" / "cv-report.json").read_text())
    classes = report["classes"]
    rows_npz = dict(np.load(cc / "cv" / "region" / "rows.npz"))
    oof = dict(np.load(cc / "cv" / "region" / "oof-new.npz"))
    page_ids = json.loads((cc / "region-sample-v4" / "pages.json").read_text())
    evaluate = rows_npz["evaluate"]
    y = rows_npz["y"].astype(np.int64)[evaluate]
    pred = oof["pred"].astype(np.int64)[evaluate]
    conf = oof["conf"].astype(np.float64)[evaluate]
    page_idx = rows_npz["page"][evaluate]
    t = thresholds(y, pred, conf, len(classes))
    f = flag(y, pred, conf, t[:, 1])
    plat = np.asarray([pages.get(page_ids[i], {}).get("platform", "unknown") for i in page_idx])
    pairs = Counter(zip(y[f].tolist(), pred[f].tolist(), strict=True))
    by_platform = {}
    for name in sorted(set(plat)):
        mask = plat == name
        by_platform[name] = {
            "flagged": int((f & mask).sum()),
            "rows": int(mask.sum()),
            "rate": round(float((f & mask).sum() / max(mask.sum(), 1)), 4),
        }
    pages_flagged = Counter(page_idx[f].tolist())
    return {
        "classes": classes,
        "evaluatedRows": int(evaluate.sum()),
        "thresholds": {c: [round(float(v), 4) for v in t[i]] for i, c in enumerate(classes)},
        "candidates": int(f.sum()),
        "givenToSuggested": {f"{classes[a]}->{classes[b]}": n for (a, b), n in pairs.most_common(25)},
        "byPlatform": by_platform,
        "pagesWithFlags": len(pages_flagged),
        "note": "region rows.npz has no node id: node-level candidates need cv_compare to save it",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-region", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pages = load_pages(args.cc)
    summary = {"pages": len(pages), "pagetype": pagetype(args.cc, pages, args.out)}
    summary["section"] = section(args.cc, pages, args.out)
    if not args.skip_region:
        summary["region"] = region(args.cc, pages)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    pt, sec = summary["pagetype"], summary["section"]
    print(
        json.dumps(
            {
                "pages": len(pages),
                "pagetype": {
                    k: pt[k] for k in ("poolRows", "oofCandidates", "jevClefDisagreements", "overlap", "total")
                },
                "section": {k: sec[k] for k in ("poolRows", "candidates", "candidatesBySrc")},
                "region": {k: summary.get("region", {}).get(k) for k in ("evaluatedRows", "candidates")},
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
