"""Slice the old vs old+new CV comparison by corpus, platform and language.

Reads the out-of-fold predictions both arms wrote (page type:
selection-predictions.jsonl from two train_pagetype_v8 --cv-folds runs, one with
--fit-exclude; region / section: cv_compare.py output) and the page tags
(cc_corpus.py tag-pages: platform + language per page id). Every arm is read at
ITS OWN thresholds, picked on its own OOF rows, so a delta is what swapping
the shipped corpus for the extended one changes. No holdout anywhere.

For each slice and class: support, then per arm precision at threshold (P@t),
coverage at threshold (share of the class's rows that fire correctly) and
fires. Slices: all, corpus old/new, platform bucket, language bucket.

  python cv_slices.py pagetype --old <dir> --new <dir> --tags tags.jsonl \
      --new-pages new-pages.txt --out report.json
  python cv_slices.py rows --cv <cv_compare out> --tags tags.jsonl \
      --new-pages new-pages.txt --out report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PLATFORM_BUCKETS = (
    "shopify",
    "woocommerce",
    "wordpress",
    "wix",
    "squarespace",
    "webflow",
    "nextjs",
)


def platform_bucket(platform: str | None) -> str:
    if platform is None:
        return "unknown"
    return platform if platform in PLATFORM_BUCKETS else "custom"


def lang_bucket(lang: str | None) -> str:
    if not lang or lang == "unk":
        return "unk"
    return "en" if lang == "en" else "non-en"


def jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def slice_masks(pages: list[str], tags: dict, new_pages: set[str]) -> dict[str, np.ndarray]:
    corpus = np.asarray(["new" if p in new_pages else "old" for p in pages])
    platform = np.asarray([platform_bucket(tags.get(p, {}).get("platform")) for p in pages])
    lang = np.asarray([lang_bucket(tags.get(p, {}).get("lang")) for p in pages])
    out = {"all": np.ones(len(pages), bool)}
    for name in ("old", "new"):
        out[f"corpus:{name}"] = corpus == name
    for name in (*PLATFORM_BUCKETS, "custom", "unknown"):
        out[f"platform:{name}"] = platform == name
    for name in ("en", "non-en", "unk"):
        out[f"lang:{name}"] = lang == name
    return out


def per_class(
    y: np.ndarray,
    pred: np.ndarray,
    conf: np.ndarray,
    thresholds: np.ndarray,
    classes: list[str],
    mask: np.ndarray,
) -> dict:
    fires = conf >= thresholds[np.clip(pred, 0, None)]
    out = {}
    for i, name in enumerate(classes):
        support = int((mask & (y == i)).sum())
        fired = mask & fires & (pred == i)
        n = int(fired.sum())
        correct = int((fired & (y == i)).sum())
        out[name] = {
            "support": support,
            "fires": n,
            "precision": round(correct / n, 4) if n else None,
            "coverage": round(correct / support, 4) if support else None,
        }
    covered = mask & fires
    out["_overall"] = {
        "rows": int(mask.sum()),
        "firing": int(covered.sum()),
        "accuracyWhenFiring": round(float((pred[covered] == y[covered]).mean()), 4)
        if covered.any()
        else None,
    }
    return out


def compare(y, arms: dict, classes, pages, tags, new_pages, skip=()) -> dict:
    masks = slice_masks(pages, tags, new_pages)
    report = {}
    for name, mask in masks.items():
        if not mask.any():
            continue
        report[name] = {
            arm: per_class(y, pred, conf, thr, classes, mask)
            for arm, (pred, conf, thr) in arms.items()
        }
        for c in classes:
            if c in skip:
                continue
            row = {arm: report[name][arm][c] for arm in arms}
            old, new = row.get("old"), row.get("new")
            if old and new and old["coverage"] is not None and new["coverage"] is not None:
                report[name].setdefault("delta", {})[c] = {
                    "coverage": round(new["coverage"] - old["coverage"], 4),
                    "precision": (
                        round(new["precision"] - old["precision"], 4)
                        if new["precision"] is not None and old["precision"] is not None
                        else None
                    ),
                }
    return report


def run_pagetype(args) -> dict:
    arms = {}
    rows: dict[str, dict] = {}
    classes = None
    for arm, folder in (("old", Path(args.old)), ("new", Path(args.new))):
        thr = json.loads((folder / "class-thresholds.json").read_text())
        classes = classes or thr["classes"]
        if thr["classes"] != classes:
            raise SystemExit("arms disagree on the class list")
        for r in jsonl(folder / "selection-predictions.jsonl"):
            rows.setdefault(r["pageId"], {"label": r["label"]})[arm] = (r["pred"], r["conf"])
        arms[arm] = np.asarray(thr["thresholds"])
    # Only pages both arms scored (the old arm predicts every pool page too).
    pages = sorted(p for p, r in rows.items() if "old" in r and "new" in r)
    index = {c: i for i, c in enumerate(classes)}
    y = np.asarray([index[rows[p]["label"]] for p in pages])
    packed = {
        arm: (
            np.asarray([index[rows[p][arm][0]] for p in pages]),
            np.asarray([rows[p][arm][1] for p in pages]),
            arms[arm],
        )
        for arm in arms
    }
    return {
        "head": "pagetype",
        "classes": classes,
        "pages": len(pages),
        "slices": compare(y, packed, classes, pages, load_tags(args), load_new(args)),
    }


def run_rows(args) -> dict:
    cv = Path(args.cv)
    report = json.loads((cv / "cv-report.json").read_text())
    classes = report["classes"]
    rows = np.load(cv / "rows.npz")
    keep = rows["evaluate"]
    page_ids = json.loads(Path(args.page_list).read_text())
    pages = [page_ids[i] for i in rows["page"][keep]]
    y = rows["y"][keep].astype(np.int64)
    arms = {}
    for arm, info in report["arms"].items():
        oof = np.load(cv / f"oof-{arm}.npz")
        arms[arm] = (
            oof["pred"][keep].astype(np.int64),
            oof["conf"][keep].astype(np.float32),
            np.asarray(info["thresholds"]),
        )
    skip = ("none",) if report["head"] == "section" else ()
    new_pages = load_new(args)
    # The old arm stands for the shipped pipeline, which never saw a new page:
    # its thresholds are re-picked on OLD pages only, by the head's own rule.
    # (cv_compare picks both arms' thresholds on the whole evaluation set.)
    old_rows = np.asarray([p not in new_pages for p in pages])
    pred, conf, pooled = arms["old"]
    arms["old"] = (pred, conf, pick_like_head(report, pred[old_rows], conf[old_rows], y[old_rows]))
    return {
        "head": report["head"],
        "classes": classes,
        "rows": int(keep.sum()),
        "oldThresholdsOn": "old pages only",
        "thresholds": {
            "old": arms["old"][2].tolist(),
            "oldPooled": pooled.tolist(),
            "new": arms["new"][2].tolist(),
        },
        "slices": compare(y, arms, classes, pages, load_tags(args), new_pages, skip),
    }


def pick_like_head(report: dict, pred: np.ndarray, conf: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Thresholds by the rule the head's trainer uses, from (argmax, max prob)."""
    classes = report["classes"]
    probs = np.zeros((len(pred), len(classes)), np.float32)
    probs[np.arange(len(pred)), pred] = conf
    if report["head"] == "region":
        from class_thresholds import sweep

        return np.asarray([sweep(probs, y, i)["threshold"] for i in range(len(classes))])
    from train_section import pick_thresholds

    out = pick_thresholds(probs, y, len(classes), 0.95, 15)
    out[classes.index("none")] = 0.0
    return np.asarray(out)


def load_tags(args) -> dict:
    return {r["p"]: r for r in jsonl(Path(args.tags))}


def load_new(args) -> set[str]:
    return {line.strip() for line in Path(args.new_pages).read_text().splitlines() if line.strip()}


def fmt(cell: dict | None) -> str:
    if not cell or cell.get("support", 0) == 0:
        return "–"
    p = "–" if cell["precision"] is None else f"{cell['precision']:.3f}"
    return f"{p}/{cell['coverage']:.2f}"


def markdown(report: dict, slices: list[str], min_support: int) -> str:
    """One table per slice: per class P@t/coverage, old -> new, and the
    coverage delta; classes under --min-support rows in the slice are skipped."""
    out = [f"## {report['head']} (old vs old+new, OOF, each arm at its own thresholds)", ""]
    for name in slices:
        block = report["slices"].get(name)
        if not block:
            continue
        overall = {arm: block[arm]["_overall"] for arm in ("old", "new")}
        out.append(
            f"### {name}: rows {overall['new']['rows']}, firing old {overall['old']['firing']} "
            f"/ new {overall['new']['firing']}, accuracy when firing "
            f"{overall['old']['accuracyWhenFiring']} -> {overall['new']['accuracyWhenFiring']}"
        )
        out.append("")
        out.append("| class | support | old P@t/cov | new P@t/cov | Δcov |")
        out.append("|---|---|---|---|---|")
        for cls in report["classes"]:
            old, new = block["old"].get(cls), block["new"].get(cls)
            if not old or old["support"] < min_support or cls == "none":
                continue
            delta = block.get("delta", {}).get(cls, {}).get("coverage")
            out.append(
                f"| {cls} | {old['support']} | {fmt(old)} | {fmt(new)} | "
                f"{'' if delta is None else f'{delta:+.2f}'} |"
            )
        out.append("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("table")
    p.add_argument("--report", required=True)
    p.add_argument(
        "--slices",
        default="all,corpus:old,corpus:new,platform:shopify,"
        "platform:woocommerce,platform:wordpress,platform:wix,platform:squarespace,"
        "platform:webflow,platform:nextjs,platform:custom,lang:en,lang:non-en",
    )
    p.add_argument("--min-support", type=int, default=30)
    p = sub.add_parser("pagetype")
    p.add_argument("--old", required=True)
    p.add_argument("--new", required=True)
    p = sub.add_parser("rows")
    p.add_argument("--cv", required=True)
    p.add_argument("--page-list", required=True, help="pages.json: page index -> page id")
    for name, p in sub.choices.items():
        if name == "table":
            continue
        p.add_argument("--tags", required=True)
        p.add_argument("--new-pages", required=True)
        p.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.cmd == "table":
        report = json.loads(Path(args.report).read_text())
        print(markdown(report, args.slices.split(","), args.min_support))
        return
    report = run_pagetype(args) if args.cmd == "pagetype" else run_rows(args)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"head": report["head"], "slices": list(report["slices"])}))


if __name__ == "__main__":
    main()
