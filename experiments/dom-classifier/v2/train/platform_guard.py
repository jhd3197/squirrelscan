"""Per-platform precision guard for strict thresholds (#2307).

The strict thresholds hold P >= 0.95 on the old and the new corpus slice, but a
platform inside a slice can still fall short (Wix blog_post .915). For every
(class, platform) with >= --min-fires fires at the strict threshold and
precision < 0.95 this reports two remedies, from the same out-of-fold rows:

  bump     the lowest threshold above the strict one at which that platform's
           precision reaches 0.95 (>= 20 fires), applied to that platform only;
  abstain  the class never fires on that platform.

Each with the coverage it costs: on the platform, and for the class overall.
Report only: nothing here changes an artifact.

  python platform_guard.py --head pagetype --pagetype-dir <cv/pt82-new> \\
      --thresholds <slices.json> --tags tags.jsonl --out guard.json
  python platform_guard.py --head region|section --cv <cv dir> --page-list pages.json \\
      --thresholds <slices.json> --tags tags.jsonl --out guard.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cv_slices import platform_bucket

TARGET = 0.95


def load_pagetype(folder: Path, classes: list[str]):
    index = {c: i for i, c in enumerate(classes)}
    pages, y, pred, conf = [], [], [], []
    for line in (folder / "selection-predictions.jsonl").read_text().splitlines():
        row = json.loads(line)
        pages.append(row["pageId"])
        y.append(index[row["label"]])
        pred.append(index[row["pred"]])
        conf.append(row["conf"])
    return pages, np.asarray(y), np.asarray(pred), np.asarray(conf, np.float32)


def load_rows(cv: Path, page_list: Path):
    rows = np.load(cv / "rows.npz")
    keep = rows["evaluate"]
    oof = np.load(cv / "oof-new.npz")
    ids = json.loads(page_list.read_text())
    pages = [ids[i] for i in rows["page"][keep]]
    return (
        pages,
        rows["y"][keep].astype(np.int64),
        oof["pred"][keep].astype(np.int64),
        oof["conf"][keep].astype(np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", choices=("pagetype", "region", "section"), required=True)
    parser.add_argument("--pagetype-dir", default="")
    parser.add_argument("--cv", default="")
    parser.add_argument("--page-list", default="")
    parser.add_argument("--thresholds", required=True, help="cv_slices slices.json (strict)")
    parser.add_argument("--tags", required=True)
    parser.add_argument("--min-fires", type=int, default=50)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report = json.loads(Path(args.thresholds).read_text())
    classes, strict = report["classes"], np.asarray(report["strict"]["thresholds"])
    if args.head == "pagetype":
        pages, y, pred, conf = load_pagetype(Path(args.pagetype_dir), classes)
    else:
        pages, y, pred, conf = load_rows(Path(args.cv), Path(args.page_list))
    tags = {}
    for line in Path(args.tags).read_text().splitlines():
        row = json.loads(line)
        tags[row["p"]] = row.get("platform")
    platform = np.asarray([platform_bucket(tags.get(p)) for p in pages])
    out = []
    for i, name in enumerate(classes):
        if name == "none" or strict[i] > 1.0:
            continue
        fires_all = (pred == i) & (conf >= strict[i])
        support_all = int((y == i).sum())
        cov_all = int((fires_all & (y == i)).sum()) / max(support_all, 1)
        for bucket in sorted(set(platform)):
            on = platform == bucket
            fires = fires_all & on
            n = int(fires.sum())
            if n < args.min_fires:
                continue
            precision = float((y[fires] == i).mean())
            if precision >= TARGET:
                continue
            support = int((on & (y == i)).sum())
            cov = int((fires & (y == i)).sum()) / max(support, 1)
            bump = None
            for t in np.arange(round(float(strict[i]) + 0.01, 2), 1.0, 0.01):
                f = (pred == i) & (conf >= t) & on
                if int(f.sum()) < 20:
                    break
                if float((y[f] == i).mean()) >= TARGET:
                    kept = int((f & (y == i)).sum())
                    new_all = int((fires_all & ~on & (y == i)).sum()) + kept
                    bump = {
                        "threshold": round(float(t), 2),
                        "fires": int(f.sum()),
                        "precision": round(float((y[f] == i).mean()), 4),
                        "platformCoverage": round(kept / max(support, 1), 4),
                        "classCoverage": round(new_all / max(support_all, 1), 4),
                    }
                    break
            abstain_all = int((fires_all & ~on & (y == i)).sum()) / max(support_all, 1)
            out.append(
                {
                    "class": name,
                    "platform": bucket,
                    "strictThreshold": float(strict[i]),
                    "fires": n,
                    "precision": round(precision, 4),
                    "support": support,
                    "platformCoverage": round(cov, 4),
                    "classCoverage": round(cov_all, 4),
                    "bump": bump,
                    "abstain": {"platformCoverage": 0.0, "classCoverage": round(abstain_all, 4)},
                }
            )
    Path(args.out).write_text(
        json.dumps({"head": args.head, "minFires": args.min_fires, "flags": out}, indent=2) + "\n"
    )
    for row in out:
        bump = row["bump"]
        print(
            f"{args.head:8s} {row['class']:18s} {row['platform']:12s} P {row['precision']:.3f} "
            f"n {row['fires']:5d} cov {row['platformCoverage']:.2f} (class {row['classCoverage']:.2f})"
            f" | bump: "
            + (
                f"t {bump['threshold']} P {bump['precision']:.3f} cov "
                f"{bump['platformCoverage']:.2f} (class {bump['classCoverage']:.2f})"
                if bump
                else "none reaches .95"
            )
            + f" | abstain: class {row['abstain']['classCoverage']:.2f}"
        )


if __name__ == "__main__":
    main()
