"""Page type v8: "this page should carry schema.org type X", trained to abstain.

Labels, one per page, first source that has one:
  1. Jev v8 (noise sample + split files), weight = Jev's confidence;
  2. Jev v7, only for the classes v8 keeps 1:1 (schema_labels_v8.V7_REUSED);
  3. the page's own schema (schema_labels_v8.py), only for classes whose free
     label agreed with Jev v8 on the noise sample at >= --free-min-agreement,
     weight = that agreement, and never on holdout (holdout is Jev only).
The featurizer never reads schema markup, so the schema-labelled pages are
trained with their schema hidden by construction.

Splits are the dataset's (domain-grouped, same holdout domains as silver-v3).
Train on train, early-stop and pick the config on val, pick per-class
thresholds on val for precision >= --target-precision, report holdout once.

Also reported on holdout:
  - schemaHidden: pages whose declared schema agrees with Jev v8, i.e. "correct
    schema", scored with that schema hidden (it always is);
  - vsV7: the v7 head on the classes both taxonomies share.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import hashlib
import re

import numpy as np
import scipy.sparse as sp

from schema_labels_v8 import V7_REUSED

N_JOBS = int(os.environ.get("DOM_CLASSIFIER_JOBS", "4"))
#: Schema family a class recommends: two classes in one family recommend
#: compatible markup (BlogPosting is an Article), so confusing them is not a
#: wrong recommendation.
FAMILY = {"article": "Article", "blog_post": "Article"}
V8_CLASSES = [
    "homepage", "product_detail", "product_listing", "article", "blog_post", "faq", "recipe",
    "event", "job_posting", "local_business", "about", "contact", "docs", "legal", "pricing",
    "search_results", "listing_index", "other",
]
# The v7 head's own class order (the artifact depends on it), NOT alphabetical:
# indexing a v7 prediction with any other order silently scores the wrong class.
from train_pagetype_jev import CLASSES as V7_CLASSES  # noqa: E402


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_features(dataset: Path, n_numeric: int, n_tokens: int) -> dict[str, dict]:
    out: dict[str, dict] = {}
    width = n_numeric + n_tokens
    for split in ("train", "val", "holdout"):
        with gzip.open(dataset / f"pages-{split}.jsonl.gz", "rt") as handle:
            for line in handle:
                row = json.loads(line)
                vector = np.zeros(width, dtype=np.float32)
                vector[:n_numeric] = row["f"]
                for bucket in row["k"]:
                    vector[n_numeric + bucket] = 1.0
                out[row["p"]] = {"x": vector, "d": row["d"], "s": split}
    return out


def per_class(y: np.ndarray, pred: np.ndarray, fires: np.ndarray, classes: list[str]) -> dict:
    """Argmax F1 per class, plus precision/coverage when only `fires` rows answer."""
    out, f1s = {}, []
    for i, name in enumerate(classes):
        support = int((y == i).sum())
        tp = int(((pred == i) & (y == i)).sum())
        fp = int(((pred == i) & (y != i)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        fired = fires & (pred == i)
        n_fired = int(fired.sum())
        out[name] = {
            "support": support, "f1": round(f1, 4), "precision": round(precision, 4),
            "recall": round(recall, 4),
            "fires": n_fired,
            "precisionAtThreshold": round(float((y[fired] == i).mean()), 4) if n_fired else None,
            "coverageAtThreshold": round(int((y[fired] == i).sum()) / support, 4) if support else None,
        }
        if support:
            f1s.append(f1)
    return {"perClass": out, "macroF1": round(float(np.mean(f1s)), 4) if f1s else 0.0,
            "accuracy": round(float((pred == y).mean()), 4) if len(y) else 0.0,
            "coverage": round(float(fires.mean()), 4) if len(y) else 0.0,
            "accuracyWhenFiring": round(float((pred[fires] == y[fires]).mean()), 4)
            if fires.any() else None, "n": int(len(y))}


def pick_thresholds(probs: np.ndarray, y: np.ndarray, classes: list[str], target: float,
                    min_fires: int) -> list[float]:
    top = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    out = []
    for i in range(len(classes)):
        chosen = 1.01
        for t in np.arange(0.3, 1.0, 0.01):
            fires = (top == i) & (conf >= t)
            if int(fires.sum()) < min_fires:
                break
            if float((y[fires] == i).mean()) >= target:
                chosen = round(float(t), 2)
                break
        out.append(chosen)
    return out


def pick_thresholds_lcb(probs: np.ndarray, y: np.ndarray, folds: np.ndarray, classes: list[str],
                        target: float, min_fires: int) -> list[float]:
    """Lowest threshold whose per-fold precision, mean minus one standard
    deviation, clears `target`: a margin sized by how much precision moves from
    one set of domains to another, measured without touching holdout."""
    top = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    out = []
    for i in range(len(classes)):
        chosen = 1.01
        for t in np.arange(0.3, 1.0, 0.01):
            fires = (top == i) & (conf >= t)
            if int(fires.sum()) < min_fires:
                break
            per_fold = [float((y[fires & (folds == f)] == i).mean())
                        for f in np.unique(folds) if int((fires & (folds == f)).sum()) >= 5]
            if len(per_fold) >= 3 and np.mean(per_fold) - np.std(per_fold) >= target:
                chosen = round(float(t), 2)
                break
        out.append(chosen)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", default="silver-v3")
    parser.add_argument("--jev-v8", required=True)
    parser.add_argument("--jev-v7", required=True)
    parser.add_argument("--schema-labels", required=True)
    parser.add_argument("--v7-model", default="", help="v7 page-type head to compare against")
    parser.add_argument("--merge", default="", help="rare:parent,... applied to every source")
    parser.add_argument("--free-min-agreement", type=float, default=0.7)
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument("--min-fires", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--extra-features", action="append", default=[],
                        help="jsonl of {p, f:[...]} appended to every page's vector (a page "
                             "missing from the file gets zeros); repeatable")
    parser.add_argument("--jev-override", action="append", default=[],
                        help="later Jev label files (e.g. a v8.1 relabel of a model-independent "
                             "page set) that replace the v8 label for their pages; repeatable")
    parser.add_argument("--drop-disagreeing", default="",
                        help="Jev relabel file: pool (train/val) pages whose label there differs "
                             "from their v8 label are dropped from TRAINING as ambiguous; "
                             "holdout labels are untouched")
    parser.add_argument("--policy-rules", action="store_true",
                        help="correct Jev labels that contradict policy-v8's own decision rules, "
                             "on every split alike (model-independent): the site root without "
                             "a query is `homepage`; a page under a /blog/ path segment that Jev "
                             "called docs/article/other is `blog_post` (rule 1: presentation wins)")
    parser.add_argument("--policy-version", choices=("v8", "v8.2"), default="v8",
                        help="which mechanical rules --policy-rules applies: v8 (site root, "
                             "/blog/) or v8.2 (adds locale roots -> homepage and Wix /post/ "
                             "-> blog_post), via adjudicate/policy_rules.page_rules_v82 so the "
                             "trainer and the adjudicator apply byte-identical rules")
    parser.add_argument("--platform-tags", default="",
                        help="jsonl {p, platform} (corpus/cc_corpus.py tag-pages); v8.2 rule 1 "
                             "(Wix /post/) needs it")
    parser.add_argument("--cv-folds", type=int, default=0,
                        help="pick thresholds on domain-grouped out-of-fold predictions over "
                             "train+val and ship a model refit on train+val; 0 = val split")
    parser.add_argument("--template-stack", default="",
                        help="jsonl {p, cluster}: with --cv-folds, add each page's leave-one-out "
                             "mean stage-1 probability over the other pages of its template "
                             "cluster, then refit (stage 2)")
    parser.add_argument("--dropout-extra", type=float, default=0.0,
                        help="share of train/val pages whose --extra-features block is set to "
                             "MISSING (NaN, LightGBM-native), so the model also works without "
                             "the link graph; holdout is then scored with and without it")
    parser.add_argument("--lcb", action="store_true",
                        help="with --cv-folds: a threshold must give mean-1sd of the per-fold "
                             "precision >= target, not just pooled precision (fold-variance margin)")
    parser.add_argument("--fit-exclude", default="",
                        help="file of page ids (one per line) that no model in this run is FIT "
                             "on (grid, CV folds, refit) and that thresholds are not picked on, "
                             "but that are still predicted out-of-fold and dumped: the "
                             "old-corpus arm of an old vs old+new comparison")
    parser.add_argument("--drop-pages", action="append", default=[],
                        help="file of page ids whose label is too uncertain to use (e.g. a "
                             "Jev/Clef disagreement): left out of every split; repeatable")
    parser.add_argument("--dump-selection", action="store_true",
                        help="write the selection split's per-page predictions (OOF with "
                             "--cv-folds) for error audits; never holdout")
    parser.add_argument("--dump-probs", action="store_true",
                        help="with --dump-selection: also write selection-probs.npy, float16 "
                             "per-class OOF probabilities in selection-predictions.jsonl row "
                             "order (columns = class-thresholds.json classes)")
    parser.add_argument("--score-holdout", action="store_true",
                        help="score holdout (only for a KEPT candidate; the loop counts touches)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import lightgbm as lgb

    root = Path(args.data_root)
    dataset = root / args.dataset
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merge = dict(pair.split(":") for pair in args.merge.split(",") if pair)
    classes = [c for c in V8_CLASSES if c not in merge]
    index = {c: i for i, c in enumerate(classes)}
    fold = lambda label: merge.get(label, label)  # noqa: E731

    schema = {r["pageId"]: r["pageType"] for r in jsonl(Path(args.schema_labels))}
    jev_dir = Path(args.jev_v8)
    jev8: dict[str, dict] = {}
    for name in ("noise", "holdout", "val", "train"):
        for row in jsonl(jev_dir / f"{name}.jsonl"):
            jev8[row["pageId"]] = row
    ambiguous: set[str] = set()
    if args.drop_disagreeing:
        for row in jsonl(Path(args.drop_disagreeing)):
            first = jev8.get(row["pageId"])
            if first is not None and first["pageType"] != row.get("pageType"):
                ambiguous.add(row["pageId"])

    def id_lines(path: str) -> set[str]:
        return {line.strip() for line in Path(path).read_text().splitlines() if line.strip()}

    fit_exclude = id_lines(args.fit_exclude) if args.fit_exclude else set()
    dropped_pages: set[str] = set()
    for path in args.drop_pages:
        dropped_pages |= id_lines(path)
    if fit_exclude and args.template_stack:
        raise SystemExit("--fit-exclude does not support --template-stack")
    overridden = 0
    for path in args.jev_override:
        for row in jsonl(Path(path)):
            if row.get("pageType") in V8_CLASSES:
                jev8[row["pageId"]] = row
                overridden += 1
    # Free-label agreement per class, measured on the noise sample BEFORE merging.
    noise = jsonl(jev_dir / "noise.jsonl")
    agree: Counter = Counter()
    seen: Counter = Counter()
    for row in noise:
        label = schema.get(row["pageId"])
        if label is None:
            continue
        seen[label] += 1
        agree[label] += row["pageType"] == label
    agreement = {c: round(agree[c] / seen[c], 3) for c in seen}
    free_ok = {c for c, a in agreement.items() if a >= args.free_min_agreement}
    v7 = {}
    for split in ("train", "val", "holdout"):
        for row in jsonl(Path(args.jev_v7) / f"{split}.jsonl"):
            v7[row["pageId"]] = row

    schema_features = json.loads((dataset / "feature-schema.json").read_text())
    features = load_features(dataset, schema_features["pageFeatureCount"],
                             schema_features["tokenBuckets"])
    for extra_path in args.extra_features:
        extra = {r["p"]: r["f"] for r in jsonl(Path(extra_path))}
        width = len(next(iter(extra.values())))
        for page_id, feat in features.items():
            feat["x"] = np.concatenate([feat["x"], np.asarray(extra.get(page_id, [0.0] * width),
                                                               dtype=np.float32)])
    base_width = schema_features["pageFeatureCount"] + schema_features["tokenBuckets"]
    if args.dropout_extra > 0:
        for page_id, feat in features.items():
            if feat["s"] == "holdout":
                continue
            bucket = int(hashlib.sha256(f"drop:{page_id}".encode()).hexdigest()[:8], 16) % 1000
            if bucket < args.dropout_extra * 1000:
                feat["x"] = feat["x"].copy()
                feat["x"][base_width:] = np.nan
    page_rules_v82 = None
    platforms: dict[str, str] = {}
    if args.policy_rules and args.policy_version == "v8.2":
        if not args.platform_tags:
            raise SystemExit("--policy-version v8.2 needs --platform-tags (rule 1 reads it)")
        # Imported here, not at the top: v8 runs must not depend on the
        # adjudicator's module. Locally it sits in ../adjudicate; on Modal the
        # image adds it beside train/ (modal_phase3.py).
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "adjudicate"))
        from policy_rules import page_rules_v82

        platforms = {r["p"]: r.get("platform") for r in jsonl(Path(args.platform_tags))}
    urls: dict[str, str] = {}
    if args.policy_rules or args.score_holdout:
        with gzip.open(dataset / "page-index.jsonl.gz", "rt") as handle:
            for line in handle:
                row = json.loads(line)
                urls[row["p"]] = row["u"]
    corrected: dict[str, Counter] = {s: Counter() for s in ("train", "val", "holdout")}
    rows = {"train": ([], [], [], []), "val": ([], [], [], []), "holdout": ([], [], [], [])}
    sources: dict[str, Counter] = {s: Counter() for s in rows}
    counts: dict[str, Counter] = {s: Counter() for s in rows}
    for page_id, feat in features.items():
        split = feat["s"]
        label = weight = source = None
        if page_id in jev8 and jev8[page_id]["pageType"] in V8_CLASSES:
            label, weight, source = jev8[page_id]["pageType"], jev8[page_id].get("confidence"), "jev-v8"
        elif page_id in v7 and v7[page_id]["pageType"] in V7_REUSED:
            label, weight, source = v7[page_id]["pageType"], v7[page_id].get("confidence"), "jev-v7"
        elif split != "holdout" and schema.get(page_id) in free_ok:
            label, weight, source = schema[page_id], agreement[schema[page_id]], "schema"
        if label is None or (split != "holdout" and page_id in ambiguous):
            continue
        if page_id in dropped_pages:
            continue
        if page_rules_v82 and source in ("jev-v8", "jev-v7") and page_id in urls:
            # v8.2 returns the folded class; count only what a RULE changed.
            fixed = page_rules_v82(label, urls[page_id], platforms.get(page_id))
            if fixed != fold(label):
                corrected[split][f"{label}->{fixed}"] += 1
            label = fixed
        elif args.policy_rules and source in ("jev-v8", "jev-v7") and page_id in urls:
            parts = urlsplit(urls[page_id])
            segments = [s for s in parts.path.lower().split("/") if s]
            fixed = label
            if not segments and not parts.query:
                fixed = "homepage"
            elif "blog" in segments[:-1] and label in ("docs", "article", "other"):
                fixed = "blog_post"
            if fixed != label:
                corrected[split][f"{label}->{fixed}"] += 1
                label = fixed
        label = fold(label)
        x, y, w, ids = rows[split]
        x.append(feat["x"])
        y.append(index[label])
        w.append(float(weight if weight is not None else 1.0))
        ids.append(page_id)
        sources[split][source] += 1
        counts[split][label] += 1

    def matrix(split):
        x, y, w, ids = rows[split]
        return sp.csr_matrix(np.vstack(x)), np.asarray(y), np.asarray(w), ids

    xt, yt, wt, ids_t = matrix("train")
    xv, yv, wv, ids_v = matrix("val")
    xh, yh, _, ids_h = matrix("holdout")
    # Rows a model may be fit on (all of them unless --fit-exclude).
    ft = np.asarray([p not in fit_exclude for p in ids_t])
    fv = np.asarray([p not in fit_exclude for p in ids_v])
    print(json.dumps({"classes": classes, "counts": {s: dict(c) for s, c in counts.items()},
                      "sources": {s: dict(c) for s, c in sources.items()},
                      "freeAgreement": agreement, "freeClassesUsed": sorted(free_ok)}), flush=True)

    base = {"objective": "multiclass", "num_class": len(classes), "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 1, "max_bin": 63, "num_threads": N_JOBS,
            "verbose": -1, "seed": 20260923}
    best = None
    grid = []
    for leaves, lr in ((31, 0.05), (63, 0.05), (127, 0.05)):
        params = {**base, "num_leaves": leaves, "learning_rate": lr, "min_data_in_leaf": 20}
        booster = lgb.train(params, lgb.Dataset(xt[ft], label=yt[ft], weight=wt[ft]),
                            num_boost_round=args.rounds,
                            valid_sets=[lgb.Dataset(xv[fv], label=yv[fv], weight=wv[fv])],
                            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        vp = booster.predict(xv, num_iteration=booster.best_iteration)
        val_f1 = per_class(yv[fv], vp[fv].argmax(1), np.ones(int(fv.sum()), bool), classes)["macroF1"]
        grid.append({"leaves": leaves, "lr": lr, "iters": booster.best_iteration, "valMacroF1": val_f1})
        print(json.dumps(grid[-1]), flush=True)
        if best is None or val_f1 > best[0]:
            best = (val_f1, booster, vp)
    _, booster, vp = best
    iterations = int(booster.best_iteration)
    selection = {"yEval": yv, "probs": vp, "name": "val", "ids": ids_v, "fit": fv}
    if args.cv_folds > 0:
        # Thresholds from domain-grouped out-of-fold predictions over train+val:
        # every pool page is scored by a model that never saw its domain, and
        # holdout plays no part. The shipped model is refit on the whole pool.
        chosen_leaves = grid[max(range(len(grid)), key=lambda i: grid[i]["valMacroF1"])]["leaves"]
        params = {**base, "num_leaves": chosen_leaves, "learning_rate": 0.05, "min_data_in_leaf": 20}
        xp = sp.vstack([xt, xv], format="csr")
        yp = np.concatenate([yt, yv])
        wp = np.concatenate([wt, wv])
        groups = [features[p]["d"] for p in [*ids_t, *ids_v]]
        fold_of = np.asarray([int(hashlib.sha256(g.encode()).hexdigest()[:8], 16) % args.cv_folds
                              for g in groups])
        fp = np.concatenate([ft, fv])
        oof = np.zeros((xp.shape[0], len(classes)))
        for f in range(args.cv_folds):
            test = fold_of == f
            fit = ~test & fp
            fb = lgb.train(params, lgb.Dataset(xp[fit], label=yp[fit], weight=wp[fit]),
                           num_boost_round=iterations)
            oof[test] = fb.predict(xp[test])
        booster = lgb.train(params, lgb.Dataset(xp[fp], label=yp[fp], weight=wp[fp]),
                            num_boost_round=iterations)
        booster.best_iteration = iterations
        if args.template_stack:
            # Stage 2: a page rendered by the same template as its siblings
            # probably has their type. Each page gets the mean stage-1
            # probability of the OTHER pages in its (domain, skeleton) cluster:
            # out-of-fold for the pool (a cluster is one domain, so one fold),
            # the pool model's for holdout (unseen domains either way).
            cluster = {r["p"]: r["cluster"] for r in jsonl(Path(args.template_stack))}
            pool_ids = [*ids_t, *ids_v]

            def stacked(ids, probs):
                sums: dict = {}
                for pid, pr in zip(ids, probs, strict=True):
                    key = cluster.get(pid)
                    if key is None:
                        continue
                    total, n = sums.get(key, (np.zeros(probs.shape[1]), 0))
                    sums[key] = (total + pr, n + 1)
                rows_out = []
                for pid, pr in zip(ids, probs, strict=True):
                    total, n = sums.get(cluster.get(pid), (np.zeros(probs.shape[1]), 0))
                    others = n - 1
                    mean = (total - pr) / others if others > 0 else np.zeros(probs.shape[1])
                    rows_out.append(np.concatenate([mean, [np.log1p(max(others, 0))]]))
                return sp.csr_matrix(np.asarray(rows_out, dtype=np.float32))

            xp = sp.hstack([xp, stacked(pool_ids, oof)], format="csr")
            xh = sp.hstack([xh, stacked(ids_h, booster.predict(xh))], format="csr")
            oof = np.zeros((xp.shape[0], len(classes)))
            for f in range(args.cv_folds):
                test = fold_of == f
                fb = lgb.train(params, lgb.Dataset(xp[~test], label=yp[~test], weight=wp[~test]),
                               num_boost_round=iterations)
                oof[test] = fb.predict(xp[test])
            booster = lgb.train(params, lgb.Dataset(xp, label=yp, weight=wp),
                                num_boost_round=iterations)
            booster.best_iteration = iterations
        selection = {"yEval": yp, "probs": oof, "name": f"oof-{args.cv_folds}fold",
                     "ids": [*ids_t, *ids_v], "folds": fold_of, "fit": fp}
    # Thresholds are picked on the rows the arm may fit on: an excluded page is
    # one the arm's corpus never had, so it cannot calibrate on it either.
    sel_fit = selection["fit"]
    if args.lcb and "folds" in selection:
        thresholds = pick_thresholds_lcb(selection["probs"][sel_fit], selection["yEval"][sel_fit],
                                         selection["folds"][sel_fit], classes,
                                         args.target_precision, args.min_fires)
        selection["name"] += "-lcb"
    else:
        thresholds = pick_thresholds(selection["probs"][sel_fit], selection["yEval"][sel_fit],
                                     classes, args.target_precision, args.min_fires)
    thr = np.asarray(thresholds)

    def score_probs(probs, y):
        pred = probs.argmax(1)
        out = per_class(y, pred, probs.max(1) >= thr[pred], classes)
        # Schema level: a recommendation is right when it names the right schema
        # family, so article <-> blog_post (BlogPosting is an Article) is not an error.
        fam = np.asarray([FAMILY.get(classes[i], classes[i]) for i in range(len(classes))])
        fp, fy = fam[pred], fam[y]
        fires = probs.max(1) >= thr[pred]
        levels = {}
        for family in sorted(set(fam)):
            fired = fires & (fp == family)
            n = int(fired.sum())
            support = int((fy == family).sum())
            levels[family] = {
                "support": support, "fires": n,
                "precisionAtThreshold": round(float((fy[fired] == family).mean()), 4) if n else None,
                "coverageAtThreshold": round(int((fy[fired] == family).sum()) / support, 4)
                if support else None}
        out["schemaLevel"] = levels
        return out

    def score(x, y):
        return score_probs(booster.predict(x, num_iteration=booster.best_iteration), y)

    report = {
        "classes": classes, "merge": merge,
        "labelProvenance": "page type v8 (dom-suggestions-v8): Jev v8 + Jev v7 (1:1 classes) "
                           "+ declared schema (agreement-qualified classes, schema hidden)",
        "counts": {s: dict(c) for s, c in counts.items()},
        "sources": {s: dict(c) for s, c in sources.items()},
        "policyRuleCorrections": {s: dict(c) for s, c in corrected.items()},
        "policyVersion": args.policy_version if args.policy_rules else None,
        "jevOverrides": overridden,
        "droppedAmbiguous": len(ambiguous),
        "freeAgreement": agreement, "freeClassesUsed": sorted(free_ok),
        "grid": grid, "iterations": int(booster.best_iteration),
        "thresholds": dict(zip(classes, thresholds, strict=True)),
        "targetPrecision": args.target_precision,
        "thresholdsFrom": selection["name"],
        "selection": score_probs(selection["probs"][sel_fit], selection["yEval"][sel_fit]),
        "fitExcluded": int((~sel_fit).sum()),
        "droppedPages": len(dropped_pages),
        "holdoutScored": bool(args.score_holdout),
    }
    if args.score_holdout:
        report["holdout"] = score(xh, yh)
        # The non-trivial slices: a site root IS a homepage (a rule, and the
        # model can read urlIsRoot), and a /blog/ path says blog_post.
        def segments_of(pid):
            parts = urlsplit(urls.get(pid, ""))
            return [s for s in parts.path.lower().split("/") if s], bool(parts.query)

        seg = [segments_of(p) for p in ids_h]
        non_root = np.asarray([bool(sg) or q for sg, q in seg])
        not_blog = np.asarray(["blog" not in sg for sg, _ in seg])
        about_url = re.compile(r"about|team|company|who-we-are|uber-uns|ueber-uns|a-propos|chi-siamo|"
                               r"sobre|quienes-somos|nosotros|over-ons|om-oss|firma|unternehmen|"
                               r"equipe|story|mission")
        contact_url = re.compile(r"contact|kontakt|contacto|contatti|support|help|hilfe|aide|ayuda")
        path_of = [urlsplit(urls.get(p, "")).path.lower() for p in ids_h]
        no_about_word = np.asarray([not about_url.search(path) for path in path_of])
        no_contact_word = np.asarray([not contact_url.search(path) for path in path_of])

        def slice_scores(xm):
            probs = booster.predict(xm, num_iteration=booster.best_iteration)
            pred, conf = probs.argmax(1), probs.max(1)
            fires = conf >= thr[pred]
            out = {}
            for name, cls, mask in (("homepageNonRoot", "homepage", non_root),
                                    ("blogPostNotUnderBlog", "blog_post", not_blog),
                                    ("aboutNoAboutWordInPath", "about", no_about_word),
                                    ("contactNoContactWordInPath", "contact", no_contact_word)):
                i = index[cls]
                fired = mask & fires & (pred == i)
                support = int((mask & (yh == i)).sum())
                n = int(fired.sum())
                out[name] = {"support": support, "fires": n,
                             "precisionAtThreshold": round(float((yh[fired] == i).mean()), 4) if n else None,
                             "coverageAtThreshold": round(int((yh[fired] == i).sum()) / support, 4)
                             if support else None}
            return out

        report["holdoutSlices"] = slice_scores(xh)
        # Per-page holdout predictions, private output, so later slices need no
        # new holdout touch.
        probs_h = booster.predict(xh, num_iteration=booster.best_iteration)
        with (out / "holdout-predictions.jsonl").open("w") as handle:
            for pid, y, pr in zip(ids_h, yh, probs_h, strict=True):
                top = int(pr.argmax())
                handle.write(json.dumps({"pageId": pid, "label": classes[int(y)],
                                         "pred": classes[top], "conf": round(float(pr[top]), 4),
                                         "fires": bool(pr[top] >= thr[top])}) + "\n")
        if xh.shape[1] > base_width:
            # Without the link graph: every cross-page column missing.
            dense = xh.toarray()
            dense[:, base_width:] = np.nan
            xm = sp.csr_matrix(dense)
            report["holdoutNoGraph"] = score(xm, yh)
            report["holdoutNoGraphSlices"] = slice_scores(xm)
    # "Correct schema" holdout pages: declared schema agrees with Jev v8.
    keep = np.asarray([schema.get(p) is not None and fold(schema[p]) == classes[y]
                       for p, y in zip(ids_h, yh, strict=True)])
    if args.score_holdout and keep.any():
        report["schemaHidden"] = score(xh[keep], yh[keep])
    if args.v7_model and args.score_holdout:
        v7_booster = lgb.Booster(model_file=args.v7_model)
        # The v7 head saw only the base page vector; --extra-features columns
        # are appended after it, so the prefix is exactly what v7 was trained on.
        v7_pred = v7_booster.predict(xh[:, : v7_booster.num_feature()]).argmax(1)
        shared = [c for c in V7_CLASSES if c in index]
        v7_as_v8 = np.asarray([index.get(fold(V7_CLASSES[i]), -1) for i in v7_pred])
        v8_pred = booster.predict(xh, num_iteration=booster.best_iteration).argmax(1)
        both = {}
        for name in shared:
            i = index[name]
            for tag, pred in (("v7", v7_as_v8), ("v8", v8_pred)):
                tp = int(((pred == i) & (yh == i)).sum())
                fp = int(((pred == i) & (yh != i)).sum())
                support = int((yh == i).sum())
                p = tp / (tp + fp) if tp + fp else 0.0
                r = tp / support if support else 0.0
                both.setdefault(name, {})[tag] = {
                    "precision": round(p, 4), "recall": round(r, 4),
                    "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}
        report["vsV7"] = both
    if args.dump_selection:
        with (out / "selection-predictions.jsonl").open("w") as handle:
            for page_id, y, probs, fit in zip(selection["ids"], selection["yEval"],
                                              selection["probs"], selection["fit"], strict=True):
                top = int(probs.argmax())
                handle.write(json.dumps({"pageId": page_id, "label": classes[int(y)],
                                         "pred": classes[top], "conf": round(float(probs[top]), 4),
                                         "fires": bool(probs[top] >= thr[top]),
                                         "fitExcluded": not bool(fit)}) + "\n")
        if args.dump_probs:
            np.save(out / "selection-probs.npy", selection["probs"].astype(np.float16))
    booster.save_model(str(out / "pagetype-model.txt"), num_iteration=booster.best_iteration)
    (out / "class-thresholds.json").write_text(json.dumps({"classes": classes,
                                                            "thresholds": thresholds}) + "\n")
    (out / "pagetype-jev-report.json").write_text(json.dumps(report, indent=2) + "\n")
    h = report.get("holdout") or report["selection"]
    print(json.dumps({"scoredOn": "holdout" if args.score_holdout else selection["name"],
                      "macroF1": h["macroF1"], "coverage": h["coverage"],
                      "accuracyWhenFiring": h["accuracyWhenFiring"]}))


if __name__ == "__main__":
    main()
