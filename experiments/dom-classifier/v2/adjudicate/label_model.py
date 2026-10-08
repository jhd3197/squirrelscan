# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3"]
# ///
"""Probabilistic label model over Treetop's label sources (#2307).

Combines Jev, Clef, the visual adjudicator, URL rules, markup rules and declared
schema into one posterior per item, weighting every source by its measured
per-class accuracy on CALIBRATION gold (train/val pages a human reviewed; never
test gold). Dawid-Skene shaped:

  P(y | votes) is proportional to  prior(y) * prod_s P(vote_s | y) ** T

- One confusion matrix per source and confidence bin (a source's confident and
  hesitant votes are separate sources: Jev at 0.95 and Jev at 0.4 are not the
  same witness). A missing vote (abstention) contributes nothing.
- Each row of a confusion matrix is estimated from calibration gold and shrunk
  toward a structured base: the source-and-bin's pooled accuracy on the
  diagonal, its pooled wrong-vote distribution off it (Dirichlet pseudo-count
  `alpha`). Gold items carry their sampling weight (the cohort oversamples
  confident-learning candidates, where sources disagree most), normalised
  within each true class so a class keeps its real sample size.
- T in (0, 1] tempers the product: Jev, Clef and the adjudicator read similar
  evidence and their errors correlate, so naive Bayes is overconfident. T is
  picked by k-fold cross-validation on gold (minimum log loss of the true label).
- prior(y): the weighted gold label distribution, smoothed.

Optional EM (`--em-iters`) re-estimates the confusions on the unlabelled pool
with gold items clamped; off by default because correlated sources make EM drift.

Output (one row per item): label, posterior (top 3), training weight = the
label's posterior, and `drop` when it is under --min-posterior. `export` writes
it in the shapes the v2 trainers already read (see `export_rows`).

    uv run --script label_model.py fit --votes votes.jsonl --gold gold.jsonl --axis page --out <dir>
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PAGE_CLASSES = [
    "homepage",
    "product_detail",
    "product_listing",
    "article",
    "blog_post",
    "about",
    "contact",
    "docs",
    "legal",
    "pricing",
    "listing_index",
    "other",
]
SECTION_CLASSES = [
    "faq_block",
    "reviews",
    "breadcrumbs",
    "business_info",
    "product_offer",
    "author_byline",
    "video_embed",
    "ad",
    "cookie_consent",
    "newsletter_signup",
    "social_share",
    "related_content",
    "comments",
    "promo_banner",
    "pagination",
    "table_of_contents",
    "account_login",
    "locale_switcher",
    "none",
]
# Confidence bins: a vote with conf >= cut is "hi", else "lo"; a source without conf has one bin.
DEFAULT_CUT = 0.8


def jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def source_bin(vote: dict, cut: float) -> str:
    conf = vote.get("conf")
    if conf is None:
        return vote["source"]
    return f"{vote['source']}:{'hi' if conf >= cut else 'lo'}"


class Votes:
    """Items x sources, as class indices (-1 = no vote)."""

    def __init__(self, rows: list[dict], classes: list[str], cut: float = DEFAULT_CUT):
        index = {c: i for i, c in enumerate(classes)}
        self.classes = classes
        by_item: dict[str, dict[str, int]] = defaultdict(dict)
        for row in rows:
            if row["label"] not in index:
                continue
            by_item[row["key"]][source_bin(row, cut)] = index[row["label"]]
        self.keys = sorted(by_item)
        self.sources = sorted({s for votes in by_item.values() for s in votes})
        self.matrix = np.full((len(self.keys), len(self.sources)), -1, np.int64)
        col = {s: j for j, s in enumerate(self.sources)}
        for i, key in enumerate(self.keys):
            for s, v in by_item[key].items():
                self.matrix[i, col[s]] = v
        self.row = {k: i for i, k in enumerate(self.keys)}


def family(source: str) -> str:
    return source.split(":")[0]


def fit_confusions(
    votes: np.ndarray,
    truth: np.ndarray,
    weight: np.ndarray,
    sources: list[str],
    k: int,
    alpha: float,
    beta: float = 2.0,
) -> np.ndarray:
    """theta[s, y, v] = P(vote v | true y) for every source column, shrunk to a structured base."""
    n_src = votes.shape[1]
    counts = np.zeros((n_src, k, k))
    for j in range(n_src):
        has = votes[:, j] >= 0
        np.add.at(counts[j], (truth[has], votes[has, j]), weight[has])
    # Pooled accuracy per family (all bins) is the prior for each bin's own accuracy.
    fam_acc = {}
    for fam in {family(s) for s in sources}:
        cols = [j for j, s in enumerate(sources) if family(s) == fam]
        c = counts[cols].sum(0)
        fam_acc[fam] = (np.trace(c) + 1.0) / (c.sum() + 2.0)
    theta = np.zeros((n_src, k, k))
    for j, s in enumerate(sources):
        c = counts[j]
        total = c.sum()
        acc = (np.trace(c) + 4.0 * fam_acc[family(s)]) / (total + 4.0)
        wrong = c.sum(0) - np.diag(c)  # wrong votes by voted class
        # A few dozen wrong votes spread over k classes: smooth toward uniform (beta per class).
        q = (wrong + beta) / (wrong.sum() + beta * k)
        base = np.zeros((k, k))
        for y in range(k):
            off = q.copy()
            off[y] = 0.0
            base[y] = (1.0 - acc) * off / off.sum()
            base[y, y] = acc
        theta[j] = (c + alpha * base) / (c.sum(1, keepdims=True) + alpha)
    return theta


def class_prior(truth: np.ndarray, weight: np.ndarray, k: int) -> np.ndarray:
    prior = np.bincount(truth, weights=weight, minlength=k) + 1.0
    return prior / prior.sum()


def posterior(votes: np.ndarray, theta: np.ndarray, prior: np.ndarray, temperature: float) -> np.ndarray:
    logp = np.tile(np.log(prior), (votes.shape[0], 1))
    log_theta = np.log(theta)
    for j in range(votes.shape[1]):
        has = votes[:, j] >= 0
        # log P(vote | y) for every y: theta[j][:, v] -> column of the voted class.
        logp[has] += temperature * log_theta[j][:, votes[has, j]].T
    logp -= logp.max(1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(1, keepdims=True)


def normalised_weights(truth: np.ndarray, weight: np.ndarray, k: int) -> np.ndarray:
    """Within each true class, rescale sampling weights to sum to that class's count."""
    out = weight.astype(np.float64).copy()
    for y in range(k):
        m = truth == y
        if m.any():
            out[m] *= m.sum() / weight[m].sum()
    return out


def folds_of(n: int, k: int) -> np.ndarray:
    return np.arange(n) % k


def cross_validate(votes, truth, weight, sources, k, alpha, temperatures, n_folds=5):
    """Out-of-fold posteriors per temperature; pick T by weighted log loss of the true class."""
    rng = np.random.default_rng(20261009)
    fold = folds_of(len(truth), n_folds)[rng.permutation(len(truth))]
    results = {}
    for t in temperatures:
        p = np.zeros((len(truth), k))
        for f in range(n_folds):
            fit, test = fold != f, fold == f
            w = normalised_weights(truth[fit], weight[fit], k)
            theta = fit_confusions(votes[fit], truth[fit], w, sources, k, alpha)
            p[test] = posterior(votes[test], theta, class_prior(truth[fit], w, k), t)
        ll = -np.log(np.clip(p[np.arange(len(truth)), truth], 1e-9, 1))
        results[t] = {
            "logLoss": float(np.average(ll, weights=weight)),
            "accuracy": float(np.average(p.argmax(1) == truth, weights=weight)),
            "oof": p,
        }
    best = min(results, key=lambda t: results[t]["logLoss"])
    return best, results


def em(votes, theta, prior, gold_rows, gold_truth, sources, k, alpha, temperature, iters):
    """Dawid-Skene EM over the pool with gold rows clamped to their label."""
    for _ in range(iters):
        p = posterior(votes, theta, prior, temperature)
        p[gold_rows] = np.eye(k)[gold_truth]
        counts = np.zeros_like(theta)
        for j in range(votes.shape[1]):
            has = votes[:, j] >= 0
            for v in range(k):
                counts[j][:, v] = p[has & (votes[:, j] == v)].sum(0)
        theta = (counts + alpha * theta) / (counts.sum(2, keepdims=True) + alpha)
        prior = (p.sum(0) + 1.0) / (p.sum() + k)
    return theta, prior


def single_source_report(votes, truth, weight, sources, classes):
    """Per source-bin: coverage on gold and weighted per-class precision of its votes."""
    out = {}
    for j, s in enumerate(sources):
        has = votes[:, j] >= 0
        if not has.any():
            continue
        correct = votes[has, j] == truth[has]
        per_class = {}
        for y, c in enumerate(classes):
            m = votes[has, j] == y
            if m.sum() >= 3:
                per_class[c] = {
                    "n": int(m.sum()),
                    "precision": round(float(np.average(correct[m], weights=weight[has][m])), 3),
                }
        out[s] = {
            "n": int(has.sum()),
            "accuracy": round(float(np.average(correct, weights=weight[has])), 3),
            "precisionByVotedClass": per_class,
        }
    return out


def fit(args) -> None:
    classes = PAGE_CLASSES if args.axis == "page" else SECTION_CLASSES
    k = len(classes)
    index = {c: i for i, c in enumerate(classes)}
    vote_rows = [r for r in jsonl(args.votes) if r.get("axis", args.axis) == args.axis]
    if args.exclude_sources:
        drop = set(args.exclude_sources.split(","))
        vote_rows = [r for r in vote_rows if r["source"] not in drop]
    v = Votes(vote_rows, classes, args.cut)
    gold = [
        g
        for g in jsonl(args.gold)
        if g.get("axis", args.axis) == args.axis and g["label"] in index and g["key"] in v.row
    ]
    gold_rows = np.asarray([v.row[g["key"]] for g in gold])
    truth = np.asarray([index[g["label"]] for g in gold])
    weight = np.asarray([float(g.get("weight", 1.0)) for g in gold])
    temperatures = [round(t, 2) for t in np.arange(0.2, 1.01, 0.1)]
    best_t, cv = cross_validate(v.matrix[gold_rows], truth, weight, v.sources, k, args.alpha, temperatures)
    w = normalised_weights(truth, weight, k)
    theta = fit_confusions(v.matrix[gold_rows], truth, w, v.sources, k, args.alpha)
    prior = class_prior(truth, w, k)
    if args.em_iters:
        theta, prior = em(v.matrix, theta, prior, gold_rows, truth, v.sources, k, args.alpha, best_t, args.em_iters)
    p = posterior(v.matrix, theta, prior, best_t)
    args.out.mkdir(parents=True, exist_ok=True)
    best = p.argmax(1)
    with (args.out / f"posteriors-{args.axis}.jsonl").open("w") as handle:
        for i, key in enumerate(v.keys):
            top = np.argsort(-p[i])[:3]
            conf = float(p[i, best[i]])
            handle.write(
                json.dumps(
                    {
                        "key": key,
                        "axis": args.axis,
                        "label": classes[best[i]],
                        "weight": round(conf, 4),
                        "drop": conf < args.min_posterior,
                        "nSources": int((v.matrix[i] >= 0).sum()),
                        "posterior": {classes[j]: round(float(p[i, j]), 4) for j in top},
                    }
                )
                + "\n"
            )
    oof = cv[best_t]["oof"]
    report = {
        "axis": args.axis,
        "items": len(v.keys),
        "gold": len(gold),
        "sources": v.sources,
        "temperature": best_t,
        "cv": {str(t): {kk: round(vv, 4) for kk, vv in r.items() if kk != "oof"} for t, r in cv.items()},
        "goldAccuracyOOF": round(float(np.average(oof.argmax(1) == truth, weights=weight)), 4),
        "coverageAtPosterior": {
            str(tau): {
                "coverage": round(float(np.average(oof.max(1) >= tau, weights=weight)), 4),
                "accuracy": round(
                    float(np.average((oof.argmax(1) == truth)[oof.max(1) >= tau], weights=weight[oof.max(1) >= tau])), 4
                )
                if (oof.max(1) >= tau).any()
                else None,
            }
            for tau in (0.5, 0.7, 0.8, 0.9, 0.95)
        },
        "singleSources": single_source_report(v.matrix[gold_rows], truth, weight, v.sources, classes),
        "dropped": int((p.max(1) < args.min_posterior).sum()),
        "labelCounts": {classes[y]: int((best == y).sum()) for y in range(k)},
    }
    (args.out / f"report-{args.axis}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({kk: report[kk] for kk in ("axis", "items", "gold", "temperature", "goldAccuracyOOF", "dropped")}))


def export_rows(args) -> None:
    """Trainer shapes: page type as --jev-override rows (pageType, confidence = weight) plus a
    --drop-pages list; sections as --labels rows (pageId, nodeId, section, weight). train_section
    reads `section` only, so section weights need a trainer change to take effect."""
    rows = jsonl(args.posteriors)
    args.out.mkdir(parents=True, exist_ok=True)
    keep = [r for r in rows if not r["drop"]]
    if args.axis == "page":
        with (args.out / "labelmodel-pagetype.jsonl").open("w") as handle:
            for r in keep:
                handle.write(
                    json.dumps(
                        {
                            "pageId": r["key"],
                            "pageType": r["label"],
                            "confidence": r["weight"],
                            "source": "label-model",
                            "posterior": r["posterior"],
                        }
                    )
                    + "\n"
                )
        (args.out / "labelmodel-drop-pages.txt").write_text("".join(f"{r['key']}\n" for r in rows if r["drop"]))
    else:
        with (args.out / "labelmodel-section.jsonl").open("w") as handle:
            for r in keep:
                page_id, node = r["key"].split("#")
                handle.write(
                    json.dumps(
                        {
                            "pageId": page_id,
                            "nodeId": int(node),
                            "section": r["label"],
                            "weight": r["weight"],
                            "source": "label-model",
                            "posterior": r["posterior"],
                        }
                    )
                    + "\n"
                )
        with (args.out / "labelmodel-section-drop-rows.jsonl").open("w") as handle:
            for r in rows:
                if r["drop"]:
                    page_id, node = r["key"].split("#")
                    handle.write(json.dumps({"pageId": page_id, "nodeId": int(node)}) + "\n")
    print(json.dumps({"kept": len(keep), "dropped": len(rows) - len(keep)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--votes", type=Path, required=True)
    f.add_argument("--gold", type=Path, required=True, help="calibration gold only, never test gold")
    f.add_argument("--axis", choices=["page", "section"], required=True)
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--alpha", type=float, default=8.0)
    f.add_argument("--cut", type=float, default=DEFAULT_CUT)
    f.add_argument("--min-posterior", type=float, default=0.6)
    f.add_argument("--em-iters", type=int, default=0)
    f.add_argument("--exclude-sources", default="")
    f.set_defaults(func=fit)
    e = sub.add_parser("export")
    e.add_argument("--posteriors", type=Path, required=True)
    e.add_argument("--axis", choices=["page", "section"], required=True)
    e.add_argument("--out", type=Path, required=True)
    e.set_defaults(func=export_rows)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
