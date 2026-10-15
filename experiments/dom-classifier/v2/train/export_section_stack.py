# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "lightgbm"]
# ///
"""Train the natural-node section stage and pack it as a drop-in section head (#2307).

Same input as the shipped section head ([197 numeric, 512 token buckets]), so the runtime needs no change: the output
`section-head.json` is a HeadJson that replaces `section` in an artifact. Per-class thresholds come from out-of-fold
predictions (lowest bar with precision >= goal and >= --min-answers answers); a class with none abstains (1.01).
Folds group by page by default. With --domains (page id -> domain hash) they group by domain, which is the honest
split: pages of one site share a template, so page folds leak it and the thresholds come out too low.
Parity: float32 reference evaluator vs LightGBM on real rows.

    uv run --script export_section_stack.py --nodes nodes-features.jsonl --answers haiku.jsonl --opus opus.jsonl --out dir \
        [--domains pool-domains.json] [--items-as-section] [--min-answers 20] [--natural-only-thresholds]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_model as em  # noqa: E402

BARS = [round(0.3 + 0.01 * i, 2) for i in range(70)]
CURVE_BARS = (0.3, 0.5, 0.7, 0.9)


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fold_of(key: str, folds: int) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % folds


def featurizer_version(nodes: Path, explicit: str | None) -> str:
    """Explicit flag, else the sidecar `<nodes>.classes.json` the exporter writes, else unknown (concatenated inputs)."""
    if explicit:
        return explicit
    sidecar = nodes.with_name(nodes.name + ".classes.json")
    if sidecar.exists():
        found = json.loads(sidecar.read_text()).get("featurizerVersion")
        if isinstance(found, str):
            return found
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--answers", type=Path, action="append", required=True)
    parser.add_argument("--opus", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--goal-precision", type=float, default=0.9)
    parser.add_argument("--min-answers", type=int, default=10)
    parser.add_argument("--min-class", type=int, default=15)
    parser.add_argument("--natural-only-thresholds", action="store_true",
                        help="choose thresholds on out-of-fold rows whose itemId does not start with secb- (mined, class-enriched)")
    parser.add_argument("--domains", type=Path,
                        help="JSON {page id: domain hash}: out-of-fold folds group by domain (fallback: page id)")
    parser.add_argument("--items-as-section", action="store_true",
                        help="a node the labeller marked as an ITEM of a class (label none, itemOf set) takes that class as its label; "
                        "without this flag only comments items do")
    parser.add_argument("--max-pages-per-domain", type=int, default=0,
                        help="keep at most N pages of any one domain (random, seed 5; all nodes of a kept page stay); needs --domains; 0 = no cap")
    parser.add_argument("--domain-weight", choices=("none", "sqrt"), default="none",
                        help="sqrt: weight each training node by 1/sqrt(pages of its domain), normalised to mean 1; needs --domains")
    parser.add_argument("--featurizer-version", help="recorded in the meta; default: the nodes sidecar, else unknown")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    truth: dict[str, str] = {}
    item_of: dict[str, str] = {}
    for path in [*args.answers, *args.opus]:
        for row in jsonl(path):
            if not row.get("error") and row.get("label"):
                truth[row["itemId"]] = row["label"]
                item_of[row["itemId"]] = row.get("itemOf") or ""
    rows = [r for r in jsonl(args.nodes) if r["itemId"] in truth]
    domains: dict[str, str] = json.loads(args.domains.read_text()) if args.domains else {}
    if (args.max_pages_per_domain or args.domain_weight != "none") and not domains:
        raise SystemExit("--max-pages-per-domain and --domain-weight need --domains")
    pages_of: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        pages_of[str(domains.get(r["pageId"], r["pageId"]))].add(r["pageId"])
    if args.max_pages_per_domain:
        rng = random.Random(5)
        keep: set[str] = set()
        for d in sorted(pages_of):
            pages = sorted(pages_of[d])
            keep.update(rng.sample(pages, min(len(pages), args.max_pages_per_domain)))
        rows = [r for r in rows if r["pageId"] in keep]
        pages_of = {d: {p for p in ps if p in keep} for d, ps in pages_of.items()}

    def label(item_id: str) -> str:
        item = item_of[item_id]
        return item if item and (args.items_as_section or item == "comments") else truth[item_id]

    label_of = {r["itemId"]: label(r["itemId"]) for r in rows}
    counts = Counter(label_of.values())
    classes = sorted(c for c, n in counts.items() if n >= args.min_class and c != "none") + ["none"]
    y = np.asarray([classes.index(label_of[r["itemId"]]) if label_of[r["itemId"]] in classes else len(classes) - 1 for r in rows])
    group_keys = [str(domains.get(r["pageId"], r["pageId"])) for r in rows]
    groups = np.asarray([fold_of(k, args.folds) for k in group_keys])
    n_dense, buckets = len(rows[0]["f"]), 512
    tok = sp.lil_matrix((len(rows), buckets), dtype=np.float32)
    for i, r in enumerate(rows):
        tok[i, r["k"]] = 1.0
    x = sp.hstack([sp.csr_matrix(np.asarray([r["f"] for r in rows], dtype=np.float32)), tok.tocsr()]).tocsr()

    params = {"objective": "multiclass", "num_class": len(classes), "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 8,
              "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1, "num_threads": args.threads, "verbose": -1,
              "seed": 5, "lambda_l2": 1.0}
    weight = None
    if args.domain_weight == "sqrt":
        weight = np.asarray([1.0 / np.sqrt(len(pages_of[k])) for k in group_keys])
        weight = weight / weight.mean()
    oof = np.zeros((len(rows), len(classes)))
    for k in range(args.folds):
        test = groups == k
        train_w = None if weight is None else weight[~test]
        oof[test] = lgb.train(params, lgb.Dataset(x[~test], label=y[~test], weight=train_w), num_boost_round=args.rounds).predict(x[test])
    pred, conf = oof.argmax(axis=1), oof.max(axis=1)
    nat = np.asarray([not r["itemId"].startswith("secb-") for r in rows]) if args.natural_only_thresholds else np.ones(len(rows), bool)
    thresholds, report = [], {}
    for i, cls in enumerate(classes):
        if cls == "none":
            thresholds.append(1.01)
            continue
        chosen = None
        for bar in BARS:
            fired = (pred == i) & (conf >= bar) & nat
            if fired.sum() < args.min_answers:
                break
            prec = float((y[fired] == i).mean())
            if prec >= args.goal_precision:
                chosen = {"bar": bar, "answered": int(fired.sum()), "precision": round(prec, 3),
                          "recall": round(float(((y == i) & fired).sum() / max(((y == i) & nat).sum(), 1)), 3)}
                break  # lowest bar holding the goal = most recall
        thresholds.append(chosen["bar"] if chosen else 1.01)
        support = max(int(((y == i) & nat).sum()), 1)
        curve = {}
        for bar in CURVE_BARS:  # what the class would score at fixed bars, certified or not (natural rows)
            fired = (pred == i) & (conf >= bar) & nat
            curve[str(bar)] = {"fired": int(fired.sum()),
                               "precision": round(float((y[fired] == i).mean()), 3) if fired.any() else None,
                               "recall": round(float(((y == i) & fired).sum() / support), 3)}
        report[cls] = {"support": int(((y == i) & nat).sum()), "supportAll": int((y == i).sum()), "threshold": chosen["bar"] if chosen else None,
                       "oof": chosen, "curve": curve}

    final = lgb.train(params, lgb.Dataset(x, label=y, weight=weight), num_boost_round=args.rounds)
    args.out.mkdir(parents=True, exist_ok=True)
    model_path = args.out / "section-stack.txt"
    final.save_model(str(model_path))
    booster, head_json, reference = em.head(model_path, None, classes, threshold=1.01, class_thresholds=thresholds)
    fv = featurizer_version(args.nodes, args.featurizer_version)
    # Featurizer 2.3.0 rows carry the text/attribute token stream: the head declares that it reads it (and gets the union at runtime).
    token_streams = ["classid", "text"] if fv.startswith("v2.3") else ["classid"]
    if token_streams != ["classid"]:
        head_json["tokenStreams"] = token_streams
    dense_x = x[:2000].toarray().astype(np.float32)
    lgb_probs, ref_probs = booster.predict(dense_x), em.evaluate(reference, dense_x)
    diff = np.abs(lgb_probs - ref_probs)
    agree = float((lgb_probs.argmax(axis=1) == ref_probs.argmax(axis=1)).mean())
    parity = {"rows": len(dense_x), "maxAbsProbabilityDiff": float(diff.max()), "argmaxAgreement": agree}
    print(json.dumps(parity))
    if agree < 0.999 or diff.max() > 1e-3:
        raise SystemExit("parity failed")
    (args.out / "section-head.json").write_text(json.dumps(head_json) + "\n")
    meta = {"classes": classes, "thresholds": thresholds, "perClass": report, "nodes": len(rows), "goalPrecision": args.goal_precision,
            "nDense": n_dense, "parity": parity,
            "foldGrouping": "domain" if args.domains else "page", "foldGroups": len(set(group_keys)), "folds": args.folds,
            "itemsAsSection": args.items_as_section, "minAnswers": args.min_answers, "minClass": args.min_class,
            "naturalOnlyThresholds": args.natural_only_thresholds,
            "maxPagesPerDomain": args.max_pages_per_domain, "domainWeight": args.domain_weight,
            "featurizerVersion": fv, "tokenStreams": token_streams}
    (args.out / "section-head.meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    (args.out / "oof-table.txt").write_text(table(report, classes) + "\n")
    print(table(report, classes), file=sys.stderr)


def table(report: dict, classes: list[str]) -> str:
    lines = [f"{'class':18s} {'support':>7s} {'thr':>5s} {'answered':>8s} {'prec':>5s} {'recall':>6s}"]
    for cls in classes:
        r = report.get(cls)
        if not r:
            continue
        o = r["oof"]
        tail = f"{o['answered']:8d} {o['precision']:5.2f} {o['recall']:6.2f}" if o else f"{'-':>8s}"
        lines.append(f"{cls:18s} {r['support']:7d} {r['threshold']!s:>5} {tail}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
