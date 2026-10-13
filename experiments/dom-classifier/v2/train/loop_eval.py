# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "lightgbm"]
# ///
"""One iteration of the page-type loop: train on weak + teacher labels, score on teacher folds (#2307).

Training rows
  weak     pages whose schema names a main entity (schema_labels_v8 filters), weight --weak-weight
  teacher  schema-less pages a teacher labelled (Opus over Haiku where both answered)
Scoring: domain-grouped K-fold over the TEACHER rows only. In each fold the booster trains on
all weak rows plus the other folds' teacher rows and predicts the held-out teacher rows, so
every number is on the distribution a "declares no schema" finding fires on, never on pages
whose schema the model could have been fitted to.

Classes are the teacher's v8.2 labels merged to what the findings need; classes with fewer
than --min-class teacher rows fold into `other`. Goal per class: precision >= --goal-precision
at recall >= --goal-recall (coverage of that class), found by the best confidence threshold.

  uv run --script train/loop_eval.py --skeletons <dir shard-*/skeletons.jsonl> --schema <schema.jsonl> \
      --teacher haiku=<answers.jsonl> --teacher opus=<answers.jsonl> --samples <sample.jsonl> ... \
      --sealed <sealed.jsonl> --out <dir> [--feature-set all|base] [--weak-weight 0.5]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("schema_labels_v8", HERE / "schema_labels_v8.py")
assert SPEC and SPEC.loader
V8 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V8)

MERGE = {
    "blog_post": "editorial", "article": "editorial", "product_listing": "listing", "listing_index": "listing",
    "product_detail": "product_detail",
}  # fmt: skip
BASE_FEATURES = 54  # the first feature set (before wp/comment/date features)
BARS = [round(0.5 + 0.05 * i, 2) for i in range(10)]


TOKEN_BUCKETS = 256


def load_features(root: Path, tokens: bool) -> tuple[dict[str, np.ndarray], list[str]]:
    out: dict[str, np.ndarray] = {}
    paths = sorted(root.glob("shard-*/skeletons.jsonl"))
    for path in paths:
        for line in path.open():
            row = json.loads(line)
            vector = np.asarray(row["f"], dtype=np.float32)
            if tokens:
                counts = np.zeros(TOKEN_BUCKETS, dtype=np.float32)
                counts[row.get("k", [])] = 1.0
                vector = np.concatenate([vector, counts])
            out[row["p"]] = vector
    names = json.loads((paths[0].parent / "features.json").read_text())["order"]
    return out, names + ([f"tok{i}" for i in range(TOKEN_BUCKETS)] if tokens else [])


def weak_labels(schema: Path, sealed: set[str]) -> dict[str, tuple[str, str]]:
    rows = [json.loads(line) for line in schema.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r.get("ok") and r["d"] not in sealed]
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_domain[row["d"]].append(row)
    site_wide: dict[str, set[str]] = {}
    for domain, pages in by_domain.items():
        presence: Counter = Counter()
        if len(pages) >= 4:
            for page in pages:
                presence.update(set(page["ld"]) | set(page["md"]))
        site_wide[domain] = {t for t, n in presence.items() if t in V8.TEMPLATE_PRONE and n / len(pages) >= 0.6}
    out: dict[str, tuple[str, str]] = {}
    for row in rows:
        label, _ = V8.page_label(row, site_wide[row["d"]])
        if label:
            out[row["sha"][:16]] = (MERGE.get(label, label), row["d"])
    return out


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skeletons", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--teacher", action="append", required=True, help="name=answers.jsonl, later wins")
    parser.add_argument("--samples", type=Path, action="append", required=True, help="sample files carrying `d` per page")
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--feature-set", choices=("all", "base"), default="all")
    parser.add_argument("--tokens", action="store_true", help="append the 256 hashed title/heading word buckets (needs v3 skeletons)")
    parser.add_argument("--train-only-teacher", action="append", default=[], help="name=answers.jsonl: extra rows used only for training")
    parser.add_argument("--train-only-samples", type=Path, action="append", default=[])
    parser.add_argument("--weak-weight", type=float, default=0.5)
    parser.add_argument("--min-class", type=int, default=40)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--leaves", type=int, default=31)
    parser.add_argument("--min-data", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--balance", action="store_true", help="weight teacher rows by inverse class frequency")
    parser.add_argument("--goal-precision", type=float, default=0.90)
    parser.add_argument("--goal-recall", type=float, default=0.50)
    parser.add_argument("--min-teacher-conf", type=float, default=0.0)
    parser.add_argument("--save-model", type=Path, help="train on every labelled row and write model.txt + meta.json (classes, feature names, per-class thresholds) here")
    parser.add_argument("--pool-out", type=Path, help="train on every labelled row and write probabilities for the unlabelled schema-less pool here (active learning)")
    parser.add_argument("--pool-exclude", type=Path, action="append", default=[], help="sample files already judged: left out of the pool file")
    parser.add_argument("--decile-pool", type=Path, help="summary.json with decilePool: weights random-dN eval rows to the natural pool")
    parser.add_argument("--manifest", type=Path, default=Path("/Volumes/BACKUP/page-corpus/pages/manifest.jsonl"))
    parser.add_argument("--keep-homepage", action="store_true",
                        help="default: homepage is answered from the URL at runtime, so root pages (manifest is_homepage) "
                        "and every `homepage` label leave both training and scoring")
    args = parser.parse_args()
    sealed = {json.loads(line)["d"] for line in args.sealed.read_text().splitlines() if line.strip()}

    features, names = load_features(args.skeletons, args.tokens)
    width = BASE_FEATURES if args.feature_set == "base" else len(names)
    weak = weak_labels(args.schema, sealed)

    domain_of: dict[str, str] = {}
    kind_of: dict[str, str] = {}
    for sample in args.samples:
        for line in sample.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                domain_of[row["p"]] = row["d"]
                kind_of[row["p"]] = row["kind"]
    def read_teacher(specs: list[str]) -> dict[str, tuple[str, float]]:
        found: dict[str, tuple[str, float]] = {}
        for spec in specs:
            _, _, path = spec.partition("=")
            for line in Path(path).read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    if "error" not in row and row["confidence"] >= args.min_teacher_conf:
                        found[row["itemId"]] = (MERGE.get(row["label"], row["label"]), row["confidence"])
        return found

    teacher = read_teacher(args.teacher)
    extra = read_teacher(args.train_only_teacher)
    for sample in args.train_only_samples:
        for line in sample.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                domain_of.setdefault(row["p"], row["d"])
    if not args.keep_homepage:
        root: set[str] = set()
        for line in args.manifest.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("is_homepage") and (row.get("raw") or {}).get("sha256"):
                root.add(row["raw"]["sha256"][:16])
        # a teacher "homepage" on a non-root page is a guess without the URL: dropped, not relabelled
        teacher = {p: v for p, v in teacher.items() if p not in root and v[0] != "homepage"}
        extra = {p: v for p, v in extra.items() if p not in root and v[0] != "homepage"}
        weak = {p: v for p, v in weak.items() if p not in root and v[0] != "homepage"}
    teacher = {p: v for p, v in teacher.items() if p in features and p in domain_of and p not in weak}
    if args.decile_pool:
        # only natural-draw rows (random-dN) are scored; hand-picked ones (article/product/negative kinds) train only
        for p in [p for p in teacher if not kind_of.get(p, "").startswith("random-d")]:
            extra[p] = teacher.pop(p)
    extra = {p: v for p, v in extra.items() if p in features and p not in weak and p not in teacher}
    counts = Counter(v[0] for v in teacher.values()) + Counter(v[0] for v in extra.values())
    classes = sorted(c for c, n in counts.items() if n >= args.min_class and c != "other") + ["other"]
    fold_label = {p: (v[0] if v[0] in classes else "other") for p, v in teacher.items()}
    weak_rows = {p: v for p, v in weak.items() if p in features and p not in teacher and p not in extra and v[0] in classes}
    extra_rows = {p: (v[0] if v[0] in classes else "other") for p, v in extra.items()}
    print(f"teacher rows {len(fold_label)} {dict(Counter(fold_label.values()))}; weak rows {len(weak_rows)} {dict(Counter(v[0] for v in weak_rows.values()))}", flush=True)

    t_ids = sorted(fold_label)
    xt = np.vstack([features[p][:width] for p in t_ids])
    yt = np.asarray([classes.index(fold_label[p]) for p in t_ids])
    gt = np.asarray([int(hashlib.sha256(domain_of[p].encode()).hexdigest()[:8], 16) % args.folds for p in t_ids])
    eval_w = np.ones(len(t_ids))
    if args.decile_pool:
        pool = json.loads(args.decile_pool.read_text())["decilePool"]
        drawn = Counter(int(kind_of[p].split("-d")[1]) for p in t_ids)
        eval_w = np.asarray([pool[str(int(kind_of[p].split("-d")[1]))] / drawn[int(kind_of[p].split("-d")[1])] for p in t_ids])
        eval_w = eval_w / eval_w.mean()
        print(f"eval rows weighted to the natural pool by article-score decile; effective n {eval_w.sum() ** 2 / (eval_w ** 2).sum():.0f} of {len(t_ids)}", flush=True)
    w_ids = sorted(weak_rows)
    xw = np.vstack([features[p][:width] for p in w_ids]) if w_ids else np.zeros((0, width), dtype=np.float32)
    yw = np.asarray([classes.index(weak_rows[p][0]) for p in w_ids])
    e_ids = sorted(extra_rows)
    xe = np.vstack([features[p][:width] for p in e_ids]) if e_ids else np.zeros((0, width), dtype=np.float32)
    ye = np.asarray([classes.index(extra_rows[p]) for p in e_ids], dtype=np.int64)
    print(f"train-only teacher rows {len(e_ids)} {dict(Counter(extra_rows.values()))}", flush=True)

    params = {"objective": "multiclass", "num_class": len(classes), "learning_rate": args.lr, "num_leaves": args.leaves,
              "min_data_in_leaf": args.min_data, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
              "num_threads": 2, "verbose": -1, "seed": 7}
    oof = np.zeros((len(t_ids), len(classes)))
    importance = np.zeros(width)
    for k in range(args.folds):
        test = gt == k
        x_train = np.vstack([xw, xe, xt[~test]])
        y_train = np.concatenate([yw, ye, yt[~test]])
        y_teacher = np.concatenate([ye, yt[~test]])
        if args.balance:
            freq = np.bincount(y_teacher, minlength=len(classes)).astype(float)
            class_w = (len(y_teacher) / (len(classes) * np.maximum(freq, 1))) ** 0.5
            teacher_w = class_w[y_teacher]
        else:
            teacher_w = np.ones(len(y_teacher))
        w_train = np.concatenate([np.full(len(yw), args.weak_weight), teacher_w])
        booster = lgb.train(params, lgb.Dataset(x_train, label=y_train, weight=w_train), num_boost_round=args.rounds)
        oof[test] = booster.predict(xt[test])
        importance += booster.feature_importance("gain")

    pred, conf = oof.argmax(axis=1), oof.max(axis=1)
    report: dict = {
        "config": {"featureSet": args.feature_set, "features": width, "weakWeight": args.weak_weight,
                   "teacherRows": len(t_ids), "weakRows": len(w_ids), "trainOnlyRows": len(e_ids), "tokens": args.tokens},
        "classes": classes, "teacherByClass": {c: int((yt == i).sum()) for i, c in enumerate(classes)},
        "accuracy": round(float(((pred == yt) * eval_w).sum() / eval_w.sum()), 4), "perClass": {}, "goalsMet": [],
    }
    for i, name in enumerate(classes):
        support = int((yt == i).sum())
        support_w = float(eval_w[yt == i].sum())
        best = None
        curve = []
        for bar in BARS:
            said = (pred == i) & (conf >= bar)
            n = int(said.sum())
            if n == 0:
                continue
            precision = float(eval_w[said & (yt == i)].sum() / eval_w[said].sum())
            recall = float(eval_w[said & (yt == i)].sum() / max(support_w, 1e-9))
            curve.append({"bar": bar, "answered": n, "precision": round(precision, 3), "recall": round(recall, 3)})
            if precision >= args.goal_precision and (best is None or recall > best["recall"]):
                best = curve[-1]
        report["perClass"][name] = {"support": support, "bestAtGoalPrecision": best, "curve": curve}
        if best and best["recall"] >= args.goal_recall:
            report["goalsMet"].append(name)
    top = np.argsort(-importance)[:12]
    report["topFeatures"] = {names[i]: round(float(importance[i] / importance.sum()), 3) for i in top}
    if args.pool_out:
        watched = {t for t, c in V8.TYPE_TO_V8.items() if c in ("article", "blog_post", "product_detail")}
        skip = set(fold_label) | set(extra_rows) | {json.loads(line)["p"] for f in args.pool_exclude for line in f.read_text().splitlines() if line.strip()}
        pool_ids = []
        for line in args.schema.read_text().splitlines():
            row = json.loads(line)
            if row.get("ok") and row["d"] not in sealed and not (set(row["ld"]) | set(row["ldNested"]) | set(row["md"])) & watched:
                p = row["sha"][:16]
                if p in features and p not in skip and p not in root:
                    pool_ids.append((p, row["d"]))
        x_all = np.vstack([xw, xe, xt])
        y_all = np.concatenate([yw, ye, yt])
        w_all = np.concatenate([np.full(len(yw), args.weak_weight), np.ones(len(ye) + len(yt))])
        final = lgb.train(params, lgb.Dataset(x_all, label=y_all, weight=w_all), num_boost_round=args.rounds)
        probs = final.predict(np.vstack([features[p][:width] for p, _ in pool_ids]))
        args.pool_out.write_text("".join(json.dumps({"p": p, "d": d, "probs": [round(float(v), 3) for v in probs[i]]}) + "\n"
                                         for i, (p, d) in enumerate(pool_ids)))
        (args.pool_out.parent / "pool_classes.json").write_text(json.dumps(classes))
        print(f"pool scored: {len(pool_ids)} pages", flush=True)
    if args.save_model:
        args.save_model.mkdir(parents=True, exist_ok=True)
        x_all = np.vstack([xw, xe, xt])
        y_all = np.concatenate([yw, ye, yt])
        w_all = np.concatenate([np.full(len(yw), args.weak_weight), np.ones(len(ye) + len(yt))])
        final = lgb.train(params, lgb.Dataset(x_all, label=y_all, weight=w_all), num_boost_round=args.rounds)
        final.save_model(str(args.save_model / "model.txt"))
        # a class that misses the goal at every bar abstains (threshold above 1): it never fires
        # ... and so does a class whose threshold rests on fewer than 10 held-out answers: that is not evidence
        def bar_for(name: str) -> float:
            best = report["perClass"][name]["bestAtGoalPrecision"]
            return best["bar"] if best and best["answered"] >= 10 else 1.01

        thresholds = [bar_for(c) for c in classes]
        (args.save_model / "meta.json").write_text(json.dumps({
            "classes": classes, "featureNames": names[:width], "classThresholds": thresholds,
            "goalPrecision": args.goal_precision, "evaluation": {c: (report["perClass"][c]["bestAtGoalPrecision"] if bar_for(c) <= 1 else None) for c in classes},
            "trainedOn": {"teacherRows": len(t_ids), "trainOnlyRows": len(e_ids), "weakRows": len(w_ids)},
            "tokens": args.tokens, "featureSet": args.feature_set}, indent=1) + "\n")
        print(f"saved model to {args.save_model}: thresholds {dict(zip(classes, thresholds))}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "oof.jsonl").write_text("".join(
        json.dumps({"p": t_ids[i], "true": classes[int(yt[i])], "pred": classes[int(pred[i])], "conf": round(float(conf[i]), 3),
                    "probs": [round(float(v), 3) for v in oof[i]]}) + "\n" for i in range(len(t_ids))))
    (args.out / "loop_eval.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"accuracy {report['accuracy']}  goals met (precision >= {args.goal_precision}, recall >= {args.goal_recall}): {report['goalsMet']}")
    for name, block in report["perClass"].items():
        b = block["bestAtGoalPrecision"]
        mid = next((c for c in block["curve"] if c["bar"] >= 0.8), None)
        print(f"  {name:14s} n={block['support']:4d}  at>=0.8: {mid}  best@{args.goal_precision}: {b}")
    print("top features:", report["topFeatures"])


if __name__ == "__main__":
    main()
