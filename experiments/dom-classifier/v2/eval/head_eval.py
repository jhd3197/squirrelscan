"""Page-type head eval at a fixed precision: one protocol for every candidate head (#2977).

Takes labels and scores only, never page text. The method is the phase 2 head-to-head, frozen:

1. Truth. Each judged page carries one truth label per judge variant; a class map folds those
   labels (and every head's own labels) onto one common vocabulary. A `consensus` variant keeps
   the pages where all variants agree on the mapped class.
2. Out-of-fold thresholds. Fold = domain bucket parity. Per head and class, the threshold applied
   to a fold is the lowest confidence whose cumulative precision on the OTHER fold reaches the
   target (DomainGenius `pick_threshold`; 1.01 when none does, so the class never answers). No
   page, and no fold, sets its own threshold.
3. Operating point. Coverage, precision and recall at those thresholds, with Wilson 95% lower
   bounds on precision.
4. Paired bootstrap. Pages are resampled (2,000 draws, seed 0) with the thresholds held fixed;
   each head minus the baseline is reported with a 95% percentile interval.
5. Repeated for every truth variant and for the schema-less and schema-declared subsets, each
   an independent analysis (its own folds, thresholds and resamples).
6. Leak check. `assert_no_overlap` counts the domains and page ids a head trained on that also
   appear in the eval rows, prints the count, and fails on any shared domain or page.

See README.md for the input formats.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]
StrArray = npt.NDArray[np.str_]
JsonObject = dict[str, object]
TiePolicy = Literal["prefix", "group"]

FOLDS = (0, 1)
NO_THRESHOLD = 1.01
NO_ANSWER = ""
CONSENSUS = "consensus"
DEFAULT_TARGET = 0.90
DEFAULT_N_BOOT = 2000
DEFAULT_SEED = 0
WILSON_Z = 1.96
SUBSETS = ("all", "schemaless", "schema_declared")
TIE_POLICIES: tuple[TiePolicy, ...] = ("prefix", "group")


class OverlapError(ValueError):
    """A head's training rows share a domain or a page with the eval rows."""


@dataclass(frozen=True)
class Item:
    page: str
    domain: str
    bucket: int
    schema: bool | None
    truth: Mapping[str, str]


@dataclass(frozen=True)
class RawAnswer:
    """One head's answer on one page, in the head's own labels. `label == NO_ANSWER`: none."""

    label: str
    confidence: float


@dataclass(frozen=True)
class HeadMap:
    labels: Mapping[str, str | None]
    default: str | None
    cannot_answer: frozenset[str]
    shipped_bars: Mapping[str, float]


@dataclass(frozen=True)
class ClassMap:
    classes: tuple[str, ...]
    other_class: str | None
    truth_labels: Mapping[str, str]
    truth_default: str
    heads: Mapping[str, HeadMap]


@dataclass(frozen=True)
class HeadArrays:
    """One head's answers on the common vocabulary, aligned with the items of a frame."""

    pred: StrArray
    conf: FloatArray
    raw: StrArray


@dataclass(frozen=True)
class Prepared:
    """Thresholded answers as float matrices, so each bootstrap draw is a few matrix products."""

    classes: tuple[str, ...]
    other_index: int | None
    acc: FloatArray
    corr: FloatArray
    onehot: FloatArray
    any_acc: FloatArray
    any_corr: FloatArray
    thresholds: Mapping[str, tuple[float, ...]]


@dataclass(frozen=True)
class Interval:
    """One paired-bootstrap row: both heads' observed values, their difference, a 95% interval."""

    base: float
    other: float
    diff: float
    lo: float
    hi: float


@dataclass(frozen=True)
class ShippedPoint:
    accepted: int
    tp: int
    precision: float
    lo: float
    recall: float


@dataclass(frozen=True)
class ClassStats:
    support: int
    accepted: int
    tp: int
    precision: float
    precision_lo: float
    recall: float


@dataclass(frozen=True)
class HeadResult:
    classes: Mapping[str, ClassStats]
    coverage: float
    precision: float
    precision_lo: float
    correct_coverage: float
    nonother: Mapping[str, float] | None
    thresholds_oof: Mapping[str, tuple[float, ...]]
    shipped: Mapping[str, ShippedPoint] | None


@dataclass(frozen=True)
class TruthResult:
    n: int
    support: Mapping[str, int]
    heads: Mapping[str, HeadResult]
    bootstrap: Mapping[str, Mapping[str, Interval]]


@dataclass(frozen=True)
class SubsetResult:
    n_pages: int
    truth: Mapping[str, TruthResult]


@dataclass(frozen=True)
class Summary:
    target: float
    ties: TiePolicy
    n_boot: int
    seed: int
    classes: tuple[str, ...]
    heads: tuple[str, ...]
    baseline: str
    variants: tuple[str, ...]
    items: int
    items_missing_a_variant: int
    no_output: Mapping[str, int]
    by_design: Mapping[str, list[str]]
    subsets: Mapping[str, SubsetResult]


# --- json plumbing ------------------------------------------------------------------------------


def as_object(value: object, where: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"{where}: expected an object")
    return {str(k): v for k, v in cast("dict[object, object]", value).items()}


def as_str(value: object, where: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{where}: expected a string")
    return value


def as_number(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: expected a number")
    return float(value)


def as_str_list(value: object, where: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{where}: expected a list")
    return [as_str(v, where) for v in cast("list[object]", value)]


def read_jsonl(path: Path) -> list[JsonObject]:
    rows: list[JsonObject] = []
    with path.open(encoding="utf8") as handle:
        for number, line in enumerate(handle, start=1):
            if line.strip():
                rows.append(as_object(json.loads(line), f"{path}:{number}"))
    return rows


# --- class map ----------------------------------------------------------------------------------


def parse_class_map(raw: JsonObject) -> ClassMap:
    classes = tuple(as_str_list(raw.get("classes"), "classes"))
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("classes: need a non-empty list without repeats")
    other = raw.get("other_class", "other")
    other_class = None if other is None else as_str(other, "other_class")
    if other_class is not None and other_class not in classes:
        raise ValueError(f"other_class {other_class!r} is not in classes")
    truth = as_object(raw.get("truth"), "truth")
    truth_map = as_object(truth.get("map", {}), "truth.map")
    truth_labels = {k: as_str(v, f"truth.map.{k}") for k, v in truth_map.items()}
    truth_default = as_str(truth.get("default"), "truth.default")
    for target in (*truth_labels.values(), truth_default):
        if target not in classes:
            raise ValueError(f"truth maps to {target!r}, which is not in classes")
    heads: dict[str, HeadMap] = {}
    for name, spec_raw in as_object(raw.get("heads"), "heads").items():
        spec = as_object(spec_raw, f"heads.{name}")
        labels: dict[str, str | None] = {}
        for label, target in as_object(spec.get("map", {}), f"heads.{name}.map").items():
            labels[label] = None if target is None else as_str(target, f"heads.{name}.map.{label}")
        default_raw = spec.get("default")
        default = None if default_raw is None else as_str(default_raw, f"heads.{name}.default")
        cannot = frozenset(
            as_str_list(spec.get("cannot_answer", []), f"heads.{name}.cannot_answer")
        )
        bars = {
            label: as_number(bar, f"heads.{name}.shipped_bars.{label}")
            for label, bar in as_object(spec.get("shipped_bars", {}), f"heads.{name}.bars").items()
        }
        for target in (*(t for t in labels.values() if t is not None), *cannot):
            if target not in classes:
                raise ValueError(f"heads.{name}: {target!r} is not in classes")
        if default is not None and default not in classes:
            raise ValueError(f"heads.{name}.default {default!r} is not in classes")
        heads[name] = HeadMap(labels, default, cannot, bars)
    return ClassMap(classes, other_class, truth_labels, truth_default, heads)


def truth_class(class_map: ClassMap, label: str) -> str:
    """A truth label on the common vocabulary: the map, else itself when a class, else default."""
    if label in class_map.truth_labels:
        return class_map.truth_labels[label]
    return label if label in class_map.classes else class_map.truth_default


def head_class(class_map: ClassMap, head: str, label: str) -> str:
    """A head label on the common vocabulary; `NO_ANSWER` when the head or the map gives none."""
    if label == NO_ANSWER:
        return NO_ANSWER
    spec = class_map.heads[head]
    if label in spec.labels:
        mapped = spec.labels[label]
    elif label in class_map.classes:
        mapped = label
    else:
        mapped = spec.default
    return NO_ANSWER if mapped is None else mapped


# --- inputs -------------------------------------------------------------------------------------


def parse_items(rows: Iterable[JsonObject]) -> list[Item]:
    items: list[Item] = []
    seen: set[str] = set()
    for number, row in enumerate(rows, start=1):
        page = as_str(row.get("p"), f"item {number}.p")
        if page in seen:
            raise ValueError(f"item {number}: page id {page!r} appears twice")
        seen.add(page)
        bucket = row.get("bucket")
        if isinstance(bucket, bool) or not isinstance(bucket, int):
            raise ValueError(f"item {page}: bucket must be an integer")
        schema = row.get("schema")
        if schema is not None and not isinstance(schema, bool):
            raise ValueError(f"item {page}: schema must be true, false or null")
        truth = {
            variant: as_str(label, f"item {page}.truth.{variant}")
            for variant, label in as_object(row.get("truth"), f"item {page}.truth").items()
        }
        if CONSENSUS in truth:
            raise ValueError(f"item {page}: {CONSENSUS!r} is a reserved truth variant name")
        items.append(Item(page, as_str(row.get("d"), f"item {page}.d"), bucket, schema, truth))
    return items


def parse_answers(rows: Iterable[JsonObject], where: str) -> dict[str, RawAnswer]:
    """First row per page wins. Needs `label` and `confidence`, or `probs` to derive them."""
    answers: dict[str, RawAnswer] = {}
    for number, row in enumerate(rows, start=1):
        page = as_str(row.get("p"), f"{where} row {number}.p")
        if page in answers:
            continue
        label_raw = row.get("label")
        label = NO_ANSWER if label_raw is None else as_str(label_raw, f"{where} {page}.label")
        probs_raw = row.get("probs")
        probs = (
            {
                k: as_number(v, f"{where} {page}.probs.{k}")
                for k, v in as_object(probs_raw, "probs").items()
            }
            if probs_raw is not None
            else {}
        )
        if label == NO_ANSWER and probs:
            label = max(probs, key=lambda key: probs[key])
        if label == NO_ANSWER:
            answers[page] = RawAnswer(NO_ANSWER, 0.0)
            continue
        if row.get("confidence") is not None:
            confidence = as_number(row.get("confidence"), f"{where} {page}.confidence")
        elif label in probs:
            confidence = probs[label]
        else:
            raise ValueError(f"{where} {page}: a label needs a confidence or probs[label]")
        answers[page] = RawAnswer(label, confidence)
    return answers


# --- leak check ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Overlap:
    domains: int
    pages: int
    train_domains: int
    train_pages: int


def overlap_counts(train_rows: Iterable[Mapping[str, str]], items: Sequence[Item]) -> Overlap:
    """Domains and page ids shared by a head's training rows (`p`, `d`) and the eval items."""
    train_pages: set[str] = set()
    train_domains: set[str] = set()
    for row in train_rows:
        if row.get("p"):
            train_pages.add(row["p"])
        if row.get("d"):
            train_domains.add(row["d"])
    shared_pages = train_pages & {item.page for item in items}
    shared_domains = train_domains & {item.domain for item in items}
    return Overlap(len(shared_domains), len(shared_pages), len(train_domains), len(train_pages))


def assert_no_overlap(
    name: str,
    train_rows: Iterable[Mapping[str, str]],
    items: Sequence[Item],
    *,
    drop_page_overlap: bool = False,
    log: list[str] | None = None,
) -> list[Item]:
    """Print the overlap count of one head's training rows with the eval rows; fail when nonzero.

    A shared domain always fails. A shared page id (same content served under another domain)
    fails too, unless `drop_page_overlap` is set: then those pages leave the eval rows and the
    count of dropped pages is printed. Returns the eval items to keep.
    """
    rows = list(train_rows)
    found = overlap_counts(rows, items)
    message = (
        f"overlap {name}: training rows cover {found.train_pages} pages / {found.train_domains} "
        f"domains; eval rows share {found.pages} pages and {found.domains} domains"
    )
    print(message)
    if log is not None:
        log.append(message)
    if found.domains:
        raise OverlapError(f"{message}: a training domain is in the eval rows")
    if not found.pages:
        return list(items)
    if not drop_page_overlap:
        raise OverlapError(f"{message}: a training page is in the eval rows")
    shared = {row["p"] for row in rows if row.get("p")} & {item.page for item in items}
    print(f"overlap {name}: dropping {len(shared)} shared page(s) from the eval rows")
    return [item for item in items if item.page not in shared]


# --- the method ---------------------------------------------------------------------------------


def wilson_lower(successes: float, total: float, z: float = WILSON_Z) -> float:
    """Lower bound of the Wilson score interval; NaN when there is nothing to measure."""
    if not total:
        return math.nan
    p = successes / total
    denom = 1 + z * z / total
    centre = p + z * z / (2 * total)
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (centre - spread) / denom


def pick_threshold(
    conf: FloatArray, ok: BoolArray, target: float, ties: TiePolicy = "prefix"
) -> float:
    """Lowest confidence whose accepted set (conf >= t) has precision >= target; 1.01 when none.

    `prefix` (the phase 2 and DomainGenius rule) checks the cumulative precision at every rank, so
    inside a group of equal confidences the row order decides, and the accepted set `conf >= t`
    can end up below the target. `group` only lets the last row of each tie group qualify, so the
    precision of the accepted set itself reaches the target. They agree when no confidences tie.
    """
    if conf.size == 0:
        return NO_THRESHOLD
    order = np.argsort(-conf, kind="stable")
    ranked = conf[order]
    cumulative = np.cumsum(ok[order].astype(np.float64)) / np.arange(1, conf.size + 1)
    qualifies = cumulative >= target
    if ties == "group":
        qualifies &= np.append(ranked[1:] != ranked[:-1], True)
    good = np.flatnonzero(qualifies)
    return NO_THRESHOLD if good.size == 0 else float(ranked[good[-1]])


def oof_accept(
    pred: StrArray,
    conf: FloatArray,
    truth: StrArray,
    fold: IntArray,
    classes: Sequence[str],
    target: float,
    ties: TiePolicy = "prefix",
) -> tuple[BoolArray, dict[str, tuple[float, ...]]]:
    """Per class, the threshold for each fold is picked on the other fold and applied to this one.

    Returns the (class, page) accepted matrix and `thresholds[class][fold]`.
    """
    accepted = np.zeros((len(classes), pred.size), dtype=bool)
    thresholds: dict[str, tuple[float, ...]] = {}
    for index, name in enumerate(classes):
        answered = pred == name
        right = truth == name
        per_fold: list[float] = []
        for current in FOLDS:
            pool = answered & (fold != current)
            threshold = pick_threshold(conf[pool], right[pool], target, ties)
            per_fold.append(threshold)
            here = fold == current
            accepted[index, here] = answered[here] & (conf[here] >= threshold)
        thresholds[name] = tuple(per_fold)
    return accepted, thresholds


def prepare(
    pred: StrArray,
    conf: FloatArray,
    truth: StrArray,
    fold: IntArray,
    classes: tuple[str, ...],
    other_class: str | None,
    target: float,
    ties: TiePolicy = "prefix",
) -> Prepared:
    accepted, thresholds = oof_accept(pred, conf, truth, fold, classes, target, ties)
    onehot = truth[None, :] == np.array(classes)[:, None]
    correct = accepted & onehot
    return Prepared(
        classes=classes,
        other_index=None if other_class is None else classes.index(other_class),
        acc=accepted.astype(np.float64),
        corr=correct.astype(np.float64),
        onehot=onehot.astype(np.float64),
        any_acc=accepted.any(axis=0).astype(np.float64),
        any_corr=correct.any(axis=0).astype(np.float64),
        thresholds=thresholds,
    )


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


def metrics(prepared: Prepared, weights: FloatArray) -> dict[str, float]:
    """Flat metric table for one head under page weights (all ones = the observed table)."""
    total = float(weights.sum())
    accepted = prepared.acc @ weights
    right = prepared.corr @ weights
    support = prepared.onehot @ weights
    out: dict[str, float] = {}
    for index, name in enumerate(prepared.classes):
        out[f"{name}:support"] = float(support[index])
        out[f"{name}:accepted"] = float(accepted[index])
        out[f"{name}:tp"] = float(right[index])
        out[f"{name}:precision"] = ratio(float(right[index]), float(accepted[index]))
        out[f"{name}:recall"] = ratio(float(right[index]), float(support[index]))
    all_accepted = float(prepared.any_acc @ weights)
    all_right = float(prepared.any_corr @ weights)
    out["all:accepted"] = all_accepted
    out["all:tp"] = all_right
    out["all:coverage"] = ratio(all_accepted, total)
    out["all:precision"] = ratio(all_right, all_accepted)
    out["all:correct_coverage"] = ratio(all_right, total)
    if prepared.other_index is not None:
        rest = [i for i in range(len(prepared.classes)) if i != prepared.other_index]
        rest_accepted = float(accepted[rest].sum())
        rest_right = float(right[rest].sum())
        out["nonother:precision"] = ratio(rest_right, rest_accepted)
        out["nonother:recall"] = ratio(rest_right, float(support[rest].sum()))
        out["nonother:accepted_share"] = ratio(rest_accepted, total)
    return out


def diff_keys(classes: Sequence[str], has_other: bool) -> list[str]:
    keys = [f"{c}:recall" for c in classes] + [f"{c}:precision" for c in classes]
    keys += ["all:coverage", "all:precision", "all:correct_coverage"]
    if has_other:
        keys += ["nonother:precision", "nonother:recall"]
    return keys


def paired_bootstrap(
    base: Prepared,
    other: Prepared,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> dict[str, Interval]:
    """`other` minus `base` over resampled pages, thresholds fixed; 95% percentile interval.

    One resample (a page-count vector) feeds both heads, so the draws are paired. A metric that
    is undefined in more than half the draws (a class a head never accepts) is left out.
    """
    n_pages = base.acc.shape[1]
    keys = diff_keys(base.classes, base.other_index is not None)
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, len(keys)))
    for draw in range(n_boot):
        counts = np.bincount(rng.integers(0, n_pages, n_pages), minlength=n_pages)
        weights = counts.astype(np.float64)
        low, high = metrics(base, weights), metrics(other, weights)
        draws[draw] = [high[key] - low[key] for key in keys]
    ones = np.ones(n_pages)
    observed_base, observed_other = metrics(base, ones), metrics(other, ones)
    result: dict[str, Interval] = {}
    for column, key in enumerate(keys):
        values = draws[:, column]
        values = values[~np.isnan(values)]
        if values.size < n_boot * 0.5:
            continue
        lower, upper = np.percentile(values, [2.5, 97.5])
        result[key] = Interval(
            base=observed_base[key],
            other=observed_other[key],
            diff=observed_other[key] - observed_base[key],
            lo=float(lower),
            hi=float(upper),
        )
    return result


def shipped_points(
    head: HeadArrays,
    truth: StrArray,
    classes: Sequence[str],
    bars: Mapping[str, float],
) -> dict[str, ShippedPoint]:
    """Reference only: the head's own shipped per-label bars (keyed by the head's raw label)."""
    bar = np.array([bars.get(label, math.inf) for label in head.raw])
    out: dict[str, ShippedPoint] = {}
    for name in classes:
        picked = (head.pred == name) & (head.conf >= bar)
        right = int((picked & (truth == name)).sum())
        count = int(picked.sum())
        support = int((truth == name).sum())
        out[name] = ShippedPoint(
            accepted=count,
            tp=right,
            precision=ratio(right, count),
            lo=wilson_lower(right, count),
            recall=ratio(right, support),
        )
    return out


# --- frames and evaluation ----------------------------------------------------------------------


@dataclass(frozen=True)
class Frame:
    """The pages of one subset, aligned with every head's answers."""

    fold: IntArray
    truth: Mapping[str, StrArray]
    heads: Mapping[str, HeadArrays]


def head_arrays(
    class_map: ClassMap,
    head: str,
    answers: Mapping[str, RawAnswer],
    items: Sequence[Item],
) -> HeadArrays:
    spec = class_map.heads[head]
    pred: list[str] = []
    conf: list[float] = []
    raw: list[str] = []
    for item in items:
        answer = answers.get(item.page, RawAnswer(NO_ANSWER, 0.0))
        mapped = head_class(class_map, head, answer.label)
        if mapped in spec.cannot_answer:
            raise ValueError(f"head {head} answered {mapped!r}, which the class map says it cannot")
        pred.append(mapped)
        conf.append(answer.confidence if mapped != NO_ANSWER else 0.0)
        raw.append(answer.label)
    return HeadArrays(
        np.array(pred, dtype=str), np.array(conf, dtype=np.float64), np.array(raw, dtype=str)
    )


def build_frame(
    items: Sequence[Item],
    variants: Sequence[str],
    answers: Mapping[str, Mapping[str, RawAnswer]],
    class_map: ClassMap,
) -> Frame:
    truth = {
        variant: np.array([truth_class(class_map, i.truth[variant]) for i in items], dtype=str)
        for variant in variants
    }
    if len(variants) > 1:
        first = truth[variants[0]]
        agree = np.ones(first.size, dtype=bool)
        for variant in variants[1:]:
            agree &= truth[variant] == first
        truth[CONSENSUS] = np.where(agree, first, NO_ANSWER).astype(str)
    return Frame(
        fold=np.array([item.bucket % 2 for item in items], dtype=np.int64),
        truth=truth,
        heads={name: head_arrays(class_map, name, answers[name], items) for name in answers},
    )


def head_result(
    prepared: Prepared,
    classes: Sequence[str],
    shipped: Mapping[str, ShippedPoint] | None,
) -> HeadResult:
    flat = metrics(prepared, np.ones(prepared.acc.shape[1]))
    per_class = {
        name: ClassStats(
            support=round(flat[f"{name}:support"]),
            accepted=round(flat[f"{name}:accepted"]),
            tp=round(flat[f"{name}:tp"]),
            precision=flat[f"{name}:precision"],
            precision_lo=wilson_lower(flat[f"{name}:tp"], flat[f"{name}:accepted"]),
            recall=flat[f"{name}:recall"],
        )
        for name in classes
    }
    nonother = None
    if "nonother:precision" in flat:
        nonother = {
            "precision": flat["nonother:precision"],
            "recall": flat["nonother:recall"],
            "accepted_share": flat["nonother:accepted_share"],
        }
    return HeadResult(
        classes=per_class,
        coverage=flat["all:coverage"],
        precision=flat["all:precision"],
        precision_lo=wilson_lower(flat["all:tp"], flat["all:accepted"]),
        correct_coverage=flat["all:correct_coverage"],
        nonother=nonother,
        thresholds_oof=prepared.thresholds,
        shipped=shipped,
    )


def evaluate_frame(
    frame: Frame,
    class_map: ClassMap,
    baseline: str,
    target: float,
    n_boot: int,
    seed: int,
    ties: TiePolicy = "prefix",
) -> dict[str, TruthResult]:
    classes = class_map.classes
    out: dict[str, TruthResult] = {}
    for variant, truth_all in frame.truth.items():
        keep = np.flatnonzero(truth_all != NO_ANSWER)
        if keep.size == 0:
            continue
        truth = truth_all[keep]
        fold = frame.fold[keep]
        prepared = {
            name: prepare(
                h.pred[keep],
                h.conf[keep],
                truth,
                fold,
                classes,
                class_map.other_class,
                target,
                ties,
            )
            for name, h in frame.heads.items()
        }
        shipped: dict[str, Mapping[str, ShippedPoint]] = {}
        for name, h in frame.heads.items():
            bars = class_map.heads[name].shipped_bars
            if bars:
                kept = HeadArrays(h.pred[keep], h.conf[keep], h.raw[keep])
                shipped[name] = shipped_points(kept, truth, classes, bars)
        out[variant] = TruthResult(
            n=int(keep.size),
            support={c: int((truth == c).sum()) for c in classes},
            heads={
                name: head_result(p, classes, shipped.get(name)) for name, p in prepared.items()
            },
            bootstrap={
                f"{name}_vs_{baseline}": paired_bootstrap(prepared[baseline], p, n_boot, seed)
                for name, p in prepared.items()
                if name != baseline
            },
        )
    return out


def subset_items(items: Sequence[Item], subset: str) -> list[Item]:
    if subset == "all":
        return list(items)
    wanted = subset == "schema_declared"
    return [item for item in items if item.schema is not None and item.schema == wanted]


def run_eval(
    items: Sequence[Item],
    answers: Mapping[str, Mapping[str, RawAnswer]],
    class_map: ClassMap,
    *,
    baseline: str,
    target: float = DEFAULT_TARGET,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    ties: TiePolicy = "prefix",
) -> Summary:
    """The whole protocol over already-checked items."""
    if baseline not in answers:
        raise ValueError(f"baseline {baseline!r} is not one of the heads {sorted(answers)}")
    if not items:
        raise ValueError("no eval items")
    variants = list(items[0].truth)
    complete = [item for item in items if all(v in item.truth for v in variants)]
    subsets: dict[str, SubsetResult] = {}
    for subset in SUBSETS:
        chosen = subset_items(complete, subset)
        if not chosen:
            continue
        frame = build_frame(chosen, variants, answers, class_map)
        subsets[subset] = SubsetResult(
            n_pages=len(chosen),
            truth=evaluate_frame(frame, class_map, baseline, target, n_boot, seed, ties),
        )
    return Summary(
        target=target,
        ties=ties,
        n_boot=n_boot,
        seed=seed,
        classes=class_map.classes,
        heads=tuple(answers),
        baseline=baseline,
        variants=(*variants, *([CONSENSUS] if len(variants) > 1 else [])),
        items=len(items),
        items_missing_a_variant=len(items) - len(complete),
        no_output={
            name: sum(1 for item in complete if item.page not in found)
            for name, found in answers.items()
        },
        by_design={name: sorted(class_map.heads[name].cannot_answer) for name in answers},
        subsets=subsets,
    )


# --- output -------------------------------------------------------------------------------------


def jsonable(value: object) -> object:
    """NaN and infinity become null, so the summary is strict JSON."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in cast("dict[object, object]", value).items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in cast("Sequence[object]", value)]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def fmt(value: float, digits: int = 2) -> str:
    return "-" if math.isnan(value) else f"{value:.{digits}f}"


def render_table(summary: Summary) -> str:
    """The short text table: per subset and truth variant, the operating point of every head."""
    classes = summary.classes
    lines = [
        f"target precision {summary.target} (ties: {summary.ties}); "
        f"paired bootstrap {summary.n_boot} draws, seed {summary.seed}; "
        f"baseline {summary.baseline}; ~ = class the head cannot answer"
    ]
    for subset, subset_result in summary.subsets.items():
        for variant, block in subset_result.truth.items():
            support = ", ".join(f"{c} {block.support[c]}" for c in classes)
            lines.append("")
            lines.append(f"## subset {subset}, truth {variant} (n={block.n}), support: {support}")
            header = f"{'class':16s}{'support':>8s}"
            for head in summary.heads:
                header += f" | {head}: acc  prec    lo recall    thr(fold0/fold1)"
            lines.append(header)
            for name in classes:
                row = f"{name:16s}{block.support[name]:8d}"
                for head in summary.heads:
                    result = block.heads[head]
                    cell = result.classes[name]
                    marker = "~" if name in summary.by_design[head] else " "
                    thresholds = "/".join(fmt(t, 3) for t in result.thresholds_oof[name])
                    row += (
                        f" | {cell.accepted:{len(head) + 6}d} {fmt(cell.precision):>5s} "
                        f"{fmt(cell.precision_lo):>5s} {fmt(cell.recall):>5s}{marker} "
                        f"{thresholds:>14s}"
                    )
                lines.append(row)
            for head in summary.heads:
                result = block.heads[head]
                lines.append(
                    f"  {head}: coverage {fmt(result.coverage, 3)}  precision "
                    f"{fmt(result.precision, 3)} (Wilson lo {fmt(result.precision_lo, 3)})  "
                    f"correct coverage {fmt(result.correct_coverage, 3)}"
                )
            for pair, intervals in block.bootstrap.items():
                lines.append(f"  paired bootstrap {pair} (95% interval; * excludes 0)")
                for key, entry in intervals.items():
                    if not (key.endswith(":recall") or key.startswith(("all:", "nonother:"))):
                        continue
                    flag = " *" if entry.lo > 0 or entry.hi < 0 else ""
                    lines.append(
                        f"    {key:24s} base {fmt(entry.base, 3)}  other {fmt(entry.other, 3)}  "
                        f"diff {entry.diff:+.3f} [{entry.lo:+.3f}, {entry.hi:+.3f}]{flag}"
                    )
    return "\n".join(lines) + "\n"


# --- cli ----------------------------------------------------------------------------------------


class Args(argparse.Namespace):
    items: Path
    class_map: Path
    head: list[str]
    baseline: str | None
    train_rows: list[str]
    drop_page_overlap: bool
    target: float
    ties: TiePolicy
    n_boot: int
    seed: int
    out: Path
    table: Path | None


def split_pair(value: str, flag: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise ValueError(f"{flag} expects NAME=PATH, got {value!r}")
    return name, Path(path)


def parse_args(argv: Sequence[str] | None) -> Args:
    parser = argparse.ArgumentParser(
        description="Page-type head eval at a fixed precision. See README.md.",
    )
    parser.add_argument("--items", type=Path, required=True, help="judged items jsonl")
    parser.add_argument("--class-map", type=Path, required=True, help="class map json")
    parser.add_argument(
        "--head", action="append", required=True, metavar="NAME=PATH", help="head outputs jsonl"
    )
    parser.add_argument("--baseline", help="head the others are compared with (default: first)")
    parser.add_argument(
        "--train-rows",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="a head's training rows (p, d) jsonl; checked for overlap with the eval rows",
    )
    parser.add_argument(
        "--drop-page-overlap",
        action="store_true",
        help="drop eval pages that are training pages instead of failing (a shared domain fails)",
    )
    parser.add_argument("--target", type=float, default=DEFAULT_TARGET, help="target precision")
    parser.add_argument(
        "--ties",
        choices=TIE_POLICIES,
        default="prefix",
        help="threshold rule inside groups of equal confidences (default: the phase 2 rule)",
    )
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, required=True, help="summary json")
    parser.add_argument("--table", type=Path, help="also write the text table here")
    return parser.parse_args(argv, namespace=Args())


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0 < args.target <= 1:
        print("--target must be in (0, 1]", file=sys.stderr)
        return 2
    class_map = parse_class_map(
        as_object(json.loads(args.class_map.read_text(encoding="utf8")), "class map")
    )
    items = parse_items(read_jsonl(args.items))
    head_paths = [split_pair(value, "--head") for value in args.head]
    answers: dict[str, Mapping[str, RawAnswer]] = {}
    for name, path in head_paths:
        if name not in class_map.heads:
            print(f"head {name!r} is not in the class map", file=sys.stderr)
            return 2
        answers[name] = parse_answers(read_jsonl(path), name)
    try:
        for value in args.train_rows:
            name, path = split_pair(value, "--train-rows")
            rows = [{k: str(v) for k, v in row.items()} for row in read_jsonl(path)]
            items = assert_no_overlap(name, rows, items, drop_page_overlap=args.drop_page_overlap)
    except OverlapError as error:
        print(f"FAIL {error}", file=sys.stderr)
        return 1
    summary = run_eval(
        items,
        answers,
        class_map,
        baseline=args.baseline or head_paths[0][0],
        target=args.target,
        ties=args.ties,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    _ = args.out.write_text(json.dumps(jsonable(asdict(summary)), indent=1) + "\n", encoding="utf8")
    table = render_table(summary)
    if args.table:
        _ = args.table.write_text(table, encoding="utf8")
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
