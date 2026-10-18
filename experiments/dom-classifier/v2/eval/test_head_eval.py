"""Tests for head_eval.py on synthetic data only: no page text, no real domains."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

import head_eval as ev

CLASSES = ("alpha", "beta", "other")


def class_map(**overrides: object) -> ev.ClassMap:
    raw: dict[str, object] = {
        "classes": list(CLASSES),
        "other_class": "other",
        "truth": {"map": {"alpha_long": "alpha"}, "default": "other"},
        "heads": {
            "old": {"map": {"home": None}, "cannot_answer": ["beta"]},
            "new": {"map": {}, "shipped_bars": {"alpha": 0.5, "beta": 0.5, "other": 0.5}},
        },
    }
    raw.update(overrides)
    return ev.parse_class_map(ev.as_object(raw, "class map"))


def item(
    page: str, bucket: int, truth: str, *, schema: bool | None = None, url: str | None = None
) -> ev.Item:
    return ev.Item(page, f"d{page}", bucket, schema, {"nourl": truth, "url": url or truth})


def synthetic_items(n: int = 60) -> list[ev.Item]:
    """Classes cycle with period 3 and buckets with period 2: every class sits in both folds."""
    return [item(f"p{i}", i, CLASSES[i % 3], schema=i % 4 < 2) for i in range(n)]


def answers_of(
    items: list[ev.Item], label_for: Mapping[str, str], confidence: float = 0.9
) -> dict[str, ev.RawAnswer]:
    return {
        i.page: ev.RawAnswer(label_for.get(i.page, i.truth["nourl"]), confidence) for i in items
    }


def arrays(
    pred: list[str], conf: list[float], truth: list[str], fold: list[int]
) -> tuple[ev.StrArray, ev.FloatArray, ev.StrArray, ev.IntArray]:
    return (
        np.array(pred, dtype=str),
        np.array(conf, dtype=np.float64),
        np.array(truth, dtype=str),
        np.array(fold, dtype=np.int64),
    )


def head_prepared(wrong: set[int], n: int = 120) -> ev.Prepared:
    """Right answers carry confidence 0.95; wrong ones distinct lower values: thresholds exist."""
    truth = [CLASSES[i % 3] for i in range(n)]
    pred = [
        ("other" if truth[i] != "other" else "alpha") if i in wrong else truth[i] for i in range(n)
    ]
    conf = [0.1 + 0.005 * i if i in wrong else 0.95 for i in range(n)]
    p, c, t, f = arrays(pred, conf, truth, [i % 2 for i in range(n)])
    return ev.prepare(p, c, t, f, CLASSES, "other", 0.9)


def run_two_heads(items: list[ev.Item]) -> ev.Summary:
    return ev.run_eval(
        items,
        {"old": answers_of(items, {}), "new": answers_of(items, {})},
        class_map(heads={"old": {}, "new": {}}),
        baseline="old",
        n_boot=50,
    )


# --- thresholds ---------------------------------------------------------------------------------


def test_pick_threshold_is_the_lowest_confidence_that_keeps_precision() -> None:
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    ok = np.array([True, True, False, True])
    assert ev.pick_threshold(conf, ok, 0.75) == 0.6
    assert ev.pick_threshold(conf, ok, 0.9) == 0.8
    assert ev.pick_threshold(conf, np.zeros(4, dtype=bool), 0.9) == ev.NO_THRESHOLD
    assert ev.pick_threshold(np.array([]), np.array([], dtype=bool), 0.9) == ev.NO_THRESHOLD


def test_tie_policies_differ_only_inside_a_group_of_equal_confidences() -> None:
    tied = np.array([0.9, 0.9])
    # Prefix rule: the first row alone clears 0.9, so the threshold 0.9 admits both rows.
    assert ev.pick_threshold(tied, np.array([True, False]), 0.9, "prefix") == 0.9
    assert ev.pick_threshold(tied, np.array([False, True]), 0.9, "prefix") == ev.NO_THRESHOLD
    # Group rule: only the whole tie group qualifies, so the answer ignores row order.
    for ok in ([True, False], [False, True]):
        assert ev.pick_threshold(tied, np.array(ok), 0.9, "group") == ev.NO_THRESHOLD
    assert ev.pick_threshold(tied, np.array([True, True]), 0.9, "group") == 0.9
    # With no ties the two rules are the same.
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    ok = np.array([True, True, False, True])
    for target in (0.6, 0.75, 0.9, 1.0):
        assert ev.pick_threshold(conf, ok, target, "group") == ev.pick_threshold(
            conf, ok, target, "prefix"
        )
    # Mixed: the group at 0.7 drags precision to 4/6, so only the 0.8 boundary qualifies at 0.9.
    conf = np.array([0.9, 0.8, 0.7, 0.7, 0.7, 0.7])
    ok = np.array([True, True, True, False, False, True])
    assert ev.pick_threshold(conf, ok, 0.9, "group") == 0.8
    assert ev.pick_threshold(conf, ok, 0.6, "group") == 0.7


def test_oof_thresholds_never_use_the_evaluated_fold() -> None:
    n = 40
    fold = [i % 2 for i in range(n)]
    # Within a fold the confidence rises with the page; truth is right for the top half only.
    conf = [0.3 + 0.6 * (i // 2) / 19 for i in range(n)]
    truth = ["alpha" if c >= 0.6 else "beta" for c in conf]
    pred = ["alpha"] * n
    clean_accepted, clean = ev.oof_accept(*arrays(pred, conf, truth, fold), CLASSES, 0.9)
    # Ruin every fold-0 answer: only the threshold picked ON fold 0 (applied to fold 1) may move.
    ruined_truth = ["beta" if fold[i] == 0 else truth[i] for i in range(n)]
    ruined_accepted, ruined = ev.oof_accept(*arrays(pred, conf, ruined_truth, fold), CLASSES, 0.9)
    assert clean["alpha"][0] < ev.NO_THRESHOLD
    assert ruined["alpha"][0] == clean["alpha"][0]
    assert ruined["alpha"][1] == ev.NO_THRESHOLD
    assert clean["alpha"][1] < ev.NO_THRESHOLD
    on_fold_0 = np.array(fold) == 0
    assert np.array_equal(ruined_accepted[:, on_fold_0], clean_accepted[:, on_fold_0])
    assert not ruined_accepted[:, ~on_fold_0].any()


def test_a_class_never_predicted_in_the_other_fold_never_answers() -> None:
    fold = [0, 0, 0, 1, 1, 1]
    pred = ["alpha", "alpha", "alpha", "beta", "beta", "beta"]
    accepted, thresholds = ev.oof_accept(*arrays(pred, [0.99] * 6, pred, fold), CLASSES, 0.9)
    assert thresholds["alpha"] == (ev.NO_THRESHOLD, 0.99)
    assert thresholds["beta"] == (0.99, ev.NO_THRESHOLD)
    assert not accepted.any()


# --- operating point ----------------------------------------------------------------------------


def test_wilson_lower_matches_known_values() -> None:
    assert ev.wilson_lower(5, 5) == pytest.approx(0.5655, abs=1e-4)
    assert ev.wilson_lower(0, 10) == pytest.approx(0.0, abs=1e-9)
    assert math.isnan(ev.wilson_lower(0, 0))


def test_a_head_that_answers_everything_correctly_reaches_full_coverage() -> None:
    summary = run_two_heads(synthetic_items())
    assert set(summary.subsets) == {"all", "schemaless", "schema_declared"}
    for subset in summary.subsets.values():
        assert set(subset.truth) == {"nourl", "url", "consensus"}
        for block in subset.truth.values():
            for result in block.heads.values():
                assert result.coverage == 1.0
                assert result.precision == 1.0
                assert result.correct_coverage == 1.0
                assert all(stats.recall == 1.0 for stats in result.classes.values())


def test_no_answer_counts_against_coverage() -> None:
    items = synthetic_items()
    home = {i.page: "home" for i in items if i.truth["nourl"] == "other"}
    summary = ev.run_eval(
        items,
        {"old": answers_of(items, home), "new": answers_of(items, {})},
        class_map(heads={"old": {"map": {"home": None}}, "new": {}}),
        baseline="new",
        n_boot=20,
    )
    result = summary.subsets["all"].truth["nourl"].heads["old"]
    assert result.coverage == pytest.approx(2 / 3)
    assert result.precision == 1.0
    assert result.classes["other"].recall == 0.0


def test_a_head_answering_a_class_it_cannot_answer_is_a_class_map_error() -> None:
    items = synthetic_items()
    with pytest.raises(ValueError, match="cannot"):
        ev.run_eval(
            items,
            {"old": answers_of(items, {})},
            class_map(heads={"old": {"cannot_answer": ["beta"]}}),
            baseline="old",
            n_boot=20,
        )


def test_by_design_classes_are_reported_and_marked_in_the_table() -> None:
    items = synthetic_items()
    no_beta = {i.page: "other" for i in items if i.truth["nourl"] == "beta"}
    summary = ev.run_eval(
        items,
        {"old": answers_of(items, no_beta), "new": answers_of(items, {})},
        class_map(),
        baseline="new",
        n_boot=20,
    )
    assert summary.by_design == {"old": ["beta"], "new": []}
    row = next(r for r in ev.render_table(summary).splitlines() if r.startswith("beta "))
    assert "~" in row


def test_class_map_resolves_truth_and_head_labels() -> None:
    cm = class_map()
    assert ev.truth_class(cm, "alpha_long") == "alpha"
    assert ev.truth_class(cm, "beta") == "beta"
    assert ev.truth_class(cm, "unknown") == "other"
    assert ev.head_class(cm, "old", "home") == ev.NO_ANSWER
    assert ev.head_class(cm, "old", "alpha") == "alpha"
    assert ev.head_class(cm, "old", "unknown") == ev.NO_ANSWER
    assert ev.head_class(cm, "old", ev.NO_ANSWER) == ev.NO_ANSWER
    with pytest.raises(ValueError, match="not in classes"):
        class_map(truth={"map": {"x": "gamma"}, "default": "other"})


def test_consensus_keeps_only_pages_the_variants_agree_on() -> None:
    items = [item("a", 0, "alpha", url="alpha"), item("b", 1, "alpha", url="beta")]
    frame = ev.build_frame(items, ["nourl", "url"], {"new": {}}, class_map())
    assert list(frame.truth[ev.CONSENSUS]) == ["alpha", ev.NO_ANSWER]


def test_shipped_bars_are_applied_to_the_raw_head_label() -> None:
    items = synthetic_items(6)
    answers = {
        i.page: ev.RawAnswer(i.truth["nourl"], 0.4 if i.page == "p0" else 0.9) for i in items
    }
    spec = {"old": {}, "new": {"shipped_bars": {"alpha": 0.5, "beta": 0.5, "other": 0.5}}}
    summary = ev.run_eval(
        items, {"old": answers, "new": answers}, class_map(heads=spec), baseline="old", n_boot=5
    )
    shipped = summary.subsets["all"].truth["nourl"].heads["new"].shipped
    assert shipped is not None
    assert shipped["alpha"].accepted == 1  # p0 (alpha) sits under the 0.5 bar, p3 clears it
    assert summary.subsets["all"].truth["nourl"].heads["old"].shipped is None


# --- paired bootstrap ---------------------------------------------------------------------------


def test_bootstrap_of_a_head_against_itself_is_centred_on_zero() -> None:
    head = head_prepared({4, 9, 14, 21})
    result = ev.paired_bootstrap(head, head, n_boot=300)
    for key in ("all:correct_coverage", "all:coverage", "alpha:recall"):
        assert result[key].diff == 0.0
        assert result[key].lo == pytest.approx(0.0, abs=1e-12)
        assert result[key].hi == pytest.approx(0.0, abs=1e-12)


def test_bootstrap_separates_a_worse_head_and_is_seeded() -> None:
    good = head_prepared(set())
    worse = head_prepared({i for i in range(120) if i % 5 in (0, 2)})
    result = ev.paired_bootstrap(good, worse, n_boot=300, seed=0)
    cover = result["all:correct_coverage"]
    assert cover.diff < 0
    assert cover.hi < 0
    assert cover.lo <= cover.diff <= cover.hi
    assert ev.paired_bootstrap(good, worse, n_boot=300, seed=0) == result
    assert ev.paired_bootstrap(good, worse, n_boot=300, seed=1) != result


def test_a_metric_undefined_in_most_draws_is_left_out() -> None:
    wrong = arrays(["alpha"] * 6, [0.1] * 6, ["beta"] * 6, [0, 1] * 3)
    nothing = ev.prepare(*wrong, CLASSES, "other", 0.9)
    result = ev.paired_bootstrap(nothing, nothing, n_boot=50)
    assert "alpha:precision" not in result
    assert "beta:recall" in result


# --- subsets and variants -----------------------------------------------------------------------


def test_schema_subsets_are_independent_analyses() -> None:
    summary = run_two_heads(synthetic_items())
    assert {k: v.n_pages for k, v in summary.subsets.items()} == {
        "all": 60,
        "schemaless": 30,
        "schema_declared": 30,
    }
    undeclared = [item(f"u{i}", i, CLASSES[i % 3]) for i in range(6)]
    only_all = run_two_heads(undeclared)
    assert list(only_all.subsets) == ["all"]


# --- leak check ---------------------------------------------------------------------------------


def test_overlap_helper_fails_on_a_shared_domain_and_prints_the_count(
    capsys: pytest.CaptureFixture[str],
) -> None:
    items = [item("p1", 0, "alpha"), item("p2", 1, "beta")]
    with pytest.raises(ev.OverlapError, match="domain"):
        _ = ev.assert_no_overlap("old", [{"p": "t1", "d": "dp2"}], items)
    assert "share 0 pages and 1 domains" in capsys.readouterr().out
    with pytest.raises(ev.OverlapError, match="domain"):
        _ = ev.assert_no_overlap("old", [{"p": "p1", "d": "dp2"}], items, drop_page_overlap=True)


def test_overlap_helper_passes_when_disjoint_and_can_drop_shared_pages(
    capsys: pytest.CaptureFixture[str],
) -> None:
    items = [item("p1", 0, "alpha"), item("p2", 1, "beta")]
    kept = ev.assert_no_overlap("old", [{"p": "t1", "d": "other_domain"}], items)
    assert kept == items
    assert "share 0 pages and 0 domains" in capsys.readouterr().out
    with pytest.raises(ev.OverlapError, match="page"):
        _ = ev.assert_no_overlap("old", [{"p": "p1", "d": "elsewhere"}], items)
    train = [{"p": "p1", "d": "elsewhere"}]
    kept = ev.assert_no_overlap("old", train, items, drop_page_overlap=True)
    assert [i.page for i in kept] == ["p2"]
    assert "dropping 1 shared page" in capsys.readouterr().out


# --- inputs and cli -----------------------------------------------------------------------------


def test_parse_answers_takes_the_first_row_and_derives_from_probs() -> None:
    rows: list[ev.JsonObject] = [
        {"p": "a", "label": "alpha", "confidence": 0.8},
        {"p": "a", "label": "beta", "confidence": 0.99},
        {"p": "b", "label": None},
        {"p": "c", "probs": {"alpha": 0.2, "beta": 0.7}},
        {"p": "d", "label": "alpha", "probs": {"alpha": 0.6}},
    ]
    got = ev.parse_answers(rows, "head")
    assert got["a"] == ev.RawAnswer("alpha", 0.8)
    assert got["b"] == ev.RawAnswer(ev.NO_ANSWER, 0.0)
    assert got["c"] == ev.RawAnswer("beta", 0.7)
    assert got["d"] == ev.RawAnswer("alpha", 0.6)
    with pytest.raises(ValueError, match="confidence"):
        _ = ev.parse_answers([{"p": "x", "label": "alpha"}], "head")


def test_parse_items_rejects_the_reserved_consensus_variant() -> None:
    row: ev.JsonObject = {"p": "a", "d": "x", "bucket": 1, "truth": {"consensus": "alpha"}}
    with pytest.raises(ValueError, match="reserved"):
        _ = ev.parse_items([row])


def test_parse_items_rejects_a_repeated_page() -> None:
    row: ev.JsonObject = {"p": "a", "d": "x", "bucket": 1, "truth": {"nourl": "alpha"}}
    with pytest.raises(ValueError, match="twice"):
        _ = ev.parse_items([row, row])


def write_jsonl(path: Path, rows: list[ev.JsonObject]) -> Path:
    _ = path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf8")
    return path


def item_rows(items: list[ev.Item]) -> list[ev.JsonObject]:
    return [
        {"p": i.page, "d": i.domain, "bucket": i.bucket, "schema": i.schema, "truth": dict(i.truth)}
        for i in items
    ]


def write_class_map(path: Path, heads: dict[str, object]) -> Path:
    spec = {"classes": list(CLASSES), "truth": {"default": "other"}, "heads": heads}
    _ = path.write_text(json.dumps(spec), encoding="utf8")
    return path


def test_cli_writes_a_strict_json_summary_and_a_table(tmp_path: Path) -> None:
    items = synthetic_items()
    new_rows: list[ev.JsonObject] = [
        {"p": i.page, "label": i.truth["nourl"], "confidence": 0.9} for i in items
    ]
    # The old head never answers beta here, so it agrees with its `cannot_answer`.
    old_rows: list[ev.JsonObject] = [
        {**row, "label": "other"} if row["label"] == "beta" else row for row in new_rows
    ]
    classes = write_class_map(
        tmp_path / "classes.json", {"old": {"cannot_answer": ["beta"]}, "new": {}}
    )
    out, table = tmp_path / "summary.json", tmp_path / "table.txt"
    code = ev.main(
        [
            *("--items", str(write_jsonl(tmp_path / "items.jsonl", item_rows(items)))),
            *("--class-map", str(classes)),
            *("--head", f"old={write_jsonl(tmp_path / 'old.jsonl', old_rows)}"),
            *("--head", f"new={write_jsonl(tmp_path / 'new.jsonl', new_rows)}"),
            *("--n-boot", "40"),
            *("--out", str(out)),
            *("--table", str(table)),
        ]
    )
    assert code == 0

    def reject(constant: str) -> None:
        raise AssertionError(f"summary holds {constant}")

    summary = json.loads(out.read_text(encoding="utf8"), parse_constant=reject)
    assert summary["baseline"] == "old"
    assert summary["by_design"] == {"old": ["beta"], "new": []}
    assert "new_vs_old" in summary["subsets"]["all"]["truth"]["nourl"]["bootstrap"]
    text = table.read_text(encoding="utf8")
    assert "subset all, truth nourl" in text
    assert "paired bootstrap new_vs_old" in text
    assert "all:correct_coverage" in text


def test_cli_fails_when_a_training_domain_is_in_the_eval_rows(tmp_path: Path) -> None:
    rows: list[ev.JsonObject] = [{"p": "p0", "label": "alpha", "confidence": 0.9}]
    train: list[ev.JsonObject] = [{"p": "zz", "d": "dp3"}]
    code = ev.main(
        [
            *("--items", str(write_jsonl(tmp_path / "items.jsonl", item_rows(synthetic_items(6))))),
            *("--class-map", str(write_class_map(tmp_path / "classes.json", {"new": {}}))),
            *("--head", f"new={write_jsonl(tmp_path / 'new.jsonl', rows)}"),
            *("--train-rows", f"new={write_jsonl(tmp_path / 'train.jsonl', train)}"),
            *("--out", str(tmp_path / "summary.json")),
        ]
    )
    assert code == 1
    assert not (tmp_path / "summary.json").exists()
