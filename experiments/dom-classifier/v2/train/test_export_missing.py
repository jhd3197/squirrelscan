"""Missing-value splits in the packed artifact (#2307): the float32 reference
walk must reproduce LightGBM's own NaN / zero-as-missing decisions, and a head
without them must still pack as version 1, unchanged."""

from __future__ import annotations

import struct

import numpy as np
import pytest

lgb = pytest.importorskip("lightgbm")

from export_model import evaluate, pack  # noqa: E402


def _data(rows: int = 3000, seed: int = 7):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, 6)).astype(np.float32)
    y = (x[:, 0] + 0.8 * x[:, 3] > 0).astype(int) + (x[:, 1] > 0.6).astype(int)
    return x, y


def _train(x, y, **params):
    booster = lgb.train(
        {
            "objective": "multiclass",
            "num_class": 3,
            "num_leaves": 15,
            "learning_rate": 0.2,
            "verbose": -1,
            "seed": 1,
            **params,
        },
        lgb.Dataset(x, label=y),
        num_boost_round=25,
    )
    return booster


def _version(blob: bytes) -> int:
    return struct.unpack("<I", blob[4:8])[0]


def test_nan_block_matches_lightgbm_with_and_without_values() -> None:
    x, y = _data()
    # The page-type setup: a trailing block dropped to NaN on 30% of rows.
    x[: len(x) * 3 // 10, 3:] = np.nan
    booster = _train(x, y)
    blob, reference = pack(booster.dump_model()["tree_info"], 3, None)
    assert _version(blob) == 2
    assert reference["flags"] is not None and (reference["flags"] & 3 == 2).any()
    test, _ = _data(400, seed=11)
    for matrix in (test, np.where(np.arange(6) >= 3, np.nan, test).astype(np.float32)):
        ref = evaluate(reference, matrix)
        assert (ref.argmax(1) == booster.predict(matrix).argmax(1)).all()
        assert np.abs(ref - booster.predict(matrix)).max() < 1e-5


def test_zero_as_missing_matches_lightgbm() -> None:
    x, y = _data()
    x[::4, 2] = 0.0
    booster = _train(x, y, zero_as_missing=True)
    blob, reference = pack(booster.dump_model()["tree_info"], 3, None)
    assert _version(blob) == 2
    test, _ = _data(400, seed=12)
    test[::3, :] = 0.0
    ref = evaluate(reference, test)
    assert (ref.argmax(1) == booster.predict(test).argmax(1)).all()


def test_a_head_without_missing_values_is_still_version_1() -> None:
    x, y = _data()
    booster = _train(x, y)
    blob, reference = pack(booster.dump_model()["tree_info"], 3, None)
    assert _version(blob) == 1
    assert reference["flags"] is None
    test, _ = _data(200, seed=13)
    assert (evaluate(reference, test).argmax(1) == booster.predict(test).argmax(1)).all()
