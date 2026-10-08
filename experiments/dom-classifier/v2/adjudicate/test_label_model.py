"""Label model on synthetic sources with known accuracy (#2307).

uv run --with numpy==2.5.3 --with pytest -m pytest test_label_model.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from label_model import PAGE_CLASSES, Votes, cross_validate, fit_confusions, normalised_weights, posterior

K = len(PAGE_CLASSES)


def synthetic(n: int, accuracies: dict[str, float], seed: int = 7):
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, K, n)
    rows = []
    for i, y in enumerate(truth):
        for position, (source, acc) in enumerate(accuracies.items()):
            if position and rng.random() < 0.1:
                continue  # abstains (the first source always votes, so every item has a row)
            v = y if rng.random() < acc else int(rng.integers(0, K))
            rows.append({"key": f"p{i}", "source": source, "label": PAGE_CLASSES[v]})
    return truth, rows


def test_posterior_beats_best_single_source_and_learns_reliability():
    truth, rows = synthetic(2000, {"good": 0.85, "ok": 0.7, "noisy": 0.4})
    votes = Votes(rows, PAGE_CLASSES)
    order = np.asarray([votes.row[f"p{i}"] for i in range(len(truth))])
    gold, pool = order[:400], order[400:]
    w = np.ones(len(gold))
    theta = fit_confusions(votes.matrix[gold], truth[:400], w, votes.sources, K, 8.0)
    acc = {s: float(np.mean(np.diag(theta[j]))) for j, s in enumerate(votes.sources)}
    assert acc["good"] > acc["ok"] > acc["noisy"]
    p = posterior(votes.matrix[pool], theta, np.full(K, 1 / K), 1.0)
    combined = float(np.mean(p.argmax(1) == truth[400:]))
    best_single = max(float(np.mean(votes.matrix[pool, j] == truth[400:])) for j in range(len(votes.sources)))
    assert combined > best_single


def test_correlated_sources_pick_a_lower_temperature():
    rng = np.random.default_rng(3)
    truth = rng.integers(0, K, 600)
    rows = []
    for i, y in enumerate(truth):
        shared = y if rng.random() < 0.7 else int(rng.integers(0, K))
        for source in ("a", "b", "c"):  # three copies of one witness
            rows.append({"key": f"p{i}", "source": source, "label": PAGE_CLASSES[shared]})
    votes = Votes(rows, PAGE_CLASSES)
    order = np.asarray([votes.row[f"p{i}"] for i in range(len(truth))])
    best, _ = cross_validate(
        votes.matrix[order], truth, np.ones(len(truth)), votes.sources, K, 8.0, [0.2, 0.4, 0.6, 0.8, 1.0]
    )
    assert best < 1.0


def test_sampling_weights_are_normalised_within_each_class():
    truth = np.asarray([0, 0, 0, 1, 1])
    w = normalised_weights(truth, np.asarray([10.0, 1.0, 1.0, 5.0, 5.0]), K)
    assert np.isclose(w[truth == 0].sum(), 3) and np.isclose(w[truth == 1].sum(), 2)


def test_cli_fit_and_export(tmp_path: Path):
    truth, rows = synthetic(300, {"jev": 0.8, "adjudicator": 0.9})
    for r in rows:
        r["axis"] = "page"
        r["conf"] = 0.9
    (tmp_path / "votes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    gold = [
        {"key": f"p{i}", "axis": "page", "label": PAGE_CLASSES[y], "weight": 1.0} for i, y in enumerate(truth[:120])
    ]
    (tmp_path / "gold.jsonl").write_text("".join(json.dumps(g) + "\n" for g in gold))
    script = Path(__file__).parent / "label_model.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "fit",
            "--votes",
            str(tmp_path / "votes.jsonl"),
            "--gold",
            str(tmp_path / "gold.jsonl"),
            "--axis",
            "page",
            "--out",
            str(tmp_path / "out"),
        ],
        check=True,
    )
    report = json.loads((tmp_path / "out" / "report-page.json").read_text())
    assert report["goldAccuracyOOF"] > 0.85
    subprocess.run(
        [
            sys.executable,
            str(script),
            "export",
            "--posteriors",
            str(tmp_path / "out" / "posteriors-page.jsonl"),
            "--axis",
            "page",
            "--out",
            str(tmp_path / "export"),
        ],
        check=True,
    )
    exported = [
        json.loads(line) for line in (tmp_path / "export" / "labelmodel-pagetype.jsonl").read_text().splitlines()
    ]
    assert exported and {"pageId", "pageType", "confidence"} <= set(exported[0])
