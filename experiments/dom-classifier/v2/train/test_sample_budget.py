"""The node cap in sample.py, pinned: it follows memory, and says when it binds."""

from __future__ import annotations

from collections import Counter

from sample import DEFAULT_MAX_TRAIN_ROWS, budget_nodes, sampling_rates


def test_budget_grows_with_memory_and_shrinks_with_row_width() -> None:
    assert budget_nodes(40.0, 8.0) > budget_nodes(40.0, 4.0)
    assert budget_nodes(80.0, 4.0) < budget_nodes(40.0, 4.0)


def test_a_bigger_corpus_is_not_a_no_op_under_a_memory_budget() -> None:
    small = Counter(content_block=1_000_000, footer=300_000, navigation=500_000, dialog=40_000)
    big = Counter({name: count * 3 for name, count in small.items()})
    cap = budget_nodes(40.0, 4.0)

    def kept_rare(counts: Counter, max_rows: int) -> float:
        rates = sampling_rates(counts, max_rows)
        return sum(counts[n] * rates[n] for n in counts if n != "content_block")

    # Under the old fixed default a 3x corpus kept the same rare rows...
    assert round(kept_rare(big, DEFAULT_MAX_TRAIN_ROWS)) == round(
        kept_rare(small, DEFAULT_MAX_TRAIN_ROWS)
    )
    # ...under a memory-sized cap it keeps more, as long as memory allows.
    assert cap > DEFAULT_MAX_TRAIN_ROWS
    assert kept_rare(big, cap) > kept_rare(small, cap)
