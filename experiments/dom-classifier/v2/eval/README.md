# Page-type head eval at a fixed precision

One script, `head_eval.py`, compares page-type heads under one protocol: the
phase 2 head-to-head of the bundled head against skeleton-v3 (#2977). Every
later candidate (a recalibrated bundled head, skeleton-v4) goes through the
same code, so a difference in a table is a difference in the head and not in
the protocol.

It reads labels and scores only. It never sees page text, URLs or markup.

```sh
cd experiments/dom-classifier/v2/eval
uv run python head_eval.py \
  --items judged-items.jsonl \
  --class-map class_map.phase2.json \
  --head bundled=heads-bundled.jsonl \
  --head skeleton=heads-skeleton.jsonl \
  --train-rows skeleton=skeleton-train-rows.jsonl \
  --baseline bundled \
  --out summary.json --table table.txt
```

Gates: `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`,
`uv run basedpyright`. The tests use synthetic data only.

## Protocol

1. **Truth.** Each judged page has one truth label per judge variant (for
   example the judge without the URL and the judge with it). The class map folds
   those labels, and every head's own labels, onto one common vocabulary. A
   `consensus` variant keeps the pages where every variant agrees on the mapped
   class and drops the rest before anything is computed.
2. **Out-of-fold thresholds.** Fold is the domain bucket parity (`bucket % 2`).
   For each head and class, the threshold applied to a fold is the lowest
   confidence whose cumulative precision on the **other** fold reaches the target
   (default 0.90; see the note on ties below). It is `1.01` when none does, so that
   class never answers. No page, and no fold, sets its own threshold. A class the
   head never predicts in the other fold gets no threshold and stays silent,
   however good it is here.
3. **Operating point.** Per class and overall: how many pages answered, precision
   with a Wilson 95% lower bound, recall, coverage (share of pages answered) and
   correct coverage (share answered correctly).
4. **Paired bootstrap.** Pages are resampled (2,000 draws, seed 0) with the
   thresholds held fixed. One resample feeds both heads, so the draws are paired.
   Each head minus the baseline is reported as the observed difference and a 95%
   percentile interval. The headline row is `all:correct_coverage`. A metric that
   is undefined in more than half the draws (a class the head never accepts) is
   left out.
5. **Repeats.** Everything runs for each truth variant and for three subsets:
   `all`, `schemaless` and `schema_declared`. Each subset is an independent
   analysis with its own folds, thresholds and resamples.
6. **Leak check.** `--train-rows NAME=PATH` counts the domains and page ids that
   a head trained on and that also appear in the eval rows, prints the counts and
   fails on any shared domain or page. `--drop-page-overlap` turns a shared page
   id (the same content served under another domain) into a drop that is counted
   and printed. A shared domain always fails.

Shipped bars (the per-label confidence bars a head carries in its artifact) are
reported next to the out-of-fold numbers for reference. They are not part of the
protocol.

### Ties in the threshold rule

The default rule is the phase 2 one (DomainGenius `pick_threshold`): it checks the
cumulative precision at every rank, so inside a group of equal confidences the row
order decides, and the accepted set `confidence >= t` can end up below the target.
A head with one constant confidence per class is the extreme case: it accepts
everything it predicts as soon as any prefix of the ranking clears the target. Use
heads that expose a real score. `--ties group` lets only the last row of each tie
group qualify, so the accepted set itself reaches the target and the row order
stops mattering. The two rules agree when nothing ties, and they gave identical
numbers on the phase 2 inputs, but a candidate head with coarse or rounded scores
should be run both ways.

## Input formats

All inputs are jsonl (one object per line) except the class map.

### Judged items (`--items`)

```json
{"p": "page id", "d": "domain id", "bucket": 22, "schema": true,
 "truth": {"nourl": "blog_post", "url": "article"}}
```

- `p` is unique. `d` is any stable domain identifier (a hash is fine).
- `bucket` is the integer domain bucket. Every page of a domain shares it.
- `schema` is `true` when the page declares a main entity in structured data,
  `false` when it does not, and `null` or absent when unknown. Pages with an
  unknown flag appear in `all` only.
- `truth` maps each judge variant to its label in the judge's own vocabulary. The
  variants are the keys of the first item; an item missing one is dropped and
  counted.

### Head outputs (`--head NAME=PATH`, repeatable)

```json
{"p": "page id", "label": "editorial", "confidence": 0.87,
 "probs": {"editorial": 0.87, "listing": 0.1}}
```

- `label` is the head's argmax in its own vocabulary. `null` means the head gave
  no answer (for example no features could be built).
- `confidence` is the probability or score of that label. It may be left out
  when `probs` is present: it is then `probs[label]`, and a missing label is the
  argmax of `probs`.
- The first row per page wins. A page with no row counts as no answer, and the
  count is in the summary under `no_output`.
- `NAME` must be a key of `heads` in the class map. The first head is the
  baseline unless `--baseline` says otherwise.

### Class map (`--class-map`)

```json
{
  "classes": ["editorial", "listing", "other"],
  "other_class": "other",
  "truth": {"map": {"blog_post": "editorial"}, "default": "other"},
  "heads": {
    "bundled": {
      "map": {"article": "editorial", "homepage": null},
      "default": null,
      "cannot_answer": ["event"],
      "shipped_bars": {"article": 0.95}
    }
  }
}
```

- `classes` is the common vocabulary, in table order.
- `other_class` names the catch-all class, used for the `nonother` rows. It may
  be `null`.
- A truth label resolves through `truth.map`, then to itself when it is a class,
  then to `truth.default`.
- A head label resolves through its `map`, then to itself when it is a class,
  then to its `default`. **No answer** is a `null` target, in `map` or as the
  `default`: the page counts as unanswered and never as correct.
- **Abstains by design** is `cannot_answer`: classes the head cannot produce at
  all. They stay in the tables (their recall is zero, which is true) but are
  marked `~` so nobody reads that zero as a failure to learn. A head that does
  answer one of them is a class map error and the run stops.
- `shipped_bars` is optional: label to bar, keyed by the head's own label. A label
  with no bar never answers at the shipped operating point.

`class_map.phase2.json` is the map the phase 2 analysis used (nine common
classes; the bundled head cannot answer `event`; its `homepage` argmax is no
answer).

### Training rows (`--train-rows NAME=PATH`, optional, repeatable)

```json
{"p": "page id", "d": "domain id"}
```

Every page and domain the named head was trained on. Either key may be absent
for a row.

## Output

`--out` is a strict JSON summary (NaN becomes `null`):

- `target`, `ties`, `n_boot`, `seed`, `classes`, `heads`, `baseline`, `variants`
- `items`, `items_missing_a_variant`, `no_output` (per head), `by_design` (per
  head)
- `subsets.<subset>.n_pages` and `subsets.<subset>.truth.<variant>`:
  - `n`, `support` (per class)
  - `heads.<name>`: `classes.<class>` with `support`, `accepted`, `tp`,
    `precision`, `precision_lo` (Wilson), `recall`; `coverage`, `precision`,
    `precision_lo`, `correct_coverage`; `nonother` (`precision`, `recall`,
    `accepted_share`); `thresholds_oof` (per class, `[fold 0, fold 1]`);
    `shipped` when bars were given
  - `bootstrap.<name>_vs_<baseline>.<metric>`: `base`, `other`, `diff`, `lo`,
    `hi` for `<class>:recall`, `<class>:precision`, `all:coverage`,
    `all:precision`, `all:correct_coverage`, `nonother:precision`,
    `nonother:recall`

`--table` (and stdout) is the short text version of the same numbers: one block
per subset and truth variant with a row per class, the overall coverage line per
head, and the recall and overall rows of each paired bootstrap, with `*` where
the interval excludes zero.
