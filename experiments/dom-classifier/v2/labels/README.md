# Label policy and vocabulary, classifier v2

Versioned inputs to the labelling pipeline. Nothing here is private: these files
are prompts and vocabularies. Labels, captures and page text stay in the private
data root.

| File | What it is |
|---|---|
| `policy-v5.md` | The shared annotation policy. Both annotators receive its bytes unchanged. |
| `page-type-map-v5.json` | The 65-to-11 page-type collapse, and the projection used to compare a v4 label set with a v5 choice. |
| `purposes-v5.json` | The 27-to-12 purpose reduction, with the support counts that justify each drop. |

## Why a policy file at all

Prompt revision v4 gave the two annotators **no shared definitions**. Measured
consequence, on the frozen 75-page test split: `componentType` agreement 57.6%,
and 33 of 34 `layout_container` / `content_section` disagreements were `<footer>`
and `<main>` alone. That is a convention clash, and a convention is the fix. See
[`../../diverse-corpus/ERROR-ANALYSIS-2026-09-19.md`](../../diverse-corpus/ERROR-ANALYSIS-2026-09-19.md).

The policy is delivered by hash, not by description. `model-suggestions/jev-v5.ts`
puts `policy-v5.md` in TypeSafe request state; `consensus/luna_v5_batch.py`
embeds it in the reviewer prompt. Both record `policySha256` on every row, so
"both annotators had the same definitions" is checkable after the fact.

## Versioning

Reword any section and you must bump the revision. A reworded policy that reaches
only one annotator reintroduces the exact failure it fixes. `policy-v5.md` is
bound to prompt revision `dom-suggestions-v5`; revision v4 remains reproducible
because it lives in a separate module (`model-suggestions/jev-adapter.ts`) that
v5 never modifies.

Label **names** are unchanged from `taxonomy.json` on the region and component
axes, so v4 and v5 outputs stay comparable. Only page types and purposes change
vocabulary, and both changes are recorded as explicit maps rather than applied
silently.

## Frozen revision: **v7**

`policy-v7.md` is the frozen labelling revision for `divsoup-eval-v1` and the
20,000-node run. No further prompt edits without agreement.

v7 is v6 plus one sentence scoping the chrome rule to the shape axis. Measured
on the same 115 nodes (all 35 `header`, all 40 `footer`, a deterministic
40-node control across the other strata):

| within reviewer B, 115 nodes | componentType | purposes micro-F1 | regions micro-F1 |
|---|---:|---:|---:|
| v5 | 67.0% | 0.710 | 0.934 |
| v6 | 88.7% | **0.187** | 0.924 |
| **v7** | **90.3%** | **0.779** | 0.925 |

The leak is closed and componentType did not pay for it: Jev's `navigation`
purpose on `header` recovered 0 → 27 and on `footer` 0 → 37.

## Pilot result, 2026-09-19: G1 passes at v6, 84.5%

200 nodes and 125 pages, stratified to oversample the measured failure
boundaries, drawn from validation and train-QA only. Aggregates are public;
per-row labels are not.

G1 is judged **within reviewer**: one reviewer answers the same nodes under both
revisions, so the reviewer, the records and the scoring rule are fixed and only
the prompt moves. Reviewer B is Claude Sonnet. Reviewer A (`gpt-5.6-luna`) was
over quota and could not be re-run under v5.

| componentType, 200 nodes | agreement |
|---|---:|
| Jev v4 vs Reviewer B v4 | 34.0% |
| Jev v5 vs Reviewer B v5 | 72.0% |
| **Jev v6 vs Reviewer B v6** | **84.5%** |
| threshold | 75% |

v6 deleted the link-menu escape hatch and gated the no-attributes abstention.
Those two clauses are exactly the two strata that moved: `header` 48.6% → 97.1%,
`footer` 75.0% → 100%, `empty_media` 80.0% → 96.0%. The 31 remaining
disagreements are led by the `<main>` container-versus-section ambiguity that v6
deliberately left alone.

**Known v6 regression, fix in v7:** `purposes` micro-F1 fell 0.698 → 0.361
because Jev generalised the componentType chrome rule into the purposes axis. On
`header` nodes its `navigation` purpose went 27 → 0 and on `footer` 20 → 0, while
the reviewer went the other way. Section 1 needs one sentence saying the chrome
rule decides shape, not what the node does for a user.

Other axes, within reviewer B, as micro-F1 over positive claims (exact-set
identity is reported too, but reviewer B declares `purposes` complete on only 22
of 200 nodes, so the exact-set rate there is scored on a subset it chose):

| Axis | v4 | v5 |
|---|---:|---:|
| regions | 0.841 | 0.888 |
| purposes | 0.576 | 0.698 |
| pageType (exact) | 42.6% | 86.4% |

Per stratum, componentType, within reviewer B: `footer` 5.0% → 75.0%, `main`
5.7% → 77.1%, `iframe` 28.6% → 85.7%, `div` 42.9% → 80.0%, `header` 28.6% →
48.6%, and `empty_media` 100% → 80.0%, the only regression.

The sample is deliberately hard. The same v4 measurement against reviewer A over
the unstratified pool is 57.6%, matching the published figure; on this sample it
is 46.7%.

**The diagnosis was right.** At v4, with no shared definitions, the two
independent reviewers agreed with *each other* 65.1% on components and 97.6% on
regions, while Jev agreed with neither (46.7% and 34.0%). The teacher was the
outlier.

Two things to fix before scaling:

1. **"Dominated by a link menu" has no test.** It is the one escape hatch in
   section 1 with no threshold, and a site footer is almost always mostly link
   labels, so the clause swallows the rule it qualifies. 26 nodes, 16 `header`
   and 10 `footer`, are 46% of all remaining disagreement. Make site chrome
   `layout_container` unconditionally and let `navigation_menu` attach to the
   `<nav>` inside. Bound if every affected node flips: 85.0%.
2. **The no-attributes abstention rule is too broad.** Section 1 says an element
   with no attributes and no accessible name is `unknown`. The frozen captures
   carry no attributes at all, so it fires on every image rather than on
   genuinely anonymous ones. An `<img>` is an `image` whatever its alt text says.
   Gate the rule on `attributesAvailable`.

## Cost

TypeSafe bills input tokens only, at $0.042 per million; output tokens are free
(docs.typesafe.ai/models.md). The 325-record Jev pilot cost **$0.076**, or $0.378
per 1,000 nodes, projecting to about **$7.55 for 20,000 nodes**. Median request
397 ms, p95 1,138 ms.

A reviewer costs **$15–17 per 1,000 nodes** (Claude Sonnet, 20-packet batches),
about 40x the teacher, so a reviewer on all 20,000 nodes is ≈$310. Label
everything with Jev and review a stratified 10% audit subset; see
[`next-corpus-sampling-v6.md`](next-corpus-sampling-v6.md).
