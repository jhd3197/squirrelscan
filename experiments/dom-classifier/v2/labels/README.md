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

## Pilot result, 2026-09-19

200 nodes and 125 pages, stratified to oversample the measured failure
boundaries, drawn from validation and train-QA only. Aggregates are public;
per-row labels are not.

| Axis | v4 baseline (this sample) | Jev v5 vs Luna v4 (one-sided proxy) |
|---|---:|---:|
| componentType | 46.7% | 49.2% |
| regions | 65.0% | 66.2% |
| purposes | 37.0% | 44.9% |
| pageType | 41.6% | 82.4% |

The sample is deliberately hard: the same v4 measurement over the unstratified
pool is 57.6%, matching the published figure.

**The G1 number is not in this table.** G1 is Jev v5 against Luna v5, both under
this policy, and the reviewer leg has not run. The proxy column compares v5
against a **v4** reviewer, so it mostly measures whether Jev adopted conventions
the reviewer already held; it moves for reasons that have nothing to do with
whether the two annotators now agree.

Per stratum, the proxy column decomposes into one large win and two large losses:
`footer` 2.7% → 67.6%, `figure` 12.5% → 87.5%, `iframe` 25% → 75%, against
`header` 71.4% → 42.9% and `empty_media` 100% → 0%.

Two things to fix before scaling, both visible in that decomposition:

1. **The no-attributes abstention rule is too broad.** Section 1 says an element
   with no attributes and no accessible name is `unknown`. The captures carry no
   attributes at all, so the rule fires on every image rather than on genuinely
   anonymous ones, turning 25 of 25 `empty_media` nodes into abstentions that
   cannot train anything. An `<img>` is an `image` whatever its alt text says.
2. **"Dominated by a link menu" has no test.** It decides `<header>` and it is
   the one escape hatch in section 1 with no stated threshold, unlike the
   `<main>` rule's "one subject or several".

## Cost

TypeSafe bills input tokens only, at $0.042 per million; output tokens are free
(docs.typesafe.ai/models.md). The 325-record pilot cost **$0.076**, or $0.378 per
1,000 nodes, which projects to about **$7.55 for a 20,000-node run**. Median
request 397 ms, p95 1,138 ms.
