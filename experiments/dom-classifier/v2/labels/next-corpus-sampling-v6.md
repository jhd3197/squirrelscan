# Sampling plan: ~20,000 nodes for labelling from the full-HTML corpus

For the corpus being featurized from the ~29.5k full-HTML pages in the local
squirrel store, rather than from the frozen 2026-09-19 captures. Written by the
labelling lane; the corpus builder owns the featurizer, this owns which nodes
get labelled and why.

Full HTML changes what a packet can say. A capture-sourced packet carries tag,
role, position, size and text. An HTML-sourced packet adds `alt`, `title`,
`aria-label`, the `src` category, the media descendants, link counts and the
repeated-sibling count — see `runtime-dom-html-v1` in
`model-suggestions/jev-v5.ts`. That is what makes the media and card rules
answerable at all: in the v5 pilot every one of 25 empty-media nodes had to be
labelled `unknown`, because the capture could not say whether an `<img>` was a
product photo, a logo or a tracking pixel.

## Blocker: the candidate filter must be widened first

`extractPage` keeps an element only when its tag is in `containerTags`, or it
carries an allowed ARIA role, or it has at least 40 characters of text **and**
at least two element children. `containerTags` is `article, aside, dialog,
footer, form, header, main, nav, section, figure, table, ul, ol, div, details,
fieldset`, and `iframe` is in `skipTags` outright.

On a synthetic shop page with a logo, a three-card product grid and a YouTube
embed, `extractPage` returns candidates for `body, header, nav*2, ul*2, main,
section, figure, footer` and **zero** `li`, `img`, `a`, `button` or `iframe`.
A test in `model-suggestions/jev-v5.test.ts` pins this.

So a labelling set sampled straight from today's `extractPage` **cannot contain
an `image`, `card`, `link`, `button`, `video_player`, `ad_unit`, `input` or
`rating_summary` node at any sample size**. Those are exactly the 22 component
classes with zero support in the current test split. Widening the filter is a
precondition for this plan, not an optimisation:

- add `li`, `a`, `button`, `img`, `picture`, `video`, `audio`, `iframe`,
  `input`, `select`, `textarea`, `label`, `blockquote`, `dl` as eligible tags;
- keep `iframe` out of the *text* walk but let it be a candidate, since its
  `title` and `src` category are the whole signal;
- keep the existing text-and-children rule as an additional way in, not as the
  only one.

Widening raises the candidate count per page by roughly an order of magnitude,
which is fine: this plan samples 20,000 nodes from that pool, it does not label
it.

## Splits before sampling

Freeze splits by **registrable domain** before drawing anything, so no domain
appears in two splits. Hold out a fresh test split that no annotator and no
error analysis has seen; the 75-page 2026-09-19 split is burned and stays
excluded entirely.

## Caps

- **At most 30 labelled nodes per registrable domain.** 20,000 nodes therefore
  need at least ~667 domains, which keeps any single large site under 0.15% of
  the set.
- **At most 4 nodes per page**, so a deep template page cannot supply a stratum
  by itself.
- **At most 2 nodes per stratum per page**, which stops a product grid from
  contributing forty near-identical cards.

Within a stratum, order by `sha256(registrableDomain || pageId || locator)` and
take greedily under the caps. Deterministic, reproducible, and independent of
crawl order.

## Strata and quotas

| Tier | Stratum | Quota | Why |
|---|---|---:|---|
| A | `div` / `span` containers, non-semantic | 6,000 | Where the tag rule scores 18.4% and a model has to earn its place. `div`'s modal label wins only 17% of the time. |
| A | `li` inside a repeated sibling group (`repeatedSiblingPatternCount >= 3`) | 2,000 | The `card` vs `list` boundary; unreachable today. |
| A | `section` with more than one heading child | 2,000 | The `content_section` vs `layout_container` boundary the policy is written for. |
| B | Commerce: pages with Product/Offer/AggregateRating JSON-LD, or price-pattern text | 2,000 | `purchase_panel`, `specification_list`, `rating_summary`, `review_list`: all zero support. |
| B | Media: nodes whose subtree holds `iframe`/`video`/`audio` with a known provider `srcCategory` | 1,200 | `video_player`, `audio_player`, `media_gallery`. |
| B | Advertising: nodes whose `srcCategory` is an ad provider | 600 | `ad_unit`, zero support, and the one class where a false positive is expensive. |
| B | Editorial furniture: Article JSON-LD plus byline or comment markers | 1,000 | `author_card`, `comment_thread`. |
| B | Form controls: `input`, `select`, `checkbox`, `radio`, `toggle`, `dropdown` | 800 | Six labels, all zero or near-zero support. |
| B | Overlays: `role=dialog`, `aria-modal`, `role=tooltip` | 400 | `dialog`, `drawer`, `tooltip`, `popover`, `notification`. |
| C | Landmarks: `header`, `footer`, `nav`, `main`, `article`, `aside` | 2,400 | Keeps the easy classes calibrated and makes a regression visible. |
| C | Media with an accessible name: `img`/`picture`/`figure` carrying `alt` or `title` | 800 | The class the capture corpus could not label at all. |
| C | `a` and `button` | 800 | The `link` vs `button` confusion, 8 records in the pilot. |
| | **Total** | **20,000** | |

Tier A is 50% and is where the model has something to add. Tier B is 30% and is
bought with cheap deterministic pre-filters over JSON-LD, ARIA and `srcCategory`
— none of which is a label, only a way to find candidates worth a judgment.
Tier C is 20% and exists so a v6 regression on the easy cases cannot hide behind
the hard ones.

If a Tier B stratum cannot be filled from the corpus, record the shortfall
rather than back-filling it from Tier A: a class the crawl does not contain is a
fact about the corpus, and silently substituting hides it.

## Masking

Two artifacts, two rules, and they must not be confused:

- the **labelling packet** keeps the node's tag and role, because
  `policy-v5.md` is written in terms of `<footer>`, `<header>` and `<main>`;
- the **training features** mask the node's own landmark tag and its ancestors'
  landmark tags, per the v2 plan, so the model learns structure rather than
  reading the answer off the tag.

## What to label with what

Full dual annotation is reviewer-bound, not Jev-bound. Measured on this pilot:

| | per 1,000 nodes | 20,000 nodes |
|---|---:|---:|
| Jev (TypeSafe, input tokens only) | $0.38 | ~$7.55 |
| Reviewer (Claude Sonnet, 20-packet batches) | see `pilot-report.json` `cost.reviewerBv5` | reviewer cost dominates |

So: **Jev on all 20,000**, and a reviewer on a stratified **10% audit subset**
(2,000 nodes, quota-proportional), which is enough to estimate agreement per
stratum and to catch a stratum where the teacher is systematically wrong. Spend
the saved budget on human adjudication of the disagreements instead, which is
the only signal here that is not another model's opinion.
