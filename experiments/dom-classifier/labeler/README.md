# Local human DOM labeler

This is a private, localhost-only review service for rendered public pages. It
stores a full-page PNG and a bounded DOM geometry snapshot for each immutable
capture. A browser is used only to make a capture, either a fresh render or a
historical replay of stored HTML (see "Historical replay captures"); the review
server itself never renders, serves archived HTML or runs page scripts.

Its fixed seed list is only a small smoke-capture mode: SquirrelScan, Taskmux,
Example, and one public MDN reference page. Larger corpus work is supplied to a
separate command-line renderer as a private, validated manifest. The review
HTTP API accepts no URLs or browser profile, so it cannot act as a proxy or
inspect a user's logged-in session.

## Run

Install the scoped experiment dependencies and its Chromium browser before
capturing or running browser checks:

```sh
(cd experiments/dom-classifier && bun install --frozen-lockfile && bunx playwright install chromium)
```

When a bundled runtime is already available, set
`LABELER_PLAYWRIGHT_MODULE` to its Playwright module instead. Capture the
bounded seed set first:

```sh
bun experiments/dom-classifier/labeler/server.ts capture
```

Then start the service through the repository's managed runtime configuration.
It defaults to `127.0.0.1:4317`; `PORT`, `LABELER_DATA_DIR` and
`LABELER_COHORTS` (see "Gold review") are supported. The service never starts a
crawl or re-captures pages on its own.

Portable local output defaults to `~/.local/share/squirrel/dom-labeler`. Set
`LABELER_DATA_DIR` to choose another review store:

```text
captures/page_<hash>.png     immutable screenshot
captures/page_<hash>.json    rendered DOM node geometry and bounded text
annotations.jsonl            append-only human decisions
page-annotations.jsonl       append-only human page-level decisions
```

Node IDs are structural IDs inside an immutable capture. A capture ID includes
the sanitized rendered-content hash, screenshot hash, and viewport. An
annotation records that capture hash and remains `source: "human"` and
`gold: false`; review/adjudication is a separate later process.

## API

- `GET /api/pages` returns the page list, node-label taxonomies, and the
  authoritative page-level `pageTypes`, `pageTypeGroups`, and `contentKinds`
  catalogues.
- `GET /api/pages/:id` returns the immutable page, nodes, node annotations,
  and `pageAnnotations`.
- `GET /api/gold-queue` returns the mounted gold questions whose captures are
  present, in review order, each with its
  item id, target, axis, shown class (`null` on a blind item), alternatives,
  status and saved answer (see "Gold review").
- `GET /api/stats` returns current queue totals, page-level labels
  (`pages.currentPageLabels`), positive element labels, and model-review
  actions. `pages.labelled` means a page has any current positive page or
  element label. Reviewed counts use only the latest effective action for each
  current model suggestion; provisional suggestions are never counted as human
  labels.
- `GET /captures/:id.png` returns an allowlisted local screenshot only.
- `POST /api/annotations` appends a validated, idempotent annotation. It needs
  a same-origin `Origin`, `labeler_csrf` cookie, and matching
  `X-Labeler-CSRF` header. Read any API endpoint first to receive the cookie.
- `POST /api/page-annotations` appends a validated, idempotent page-level
  annotation with the same localhost CSRF protection.
- `GET /api/export` downloads node annotation JSONL. Page annotations remain in the separate private `page-annotations.jsonl` stream.
- `GET /api/export/reviews` downloads a private JSON review bundle with the
  stats snapshot and append-only node, page, and model-review histories.

The write body is `{pageId,nodeId,decision,regions?,purposes?,componentType?,
componentSubtype?,observedState?,role?,context?,comment?,boundary,
clientRequestId,captureHash?,supersedes?}`. `captureHash` is optional for the
base contract but lets the UI reject an optimistic write against an older
capture. `clientRequestId` makes retrying a save safe.

Earlier schema-v2 rows separated independent axes:

- `regions`: `site_header`, `footer`, `sidebar`, `main_content`,
  `article_header`, or `unknown`.
- `functions`: `navigation`, `card`, `form`, `consent_banner`, or `unknown`.

Each array may be empty and has no negative implication. Labels are unique and
`unknown` must be the sole label in its own axis. This permits combinations
such as `regions: ["footer"]` plus `functions: ["navigation"]`. These raw
values remain preserved in the append-only history; the current picker does
not write a second legacy-function selector.

The raw legacy `role` and `context` fields remain in every existing annotation.
When old records are read, the service exposes a `labelSchemaVersion: 1`
compatibility projection: legacy `aside` becomes `sidebar`, legacy navigation
becomes a purpose, and explicit legacy `header`, `footer`, or `main` context
adds the corresponding region. It does not rewrite the JSONL or treat that
projection as a new human claim. A visible editor note identifies this
projection before a reviewer saves a newer revision.

`label` needs one or more region/purpose/component labels. An `accept` must include its
weak hint on the hint's axis, but may retain independently selected labels on
the other axis. If an accept omits both arrays, the service records only the
matching weak-hint projection. A `reject` keeps supplied independent labels and
context but cannot retain the rejected hint or carry a legacy role. `unsure` carries
no labels. A correction can set `supersedes` only to the latest annotation for
the same captured node, so the JSONL preserves the full revision trail. Page
progress counts unique nodes, not revisions.

## Component schema v3

V3 records the observed component shape separately from its layout and purpose.
It does not identify framework/source components, component families, or link
graph membership. The picker supports primitives (`button`, `link`, `input`,
`checkbox`, `radio`, `select`, `toggle`, `heading`, `text`, `icon`, `image`),
widgets/content (`search`, `dropdown`, `tabs`, `accordion`, `pagination`,
`card`, `form`, `list`, `table`, `article`, `media`), sections/layout (`hero`,
`navigation_menu`, `content_section`, `layout_container`), and overlays
(`dialog`, `drawer`, `tooltip`, `notification`, `banner`, `popover`), plus
`unknown`. Regions separately describe placement, including `hero`,
`top_banner`, `bottom_banner`, and `overlay`.
A `navigation_menu` is a composite type; its descendant links remain separate
observed nodes and are never inferred as primitive labels merely from that
container. DOM containment is capture evidence, not a source-component tree.

`componentSubtype` is optional and accepted only for component types and values
published by `GET /api/pages`. The current catalogue includes variants for
`button`, `input`, `card`, `navigation_menu`, `form`, `dialog`, `banner`,
`drawer`, `popover`, and `content_section`; the server catalogue remains
authoritative as it evolves.
`observedState` is optional, has per-type validation, and is only what the
annotator saw: `visible`, `disabled`, `expanded`, `selected`, `open`, `sticky`,
or `unknown`. `purposes` are independent of type: `navigation`, `submit`,
`search`, `filter`, `consent`, `share`, `account`, `dismiss`, `download`,
`toggle`, `promotion`, `announcement`, `authentication`, `subscription`,
`feedback`, `support`, or `unknown`. The current write field is `purposes`; it
is distinct from the v2 `functions` archive field. `card`, `form`, and
`banner` describe the observed component shape, while a purpose such as
`submit` or `consent` describes what it is for. Page region answers where it
appears. None of these axes identifies a framework component or source-code
family.

V1/V2 rows remain byte-for-byte unchanged. They read with
`componentSchemaVersion: 0` and, when unambiguous, a
`componentProjection` or `purposeProjection` marked `legacy-v1-v2`; these are
display aids, not new human v3 labels. In particular, legacy navigation reads
with `purposeProjection: ["navigation"]` while raw `purposes` remains empty.
New component-picker saves have `componentSchemaVersion: 3`.
`GET /api/export` returns the original JSONL lines rather than normalized read
projections.

## Taxonomy revision v2

`dom-taxonomy-v2` is additive: every existing canonical label remains valid
with its prior meaning. The version adds article-body and paid-ad regions,
separate product gallery, buy-box, details, and review regions; author, related,
and comments regions; purposes for paid advertising, purchase, media playback,
reviews, information, editorial, instruction, product information, comparison,
and social proof; and component shapes for ad units, media galleries, purchase
panels, specification lists, rating summaries, review lists, media players,
author cards, and comment threads. It deliberately does **not** duplicate existing
`image`, `media`, `banner`, `promotion`, product page types, or editorial page
types. The full machine-readable vocabulary and definitions are in
[`../taxonomy.json`](../taxonomy.json).

Axes remain independent: a `media_gallery` component can sit in a
`product_gallery` region without implying `media_playback`; a first-party
`banner` can have `promotion` without becoming an `advertisement`; and a
`purchase_panel` may have `purchase` only when it actually supports a purchase.
An omitted or empty axis is unobserved, never a negative label. Only
`["unknown"]` records an explicit unknown for one multi-label axis. Legacy
records and sidecars are never silently mapped to new v2 labels.

## Page schema v1

Page labels are a separate review stream: they never add, modify, or infer DOM
component labels. `POST /api/page-annotations` accepts
`{pageId,decision,pageTypes?,contentKinds?,comment?,clientRequestId,captureHash?,supersedes?}`.
The response is `{annotation}`, and a page detail includes its complete
append-only `pageAnnotations` trail.

`pageTypes` describes the page's role (for example `homepage`, `blog_post`,
`docs_article`, `product_detail`, `pricing`, `contact`, `news_article`, or
`case_study`). The API publishes a grouped catalogue for entry/company,
commercial, documentation/releases, publishing/media, community/events,
account/transaction, and utility/policy uses. `contentKinds` is an independent
optional axis: `article`, `product`, `person`, `event`, `review`, `offer`,
`job`, `software`, `service`, `organization`, `reference`, `media`, `other`,
or `unknown`. This permits hybrids such as a product detail page whose content
is software, without making a cross-product taxonomy.

Both axes allow several values for hybrids. Values are unique and `unknown` is
exclusive within its own axis. A `label` decision needs at least one page type
or content kind; an `unsure` decision carries neither. Thus an explicit
unknown is a `label` with `pageTypes: ["unknown"]` or
`contentKinds: ["unknown"]`, while `unsure` records no classification. A
correction may supersede only the latest page annotation for that capture, and
the same `clientRequestId` may safely retry exactly the same request. Page
labels keep the immutable capture hash as provenance. URL-derived hints are not
stored as human labels or prefilled as confirmed labels.

Historical replay captures may add non-label provenance metadata:
`sourceKind`, `originalCrawledAt`, `sourceUrl`, opaque `corpusRef`,
`contentOriginDate`, and `assetMode`. `capturedAt` remains the time the
rendered screenshot was made. These fields describe capture origin for the UI;
they do not affect page or node labels.

## Provisional Jev suggestions and keyboard review

When the current capture has a Jev sidecar suggestion for an unreviewed node or
page, the labeler starts in fast model-review mode. For element review, the
card names the proposed **Page region**, **Element type**, and any **Purpose**,
and asks the concrete question those values imply. An axis Jev did not map is
shown as **Not suggested**; the labeler never infers it from the DOM tag. The
exact captured target receives a strong outline and is panned into view. Page
review shows page classification only and does not highlight an element. The
card then offers **OK** and **Not OK**. OK saves the
suggestion's exact mapped labels as a human acceptance and advances. Not OK
saves a standalone model rejection and advances; it does not require a
replacement label. A comment is optional in the advanced editor. Model scores
are never presented as verified accuracy.

An existing human annotation or saved review is preserved and cannot be
overwritten by the fast controls. Open the advanced editor to revise an earlier
human label, add a manual label, or make a correction. Changing a region,
purpose, component type, subtype, or observed state records a model `correct`
review; “Reset to model labels” intentionally replaces a changed draft with the
model labels.

Use the dedicated model-review card for one prepared target at a time: swipe
right to approve or left to reject. The gesture only triggers after a deliberate
horizontal drag; vertical scrolling, short drags, and gestures that start on a
control never save a review. **Skip** (or `J`) defers the current target for
this browser session without writing a label or rejection, then moves to the
next available target or page. In Page mode, Skip moves to the next captured
page. `Enter` outside form fields marks the displayed model label OK and
continues; `X` marks it Not OK and continues. **Undo last review** (or
`Cmd/Ctrl` + `Z` outside a text field) reverses your most recent successful
model acceptance, correction, page label, or rejection with an append-only
undo record, then returns to the exact target. Finish or discard a current
manual draft before undoing so that draft is preserved. `K` moves back through the current
capture's model targets, including saved labels so they can be corrected. In
Page mode, `K` moves to the previous captured page. `[` and `]` move between
captured pages; `P` and `C` select the parent or first child; `Esc` clears the
current element; and `?` opens the on-screen shortcut reference. `Cmd/Ctrl`
`Enter` saves the current advanced draft from any field, including a comment;
add `Shift` to save and continue. These shortcuts do not intercept ordinary
typing or native button/link activation. Unsaved human edits still use the
inline keep-or-discard prompt before a page, mode, or selection change.

Run the focused checks with:

```sh
bun test experiments/dom-classifier/labeler/server.test.ts
```

## Gold review

Gold review collects human answers to the exact questions the Treetop heads
are scored on: test gold, which measures the model and the label sources, and
calibration gold, which may weight the sources in the label model (see
`../DEVELOPMENT.md`, rule 4). One question per item, one class per answer.

### Axes

Two additive answer axes sit beside the descriptive v2 taxonomy. Rows without
them are unchanged, and each field is written only when present.

- `pageTypesV8`: the 12 page-type classes. The card text follows the current
  page-type policy (`v2/labels/policy-v8.2.md`), and
  `GOLD_POLICY_REVISIONS.pageTypeV8` names it.
- `sectionPurposesV3`: the section classes of `v2/labels/section-classes-v2.json`
  at `policy-section-v3.md` granularity. The table of contents is judged at its
  tightest node, and `ad` means third-party only (the site's own promotion is
  `promo_banner`).
- `sectionItemOfV3`: an item answer (see "Keys"). It always sits next to
  `sectionPurposesV3: ["none"]`, so v3 scoring reads it as none.

### Cohorts

A cohort is a directory holding `captures/` and a `model-suggestions.jsonl`
sidecar of gold questions. `LABELER_COHORTS` mounts cohorts as comma-separated
`id=/absolute/dir` pairs, and their order is the queue order: the primary data
directory's questions first, then each cohort in turn. Answers always go to the
primary data directory's journals, stamped with the cohort id, so mounting or
unmounting a cohort never moves an answer. A gold answer is also stamped with
its axis's `goldPolicyRevision`, the policy its card text stated when it was
saved. Answers saved before that field existed carry none, and the scorer
dates them. The server reads cohorts at start:
restart it after changing `LABELER_COHORTS`, and reload the page afterwards. The
current cohorts (test gold and calibration gold sets), their paths and their
sizes are in the private runbook.

Scoring keys (which items are attention checks, every source's label, the
sampling weights) live outside anything the server reads. Never put them in a
sidecar or a capture. A blind item is marked in the sidecar, because the card
has to ask it differently, but its sources' labels are not.

### Questions

- **Shown item**: the sidecar row's `mappedLabels` carries one class on one
  axis (`pageTypeV8` for a page, `sectionPurposeV3` for a node), with optional
  `alternatives` on the same axis. The card asks a concrete question ("Is this
  page an About page?", "Is the outlined element a cookie consent banner?")
  and shows the class's policy one-liner.
- **Blind item**: the row carries only `blindAxis`, with empty `mappedLabels`
  and no alternatives. The card asks "What is this page?" or "What is the
  outlined element?" with the class picker already open, so the reviewer
  answers without an anchor.
- **Attention check**: a small share of shown items carry a deliberately
  wrong class. They look exactly like any other shown item; which items they
  are is recorded only in a private key.

The card never names a source and shows no model score, and the cohort
builders give every gold row the same neutral `modelId`. Gold questions never appear on the
element or page cards.

### Keys

| key | on a shown item | in the picker |
|---|---|---|
| `Y` / `Enter` | yes, it is this class | `Enter` saves the top match once something is typed |
| `N` | no, it is not this class (a rejection with no fix) | |
| `S` | suggest the right class: opens the picker | |
| `I` (sections) | an item answer of the shown class | switches to (or out of) "an item of which section?" |
| letters | | filter by the start of any word ("coo" finds cookie consent) |
| `1`–`9`, `0` | | pick one of the first ten classes |
| `U` | | unsure: a "can't tell" bound to the question |
| `Esc` | | clear the filter, leave item mode, or close the picker |
| `J` / `K` | skip for this session / go back one question | same, when the filter is empty |
| `Cmd`/`Ctrl` + `Z` | undo the last answer | same |

Section cards show the buttons No, Suggest, Item and Yes; page cards show No,
Suggest and Yes. On a section, the picker also has an "Item of…" button. With
the filter empty, `u`, `j` and `k` (and `i` on a section) act as commands, so a
class with a word starting with one of them is reached by typing a different
word of it ("bus" for Business info).

An **item answer** says the outline covers one item of a section: one comment,
review, FAQ question, product card or breadcrumb link. On a shown section, `I`
records an item of the shown class. On a blind section (or a shown `none`), `I`
opens the picker to choose the section. It is stored as
`sectionPurposesV3: ["none"]` plus `sectionItemOfV3: "<class>"`; the store
refuses an item on any other purpose, or one that names `none`. The private
scorer reports item answers apart from the v3 scores; they seed a later item
layer.

### Item ids and deep links

Every gold question has a short, stable id, `<cohort short>-<position>`.
`gold-v<N>` becomes `g<N>` (so `gold-v1b` is `g1b`), `gold-cal-v1` becomes `cal`
(`gold-cal-v<N>` becomes `cal<N>`), and any other cohort keeps its own id.
Position counts the cohort's well-formed gold rows in sidecar order, including
rows later skipped for a missing capture, so a missing capture never shifts
an id. The id shows
in the card's corner (select it to copy), and the address bar follows the
question on screen (`#g1-234`, via `history.replaceState`). Opening
`/#<id>`, or `/#msug_…` with a suggestion id, jumps to that question, answered
or not; pasting another id into the address bar jumps without a reload. An
unknown id shows "No such item" and opens the next open question.

### Answers and undo

| answer | stored as |
|---|---|
| yes | a page or node annotation with the shown class, `modelReview: "accept"` |
| a picked class (fix or blind) | the same with the picked class, `modelReview: "correct"` |
| no | a standalone model review, `review: "reject"` |
| unsure | an annotation with `decision: "unsure"`, bound to the question |
| item | `sectionPurposesV3: ["none"]` + `sectionItemOfV3`, `modelReview: "correct"` |

Every answer is bound to its question by `modelSuggestionId`, and the latest
effective action per question is the answer. Answering an answered question
appends a newer answer, and the card says what is saved ("Saved: Contact",
"Saved: an item of Comments"). `Cmd`/`Ctrl` + `Z` undoes answers one by one,
newest first, within the browser session, each with an append-only undo
record, so the journals keep every revision. The queue opens at the first
unanswered question.

## Historical replay captures

`replay-capture.ts` renders STORED HTML rather than fetching the page, so a
reviewer sees the bytes a model was given (`sourceKind: "historical_replay"`,
`assetMode: "hybrid"`). The navigation to the original URL is answered with the
stored document; stylesheets, images and fonts load from their public hosts.
Fixed elements are pinned where they were measured, so the full-page
screenshot and the node boxes agree. Elements a caller marked with
`data-replay-target` are reported as located, hidden, or gone (a script
replaced them). It is a library for a private, command-line builder; the
review server never renders anything.

Page content is hostile, and a replayed page runs on a machine where local
services listen (`../DEVELOPMENT.md`, rule 8). The hardening:

- **Deny-by-default addresses.** `isPublicAddress` (in `capture.ts`) refuses
  the IPv4 special-purpose blocks (private, loopback, link-local, shared,
  benchmarking, documentation, multicast, reserved). It parses every IPv6
  spelling and judges mapped, compatible, translated and NAT64 forms by the
  IPv4 inside them. Any other IPv6 address must be in 2000::/3, and
  2001::/23 (Teredo and other protocol assignments), 2001:db8::/32, 2002::/16
  (6to4) and 3fff::/20 are refused too, so `::`, loopback, link-local,
  site-local, unique-local and multicast addresses never pass. `isPrivateHostname` uses it for IP literals and also refuses
  `localhost` and `.local` names. Fresh captures and replays both check
  through it.
- **Egress guard.** All replay traffic leaves through `egress-guard.ts`, a
  loopback CONNECT proxy. It resolves each host itself (caching the decision
  for ten minutes), refuses the host if any address is not public, and connects
  to exactly the address it checked, which closes DNS rebinding. Playwright sends loopback through the proxy too. Plain
  HTTP subresources are refused.
- **Blocked requests.** Data requests (`fetch`, XHR, event streams), beacons,
  WebSockets (closed by a WebSocket route), popups and any top-level
  navigation after the stored document are refused. Launch the browser with
  `replayLaunchOptions()`, which also stops WebRTC from using non-proxied UDP.
- **Scripts off by default.** A replay serves the stored document under
  `script-src 'none'` (`scriptMode: "disabled"`). A caller turns scripts on per
  page only where the page needs them: client-rendered builders, and SPA shells
  whose scripts-off render comes out blank. The private render builder records
  why (`scriptReason`).
- **Deferred stylesheets.** `script-src 'none'` also blocks the inline handler
  that loadCSS-style pages use to switch a stylesheet on (`rel="preload"
  as="style"` or a `media="print"`/`media="none"` switch), and `<noscript>`
  stays unrendered. Scripts-off replays promote those links to plain
  stylesheets (`promoteDeferredStyles`), so such pages are not drawn
  unstyled.
- **Deadline.** Once its browser context exists, a replay that has not finished
  by the deadline closes that context, so a page that never settles cannot hold
  a render worker.

Chromium's Private/Local Network Access features were measured ineffective in
the bundled headless shell (140), so they are not relied on. Residual risks:
with scripts on, a WebSocket opened from a worker is not closed by the page's
WebSocket route, but it still leaves through the guard and so reaches public
hosts only. The page's JavaScript still runs in Chromium, and
`replayLaunchOptions()` keeps Playwright's default of no Chromium OS sandbox.

## Private corpus rendering

A separate command-line-only queue may render up to 200 pre-reviewed public
HTTPS manifest URLs. It is never available through the labeler API. Each JSONL
row is exactly `{url,originalCrawledAt?,corpusRef?}`. Query strings, credentials,
private hosts, authentication paths, and attachment URLs are rejected before a
browser starts.

The queue stages immutable PNG/manifest pairs first. Every candidate is an
anonymous **fresh** browser render (`sourceKind: "fresh_capture"`,
`assetMode: "live"`) at `capturedAt`; an `originalCrawledAt` or opaque
`corpusRef` records where a URL was discovered, not a claim that archived HTML
or screenshot data was imported. A true replay, when one is supported, must be
marked separately with its historical provenance and must never be presented as
a fresh render.

Before import, run the read-only quality review over the stage. It checks that
the PNG and manifest dimensions agree with `min(ceil(documentHeight), 12000)`,
that retained DOM rectangles fit the PNG, and that obvious challenge or
obscured captures are held out. The importer requires a matching private
`--review` file and copies only records marked `eligible`; it leaves every
held-out pair staged. The queue uses isolated contexts, four concurrent
distinct hosts, a 60-second capture deadline, 20 lazy-load steps, and a
12,000px image cap.
