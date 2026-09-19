# Shared annotation policy, revision `dom-suggestions-v6`

This file is the **only** source of label meaning for prompt revision
`dom-suggestions-v6`. Both annotators receive its bytes unchanged: the Jev
adapter puts it in request state, and the reviewer prompt embeds it. Each
emitted row carries `policySha256` so a row can be proven to have been labelled
under this exact text.

Revision `v4` gave the two annotators **no shared definitions at all** and they
agreed on `componentType` 57.6% of the time. Revision `v5` added this file and
moved within-reviewer agreement from 34.0% to 72.0% on a deliberately hard
200-node pilot, against a 75% gate. Revision `v6` changes four things, each
because the v5 pilot measured a failure rather than predicted one:

1. **The link-menu escape hatch is gone.** v5 said site chrome is
   `layout_container`, then added that `navigation_menu` wins "if the footer or
   header is dominated by a link menu". "Dominated" had no test, and a site
   footer is almost always mostly link labels, so the clause swallowed the rule
   it qualified. It produced 26 of the 56 remaining disagreements, 46% of the
   gap, on 16 `<header>` and 10 `<footer>` nodes.
2. **The no-attributes abstention is gated on the evidence actually being
   available.** v5 said an element with no attributes and no accessible name is
   `unknown`. The frozen captures carry no attributes at all, so the rule fired
   on every image rather than on genuinely anonymous ones, and it was the only
   stratum that went backwards, 100% to 80%.
3. **`media` versus `image` and `card` versus `list` are defined.** Both pairs
   produced disagreements and neither had a rule; `media` was in the vocabulary
   with no definition at all.
4. **`<main>` is deliberately unchanged.** Ten disagreements are `<main>`
   container-versus-section, and the reviewers split 13/12 on the same nodes
   under v4. That is real ambiguity. Forcing a convention there would buy
   agreement by destroying the distinction; `unknown` carries it instead.

See [`../../diverse-corpus/ERROR-ANALYSIS-2026-09-19.md`](../../diverse-corpus/ERROR-ANALYSIS-2026-09-19.md)
and the pilot record in the private `classifier-v2/label-pilot-v5/` directory.

Do not reword any section below without bumping the revision. A reworded policy
that reaches only one annotator reintroduces exactly the failure it fixes.

---

## 1. componentType

Label names are exactly the `allowedLabels.componentTypes` values in
[`../../taxonomy.json`](../../taxonomy.json). Section 1 follows
[`../../LABEL-DEFINITIONS.md`](../../LABEL-DEFINITIONS.md) except where this
revision's changes say otherwise.

Pick exactly one `componentType` for the node. Work down the precedence ladder
and stop at the first rule that fires. Do not revisit an earlier step because a
later one looks like a better fit.

### Precedence

1. **A specific component**, if the node matches one of the named shapes below.
2. **`content_section`**, if the node is a section of the page's own content,
   with its own subject.
3. **`layout_container`**, if the node only groups or positions other things.
4. **`unknown`**, if there is genuinely no evidence. Abstention is a valid
   answer and is never penalised. Do not guess from a class name alone.

A more specific label always wins. A node that is both a grouping wrapper and a
navigation menu is `navigation_menu`. A node that is both a content section and
a card is `card`.

### The two rules that settle the common case

**`<footer>` and `<header>` elements, and any node whose role is `contentinfo`
or `banner`, are `layout_container`. No exception.** A site footer or masthead
groups links, legal text and contact details that belong to the whole site, not
to this page's subject. It is a container.

  - **Being mostly links does not change this.** A site footer is almost always
    mostly link labels; that is what a footer is. `navigation_menu` belongs to
    the `<nav>`, `<ul>` or menu element **inside** the header or footer, which
    is a different node and gets its own judgment. Label the chrome element as
    the container it is, and let the menu inside it be the menu.
  - The only thing that overrides this is the element carrying substantive
    content about this page's own subject, which is rare, and which makes it
    `content_section`.

```
footer > [ nav > ul > li>a * 20 , p "© 2026" ]   -> layout_container
    the footer; mostly links, still the footer

    ...and the nav inside it                      -> navigation_menu
    a different node, judged on its own

header > [ img (logo) , nav > ul > li>a * 6 ]     -> layout_container
    the masthead

    ...and the nav inside it                      -> navigation_menu
```

**`<main>` elements, and any node whose role is `main`: label it
`layout_container` when it only wraps the page's sections, and
`content_section` when it is itself the single body of content.** The test is
whether the node has one subject of its own or several. A `<main>` holding a
hero, three feature blocks and a call to action is a `layout_container`. A
`<main>` holding one article's prose is a `content_section`, or `article` if the
article shape below fires.

### `content_section` versus `layout_container`

Ask one question: **does this node have a subject of its own?**

- **`content_section`**: yes. The node presents one topic, and its heading, prose
  or media are about that topic. Removing it would remove content from the page.
- **`layout_container`**: no. The node exists to group, position, wrap or space
  other nodes. Removing it would change the arrangement, not the content. Its
  text is whatever its children happen to contain.

Tie-breakers, in order, when both still look plausible:

1. **Own heading.** A direct heading child that names the node's topic makes it
   `content_section`.
2. **Mixed subjects.** Children covering unrelated topics make it
   `layout_container`.
3. **Pure wrapper shape.** A single element child, or children that are
   themselves sections, makes it `layout_container`.
4. **Still tied:** choose `layout_container`. It is the weaker claim, and the
   ladder above means a genuine content section will normally have already
   matched a specific component or rule 1.

Do not use size, depth, child count or text length to decide this. Those
measurements were checked against the labels and they do not separate the two.

#### Abstract shapes

```
div > [ h2 "Pricing" , p , ul ]                      -> content_section
    one heading naming one subject, prose beneath it

div > [ section#pricing , section#faq , section#cta ] -> layout_container
    three unrelated subjects, the node itself has none

div > div > section > [ h2 , p ]                      -> layout_container
    single-child wrapper chain; the content lives further down

main > [ h1 "How we test" , p , p , figure ]          -> content_section
    one subject, so the main element is the body, not a wrapper

footer > [ nav , p "© 2026" , ul ]                    -> layout_container
    site chrome grouping links and legal text
```

### Specific components

Each fires only when its shape is present. When two fire, take the one listed
first.

**`navigation_menu`** A node whose main purpose is moving between destinations:
a menu, breadcrumb trail, pagination strip, table of contents or tab bar. The
test is link density and uniformity: most of its text is link labels, and the
links are siblings of the same shape. A `<nav>` element is almost always this.
Not a node that merely contains a link, and not prose with incidental links.

```
nav > ul > [ li>a , li>a , li>a ]                     -> navigation_menu
p > [ text , a , text ]                               -> not navigation_menu
header > [ img , nav > ul > li>a * 5 ]                -> layout_container
    the header itself is chrome; the nav inside it is the navigation_menu
```

**`card` versus `list`: the item is a `card`, the wrapper is a `list`.**
Judge the node in front of you, not the group it belongs to.

- **`card`** is **one** repeated teaser unit: its own link, its own title, and
  usually an image or summary, standing beside siblings of the same shape. The
  repetition is the signal. A single unit alone is a `card` only if it clearly
  has the teaser shape; otherwise prefer `content_section`.
- **`list`** is the **container** of short, uniform items, or a run of items
  that are themselves the content rather than links to elsewhere. If the node
  is the `<ul>`, `<ol>` or grid wrapper, it is a `list` (or a
  `navigation_menu`, if its items are navigation links), whatever its children
  are. If the node is one item inside it, it is a `card` when it has the teaser
  shape and part of the `list` otherwise.

```
li > [ img , h3 , p , a "Read more" ]  (5 identical siblings) -> card
ul > li > [ img , h3 , a ] * 5   (the ul)                     -> list
figure > [ img , figcaption > [ h3 , a ] ]                    -> card
div > [ h2 , p , p , p ]  (no siblings of this shape)         -> content_section
ul > [ li "Fast" , li "Secure" , li "Open" ]                  -> list
ol > li > [ h4 , p ] * 8   (the ol)                           -> list
ul > li > a * 6   (all link labels)                           -> navigation_menu
```

**`hero`** The single leading block at the top of the page, carrying the page's
headline, a short supporting line, and usually one or two prominent calls to
action or a large background image. There is at most one per page, and it comes
before the page's other sections. A second, later banner-shaped block is
`banner`, not `hero`.

```
section (first in main) > [ h1 , p , a.button , a.button ]  -> hero
div (first in main) > [ h1 , p , img (full width) ]         -> hero
section (mid page) > [ h2 , p , a.button ]                  -> content_section
```

**`article`** A complete piece of editorial or documentation content: the body
prose of one story, post, guide or reference page, usually with a heading and
several paragraphs. Use it for the content itself, not for a teaser pointing at
it, which is a `card`.

```
article > [ h1 , time , p * 12 ]                      -> article
div > [ h1 , p * 9 , figure , p * 4 ]                 -> article
article > [ h3 , p , a "Read more" ]  (one of many)   -> card
```

**`banner`** A strip that promotes, announces or notifies, spanning the content
width and separable from the page's own subject: a promotion bar, an
announcement strip, a cookie or consent bar, a subscription prompt. Not the
page's leading block, which is `hero`. Not paid third-party advertising, which
is `ad_unit`.

```
div (top of page) > [ p "Free shipping this week" , a ] -> banner
div > [ p "We use cookies" , button , button ]          -> banner
section (first in main) > [ h1 , p , a.button ]         -> hero
```

### Elements with no text

An `<img>`, `<iframe>`, `<video>`, `<audio>`, `<svg>` or `<input>` often carries
no text at all. Label it from its tag, its attributes and its container, not
from the empty text.

**First check whether the packet carries attributes at all.** A packet built
from a legacy browser capture has none: it says so, and its `attributes=` and
`src=` lines are absent or `none` for every element on every page. A packet
built from HTML carries them.

- **When attributes are unavailable, label from the tag.** An `<img>` is an
  `image`, a `<video>` is `media`, an `<input>` is `input`. The element *is*
  what its tag says regardless of what its `alt` would have said, and abstaining
  on all of them produces a page of `unknown` that teaches nothing. Use
  `unknown` only for a tag that genuinely does not decide a component, such as a
  bare `<iframe>` or `<div>`.
- **When attributes are available**, use them: `alt`, `title`, `aria-label` and
  `role` describe what the element is, and the `src` category says whether it is
  a known embed provider (`youtube`, `google`, `recaptcha`, and so on), the
  site's own asset (`same-site`) or an unrecognised third party. The host itself
  is never carried. Only here does the next rule apply: an element that carries
  attributes, and whose attributes give it no accessible name and no recognised
  source, is `unknown` rather than a guess.

**`image` versus `media`.** `image` is a still picture: `<img>`, `<picture>`, or
a `<figure>` whose only content is one. `media` is time-based or playable
content that is not itself a player surface: `<video>`, `<audio>`, or a wrapper
holding one. If the node is the interactive player surface with controls, it is
`video_player` or `audio_player`. If it is a coordinated set of several visual
items, it is `media_gallery`. When a node is both plausible, prefer the more
specific of `video_player`, `audio_player` and `media_gallery`, then `image` for
a still and `media` for anything time-based.

```
img[alt="Product photo, front"]                       -> image
img, no attributes available (capture packet)         -> image
video, no attributes available (capture packet)       -> media
iframe[title="Install walkthrough", src=youtube]      -> video_player
iframe[no title, src=an ad provider]                  -> ad_unit
iframe, no attributes available (capture packet)      -> unknown
img[alt="", 1x1]  (attributes available)              -> unknown
```

---

## 2. regions

`regions` is multi-label and answers **where the node sits in the rendered
page**, which is a different question from what shape it has. A node is commonly
in one region and occasionally in two nested ones (an `article_body` inside
`main_content`). Answer each region independently; a region is yes only when the
node is inside or is that area.

The three conventions that matter, matching section 1 so the two axes cannot
contradict each other:

- **A `<header>` element, or a node with role `banner`, that sits at the top of
  the page and belongs to every page of the site, is `site_header`.** It is not
  `top_banner`. `top_banner` is for a separable promotional or consent strip
  above or inside the header, not for the masthead itself.
- **A `<footer>` element, or a node with role `contentinfo`, that belongs to
  every page of the site, is `footer`.** A footer nested inside one article is
  not the site footer; judge it on what it belongs to.
- **A `<main>` element, or a node with role `main`, is `main_content`.** Its
  descendants are also `main_content`; add `article_body` as well only when the
  node is the prose body of one editorial article, and `article_header` only for
  that article's own title block.

`unknown` is the whole answer for the axis or it is not used. Prefer an empty,
complete `regions` observation over guessing when the node is genuinely
page-generic.

## 3. purposes

`purposes` is multi-label and answers **what the node does for a user**.
Revision v5 carries only the twelve purposes with measured support in the v4
labels; the fifteen dropped values are listed in
[`purposes-v5.json`](purposes-v5.json) and must not be emitted.

A purpose is yes when the node's own content or controls serve it, not when a
neighbour does. A node may serve none: an empty complete `purposes` observation
is a valid answer and is preferred over a weak yes.

## 4. pageType

`pageType` is **one** primary choice per page, not a set. Revision v4 asked 65
independent yes/no questions and the two annotators agreed on the resulting set
1.3% of the time on the test split, which is a measurement of the question, not
of the page. v5 asks the single question the rules actually need: what is this
page for?

Judge the page as a whole from its title, URL path and body evidence. Navigation
labels, footer links and incidental words are not evidence that the page itself
has a type: every page on a shop links to the pricing page. The
sixty-five-to-eleven collapse is recorded in
[`page-type-map-v5.json`](page-type-map-v5.json).

- **`article`** One editorial or news piece: a blog post, news story, press
  release or case study, with a body of prose written to be read start to end.
- **`product_detail`** One purchasable product or plan, with its price,
  variants, availability or buy control.
- **`product_listing`** Many purchasable products together: a shop category, a
  search result grid, a collection page.
- **`docs`** Product or API documentation, a reference page, a how-to guide or a
  tutorial, written to be consulted while doing something.
- **`homepage`** The site's root page: the top-level entry point that introduces
  the whole site rather than one subject.
- **`pricing`** Plans, tiers or rates presented for comparison, with the
  intention of choosing one.
- **`contact`** How to reach the organisation: a contact form, address, phone
  number, office locations or a support request page.
- **`about`** Who the organisation or person is: company story, mission, team,
  history, careers overview.
- **`legal`** Terms, privacy policy, cookie policy, accessibility statement,
  licence or other formal policy text.
- **`listing_index`** An index of other pages that are not products: a blog
  index, news index, category or tag archive, docs index, event or directory
  listing. The page's own content is the list.
- **`other`** Anything the ten above do not describe, including account,
  checkout, login, search-result, error and status pages. Use it rather than
  stretching a label.

When two fit, choose the one the page is **for**. A pricing page that also
describes the company is `pricing`. A docs index is `listing_index` only if it
is purely a table of contents; if it explains anything, it is `docs`.

## 5. Abstention

Abstention is a first-class answer on every axis and is never penalised.

- `componentType`: emit `unknown` when the evidence does not decide it.
- `regions` and `purposes`: emit the axis with `complete: true` and an empty
  label list when the packet rules every label out, or omit the axis when the
  packet cannot support a judgment at all.
- `pageType`: emit `other` rather than guessing between two specific types.

A confident wrong label is worse than an abstention, because a wrong label
trains a model and an abstention only costs a row.
