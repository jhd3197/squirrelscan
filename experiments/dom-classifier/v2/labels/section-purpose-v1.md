# Section purpose, revision `section-purpose-v1`

A third head on the region head's candidate nodes: **which schema.org-relevant
section is this node?** It exists so an audit can say "this page has a FAQ
block (or reviews, breadcrumbs, a business address, an offer, a byline, a
video) but no matching structured data". One label per node; most nodes are
`none`.

The featurizer never reads JSON-LD, microdata (`itemscope`/`itemtype`/
`itemprop`) or `og:*` (featurize.ts rule 1; `featurize.test.ts` pins that a
node's vector is byte-identical with and without microdata attributes), so a
label derived from schema can train the head without the head seeing its
source.

| class | the node is | schema.org type it should carry |
|---|---|---|
| `faq_block` | a container of two or more question-and-answer pairs | FAQPage / Question |
| `reviews` | a container of one or more customer reviews, or the rating summary | Review / AggregateRating |
| `breadcrumbs` | the trail of links to this page's ancestors | BreadcrumbList |
| `business_info` | the business's address, opening hours and/or phone, as a block | LocalBusiness / PostalAddress |
| `product_offer` | the price and buy control of the product this page sells | Offer |
| `author_byline` | the author's name (and date) attached to this page's content | Person (author) |
| `video_embed` | an embedded video player | VideoObject |
| `none` | anything else | none |

Label the TIGHTEST candidate that contains the whole section: the list, not the
page wrapper around it; the FAQ container, not each question.

## Changes from the proposal, and why

- **`recipe_card` dropped.** 22 of 74,380 corpus pages declare a Recipe; there
  is nothing to train or evaluate on. Revisit with a targeted crawl (page-type
  v8 has the same gap).
- **`event_details` dropped.** 93 pages declare an Event, 5 in microdata. Same
  reason.
- Everything else is kept; the free-label noise report
  (`section_labels_v1` output) decides which classes are usable.
