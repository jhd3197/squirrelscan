# Shared annotation policy, revision `dom-suggestions-v8.2`

This file is the **only** source of label meaning for prompt revision
`dom-suggestions-v8`. Unlike v5 to v7 it covers **one axis only, `pageType`**:
v8 exists to train a page-type head whose answer is "this page should carry
schema.org type X", so a missing-schema finding can be raised when it does not.
The node axes (componentType, regions, purposes) stay on v7 and are not asked
under this revision. Carrying their text in every request cost roughly two
thirds of each v7 page-type call for nothing.

Each emitted row carries `policySha256` so it can be proven to have been
labelled under this exact text. Do not reword a section without bumping the
revision.

v8.1 is v8 plus decision rule 7 (support and help pages), because the v8
labels split help centres between `contact` and `docs` with no rule to decide,
and that ambiguity capped `contact` precision.

v8.2 (2026-10-09, owner decision) widens `product_detail` and
`product_listing` from purchasable products to products, plans and service
offerings: a page about ONE product, plan or service is `product_detail` even
with no price, cart or buy control (SaaS product and feature pages, a single
service page). `other` no longer covers those pages. It also adds rule 8
(locale roots are homepages) and names Wix's `/post/` path as a blog path in
rule 1. Labels made under v8 or v8.1 for landing and service pages are stale.

What changed from v7's section 4, and why:

1. **Every class maps to one expected schema.org type**, or explicitly to none.
   A class that cannot say which schema a page should carry cannot produce a
   missing-schema finding, so it is only kept where the distinction matters for
   one: `blog_post` splits out of `article`, and `faq`, `recipe`, `event`,
   `job_posting`, `local_business` and `search_results` are new.
2. **The confusable pairs have decision rules** (below). v7 left
   `product_listing` versus `listing_index` and `article` versus blog content to
   judgment, and they were two of its weakest classes.

---

## 4. pageType

`pageType` is **one** primary choice per page. Judge the page as a whole from its
title, URL path and body evidence. Navigation labels, footer links and
incidental words are not evidence that the page itself has a type: every page
on a shop links to the cart.

The schema column is what a correctly marked-up page of that class carries. It
is context for the annotator, not evidence: the packet never contains the page's
own markup, and the label must be decidable without it.

| class | expected schema.org type |
|---|---|
| `homepage` | Organization + WebSite |
| `product_detail` | Product |
| `product_listing` | CollectionPage / ItemList |
| `article` | Article / NewsArticle |
| `blog_post` | BlogPosting |
| `faq` | FAQPage |
| `recipe` | Recipe |
| `event` | Event |
| `job_posting` | JobPosting |
| `local_business` | LocalBusiness |
| `about` | AboutPage |
| `contact` | ContactPage |
| `docs` | none required (TechArticle optional) |
| `legal` | none |
| `pricing` | none required (Offer optional) |
| `search_results` | SearchResultsPage (optional) |
| `listing_index` | CollectionPage (optional) |
| `other` | none |

Definitions:

- **`homepage`** The site's root page: the top-level entry point that introduces
  the whole site rather than one subject.
- **`product_detail`** ONE specific product, plan or service offering is the
  page's subject: a shop product page, a software product or feature page, a
  single service page. A price, cart or buy control is NOT required.
- **`product_listing`** MANY products, plans or service offerings presented
  together: a shop category, a collection, a product search grid, a services
  overview whose items each link to their own page.
- **`article`** One editorial, news or informational piece written to be read
  start to end: a news story, press release, feature, case study or guide that
  is not presented as a blog entry.
- **`blog_post`** One entry of a blog: an individual post, usually dated, often
  with an author, sitting under a blog section of the site.
- **`faq`** A page whose main content is a list of questions each with its
  answer.
- **`recipe`** One recipe: ingredients and preparation steps for a dish.
- **`event`** One event with its own date, time and place (physical or online):
  a concert, webinar, conference session, class.
- **`job_posting`** One job vacancy: the role, its requirements and how to
  apply.
- **`local_business`** One physical location of a business: a store, branch,
  clinic, restaurant or office page with its address and opening hours.
- **`about`** Who the organisation or person is: story, mission, team, history,
  careers overview.
- **`contact`** How to reach the organisation: a contact form, address, phone,
  support request page. One page listing MANY locations is `contact`; one
  location's own page is `local_business`.
- **`docs`** Product or API documentation, reference, how-to or tutorial written
  to be consulted while doing something.
- **`legal`** Terms, privacy, cookie or accessibility policy, licence or other
  formal policy text.
- **`pricing`** Plans, tiers or rates presented for comparison, with the
  intention of choosing one.
- **`search_results`** The results of a site search the visitor ran (a query is
  visible in the path, title or body), for anything other than products.
- **`listing_index`** An index of other pages that are NOT products: blog index,
  news index, category or tag archive, docs index, event or job listing,
  directory. The page's own content is the list.
- **`other`** Anything the seventeen above do not describe: account, login,
  checkout, cart, error and status pages, and campaign or landing pages that are
  not about one specific product, plan or service. Use it rather than
  stretching a label.

Decision rules for the confusable pairs:

1. **`article` versus `blog_post`.** `blog_post` when the page presents itself as
   a blog entry: a `/blog/` (or equivalent, e.g. Wix's `/post/<slug>`) path, a "blog" section label in the
   title or breadcrumb, or blog chrome (post date + author + categories/tags +
   "previous/next post"). News outlets, press releases, magazine features and
   guides are `article`. When nothing says "blog", choose `article`.
2. **`product_detail` versus `product_listing`.** Count the products, plans or
   services the page is ABOUT, not the ones it recommends. One item as the
   page's subject is `product_detail` even with a "related products" row below it.
   Two or more items presented as equals, each linking to its own page, is
   `product_listing`.
3. **`listing_index` versus `product_listing`.** Are the listed items products,
   plans, services or bookable items? Then `product_listing`. Posts, articles,
   events, jobs, documents or places: `listing_index`. A mixed index is decided
   by the majority of its items.
4. **`event` / `job_posting` versus `listing_index`.** ONE event or vacancy is
   its own class; a list of them is `listing_index`.
5. **`docs` versus `article`.** Written to be consulted while doing a task
   (steps, parameters, API) is `docs`; written to be read through is `article`.
6. **`search_results` versus `product_listing`.** A product search grid is
   `product_listing`; search results over non-product content are
   `search_results`.

7. **`contact` versus `docs` on support and help pages.** A support or help
   page whose main content is self-service answers (FAQs, how-to articles,
   links into documentation, a searchable knowledge base) is `docs`, even when
   it also shows an email address or a "contact us" button. It is `contact`
   only when its main content is a way to reach a person: a contact or support
   request form, phone numbers, addresses, office hours, a chat entry point.
8. **Locale roots are homepages.** A language or region root (`/de/`,
   `/en-us/`, `/fr-ca/`) is that locale's `homepage`, like the site root.
9. **`homepage` versus `product_detail` on single-product sites.** The site root
   is always `homepage`, even when the whole site is about one product.

When two still fit, choose the one the page is **for**.

## 5. Abstention

Abstention is a first-class answer and is never penalised. For `pageType`, emit
`other` rather than guessing between two specific types. A confident wrong
label is worse than an abstention, because a wrong label trains a model and an
abstention only costs a row.
