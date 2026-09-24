# Section purpose policy, revision `section-purpose-v3`

One label per page element: **what is this section of the page for?** The
answer drives two audit features: de-duplicating an issue that sits in a widget
repeated across many pages (one finding, not one per page), and noticing a
section that should carry structured data but does not. Class list and schema
mapping: `section-classes-v2.json`.

Judge the element from its packet: tag, role, position, its own text, its
children and links.

v3 changes the granularity rule, and only that. v2 asked for the TIGHTEST
container, so a wrapper around the FAQ list and the list itself could not both
be `faq_block`, and which of two near-identical nodes was "tightest" became the
labelling noise. v3 labels at REGION level:

- Label the element with a purpose when its content IS that section: the
  section itself, a wrapper whose content is only (or almost only) that
  section, or the section's main list. Several nested elements can carry the
  same label.
- Answer `none` when the element is only a FRAGMENT of the section (one
  question of a FAQ, one link of a breadcrumb trail, one review of several), or
  when it also contains substantial unrelated content (a page column, the main
  content, a whole footer that merely includes a newsletter form).

## Classes

Schema-mapped:
- **`faq_block`** two or more question-and-answer pairs.
- **`reviews`** customer reviews or testimonials, or the rating summary for them.
- **`breadcrumbs`** the trail of links from the site root to this page.
- **`business_info`** a business's postal address, opening hours and/or phone as
  a block (not a single phone link in a menu).
- **`product_offer`** the price and buy control (add to cart, buy now, book) of
  the thing this page sells.
- **`recipe_card`** ingredients and steps of a recipe.
- **`event_details`** one event's date, time and place.
- **`author_byline`** the name of the author of this page's content, often with
  a date. Comment authors and testimonial names are not bylines.
- **`video_embed`** an embedded video player the visitor can play. A muted,
  autoplaying background video is `none`.

Non-schema:
- **`ad`** a display, native or sponsored advertisement, an ad slot or an ad
  iframe, including "sponsored"/"promoted" recommendation widgets from ad
  networks.
- **`cookie_consent`** a cookie or privacy consent banner or dialog.
- **`newsletter_signup`** a form or call to action to subscribe by email.
- **`social_share`** buttons that share THIS page to social networks (not links
  to the site's own social profiles).
- **`related_content`** a list of other articles, posts or products offered as
  "related", "recommended" or "read next".
- **`comments`** the discussion thread under the content, or its reply form.
- **`chat_widget`** a live chat or chatbot launcher or window.
- **`promo_banner`** a site-wide announcement or promotion bar (shipping offer,
  sale, notice), usually a thin strip at the top.
- **`pagination`** controls to move between pages of a list (1 2 3 next).
- **`table_of_contents`** a list of in-page links to this page's own sections.
- **`account_login`** a sign-in or registration form or panel.
- **`locale_switcher`** a language, region or currency selector.

- **`none`** anything else, including headers, footers, menus and main content.

When two fit, choose the more specific purpose. When unsure, answer `none`: a
confident wrong label is worse than an abstention.
