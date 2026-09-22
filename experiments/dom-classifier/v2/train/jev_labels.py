"""Map Jev's three label axes onto the region head's ten mutually-exclusive classes.

Why this file exists
--------------------
Silver region labels come from the page's own landmarks, so a landmark-less
page produces no `footer` and no `navigation` row at ANY corpus size. The
region head therefore learns those classes only from pages that declare them,
and is asked at runtime about pages that do not. Growing the crawl does not
touch that: the labels are missing exactly where the model is weak.

Jev reads a node's evidence rather than its tag, so it can put `footer` on a
`<div class="site-foot">`. That makes it the only label source that can cover
the div-soup half of the distribution.

The mapping problem
-------------------
Jev answers three axes and the region head has one. The axes are not
interchangeable:

* `regions` (multi-label) answers WHERE the node sits: `site_header`, `footer`,
  `sidebar`, `main_content`, `article_header`, `overlay`, `article_body`.
* `componentTypes` (single) answers WHAT SHAPE it is: `navigation_menu`,
  `form`, `dialog`, `search`, `content_section`, `layout_container`, `card`.
* `purposes` (multi-label) answers WHAT IT DOES: `navigation`, `search`.

`navigation` is the case that proves the axes have to be combined rather than
picked between. It is NOT in the regions vocabulary at all — the 822-node
div-soup sample recorded `regionsTop` of footer/main_content/site_header/
article_body and zero navigation — while `navigation_menu` is the second
largest componentType. A mapping that read only `regions` would produce no
navigation labels whatsoever, which is the exact gap this is meant to close.

Precedence
----------
The ladder below mirrors what `silver-labels.ts` does with landmarks, so a node
Jev and the silver labeller both see gets the same answer from both wherever
they agree, and a disagreement is a real disagreement rather than an artefact
of two different conventions:

  1. dialog       - an overlay is outside the page flow, so it wins over where
                    it happens to sit in the DOM.
  2. navigation   - a `<nav>` inside a `<header>` is `navigation` under the
                    silver rules (its own landmark beats its container), so
                    navigation_menu has to beat site_header here too.
  3. search, form - same reason: both are landmark roles in their own right.
  4. site_header, footer, sidebar - site chrome.
  5. article_chrome - a section's own header, which is NOT site chrome.
  6. main_content - only for a node whose SHAPE is a wrapper. Policy section 2
                    says a `<main>`'s descendants are also `main_content`, so
                    the regions axis alone cannot tell the `<main>` element
                    from a card inside it; the componentType breaks the tie.
  7. content_block - content sitting inside main.

A node whose answers reach none of these returns None and is simply not
trained on. Abstention costs a row; a confident wrong label trains a model.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

# The region head's classes, matching REGION_CLASSES in sample.py.
DIALOG_SHAPES = {"dialog", "drawer", "popover", "tooltip", "notification"}
WRAPPER_SHAPES = {"layout_container"}
CONTENT_SHAPES = {
    "content_section",
    "card",
    "article",
    "list",
    "hero",
    "banner",
    "table",
    "media_gallery",
    "purchase_panel",
    "specification_list",
    "rating_summary",
    "review_list",
    "author_card",
    "comment_thread",
    "accordion",
    "tabs",
}

#: `noul` at or above this counts as the axis asserting the label. It is the
#: same threshold `run-divsoup-v7.ts` uses to build `observedAxes`, so a label
#: read from `observedAxes` and one recomputed from `rawAnswers` agree.
NOUL_THRESHOLD = 0.75


def _noul(raw: dict, node_id: str, kind: str, label: str) -> float:
    """The model's probability for one multi-label answer, 0.0 when absent."""
    answer = raw.get(f"node_{node_id}_{kind}{label}")
    if isinstance(answer, dict):
        value = answer.get("noul")
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def region_from_answers(
    node_id: str,
    raw: dict,
    component_type: str | None,
    regions: list[str],
    purposes: list[str],
) -> tuple[str, float] | None:
    """One region class and a confidence, or None to abstain.

    `regions` and `purposes` are the noul-filtered sets the runner already
    wrote; `raw` is consulted only for the confidence of the label that wins,
    so the decision and its strength come from the same number.
    """
    region_set = set(regions)
    purpose_set = set(purposes)
    shape = component_type or "unknown"

    def region_conf(label: str, floor: float = NOUL_THRESHOLD) -> float:
        return max(_noul(raw, node_id, "region_", label), floor)

    def purpose_conf(label: str, floor: float = NOUL_THRESHOLD) -> float:
        return max(_noul(raw, node_id, "purpose_", label), floor)

    def shape_conf(floor: float = NOUL_THRESHOLD) -> float:
        answer = raw.get(f"node_{node_id}_component_type")
        if isinstance(answer, dict):
            value = answer.get("confidence")
            if isinstance(value, (int, float)):
                return float(value)
        return floor

    # 1. overlays
    if shape in DIALOG_SHAPES:
        return "dialog", shape_conf()
    if "overlay" in region_set:
        return "dialog", region_conf("overlay")

    # 2. navigation: componentType is the ONLY axis that carries it.
    if shape == "navigation_menu":
        return "navigation", shape_conf()

    # 3. the other landmark roles
    if shape == "search":
        return "search", shape_conf()
    if shape == "form":
        return "form", shape_conf()

    # 4. site chrome
    if "site_header" in region_set:
        return "site_header", region_conf("site_header")
    if "footer" in region_set:
        return "footer", region_conf("footer")
    if "sidebar" in region_set:
        return "sidebar", region_conf("sidebar")

    # 5. a section's own chrome, which is not the site's
    if "article_header" in region_set:
        return "article_chrome", region_conf("article_header")

    # 6 / 7. inside main: the shape decides wrapper from content
    inside_main = bool(region_set & {"main_content", "article_body"})
    if inside_main:
        if shape in WRAPPER_SHAPES:
            return "main_content", region_conf("main_content")
        if shape in CONTENT_SHAPES:
            return "content_block", shape_conf()
        return None

    # A purpose of navigation with no navigation shape is weak evidence on its
    # own: a footer serves navigation and is still a footer. Not a label.
    if "navigation" in purpose_set and shape in WRAPPER_SHAPES and not region_set:
        return None
    return None


def load_jev_region_labels(path: Path) -> dict[tuple[str, int], tuple[str, float]]:
    """Read a `run-divsoup-v7.ts` output file into {(pageId, nodeId): (class, conf)}.

    The runner appends, and a resumed run can write a page twice; the LAST row
    for a node wins, which is the one from the most recent complete request.
    """
    out: dict[tuple[str, int], tuple[str, float]] = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:  # type: ignore[operator]
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("recordType") != "node":
                continue
            node_ref = row.get("nodeId")
            if not isinstance(node_ref, str) or "#" not in node_ref:
                continue
            page_id, _, index = node_ref.rpartition("#")
            try:
                node_id = int(index)
            except ValueError:
                continue
            axes = row.get("observedAxes") or {}
            component = (axes.get("componentType") or {}).get("labels")
            regions = (axes.get("regions") or {}).get("labels") or []
            purposes = (axes.get("purposes") or {}).get("labels") or []
            mapped = region_from_answers(
                node_ref,
                row.get("rawAnswers") or {},
                component if isinstance(component, str) else None,
                list(regions),
                list(purposes),
            )
            if mapped is not None:
                out[(page_id, node_id)] = mapped
    return out
