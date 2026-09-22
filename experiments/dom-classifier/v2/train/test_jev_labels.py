"""The precedence ladder in jev_labels.py, pinned.

Every case here is one the silver labeller already answers a particular way.
The point of the tests is that Jev and silver give the SAME answer wherever
both can see a node, so a disagreement counted during sampling is a real
disagreement about the page rather than two different conventions.
"""

from __future__ import annotations

from jev_labels import region_from_answers


def answers(node: str, **noul: float) -> dict:
    """Raw answers with one noul per `axis_label` key, e.g. region_footer=0.9."""
    return {f"node_{node}_{key}": {"noul": value} for key, value in noul.items()}


def call(component=None, regions=(), purposes=(), raw=None, node="p#1"):
    return region_from_answers(node, raw or {}, component, list(regions), list(purposes))


def test_nav_inside_a_header_is_navigation_not_site_header():
    # The case the whole ladder exists for: silver labels a <nav> inside a
    # <header> `navigation`, because a node's own landmark beats its container.
    # `navigation` is not in the regions vocabulary at all, so only the
    # componentType axis can carry it.
    assert call(component="navigation_menu", regions=["site_header"])[0] == "navigation"


def test_the_header_itself_is_site_header():
    # Policy section 1 makes every <header> a layout_container, so the shape
    # axis cannot distinguish chrome from any other wrapper; regions does.
    assert call(component="layout_container", regions=["site_header"])[0] == "site_header"


def test_a_link_heavy_footer_is_still_footer():
    # A site footer is almost always mostly link labels. Policy v6 removed the
    # "dominated by a link menu" escape hatch precisely because it swallowed
    # the rule it qualified; the mapping must not reintroduce it.
    result = call(
        component="layout_container",
        regions=["footer"],
        purposes=["navigation"],
        raw=answers("p#1", region_footer=0.93),
    )
    assert result == ("footer", 0.93)


def test_a_menu_inside_the_footer_is_navigation():
    assert call(component="navigation_menu", regions=["footer"])[0] == "navigation"


def test_overlay_beats_where_it_sits():
    assert call(component="dialog", regions=["main_content"])[0] == "dialog"
    assert call(component="layout_container", regions=["overlay", "main_content"])[0] == "dialog"


def test_article_header_is_section_chrome_not_site_chrome():
    assert call(component="layout_container", regions=["article_header"])[0] == "article_chrome"


def test_inside_main_the_shape_decides_wrapper_from_content():
    # Policy section 2 says a <main>'s DESCENDANTS are also main_content, so
    # the regions axis alone cannot tell the <main> from a card inside it.
    assert call(component="layout_container", regions=["main_content"])[0] == "main_content"
    assert call(component="card", regions=["main_content"])[0] == "content_block"
    assert call(component="content_section", regions=["article_body"])[0] == "content_block"


def test_abstention_is_a_real_answer():
    assert call(component="unknown", regions=[]) is None
    assert call(component="text", regions=["main_content"]) is None
    # A wrapper that merely serves navigation is not a navigation menu.
    assert call(component="layout_container", regions=[], purposes=["navigation"]) is None


def test_confidence_comes_from_the_label_that_won():
    result = call(
        component="layout_container",
        regions=["sidebar"],
        raw=answers("p#1", region_sidebar=0.81, region_footer=0.99),
    )
    # 0.81 is the sidebar's own noul; the footer's 0.99 belongs to a label that
    # did not win and must not be reported as this answer's strength.
    assert result == ("sidebar", 0.81)


def test_shape_confidence_is_read_from_the_component_answer():
    raw = {"node_p#1_component_type": {"choice": "navigation_menu", "confidence": 0.88}}
    assert call(component="navigation_menu", regions=[], raw=raw) == ("navigation", 0.88)


def test_a_missing_noul_falls_back_to_the_threshold_not_to_zero():
    # A label present in `observedAxes` cleared 0.75 by construction, so the
    # floor is 0.75 and never 0: a weight of 0 would silently drop the row.
    label, confidence = call(component="layout_container", regions=["footer"])
    assert label == "footer"
    assert confidence == 0.75
