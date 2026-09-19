"""Tests for the v5 G1 pilot: sampler, reviewer batch, scorer.

Loaded by path because the experiment directory is not a package and both
modules read ``DOM_CLASSIFIER_DATA_ROOT`` at import time. The root is set to a
temporary directory here, so importing never touches the private corpus.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

_SANDBOX = tempfile.mkdtemp(prefix="dom-classifier-test-root-")
os.environ.setdefault("DOM_CLASSIFIER_DATA_ROOT", _SANDBOX)
os.environ["DOM_CLASSIFIER_DATA_ROOT"] = os.environ.get("DOM_CLASSIFIER_DATA_ROOT", _SANDBOX)


def _load(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


build_pilot = _load("build_pilot_v5")
luna_v5 = _load("luna_v5_batch")
score = _load("score_pilot_v5")

LABELS = Path(__file__).resolve().parents[1] / "v2" / "labels"


def sha(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def luna_row(**overrides: Any) -> dict[str, Any]:
    row = {
        "recordType": "node",
        "split": "validation",
        "pageId": "page_1",
        "nodeId": "node_1",
        "captureHash": "sha256:c",
        "inputSha256": "sha256:i",
        "input": "runtime-dom-text-v1\ntag=footer\nancestors=body\ntext=Contact and legal",
    }
    row.update(overrides)
    return row


# --- stratification -------------------------------------------------------


@pytest.mark.parametrize(
    ("tag", "text", "expected"),
    [
        ("footer", "Contact", "footer"),
        ("main", "Body", "main"),
        ("header", "Logo", "header"),
        ("div", "Something", "div"),
        ("figure", "Caption", "figure"),
        ("iframe", "[empty]", "iframe"),
        ("img", "[empty]", "empty_media"),
        ("img", "", "empty_media"),
        ("img", "A photo of a mug", "other"),
        ("nav", "Home About", "other"),
    ],
)
def test_stratum_of(tag: str, text: str, expected: str) -> None:
    row = luna_row(input=f"runtime-dom-text-v1\ntag={tag}\nancestors=body\ntext={text}")
    assert build_pilot.stratum_of(row) == expected


def test_media_tag_with_text_is_not_empty_media() -> None:
    """`empty_media` is about the *absence of text*, not about the tag."""
    assert build_pilot.stratum_of(luna_row(input="x\ntag=img\ntext=Product photo")) == "other"


def test_order_key_is_content_derived_and_stable() -> None:
    row = luna_row()
    assert build_pilot.order_key(row) == build_pilot.order_key(dict(row))
    moved = dict(row, nodeId="node_2")
    assert build_pilot.order_key(row) != build_pilot.order_key(moved)


def test_allocate_fills_quotas_and_redistributes_a_short_stratum() -> None:
    strata = {
        name: [luna_row(nodeId=f"{name}-{index}") for index in range(quota)]
        for name, quota in build_pilot.QUOTAS.items()
    }
    strata["iframe"] = strata["iframe"][:3]  # five short
    strata["div"] = strata["div"] + [luna_row(nodeId=f"div-extra-{i}") for i in range(10)]
    selected = build_pilot.allocate(strata, sum(build_pilot.QUOTAS.values()))
    assert len(selected) == sum(build_pilot.QUOTAS.values())
    ids = [row["nodeId"] for row in selected]
    assert len(set(ids)) == len(ids)
    assert sum(1 for value in ids if value.startswith("div-")) == build_pilot.QUOTAS["div"] + 5


def test_allocate_is_deterministic() -> None:
    strata = {
        name: [luna_row(nodeId=f"{name}-{index}") for index in range(quota + 5)]
        for name, quota in build_pilot.QUOTAS.items()
    }
    first = [row["nodeId"] for row in build_pilot.allocate(strata, 200)]
    second = [row["nodeId"] for row in build_pilot.allocate(strata, 200)]
    assert first == second


def test_quotas_sum_to_the_pilot_size() -> None:
    assert sum(build_pilot.QUOTAS.values()) == 200


# --- the reviewer leg -----------------------------------------------------


def packet(**overrides: Any) -> dict[str, Any]:
    body = overrides.pop("input", "runtime-dom-text-v2\ntag=footer\nrole=none\ntext=Contact")
    row = {
        "recordType": "node",
        "split": "validation",
        "pageId": "page_1",
        "nodeId": "node_1",
        "captureHash": "sha256:c",
        "input": body,
        "inputSha256": sha(body),
        "stratum": "footer",
    }
    row.update(overrides)
    return row


def test_validate_packet_rejects_an_input_hash_that_does_not_bind() -> None:
    with pytest.raises(ValueError, match="inputSha256"):
        luna_v5.validate_packet(packet(inputSha256="sha256:wrong"))


def test_validate_packet_rejects_a_page_packet_carrying_a_node_id() -> None:
    with pytest.raises(ValueError, match="nodeId must be null"):
        luna_v5.validate_packet(packet(recordType="page"))


def test_blind_packet_carries_no_identity() -> None:
    blind = luna_v5.blind(packet())
    assert set(blind) == {"packetKey", "recordType", "input"}
    serialized = json.dumps(blind)
    for secret in ("page_1", "node_1", "sha256:c"):
        assert secret not in serialized


def test_packet_key_changes_when_the_input_changes() -> None:
    first = luna_v5.packet_key(packet())
    second = luna_v5.packet_key(packet(input="runtime-dom-text-v2\ntag=footer\ntext=Other"))
    assert first != second


def test_prompt_embeds_the_policy_bytes_verbatim() -> None:
    policy, policy_hash = luna_v5.load_policy()
    vocabulary = luna_v5.load_vocabulary()
    prompt = luna_v5.prompt_for([packet()], vocabulary, policy, policy_hash)
    assert policy in prompt
    assert policy_hash in prompt
    assert "label the site-wide chrome `layout_container`" in prompt


def test_vocabulary_matches_the_versioned_mapping_files() -> None:
    vocabulary = luna_v5.load_vocabulary()
    assert vocabulary["pageType"] == json.loads((LABELS / "page-type-map-v5.json").read_text())["classes"]
    assert len(vocabulary["pageType"]) == 11
    assert vocabulary["purposes"] == sorted(
        json.loads((LABELS / "purposes-v5.json").read_text())["retained"]
    )
    assert len(vocabulary["purposes"]) == 12
    assert "media_playback" not in vocabulary["purposes"]


def observation(axis: str, labels: list[str], complete: bool = True) -> dict[str, Any]:
    return {"axis": axis, "complete": complete, "labels": labels}


def record(packet_row: dict[str, Any], observations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "packetKey": luna_v5.packet_key(packet_row),
        "source": "luna",
        "gold": False,
        "model": luna_v5.MODEL,
        "labelingMethod": "independent_reasoned_review",
        "promptRevision": luna_v5.PROMPT_REVISION,
        "evidence": "Site chrome grouping links and legal text.",
        "observations": observations,
    }


def test_validate_record_accepts_a_well_formed_node_answer() -> None:
    vocabulary = luna_v5.load_vocabulary()
    row = packet()
    result = luna_v5.validate_record(
        record(
            row,
            [
                observation("componentType", ["layout_container"]),
                observation("regions", ["footer"]),
                observation("purposes", []),
            ],
        ),
        row,
        vocabulary,
    )
    assert result["observedAxes"]["componentType"]["labels"] == ["layout_container"]
    assert result["observedAxes"]["purposes"] == {"complete": True, "labels": []}
    assert result["pageId"] == "page_1"
    assert result["stratum"] == "footer"


@pytest.mark.parametrize(
    ("observations", "message"),
    [
        ([observation("pageType", ["docs"])], "invalid for a node"),
        ([observation("componentType", ["layout_container", "card"])], "exactly one label"),
        ([observation("purposes", ["media_playback"])], "off-taxonomy"),
        ([observation("regions", ["unknown", "footer"])], "unknown must stand alone"),
        ([observation("regions", ["footer"]), observation("regions", ["sidebar"])], "duplicate"),
    ],
)
def test_validate_record_rejects(observations: list[dict[str, Any]], message: str) -> None:
    row = packet()
    with pytest.raises(ValueError, match=message):
        luna_v5.validate_record(record(row, observations), row, luna_v5.load_vocabulary())


def test_validate_record_rejects_a_v4_prompt_revision() -> None:
    row = packet()
    stale = record(row, [observation("componentType", ["layout_container"])])
    stale["promptRevision"] = "dom-suggestions-v4"
    with pytest.raises(ValueError, match="provenance"):
        luna_v5.validate_record(stale, row, luna_v5.load_vocabulary())


def test_a_batch_is_all_or_nothing() -> None:
    vocabulary = luna_v5.load_vocabulary()
    first, second = packet(), packet(nodeId="node_2", input="runtime-dom-text-v2\ntag=nav\ntext=Home")
    second["inputSha256"] = sha(second["input"])
    payload = {
        "records": [
            record(first, [observation("componentType", ["layout_container"])]),
            record(second, [observation("componentType", ["navigation_menu", "list"])]),
        ]
    }
    with pytest.raises(ValueError):
        luna_v5.validated_batch(json.dumps(payload), [first, second], vocabulary, "prompt", 1, 10)


def test_page_type_schema_offers_exactly_one_label() -> None:
    row = packet(recordType="page", nodeId=None)
    schema = luna_v5.response_schema([row], luna_v5.load_vocabulary())
    options = schema["properties"]["records"]["items"]["properties"]["observations"]["items"]["anyOf"]
    page_schema = next(o for o in options if o["properties"]["axis"]["const"] == "pageType")
    assert page_schema["properties"]["labels"]["minItems"] == 1
    assert page_schema["properties"]["labels"]["maxItems"] == 1


# --- scoring --------------------------------------------------------------


def test_jev_v4_sets_thresholds_at_075_inclusive() -> None:
    row = {
        "softTargets": {
            "componentType": {"choice": "card"},
            "regions": [
                {"label": "footer", "yesProbability": 0.75},
                {"label": "sidebar", "yesProbability": 0.7499},
            ],
        }
    }
    parsed = score.jev_v4_sets(row)
    assert parsed["componentType"] == "card"
    assert parsed["regions"] == {"footer"}


def test_incomplete_axes_are_not_read_as_empty_observations() -> None:
    parsed = score.luna_v4_sets(
        {"observedAxes": {"regions": {"complete": False, "labels": ["footer"]}}}
    )
    assert "regions" not in parsed


def test_agreement_counts_only_rows_where_both_observed_the_axis() -> None:
    left = {("v", "p1", "n1"): {"componentType": "card"}, ("v", "p2", "n2"): {"componentType": "list"}}
    right = {("v", "p1", "n1"): {"componentType": "card"}, ("v", "p2", "n2"): {}}
    result = score.agreement(left, right, "componentType")
    assert (result["n"], result["agreed"], result["rate"]) == (1, 1, 1.0)


def test_agreement_records_the_direction_of_a_confusion() -> None:
    left = {("v", "p", "n"): {"componentType": "content_section"}}
    right = {("v", "p", "n"): {"componentType": "layout_container"}}
    result = score.agreement(left, right, "componentType")
    assert result["topConfusions"] == [
        {"left": "content_section", "right": "layout_container", "count": 1}
    ]


def test_page_type_projection_collapses_to_eleven_classes() -> None:
    mapping = json.loads((LABELS / "page-type-map-v5.json").read_text())["map"]
    assert score.project_page_types({"blog_post", "news_article"}, mapping) == {"article"}
    assert score.project_page_types({"docs_article", "tutorial"}, mapping) == {"docs"}
    assert score.project_page_types({"checkout"}, mapping) == {"other"}
    assert score.project_page_types("blog_post", mapping) == {"article"}


def test_every_taxonomy_page_type_has_a_projection() -> None:
    taxonomy = json.loads((LABELS.parents[1] / "taxonomy.json").read_text())
    mapping = json.loads((LABELS / "page-type-map-v5.json").read_text())
    for label in taxonomy["allowedLabels"]["pageTypes"]:
        assert label in mapping["map"], label
        assert mapping["map"][label] in mapping["classes"]


def test_purpose_projection_drops_only_unsupported_labels() -> None:
    retained = set(json.loads((LABELS / "purposes-v5.json").read_text())["retained"])
    assert score.project_purposes({"navigation", "media_playback"}, retained) == {"navigation"}


def test_abstention_scopes_to_the_records_the_axis_applies_to() -> None:
    node_key, page_key = ("v", "p1", "n1"), ("v", "p2", "")
    rows = {node_key: {"componentType": "unknown"}, page_key: {"pageType": {"docs"}}}
    result = score.abstention(rows, "componentType", "unknown", {node_key})
    assert result == {
        "n": 1,
        "eligible": 1,
        "abstainToken": "unknown",
        "abstainRate": 1.0,
        "emptySet": 0.0,
        "omitted": 0.0,
    }


def test_abstention_reports_an_axis_the_annotator_omitted() -> None:
    keys = {("v", "p1", "n1"), ("v", "p2", "n2")}
    rows = {("v", "p1", "n1"): {"componentType": "card"}, ("v", "p2", "n2"): {}}
    assert score.abstention(rows, "componentType", "unknown", keys)["omitted"] == 0.5
