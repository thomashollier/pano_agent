"""Tests for pano_agent. Run with: python -m pytest tests/ -v"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pano_agent.brief import (
    SceneBrief,
    StyleSpec, SpaceSpec, WindowSpec, DoorSpec, LightingSpec, CameraSpec,
    CardinalView, InterstitialView,
    cardinal_angles,
    interstitial_angles,
)
from pano_agent.build import build_workflow
from pano_agent.prompts import global_prompt, cardinal_task_prompt, interstitial_task_prompt


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def test_cardinal_angles_4():
    assert cardinal_angles(4) == [0, 90, 180, 270]

def test_cardinal_angles_6():
    assert cardinal_angles(6) == [0, 60, 120, 180, 240, 300]

def test_cardinal_angles_8():
    assert cardinal_angles(8) == [0, 45, 90, 135, 180, 225, 270, 315]

def test_cardinal_angles_invalid():
    with pytest.raises(ValueError):
        cardinal_angles(7)
    with pytest.raises(ValueError):
        cardinal_angles(1)

def test_interstitial_angles_4():
    # cardinals at 0, 90, 180, 270 → midpoints at 45, 135, 225, 315
    assert interstitial_angles([0, 90, 180, 270]) == [45, 135, 225, 315]

def test_interstitial_angles_loop_closure():
    # Last interstitial wraps from 270 to 0 → (270 + 360) // 2 % 360 = 315
    angles = interstitial_angles([0, 90, 180, 270])
    assert angles[-1] == 315

def test_interstitial_angles_6():
    assert interstitial_angles([0, 60, 120, 180, 240, 300]) == [30, 90, 150, 210, 270, 330]


# ---------------------------------------------------------------------------
# Brief serialization round-trip
# ---------------------------------------------------------------------------

def _make_test_brief(n_cards: int = 4) -> SceneBrief:
    """A minimal brief usable for build/prompt tests."""
    angles = cardinal_angles(n_cards)
    inter_angles = interstitial_angles(angles)
    cardinals = [
        CardinalView(
            key=f"C{i+1}",
            angle=ang,
            label=f"View {i+1} — {ang}°",
            wall_role=f"wall {i}",
            wall_inventory=f"UPPER: stuff at angle {ang}\nMID: more stuff\nLOWER: floor stuff",
            left_edge_content=f"left of {ang}",
            right_edge_content=f"right of {ang}",
        )
        for i, ang in enumerate(angles)
    ]
    interstitials = []
    for i, ang in enumerate(inter_angles):
        l_idx = i
        r_idx = (i + 1) % n_cards
        interstitials.append(InterstitialView(
            key=f"I{i+1}",
            angle=ang,
            label=f"Corner {i+1} — {ang}°",
            left_neighbor_key=f"C{l_idx+1}",
            right_neighbor_key=f"C{r_idx+1}",
            left_half_content=f"L half {ang}",
            right_half_content=f"R half {ang}",
            corner_description="clean 90° corner",
        ))
    return SceneBrief(
        style=StyleSpec(art_style="painted", palette="warm",
                       materials="wood", atmosphere="cozy",
                       rendering_notes="painterly"),
        space=SpaceSpec(type_label="test room", dimensions="10x10",
                       geometry_rules="flat walls", floor="wood",
                       ceiling="flat", walls="painted"),
        windows=WindowSpec(description="square", placement_rules="evenly spaced"),
        doors=DoorSpec(exterior="front", interior="back"),
        lighting=LightingSpec(
            key_light="window", fill_light="lamps",
            darkest_zone="back", fixed_world_space_description="lighting fixed in space",
        ),
        camera=CameraSpec(),
        cardinals=cardinals,
        interstitials=interstitials,
        reference_cardinal_key="C1",
    )


def test_brief_roundtrip():
    brief = _make_test_brief()
    data = brief.to_dict()
    text = json.dumps(data)
    restored = SceneBrief.from_dict(json.loads(text))
    assert restored.cardinals[0].angle == brief.cardinals[0].angle
    assert restored.cardinals[0].wall_inventory == brief.cardinals[0].wall_inventory
    assert restored.reference_cardinal_key == "C1"


def test_brief_lookup_by_key():
    brief = _make_test_brief(8)
    assert brief.cardinal_by_key("C5").angle == 180


# ---------------------------------------------------------------------------
# Prompt synthesis
# ---------------------------------------------------------------------------

def test_global_prompt_includes_essentials():
    brief = _make_test_brief()
    p = global_prompt(brief)
    assert "GLOBAL CONTEXT" in p
    assert "FLAT-ON" in p  # flat-on block included by default
    assert "90° corners" in p or "90 degree corners" in p.lower()
    # No-recess rule appears in some form
    assert "recess" in p.lower() and "alcove" in p.lower()
    assert "windows" in p.lower()
    assert brief.windows.description in p


def test_global_prompt_skips_flaton_when_disabled():
    brief = _make_test_brief()
    brief.camera.flat_on_required = False
    p = global_prompt(brief)
    assert "PERPENDICULAR ORIENTATION" not in p


def test_cardinal_task_reference_view_uses_flatten_framing():
    brief = _make_test_brief()
    ref = brief.cardinals[0]
    p = cardinal_task_prompt(brief, ref, is_reference=True, has_extra_references=False)
    assert "RE-RENDER" in p or "re-render" in p.lower()
    assert "DO NOT drop" in p


def test_cardinal_task_extra_refs_view_acknowledges_neighbors():
    brief = _make_test_brief()
    side = brief.cardinals[1]
    p = cardinal_task_prompt(brief, side, is_reference=False, has_extra_references=True)
    assert "neighboring" in p.lower() or "neighbor" in p.lower()


def test_cardinal_task_includes_wall_inventory():
    brief = _make_test_brief()
    side = brief.cardinals[1]
    p = cardinal_task_prompt(brief, side, is_reference=False, has_extra_references=False)
    assert side.wall_inventory in p
    assert side.left_edge_content in p
    assert side.right_edge_content in p


def test_interstitial_task_includes_corner_rule():
    brief = _make_test_brief()
    inter = brief.interstitials[0]
    p = interstitial_task_prompt(brief, inter)
    assert "90°" in p or "90 degree" in p.lower()
    assert "no recess" in p.lower() or "NO recess" in p
    assert "Reference A" in p
    assert "Reference B" in p


def test_interstitial_task_references_correct_neighbors():
    brief = _make_test_brief()
    inter = brief.interstitials[0]  # I1: between C1 and C2
    p = interstitial_task_prompt(brief, inter)
    assert "0°" in p  # C1's angle
    assert "90°" in p  # C2's angle


# ---------------------------------------------------------------------------
# Build integrity
# ---------------------------------------------------------------------------

def _validate_workflow_links(wf: dict) -> list[str]:
    """Return list of error strings for any broken/inconsistent links."""
    errors = []
    node_by_id = {n["id"]: n for n in wf["nodes"]}

    # Every link references valid nodes and slots
    for link in wf["links"]:
        lid, src, src_slot, dst, dst_slot, type_ = link
        if src not in node_by_id:
            errors.append(f"link {lid}: src {src} missing")
            continue
        if dst not in node_by_id:
            errors.append(f"link {lid}: dst {dst} missing")
            continue
        src_node = node_by_id[src]
        dst_node = node_by_id[dst]
        if src_slot >= len(src_node.get("outputs", [])):
            errors.append(f"link {lid}: src slot {src_slot} OOB on {src_node['type']}")
        if dst_slot >= len(dst_node.get("inputs", [])):
            errors.append(f"link {lid}: dst slot {dst_slot} OOB on {dst_node['type']}")

    # Every input.link must match the link table
    for n in wf["nodes"]:
        for slot, inp in enumerate(n.get("inputs", [])):
            actual = inp.get("link")
            expected = [l[0] for l in wf["links"] if l[3] == n["id"] and l[4] == slot]
            if expected and actual != expected[0]:
                errors.append(f"node {n['id']} in[{slot}] link mismatch")
            if not expected and actual is not None:
                errors.append(f"node {n['id']} in[{slot}] has stale link {actual}")

    # Every output.links must match the link table
    for n in wf["nodes"]:
        for slot, out in enumerate(n.get("outputs", [])):
            actual = sorted(out.get("links") or [])
            expected = sorted(l[0] for l in wf["links"]
                              if l[1] == n["id"] and l[2] == slot)
            if actual != expected:
                errors.append(f"node {n['id']} out[{slot}] mismatch: have {actual} expected {expected}")
    return errors


def test_build_4_cardinals():
    brief = _make_test_brief(4)
    wf = build_workflow(brief)
    errors = _validate_workflow_links(wf)
    assert not errors, "\n".join(errors)
    # Expect 8 views = 4 cardinals + 4 interstitials
    gems = [n for n in wf["nodes"] if n["type"] == "GeminiImage2Node"]
    assert len(gems) == 8
    saves = [n for n in wf["nodes"] if n["type"] == "SaveImage"]
    assert len(saves) == 8


def test_build_6_cardinals():
    brief = _make_test_brief(6)
    wf = build_workflow(brief)
    errors = _validate_workflow_links(wf)
    assert not errors, "\n".join(errors)
    gems = [n for n in wf["nodes"] if n["type"] == "GeminiImage2Node"]
    assert len(gems) == 12  # 6 cardinals + 6 interstitials


def test_build_8_cardinals():
    brief = _make_test_brief(8)
    wf = build_workflow(brief)
    errors = _validate_workflow_links(wf)
    assert not errors, "\n".join(errors)
    gems = [n for n in wf["nodes"] if n["type"] == "GeminiImage2Node"]
    assert len(gems) == 16


def test_build_minimum_2_cardinals():
    """Edge case: 2 cardinals (180° apart). No 'extra references' batch logic."""
    brief = _make_test_brief(2)
    wf = build_workflow(brief)
    errors = _validate_workflow_links(wf)
    assert not errors, "\n".join(errors)
    # With only 2 cardinals, no cardinal gets the "extra refs" batch
    batches = [n for n in wf["nodes"] if n["type"] == "BatchImagesNode"]
    # Only the interstitial batches (2 of them, halfway between the 2 cardinals)
    assert len(batches) == 2


def test_build_execution_order():
    """Reference cardinal generates first, opposite second, then the rest."""
    brief = _make_test_brief(4)
    wf = build_workflow(brief)
    gems = [n for n in wf["nodes"] if n["type"] == "GeminiImage2Node"]
    # Sort by execution order
    gems_sorted = sorted(gems, key=lambda n: n["order"])
    titles = [n["title"] for n in gems_sorted]
    # First gem must be the reference cardinal (C1)
    assert "[C1]" in titles[0]
    # Second must be the opposite (C3 at 180°)
    assert "[C3]" in titles[1]
    # Then C2 and C4 (in some order, both depend on C1+C3 outputs)
    assert "[C2]" in titles[2] or "[C4]" in titles[2]
    assert "[C2]" in titles[3] or "[C4]" in titles[3]
    # Then interstitials
    assert "[I" in titles[4]


def test_build_reference_cardinal_takes_only_ref_image():
    """The reference cardinal's image input must come directly from LoadImage."""
    brief = _make_test_brief(4)
    wf = build_workflow(brief)
    node_by_id = {n["id"]: n for n in wf["nodes"]}
    ref_gem = next(n for n in wf["nodes"]
                   if n["type"] == "GeminiImage2Node" and "[C1]" in n["title"])
    images_input = ref_gem["inputs"][0]
    assert images_input["link"] is not None
    link = next(l for l in wf["links"] if l[0] == images_input["link"])
    src = node_by_id[link[1]]
    assert src["type"] == "LoadImage"


def test_build_side_cardinal_takes_batch():
    """A side cardinal in 4+ cardinal mode must take a batch as image input."""
    brief = _make_test_brief(4)
    wf = build_workflow(brief)
    node_by_id = {n["id"]: n for n in wf["nodes"]}
    side_gem = next(n for n in wf["nodes"]
                    if n["type"] == "GeminiImage2Node" and "[C2]" in n["title"])
    images_input = side_gem["inputs"][0]
    link = next(l for l in wf["links"] if l[0] == images_input["link"])
    src = node_by_id[link[1]]
    assert src["type"] == "BatchImagesNode"


def test_build_interstitial_takes_batch():
    brief = _make_test_brief(4)
    wf = build_workflow(brief)
    node_by_id = {n["id"]: n for n in wf["nodes"]}
    inter_gem = next(n for n in wf["nodes"]
                     if n["type"] == "GeminiImage2Node" and "[I1]" in n["title"])
    images_input = inter_gem["inputs"][0]
    link = next(l for l in wf["links"] if l[0] == images_input["link"])
    src = node_by_id[link[1]]
    assert src["type"] == "BatchImagesNode"


def test_build_workflow_serializes_to_json():
    """The output must round-trip through JSON without errors."""
    brief = _make_test_brief()
    wf = build_workflow(brief)
    text = json.dumps(wf)
    restored = json.loads(text)
    assert restored["nodes"]
    assert restored["links"]
