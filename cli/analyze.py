"""
Vision pass: image → SceneBrief.

Uses the Anthropic API with a vision-capable Claude model. The prompt asks
Claude to fill out a structured JSON matching SceneBrief, with strong
guidance on what each field should contain (especially wall_inventory,
which is the field that most affects render quality).

The agent makes TWO calls:

  1. ANALYSIS — produces global style/space/lighting and the inventory
     for the *visible* cardinal (the one that matches the reference image).

  2. EXTRAPOLATION — given the analysis from step 1, infer the contents
     of the remaining cardinals and corners. This call doesn't see the
     image again; it works from the structured data so its outputs are
     internally consistent.

Splitting the calls keeps each prompt focused and lets the user inspect
the analysis before the extrapolation runs (with --analysis-only).
"""
from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any

from pano_agent.brief import (
    CardinalView,
    InterstitialView,
    SceneBrief,
    StyleSpec, SpaceSpec, WindowSpec, DoorSpec, LightingSpec, CameraSpec,
    cardinal_angles,
    interstitial_angles,
)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

ANALYSIS_SYSTEM_PROMPT = """You are an architectural and visual analyst examining a single illustrated interior.
Your job is to produce a structured JSON description of the space that another LLM
will use to write image-generation prompts for a 360° panorama. The description must
be specific enough that someone could redraw the scene from the JSON alone.

Key principles:

1. WALL INVENTORY format. For the wall visible in the reference, describe its
   contents as horizontal zones from top to bottom (UPPER, MID, LOWER), then
   list every furniture / appliance / fixture / opening on that wall by
   position. Empty wall is rare; if you see paneling above a counter, ask
   yourself "is there really nothing there, or am I about to drop the upper
   cabinets that the model will then drop too?" Be exhaustive.

2. WORLD-SPACE LIGHTING. Lighting should be described as fixed in the world,
   not relative to the camera. Identify the dominant light source and its
   position relative to the visible wall (front/back/side).

3. CANONICAL specifications. For repeated elements (windows, doors), establish
   ONE canonical description that all instances must match. The most common
   panorama failure is windows changing shape between views; the canon prevents
   that.

4. NO speculation about what you cannot see. Fields about unseen walls go in
   the extrapolation pass, not this one.
"""


def _analysis_user_prompt(n_cardinals: int) -> str:
    angles = cardinal_angles(n_cardinals)
    return f"""Analyze the attached reference image and produce a JSON object with this exact structure:

{{
  "style": {{
    "art_style": "...",
    "palette": "...",
    "materials": "...",
    "atmosphere": "...",
    "rendering_notes": "..."
  }},
  "space": {{
    "type_label": "...",
    "dimensions": "...",
    "geometry_rules": "...",
    "floor": "...",
    "ceiling": "...",
    "walls": "..."
  }},
  "windows": {{
    "description": "...",
    "placement_rules": "..."
  }},
  "doors": {{
    "exterior": "...",
    "interior": "..."
  }},
  "lighting": {{
    "key_light": "...",
    "fill_light": "...",
    "darkest_zone": "...",
    "fixed_world_space_description": "..."
  }},
  "visible_cardinal": {{
    "wall_role": "...",
    "wall_inventory": "...",
    "left_edge_content": "...",
    "right_edge_content": "..."
  }},
  "spatial_inferences": {{
    "long_axis": "...",
    "wall_assignments": {{
       "front_short_wall_at_0": "what wall the camera is facing in the reference",
       "back_short_wall_at_180": "what's on the opposite wall",
       "left_long_wall_at_270": "what's on the camera's left wall",
       "right_long_wall_at_90": "what's on the camera's right wall"
    }},
    "notes": "any spatial cues you used (door placements visible in frame edges, depth cues, etc.)"
  }}
}}

Field guidance:

- space.geometry_rules: A strict rule statement like "all walls flat; 90°
  corners; no recesses, alcoves, closets, bump-outs, or wall-jogs anywhere."
  This will be repeated to the image generator to prevent it from inventing
  architectural depth.

- windows.description: One canonical shape/size/material for ALL windows.
  Sentence form. e.g. "rounded-top / squared-bottom rectangle, ~3 ft × 2.5 ft,
  thin dark wooden frame, frosted glass with a few horizontal painted highlight
  streaks." DO NOT describe variation between windows here.

- windows.placement_rules: Per-wall counts and order, e.g. "left wall: 2
  windows + 1 exterior door, in order WINDOW→DOOR→WINDOW from front to back."

- lighting.fixed_world_space_description: A paragraph explaining that the
  lighting is fixed in world space — naming the position of the key light,
  the fill, and what zone is darkest. This paragraph will be embedded in
  every per-view prompt.

- visible_cardinal.wall_inventory: The single most important field. Describe
  the wall facing the camera in the reference image as horizontal zones:
    UPPER ZONE (top of wall to mid-wall): [explicit list]
    MID ZONE (counter / fixture level): [explicit list]
    LOWER ZONE (mid-wall to floor): [explicit list]
  For each zone, list every visible item left to right. Include details like
  "brown wooden upper cabinets fill most of the upper zone" if that's what
  you see — the goal is that no zone gets dropped by the image generator.

- visible_cardinal.left_edge_content / right_edge_content: What's visible at
  the left and right edges of the reference frame, where the side walls start
  to enter. These define seams to neighboring views.

- spatial_inferences: A best-guess spatial map. The reference image will be
  treated as the 0° cardinal of a {n_cardinals}-cardinal panorama with views
  at angles {angles}. Identify which wall the camera faces (front_short),
  what's opposite (back_short), and what's on the left and right long walls.
  Use any visible cues — partial doorways at edges, depth perception, items
  bleeding in from adjacent walls.

Output ONLY valid JSON. No markdown fences, no preamble, no commentary.
"""


EXTRAPOLATION_SYSTEM_PROMPT = """You are a writer continuing a structured scene description.
Given a JSON analysis of one cardinal view of an interior space, you must
write the corresponding inventories for the remaining cardinals and the
interstitial corner views.

You do NOT have access to the image. Work entirely from the analysis JSON,
and from the user-provided spatial inferences. Be internally consistent:
windows must match the canonical spec, doors must match their canon,
elements that bridge views must be referenced in both views.

For each unseen view, your wall_inventory should follow the same UPPER /
MID / LOWER zone format as the visible cardinal. If the user's spatial
inferences say "the back wall has a TV and a bedroom door," your inventory
must include those items in the right zones with reasonable detail.
"""


def _extrapolation_user_prompt(analysis: dict[str, Any], n_cardinals: int) -> str:
    angles = cardinal_angles(n_cardinals)
    inter_angles = interstitial_angles(angles)
    return f"""Given this analysis of the visible cardinal (0°), write the rest of the panorama.

Analysis:
```json
{json.dumps(analysis, indent=2)}
```

The panorama has {n_cardinals} cardinal views at angles {angles} and
{len(inter_angles)} interstitial corner views at angles {inter_angles}.

The visible cardinal is at 0° and is already described. You must produce:

1. The remaining {n_cardinals - 1} cardinal views (angles {angles[1:]}).
2. All {len(inter_angles)} interstitial views.

Output JSON with this exact structure:

{{
  "remaining_cardinals": [
    {{
      "angle": 90,
      "wall_role": "...",
      "label": "...",
      "wall_inventory": "...",
      "left_edge_content": "...",
      "right_edge_content": "..."
    }}
    // ... one entry per remaining cardinal
  ],
  "interstitials": [
    {{
      "angle": 45,
      "left_neighbor_angle": 0,
      "right_neighbor_angle": 90,
      "label": "...",
      "left_half_content": "...",
      "right_half_content": "...",
      "corner_description": "..."
    }}
    // ... one entry per interstitial
  ]
}}

Field guidance:

- For each cardinal beyond the first, treat the camera as having pivoted
  to face that wall flat-on. The wall_inventory should describe what's on
  THAT wall, in UPPER/MID/LOWER zones, with windows / doors / fixtures /
  furniture as inferred from the spatial_inferences in the analysis.

- left_edge_content for each cardinal should describe the corner where
  this wall meets the previous wall (counter-clockwise neighbor); right_edge
  is the clockwise neighbor. These edges will be where the panorama seams.

- Cardinal labels should be human-readable like "RIGHT (90°) — [what's
  notable about this wall]". Use angles + role.

- For interstitials, the corner_description must explicitly state that the
  two walls meet at a clean 90° corner with no recess, alcove, closet, or
  setback — image models love to invent architectural depth at corners
  and we have to tell them not to.

- left_half_content and right_half_content for interstitials describe what
  fills each half of the frame when the camera is pointed into the corner.
  The center of the frame is the vertical corner seam itself.

- Continuity: if the analysis lists cardboard boxes at the front-left of
  the visible cardinal, those same boxes appear at the right-edge of the
  90°-counter-clockwise cardinal AND across the entire 315° interstitial.
  Track these continuity anchors and reference them by name in every view
  they appear in.

Output ONLY valid JSON. No markdown fences, no preamble.
"""


# ---------------------------------------------------------------------------
# API call machinery
# ---------------------------------------------------------------------------

def _call_claude_vision(
    image_bytes: bytes,
    image_media_type: str,
    system: str,
    user: str,
    model: str,
    api_key: str,
) -> str:
    """One vision call. Returns assistant text."""
    try:
        import anthropic
    except ImportError:
        raise RuntimeError(
            "cli.analyze requires the `anthropic` package. "
            "Install with: pip install anthropic"
        )

    client = anthropic.Anthropic(api_key=api_key)
    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    msg = client.messages.create(
        model=model,
        max_tokens=8000,
        system=system,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_media_type,
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": user},
                ],
            }
        ],
    )
    return msg.content[0].text


def _call_claude_text(
    system: str,
    user: str,
    model: str,
    api_key: str,
) -> str:
    """Text-only call (for the extrapolation pass)."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=8000,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text


def _parse_json(text: str) -> dict[str, Any]:
    """Tolerate markdown fences and stray prose around JSON."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    # Find first { and last } — handles models that add commentary.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object found in response: {text[:200]}")
    return json.loads(text[start : end + 1])


def _media_type_for(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(ext, "image/png")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyze_image(
    image_path: Path,
    n_cardinals: int,
    anthropic_key: str | None = None,
    model: str = "claude-opus-4-5",
    verbose: bool = False,
) -> SceneBrief:
    """End-to-end vision pass: image → SceneBrief."""
    api_key = anthropic_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "no Anthropic API key (pass --api-key or set ANTHROPIC_API_KEY)"
        )

    image_bytes = image_path.read_bytes()
    media_type = _media_type_for(image_path)

    if verbose:
        print(f"[analyze] reading {image_path} ({len(image_bytes)} bytes, {media_type})")
        print(f"[analyze] step 1: vision analysis ({model})")

    analysis_text = _call_claude_vision(
        image_bytes=image_bytes,
        image_media_type=media_type,
        system=ANALYSIS_SYSTEM_PROMPT,
        user=_analysis_user_prompt(n_cardinals),
        model=model,
        api_key=api_key,
    )
    analysis = _parse_json(analysis_text)

    if verbose:
        print(f"[analyze] step 2: extrapolation ({model})")

    extrapolation_text = _call_claude_text(
        system=EXTRAPOLATION_SYSTEM_PROMPT,
        user=_extrapolation_user_prompt(analysis, n_cardinals),
        model=model,
        api_key=api_key,
    )
    extrapolation = _parse_json(extrapolation_text)

    return _assemble_brief(analysis, extrapolation, n_cardinals)


# ---------------------------------------------------------------------------
# Brief assembly
# ---------------------------------------------------------------------------

def _assemble_brief(
    analysis: dict[str, Any],
    extrapolation: dict[str, Any],
    n_cardinals: int,
) -> SceneBrief:
    """Merge the two API responses into a SceneBrief."""
    brief = SceneBrief(
        style=StyleSpec(**analysis.get("style", {})),
        space=SpaceSpec(**analysis.get("space", {})),
        windows=WindowSpec(**analysis.get("windows", {})),
        doors=DoorSpec(**analysis.get("doors", {})),
        lighting=LightingSpec(**analysis.get("lighting", {})),
        camera=CameraSpec(),
    )

    # Cardinal 0 is the reference image (described in analysis.visible_cardinal).
    angles = cardinal_angles(n_cardinals)
    visible = analysis.get("visible_cardinal", {})
    brief.cardinals.append(CardinalView(
        key="C1",
        angle=angles[0],
        label=f"View 1 — {angles[0]}° — {visible.get('wall_role', 'reference view')}",
        wall_role=visible.get("wall_role", ""),
        wall_inventory=visible.get("wall_inventory", ""),
        left_edge_content=visible.get("left_edge_content", ""),
        right_edge_content=visible.get("right_edge_content", ""),
    ))

    # Remaining cardinals from extrapolation
    extra_cards = extrapolation.get("remaining_cardinals", [])
    extra_by_angle = {c.get("angle"): c for c in extra_cards}
    for i, ang in enumerate(angles[1:], start=2):
        c = extra_by_angle.get(ang, {})
        brief.cardinals.append(CardinalView(
            key=f"C{i}",
            angle=ang,
            label=c.get("label", f"View {i} — {ang}°"),
            wall_role=c.get("wall_role", ""),
            wall_inventory=c.get("wall_inventory", ""),
            left_edge_content=c.get("left_edge_content", ""),
            right_edge_content=c.get("right_edge_content", ""),
        ))

    # Interstitials
    inter_angles = interstitial_angles(angles)
    extra_inters = extrapolation.get("interstitials", [])
    inters_by_angle = {it.get("angle"): it for it in extra_inters}
    angle_to_key = {c.angle: c.key for c in brief.cardinals}
    for i, ang in enumerate(inter_angles, start=1):
        it = inters_by_angle.get(ang, {})
        l_neighbor_angle = it.get("left_neighbor_angle", angles[i - 1])
        r_neighbor_angle = it.get("right_neighbor_angle", angles[i % len(angles)])
        brief.interstitials.append(InterstitialView(
            key=f"I{i}",
            angle=ang,
            label=it.get("label", f"View {i}.5 — {ang}°"),
            left_neighbor_key=angle_to_key.get(l_neighbor_angle, "C1"),
            right_neighbor_key=angle_to_key.get(r_neighbor_angle, "C1"),
            left_half_content=it.get("left_half_content", ""),
            right_half_content=it.get("right_half_content", ""),
            corner_description=it.get("corner_description", ""),
        ))

    return brief
