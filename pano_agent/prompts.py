"""
Prompt synthesis: SceneBrief → the strings that go into Comfy's prompt nodes.

Three template families:
  - global_prompt(brief): the shared prefix attached to every view
  - cardinal_task_prompt(brief, cardinal): the per-view task for a flat-on cardinal
  - interstitial_task_prompt(brief, interstitial): the per-view task for a corner

These are pure functions — no API calls. They encode the rules we've worked
out: flat-on rendering, wall-inventory format, no-recess corner rule,
canonical window/door spec, world-space lighting, edge seaming.
"""
from __future__ import annotations

from .brief import (
    CardinalView,
    InterstitialView,
    SceneBrief,
)


# ---------------------------------------------------------------------------
# GLOBAL prompt
# ---------------------------------------------------------------------------

def global_prompt(brief: SceneBrief) -> str:
    s = brief.style
    sp = brief.space
    w = brief.windows
    d = brief.doors
    lg = brief.lighting
    c = brief.camera

    return f"""GLOBAL CONTEXT — applies to every view (do not deviate):

STYLE: {s.art_style}. {s.rendering_notes} Palette: {s.palette}. Materials: {s.materials}. Atmosphere: {s.atmosphere}.

SPACE — STRICT GEOMETRY (do NOT deviate):
- {sp.type_label}, {sp.dimensions}.
- {sp.geometry_rules}
- Floor: {sp.floor}
- Ceiling: {sp.ceiling}
- Walls: {sp.walls}
- All walls are FLAT and meet at simple 90° corners. There are NO architectural recesses, NO alcoves, NO closets, NO bump-outs, NO wall-jogs, NO setbacks, NO partition walls, NO door-frames-set-back-into-a-recess. Every corner is a clean perpendicular meeting of two flat planes. If you find yourself adding depth to a corner — STOP. The corner is just two flat walls touching.

CANONICAL WINDOW SPECIFICATION (MUST be identical in every view that shows a window):
- {w.description}
- DO NOT vary window shape, size, or framing between views.

WINDOW PLACEMENT — fixed across all views:
{_indent(w.placement_rules)}

CANONICAL DOOR SPECIFICATION:
- Exterior door: {d.exterior}
- Interior door(s): {d.interior}

CAMERA RIG (identical for every view):
- Position: fixed at one spot near the geometric center of the space.
- Height: standing eye-level, exactly {c.height_meters} m off the floor.
- Frame: square, aspect ratio {c.aspect_ratio}.
- FOV: {c.fov_horizontal_degrees}° horizontal × {c.fov_vertical_degrees}° vertical.
- Projection: rectilinear (no fisheye, no barrel distortion).
- Horizon: dead center vertically.
{"- *** CRITICAL — FLAT-ON / PERPENDICULAR ORIENTATION ***" if c.flat_on_required else ""}
{_flat_on_block() if c.flat_on_required else ""}{(_indent(c.extra_notes) + chr(10)) if c.extra_notes else ""}
WORLD-SPACE LIGHTING (fixed — does NOT rotate with the camera):
- KEY LIGHT: {lg.key_light}
- FILL: {lg.fill_light}
- DARKEST ZONE: {lg.darkest_zone}
{lg.fixed_world_space_description}

ARCHITECTURAL CONTINUITY (CRITICAL):
- Floor pattern, ceiling, wall paneling, and horizon line MUST match exactly across every view's edges.
- Repeated elements (windows, doors) must be identical across all views in which they appear.

The reference image attached shows the art style, palette, materials, and brushwork to match. Match the reference's visual treatment exactly. If the reference is at a slight angle, your renders must flatten that perspective into perfectly perpendicular flat-on views as described above.
""".strip()


def _flat_on_block() -> str:
    return """  The camera's optical axis is EXACTLY perpendicular to the wall it is facing. Zero pitch (no tilt up or down). Zero roll (no tilt sideways). Zero yaw deviation. The camera looks STRAIGHT AT the wall, head-on.
  Visual consequences of being perfectly flat-on:
  - All vertical lines in the scene appear PERFECTLY VERTICAL — not converging, not tilted.
  - All horizontal lines appear PERFECTLY HORIZONTAL — not tilted, not converging.
  - The wall directly faced fills the frame symmetrically — its left edge and right edge are equidistant from the image center.
  - The horizon line passes through the exact vertical center of the image.
  - There is NO three-quarter angle, NO oblique perspective on the faced wall, NO casual-snapshot tilt.
  Think architectural elevation drawing or a perfectly aligned product shot, not a candid photo."""


def _indent(text: str, prefix: str = "  ") -> str:
    if not text:
        return ""
    return "\n".join(prefix + line for line in text.splitlines())


# ---------------------------------------------------------------------------
# Cardinal task prompt
# ---------------------------------------------------------------------------

def cardinal_task_prompt(
    brief: SceneBrief,
    cardinal: CardinalView,
    is_reference: bool,
    has_extra_references: bool,
) -> str:
    """
    Generate the task prompt for one cardinal view.

    is_reference: this is the cardinal that matches the input image. The
    prompt frames it as "flatten the reference."

    has_extra_references: true for cardinals that get the front+back outputs
    as additional inputs (e.g. left/right walls in a 4-cardinal panorama).
    """
    angle = cardinal.angle
    label = cardinal.label

    if is_reference:
        framing = (
            f"Render this view as a perfectly perpendicular flat-on architectural "
            f"elevation. The attached reference image shows the SAME ENVIRONMENT, "
            f"ART STYLE, COLOR PALETTE, AND CONTENT — but it may have been shot at "
            f"a slight oblique angle. Your job is to RE-RENDER the same scene with "
            f"the same elements as if the camera were standing in the geometric "
            f"center of the space, pointed STRAIGHT AT this wall, with zero rotation. "
            f"Match the reference's art style, brushwork, colors, lighting, and "
            f"material treatment EXACTLY. Preserve every element visible in the "
            f"reference — DO NOT drop or simplify any furniture, appliance, or "
            f"fixture. Just correct the camera angle."
        )
    elif has_extra_references:
        framing = (
            f"Camera pivoted to {angle}°, same fixed standing position. You are "
            f"given multiple references: the original illustration, plus the "
            f"newly-generated flat-on views of neighboring walls. Use those "
            f"neighbor views' edge content to anchor THIS view's left and right "
            f"edges. The view must be perfectly perpendicular to the wall — "
            f"all vertical lines vertical, all horizontal lines horizontal, "
            f"the wall filling the frame symmetrically, horizon dead-center."
        )
    else:
        framing = (
            f"Camera pivoted to {angle}°, same fixed standing position. The view "
            f"must be perfectly perpendicular to this wall — flat-on, no tilt, "
            f"horizon dead-center. Match the art style of the attached reference."
        )

    return f"""TASK — {label} — FLAT-ON:

{framing}

THE WALL — top to bottom, left to right (do not omit anything):

{cardinal.wall_inventory}

LEFT EDGE OF FRAME — the adjacent wall enters at a clean 90° corner:
{_indent(cardinal.left_edge_content) if cardinal.left_edge_content else "  (continuity with the counter-clockwise neighbor view)"}

RIGHT EDGE OF FRAME — the adjacent wall enters at a clean 90° corner:
{_indent(cardinal.right_edge_content) if cardinal.right_edge_content else "  (continuity with the clockwise neighbor view)"}

CAMERA — flat-on / perpendicular:
- All vertical lines (paneling seams, window frames, door edges, corner seams) PERFECTLY VERTICAL.
- All horizontal lines (counter, beam, floor seam) PERFECTLY HORIZONTAL.
- The wall fills the center of the frame symmetrically.
- Horizon dead-center. No tilt, no three-quarter angle, no recesses at corners.
""".strip()


# ---------------------------------------------------------------------------
# Interstitial task prompt
# ---------------------------------------------------------------------------

def interstitial_task_prompt(
    brief: SceneBrief,
    interstitial: InterstitialView,
) -> str:
    angle = interstitial.angle
    label = interstitial.label
    ln = brief.cardinal_by_key(interstitial.left_neighbor_key)
    rn = brief.cardinal_by_key(interstitial.right_neighbor_key)

    return f"""TASK — {label}:

A NEW camera angle exactly halfway between {ln.label} (Reference A, {ln.angle}°, attached) and {rn.label} (Reference B, {rn.angle}°, attached). Same fixed standing position, square frame, rectilinear FOV.

CRITICAL ARCHITECTURAL RULE: The corner shown is a SIMPLE 90° INTERIOR CORNER where two flat walls meet. Both walls are FLAT vertical surfaces meeting at a clean perpendicular vertical seam. NO recess, NO alcove, NO closet, NO bump-out, NO setback, NO partition wall, NO door at the corner. Do NOT make either wall step backward before continuing — the corner is just two flat planes touching. The space is NARROW — the camera is close to the adjacent wall.

This is NOT a blend of two pictures — render as if photographing the actual 3D space from a new camera rotation.

Frame content:

LEFT HALF: {interstitial.left_half_content or "(continuation of Reference A's right edge)"}

CENTER (image center): {interstitial.corner_description or "the clean vertical CORNER seam where the two flat walls meet at 90°. Just a vertical line where two flat wall surfaces meet."}

RIGHT HALF: {interstitial.right_half_content or "(continuation of Reference B's left edge)"}

LEFT edge of this image must align with the right edge of Reference A. RIGHT edge must align with the left edge of Reference B. All architectural lines (floor seam, ceiling, horizon, wall paneling) must seam continuously to BOTH references.
""".strip()
