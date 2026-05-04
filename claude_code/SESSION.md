# Pano-agent — Claude Code session guide

You are helping a designer build a ComfyUI panorama-generation workflow from a single reference illustration of an interior space. The user will share an image; your job is to analyze it, write a `scene_brief.json` describing the space, then assemble that into a ComfyUI workflow JSON via a build script.

## The workflow

```
reference.png   ──  YOU analyze   ──→  scene_brief.json
                                         │
                                         ├── user reviews + edits
                                         │
                                         ▼
                              YOU run build_workflow.py
                                         │
                                         ▼
                                   workflow.json   ──→  load in ComfyUI
```

## What you do at each step

### Step 1: Analyze the image

When the user shares a reference image, look at it carefully and produce a `scene_brief.json` with this shape (see `schema/scene_brief.schema.json` for the formal structure):

```jsonc
{
  "style": {
    "art_style": "...",                    // e.g. "stylized 3D render, hand-painted illustrative"
    "palette": "...",
    "materials": "...",
    "atmosphere": "...",
    "rendering_notes": "..."
  },
  "space": {
    "type_label": "...",                   // e.g. "single-wide vintage trailer interior"
    "dimensions": "...",                   // best estimate, e.g. "12 ft × 30 ft × 7 ft ceiling"
    "geometry_rules": "...",               // strict rules: flat walls, 90° corners, no recesses
    "floor": "...",
    "ceiling": "...",
    "walls": "..."
  },
  "windows": {
    "description": "...",                  // ONE canonical shape/size — all windows match this
    "placement_rules": "..."               // per-wall counts and order
  },
  "doors": {
    "exterior": "...",
    "interior": "..."
  },
  "lighting": {
    "key_light": "...",
    "fill_light": "...",
    "darkest_zone": "...",
    "fixed_world_space_description": "..."
  },
  "camera": {
    "height_meters": 1.6,
    "fov_horizontal_degrees": 90,
    "fov_vertical_degrees": 90,
    "aspect_ratio": "1:1",
    "flat_on_required": true
  },
  "cardinals": [
    {
      "key": "C1", "angle": 0,
      "label": "View 1 — FRONT (0°) — [wall name]",
      "wall_role": "...",
      "wall_inventory": "UPPER ZONE: ...\nMID ZONE: ...\nLOWER ZONE: ...",
      "left_edge_content": "...",
      "right_edge_content": "..."
    }
    // C2 (90°), C3 (180°), C4 (270°) for an 8-view panorama
  ],
  "interstitials": [
    {
      "key": "I1", "angle": 45,
      "label": "View 1.5 — FRONT-RIGHT (45°) — [corner name]",
      "left_neighbor_key": "C1",
      "right_neighbor_key": "C2",
      "left_half_content": "...",
      "right_half_content": "...",
      "corner_description": "..."
    }
    // I2 (135°), I3 (225°), I4 (315°)
  ],
  "reference_cardinal_key": "C1"           // which cardinal matches the reference image
}
```

### Critical rules for the brief

These rules emerged from extensive iteration. Following them is how you avoid the common failure modes.

1. **Wall inventory format — exhaustive, top to bottom.** For each cardinal's `wall_inventory`, describe the wall as horizontal zones:

   ```
   UPPER ZONE (top of wall to mid-wall): [list every visible item, left to right]
   MID ZONE (counter / fixture level):   [list every item]
   LOWER ZONE (mid-wall to floor):       [list every item]
   ```

   The most common failure is dropping fixtures (cabinets, fridges, beams). If a wall has upper cabinets, you MUST say "brown wooden upper cabinets fill most of the upper zone" — leaving the upper zone vague results in the image generator filling it with empty paneling.

2. **Canonical window spec.** All windows in the space MUST have one shared description. Don't say "windows are arched" for one view and "windows are rectangular" for another. Pick one ("rounded-top / squared-bottom rectangle, ~3 ft × 2.5 ft, frosted glass") and use it for every window across every view.

3. **Window placement is per-wall.** State exactly how many windows are on each wall and in what order. e.g. "Left long wall: 2 windows + 1 exterior door, in order WINDOW→DOOR→WINDOW from front to back."

4. **No-recess rule for corners.** Every interstitial's `corner_description` must explicitly state that the corner is a clean 90° meeting of two flat walls — no closets, alcoves, recesses, bump-outs, or setbacks. Image models default to inventing depth at corners; you have to forbid it.

5. **World-space lighting.** Describe the dominant light source as fixed in the world (e.g. "warm pendant at the kitchen end"), not relative to the camera. This keeps the panorama internally consistent.

6. **Continuity anchors.** Items that bridge views must be referenced in every view they appear in. If the reference image shows boxes at the front-left, those same boxes appear at the right-edge of the 270° cardinal AND across the entire 315° interstitial — name them ("boxes labeled BOOKS / KITCHEN / STUFF") in all three so the model treats them as the same objects.

7. **Edge content describes seams.** `left_edge_content` and `right_edge_content` for cardinals describe the corner where this wall meets the neighbor wall, plus the partial sliver of the neighbor wall visible at the frame's edge. These define how views seam together.

8. **The reference cardinal** is the cardinal that best matches the reference image. Default `C1` (0°). The reference image is treated as that cardinal during generation, and prompts for that cardinal will say "flatten this reference."

### How the user might guide you

The user might:

- Hand you the image with no other context. Analyze it, infer the geometry, write the full brief.
- Tell you the space dimensions or wall layout ("it's a 12×30 trailer, two windows on the left wall plus a door, three windows on the right"). Use their description verbatim for `placement_rules` and infer the rest.
- Ask you to revise specific cardinals after seeing the build script's output. Edit just those cardinals in the brief.

### Multi-cardinal panoramas

The default is 4 cardinals + 4 interstitials = 8 total views. The user can request other configurations:

- 2 cardinals (180°) — minimal, no side walls
- 4 cardinals (90° each) — default
- 6 cardinals (60° each) — denser, more overlap
- 8 cardinals (45° each) — densest

Cardinals must divide 360° evenly. Generate cardinal angles as `[0, 360/N, 2·360/N, ...]` and interstitial angles as midpoints between consecutive cardinals.

## Step 2: Build the workflow

Once the user has reviewed/edited `scene_brief.json`, run:

```bash
python build.py scene_brief.json -o workflow.json
```

If the reference image isn't `reference.png`:

```bash
python build.py scene_brief.json -o workflow.json --reference-filename my_image.png
```

The build script is deterministic — same brief always produces the same workflow. It loads the brief, generates all the prompt strings using internal templates that encode the rules above, and emits a ComfyUI JSON with all nodes, links, batch wiring, and execution order set up correctly.

You don't need to hand-write any prompts. The build script handles that. Your job is just to produce a faithful, complete `scene_brief.json`.

## What you do NOT do

- Don't generate any images yourself. The user will run the resulting workflow in ComfyUI.
- Don't write the prompt strings into the brief. The build script generates them from the structured fields.
- Don't include any panagent-specific JSON fields not in the schema. Stay strict.

## When the user shares output images

If the user runs the workflow and shares a generated image saying "this view came out wrong":

1. Look at it carefully against the brief.
2. Diagnose what's wrong: is a fixture missing? Wrong window shape? False architectural recess? Camera not flat-on?
3. Identify which brief field needs editing to fix it.
4. Edit the brief and propose re-running the build.

This diagnostic loop is the main reason for using the Claude Code pipeline over the standalone CLI — you can iterate naturally on the brief based on what you see in outputs.

## File layout

In your working directory you should have:

- `build.py` — the build script (do not modify, it's the same code as the CLI's build pass)
- `pano_agent/` — the shared library it imports from
- `scene_brief.json` — what you produce in step 1
- `workflow.json` — what the build script produces in step 2
- `reference.png` (or whatever name) — the input image
- `outputs/` — generated images go here (ComfyUI default)
