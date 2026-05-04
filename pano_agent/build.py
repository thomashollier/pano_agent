"""
Workflow builder: SceneBrief → ComfyUI workflow JSON.

Generalized to any even N ≥ 4 cardinals. Layout is a left-to-right flowchart:

  COL 1 SHARED       — reference image, global prompt
  COL 2 CARDINAL     — task + custom prompt pairs, one row per cardinal
  COL 3 INTERSTITIAL — task + custom prompt pairs, one row per interstitial
  COL 4 CONCAT       — text concatenators (one per view)
  COL 5 BATCH+GEMINI — batch nodes (where needed) + gemini generation nodes
  COL 6 SAVE         — SaveImage nodes

Generation order (enforced by data dependencies):
  - The "reference cardinal" (the one matching the input image) generates first
    using only the reference image as input.
  - The "opposite cardinal" (180° from reference) generates next, also from
    the reference image only. This anchors the back wall.
  - Remaining cardinals receive [reference + reference-cardinal output +
    opposite-cardinal output] via batch nodes — they execute after both
    anchors are done.
  - Interstitials receive [reference + their two neighbor cardinals] via
    batch nodes — they execute after all cardinals are done.

Skipping the "extra references" step for non-reference / non-opposite cardinals
when N=2 (only two cardinals exist; nothing to batch). Otherwise it's always on.
"""
from __future__ import annotations

import json
from typing import Any

from .brief import SceneBrief
from .prompts import (
    cardinal_task_prompt,
    global_prompt,
    interstitial_task_prompt,
)


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

COL_REF = -4400
COL_CARDINAL = -3700
COL_INTERST = -2900
COL_CONCAT = -2100
COL_BATCH = -1600
COL_GEMINI = -1100
COL_SAVE = -400

PROMPT_W = 600
PROMPT_H = 320
PROMPT_GAP_V = 40
ROW_HEIGHT = 800

CARDINAL_BAND_TOP = 400
INTERSTITIAL_BAND_GAP = 400  # gap between cardinal band and interstitial band


# Default Nano Banana system instruction (matches user's existing workflow)
DEFAULT_GEMINI_SYSTEM = """You are an expert image-generation engine. You must ALWAYS produce an image.
Interpret all user input—regardless of format, intent, or abstraction—as literal visual directives for image composition.
If a prompt is conversational or lacks specific visual details, you must creatively invent a concrete visual scenario that depicts the concept.
Prioritize generating the visual representation above any text, formatting, or conversational requests."""

GEMINI_MODEL = "gemini-3-pro-image-preview"


# ---------------------------------------------------------------------------
# Workflow assembler
# ---------------------------------------------------------------------------

class _Builder:
    def __init__(self):
        self.nodes: list[dict[str, Any]] = []
        self.links: list[list] = []
        self._next_node_id = 0
        self._next_link_id = 0

    def node_id(self) -> int:
        self._next_node_id += 1
        return self._next_node_id

    def link_id(self) -> int:
        self._next_link_id += 1
        return self._next_link_id

    def add_link(self, src_id: int, src_slot: int, dst_id: int, dst_slot: int, type_: str) -> int:
        lid = self.link_id()
        self.links.append([lid, src_id, src_slot, dst_id, dst_slot, type_])
        # Patch source's outputs.links list
        src = next(n for n in self.nodes if n["id"] == src_id)
        out = src["outputs"][src_slot]
        if out.get("links") is None:
            out["links"] = []
        out["links"].append(lid)
        # Patch dest's inputs.link
        dst = next(n for n in self.nodes if n["id"] == dst_id)
        dst["inputs"][dst_slot]["link"] = lid
        return lid

    # ---- node templates ----

    def make_load_image(self, title: str, pos: tuple[int, int], filename: str) -> dict:
        n = {
            "id": self.node_id(),
            "type": "LoadImage",
            "pos": list(pos),
            "size": [620, 600],
            "flags": {}, "order": 0, "mode": 0,
            "inputs": [],
            "outputs": [
                {"name": "IMAGE", "type": "IMAGE", "links": []},
                {"name": "MASK", "type": "MASK", "links": None},
            ],
            "title": title,
            "properties": {
                "Node name for S&R": "LoadImage",
                "cnr_id": "comfy-core",
                "ver": "0.3.52",
            },
            "widgets_values": [filename, "image"],
        }
        self.nodes.append(n)
        return n

    def make_string_node(self, title: str, pos: tuple[int, int], text: str,
                         size: tuple[int, int] = (PROMPT_W, PROMPT_H)) -> dict:
        n = {
            "id": self.node_id(),
            "type": "PrimitiveStringMultiline",
            "pos": list(pos),
            "size": list(size),
            "flags": {}, "order": 0, "mode": 0,
            "inputs": [],
            "outputs": [{"name": "STRING", "type": "STRING", "links": []}],
            "title": title,
            "properties": {"Node name for S&R": "PrimitiveStringMultiline"},
            "widgets_values": [text],
        }
        self.nodes.append(n)
        return n

    def make_concat(self, title: str, pos: tuple[int, int]) -> dict:
        n = {
            "id": self.node_id(),
            "type": "Text Concatenate",
            "pos": list(pos),
            "size": [330, 240],
            "flags": {}, "order": 0, "mode": 0,
            "inputs": [
                {"name": "text_a", "shape": 7, "type": "STRING", "link": None},
                {"name": "text_b", "shape": 7, "type": "STRING", "link": None},
                {"name": "text_c", "shape": 7, "type": "STRING", "link": None},
                {"name": "text_d", "shape": 7, "type": "STRING", "link": None},
            ],
            "outputs": [{"name": "STRING", "type": "STRING", "links": []}],
            "title": title,
            "properties": {"Node name for S&R": "Text Concatenate"},
            "widgets_values": ["\n\n", "true"],
        }
        self.nodes.append(n)
        return n

    def make_batch(self, title: str, pos: tuple[int, int]) -> dict:
        n = {
            "id": self.node_id(),
            "type": "BatchImagesNode",
            "pos": list(pos),
            "size": [260, 140],
            "flags": {}, "order": 0, "mode": 0,
            "inputs": [
                {"label": "image0", "name": "images.image0", "type": "IMAGE", "link": None},
                {"label": "image1", "name": "images.image1", "type": "IMAGE", "link": None},
                {"label": "image2", "name": "images.image2", "shape": 7, "type": "IMAGE", "link": None},
                {"label": "image3", "name": "images.image3", "shape": 7, "type": "IMAGE", "link": None},
            ],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": []}],
            "title": title,
            "properties": {"Node name for S&R": "BatchImagesNode"},
            "widgets_values": [],
        }
        self.nodes.append(n)
        return n

    def make_gemini(self, title: str, pos: tuple[int, int], seed: int) -> dict:
        n = {
            "id": self.node_id(),
            "type": "GeminiImage2Node",
            "pos": list(pos),
            "size": [460, 600],
            "flags": {}, "order": 0, "mode": 0,
            "showAdvanced": False,
            "inputs": [
                {"name": "images", "shape": 7, "type": "IMAGE", "link": None},
                {"name": "files", "shape": 7, "type": "GEMINI_INPUT_FILES", "link": None},
                {"name": "prompt", "type": "STRING", "widget": {"name": "prompt"}, "link": None},
            ],
            "outputs": [
                {"name": "IMAGE", "type": "IMAGE", "links": []},
                {"name": "STRING", "type": "STRING", "links": []},
            ],
            "title": title,
            "properties": {"Node name for S&R": "GeminiImage2Node"},
            "widgets_values": [
                "",  # widget prompt (overridden by input link)
                GEMINI_MODEL,
                seed,
                "randomize",
                "1:1",
                "1K",
                "IMAGE",
                DEFAULT_GEMINI_SYSTEM,
            ],
            "color": "#432",
            "bgcolor": "#653",
        }
        self.nodes.append(n)
        return n

    def make_save(self, title: str, pos: tuple[int, int], filename_prefix: str) -> dict:
        n = {
            "id": self.node_id(),
            "type": "SaveImage",
            "pos": list(pos),
            "size": [500, 600],
            "flags": {}, "order": 0, "mode": 0,
            "inputs": [{"name": "images", "type": "IMAGE", "link": None}],
            "outputs": [],
            "title": title,
            "properties": {"cnr_id": "comfy-core", "ver": "0.3.56"},
            "widgets_values": [filename_prefix],
        }
        self.nodes.append(n)
        return n


# ---------------------------------------------------------------------------
# Topological order (used to compute node 'order' field)
# ---------------------------------------------------------------------------

def _compute_order(nodes: list[dict], links: list[list]) -> dict[int, int]:
    from collections import defaultdict, deque
    in_deg: dict[int, int] = defaultdict(int)
    adj: dict[int, list[int]] = defaultdict(list)
    node_ids = {n["id"] for n in nodes}
    for link in links:
        _, src, _, dst, _, _ = link
        if src in node_ids and dst in node_ids:
            adj[src].append(dst)
            in_deg[dst] += 1

    queue = deque(sorted(n["id"] for n in nodes if in_deg[n["id"]] == 0))
    order_map: dict[int, int] = {}
    i = 0
    while queue:
        nid = queue.popleft()
        order_map[nid] = i
        i += 1
        for nxt in sorted(adj[nid]):
            in_deg[nxt] -= 1
            if in_deg[nxt] == 0:
                queue.append(nxt)
    return order_map


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_workflow(brief: SceneBrief, reference_filename: str = "reference.png") -> dict:
    b = _Builder()
    n_cards = len(brief.cardinals)
    n_inters = len(brief.interstitials)

    # Identify the reference cardinal (the one matching the input image)
    # and its opposite (the one 180° away).
    ref_cardinal = brief.cardinal_by_key(brief.reference_cardinal_key)
    opposite_angle = (ref_cardinal.angle + 180) % 360
    opposite_cardinal = next(
        (c for c in brief.cardinals if c.angle == opposite_angle), None
    )

    # ---- Column 1: shared (reference image + global prompt) ----
    ref_node = b.make_load_image(
        title=f"REFERENCE IMAGE — original illustration (= {ref_cardinal.label})",
        pos=(COL_REF, -300),
        filename=reference_filename,
    )
    global_node = b.make_string_node(
        title="GLOBAL PROMPT — style, space, lighting, continuity (shared by ALL views)",
        pos=(COL_REF, 350),
        text=global_prompt(brief),
        size=(620, 700),
    )

    # Track node refs for later wiring
    cardinal_records = []  # (cardinal, task_node, custom_node, concat_node, batch_node, gem_node, save_node)

    # ---- Column 2 + downstream: cardinals ----
    for i, cardinal in enumerate(brief.cardinals):
        row_y = CARDINAL_BAND_TOP + i * ROW_HEIGHT
        is_ref = (cardinal.key == brief.reference_cardinal_key)
        is_opposite = (opposite_cardinal is not None and cardinal.key == opposite_cardinal.key)
        # "extra references" cardinals get [ref + ref-cardinal output + opposite-cardinal output]
        # as additional inputs. Skip when there are only 2 cardinals (nothing to batch beyond ref).
        has_extras = (not is_ref and not is_opposite and n_cards >= 4
                      and opposite_cardinal is not None)

        # Task prompt
        task_node = b.make_string_node(
            title=f"[{cardinal.key}] TASK — {cardinal.label}",
            pos=(COL_CARDINAL, row_y),
            text=cardinal_task_prompt(brief, cardinal, is_ref, has_extras),
        )
        # Custom notes
        custom_node = b.make_string_node(
            title=f"[{cardinal.key}] YOUR NOTES — {cardinal.label}",
            pos=(COL_CARDINAL, row_y + PROMPT_H + PROMPT_GAP_V),
            text=cardinal.custom_notes or "[Add scene-specific direction or corrections here.]",
        )
        # Concat
        concat_node = b.make_concat(
            title=f"[{cardinal.key}] Concat → prompt",
            pos=(COL_CONCAT, row_y + PROMPT_H // 2 - 100),
        )
        b.add_link(task_node["id"], 0, concat_node["id"], 0, "STRING")
        b.add_link(custom_node["id"], 0, concat_node["id"], 1, "STRING")
        b.add_link(global_node["id"], 0, concat_node["id"], 2, "STRING")

        # Gemini node
        gem_node = b.make_gemini(
            title=f"[{cardinal.key}] Nano Banana Pro — {cardinal.label}",
            pos=(COL_GEMINI, row_y),
            seed=42 + i,
        )
        b.add_link(concat_node["id"], 0, gem_node["id"], 2, "STRING")

        # Image input — either direct ref, or via batch
        batch_node = None
        if has_extras:
            batch_node = b.make_batch(
                title=f"[{cardinal.key}] Batch: ref + {brief.reference_cardinal_key} + {opposite_cardinal.key}",
                pos=(COL_BATCH, row_y + 200),
            )
            b.add_link(ref_node["id"], 0, batch_node["id"], 0, "IMAGE")
            # Note: the actual gemini for ref/opposite is wired later; we
            # store batch-node refs and patch them in a second pass.
            b.add_link(batch_node["id"], 0, gem_node["id"], 0, "IMAGE")
        else:
            b.add_link(ref_node["id"], 0, gem_node["id"], 0, "IMAGE")

        # Save
        save_node = b.make_save(
            title=f"[{cardinal.key}] SAVE — {cardinal.label}",
            pos=(COL_SAVE, row_y),
            filename_prefix=f"panorama/{cardinal.key}_{cardinal.angle:03d}",
        )
        b.add_link(gem_node["id"], 0, save_node["id"], 0, "IMAGE")

        cardinal_records.append({
            "cardinal": cardinal,
            "task": task_node, "custom": custom_node, "concat": concat_node,
            "batch": batch_node, "gem": gem_node, "save": save_node,
        })

    # Second pass: wire batch nodes' image1/image2 from the reference and
    # opposite cardinal Geminis (we needed all gem nodes to exist first).
    ref_gem = next(r["gem"] for r in cardinal_records if r["cardinal"].key == brief.reference_cardinal_key)
    opposite_gem = (next(r["gem"] for r in cardinal_records if r["cardinal"].key == opposite_cardinal.key)
                    if opposite_cardinal else None)

    for rec in cardinal_records:
        if rec["batch"] is not None:
            b.add_link(ref_gem["id"], 0, rec["batch"]["id"], 1, "IMAGE")
            if opposite_gem is not None:
                b.add_link(opposite_gem["id"], 0, rec["batch"]["id"], 2, "IMAGE")

    # ---- Interstitial band ----
    interstitial_top = CARDINAL_BAND_TOP + n_cards * ROW_HEIGHT + INTERSTITIAL_BAND_GAP
    cardinal_gem_by_key = {r["cardinal"].key: r["gem"] for r in cardinal_records}

    for i, inter in enumerate(brief.interstitials):
        row_y = interstitial_top + i * ROW_HEIGHT

        task_node = b.make_string_node(
            title=f"[{inter.key}] TASK — {inter.label}",
            pos=(COL_INTERST, row_y),
            text=interstitial_task_prompt(brief, inter),
        )
        custom_node = b.make_string_node(
            title=f"[{inter.key}] YOUR NOTES — {inter.label}",
            pos=(COL_INTERST, row_y + PROMPT_H + PROMPT_GAP_V),
            text=inter.custom_notes or "[Add scene-specific direction here.]",
        )
        concat_node = b.make_concat(
            title=f"[{inter.key}] Concat → prompt",
            pos=(COL_CONCAT, row_y + PROMPT_H // 2 - 100),
        )
        b.add_link(task_node["id"], 0, concat_node["id"], 0, "STRING")
        b.add_link(custom_node["id"], 0, concat_node["id"], 1, "STRING")
        b.add_link(global_node["id"], 0, concat_node["id"], 2, "STRING")

        batch_node = b.make_batch(
            title=f"[{inter.key}] Batch: ref + {inter.left_neighbor_key} + {inter.right_neighbor_key}",
            pos=(COL_BATCH, row_y + 200),
        )
        b.add_link(ref_node["id"], 0, batch_node["id"], 0, "IMAGE")
        ln_gem = cardinal_gem_by_key[inter.left_neighbor_key]
        rn_gem = cardinal_gem_by_key[inter.right_neighbor_key]
        b.add_link(ln_gem["id"], 0, batch_node["id"], 1, "IMAGE")
        b.add_link(rn_gem["id"], 0, batch_node["id"], 2, "IMAGE")

        gem_node = b.make_gemini(
            title=f"[{inter.key}] Nano Banana Pro — {inter.label}",
            pos=(COL_GEMINI, row_y),
            seed=142 + i,
        )
        b.add_link(batch_node["id"], 0, gem_node["id"], 0, "IMAGE")
        b.add_link(concat_node["id"], 0, gem_node["id"], 2, "STRING")

        save_node = b.make_save(
            title=f"[{inter.key}] SAVE — {inter.label}",
            pos=(COL_SAVE, row_y),
            filename_prefix=f"panorama/{inter.key}_{inter.angle:03d}",
        )
        b.add_link(gem_node["id"], 0, save_node["id"], 0, "IMAGE")

    # ---- Finalize ----
    order_map = _compute_order(b.nodes, b.links)
    for n in b.nodes:
        n["order"] = order_map.get(n["id"], 0)

    # Group bands
    canvas_right = COL_SAVE + 500 + 80
    cardinal_bottom = CARDINAL_BAND_TOP + n_cards * ROW_HEIGHT + 100
    interstitial_bottom = interstitial_top + n_inters * ROW_HEIGHT + 100

    groups = [
        {
            "id": 1,
            "title": "▼ COL 1: SHARED — reference + global notes",
            "bounding": [COL_REF - 80, -400, 720, 1500],
            "color": "#3f5b7e",
            "font_size": 28,
        },
        {
            "id": 2,
            "title": f"▼ COL 2: CARDINAL PROMPTS ({n_cards} views)",
            "bounding": [COL_CARDINAL - 80, CARDINAL_BAND_TOP - 100, 720, cardinal_bottom - CARDINAL_BAND_TOP + 100],
            "color": "#2d5f7b",
            "font_size": 28,
        },
        {
            "id": 3,
            "title": f"▼ COL 3: INTERSTITIAL PROMPTS ({n_inters} corners)",
            "bounding": [COL_INTERST - 80, interstitial_top - 100, 720, interstitial_bottom - interstitial_top + 100],
            "color": "#7b2d6b",
            "font_size": 28,
        },
        {
            "id": 4,
            "title": "▼ PHASE 1: Cardinal generation",
            "bounding": [COL_CONCAT - 80, CARDINAL_BAND_TOP - 100,
                         canvas_right - (COL_CONCAT - 80), cardinal_bottom - CARDINAL_BAND_TOP + 100],
            "color": "#3f789e",
            "font_size": 28,
        },
        {
            "id": 5,
            "title": "▼ PHASE 2: Interstitial generation",
            "bounding": [COL_CONCAT - 80, interstitial_top - 100,
                         canvas_right - (COL_CONCAT - 80), interstitial_bottom - interstitial_top + 100],
            "color": "#a1309b",
            "font_size": 28,
        },
    ]

    return {
        "id": "pano_agent-panorama",
        "revision": 1,
        "last_node_id": b._next_node_id,
        "last_link_id": b._next_link_id,
        "nodes": b.nodes,
        "links": b.links,
        "groups": groups,
        "config": {},
        "extra": {
            "ds": {"scale": 0.18, "offset": [4800, 600]},
            "frontendVersion": "1.42.6",
        },
        "version": 0.4,
    }
