"""
SceneBrief — the intermediate representation between vision and build.

The brief describes the space, the camera rig, and per-view content with
enough detail that prompt synthesis is a deterministic template fill.

Designed to be edited by hand in JSON between analyze and build.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def cardinal_angles(n_cardinals: int) -> list[int]:
    """N cardinals evenly spaced around the circle starting at 0°."""
    if n_cardinals < 2 or 360 % n_cardinals != 0:
        raise ValueError(
            f"n_cardinals must divide 360 (got {n_cardinals}); "
            "use 2, 3, 4, 5, 6, 8, 9, 10, 12, ..."
        )
    step = 360 // n_cardinals
    return [i * step for i in range(n_cardinals)]


def interstitial_angles(cardinals: list[int]) -> list[int]:
    """Halfway points between consecutive cardinals (closing the loop)."""
    out = []
    for i, a in enumerate(cardinals):
        b = cardinals[(i + 1) % len(cardinals)]
        # midpoint, accounting for wrap-around at 360
        mid = (a + b) // 2 if b > a else (a + (b + 360)) // 2 % 360
        out.append(mid)
    return out


# ---------------------------------------------------------------------------
# Brief fields
# ---------------------------------------------------------------------------

@dataclass
class StyleSpec:
    art_style: str = ""              # "stylized 3D render, hand-painted illustrative"
    palette: str = ""                # "warm browns, olive-sage greens, mint accents"
    materials: str = ""              # "vertical wood paneling, linoleum floor"
    atmosphere: str = ""             # "lived-in, melancholic moving day"
    rendering_notes: str = ""        # "painterly brushwork visible on wood and fabric"


@dataclass
class SpaceSpec:
    type_label: str = ""             # "single-wide vintage trailer interior"
    dimensions: str = ""             # "12 ft × 30 ft × 7 ft ceiling"
    geometry_rules: str = ""         # "walls flat, 90° corners, no recesses..."
    floor: str = ""                  # "olive-green linoleum with confetti speckles"
    ceiling: str = ""                # "low wood-plank with one exposed beam across the width"
    walls: str = ""                  # "vertical wood paneling on all surfaces"


@dataclass
class WindowSpec:
    description: str = ""            # "rounded-top / squared-bottom, ~3×2.5 ft, frosted"
    placement_rules: str = ""        # "left wall: window-door-window; right wall: 3 windows..."


@dataclass
class DoorSpec:
    exterior: str = ""               # "left wall, full-height, rounded top..."
    interior: str = ""               # "back wall, plain rectangular bedroom door..."


@dataclass
class LightingSpec:
    key_light: str = ""              # "warm tulip pendant at kitchen end (0°)"
    fill_light: str = ""             # "cool overcast daylight from side windows"
    darkest_zone: str = ""           # "TV/back end (180°), only daylight spill"
    fixed_world_space_description: str = ""  # the canonical "lighting is fixed in world space" paragraph


@dataclass
class CameraSpec:
    height_meters: float = 1.6
    fov_horizontal_degrees: int = 90
    fov_vertical_degrees: int = 90
    aspect_ratio: str = "1:1"
    flat_on_required: bool = True
    extra_notes: str = ""


@dataclass
class CardinalView:
    """One cardinal view of the panorama (a wall faced flat-on)."""
    key: str                          # "C1", "C2", ...
    angle: int                        # 0, 90, 180, 270, etc.
    label: str                        # "FRONT (0°) — Kitchen end"
    wall_role: str                    # "kitchen short wall" | "right long wall" | etc.
    wall_inventory: str               # full top-to-bottom description (the big payload)
    left_edge_content: str            # what the left edge of frame shows (entering side wall)
    right_edge_content: str           # what the right edge of frame shows
    custom_notes: str = ""            # user-editable secondary direction


@dataclass
class InterstitialView:
    """A corner view, halfway between two cardinals."""
    key: str                          # "I1", "I2", ...
    angle: int                        # 45, 135, ...
    label: str                        # "FRONT-RIGHT (45°) — Kitchen→Right corner"
    left_neighbor_key: str            # cardinal feeding the left half
    right_neighbor_key: str           # cardinal feeding the right half
    left_half_content: str            # what's on the left half of frame
    right_half_content: str           # what's on the right half of frame
    corner_description: str           # what the architectural corner looks like
    custom_notes: str = ""


@dataclass
class SceneBrief:
    style: StyleSpec = field(default_factory=StyleSpec)
    space: SpaceSpec = field(default_factory=SpaceSpec)
    windows: WindowSpec = field(default_factory=WindowSpec)
    doors: DoorSpec = field(default_factory=DoorSpec)
    lighting: LightingSpec = field(default_factory=LightingSpec)
    camera: CameraSpec = field(default_factory=CameraSpec)
    cardinals: list[CardinalView] = field(default_factory=list)
    interstitials: list[InterstitialView] = field(default_factory=list)

    # Reference view: which cardinal corresponds to the input image?
    # Default 0 (the first/front cardinal); can be overridden if the
    # input image isn't a head-on view.
    reference_cardinal_key: str = "C1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SceneBrief":
        return cls(
            style=StyleSpec(**data.get("style", {})),
            space=SpaceSpec(**data.get("space", {})),
            windows=WindowSpec(**data.get("windows", {})),
            doors=DoorSpec(**data.get("doors", {})),
            lighting=LightingSpec(**data.get("lighting", {})),
            camera=CameraSpec(**data.get("camera", {})),
            cardinals=[CardinalView(**c) for c in data.get("cardinals", [])],
            interstitials=[InterstitialView(**i) for i in data.get("interstitials", [])],
            reference_cardinal_key=data.get("reference_cardinal_key", "C1"),
        )

    # ---- convenience lookups ----

    def cardinal_by_key(self, key: str) -> CardinalView:
        for c in self.cardinals:
            if c.key == key:
                return c
        raise KeyError(f"no cardinal with key {key}")
