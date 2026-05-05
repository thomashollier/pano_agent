#!/usr/bin/env python3
"""
worldgen_pano.py — perspective image → 360° equirectangular panorama on Modal GPU.

Custom pipeline (geometrically correct):
  1. DepthPro (Apple) estimates the focal length (FOV) of the perspective input
  2. Inverse equirectangular→perspective mapping projects the reference image onto
     the pano canvas using the correct spherical geometry
  3. FLUX.1-dev Fill (WorldGen) inpaints the unseen 270°+ of the scene

The key improvement over WorldGen's stock i2s pipeline: DA-2 treated the
perspective image as if it were already a panorama (SphereViT), misplacing
pixels. Here, DepthPro gives us the real FOV so the projection is correct.

Setup (one-time):
    pip install modal
    modal setup
    modal secret create huggingface HF_TOKEN=hf_yourtoken

Usage:
    modal run worldgen_pano.py --image input.jpg
    modal run worldgen_pano.py --image input.jpg --prompt "a cozy vintage interior"
    modal run worldgen_pano.py --image input.jpg --output pano.png --high-vram
"""

import io
import sys
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal app & container image
# ---------------------------------------------------------------------------

app = modal.App("worldgen-pano")

hf_cache_vol = modal.Volume.from_name("worldgen-hf-cache", create_if_missing=True)

worldgen_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "build-essential", "gcc", "g++", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.7.0",
        "torchvision==0.22.0",
        index_url="https://download.pytorch.org/whl/cu126",
    )
    .run_commands(
        # WorldGen: provides FLUX.1-dev Fill + the WorldGen panoramic LoRA weights.
        # Skip ml-sharp submodule (SSH-only, not needed for inpainting).
        "git clone --depth 1 --no-recurse-submodules https://github.com/ZiYang-xie/WorldGen.git /opt/worldgen",
        "pip install /opt/worldgen",
        # DepthPro: metric depth + focal-length (FOV) estimation from a perspective image.
        # Checkpoint (~2 GB) is downloaded on first run and stored in the HF cache volume.
        "pip install git+https://github.com/apple/ml-depth-pro.git",
    )
    .env({"HF_HOME": "/hf_cache"})
)


# ---------------------------------------------------------------------------
# Remote GPU function
# ---------------------------------------------------------------------------

@app.function(
    image=worldgen_image,
    gpu="A100",
    timeout=600,
    volumes={"/hf_cache": hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def generate_pano(
    image_bytes: bytes,
    prompt: str = "",
    low_vram: bool = True,
    pano_h: int = 512,
    pano_w: int = 1024,
) -> bytes:
    """
    Perspective image → 360° equirectangular panorama (PNG bytes).

    Step 1 — DepthPro estimates the camera focal length (FOV).
    Step 2 — Inverse equirectangular projection: for every pano pixel we compute
              its 3D ray on the sphere, project back into the perspective image,
              and sample with bilinear interpolation. Pixels outside the camera's
              view stay black and are marked for FLUX to generate.
    Step 3 — FLUX.1-dev Fill (WorldGen) inpaints the unseen regions.
    """
    import sys
    import tempfile
    import os
    import numpy as np
    import cv2
    import torch
    from PIL import Image
    from unittest.mock import MagicMock

    # WorldGen's __init__ imports pano_depth.py which imports da2 (SphereViT)
    # at the top level. Stub both da2 and pytorch3d so worldgen imports cleanly.
    _da2 = MagicMock()
    sys.modules.setdefault("da2", _da2)
    sys.modules.setdefault("da2.model", _da2)
    sys.modules.setdefault("da2.model.spherevit", _da2)
    _p3d = MagicMock()
    sys.modules.setdefault("pytorch3d", _p3d)
    sys.modules.setdefault("pytorch3d.transforms", _p3d)

    device = torch.device("cuda")
    input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    W_in, H_in = input_image.size
    print(f"Input: {W_in}×{H_in}  prompt={prompt!r}  low_vram={low_vram}")

    # -------------------------------------------------------------------------
    # Step 1: DepthPro — estimate focal length
    # -------------------------------------------------------------------------
    print("Running DepthPro (depth + FOV estimation)...")

    import depth_pro
    from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
    from huggingface_hub import hf_hub_download

    # Download checkpoint to the mounted HF cache volume on first run.
    checkpoint_path = hf_hub_download(repo_id="apple/DepthPro", filename="depth_pro.pt")
    DEFAULT_MONODEPTH_CONFIG_DICT.checkpoint_uri = checkpoint_path

    dp_model, dp_transform = depth_pro.create_model_and_transforms(
        config=DEFAULT_MONODEPTH_CONFIG_DICT, device=device
    )
    dp_model.eval()

    # DepthPro's load_rgb expects a file path; write image to a temp file.
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        input_image.save(tmp_path)

    img_for_dp, _, _ = depth_pro.load_rgb(tmp_path)
    os.unlink(tmp_path)

    with torch.no_grad():
        prediction = dp_model.infer(dp_transform(img_for_dp), f_px=None)

    focal_px = float(prediction["focallength_px"])
    print(f"DepthPro: focal_length={focal_px:.1f}px  "
          f"(hFOV={2 * np.degrees(np.arctan(W_in / (2 * focal_px))):.1f}°)")

    del dp_model
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Step 2: Perspective → equirectangular projection (inverse mapping)
    # -------------------------------------------------------------------------
    # For each pano pixel we compute the 3D ray on the unit sphere, then project
    # it back into the perspective image.  This guarantees a dense, gap-free
    # canvas with correct geometry.
    #
    # Coordinate system (right-hand, camera looking along +Z):
    #   X = right,  Y = up,  Z = forward (into scene)
    #
    # Equirectangular conventions:
    #   u ∈ [0, pano_w]  →  azimuth ∈ [-π, π]   (0 = center = camera forward)
    #   v ∈ [0, pano_h]  →  elevation ∈ [π/2, -π/2]  (0 = top = zenith)
    print("Projecting perspective image onto equirectangular canvas...")

    img_np = np.array(input_image)  # (H_in, W_in, 3)
    cx = W_in / 2.0
    cy = H_in / 2.0

    # Grid of pano pixel indices
    pano_v, pano_u = np.mgrid[0:pano_h, 0:pano_w].astype(np.float32)

    azimuth  = (pano_u / pano_w - 0.5) * 2.0 * np.pi  # [-π, +π]
    elevation = (0.5 - pano_v / pano_h) * np.pi        # [+π/2, -π/2]

    # Unit sphere directions
    Xe = np.sin(azimuth)  * np.cos(elevation)   # right
    Ye = np.sin(elevation)                        # up
    Ze = np.cos(azimuth)  * np.cos(elevation)    # forward

    # Perspective projection  (valid only for Ze > 0 = in front of camera)
    eps = 1e-6
    valid = Ze > eps
    with np.errstate(divide="ignore", invalid="ignore"):
        col = np.where(valid,  focal_px * Xe / Ze + cx, 0.0)  # image column
        row = np.where(valid, -focal_px * Ye / Ze + cy, 0.0)  # image row (Y flipped)

    # Pixels whose projected column/row fall outside the image
    in_bounds = (
        valid &
        (col >= 0) & (col < W_in - 1) &
        (row >= 0) & (row < H_in - 1)
    )

    # cv2.remap samples from img_np using the (col, row) map.
    # Out-of-bounds entries are zeroed out afterwards.
    map_x = col.astype(np.float32)
    map_y = row.astype(np.float32)
    map_x[~in_bounds] = 0.0
    map_y[~in_bounds] = 0.0

    projected = cv2.remap(
        img_np,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    projected[~in_bounds] = 0  # zero out anything sampled from invalid coords

    covered_pct = in_bounds.mean() * 100
    print(f"Coverage: {covered_pct:.1f}% of pano canvas filled by reference image")

    canvas_img = Image.fromarray(projected.astype(np.uint8))

    # Inpainting mask: white (255) = generate, black (0) = keep
    mask_np = (~in_bounds).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask_np)

    # -------------------------------------------------------------------------
    # Step 3: FLUX.1-dev Fill — inpaint the unseen regions
    # -------------------------------------------------------------------------
    print("Loading FLUX.1-dev Fill model (WorldGen)...")
    from worldgen.pano_gen import build_pano_fill_model, gen_pano_fill_image

    flux_model = build_pano_fill_model(device=device, low_vram=low_vram)

    print("Inpainting unseen regions...")
    pano = gen_pano_fill_image(
        flux_model,
        image=canvas_img,
        mask=mask_img,
        prompt=prompt,
        height=pano_h,
        width=pano_w,
    )
    print(f"Panorama: {pano.size}")

    buf = io.BytesIO()
    pano.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Local CLI entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    image: str,
    prompt: str = "",
    output: str = "",
    high_vram: bool = False,
):
    """
    image     -- path to the input perspective image (JPG, PNG, …)
    prompt    -- optional text description to guide outpainting
    output    -- output path for the panorama PNG (default: <stem>_pano.png)
    high_vram -- disable low-VRAM mode (upgrade gpu= to A100-80GB too)
    """
    image_path = Path(image)
    if not image_path.exists():
        print(f"error: {image_path} not found", file=sys.stderr)
        raise SystemExit(1)

    out_path = Path(output) if output else image_path.with_name(f"{image_path.stem}_pano.png")

    print(f"Input:   {image_path}")
    print(f"Output:  {out_path}")
    if prompt:
        print(f"Prompt:  {prompt}")
    print()

    pano_bytes = generate_pano.remote(
        image_bytes=image_path.read_bytes(),
        prompt=prompt,
        low_vram=not high_vram,
    )

    out_path.write_bytes(pano_bytes)
    print(f"Saved {out_path}  ({len(pano_bytes) // 1024} KB)")
