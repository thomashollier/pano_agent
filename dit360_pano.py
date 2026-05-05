#!/usr/bin/env python3
"""
dit360_pano.py — perspective image → 360° equirectangular panorama using DiT360.

Pipeline:
  1. DepthPro (Apple) estimates the focal length (FOV) from the perspective image
  2. Inverse equirectangular projection places the reference image on a 1024×2048 canvas
  3. DiT360 (FLUX.1-dev + panoramic LoRA + Personalize Anything attention) outpaints
     the unseen regions while preserving the projected pixels

DiT360 paper: https://arxiv.org/abs/2510.11712
GitHub:       https://github.com/Insta360-Research-Team/DiT360

Key differences from worldgen_pano.py:
  - DiT360 uses FLUX.1-dev (not Fill) with a panoramic LoRA and circular-padded attention
  - Output is full-resolution 1024×2048 (DiT360's fixed training size)
  - The outpainting uses RFID inversion + masked attention rather than a fill/inpaint pipeline
  - Requires A100-80GB (~37 GB VRAM)

Setup (one-time):
    pip install modal
    modal setup
    modal secret create huggingface HF_TOKEN=hf_yourtoken

Usage:
    modal run dit360_pano.py --image input.jpg
    modal run dit360_pano.py --image input.jpg --prompt "a cozy vintage trailer interior"
    modal run dit360_pano.py --image input.jpg --output pano.png
"""

import io
import sys
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal app & container image
# ---------------------------------------------------------------------------

app = modal.App("dit360-pano")

hf_cache_vol = modal.Volume.from_name("worldgen-hf-cache", create_if_missing=True)

dit360_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "build-essential", "gcc", "g++", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        index_url="https://download.pytorch.org/whl/cu126",
    )
    .run_commands(
        # DiT360: FLUX.1-dev + panoramic LoRA + pa_src (Personalize Anything) modules
        "git clone --depth 1 https://github.com/Insta360-Research-Team/DiT360.git /opt/dit360",
        # diffusers must be installed from git HEAD to avoid a known bug
        # (https://github.com/huggingface/diffusers/issues/12436)
        "pip install git+https://github.com/huggingface/diffusers",
        "pip install accelerate peft 'transformers==4.52.4' sentencepiece protobuf"
        " opencv-python-headless py360convert",
        # DepthPro: focal-length estimation from perspective images
        "pip install git+https://github.com/apple/ml-depth-pro.git",
    )
    .env({"HF_HOME": "/hf_cache", "PYTHONPATH": "/opt/dit360"})
)


# ---------------------------------------------------------------------------
# Remote GPU function
# ---------------------------------------------------------------------------

@app.function(
    image=dit360_image,
    gpu="A100-80GB",   # DiT360 needs ~37 GB VRAM at 1024×2048
    timeout=900,
    volumes={"/hf_cache": hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def generate_pano(
    image_bytes: bytes,
    prompt: str = "",
    tau: int = 50,      # 0–100: lower = stronger source consistency, higher = more freedom
    seed: int = 0,
) -> bytes:
    """
    Perspective image → 360° equirectangular panorama (PNG bytes, 1024×2048).

    Step 1 — DepthPro estimates the camera focal length (FOV).
    Step 2 — Inverse equirectangular projection: for every pano pixel we compute
              its 3D ray on the sphere, project back into the perspective image,
              and sample with bilinear interpolation. Pixels outside the camera's
              view stay black and are marked for DiT360 to generate.
    Step 3 — DiT360 outpainting: RFID inversion + Personalize Anything attention
              preserves the projected pixels while generating the unseen ~75%+ of
              the sphere with FLUX.1-dev guided by the panoramic LoRA.
    """
    import os
    import sys
    import tempfile
    import numpy as np
    import cv2
    import torch
    import torch.nn.functional as F
    from PIL import Image

    # DiT360's pa_src and src packages live at /opt/dit360
    sys.path.insert(0, "/opt/dit360")

    PANO_H, PANO_W = 1024, 2048  # DiT360 is fixed at this resolution

    device = torch.device("cuda")
    input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    W_in, H_in = input_image.size
    print(f"Input: {W_in}×{H_in}  prompt={prompt!r}  tau={tau}  seed={seed}")

    # -------------------------------------------------------------------------
    # Step 1: DepthPro — estimate focal length
    # -------------------------------------------------------------------------
    print("Running DepthPro (FOV estimation)...")
    import depth_pro
    from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
    from huggingface_hub import hf_hub_download

    checkpoint_path = hf_hub_download(repo_id="apple/DepthPro", filename="depth_pro.pt")
    DEFAULT_MONODEPTH_CONFIG_DICT.checkpoint_uri = checkpoint_path

    dp_model, dp_transform = depth_pro.create_model_and_transforms(
        config=DEFAULT_MONODEPTH_CONFIG_DICT, device=device
    )
    dp_model.eval()

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
    # Step 2: Inverse equirectangular projection
    # -------------------------------------------------------------------------
    # For each pano pixel we compute the 3D ray on the unit sphere, then project
    # it back into the perspective image. Gap-free, geometrically correct.
    #
    # Coordinate system (camera looks along +Z):
    #   X = right,  Y = up,  Z = forward
    print("Projecting perspective image onto equirectangular canvas...")
    img_np = np.array(input_image)
    cx, cy = W_in / 2.0, H_in / 2.0

    pano_v, pano_u = np.mgrid[0:PANO_H, 0:PANO_W].astype(np.float32)
    azimuth   = (pano_u / PANO_W - 0.5) * 2.0 * np.pi   # [-π, +π]
    elevation = (0.5 - pano_v / PANO_H) * np.pi          # [+π/2, -π/2]

    Xe = np.sin(azimuth)  * np.cos(elevation)   # right
    Ye = np.sin(elevation)                        # up
    Ze = np.cos(azimuth)  * np.cos(elevation)    # forward

    eps = 1e-6
    valid = Ze > eps
    with np.errstate(divide="ignore", invalid="ignore"):
        col = np.where(valid,  focal_px * Xe / Ze + cx, 0.0)
        row = np.where(valid, -focal_px * Ye / Ze + cy, 0.0)

    in_bounds = (
        valid &
        (col >= 0) & (col < W_in - 1) &
        (row >= 0) & (row < H_in - 1)
    )

    map_x = col.astype(np.float32); map_x[~in_bounds] = 0.0
    map_y = row.astype(np.float32); map_y[~in_bounds] = 0.0

    projected = cv2.remap(
        img_np, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    projected[~in_bounds] = 0

    covered_pct = in_bounds.mean() * 100
    print(f"Coverage: {covered_pct:.1f}% of pano canvas filled by reference image")

    canvas_img = Image.fromarray(projected.astype(np.uint8))

    # -------------------------------------------------------------------------
    # Step 3: DiT360 outpainting
    # -------------------------------------------------------------------------
    print("Loading DiT360 pipeline (FLUX.1-dev + DiT360 panoramic LoRA)...")

    from pa_src.pipeline import RFPanoInversionParallelFluxPipeline
    from pa_src.attn_processor import (
        PersonalizeAnythingAttnProcessor,
        set_flux_transformer_attn_processor,
    )

    dtype = torch.float16
    pipe = RFPanoInversionParallelFluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    pipe.load_lora_weights("Insta360-Research/DiT360-Panorama-Image-Generation")

    # Build latent-resolution mask: 1 = preserve projected pixels, 0 = generate
    # DiT360 uses vae_scale_factor=8 and halves again → effective factor=16
    latent_h = PANO_H // (pipe.vae_scale_factor * 2)   # 1024 // 16 = 64
    latent_w = PANO_W // (pipe.vae_scale_factor * 2)   # 2048 // 16 = 128
    img_dims  = latent_h * (latent_w + 2)               # circular-padded token count

    mask_full   = torch.from_numpy(in_bounds.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    mask_latent = F.interpolate(mask_full, size=(latent_h, latent_w), mode="nearest").squeeze()
    # Circular-pad left/right edges to match DiT360's boundary-continuity tokens
    mask = torch.cat([mask_latent[:, 0:1], mask_latent, mask_latent[:, -1:]], dim=-1).view(-1, 1)

    # Invert the projected canvas so we can re-generate while anchoring known pixels
    pano_prompt = f"This is a panorama image. {prompt}" if prompt else "This is a panorama image."
    print("Inverting source canvas...")
    inverted_latents, image_latents, latent_image_ids = pipe.invert(
        source_prompt="",
        image=canvas_img,
        height=PANO_H,
        width=PANO_W,
        num_inversion_steps=50,
        gamma=1.0,
    )

    # Personalize Anything attention: mask controls which tokens are locked to source
    set_flux_transformer_attn_processor(
        pipe.transformer,
        set_attn_proc_func=lambda name, dh, nh, ap: PersonalizeAnythingAttnProcessor(
            name=name,
            tau=tau / 100.0,   # tau: lower = stronger source lock
            mask=mask,
            device=device,
            img_dims=img_dims,
        ),
    )

    print("Outpainting unseen regions...")
    result = pipe(
        [pano_prompt, pano_prompt],
        inverted_latents=inverted_latents,
        image_latents=image_latents,
        latent_image_ids=latent_image_ids,
        height=PANO_H,
        width=PANO_W,
        start_timestep=0.0,
        stop_timestep=0.99,
        num_inference_steps=50,
        eta=1.0,
        generator=torch.Generator(device=device).manual_seed(seed),
        mask=mask,
        use_timestep=True,
    )
    pano = result.images[1]
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
    tau: int = 50,
    seed: int = 0,
):
    """
    image   -- path to the input perspective image (JPG, PNG, …)
    prompt  -- text description to guide the outpainting
    output  -- output path (default: <stem>_dit360.png)
    tau     -- source-consistency strength 0–100 (default 50; lower = tighter lock)
    seed    -- random seed for reproducibility (default 0)
    """
    image_path = Path(image)
    if not image_path.exists():
        print(f"error: {image_path} not found", file=sys.stderr)
        raise SystemExit(1)

    out_path = Path(output) if output else image_path.with_name(f"{image_path.stem}_dit360.png")

    print(f"Input:  {image_path}")
    print(f"Output: {out_path}")
    if prompt:
        print(f"Prompt: {prompt}")
    print(f"tau={tau}  seed={seed}")
    print()

    pano_bytes = generate_pano.remote(
        image_bytes=image_path.read_bytes(),
        prompt=prompt,
        tau=tau,
        seed=seed,
    )

    out_path.write_bytes(pano_bytes)
    print(f"Saved {out_path}  ({len(pano_bytes) // 1024} KB)")
