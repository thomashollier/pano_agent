#!/usr/bin/env python3
"""
hunyuan_pano.py — perspective image → 360° equirectangular panorama using HunyuanWorld 1.0.

Pipeline:
  1. DepthPro (Apple) estimates the focal length (FOV) from the perspective image
  2. HunyuanWorld's Perspective class projects the image onto a 960×1920 equirectangular canvas
  3. Image2PanoramaPipelines (FLUX.1-Fill-dev + HunyuanWorld panoramic LoRA) outpaints
     the unseen regions

HunyuanWorld 1.0:  https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0
Model weights:     https://huggingface.co/tencent/HunyuanWorld-1

Setup (one-time):
    pip install modal
    modal setup
    modal secret create huggingface HF_TOKEN=hf_yourtoken

Usage:
    modal run hunyuan_pano.py --image input.jpg
    modal run hunyuan_pano.py --image input.jpg --prompt "a cozy vintage trailer interior"
    modal run hunyuan_pano.py --image input.jpg --output pano.png
"""

import io
import sys
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal app & container image
# ---------------------------------------------------------------------------

app = modal.App("hunyuan-pano")

hf_cache_vol = modal.Volume.from_name("worldgen-hf-cache", create_if_missing=True)

hunyuan_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "build-essential", "gcc", "g++", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.5.0",
        "torchvision==0.20.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .run_commands(
        # HunyuanWorld-1.0: provides hy3dworld package
        "git clone --depth 1 https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0.git /opt/hunyuanworld",
        # Patch the three __init__.py files to keep only pano-generation imports,
        # stripping out the 3D scene-gen deps (open3d, utils3d, moge, Real-ESRGAN, ZIM, trimesh)
        # that are not needed and would require heavy installs.
        "python3 -c \""
        "import pathlib; base = pathlib.Path('/opt/hunyuanworld/hy3dworld');"
        "(base / '__init__.py').write_text("
        "    'from .models import Image2PanoramaPipelines, Text2PanoramaPipelines\\n'"
        "    'from .utils import Perspective\\n'"
        ");"
        "(base / 'models/__init__.py').write_text("
        "    'from .pano_generator import Image2PanoramaPipelines, Text2PanoramaPipelines\\n'"
        "    'from .pipelines import FluxPipeline, FluxFillPipeline\\n'"
        ");"
        "(base / 'utils/__init__.py').write_text("
        "    'from .perspective_utils import Perspective\\n'"
        ");"
        "\"",
        "pip install accelerate==1.6.0 diffusers peft transformers sentencepiece"
        " protobuf safetensors huggingface_hub opencv-python-headless einops",
        # DepthPro: focal-length estimation from perspective images
        "pip install git+https://github.com/apple/ml-depth-pro.git",
    )
    .env({"HF_HOME": "/hf_cache", "PYTHONPATH": "/opt/hunyuanworld"})
)


# ---------------------------------------------------------------------------
# Remote GPU function
# ---------------------------------------------------------------------------

@app.function(
    image=hunyuan_image,
    gpu="A100",
    timeout=900,
    volumes={"/hf_cache": hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def generate_pano(
    image_bytes: bytes,
    prompt: str = "",
    seed: int = 0,
    pano_h: int = 1024,
    pano_w: int = 2048,
    steps: int = 75,
) -> bytes:
    """
    Perspective image → 360° equirectangular panorama.

    Step 1 — DepthPro estimates the camera focal length, giving us the real hFOV.
    Step 2 — HunyuanWorld's Perspective class projects the image onto the equirectangular
              canvas using the DepthPro-derived FOV, producing the filled region and a mask.
    Step 3 — Image2PanoramaPipelines (FLUX.1-Fill-dev + panoramic LoRA) outpaints
              the unseen regions guided by the prompt.
    """
    import os
    import sys
    import tempfile
    import numpy as np
    import cv2
    import torch
    from PIL import Image

    sys.path.insert(0, "/opt/hunyuanworld")

    PANO_H, PANO_W = pano_h, pano_w

    device = torch.device("cuda")
    input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    W_in, H_in = input_image.size
    print(f"Input: {W_in}×{H_in}  prompt={prompt!r}  seed={seed}")

    # -------------------------------------------------------------------------
    # Step 1: DepthPro — estimate focal length → hFOV
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
    hFOV_deg = float(2 * np.degrees(np.arctan(W_in / (2 * focal_px))))
    print(f"DepthPro: focal_length={focal_px:.1f}px  hFOV={hFOV_deg:.1f}°")

    del dp_model
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Step 2: HunyuanWorld Perspective projection
    # -------------------------------------------------------------------------
    # The Perspective class takes: (img_bgr, FOV_deg, THETA_deg, PHI_deg)
    # THETA=0, PHI=0 = camera looking straight forward (no yaw/pitch offset)
    # GetEquirec returns: (equirectangular_canvas_bgr, binary_mask)
    print("Projecting perspective image onto equirectangular canvas...")
    from hy3dworld.utils.perspective_utils import Perspective

    img_bgr = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    equ = Perspective(img_bgr, hFOV_deg, THETA=0, PHI=0, crop_bound=False)
    canvas_bgr, mask_np = equ.GetEquirec(PANO_H, PANO_W)

    # Erode mask slightly to avoid hard projection edges at the boundary
    mask_np = cv2.erode(
        mask_np.astype(np.uint8),
        np.ones((3, 3), np.uint8),
        iterations=5,
    )

    canvas_rgb = cv2.cvtColor(canvas_bgr.astype(np.uint8), cv2.COLOR_BGR2RGB)
    canvas_pil = Image.fromarray(canvas_rgb)

    # FLUX Fill expects mask: white (255) = generate, black (0) = keep.
    # mask_np from Perspective: 1 = filled, 0 = empty → invert for Fill.
    mask_pil = Image.fromarray(((1 - mask_np) * 255).astype(np.uint8))

    covered_pct = mask_np.mean() * 100
    print(f"Coverage: {covered_pct:.1f}% of pano canvas filled by reference image")

    # -------------------------------------------------------------------------
    # Step 3: HunyuanWorld Image2PanoramaPipelines outpainting
    # -------------------------------------------------------------------------
    print("Loading HunyuanWorld pipeline (FLUX.1-Fill-dev + panoramic LoRA)...")
    from hy3dworld.models.pano_generator import Image2PanoramaPipelines

    pipe = Image2PanoramaPipelines.from_pretrained(
        "black-forest-labs/FLUX.1-Fill-dev",
        torch_dtype=torch.bfloat16,
    )
    pipe.load_lora_weights(
        "tencent/HunyuanWorld-1",
        subfolder="HunyuanWorld-PanoDiT-Image",
        weight_name="lora.safetensors",
        torch_dtype=torch.bfloat16,
    )
    pipe.fuse_lora()
    pipe.unload_lora_weights()
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()

    pano_prompt = prompt if prompt else ""
    print("Outpainting unseen regions...")
    result = pipe(
        prompt=pano_prompt,
        image=canvas_pil,
        mask_image=mask_pil,
        height=PANO_H,
        width=PANO_W,
        negative_prompt="",
        guidance_scale=30,
        num_inference_steps=steps,
        generator=torch.Generator("cpu").manual_seed(seed),
        blend_extend=6,
        shifting_extend=0,
        true_cfg_scale=0.0,
    )
    pano = result.images[0]
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
    seed: int = 0,
    pano_h: int = 1024,
    pano_w: int = 2048,
    steps: int = 75,
):
    """
    image   -- path to the input perspective image (JPG, PNG, …)
    prompt  -- text description to guide the outpainting
    output  -- output path (default: <stem>_hunyuan.png)
    seed    -- random seed (default 0)
    """
    image_path = Path(image)
    if not image_path.exists():
        print(f"error: {image_path} not found", file=sys.stderr)
        raise SystemExit(1)

    out_path = Path(output) if output else image_path.with_name(f"{image_path.stem}_hunyuan.png")

    print(f"Input:  {image_path}")
    print(f"Output: {out_path}")
    if prompt:
        print(f"Prompt: {prompt}")
    print(f"seed={seed}")
    print()

    print(f"Resolution: {pano_w}×{pano_h}  steps={steps}")
    print()

    pano_bytes = generate_pano.remote(
        image_bytes=image_path.read_bytes(),
        prompt=prompt,
        seed=seed,
        pano_h=pano_h,
        pano_w=pano_w,
        steps=steps,
    )

    out_path.write_bytes(pano_bytes)
    print(f"Saved {out_path}  ({len(pano_bytes) // 1024} KB)")
