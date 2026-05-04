# pano_agent

Build a ComfyUI panorama-generation workflow from a single reference image of an interior space.

The agent analyzes the image, infers the geometry of the unseen parts of the space, and emits a ComfyUI workflow JSON with all prompt nodes, batch nodes, and Nano Banana generation nodes pre-wired in the right execution order to produce N synchronized panorama views.

Two pipelines are available:

```
pano_agent/
├── pano_agent/        ← shared library (brief, build, prompts)
├── cli/               ← Python CLI pipeline (one-shot, scriptable)
├── claude_code/       ← Claude Code pipeline (interactive, iterative)
└── tests/             ← test suite for the shared library
```

## Which pipeline?

| | CLI | Claude Code |
|---|---|---|
| Best for | Reproducible, headless, batch | Interactive, iterative, diagnostic |
| Setup | `pip install anthropic` + API key | Claude Code installed + authenticated |
| Workflow | One-shot, scriptable | Conversational |
| Can iterate on output images | No | Yes |
| Determinism | Same input → same brief | Conversational drift possible |

Both pipelines produce the same shape of output (`scene_brief.json` → `workflow.json`) and share the build pass, prompt templates, and tests. Switching between them at any point works fine.

## Quick start — CLI

```bash
cd cli/
pip install anthropic
export ANTHROPIC_API_KEY=sk-...

python pano_agent_cli.py run ../reference.png -o ../workflow.json
```

See [`cli/README.md`](cli/README.md) for full documentation.

## Quick start — Claude Code

```bash
cd claude_code/
claude --append-system-prompt "$(cat SESSION.md)" \
       "Analyze this reference image and write scene_brief.json. <attach reference.png>"

# After Claude writes the brief:
python build.py scene_brief.json -o workflow.json
```

See [`claude_code/README.md`](claude_code/README.md) for full documentation.

## What the workflow does

For an N-view panorama (N=8 by default), the generated ComfyUI workflow:

1. Generates the **reference cardinal** flat-on (the wall the input image faces) using only the reference image as input.
2. Generates the **opposite cardinal** (180° away) flat-on, also from the reference image.
3. Generates the remaining cardinals using `[reference + ref-cardinal output + opposite-cardinal output]` as inputs — the side walls have anchor views to seam to on both ends.
4. Generates the interstitial **corner views** using `[reference + flanking cardinals]` as inputs.

ComfyUI's topological execution naturally enforces this order. One queued run produces all 8 views in the correct sequence with consistent style, lighting, and architectural geometry.

## What's special about the prompts

The build pass embeds rules that emerged from extensive iteration on real generations. Without them, image models reliably:

- Drop fixtures (kitchen cabinets disappear)
- Vary window shape between views (some arched, some rectangular)
- Invent architectural recesses at corners (closets, alcoves, bump-outs)
- Re-light each view relative to the camera instead of in world space
- Render walls obliquely instead of flat-on

The prompt templates explicitly forbid each of these. See `pano_agent/prompts.py`.

## Tests

```bash
python -m pytest tests/ -v
```

25 tests covering geometry helpers, brief round-tripping, prompt synthesis, and build integrity (link validation, execution order, image-input wiring) for N ∈ {2, 4, 6, 8} cardinals.

## Background

This was iterated through a long conversation about generating a 360° panorama from a single hand-painted reference of a vintage trailer interior. The conversation produced both pipelines, the rule set that goes into the prompt templates, and the workflow structure. See `tests/test_pano_agent.py` for what each rule looks like in code.
