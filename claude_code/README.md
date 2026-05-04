# pano_agent — Claude Code pipeline

An interactive, conversational alternative to the standalone CLI. You drive the analysis through a Claude Code session, then run the build script to assemble the workflow.

## Quick start

```bash
cd claude_code/

# Start a Claude Code session, attaching the session prompt and your image
claude --append-system-prompt "$(cat SESSION.md)" \
       "Analyze this reference image and write scene_brief.json. <attach reference.png>"

# Claude analyzes, writes scene_brief.json. Review/edit it.

# Then ask Claude (or run directly):
python build.py scene_brief.json -o workflow.json
```

Or, if you'd rather work directly with `claude -p`:

```bash
claude -p "$(cat SESSION.md)

Now analyze this image and write scene_brief.json:" --image reference.png > scene_brief.json

# review/edit

python build.py scene_brief.json -o workflow.json
```

## When to use this over the CLI

This pipeline is better for:

- **Iterative scene analysis.** You can ask follow-up questions ("are you sure the back wall has no windows?", "extend the brief to handle a 6-cardinal panorama instead of 4"), refine, regenerate. The CLI is one-shot.
- **Diagnostic loops.** When a generated image looks wrong, share it back into the session and ask Claude to figure out which brief field needs editing. The CLI can't see outputs.
- **Custom analysis depth.** Tell Claude to spend more time on certain aspects, ignore others, or use specific terminology. The CLI runs a fixed analysis prompt.

The CLI is better for:

- **Reproducibility.** Same image → same brief, no conversational drift.
- **Headless / scripted runs.** Wrapping in a service, batch processing many images.
- **No Claude Code required.** Just a Python install and an API key.

## How it works

The session prompt (`SESSION.md`) tells Claude:

- The panorama-generation task and pipeline shape
- The exact JSON structure to produce (`scene_brief.schema.json`)
- The hard-won rules about wall inventory format, no-recess corners, canonical window spec, world-space lighting, etc.
- How to invoke `build.py` once the brief is ready

Claude reads the prompt, analyzes the user's image, writes `scene_brief.json`, and (when asked) runs the build script. The build script is shared with the CLI pipeline — it imports from the same `pano_agent/` library.

## Files

- `SESSION.md` — system prompt that frames the task for Claude Code
- `scene_brief.schema.json` — JSON schema describing the brief structure
- `build.py` — thin wrapper around `pano_agent.build.build_workflow`. Invoke directly with `python build.py brief.json -o workflow.json`. Runs offline with no API calls.

## Prerequisites

- Claude Code installed and authenticated
- Python 3.9+
- The `pano_agent/` package directory in the parent directory (or copied next to `build.py`)

## Tip

Once you have a brief that works for one scene, copy it as a template for similar scenes. Most edits are scene-specific (what's on each wall) — the style/space/window/lighting structure stays similar across "vintage trailer" or "1970s motel room" or "ship's cabin" type briefs.
