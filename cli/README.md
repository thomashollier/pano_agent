# pano_agent — Python CLI pipeline

Standalone CLI that calls the Anthropic API to analyze a reference image and emit a ComfyUI panorama workflow.

## Install

```bash
pip install anthropic
```

## Quick start

```bash
export ANTHROPIC_API_KEY=sk-...

# Two-stage (recommended — review the brief before building)
python pano_agent_cli.py analyze reference.png -o brief.json
# review/edit brief.json
python pano_agent_cli.py build brief.json -o workflow.json

# Or one-shot
python pano_agent_cli.py run reference.png -o workflow.json --brief-output brief.json
```

## Subcommands

### `analyze IMAGE -o BRIEF`

Calls the Anthropic vision API. Two API calls under the hood: one with the image to describe what's visible, one text-only to extrapolate to the unseen walls and corners. Output is `scene_brief.json`.

Options:
- `--views N` — total panorama views (default 8). Cardinals = N/2.
- `--api-key KEY` — Anthropic key. Or set `ANTHROPIC_API_KEY`.
- `--model NAME` — vision model (default `claude-opus-4-5`).
- `--no-review-hint` — suppress the "review the brief" message.

### `build BRIEF -o WORKFLOW`

Local, deterministic. Loads the brief, runs prompt synthesis, emits ComfyUI workflow JSON.

Options:
- `--reference-filename NAME` — filename to embed in the LoadImage node. Must match the file in ComfyUI's `input/` directory. Default `reference.png`.

### `run IMAGE -o WORKFLOW`

`analyze` + `build` in one shot.

Options:
- `--brief-output PATH` — also save the intermediate brief.

## When to use this over the Claude Code pipeline

The CLI is better for:
- **Reproducibility.** Same input image always produces a similar brief — no conversational drift.
- **Headless automation.** Wrapping in a service, batch processing many images, CI/CD.
- **No Claude Code required.** Works with just `pip install anthropic`.

The Claude Code pipeline (in `claude_code/`) is better for:
- **Iterative analysis.** Refining the brief through follow-up questions.
- **Diagnostic loops.** Showing generated images back to Claude and asking for brief edits.

## Files

- `pano_agent_cli.py` — CLI entry point with `analyze`, `build`, `run` subcommands.
- `analyze.py` — vision API call logic. Two-call structure (analyze visible cardinal, extrapolate the rest), markdown-fence-tolerant JSON parsing.

The actual prompt synthesis and workflow assembly live in the parent `pano_agent/` package — both pipelines share that code.
