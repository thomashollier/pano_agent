#!/usr/bin/env python3
"""
build.py — Claude Code helper. Builds a ComfyUI workflow from a scene brief.

Designed to be invoked by Claude during a Code session:

    python build.py scene_brief.json -o workflow.json

This is a thin wrapper around pano_agent.build.build_workflow. The actual
prompt synthesis and workflow assembly live in the pano_agent/ package
shared with the CLI pipeline. Updates to prompt templates apply here
automatically.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Resolve the shared library path. Works whether build.py is run from
# the claude_code/ directory or copied to a working directory next to
# pano_agent/.
HERE = Path(__file__).parent.resolve()
for candidate in (HERE.parent, HERE):
    if (candidate / "pano_agent" / "build.py").exists():
        sys.path.insert(0, str(candidate))
        break

from pano_agent.brief import SceneBrief  # noqa: E402
from pano_agent.build import build_workflow  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a ComfyUI panorama workflow from a scene brief.",
    )
    parser.add_argument("brief", help="path to scene_brief.json")
    parser.add_argument("-o", "--output", default="workflow.json")
    parser.add_argument(
        "--reference-filename",
        default="reference.png",
        help="filename for the LoadImage node (must exist in ComfyUI's input/)",
    )
    args = parser.parse_args()

    brief_path = Path(args.brief)
    if not brief_path.exists():
        print(f"error: {brief_path} not found", file=sys.stderr)
        return 1

    try:
        data = json.loads(brief_path.read_text())
        brief = SceneBrief.from_dict(data)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        print(f"error: invalid scene_brief.json: {e}", file=sys.stderr)
        return 1

    workflow = build_workflow(brief, reference_filename=args.reference_filename)
    out = Path(args.output)
    out.write_text(json.dumps(workflow, indent=2))

    n_cards = len(brief.cardinals)
    n_inters = len(brief.interstitials)
    print(f"wrote {out}")
    print(f"  {n_cards} cardinals, {n_inters} interstitials")
    print(f"  {len(workflow['nodes'])} nodes, {len(workflow['links'])} links")
    print()
    print(f"Load this in ComfyUI and place {args.reference_filename} in ComfyUI's input/ directory.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
