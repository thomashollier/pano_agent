#!/usr/bin/env python3
"""
pano_agent CLI — Python pipeline for building a ComfyUI panorama workflow.

Two stages with a human-editable JSON between them:

    pano_agent_cli.py analyze ref.png -o brief.json     (calls Anthropic API)
    pano_agent_cli.py build   brief.json -o workflow.json   (local, deterministic)

Or run both back-to-back:

    pano_agent_cli.py run ref.png -o workflow.json --brief-output brief.json

For an interactive session-based alternative, see claude_code/README.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path so we can import pano_agent and cli.analyze
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.analyze import analyze_image
from pano_agent.brief import SceneBrief
from pano_agent.build import build_workflow


def cmd_analyze(args: argparse.Namespace) -> int:
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"error: {image_path} not found", file=sys.stderr)
        return 1

    brief = analyze_image(
        image_path=image_path,
        n_cardinals=args.views // 2,
        anthropic_key=args.api_key,
        model=args.model,
        verbose=args.verbose,
    )
    out = Path(args.output)
    out.write_text(json.dumps(brief.to_dict(), indent=2))
    print(f"wrote {out}")
    print(f"  cardinals: {len(brief.cardinals)}")
    print(f"  interstitials: {len(brief.interstitials)}")
    if not args.no_review_hint:
        print()
        print("Review the brief and edit anything wrong before building.")
        print("Common things to check:")
        print("  - cardinals[*].wall_inventory (top-to-bottom of each wall)")
        print("  - windows.description / doors.exterior / doors.interior")
        print("  - lighting.fixed_world_space_description")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    brief_path = Path(args.brief)
    if not brief_path.exists():
        print(f"error: {brief_path} not found", file=sys.stderr)
        return 1

    brief = SceneBrief.from_dict(json.loads(brief_path.read_text()))
    workflow = build_workflow(brief, reference_filename=args.reference_filename)
    out = Path(args.output)
    out.write_text(json.dumps(workflow, indent=2))
    print(f"wrote {out}")
    print(f"  {len(workflow['nodes'])} nodes, {len(workflow['links'])} links")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"error: {image_path} not found", file=sys.stderr)
        return 1

    brief = analyze_image(
        image_path=image_path,
        n_cardinals=args.views // 2,
        anthropic_key=args.api_key,
        model=args.model,
        verbose=args.verbose,
    )
    workflow = build_workflow(brief, reference_filename=image_path.name)
    out = Path(args.output)
    out.write_text(json.dumps(workflow, indent=2))
    print(f"wrote {out} ({len(workflow['nodes'])} nodes)")

    if args.brief_output:
        brief_out = Path(args.brief_output)
        brief_out.write_text(json.dumps(brief.to_dict(), indent=2))
        print(f"wrote {brief_out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pano_agent_cli", description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--api-key", help="Anthropic API key (or set ANTHROPIC_API_KEY)")
    parser.add_argument("--model", default="claude-opus-4-5", help="Vision model name")
    parser.add_argument("--verbose", "-v", action="store_true")

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_an = sub.add_parser("analyze", help="reference image → scene brief JSON")
    p_an.add_argument("image", help="reference image path")
    p_an.add_argument("-o", "--output", default="scene_brief.json")
    p_an.add_argument("--views", type=int, default=8,
                      help="total views (must be even and divide 360); cardinals = views/2")
    p_an.add_argument("--no-review-hint", action="store_true")
    p_an.set_defaults(func=cmd_analyze)

    p_bd = sub.add_parser("build", help="scene brief JSON → ComfyUI workflow JSON")
    p_bd.add_argument("brief", help="scene_brief.json path")
    p_bd.add_argument("-o", "--output", default="workflow.json")
    p_bd.add_argument("--reference-filename", default="reference.png",
                      help="filename to put in the LoadImage node (image must be in Comfy's input/)")
    p_bd.set_defaults(func=cmd_build)

    p_rn = sub.add_parser("run", help="analyze + build in one shot")
    p_rn.add_argument("image", help="reference image path")
    p_rn.add_argument("-o", "--output", default="workflow.json")
    p_rn.add_argument("--brief-output", help="also save the intermediate brief JSON")
    p_rn.add_argument("--views", type=int, default=8)
    p_rn.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
