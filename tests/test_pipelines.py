"""
Integration tests for the two pipeline entry points.

The Python CLI's `build` subcommand and the Claude Code pipeline's `build.py`
must both:
  - load a scene_brief.json correctly
  - emit a valid workflow.json
  - share the same output (since they share the same build code)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()


def _make_minimal_brief() -> dict:
    """A minimal but valid brief for end-to-end tests."""
    return {
        "style": {
            "art_style": "test", "palette": "test",
            "materials": "test", "atmosphere": "test",
            "rendering_notes": "test",
        },
        "space": {
            "type_label": "test", "dimensions": "test",
            "geometry_rules": "test", "floor": "test",
            "ceiling": "test", "walls": "test",
        },
        "windows": {"description": "test", "placement_rules": "test"},
        "doors": {"exterior": "test", "interior": "test"},
        "lighting": {
            "key_light": "test", "fill_light": "test",
            "darkest_zone": "test",
            "fixed_world_space_description": "test",
        },
        "camera": {
            "height_meters": 1.6,
            "fov_horizontal_degrees": 90,
            "fov_vertical_degrees": 90,
            "aspect_ratio": "1:1",
            "flat_on_required": True,
            "extra_notes": "",
        },
        "cardinals": [
            {
                "key": f"C{i+1}", "angle": i * 90,
                "label": f"View {i+1}",
                "wall_role": f"wall {i}",
                "wall_inventory": f"UPPER: {i}\nMID: {i}\nLOWER: {i}",
                "left_edge_content": f"left {i}",
                "right_edge_content": f"right {i}",
                "custom_notes": "",
            }
            for i in range(4)
        ],
        "interstitials": [
            {
                "key": f"I{i+1}", "angle": 45 + i * 90,
                "label": f"Corner {i+1}",
                "left_neighbor_key": f"C{i+1}",
                "right_neighbor_key": f"C{((i+1) % 4) + 1}",
                "left_half_content": f"L {i}",
                "right_half_content": f"R {i}",
                "corner_description": f"corner {i}",
                "custom_notes": "",
            }
            for i in range(4)
        ],
        "reference_cardinal_key": "C1",
    }


@pytest.fixture
def brief_file(tmp_path):
    """Write a minimal brief to disk and return its path."""
    path = tmp_path / "scene_brief.json"
    path.write_text(json.dumps(_make_minimal_brief()))
    return path


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, check=False,
    )


# ---------------------------------------------------------------------------
# Claude Code pipeline: claude_code/build.py
# ---------------------------------------------------------------------------

def test_claude_code_build_runs(brief_file: Path, tmp_path: Path):
    out = tmp_path / "workflow.json"
    result = _run(
        [sys.executable,
         str(REPO_ROOT / "claude_code" / "build.py"),
         str(brief_file), "-o", str(out)],
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, f"build.py failed:\nstderr: {result.stderr}\nstdout: {result.stdout}"
    assert out.exists()
    wf = json.loads(out.read_text())
    assert wf["nodes"]
    assert wf["links"]


def test_claude_code_build_handles_missing_brief(tmp_path: Path):
    result = _run(
        [sys.executable,
         str(REPO_ROOT / "claude_code" / "build.py"),
         str(tmp_path / "nonexistent.json"),
         "-o", str(tmp_path / "out.json")],
        cwd=REPO_ROOT,
    )
    assert result.returncode == 1
    assert "not found" in result.stderr


def test_claude_code_build_handles_invalid_json(tmp_path: Path):
    bad_brief = tmp_path / "bad.json"
    bad_brief.write_text("{this is not valid JSON")
    result = _run(
        [sys.executable,
         str(REPO_ROOT / "claude_code" / "build.py"),
         str(bad_brief),
         "-o", str(tmp_path / "out.json")],
        cwd=REPO_ROOT,
    )
    assert result.returncode == 1


# ---------------------------------------------------------------------------
# CLI pipeline: cli/pano_agent_cli.py build
# ---------------------------------------------------------------------------

def test_cli_build_runs(brief_file: Path, tmp_path: Path):
    out = tmp_path / "workflow.json"
    result = _run(
        [sys.executable,
         str(REPO_ROOT / "cli" / "pano_agent_cli.py"),
         "build", str(brief_file), "-o", str(out)],
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, f"CLI build failed:\nstderr: {result.stderr}\nstdout: {result.stdout}"
    assert out.exists()
    wf = json.loads(out.read_text())
    assert wf["nodes"]


# ---------------------------------------------------------------------------
# Both pipelines must produce identical output for the same brief
# ---------------------------------------------------------------------------

def test_both_pipelines_produce_identical_workflow(brief_file: Path, tmp_path: Path):
    cli_out = tmp_path / "cli_workflow.json"
    cc_out = tmp_path / "cc_workflow.json"

    cli_result = _run(
        [sys.executable,
         str(REPO_ROOT / "cli" / "pano_agent_cli.py"),
         "build", str(brief_file), "-o", str(cli_out)],
        cwd=REPO_ROOT,
    )
    assert cli_result.returncode == 0

    cc_result = _run(
        [sys.executable,
         str(REPO_ROOT / "claude_code" / "build.py"),
         str(brief_file), "-o", str(cc_out)],
        cwd=REPO_ROOT,
    )
    assert cc_result.returncode == 0

    cli_wf = json.loads(cli_out.read_text())
    cc_wf = json.loads(cc_out.read_text())

    # Both pipelines call the same build_workflow function with the same args,
    # so output must be byte-identical.
    assert cli_wf == cc_wf, "CLI and Claude Code build outputs differ"
