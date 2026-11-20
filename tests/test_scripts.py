"""The numbered scripts: one sequence, each answering ``--help`` without data, a GPU or a stored artefact.

The scripts run in a fresh interpreter with ``configs/paths.toml`` replaced
by a table whose every location is an empty folder, so a script that read an
artefact while starting up would fail here.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from support import REPO as ROOT

SCRIPTS = sorted(p.name for p in (ROOT / "scripts").glob("*.py"))
NUMBERED = [name for name in SCRIPTS if name[:2].isdigit()]


def test_the_numbered_scripts_form_one_sequence():
    assert [int(name[:2]) for name in NUMBERED] == list(range(1, len(NUMBERED) + 1))
    assert set(SCRIPTS) - set(NUMBERED) == {"export_catalogue.py", "freeze_check.py"}


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_help_runs_without_data(tmp_path: Path, script: str):
    empty = tmp_path / "nothing"
    empty.mkdir()
    (tmp_path / "paths.toml").write_text(f'[data]\nflat = "{empty}"\n')
    python_path = [str(ROOT), *filter(None, os.environ.get("PYTHONPATH", "").split(os.pathsep))]
    env = {**os.environ, "AVITO_PATHS": str(tmp_path / "paths.toml"), "CUDA_VISIBLE_DEVICES": "",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": os.pathsep.join(python_path)}
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--help"], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith(f"usage: {script}")
    assert "python scripts/" in result.stdout          # the docstring with its example commands is the description
