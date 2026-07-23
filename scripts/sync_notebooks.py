#!/usr/bin/env python3
"""Jupytext 同步脚本：将 notebook_sources/*.py 同步为 notebooks/*.ipynb"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "notebook_sources"
NOTEBOOKS = ROOT / "notebooks"

def sync_all():
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    py_files = sorted(SOURCES.glob("*.py"))
    for py_file in py_files:
        ipynb_file = NOTEBOOKS / py_file.with_suffix(".ipynb").name
        print(f"Syncing: {py_file.name} -> {ipynb_file.name}")
        subprocess.run(
            ["uv", "run", "jupytext", "--to", "notebook",
             str(py_file), "-o", str(ipynb_file)],
            check=True, cwd=ROOT
        )
    print(f"\n✅ Synced {len(py_files)} notebooks.")

if __name__ == "__main__":
    sync_all()
