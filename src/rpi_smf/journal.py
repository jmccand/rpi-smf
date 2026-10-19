"""Rebalance journal (IPS 4 step 7, 4.3, 9B): everything needed to reproduce a run.

Each rebalance writes runs/<date>/ (tables as CSV plus manifest.json with config
hashes, input-file hashes, git commit, solver settings and lambda) and appends one
line to runs/journal.csv.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

from rpi_smf.config import REPO_ROOT


def sha256(path) -> Optional[str]:
    """Hash of a file, or of all CSVs under a directory; None if the path is missing."""
    p = Path(path)
    if not p.exists():
        return None
    # sorted so the directory hash is order-independent
    h = hashlib.sha256()
    for f in (sorted(p.rglob("*.csv")) if p.is_dir() else [p]):
        h.update(f.read_bytes())
    return h.hexdigest()


def git_commit() -> Optional[str]:
    """HEAD commit hash, suffixed '+uncommitted' if src/config have local changes."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "config"],
                               cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
        return out + ("+uncommitted" if dirty else "")
    except Exception:
        return None


def package_versions() -> Dict[str, str]:
    """Installed versions of the numerical dependencies, for reproducibility."""
    out = {}
    for p in ("pandas", "numpy", "scipy", "cvxpy", "yfinance"):
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return out


def write_run(runs_dir, run_name: str, tables: Dict[str, pd.DataFrame],
              manifest: dict, input_files: Iterable = (), config_files: Iterable = ()) -> Path:
    """Write a run's tables and manifest under runs/<run_name>/; returns that directory."""
    run = Path(runs_dir) / run_name
    run.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        t.to_csv(run / f"{name}.csv")
    # enrich a copy of the manifest with provenance (code, packages, inputs, config)
    manifest = dict(manifest)
    manifest["git_commit"] = git_commit()
    manifest["package_versions"] = package_versions()
    manifest["input_hashes"] = {str(Path(p).name): sha256(p) for p in input_files}
    manifest["config_hashes"] = {str(Path(p).name): sha256(p) for p in config_files}
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return run


def append_journal(runs_dir, row: dict) -> None:
    """Append one summary row to runs/journal.csv, writing the header on first use."""
    path = Path(runs_dir) / "journal.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
