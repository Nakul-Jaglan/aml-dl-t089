"""Small helpers for artifact provenance and stable file formats."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = Path(os.environ.get("CLOTHING_RESULTS_ROOT", ROOT / "results")).resolve()


def jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=jsonable, allow_nan=False) + "\n")
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=jsonable).encode()).hexdigest()


def environment():
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "scikit-image", "matplotlib", "joblib", "threadpoolctl"]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except subprocess.CalledProcessError:
        commit = "uncommitted-initial-build"
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "machine": platform.machine(),
        "git_commit": commit,
        "packages": {p: importlib.metadata.version(p) for p in packages},
    }

