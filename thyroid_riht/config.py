"""Configuration loading and project paths."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Load the YAML config; ``overrides`` is a (possibly nested) dict merged on top."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if overrides:
        cfg = _deep_update(cfg, overrides)
    cfg["_config_path"] = str(path)
    return cfg


def _deep_update(base: dict, upd: dict) -> dict:
    base = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def resolve(cfg: dict, key: str) -> Path:
    """Resolve cfg['paths'][key] relative to the project root."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else PROJECT_ROOT / p
