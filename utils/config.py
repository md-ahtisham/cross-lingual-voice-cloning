"""
utils/config.py
~~~~~~~~~~~~~~~
YAML config loader and helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Load and return the YAML config as a plain dict."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p.resolve()}")
    with open(p, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file must be a YAML mapping: {p}")
    return cfg


def enabled_models(cfg: Dict[str, Any]) -> List[str]:
    """Return list of model names that have ``enabled: true`` in config."""
    models_section = cfg.get("models", {})
    return [
        name
        for name, mcfg in models_section.items()
        if isinstance(mcfg, dict) and mcfg.get("enabled", False)
    ]
