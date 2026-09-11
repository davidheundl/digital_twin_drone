"""Config loading. One YAML file is the single source of truth for the world."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


class Config(dict):
    """Dict with attribute-ish nested access helpers."""

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    path = Path(path) if path else DEFAULT_PATH
    with open(path, "r") as fh:
        data = yaml.safe_load(fh)
    if overrides:
        data = _deep_merge(data, overrides)
    return Config(data)


def _deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out
