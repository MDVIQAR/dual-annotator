"""
mlops/registry/projects_config.py

Persists user-defined project names and categories shared across all MLOps tabs.
Stored in %LOCALAPPDATA%\DualAnnotator\projects_config.json.
"""

import os
import json

_CONFIG_PATH = os.path.join(
    os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
    "DualAnnotator", "projects_config.json"
)


def load() -> dict:
    if not os.path.isfile(_CONFIG_PATH):
        return {"projects": [], "categories": []}
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return {
            "projects":   list(data.get("projects", [])),
            "categories": list(data.get("categories", [])),
        }
    except (json.JSONDecodeError, OSError):
        return {"projects": [], "categories": []}


def save(data: dict):
    os.makedirs(os.path.dirname(_CONFIG_PATH), exist_ok=True)
    tmp = _CONFIG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, _CONFIG_PATH)


def get_projects() -> list:
    return load()["projects"]


def get_categories() -> list:
    return load()["categories"]


def ensure_project(name: str):
    """Add project name to saved list if not already present."""
    name = name.strip()
    if not name:
        return
    data = load()
    if name not in data["projects"]:
        data["projects"].append(name)
        save(data)


def ensure_category(name: str):
    """Add category to saved list if not already present."""
    name = name.strip()
    if not name:
        return
    data = load()
    if name not in data["categories"]:
        data["categories"].append(name)
        save(data)
