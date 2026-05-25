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


# ---------------------------------------------------------------------------
# 4-level hierarchy persistence for cascading dropdowns
# ---------------------------------------------------------------------------
# Storage structure inside the JSON under "hierarchy":
# {
#   "yolo": {
#     "SnipeX": {
#       "ids": {
#         "SLXCAP": {
#           "variants": {
#             "CRC": { "cameras": ["cam0", "cam1"] }
#           }
#         }
#       }
#     }
#   }
# }

def _get_hierarchy(data: dict) -> dict:
    return data.get("hierarchy", {})


def ensure_hierarchy(model_type: str, project_name: str, project_id: str,
                     variant: str, camera: str):
    """Persist a complete 4-level entry so it appears in future dropdown lists."""
    if not all([model_type.strip(), project_name.strip(), project_id.strip(),
                variant.strip(), camera.strip()]):
        return
    data = load()
    h = data.setdefault("hierarchy", {})
    m = h.setdefault(model_type, {})
    pn = m.setdefault(project_name, {})
    ids = pn.setdefault("ids", {})
    pid = ids.setdefault(project_id, {})
    variants = pid.setdefault("variants", {})
    var = variants.setdefault(variant, {})
    cameras = var.setdefault("cameras", [])
    if camera not in cameras:
        cameras.append(camera)
        cameras.sort()
    save(data)


def get_custom_project_names(model_type: str) -> list:
    data = load()
    return sorted(_get_hierarchy(data).get(model_type, {}).keys())


def get_custom_project_ids(model_type: str, project_name: str) -> list:
    data = load()
    pn = _get_hierarchy(data).get(model_type, {}).get(project_name, {})
    return sorted(pn.get("ids", {}).keys())


def get_custom_variants(model_type: str, project_name: str, project_id: str) -> list:
    data = load()
    pn = _get_hierarchy(data).get(model_type, {}).get(project_name, {})
    pid = pn.get("ids", {}).get(project_id, {})
    return sorted(pid.get("variants", {}).keys())


def get_custom_cameras(model_type: str, project_name: str,
                       project_id: str, variant: str) -> list:
    data = load()
    pn = _get_hierarchy(data).get(model_type, {}).get(project_name, {})
    pid = pn.get("ids", {}).get(project_id, {})
    var = pid.get("variants", {}).get(variant, {})
    return sorted(var.get("cameras", []))
