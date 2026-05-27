"""
mlops/registry/csv_exporter.py

Auto-generates and updates models_registry.csv inside the registry root.
Preserves manually-entered 'deployed' and 'deployed_date' values across regenerations.
"""

import csv
import os

from mlops.registry.scanner import RegistryScanner

_COLUMNS = [
    "version_id", "project", "version_count", "architecture", "commit_message",
    "encoder", "encoder_weights", "in_channels", "out_classes",
    "batch_size", "learning_rate", "image_width", "image_height", "device",
    "train_count", "val_count", "test_count", "num_classes", "class_names",
    "deployed", "deployed_date",
]

_CSV_FILENAME = "models_registry.csv"


def refresh_registry_csv(registry_root: str) -> int:
    """
    Scan registry_root and write/update models_registry.csv there.
    Existing 'deployed' and 'deployed_date' values are preserved by version_id.
    Returns the number of model rows written, or -1 if registry_root is invalid.
    """
    if not registry_root or not os.path.isdir(registry_root):
        return -1

    csv_path = os.path.join(registry_root, _CSV_FILENAME)

    # Load any existing deployed/deployed_date values so we don't wipe manual entries
    existing_deploy = {}
    if os.path.isfile(csv_path):
        try:
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    vid = row.get("version_id", "")
                    if vid:
                        existing_deploy[vid] = {
                            "deployed":      row.get("deployed", ""),
                            "deployed_date": row.get("deployed_date", ""),
                        }
        except Exception:
            pass

    scanner = RegistryScanner(registry_root)
    rows = []
    for proj in scanner.get_projects():
        versions = scanner.get_versions(proj)
        version_count = len(versions)
        for s in versions:
            m = scanner.get_version(proj, s["version_id"])
            if not m:
                continue
            hp = m.get("hyperparams", {})
            ds = m.get("dataset", {})
            vid = m.get("version_id", "")
            prev = existing_deploy.get(vid, {})
            rows.append({
                "version_id":      vid,
                "project":         m.get("project", ""),
                "version_count":   version_count,
                "architecture":    hp.get("architecture", ""),
                "commit_message":  m.get("commit_message", ""),
                "encoder":         hp.get("encoder", ""),
                "encoder_weights": hp.get("encoder_weights", ""),
                "in_channels":     hp.get("in_channels", ""),
                "out_classes":     hp.get("out_classes", ""),
                "batch_size":      hp.get("batch_size", ""),
                "learning_rate":   hp.get("learning_rate", ""),
                "image_width":     hp.get("image_width", ""),
                "image_height":    hp.get("image_height", ""),
                "device":          hp.get("device", ""),
                "train_count":     ds.get("train_count", ""),
                "val_count":       ds.get("val_count", ""),
                "test_count":      ds.get("test_count", ""),
                "num_classes":     ds.get("num_classes", ""),
                "class_names":     "|".join(ds.get("classes", [])),
                "deployed":        prev.get("deployed", ""),
                "deployed_date":   prev.get("deployed_date", ""),
            })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)
