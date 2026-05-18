"""
mlops/training/config_builder.py

Pure Python — no PyQt5 imports.
Provides config validation, metric line parsing, and pre-flight checks
for the training pipeline.
"""

import os
import json


# ---------------------------------------------------------------------------
# build_training_config
# ---------------------------------------------------------------------------

def build_training_config(form: dict) -> dict:
    """
    Validate the training form and return a structured config dict.
    Raises ValueError with a human-readable message on any validation failure.
    """
    # --- required string fields ---
    project = str(form.get("project", "")).strip()
    if not project:
        raise ValueError("Project name is required.")

    category = str(form.get("category", "")).strip()

    pretrained_weights = str(form.get("pretrained_weights", "")).strip()
    augmentations      = dict(form.get("augmentations", {}))

    commit_message = str(form.get("commit_message", "")).strip()
    if not commit_message:
        raise ValueError("Commit message is required.")

    annotator_initials = str(form.get("annotator_initials", "")).strip()
    if not annotator_initials:
        raise ValueError("Annotator initials are required.")

    annotator_name = str(form.get("annotator_name", "")).strip()

    # --- dataset folder ---
    dataset_folder = str(form.get("dataset_folder", "")).strip()
    if not dataset_folder or not os.path.isdir(dataset_folder):
        raise ValueError(f"Dataset folder does not exist: {dataset_folder!r}")

    info_path = os.path.join(dataset_folder, "dataset_info.json")
    if not os.path.isfile(info_path):
        raise ValueError(f"dataset_info.json not found in: {dataset_folder}")

    with open(info_path, "r", encoding="utf-8") as fh:
        dataset_info_raw = json.load(fh)

    # form["task_type"] (from radio button) takes precedence over dataset_info.json
    task_type   = str(form.get("task_type", dataset_info_raw.get("task_type", "unet")))
    train_count = int(dataset_info_raw.get("train_count", 0))
    val_count   = int(dataset_info_raw.get("val_count", 0))
    test_count  = int(dataset_info_raw.get("test_count", 0))
    class_names = dataset_info_raw.get("class_names", [])
    num_classes = int(dataset_info_raw.get("num_classes", 0))
    sha256      = dataset_info_raw.get("sha256", "")

    # --- hyperparams ---
    epochs = int(form.get("epochs", 100))
    if not (1 <= epochs <= 1000):
        raise ValueError(f"Epochs must be between 1 and 1000 (got {epochs}).")

    batch_size = int(form.get("batch_size", 8))
    if not (1 <= batch_size <= 128):
        raise ValueError(f"Batch size must be between 1 and 128 (got {batch_size}).")

    learning_rate = float(form.get("learning_rate", 0.001))
    if learning_rate <= 0:
        raise ValueError(f"Learning rate must be greater than 0 (got {learning_rate}).")

    image_width = int(form.get("image_width", 640))
    if image_width <= 0:
        raise ValueError(f"Image width must be > 0 (got {image_width}).")

    image_height = int(form.get("image_height", 512))
    if image_height <= 0:
        raise ValueError(f"Image height must be > 0 (got {image_height}).")

    architecture    = str(form.get("architecture", "Unet"))
    device          = str(form.get("device", "cpu"))
    encoder         = str(form.get("encoder", "")) if task_type == "unet" else ""
    encoder_weights = str(form.get("encoder_weights", "imagenet")) if task_type == "unet" else ""
    in_channels     = int(form.get("in_channels", 3))
    # out_classes from form overrides dataset num_classes (user may need to adjust)
    out_classes     = int(form.get("out_classes", num_classes or 2))
    num_workers     = int(form.get("num_workers", 0))
    repeat_factor   = int(form.get("repeat_factor", 2))

    # Issue 14 fix: Validate architecture and encoder names upfront
    VALID_UNET_ARCHS = {"Unet", "UnetPlusPlus", "DeepLabV3", "DeepLabV3Plus", "FPN", "PSPNet", "PAN", "Linknet"}
    VALID_YOLO_ARCHS = {"yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"}
    if task_type == "unet" and architecture not in VALID_UNET_ARCHS:
        raise ValueError(f"Unknown UNet architecture '{architecture}'. Valid: {sorted(VALID_UNET_ARCHS)}")
    if task_type == "yolo" and architecture not in VALID_YOLO_ARCHS:
        raise ValueError(f"Unknown YOLO architecture '{architecture}'. Valid: {sorted(VALID_YOLO_ARCHS)}")

    return {
        "model_type":     task_type,
        "annotator":      annotator_initials,
        "annotator_name": annotator_name,
        "project":             project,
        "category":            category,
        "pretrained_weights":  pretrained_weights,
        "augmentations":       augmentations,
        "commit_message":      commit_message,
        "dataset_folder": os.path.abspath(dataset_folder),
        "dataset_info": {
            "train_count":   train_count,
            "val_count":     val_count,
            "test_count":    test_count,
            "class_names":   class_names,
            "num_classes":   num_classes,
            "sha256":        sha256,
            "absolute_path": os.path.abspath(dataset_folder),
            "relative_path": "dataset",
        },
        "hyperparams": {
            "architecture":    architecture,
            "encoder":         encoder,
            "encoder_weights": encoder_weights,
            "in_channels":     in_channels,
            "out_classes":     out_classes,
            "epochs":          epochs,
            "batch_size":      batch_size,
            "image_width":     image_width,
            "image_height":    image_height,
            "learning_rate":   learning_rate,
            "device":          device,
            "num_workers":     num_workers,
            "repeat_factor":   repeat_factor,
        },
    }


# ---------------------------------------------------------------------------
# parse_metric_line
# ---------------------------------------------------------------------------

def parse_metric_line(line: str) -> dict | None:
    """
    Parse a METRIC line emitted by a training script.

    Expected format:
      METRIC epoch=5 train_loss=0.3421 val_loss=0.2890 train_iou=0.85 val_iou=0.82

    Returns a dict with parsed values (missing keys default to None).
    Returns None if the line does not start with 'METRIC '.
    """
    if not line.startswith("METRIC "):
        return None

    result = {"epoch": None, "train_loss": None, "val_loss": None,
              "train_iou": None, "val_iou": None,
              "train_per_image_iou": None, "val_per_image_iou": None,
              "map50": None}
    tokens = line[len("METRIC "):].split()
    for token in tokens:
        if "=" not in token:
            continue
        key, _, raw_val = token.partition("=")
        try:
            if key == "epoch":
                result[key] = int(raw_val)
            else:
                result[key] = float(raw_val)
        except ValueError:
            result[key] = None
    return result


# ---------------------------------------------------------------------------
# run_preflight
# ---------------------------------------------------------------------------

def run_preflight(config: dict, registry_root: str, scripts_dir: str) -> list:
    """
    Run four pre-flight checks and return a list of result dicts.

    Each result: {"name": str, "ok": bool, "warn": bool, "msg": str}

    warn=True items are shown as warnings (yellow) — not hard blockers.
    ok=False, warn=False items are hard blockers for the Start button.
    """
    results = []

    # 1. Dataset loaded
    train_count = config.get("dataset_info", {}).get("train_count", 0)
    val_count   = config.get("dataset_info", {}).get("val_count", 0)
    if train_count > 0:
        results.append({
            "name": "Dataset loaded",
            "ok": True,
            "warn": False,
            "msg": f"{train_count} train / {val_count} val",
        })
    else:
        results.append({
            "name": "Dataset loaded",
            "ok": False,
            "warn": False,
            "msg": "No pairs found",
        })

    # 2. Registry configured
    if registry_root and os.path.isdir(registry_root):
        results.append({
            "name": "Registry configured",
            "ok": True,
            "warn": False,
            "msg": registry_root,
        })
    else:
        results.append({
            "name": "Registry configured",
            "ok": False,
            "warn": False,
            "msg": "GDrive root not set or not found",
        })

    # 3. Training script found
    model_type  = config.get("model_type", "unet")
    script_name = f"train_{model_type}.py"
    script_path = os.path.join(scripts_dir, script_name)
    if os.path.isfile(script_path):
        results.append({
            "name": "Training script found",
            "ok": True,
            "warn": False,
            "msg": script_path,
        })
    else:
        results.append({
            "name": "Training script found",
            "ok": False,
            "warn": False,
            "msg": f"{script_path} not found",
        })

    # 4. GPU available
    try:
        import torch
        if torch.cuda.is_available():
            results.append({
                "name": "GPU available",
                "ok": True,
                "warn": False,
                "msg": "CUDA available",
            })
        else:
            results.append({
                "name": "GPU available",
                "ok": False,
                "warn": True,
                "msg": "No CUDA — will train on CPU",
            })
    except Exception:
        results.append({
            "name": "GPU available",
            "ok": False,
            "warn": True,
            "msg": "torch not installed or broken",
        })

    return results
