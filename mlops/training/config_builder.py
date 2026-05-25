"""
mlops/training/config_builder.py

Pure Python — no PyQt5 imports.
Provides config validation, metric line parsing, and pre-flight checks
for the training pipeline.
"""

import os
import json

from mlops.registry.utils import sanitize_path_component


# ---------------------------------------------------------------------------
# build_training_config
# ---------------------------------------------------------------------------

def build_training_config(form: dict) -> dict:
    """
    Validate the training form and return a structured config dict.
    Raises ValueError with a human-readable message on any validation failure.
    """
    # --- hierarchy fields ---
    project_name = str(form.get("project_name", "")).strip()
    project_id   = str(form.get("project_id", "")).strip()
    variant      = str(form.get("variant", "")).strip()
    camera       = str(form.get("camera", "")).strip()

    if not project_name:
        raise ValueError("Project Name is required.")
    if not project_id:
        raise ValueError("Project ID is required.")
    if not variant:
        raise ValueError("Variant is required.")
    if not camera:
        raise ValueError("Camera is required.")

    pretrained_weights = str(form.get("pretrained_weights", "")).strip()
    augmentations      = dict(form.get("augmentations", {}))

    commit_message = str(form.get("commit_message", "")).strip()
    if not commit_message:
        raise ValueError("Commit message is required.")

    annotator_initials = str(form.get("annotator_initials", "")).strip()
    if not annotator_initials:
        raise ValueError("Annotator initials are required.")

    annotator_name = str(form.get("annotator_name", "")).strip()

    gdrive_root = str(form.get("gdrive_root", "")).strip()

    # --- dataset folder (staged dataset) ---
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

    # Build the forward-slash project key used in manifest + version folder path.
    # Sanitize each component so user-facing names like "Snipe MX/Cx" don't create
    # unintended nested directories on disk.
    project = "/".join([
        task_type,
        sanitize_path_component(project_name),
        sanitize_path_component(project_id),
        sanitize_path_component(variant),
        sanitize_path_component(camera),
    ])

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
    num_workers             = int(form.get("num_workers", 0))
    repeat_factor           = int(form.get("repeat_factor", 2))
    early_stopping_patience = int(form.get("early_stopping_patience", 15))
    if not (1 <= early_stopping_patience <= 200):
        raise ValueError(f"Early stopping patience must be between 1 and 200 (got {early_stopping_patience}).")

    _valid_metrics = {"precision", "recall", "f1"}
    extra_metrics  = [m for m in list(form.get("extra_metrics", [])) if m in _valid_metrics]

    # --- YOLO advanced params (ignored for UNet, passed through for YOLO) ---
    _valid_optimizers = {"AdamW", "Adam", "SGD", "auto"}
    yolo_optimizer    = str(form.get("yolo_optimizer", "AdamW"))
    if yolo_optimizer not in _valid_optimizers:
        yolo_optimizer = "AdamW"
    yolo_lrf           = float(form.get("yolo_lrf",           0.01))
    yolo_momentum      = float(form.get("yolo_momentum",      0.937))
    yolo_weight_decay  = float(form.get("yolo_weight_decay",  0.0005))
    yolo_warmup_epochs = float(form.get("yolo_warmup_epochs", 5.0))
    yolo_cos_lr        = bool(form.get("yolo_cos_lr",         False))
    yolo_box           = float(form.get("yolo_box",           7.5))
    yolo_cls           = float(form.get("yolo_cls",           3.0))
    yolo_dfl           = float(form.get("yolo_dfl",           1.5))
    yolo_dropout       = float(form.get("yolo_dropout",       0.1))
    yolo_overlap_mask  = bool(form.get("yolo_overlap_mask",   True))
    yolo_mask_ratio    = int(form.get("yolo_mask_ratio",      4))
    yolo_hsv_h         = float(form.get("yolo_hsv_h",         0.01))
    yolo_hsv_s         = float(form.get("yolo_hsv_s",         0.30))
    yolo_hsv_v         = float(form.get("yolo_hsv_v",         0.20))
    yolo_fliplr        = float(form.get("yolo_fliplr",        0.5))
    yolo_flipud        = float(form.get("yolo_flipud",        0.0))
    yolo_degrees       = float(form.get("yolo_degrees",       5.0))
    yolo_translate     = float(form.get("yolo_translate",     0.05))
    yolo_scale         = float(form.get("yolo_scale",         0.25))
    yolo_mosaic        = float(form.get("yolo_mosaic",        0.3))
    yolo_mixup         = float(form.get("yolo_mixup",         0.0))
    yolo_copy_paste    = float(form.get("yolo_copy_paste",    0.1))
    yolo_close_mosaic  = int(form.get("yolo_close_mosaic",    20))
    yolo_amp           = bool(form.get("yolo_amp",            True))
    yolo_cache         = bool(form.get("yolo_cache",          False))
    yolo_save_period   = int(form.get("yolo_save_period",     10))
    yolo_plots         = bool(form.get("yolo_plots",          True))

    # Issue 14 fix: Validate architecture and encoder names upfront
    VALID_UNET_ARCHS = {"Unet", "UnetPlusPlus", "MAnet", "DeepLabV3", "DeepLabV3Plus", "FPN", "PSPNet", "PAN", "Linknet"}
    VALID_YOLO_ARCHS = {
        "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
        "yolov8n-seg", "yolov8s-seg", "yolov8m-seg", "yolov8l-seg", "yolov8x-seg",
    }
    if task_type == "unet" and architecture not in VALID_UNET_ARCHS:
        raise ValueError(f"Unknown UNet architecture '{architecture}'. Valid: {sorted(VALID_UNET_ARCHS)}")
    if task_type == "yolo" and architecture not in VALID_YOLO_ARCHS:
        raise ValueError(f"Unknown YOLO architecture '{architecture}'. Valid: {sorted(VALID_YOLO_ARCHS)}")

    return {
        "model_type":     task_type,
        "annotator":      annotator_initials,
        "annotator_name": annotator_name,
        "project":        project,
        "project_name":   project_name,
        "project_id":     project_id,
        "variant":        variant,
        "camera":         camera,
        "gdrive_root":    gdrive_root,
        "pretrained_weights":  pretrained_weights,
        "augmentations":       augmentations,
        "commit_message":      commit_message,
        # dataset_folder is the staged path; train_worker will copy it into the version folder
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
            "num_workers":              num_workers,
            "repeat_factor":            repeat_factor,
            "early_stopping_patience":  early_stopping_patience,
            "extra_metrics":            extra_metrics,
            # YOLO advanced
            "yolo_optimizer":     yolo_optimizer,
            "yolo_lrf":           yolo_lrf,
            "yolo_momentum":      yolo_momentum,
            "yolo_weight_decay":  yolo_weight_decay,
            "yolo_warmup_epochs": yolo_warmup_epochs,
            "yolo_cos_lr":        yolo_cos_lr,
            "yolo_box":           yolo_box,
            "yolo_cls":           yolo_cls,
            "yolo_dfl":           yolo_dfl,
            "yolo_dropout":       yolo_dropout,
            "yolo_overlap_mask":  yolo_overlap_mask,
            "yolo_mask_ratio":    yolo_mask_ratio,
            "yolo_hsv_h":         yolo_hsv_h,
            "yolo_hsv_s":         yolo_hsv_s,
            "yolo_hsv_v":         yolo_hsv_v,
            "yolo_fliplr":        yolo_fliplr,
            "yolo_flipud":        yolo_flipud,
            "yolo_degrees":       yolo_degrees,
            "yolo_translate":     yolo_translate,
            "yolo_scale":         yolo_scale,
            "yolo_mosaic":        yolo_mosaic,
            "yolo_mixup":         yolo_mixup,
            "yolo_copy_paste":    yolo_copy_paste,
            "yolo_close_mosaic":  yolo_close_mosaic,
            "yolo_amp":           yolo_amp,
            "yolo_cache":         yolo_cache,
            "yolo_save_period":   yolo_save_period,
            "yolo_plots":         yolo_plots,
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

    result = {
        "epoch": None, "train_loss": None, "val_loss": None,
        "train_iou": None, "val_iou": None,
        "train_per_image_iou": None, "val_per_image_iou": None,
        "map50": None,
        "train_precision": None, "val_precision": None,
        "train_per_image_precision": None, "val_per_image_precision": None,
        "train_recall": None, "val_recall": None,
        "train_per_image_recall": None, "val_per_image_recall": None,
        "train_f1": None, "val_f1": None,
        "train_per_image_f1": None, "val_per_image_f1": None,
        "learning_rate": None,
        "mask_map50": None,
    }
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

    # 5. YOLO label validation (YOLO only)
    if model_type == "yolo":
        results.append(_check_yolo_labels(config))

    return results


def _check_yolo_labels(config: dict) -> dict:
    """Scan YOLO label files: check class indices vs nc, and polygon format for seg."""
    dataset_folder = config.get("dataset_folder", "")
    hp             = config.get("hyperparams", {})
    arch           = str(hp.get("architecture", ""))
    is_seg         = arch.endswith("-seg")

    # Read nc from data.yaml
    yaml_path = os.path.join(dataset_folder, "data.yaml")
    nc = 1
    try:
        with open(yaml_path, "r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped.startswith("nc:"):
                    nc = int(stripped.split(":", 1)[1].strip())
                    break
    except Exception:
        return {
            "name": "YOLO labels",
            "ok": False,
            "warn": True,
            "msg": f"data.yaml not found or unreadable: {yaml_path}",
        }

    # Find labels/train directory
    candidates = [
        os.path.join(dataset_folder, "labels", "train"),
        os.path.join(dataset_folder, "train", "labels"),
    ]
    labels_dir = next((d for d in candidates if os.path.isdir(d)), None)
    if labels_dir is None:
        return {
            "name": "YOLO labels",
            "ok": False,
            "warn": True,
            "msg": "labels/train directory not found",
        }

    txt_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
    if not txt_files:
        return {
            "name": "YOLO labels",
            "ok": False,
            "warn": True,
            "msg": "No .txt label files found in labels/train",
        }

    max_class  = -1
    has_polygon = False
    for fname in txt_files[:100]:
        path = os.path.join(labels_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        cls_idx = int(float(parts[0]))
                    except (ValueError, IndexError):
                        continue
                    max_class = max(max_class, cls_idx)
                    if len(parts) > 5:
                        has_polygon = True
        except OSError:
            continue

    if max_class < 0:
        return {
            "name": "YOLO labels",
            "ok": False,
            "warn": True,
            "msg": "Label files appear empty",
        }

    if max_class >= nc:
        return {
            "name": "YOLO labels",
            "ok": False,
            "warn": False,
            "msg": (
                f"Class index {max_class} found but nc={nc} "
                f"(valid: 0–{nc-1}). Re-export dataset as 'YOLOv8' format."
            ),
        }

    if is_seg and not has_polygon:
        return {
            "name": "YOLO labels",
            "ok": False,
            "warn": True,
            "msg": (
                f"Arch ends in '-seg' but labels have bbox format only (5 values). "
                f"Re-export as 'YOLOv8 Segmentation' to train masks."
            ),
        }

    label_type = "polygon" if has_polygon else "bbox"
    return {
        "name": "YOLO labels",
        "ok": True,
        "warn": False,
        "msg": f"{len(txt_files)} files, nc={nc}, max_class={max_class}, format={label_type}",
    }
