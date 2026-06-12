# -*- mode: python ; coding: utf-8 -*-
# DualAnnotator.spec — PyInstaller build specification
# ─────────────────────────────────────────────────────
# Build command:
#   pyinstaller DualAnnotator.spec
#
# torch/torchvision are NOT bundled — training/export/inference run via
# an external venv Python (see find_python() in mlops/utils.py). torch is
# only installed in CI so PyInstaller can trace imports from packages
# like segmentation_models_pytorch.
# ─────────────────────────────────────────────────────

import os
import sys
from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

# ── Collect packages ──────────────────────────────────────────
# collect_all grabs data files, binaries (DLLs), and hidden
# imports for each package.

packages_to_collect = [
    "ultralytics",
    "segmentation_models_pytorch",
    "albumentations",
    "pytorch_lightning",
    "torchmetrics",
    "onnxruntime",
]

all_datas = []
all_binaries = []
all_hiddenimports = []

for pkg in packages_to_collect:
    try:
        d, b, h = collect_all(pkg)
        all_datas += d
        all_binaries += b
        all_hiddenimports += h
    except Exception as e:
        print(f"[WARN] collect_all('{pkg}') failed: {e}")

# ── Analysis ──────────────────────────────────────────────────
a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=all_binaries,
    datas=[
        # Full mlops package (scripts import from mlops.registry, mlops.training etc.)
        ("mlops", "mlops"),
        *all_datas,
    ],
    hiddenimports=[
        # Explicit hidden imports that collect_all sometimes misses
        "pytorch_lightning",
        "pytorch_lightning.callbacks",
        "pytorch_lightning.loggers",
        "torchmetrics",
        "torchmetrics.functional",
        "segmentation_models_pytorch",
        "albumentations",
        "onnx",
        "onnxruntime",
        "cv2",
        "PIL",
        "yaml",
        "matplotlib",
        "matplotlib.backends.backend_agg",
        "numpy",
        *all_hiddenimports,
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # torch is not bundled — training runs via an external venv
        "torch",
        "torchvision",
        # These cause import errors and aren't needed at runtime
        "onnx.reference",
        "onnxscript",
    ],
    noarchive=False,
    optimize=0,
)

# ── Strip stray torch/torchvision/torchaudio files ─────────────
# `excludes` only removes them from the Python module graph, but
# PyInstaller's binary dependency walker still pulls in their DLLs
# and data files via other packages (e.g. onnxruntime, segmentation
# models). Drop anything destined for a torch* folder so the exe
# stays small and never touches c10.dll / torch_cpu.dll.
_TORCH_PREFIXES = ("torch", "torchvision", "torchaudio")


def _is_torch_entry(entry):
    dest = entry[0].lower().replace("\\", "/")
    return dest.split("/", 1)[0].startswith(_TORCH_PREFIXES)


a.binaries = [x for x in a.binaries if not _is_torch_entry(x)]
a.datas = [x for x in a.datas if not _is_torch_entry(x)]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DualAnnotator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="DualAnnotator",
)
