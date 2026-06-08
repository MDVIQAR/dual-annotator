# -*- mode: python ; coding: utf-8 -*-
# DualAnnotator.spec — PyInstaller build specification
# ─────────────────────────────────────────────────────
# Build command:
#   pyinstaller DualAnnotator.spec
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
# imports for each package.  This is what makes torch's ~15
# DLLs land inside _internal/torch/lib/.

packages_to_collect = [
    "torch",
    "torchvision",
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
    binaries=[
        *all_binaries,
        # Force-include ALL torch DLLs — collect_all misses some (e.g. asmjit.dll, fbgemm.dll)
        ("torch_dlls/*.dll", "torch/lib"),
    ],
    datas=[
        # Training / inference scripts (run via runpy in-process)
        ("mlops/scripts/*.py", "mlops/scripts"),
        # Help page and all its assets
        ("docs", "docs"),
        *all_datas,
    ],
    hiddenimports=[
        # Explicit hidden imports that collect_all sometimes misses
        "torch",
        "torch.utils",
        "torch.utils.data",
        "torch._C",
        "torch.lib",
        "torchvision",
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
    # ── CRITICAL: runtime hook sets DLL paths before anything runs ──
    runtime_hooks=["hooks/rthook_torch_dlls.py"],
    excludes=[
        # These cause import errors and aren't needed at runtime
        "onnx.reference",
        "onnxscript",
    ],
    noarchive=False,
    optimize=0,
)

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
    upx=False,          # CRITICAL: never UPX torch DLLs — it corrupts them
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
    upx=False,           # CRITICAL: same reason — don't compress DLLs
    upx_exclude=[],
    name="DualAnnotator",
)
