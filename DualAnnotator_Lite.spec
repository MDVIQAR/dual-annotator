# -*- mode: python ; coding: utf-8 -*-
# DualAnnotator_Lite.spec — Annotation-only build (no ML dependencies)
#
# Build command:
#   pyinstaller DualAnnotator_Lite.spec
#
# Requires the `_lite_mode` marker file to exist in the project root
# (created automatically by build.yml, or manually for local builds):
#   New-Item -ItemType File -Path "_lite_mode" -Force

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=[
        # Marker file that tells the app to run in Lite mode
        ("_lite_mode", "."),
    ],
    hiddenimports=[
        "cv2",
        "PIL",
        "yaml",
        "numpy",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude ALL ML packages — not needed in Lite
        "torch",
        "torchvision",
        "ultralytics",
        "pytorch_lightning",
        "lightning",
        "lightning_fabric",
        "torchmetrics",
        "segmentation_models_pytorch",
        "onnx",
        "onnxscript",
        "onnxruntime",
        "albumentations",
        "matplotlib",
        # Exclude the mlops package entirely — Settings (the only Lite tab
        # that used mlops.registry) now lives in the Full build only.
        "mlops",
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
    name="DualAnnotator_Lite",
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
    name="DualAnnotator_Lite",
)
