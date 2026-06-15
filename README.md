# Dual Annotator

A desktop application for annotating images and training YOLO / U-Net models — all in one place.
Built with Python and PyQt5.

---

## What It Does

Dual Annotator covers the full ML pipeline from raw images to a tested model:

1. **Convert** raw images (RAW → PNG)
2. **Annotate** with 8 shape types across YOLO and U-Net modes
3. **Prepare** a train/val/test dataset split
4. **Train** a YOLO or U-Net model
5. **Export** the trained model to ONNX and run inference on test images
6. **Registry** to track all trained model versions
7. **Collaborate** with teammates over a shared server

---

## Documentation

Full documentation with tutorials and GIF walkthroughs:
[dualannotator-docs.netlify.app](https://dualannotator-docs.netlify.app)

---

## Requirements

- Python 3.11.x (required — other versions are untested)
- Windows 10/11 (primary), macOS, Linux
- Git

---

## Installation

### 1. Install Python 3.11.9

This project requires **Python 3.11.9** exactly (other versions are untested or unsupported).

Download: [python.org/downloads/release/python-3119](https://www.python.org/downloads/release/python-3119/)

After installing, verify:

```bash
python --version
# Expected: Python 3.11.9
```

### 2. Clone the repository

```bash
git clone https://github.com/MDVIQAR/dual-annotator.git
cd dual-annotator
```

### 3. Create a virtual environment with Python 3.11

**Windows:**

```bash
py -3.11 -m venv venv
```

**macOS / Linux:**

```bash
python3.11 -m venv venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run

```bash
python main.py
```

---

## Building the .exe from source

### Prerequisites

- Python 3.11.x

### Steps

```powershell
git clone https://github.com/MDVIQAR/dual-annotator.git
cd dual-annotator

# Create venv and install dependencies
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
# Install torch for PyInstaller import tracing (NOT bundled into the exe)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pyinstaller==6.11.1

# Build (torch is not bundled into the exe — see "Setting up training" below)
pyinstaller DualAnnotator.spec
```

The exe will be at `dist\DualAnnotator\DualAnnotator.exe`.

### Setting up training

When you click **Start Training** for the first time, the app automatically
creates a virtual environment and installs all ML packages (~2 GB download,
takes 5-10 minutes). This only happens once.

**Prerequisite:** Python 3.11 must be installed on the system with
"Add Python to PATH" checked during installation.

Download: [python.org/downloads](https://www.python.org/downloads/)

> The GUI works immediately without Python installed. Python is only
> needed when you use training, inference, or ONNX export features.

---

## Lite Build (annotation only)

A standalone annotation-only exe is also available — no Python or setup needed.
Download **DualAnnotator-Lite.zip** from the
[Releases](https://github.com/MDVIQAR/dual-annotator/releases) page.

Includes: Annotate, RAW → PNG, Coin Rotator, Collaborate.
Does not include: Train, Data Prep, Export & Test, Registry, Settings.

---

## Tabs Overview

| Tab | What it does |
| --- | ------------ |
| **RAW → PNG** | Batch convert RAW camera files to PNG with a preview gallery |
| **Annotate** | Draw and manage annotations on images |
| **Data Prep** | Build train/val/test splits and generate `dataset_info.json` |
| **Train** | Configure hyperparameters and train YOLO or U-Net models |
| **Export Test** | Export a trained model to ONNX and run inference on test images |
| **Registry** | Browse all trained model versions with metrics and metadata |
| **Coin Rotator** | Specialized tool for rotational image augmentation |
| **Settings** | App-wide preferences |
| **Collaborate** | Connect to a shared server to annotate with teammates |

---

## Annotation

### Modes

- **YOLO mode** — bounding box annotation for object detection
- **UNet mode** — segmentation annotation with solid and hollow shapes

### Shape Types

| Key | Shape |
| --- | ----- |
| `B` | Bounding Box |
| `P` | Polygon |
| `Q` | Bezier Polygon |
| `C` | Circle |
| `E` | Ellipse |
| `F` | Frame (hollow rectangle) |
| `O` | Donut (hollow circle) |
| `H` | Hollow Ellipse |

### Hollow Shapes

Right-click any shape → **Create Inner Shape** or **Create Outer Shape**.
A thickness slider gives a live preview before confirming.

### Templates

Right-click a shape → **Save as Template** to store it as a reusable stamp.
Select a template from the dropdown and click anywhere to place it.

### Editing

- **Resize** — drag any handle
- **Move** — drag the interior
- **Nudge** — arrow keys (configurable step size)
- **Clone** — Ctrl + drag
- **Copy / Paste** — Ctrl+C / Ctrl+V (pastes at cursor position)
- **Multi-select** — rubber band drag, then move or copy the group
- **Undo / Redo** — Ctrl+Z / Ctrl+Y (50 steps)

### Navigation

- **A / D** — previous / next image
- **Tab** — jump to next unannotated image
- **Scroll wheel** — zoom in/out
- **Middle-click drag or Space** — pan
- **Ctrl+0** — fit image to screen

### Autosave

Every change saves automatically. There is no Save button.
Annotations are stored in a hidden `.dualannotator/` folder inside the image directory — they travel with the images if you move the folder.

```text
your_images/
├── image_001.jpg
├── image_002.jpg
└── .dualannotator/
    ├── project.json
    └── annotations/
        ├── image_001.jpg.json
        └── image_002.jpg.json
```

---

## Export (Annotations)

**File → Export Annotations… (Ctrl+E)**

Supported formats:

- **YOLO** — `.txt` label files + `data.yaml` + `classes.txt`
- **COCO JSON**
- **Pascal VOC XML**

Features:

- Configurable train / val / test split with fixed random seed
- Delta export — only re-exports images that changed since last export
- Annotated image copies with shapes drawn on them

YOLO output structure:

```text
exports/yolo_2026-06-05/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── annotated/
├── data.yaml
└── classes.txt
```

---

## Training Pipeline

1. **Data Prep tab** — point to your images and labels folders, set the split, click **Prepare Dataset**. Or use **Import Existing Dataset** if you already have a split.
2. **Train tab** — pick YOLO or U-Net, set hyperparameters (epochs, batch size, learning rate, image size, etc.), run pre-flight checks, then click **Start Training**.
3. Training metrics (loss, IoU / mAP) plot live as training runs.
4. When training finishes, click **Export to ONNX** to produce a cross-platform model file.

---

## Export Test (Inference)

Drop a trained version folder onto the Export Test tab (or browse to it).
The tab auto-fills the model path (`best.pt`) and test images folder.
Click **Run Inference** — results appear in a zoomable gallery.

- Both `.pt` (PyTorch) and `.onnx` are accepted for inference.
- ONNX models are exported at **opset 11** (IR version 6) for broad runtime compatibility.

---

## Registry

All trained versions are listed with their metrics, hyperparameters, training date, and model type.
You can compare versions and open any version folder directly.

---

## Collaboration

Connect to a shared annotation server so multiple users can annotate the same dataset.
The Collaborate tab handles connection, sync, and conflict resolution.

---

## Keyboard Shortcuts

| Key | Action |
| --- | ------ |
| `B` | Bounding Box tool |
| `P` | Polygon tool |
| `Q` | Bezier Polygon tool |
| `C` | Circle tool |
| `E` | Ellipse tool |
| `F` | Frame tool |
| `O` | Donut tool |
| `H` | Hollow Ellipse tool |
| `A` / `D` | Previous / next image |
| `Tab` | Next unannotated image |
| `Space` | Toggle pan mode |
| `Ctrl+0` | Fit image to screen |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |
| `Ctrl+C` | Copy selected shape / group |
| `Ctrl+V` | Paste at cursor |
| `Ctrl+A` | Select all shapes |
| `Ctrl+Drag` | Clone shape |
| `Del` | Delete selected shape / group |
| `Arrow keys` | Nudge selected shape |
| `Ctrl+E` | Export annotations |
| `Enter` | Finish polygon / bezier drawing |
| `Esc` | Cancel drawing |

---

## Project Structure

```text
DualAnnotator/
├── main.py                        ← entry point
├── requirements.txt
├── core/
│   ├── annotation.py              ← BoundingBox
│   ├── bezier_shape.py            ← BezierPolygonShape
│   ├── circle_shape.py            ← CircleShape
│   ├── class_manager.py           ← ClassCategory, ClassManager
│   ├── ellipse_shape.py           ← EllipseShape
│   ├── hollow_ops.py              ← geometry offset functions
│   ├── polygon_shape.py           ← PolygonShape
│   ├── project_manager.py         ← autosave, file I/O, session management
│   ├── ring_shape.py              ← FrameShape, DonutShape, HollowEllipseShape
│   ├── shape_base.py              ← abstract Shape base class
│   ├── template_manager.py        ← stamp template storage
│   ├── export_manager.py          ← export orchestrator
│   └── exporters/
│       ├── yolo_exporter.py       ← YOLO .txt + data.yaml + classes.txt
│       ├── coco_exporter.py       ← COCO JSON
│       └── voc_exporter.py        ← Pascal VOC XML
├── gui/
│   ├── canvas.py                  ← drawing canvas
│   ├── class_panel.py             ← class list UI
│   ├── context_menu.py            ← right-click shape menu
│   ├── main_window.py             ← main application window
│   ├── shape_toolbar.py           ← left toolbar
│   └── tabs/
│       ├── raw_to_png_tab.py
│       ├── data_prep_tab.py
│       ├── training_tab.py
│       ├── export_test_tab.py
│       ├── registry_tab.py
│       ├── collab_tab.py
│       ├── coin_rotator_tab.py
│       └── settings_tab.py
├── mlops/
│   ├── utils.py                   ← find_python() for venv delegation
│   ├── engine_manager.py          ← auto venv setup
│   ├── training/
│   │   ├── train_worker.py        ← training subprocess manager
│   │   └── config_builder.py      ← hyperparameter validation
│   ├── export/
│   │   ├── onnx_worker.py
│   │   └── infer_worker.py
│   └── scripts/
│       ├── train_unet.py
│       ├── train_yolo.py
│       ├── seg_model.py
│       ├── dataloader_unet.py
│       ├── export_onnx.py
│       ├── infer_unet.py
│       └── infer_yolo.py
├── DualAnnotator.spec             ← Full build spec
└── DualAnnotator_Lite.spec        ← Lite build spec
```

---

## License

MIT
