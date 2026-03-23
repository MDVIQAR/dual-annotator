# Dual Annotator

A desktop annotation tool for building YOLO object detection and U-Net segmentation datasets.
Built with Python and PyQt5.

## Features
# Annotation:

YOLO mode — bounding box annotation for object detection
UNet mode — segmentation annotation with solid and hollow shapes
8 shape types — box, polygon, bezier curve, circle, ellipse, frame (hollow rect), donut (hollow circle), hollow ellipse
Hollow shapes — create inner/outer offset shapes via right-click context menu with live preview
Multi-class support — colour-coded classes with add/edit/delete
Template system — save any shape as a reusable stamp template

Editing:

Copy, paste, and resize shapes
Ctrl+drag to clone a shape
Undo / redo (up to 50 steps)
Pan (middle-click drag or Space) and zoom (scroll wheel)
Right-click context menu on any shape

Project & Save:

Open any image folder as a project
Autosave — every annotation change is saved automatically (800ms debounce)
Resume previous projects — annotations persist across sessions
Per-image status tracking (unannotated / in progress / annotated / skipped)
Image hash validation — warns if source image changes after annotation

Export (YOLO):

Export annotated images as YOLO .txt label files
Generates data.yaml and classes.txt automatically
Train / val / test split with configurable percentages and fixed random seed
Delta export — only re-exports images that changed since the last export
Copies source images into the correct split folders
Saves annotated image copies with bounding boxes drawn on them
Additional formats: COCO JSON, Pascal VOC XML

## Installation
Requirements

Python 3.9 or higher
Windows, macOS, or Linux

# 1. Clone the repository
git clone https://github.com/yourusername/DualAnnotator.git
cd DualAnnotator

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run
python main.py

DualAnnotator/
├── main.py                        ← entry point
├── requirements.txt
├── README.md
├── .gitignore
├── core/
│   ├── annotation.py              ← BoundingBox (YOLO bbox)
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
└── gui/
    ├── canvas.py                  ← drawing canvas (AnnotationCanvas)
    ├── class_panel.py             ← class list UI
    ├── context_menu.py            ← right-click shape menu
    ├── export_dialog.py           ← export settings dialog
    ├── export_summary_dialog.py   ← post-export summary
    ├── main_window.py             ← main application window
    ├── mode_switch_dialog.py      ← YOLO ↔ UNet mode switch dialog
    ├── shape_toolbar.py           ← left toolbar
    └── thickness_dialog.py        ← hollow shape offset slider

How It Works
Annotation workflow

File → Open Image Folder — select a folder of images
Select a class from the right panel (or add one with the + button)
Select a shape tool from the left toolbar
Draw on the canvas
Annotations are saved automatically — no save button needed

Hollow shapes
Draw any shape, then right-click it and choose Create Inner Shape or
Create Outer Shape. A slider lets you set the offset with a live preview.
For YOLO export, hollow shapes export as the outer bounding box only.
Exporting

File → Export Annotations... (Ctrl+E)
Choose format, output folder, and split percentages
Click Export — runs in background, progress bar updates live
A summary shows counts, split distribution, and opens the output folder

Save file location
Annotations are saved in a hidden .dualannotator/ folder inside your image
directory. This folder travels with your images if you move the folder.
your_images/
├── image_001.jpg
├── image_002.jpg
└── .dualannotator/
    ├── project.json
    └── annotations/
        ├── image_001.jpg.json
        └── image_002.jpg.json

Export output structure
your_images/exports/yolo_2026-03-14_18-44/
├── labels/
│   ├── train/   image_001.txt ...
│   ├── val/     image_007.txt ...
│   └── test/    image_009.txt ...
├── images/
│   ├── train/   image_001.jpg ...
│   ├── val/     ...
│   └── test/    ...
├── annotated/   image_001.jpg (with boxes drawn)
├── data.yaml
└── classes.txt


Keyboard Shortcuts


YOLO Label Format:
Each .txt file contains one line per annotation:
<class_index> <cx_norm> <cy_norm> <w_norm> <h_norm>

All values are normalised to [0.0, 1.0]. For all shape types, the export
uses the bounding box of the shape. Hollow shapes export the outer boundary
bounding box only.

Example data.yaml:
path: /path/to/exports/yolo_2026-03-14/
train: images/train
val: images/val
test: images/test
nc: 3
names:
  - seal
  - cap
  - packet

  