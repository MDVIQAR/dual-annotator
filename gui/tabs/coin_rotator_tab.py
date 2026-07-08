"""
gui/tabs/coin_rotator_tab.py

Coin Year Auto-Rotation tab for DualAnnotator.

Phase 1 — pick any coin, rotate with a slider until the year faces up, confirm.
Phase 2 — batch SIFT + FLANN feature matching warps every coin to the reference
           orientation via estimateAffinePartial2D + warpAffine.
"""

import os
import glob
import json

import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QLineEdit, QPushButton, QSlider, QSpinBox, QTextEdit,
    QProgressBar, QFileDialog, QMessageBox, QGroupBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont, QPixmap, QImage
import logging
import traceback

# ──────────────────────────────────────────────────────────────────────────────
# CV2 pipeline constants
# ──────────────────────────────────────────────────────────────────────────────

SIFT_FEATURES     = 8000
FLANN_INDEX_KDTREE = 1
FLANN_TREES       = 5
FLANN_CHECKS      = 200
LOWE_RATIO        = 0.70
MIN_GOOD_MATCHES  = 10
RANSAC_THRESHOLD  = 5.0
RANSAC_MAX_ITERS  = 5000
RANSAC_CONFIDENCE = 0.99
MIN_INLIER_RATIO  = 0.10   # coins have low ratio due to repetitive circular patterns
MIN_ABS_INLIERS   = 6      # absolute floor — affine needs 3, 6 gives confidence
ROT_SEARCH_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]

# ──────────────────────────────────────────────────────────────────────────────
# Feature pipeline (pure CV2 — unchanged from original)
# ──────────────────────────────────────────────────────────────────────────────

def _build_sift():
    try:
        det = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
        det.detect(np.zeros((64, 64), dtype=np.uint8))
        return det, "SIFT"
    except Exception:
        pass
    return cv2.ORB_create(nfeatures=SIFT_FEATURES), "ORB"


_DETECTOR, _DETECTOR_NAME = _build_sift()


def _preprocess(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    b = cv2.medianBlur(gray, 5)
    return cv2.adaptiveThreshold(
        b, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )


def _compute_features(gray):
    return _DETECTOR.detectAndCompute(gray, None)


def _build_flann():
    if _DETECTOR_NAME == "SIFT":
        ip = dict(algorithm=FLANN_INDEX_KDTREE, trees=FLANN_TREES)
        sp = dict(checks=FLANN_CHECKS)
    else:
        ip = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        sp = dict(checks=50)
    return cv2.FlannBasedMatcher(ip, sp)


def _match_features(des_ref, des_tgt):
    if des_ref is None or des_tgt is None:
        return []
    if len(des_ref) < 2 or len(des_tgt) < 2:
        return []
    try:
        raw = _build_flann().knnMatch(des_ref, des_tgt, k=2)
    except cv2.error:
        return []
    good = []
    for pair in raw:
        if len(pair) == 2:
            m, n = pair
            if m.distance < LOWE_RATIO * n.distance:
                good.append(m)
    return good


def _try_match(ref_kp, ref_des, tgt_kp, tgt_des, min_matches):
    good = _match_features(ref_des, tgt_des)
    if len(good) < min_matches:
        return None, 0, 0.0, len(good)

    src_pts = np.float32([tgt_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.estimateAffinePartial2D(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESHOLD,
        maxIters=RANSAC_MAX_ITERS,
        confidence=RANSAC_CONFIDENCE,
    )
    if M is None:
        return None, 0, 0.0, len(good)

    n_in = int(mask.sum()) if mask is not None else 0
    return M, n_in, n_in / max(len(good), 1), len(good)


def _rotate_gray(gray, angle_deg):
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def _process_one(ref_gray, ref_kp, ref_des, ref_shape, img_path, out_dir, min_matches):
    record = dict(file=os.path.basename(img_path), success=False,
                  ref_kp=len(ref_kp), tgt_kp=0,
                  good_matches=0, inliers=0, inlier_ratio=None,
                  search_angle=None, message="")

    tgt_img = cv2.imread(img_path)
    if tgt_img is None:
        record["message"] = "Cannot read image"
        return record

    tgt_gray = _preprocess(tgt_img)

    best_M = best_inliers = best_good = best_angle = best_kp = 0
    best_ratio = 0.0

    for angle in ROT_SEARCH_ANGLES:
        g = _rotate_gray(tgt_gray, angle) if angle != 0 else tgt_gray
        kp, des = _compute_features(g)
        if des is None or len(kp) < 2:
            continue

        M, n_in, ratio, n_good = _try_match(ref_kp, ref_des, kp, des, min_matches)
        if M is not None and n_in > best_inliers:
            if angle != 0:
                h, w = tgt_img.shape[:2]
                R = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
                R3 = np.vstack([R, [0, 0, 1]])
                M3 = np.vstack([M, [0, 0, 1]])
                M_combined = (M3 @ R3)[:2, :]
            else:
                M_combined = M

            best_M, best_inliers, best_ratio = M_combined, n_in, ratio
            best_good, best_angle, best_kp = n_good, angle, len(kp)

    record.update(tgt_kp=best_kp, good_matches=best_good,
                  inliers=best_inliers, search_angle=best_angle)

    if not isinstance(best_M, np.ndarray):
        record["message"] = f"No candidate rotation yielded ≥{min_matches} good matches"
        return record

    record["inlier_ratio"] = round(best_ratio, 4)
    if best_ratio < MIN_INLIER_RATIO and best_inliers < MIN_ABS_INLIERS:
        record["message"] = (f"Best inlier ratio {best_ratio:.2f} < {MIN_INLIER_RATIO} "
                             f"and inliers {best_inliers} < {MIN_ABS_INLIERS} "
                             f"(at pre-rot {best_angle}°)")
        return record

    h_ref, w_ref = ref_shape[:2]
    aligned = cv2.warpAffine(tgt_img, best_M, (w_ref, h_ref),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0))
    cv2.imwrite(os.path.join(out_dir, os.path.basename(img_path)), aligned)
    record["success"] = True
    record["message"] = f"OK  (pre-rot {best_angle}°)"
    return record


# ──────────────────────────────────────────────────────────────────────────────
# Batch worker (runs in QThread)
# ──────────────────────────────────────────────────────────────────────────────

class _BatchWorker(QThread):
    log_line      = pyqtSignal(str, str)   # (message, tag)
    progress      = pyqtSignal(int)        # 0-100
    status        = pyqtSignal(str)
    batch_finished = pyqtSignal(int, int, str)  # ok, skip, log_path

    def __init__(self, ref_img_bgr, coin_files, out_dir, min_matches):
        super().__init__()
        self._ref    = ref_img_bgr
        self._files  = coin_files
        self._out    = out_dir
        self._min_m  = min_matches

    def run(self):
        ref_gray = _preprocess(self._ref)
        ref_kp, ref_des = _compute_features(ref_gray)
        ref_shape = self._ref.shape

        self.log_line.emit(
            f"Reference  →  KP: {len(ref_kp)}  size: {ref_shape[1]}×{ref_shape[0]}", "info")
        self.log_line.emit("─" * 62, "dim")

        if ref_des is None or len(ref_kp) < 10:
            self.log_line.emit("Reference has too few keypoints!", "err")
            return

        os.makedirs(self._out, exist_ok=True)
        self.log_line.emit(f"Output  →  {self._out}", "info")
        self.log_line.emit(
            f"Matcher  →  FLANN KD-Tree · ratio={LOWE_RATIO} · "
            f"RANSAC={RANSAC_THRESHOLD}px · inlier≥{int(MIN_INLIER_RATIO*100)}%", "info")
        self.log_line.emit("─" * 62, "dim")

        n = len(self._files)
        ok = skip = 0
        log_data = []

        for i, fpath in enumerate(self._files):
            rec = _process_one(ref_gray, ref_kp, ref_des, ref_shape,
                               fpath, self._out, self._min_m)
            log_data.append(rec)
            bname = rec["file"]

            if rec["success"]:
                ok += 1
                tag = "ok"
                line = (f"[✓] {bname:35s}"
                        f"  kp={rec['tgt_kp']:4d}"
                        f"  good={rec['good_matches']:3d}"
                        f"  inliers={rec['inliers']:3d}/{rec['good_matches']:3d}"
                        f"  ({rec['inlier_ratio']:.0%})")
            else:
                skip += 1
                tag = "warn"
                line = f"[✗] {bname:35s}  good={rec['good_matches']:3d}  → {rec['message']}"

            self.log_line.emit(line, tag)
            self.progress.emit(int((i + 1) / n * 100))
            self.status.emit(f"Processing {i + 1}/{n} — {bname}")

        log_path = os.path.join(self._out, "rotation_log.json")
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        self.log_line.emit("─" * 62, "dim")
        self.log_line.emit(
            f"Done!  ✓ {ok} aligned   ✗ {skip} skipped   → {log_path}", "ok")
        self.batch_finished.emit(ok, skip, log_path)


# ──────────────────────────────────────────────────────────────────────────────
# Shared style helpers
# ──────────────────────────────────────────────────────────────────────────────

_PANEL_STYLE = """
QGroupBox {
    background-color: #252526;
    border: 1px solid #3a3a3a;
    border-radius: 6px;
    margin-top: 10px;
    font-weight: bold;
    color: #cba6f7;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
"""

_BTN_PRIMARY = """
QPushButton {
    background-color: #6c63ff;
    color: #ffffff;
    border: none;
    border-radius: 5px;
    padding: 7px 14px;
    font-weight: bold;
}
QPushButton:hover  { background-color: #7d75ff; }
QPushButton:pressed { background-color: #5a52e0; }
QPushButton:disabled { background-color: #3a3a3a; color: #666666; }
"""

_BTN_SUCCESS = """
QPushButton {
    background-color: #40a870;
    color: #ffffff;
    border: none;
    border-radius: 5px;
    padding: 7px 14px;
    font-weight: bold;
}
QPushButton:hover  { background-color: #4ec484; }
QPushButton:pressed { background-color: #318a58; }
QPushButton:disabled { background-color: #3a3a3a; color: #666666; }
"""

_BTN_SMALL = """
QPushButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 11px;
}
QPushButton:hover  { background-color: #45475a; }
QPushButton:pressed { background-color: #1e1e2e; }
"""

_SLIDER_STYLE = """
QSlider::groove:horizontal {
    height: 6px;
    background: #313244;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    width: 16px; height: 16px;
    margin: -5px 0;
    background: #6c63ff;
    border-radius: 8px;
}
QSlider::sub-page:horizontal {
    background: #6c63ff;
    border-radius: 3px;
}
"""

_PROGRESS_STYLE = """
QProgressBar {
    background-color: #313244;
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
    color: transparent;
}
QProgressBar::chunk {
    background-color: #6c63ff;
    border-radius: 4px;
}
"""


def _bgr_to_qpixmap(img_bgr, max_size=420):
    """Convert OpenCV BGR image to a QPixmap, scaled to max_size."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    pix = QPixmap.fromImage(qimg)
    return pix.scaled(max_size, max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)


# ──────────────────────────────────────────────────────────────────────────────
# Main tab widget
# ──────────────────────────────────────────────────────────────────────────────

class CoinRotatorTab(QWidget):

    CANVAS = 420   # preview canvas size (px)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #1e1e1e; color: #cdd6f4;")

        # State
        self._coin_files: list[str] = []
        self._coin_idx   = 0
        self._orig_img   = None      # current coin BGR
        self._rotation   = 0.0
        self._ref_bgr    = None      # confirmed reference BGR
        self._worker     = None

        self._setup_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 10, 14, 10)
        root.setSpacing(8)

        # Title
        title = QLabel("🪙  Coin Year Auto-Rotation")
        title.setFont(QFont("Inter", 15, QFont.Bold))
        title.setStyleSheet("color: #cba6f7;")
        title.setAlignment(Qt.AlignCenter)
        root.addWidget(title)

        sub = QLabel(f"SIFT {SIFT_FEATURES}ft · FLANN KD-Tree · estimateAffinePartial2D · Direct Warp")
        sub.setFont(QFont("Inter", 9))
        sub.setStyleSheet("color: #6c7086;")
        sub.setAlignment(Qt.AlignCenter)
        root.addWidget(sub)

        # Input folder row
        folder_row = QHBoxLayout()
        folder_row.addWidget(QLabel("Input folder:"))
        self._input_edit = self._line_edit("/path/to/coins")
        folder_row.addWidget(self._input_edit, 1)
        btn_browse = QPushButton("Browse")
        btn_browse.setStyleSheet(_BTN_PRIMARY)
        btn_browse.clicked.connect(self._browse_input)
        folder_row.addWidget(btn_browse)
        btn_load = QPushButton("Load Coins")
        btn_load.setStyleSheet(_BTN_PRIMARY)
        btn_load.clicked.connect(self._load_coins)
        folder_row.addWidget(btn_load)
        root.addLayout(folder_row)

        # Main splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet("QSplitter::handle { background: #3a3a3a; }")
        root.addWidget(splitter, 1)

        splitter.addWidget(self._build_phase1())
        splitter.addWidget(self._build_phase2())
        splitter.setSizes([500, 600])

    def _line_edit(self, placeholder=""):
        w = QLineEdit()
        w.setPlaceholderText(placeholder)
        w.setStyleSheet("""
            QLineEdit {
                background-color: #252526;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                color: #cdd6f4;
                padding: 4px 8px;
            }
        """)
        return w

    def _build_phase1(self):
        box = QGroupBox(" Phase 1 — Manual Reference Alignment ")
        box.setStyleSheet(_PANEL_STYLE)
        lay = QVBoxLayout(box)
        lay.setSpacing(6)

        # Navigation row
        nav = QHBoxLayout()
        btn_prev = QPushButton("◀ Prev")
        btn_prev.setStyleSheet(_BTN_SMALL)
        btn_prev.clicked.connect(self._prev_coin)
        nav.addWidget(btn_prev)

        self._coin_label = QLabel("No coins loaded")
        self._coin_label.setAlignment(Qt.AlignCenter)
        self._coin_label.setStyleSheet("color: #89b4fa; font-size: 11px;")
        nav.addWidget(self._coin_label, 1)

        btn_next = QPushButton("Next ▶")
        btn_next.setStyleSheet(_BTN_SMALL)
        btn_next.clicked.connect(self._next_coin)
        nav.addWidget(btn_next)
        lay.addLayout(nav)

        # Canvas
        self._canvas = QLabel()
        self._canvas.setFixedSize(self.CANVAS, self.CANVAS)
        self._canvas.setAlignment(Qt.AlignCenter)
        self._canvas.setStyleSheet(
            "background-color: #181825; border: 2px solid #313244; border-radius: 4px;")
        self._canvas.setText("Load coins → pick one → rotate\n\n↑  YEAR TOP  ↑")
        self._canvas.setFont(QFont("Inter", 10))
        lay.addWidget(self._canvas, alignment=Qt.AlignHCenter)

        # Rotation slider + value
        rot_row = QHBoxLayout()
        rot_row.addWidget(QLabel("Rotation:"))
        self._rot_label = QLabel("  0.0°")
        self._rot_label.setStyleSheet("color: #89dceb; font-weight: bold; min-width: 55px;")
        rot_row.addStretch()
        rot_row.addWidget(self._rot_label)
        lay.addLayout(rot_row)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(-1800, 1800)   # ×10 for 0.1° precision
        self._slider.setValue(0)
        self._slider.setStyleSheet(_SLIDER_STYLE)
        self._slider.valueChanged.connect(self._on_slider)
        lay.addWidget(self._slider)

        # Fine nudge buttons
        nudge_row = QHBoxLayout()
        nudge_row.setSpacing(4)
        for delta, lbl in [(-50, "–5°"), (-10, "–1°"), (-1, "–0.1°"),
                            (+1, "+0.1°"), (+10, "+1°"), (+50, "+5°")]:
            b = QPushButton(lbl)
            b.setStyleSheet(_BTN_SMALL)
            b.setFixedWidth(58)
            b.clicked.connect(lambda _, d=delta: self._nudge(d))
            nudge_row.addWidget(b)
        lay.addLayout(nudge_row)

        hint = QLabel("↑  Rotate until the year text faces this arrow  ↑")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet("color: #f9e2af; font-style: italic; font-size: 10px;")
        lay.addWidget(hint)

        self._confirm_btn = QPushButton("✅  Confirm Reference  →  Run Batch")
        self._confirm_btn.setStyleSheet(_BTN_SUCCESS)
        self._confirm_btn.setEnabled(False)
        self._confirm_btn.clicked.connect(self._confirm_reference)
        lay.addWidget(self._confirm_btn)

        return box

    def _build_phase2(self):
        box = QGroupBox(" Phase 2 — Batch FLANN Matching ")
        box.setStyleSheet(_PANEL_STYLE)
        lay = QVBoxLayout(box)
        lay.setSpacing(6)

        # Output folder
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output folder:"))
        self._out_edit = self._line_edit("/path/to/coins_rotated")
        out_row.addWidget(self._out_edit, 1)
        btn_out = QPushButton("Browse")
        btn_out.setStyleSheet(_BTN_SMALL)
        btn_out.clicked.connect(self._browse_output)
        out_row.addWidget(btn_out)
        lay.addLayout(out_row)

        # Min matches
        mm_row = QHBoxLayout()
        mm_row.addWidget(QLabel("Min good matches:"))
        self._min_matches = QSpinBox()
        self._min_matches.setRange(1, 500)
        self._min_matches.setValue(MIN_GOOD_MATCHES)
        self._min_matches.setStyleSheet(
            "background:#252526; border:1px solid #3a3a3a; color:#cdd6f4; padding:3px;")
        mm_row.addWidget(self._min_matches)
        mm_row.addStretch()
        lay.addLayout(mm_row)

        # Pipeline info
        info_box = QGroupBox(" Pipeline ")
        info_box.setStyleSheet(_PANEL_STYLE)
        info_lay = QVBoxLayout(info_box)
        for line in [
            f"Detector : {_DETECTOR_NAME}  ({SIFT_FEATURES} features)",
            f"Matcher  : FLANN KD-Tree (trees={FLANN_TREES}, checks={FLANN_CHECKS})",
            f"Ratio    : {LOWE_RATIO}   |  RANSAC thresh : {RANSAC_THRESHOLD}px",
            f"Inlier   : ≥ {int(MIN_INLIER_RATIO*100)}%   |  Warp: direct warpAffine(M)",
        ]:
            lbl = QLabel(line)
            lbl.setFont(QFont("Courier New", 8))
            lbl.setStyleSheet("color: #a6e3a1;")
            info_lay.addWidget(lbl)
        lay.addWidget(info_box)

        # Reference thumbnail
        ref_box = QGroupBox(" Confirmed Reference ")
        ref_box.setStyleSheet(_PANEL_STYLE)
        ref_lay = QVBoxLayout(ref_box)
        self._ref_canvas = QLabel("No reference yet")
        self._ref_canvas.setFixedSize(200, 200)
        self._ref_canvas.setAlignment(Qt.AlignCenter)
        self._ref_canvas.setStyleSheet(
            "background-color: #181825; border: 1px solid #313244; color: #6c7086;")
        ref_lay.addWidget(self._ref_canvas, alignment=Qt.AlignHCenter)
        self._ref_name_lbl = QLabel("(none)")
        self._ref_name_lbl.setAlignment(Qt.AlignCenter)
        self._ref_name_lbl.setStyleSheet("color: #6c7086; font-size: 9px;")
        ref_lay.addWidget(self._ref_name_lbl)
        lay.addWidget(ref_box)

        # Progress
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setStyleSheet(_PROGRESS_STYLE)
        self._progress.setFixedHeight(10)
        lay.addWidget(self._progress)

        # Status label
        self._status_lbl = QLabel("Step 1: Load coins → rotate year to ↑ → Confirm.")
        self._status_lbl.setStyleSheet("color: #89dceb; font-size: 10px;")
        self._status_lbl.setWordWrap(True)
        lay.addWidget(self._status_lbl)

        # Log
        log_box = QGroupBox(" Log ")
        log_box.setStyleSheet(_PANEL_STYLE)
        log_lay = QVBoxLayout(log_box)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Courier New", 9))
        self._log.setStyleSheet(
            "background-color: #181825; border: none; color: #cdd6f4;")
        log_lay.addWidget(self._log)
        lay.addWidget(log_box, 1)

        return box

    # ── File ops ──────────────────────────────────────────────────────────────

    def _browse_input(self):
        try:
            d = QFileDialog.getExistingDirectory(self, "Select input folder",
                                                 self._input_edit.text())
            if d:
                self._input_edit.setText(d)
                self._out_edit.setText(os.path.join(d, "coins_rotated"))

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def _browse_output(self):
        try:
            d = QFileDialog.getExistingDirectory(self, "Select output folder",
                                                 self._out_edit.text())
            if d:
                self._out_edit.setText(d)

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def _load_coins(self):
        try:
            inp = self._input_edit.text().strip()
            if not os.path.isdir(inp):
                QMessageBox.warning(self, "Error", "Invalid input folder.")
                return
            exts = ["*.bmp", "*.png", "*.jpg", "*.jpeg", "*.tiff", "*.tif"]
            files = []
            for e in exts:
                files += glob.glob(os.path.join(inp, e))
                files += glob.glob(os.path.join(inp, e.upper()))
            files = sorted(set(
                f for f in files
                if "coins_rotated" not in f
                and "reference_aligned" not in os.path.basename(f)
            ))
            if not files:
                QMessageBox.warning(self, "No images", "No images found in that folder.")
                return
            self._coin_files = files
            self._coin_idx   = 0
            self._rotation   = 0.0
            self._slider.setValue(0)
            self._show_current_coin()
            self._status_lbl.setText(
                f"Loaded {len(files)} images.  Rotate so year faces ↑, then Confirm.")

        # ── Coin navigation ────────────────────────────────────────────────────────

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def _prev_coin(self):
        try:
            if not self._coin_files:
                return
            self._coin_idx = (self._coin_idx - 1) % len(self._coin_files)
            self._rotation = 0.0
            self._slider.blockSignals(True)
            self._slider.setValue(0)
            self._slider.blockSignals(False)
            self._show_current_coin()

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def _next_coin(self):
        try:
            if not self._coin_files:
                return
            self._coin_idx = (self._coin_idx + 1) % len(self._coin_files)
            self._rotation = 0.0
            self._slider.blockSignals(True)
            self._slider.setValue(0)
            self._slider.blockSignals(False)
            self._show_current_coin()

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def _show_current_coin(self):
        path = self._coin_files[self._coin_idx]
        self._orig_img = cv2.imread(path)
        if self._orig_img is None:
            self._coin_label.setText(f"Cannot read: {os.path.basename(path)}")
            return
        self._coin_label.setText(
            f"{os.path.basename(path)}  ({self._coin_idx + 1}/{len(self._coin_files)})")
        self._confirm_btn.setEnabled(True)
        self._draw_rotated()

    # ── Rotation controls ──────────────────────────────────────────────────────

    def _on_slider(self, val):
        self._rotation = val / 10.0      # slider stores ×10 for sub-degree precision
        self._rot_label.setText(f"{self._rotation:+.1f}°")
        self._draw_rotated()

    def _nudge(self, delta_10th):
        new_val = max(-1800, min(1800, self._slider.value() + delta_10th))
        self._slider.setValue(new_val)   # triggers _on_slider

    def _draw_rotated(self):
        if self._orig_img is None:
            return
        h, w = self._orig_img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), self._rotation, 1.0)
        rotated = cv2.warpAffine(self._orig_img, M, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))
        pix = _bgr_to_qpixmap(rotated, self.CANVAS)
        self._canvas.setPixmap(pix)

    # ── Confirm reference ──────────────────────────────────────────────────────

    def _confirm_reference(self):
        try:
            if self._orig_img is None:
                QMessageBox.warning(self, "Error", "No coin loaded.")
                return

            h, w = self._orig_img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), self._rotation, 1.0)
            ref_bgr = cv2.warpAffine(self._orig_img, M, (w, h),
                                      flags=cv2.INTER_LANCZOS4,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(0, 0, 0))

            save_path = os.path.join(self._input_edit.text(), "reference_aligned.bmp")
            cv2.imwrite(save_path, ref_bgr)
            self._ref_bgr = ref_bgr

            orig_name = os.path.basename(self._coin_files[self._coin_idx])
            self._log_line(f"Reference saved: reference_aligned.bmp", "ok")
            self._log_line(f"  from : {orig_name}  rotation = {self._rotation:+.1f}°", "dim")
            self._log_line(f"  path : {save_path}", "dim")

            # Update reference thumbnail
            pix = _bgr_to_qpixmap(ref_bgr, 200)
            self._ref_canvas.setPixmap(pix)
            self._ref_name_lbl.setText(f"{orig_name}  (rot={self._rotation:+.1f}°)")

            self._status_lbl.setText("Reference confirmed — starting batch…")
            self._start_batch()

        # ── Batch ─────────────────────────────────────────────────────────────────

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def _start_batch(self):
        if self._worker and self._worker.isRunning():
            return
        if self._ref_bgr is None or not self._coin_files:
            return

        out_dir = self._out_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "Error", "Please set an output folder.")
            return

        self._confirm_btn.setEnabled(False)
        self._progress.setValue(0)

        self._worker = _BatchWorker(
            self._ref_bgr, self._coin_files, out_dir, self._min_matches.value()
        )
        self._worker.log_line.connect(self._log_line)
        self._worker.progress.connect(self._progress.setValue)
        self._worker.status.connect(self._status_lbl.setText)
        self._worker.batch_finished.connect(self._on_batch_done)
        self._worker.start()

    @pyqtSlot(int, int, str)
    def _on_batch_done(self, ok, skip, log_path):
        self._confirm_btn.setEnabled(True)
        QMessageBox.information(
            self, "Complete",
            f"Batch finished!\n\n"
            f"✓ Aligned : {ok}\n"
            f"✗ Skipped : {skip}\n\n"
            f"Log saved to:\n{log_path}"
        )

    # ── Log helper ─────────────────────────────────────────────────────────────

    TAG_COLORS = {
        "ok":   "#a6e3a1",
        "warn": "#f9e2af",
        "err":  "#f38ba8",
        "info": "#89b4fa",
        "dim":  "#6c7086",
    }

    @pyqtSlot(str, str)
    def _log_line(self, msg: str, tag: str = "info"):
        color = self.TAG_COLORS.get(tag, "#cdd6f4")
        self._log.append(f'<span style="color:{color}; font-family:monospace;">'
                         f'{msg.replace(" ", "&nbsp;")}</span>')
        self._log.verticalScrollBar().setValue(
            self._log.verticalScrollBar().maximum()
        )
