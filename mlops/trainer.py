# mlops/trainer.py
# Phase 2: Training Orchestration via QThread + Subprocess
#
# Design rationale:
#   YOLO training is launched as a SUBPROCESS (not an in-process import).
#   This prevents PyTorch from blocking the Qt event loop, allows live stdout
#   streaming, and means the training can be cancelled cleanly by killing
#   the subprocess — without crashing the parent app.
#
# Output folder structure (YOLO default):
#   <version_folder>/run/
#       weights/   best.pt, last.pt
#       results.csv
#       results.png
#
# Phase 3 (registry_manager) will scan this 'run/' folder and archive
# best.pt, best.onnx, results.png into the version root.

import os
import sys
import json
import subprocess
import tempfile
import re
from datetime import datetime
from PyQt5.QtCore import QThread, pyqtSignal


# ---------------------------------------------------------------------------
# Training configuration dataclass (plain dict for JSON-serialisability)
# ---------------------------------------------------------------------------

def build_training_config(
    *,
    task_type:       str,   # "detect" | "segment"
    base_model:      str,   # e.g. "yolov8n.pt", "yolov8n-seg.pt"
    data_yaml:       str,   # absolute path to data.yaml
    version_folder:  str,   # registry version folder (e.g. …/YOLO/Project/v1)
    epochs:          int,
    batch:           int,
    imgsz:           int,
    learning_rate:   float,
    annotator_name:  str,
    commit_message:  str,
    dataset_hash:    str    = "",
) -> dict:
    """Build a serialisable training config dict."""
    return {
        "task_type":      task_type,
        "base_model":     base_model,
        "data_yaml":      data_yaml,
        "version_folder": version_folder,
        "run_name":       "run",                     # YOLO saves to version_folder/run/
        "epochs":         epochs,
        "batch":          batch,
        "imgsz":          imgsz,
        "lr0":            learning_rate,
        "annotator":      annotator_name,
        "commit_message": commit_message,
        "dataset_hash":   dataset_hash,
        "started_at":     datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Training Worker
# ---------------------------------------------------------------------------

# ── Inline training script ──────────────────────────────────────────────────
# This string is written to a temp .py file and executed as a subprocess.
# Kept as a module-level constant so PyInstaller can detect it statically.
_TRAIN_SCRIPT = """
import sys
import json

cfg_path = sys.argv[1]
with open(cfg_path, "r") as f:
    cfg = json.load(f)

from ultralytics import YOLO

model = YOLO(cfg["base_model"])
results = model.train(
    data      = cfg["data_yaml"],
    epochs    = cfg["epochs"],
    batch     = cfg["batch"],
    imgsz     = cfg["imgsz"],
    lr0       = cfg["lr0"],
    project   = cfg["version_folder"],
    name      = cfg["run_name"],
    exist_ok  = True,
    verbose   = True,
)
# Signal completion with the output directory so the parent can locate files
print("TRAIN_DONE::" + str(results.save_dir))
"""


class TrainingWorker(QThread):
    """
    Executes YOLO training in a subprocess and streams output to the UI.

    Signals:
        log(str)              — one line of training stdout/stderr
        progress(int)         — 0–100, parsed from epoch numbers
        epoch_update(int,int) — (current_epoch, total_epochs)
        finished(bool, str)   — (success, output_run_folder)
    """
    log          = pyqtSignal(str)
    progress     = pyqtSignal(int)
    epoch_update = pyqtSignal(int, int)
    finished     = pyqtSignal(bool, str)   # success, run_output_dir

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config   = config
        self._process = None
        self._running = True

    # -----------------------------------------------------------------------
    def run(self):
        success    = False
        output_dir = ""
        try:
            success, output_dir = self._execute_training()
        except Exception as exc:
            self.log.emit(f"[ERROR] Trainer exception: {exc}")
        finally:
            self.finished.emit(success, output_dir)

    def stop(self):
        """Request cancellation. Kills the subprocess if running."""
        self._running = False
        if self._process and self._process.poll() is None:
            self._process.kill()
            self.log.emit("\n[CANCELLED] Training stopped by user.")

    # -----------------------------------------------------------------------
    def _execute_training(self):
        cfg    = self.config
        epochs = cfg.get("epochs", 100)

        self.log.emit("=" * 52)
        self.log.emit(f"  Training started: {cfg.get('annotator', '')} — {cfg.get('commit_message', '')}")
        self.log.emit(f"  Model  : {cfg['base_model']}")
        self.log.emit(f"  Data   : {cfg['data_yaml']}")
        self.log.emit(f"  Epochs : {epochs}  |  Batch: {cfg['batch']}  |  Img: {cfg['imgsz']}px")
        self.log.emit(f"  Output : {cfg['version_folder']}/{cfg['run_name']}/")
        self.log.emit("=" * 52)
        self.progress.emit(2)

        # ── Write temp config JSON ─────────────────────────────────────
        tmp_dir  = tempfile.mkdtemp(prefix="dualannotator_train_")
        cfg_path = os.path.join(tmp_dir, "train_cfg.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        # ── Write temp training script ─────────────────────────────────
        script_path = os.path.join(tmp_dir, "run_training.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(_TRAIN_SCRIPT)

        # ── Launch subprocess ──────────────────────────────────────────
        cmd = [sys.executable, script_path, cfg_path]
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
            cwd=cfg.get("version_folder", "."),
        )

        output_dir    = ""
        success       = False
        epoch_pattern = re.compile(r"^\s*(\d+)/(\d+)\s")   # "  3/100   ..."

        # ── Stream output ──────────────────────────────────────────────
        for raw_line in self._process.stdout:
            if not self._running:
                break

            line = raw_line.rstrip()
            self.log.emit(line)

            # Parse epoch progress
            m = epoch_pattern.match(line)
            if m:
                current = int(m.group(1))
                total   = int(m.group(2))
                pct     = max(2, min(98, int(current / total * 96) + 2))
                self.progress.emit(pct)
                self.epoch_update.emit(current, total)

            # Capture output directory signal from training script
            if line.startswith("TRAIN_DONE::"):
                output_dir = line.split("::", 1)[1].strip()

        self._process.wait()
        returncode = self._process.returncode

        if returncode == 0 and self._running:
            success = True
            self.progress.emit(100)
            self.log.emit("\n" + "=" * 52)
            self.log.emit("  Training complete!")
            if output_dir:
                self.log.emit(f"  Results saved to: {output_dir}")
            self.log.emit("=" * 52)
        elif not self._running:
            pass  # cancelled — message already emitted by stop()
        else:
            self.log.emit(f"\n[ERROR] Training process exited with code {returncode}")

        return success, output_dir
