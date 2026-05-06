"""
mlops/training/train_worker.py

QThread worker that:
  1. Creates a version folder + manifest.json in the registry
  2. Writes config.json to the version folder
  3. Launches the bundled training script as a subprocess
  4. Parses stdout for metric/progress lines and emits Qt signals
  5. On completion writes final manifest fields; on failure writes crash_info
"""

import os
import json
import subprocess
import sys

from PyQt5.QtCore import QThread, pyqtSignal

from mlops.registry.manifest import generate_version_id, ManifestWriter
from mlops.registry.utils import create_version_folder
from mlops.training.config_builder import parse_metric_line


class TrainWorker(QThread):
    """Background thread that manages the full training lifecycle."""

    log      = pyqtSignal(str)              # one text line for the console
    progress = pyqtSignal(int)              # 0–100
    metric   = pyqtSignal(int, float, float)  # epoch, train_loss, val_loss
    finished = pyqtSignal(bool, str)        # success, version_id

    def __init__(self, config: dict, registry_root: str, scripts_dir: str, parent=None):
        super().__init__(parent)
        self._config        = config
        self._registry_root = registry_root
        self._scripts_dir   = scripts_dir
        self._cancelled     = False
        self._proc          = None

    # ------------------------------------------------------------------
    # QThread entry point
    # ------------------------------------------------------------------

    def run(self):
        try:
            version_id = self._execute()
            self.finished.emit(True, version_id)
        except Exception as exc:
            self.log.emit(f"[ERROR] {exc}")
            self.finished.emit(False, "")

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _execute(self) -> str:
        # Step 1 — Generate version ID and create folder
        version_id     = generate_version_id(self._config["annotator"])
        project        = self._config["project"]
        version_folder = create_version_folder(self._registry_root, project, version_id)

        self.log.emit(f"[INFO] Version: {version_id}")
        self.log.emit(f"[INFO] Folder : {version_folder}")

        # Step 2 — Write config.json atomically
        config_json_path = os.path.join(version_folder, "config.json")
        _write_atomic(config_json_path, self._config)

        # Step 3 — Create manifest
        writer = ManifestWriter(version_folder)
        writer.create(
            version_id     = version_id,
            model_type     = self._config["model_type"],
            annotator      = self._config["annotator"],
            project        = project,
            commit_message = self._config["commit_message"],
            dataset_info   = self._config["dataset_info"],
            hyperparams    = self._config["hyperparams"],
        )
        self.log.emit("[INFO] Manifest created.")
        self.progress.emit(5)

        # Step 4 — Determine script path and launch subprocess
        model_type  = self._config["model_type"]
        script_path = os.path.join(self._scripts_dir, f"train_{model_type}.py")

        self.log.emit(f"[INFO] Launching: {script_path}")
        cmd = [sys.executable, script_path, "--config", config_json_path, "--out", version_folder]

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        # Step 5 — Read stdout line by line
        total_epochs = self._config["hyperparams"]["epochs"]
        for raw_line in self._proc.stdout:
            if self._cancelled:
                self._proc.terminate()
                break
            line = raw_line.rstrip()
            parsed = parse_metric_line(line)
            if parsed and parsed.get("epoch") and parsed.get("train_loss") and parsed.get("val_loss"):
                self.metric.emit(
                    int(parsed["epoch"]),
                    float(parsed["train_loss"]),
                    float(parsed["val_loss"]),
                )
                pct = int(parsed["epoch"] / total_epochs * 90)
                self.progress.emit(min(pct, 90))
            elif line.startswith("PROGRESS "):
                try:
                    self.progress.emit(int(line.split()[1]))
                except (IndexError, ValueError):
                    pass
            else:
                self.log.emit(line)

        self._proc.wait()

        # Step 6 — Handle outcome
        if self._cancelled:
            writer.mark_cancelled()
            self.log.emit("[CANCELLED] Training was stopped by the user.")
            raise RuntimeError("cancelled")
        elif self._proc.returncode != 0:
            writer.mark_crashed(
                reason=f"Script exited with code {self._proc.returncode}",
                hint="Check the console for details.",
            )
            raise RuntimeError(f"Training script failed (exit {self._proc.returncode})")
        else:
            self.progress.emit(100)
            writer.finalize()
            self.log.emit(f"[DONE] Version {version_id} saved to registry.")
            return version_id

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def cancel(self):
        """Signal the worker to stop after the current stdout line."""
        self._cancelled = True
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_atomic(path: str, data: dict):
    """Write data as JSON atomically using a .tmp intermediary."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, path)
