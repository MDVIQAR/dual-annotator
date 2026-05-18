"""
mlops/training/train_worker.py

QThread worker that:
  1. Creates a version folder + manifest.json in the registry
  2. Writes config.json to the version folder
  3. Launches the bundled training script as a subprocess
  4. Parses stdout for metric/progress lines and emits Qt signals
  5. On completion writes final manifest fields; on failure writes crash_info

Cancel behaviour:
  Writes a 'stop_requested' sentinel file into the version folder so the
  training script can finish its current epoch gracefully before exiting.
  If the script exits with code 0 after a cancel request the run is marked
  'partial' (usable for fine-tuning).  A 5-minute grace window is given;
  if the script has not exited by then it is force-killed.
"""

import os
import json
import shutil
import subprocess
import sys

from PyQt5.QtCore import QThread, pyqtSignal

from mlops.registry.manifest import generate_version_id, ManifestWriter
from mlops.registry.utils import create_version_folder
from mlops.training.config_builder import parse_metric_line

_GRACEFUL_TIMEOUT = 300   # seconds to wait for graceful stop before force-kill


class TrainWorker(QThread):
    """Background thread that manages the full training lifecycle."""

    log      = pyqtSignal(str)                    # one text line for the console
    progress = pyqtSignal(int)                    # 0–100
    metric   = pyqtSignal(int, float, float, float, float, float, float)
    # (success, version_id, local_run_dir, is_partial)
    finished = pyqtSignal(bool, str, str, bool)

    def __init__(self, config: dict, registry_root: str, scripts_dir: str, parent=None):
        super().__init__(parent)
        self._config          = config
        self._registry_root   = registry_root
        self._scripts_dir     = scripts_dir
        self._cancelled       = False
        self._proc            = None
        self._final_metrics   = {}
        self._version_folder  = ""   # set in _execute(); needed by cancel()

    # ------------------------------------------------------------------
    # QThread entry point
    # ------------------------------------------------------------------

    def run(self):
        try:
            version_id, local_run_dir, is_partial = self._execute()
            self.finished.emit(True, version_id, local_run_dir, is_partial)
        except Exception as exc:
            self.log.emit(f"[ERROR] {exc}")
            self.finished.emit(False, "", "", False)

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _execute(self) -> tuple:
        # Step 1 — Generate version ID and create folder
        version_id            = generate_version_id(self._config["annotator"])
        project               = self._config["project"]
        self._version_folder  = create_version_folder(self._registry_root, project, version_id)

        self.log.emit(f"[INFO] Version: {version_id}")
        self.log.emit(f"[INFO] Folder : {self._version_folder}")

        # Step 2 — Write config.json atomically
        config_json_path = os.path.join(self._version_folder, "config.json")
        _write_atomic(config_json_path, self._config)

        # Step 3 — Create manifest
        writer = ManifestWriter(self._version_folder)
        writer.create(
            version_id     = version_id,
            model_type     = self._config["model_type"],
            annotator      = self._config["annotator"],
            project        = project,
            commit_message = self._config["commit_message"],
            dataset_info   = self._config["dataset_info"],
            hyperparams    = self._config["hyperparams"],
            augmentations  = self._config.get("augmentations", {}),
        )
        self.log.emit("[INFO] Manifest created.")
        self.progress.emit(5)

        # Step 4 — Launch subprocess
        model_type  = self._config["model_type"]
        script_path = os.path.join(self._scripts_dir, f"train_{model_type}.py")

        self.log.emit(f"[INFO] Launching: {script_path}")
        cmd = [sys.executable, script_path, "--config", config_json_path, "--out", self._version_folder]

        import pathlib
        project_root = str(pathlib.Path(__file__).resolve().parents[2])
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )

        # Step 5 — Read stdout line by line
        # When cancelled we do NOT terminate here — the stop_requested sentinel
        # file was already written by cancel().  We keep draining stdout so the
        # script can finish its current epoch and exit cleanly.
        total_epochs = self._config["hyperparams"]["epochs"]
        for raw_line in self._proc.stdout:
            line = raw_line.rstrip()
            parsed = parse_metric_line(line)
            if (parsed is not None
                    and parsed.get("epoch") is not None
                    and parsed.get("train_loss") is not None
                    and parsed.get("val_loss") is not None):
                epoch       = int(parsed["epoch"])
                t_loss      = float(parsed["train_loss"])
                v_loss      = float(parsed["val_loss"])
                t_iou       = float(parsed.get("train_iou")           or 0.0)
                v_iou       = float(parsed.get("val_iou") or parsed.get("map50") or 0.0)
                t_per_iou   = float(parsed.get("train_per_image_iou") or 0.0)
                v_per_iou   = float(parsed.get("val_per_image_iou")   or 0.0)

                self.metric.emit(epoch, t_loss, v_loss, t_iou, v_iou, t_per_iou, v_per_iou)
                self.progress.emit(min(int(epoch / total_epochs * 90), 90))
                self.log.emit(
                    f"Epoch {epoch}/{total_epochs}  |  "
                    f"train_loss: {t_loss:.4f}  val_loss: {v_loss:.4f}  |  "
                    f"train_iou: {t_iou:.4f}  val_iou: {v_iou:.4f}  |  "
                    f"train_per_img: {t_per_iou:.4f}  val_per_img: {v_per_iou:.4f}"
                )
            elif line.startswith("PROGRESS "):
                try:
                    self.progress.emit(int(line.split()[1]))
                except (IndexError, ValueError):
                    pass
            elif line.startswith("METRICS_FINAL "):
                tokens = line[len("METRICS_FINAL "):].split()
                for tok in tokens:
                    if "=" in tok:
                        k, _, v = tok.partition("=")
                        try:
                            self._final_metrics[k] = float(v)
                        except ValueError:
                            pass
            else:
                self.log.emit(line)

        # Step 6 — Wait for the process to exit
        # Give up to _GRACEFUL_TIMEOUT seconds for a graceful stop; force-kill after.
        try:
            self._proc.wait(timeout=_GRACEFUL_TIMEOUT)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self.log.emit("[WARN] Training did not stop in time — force killed.")

        # Clean up sentinel file regardless of outcome
        _remove_sentinel(self._version_folder)

        # Step 7 — Handle outcome
        exit_ok = (self._proc.returncode == 0)

        if self._cancelled and exit_ok:
            # Graceful partial stop — at least one epoch completed, best.pt saved
            if self._final_metrics:
                writer.update_metrics(
                    best_val_loss    = self._final_metrics.get("best_val_loss", 0.0),
                    best_epoch       = int(self._final_metrics.get("best_epoch", 0)),
                    final_train_loss = self._final_metrics.get("final_train_loss", 0.0),
                )
            if os.path.isfile(os.path.join(self._version_folder, "best.pt")):
                writer.update_artifacts(weights="best.pt")
            writer.mark_partial()
            self.log.emit("[PARTIAL] Training stopped early — checkpoint preserved.")
            local_run_dir = self._copy_to_local(version_id)
            return version_id, local_run_dir, True

        elif self._cancelled:
            # Hard cancel (script crashed or was force-killed)
            writer.mark_cancelled()
            self.log.emit("[CANCELLED] Training was stopped by the user.")
            raise RuntimeError("cancelled")

        elif not exit_ok:
            writer.mark_crashed(
                reason=f"Script exited with code {self._proc.returncode}",
                hint="Check the console for details.",
            )
            raise RuntimeError(f"Training script failed (exit {self._proc.returncode})")

        else:
            # Full success
            if self._final_metrics:
                writer.update_metrics(
                    best_val_loss    = self._final_metrics.get("best_val_loss", 0.0),
                    best_epoch       = int(self._final_metrics.get("best_epoch", 0)),
                    final_train_loss = self._final_metrics.get("final_train_loss", 0.0),
                )
            writer.update_artifacts(weights="best.pt")
            self.progress.emit(100)
            writer.finalize()
            try:
                from mlops.registry.utils import write_project_csv
                write_project_csv(self._registry_root, project)
            except Exception:
                pass
            self.log.emit(f"[DONE] Version {version_id} saved to registry.")
            local_run_dir = self._copy_to_local(version_id)
            return version_id, local_run_dir, False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _copy_to_local(self, version_id: str) -> str:
        """Copy the version folder into {dataset_folder}/runs/{version_id}/."""
        dataset_folder = self._config.get("dataset_folder", "")
        if not dataset_folder or not os.path.isdir(dataset_folder):
            return ""
        local_run_dir = os.path.join(dataset_folder, "runs", version_id)
        try:
            shutil.copytree(self._version_folder, local_run_dir)
            self.log.emit(f"[INFO] Local copy saved: {local_run_dir}")
            return local_run_dir
        except Exception as copy_err:
            self.log.emit(f"[WARN] Local copy failed: {copy_err}")
            return ""

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def cancel(self):
        """
        Request a graceful stop by writing a sentinel file the training script
        checks at the end of each epoch.  Falls back to hard terminate if the
        version folder is not yet known.
        """
        self._cancelled = True
        if self._proc and self._proc.poll() is None:
            if self._version_folder:
                stop_file = os.path.join(self._version_folder, "stop_requested")
                try:
                    open(stop_file, "w").close()
                    self.log.emit("[INFO] Stop requested — finishing current epoch before stopping...")
                    return
                except Exception:
                    pass
            # Fallback: hard terminate
            self._proc.terminate()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_atomic(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, path)


def _remove_sentinel(version_folder: str):
    stop_file = os.path.join(version_folder, "stop_requested")
    try:
        if os.path.isfile(stop_file):
            os.remove(stop_file)
    except Exception:
        pass
