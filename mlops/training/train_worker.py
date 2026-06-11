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
import re
import shutil
import subprocess
import sys

from mlops.registry import RegistrySettings

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]|\x1b\][^\x07]*\x07|\r')

from PyQt5.QtCore import QThread, pyqtSignal

from mlops.registry.manifest import generate_version_id, ManifestWriter
from mlops.registry.utils import create_version_folder
from mlops.training.config_builder import parse_metric_line
from mlops.utils import find_python

_GRACEFUL_TIMEOUT = 300   # seconds to wait for graceful stop before force-kill


class _TrainingDiagnostics:
    """Tracks metric history and emits training health diagnostics."""

    def __init__(self, total_epochs: int):
        self._total_epochs = total_epochs
        self._train_losses = []    # list of values
        self._val_losses = []
        self._train_ious = []
        self._val_ious = []
        self._best_val_loss = float("inf")
        self._epochs_since_improvement = 0
        self._val_rising_streak = 0
        self._train_dropping_streak = 0
        self._crossed_50_iou = False
        self._last_healthy_msg_epoch = 0

    def update(self, epoch, t_loss, v_loss, t_iou, v_iou):
        """Record metrics for this epoch."""
        self._train_losses.append(t_loss)
        self._val_losses.append(v_loss)
        self._train_ious.append(t_iou)
        self._val_ious.append(v_iou)

        # Track val loss improvement
        if v_loss < self._best_val_loss:
            self._best_val_loss = v_loss
            self._epochs_since_improvement = 0
        else:
            self._epochs_since_improvement += 1

        # Track val loss rising streak
        if len(self._val_losses) >= 2:
            if v_loss > self._val_losses[-2]:
                self._val_rising_streak += 1
            else:
                self._val_rising_streak = 0

        # Track train loss dropping streak
        if len(self._train_losses) >= 2:
            if t_loss < self._train_losses[-2]:
                self._train_dropping_streak += 1
            else:
                self._train_dropping_streak = 0

    def diagnose(self, epoch, t_loss, v_loss, t_iou, v_iou) -> list:
        """Return a list of diagnostic message strings for this epoch.
        Returns empty list if nothing noteworthy."""
        msgs = []

        # Skip first 3 epochs — metrics are noisy at the start
        if epoch <= 3:
            return msgs

        # ── Overfitting: val rising while train dropping ──
        if (self._val_rising_streak >= 3 and self._train_dropping_streak >= 3):
            msgs.append(
                "  ⚠ Possible overfitting — val_loss rising while train_loss "
                f"dropping ({self._val_rising_streak} epochs)"
            )
            msgs.append(
                "  💡 Try: increase augmentation, reduce model size, or add weight decay"
            )

        # ── Overfitting: large IoU gap ──
        elif epoch >= 5 and t_iou > 0.1 and v_iou > 0.1:
            gap = t_iou - v_iou
            if gap > 0.15:
                msgs.append(
                    f"  ⚠ Train/val IoU gap is large ({gap:.2f}) — model may be overfitting"
                )
                msgs.append(
                    "  💡 Try: more augmentation, dropout, or weight decay"
                )

        # ── Val loss spike ──
        if len(self._val_losses) >= 2:
            prev_val = self._val_losses[-2]
            if prev_val > 0 and v_loss > prev_val * 1.5:
                msgs.append(
                    f"  ⚠ Val loss spiked ({prev_val:.4f} → {v_loss:.4f}) "
                    "— possible data issue or learning rate too high"
                )

        # ── Underfitting: losses still high after 20% of training ──
        pct_done = epoch / self._total_epochs
        if pct_done >= 0.2 and t_loss > 0.7 and v_loss > 0.7 and not msgs:
            msgs.append(
                f"  ⚠ Possible underfitting — both losses still high at "
                f"{int(pct_done * 100)}% through training"
            )
            msgs.append(
                "  💡 Try: increase learning rate, train longer, or use a larger encoder"
            )

        # ── Stale training: no improvement for 5+ epochs ──
        if self._epochs_since_improvement >= 5 and not msgs:
            msgs.append(
                f"  ⚠ No val_loss improvement in {self._epochs_since_improvement} epochs "
                "— training may be stalling"
            )
            msgs.append(
                "  💡 Early stopping will trigger soon if this continues"
            )

        # ── Milestone: first time val IoU crosses 0.5 ──
        if not self._crossed_50_iou and v_iou >= 0.5:
            self._crossed_50_iou = True
            msgs.append("  ✅ Model is learning well — val IoU passed 0.50")

        # ── Healthy: periodic reassurance every 10 epochs ──
        if (not msgs
                and epoch >= 5
                and epoch - self._last_healthy_msg_epoch >= 10
                and self._epochs_since_improvement <= 2
                and v_loss < self._best_val_loss * 1.1):
            iou_gap = abs(t_iou - v_iou)
            if iou_gap < 0.1:
                msgs.append(
                    "  ✅ Training looks healthy — losses decreasing, train/val gap is small"
                )
                self._last_healthy_msg_epoch = epoch

        return msgs


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
        self._prev_lr         = None

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
        project               = self._config["project"]
        version_id            = generate_version_id(self._registry_root, project)
        self._version_folder  = create_version_folder(self._registry_root, project, version_id)

        self.log.emit(f"[INFO] Version: {version_id}")
        self.log.emit(f"[INFO] Folder : {self._version_folder}")

        # Step 2 — Copy staged dataset into {version_folder}/dataset/
        staged_dataset = self._config.get("dataset_folder", "")
        version_dataset = os.path.join(self._version_folder, "dataset")
        if staged_dataset and os.path.isdir(staged_dataset):
            self.log.emit("[INFO] Copying dataset snapshot into version folder...")
            shutil.copytree(staged_dataset, version_dataset)
            # Update dataset_info.json inside the copy to reflect the new absolute path
            _patch_dataset_info(version_dataset, version_dataset)
            # For YOLO: rewrite data.yaml path too
            _patch_data_yaml(version_dataset, version_dataset)
            self.log.emit(f"[INFO] Dataset snapshot: {version_dataset}")
        else:
            os.makedirs(version_dataset, exist_ok=True)
            self.log.emit("[WARN] No staged dataset found — version_folder/dataset/ is empty.")

        # Update config to point at the version-local dataset copy
        config_to_save = dict(self._config)
        config_to_save["dataset_folder"] = version_dataset
        config_to_save["dataset_info"]["absolute_path"] = version_dataset

        # Step 3 — Write config.json atomically
        config_json_path = os.path.join(self._version_folder, "config.json")
        _write_atomic(config_json_path, config_to_save)

        # Step 4 — Create manifest
        dataset_info = config_to_save["dataset_info"]
        if self._config["model_type"] in ("unet", "concentric"):
            out_classes = int(self._config["hyperparams"].get("out_classes", 2))
            dataset_info["num_classes"] = max(out_classes - 1, 1)

        writer = ManifestWriter(self._version_folder)
        writer.create(
            version_id     = version_id,
            model_type     = self._config["model_type"],
            annotator      = self._config["annotator"],
            project        = project,
            project_name   = self._config.get("project_name", ""),
            project_id     = self._config.get("project_id", ""),
            variant        = self._config.get("variant", ""),
            camera         = self._config.get("camera", ""),
            commit_message = self._config["commit_message"],
            dataset_info   = dataset_info,
            hyperparams    = self._config["hyperparams"],
            augmentations  = self._config.get("augmentations", {}),
        )
        self.log.emit("[INFO] Manifest created.")
        self.progress.emit(5)

        # Step 4 — Launch subprocess
        model_type  = self._config["model_type"]
        script_path = os.path.join(self._scripts_dir, f"train_{model_type}.py")

        python_exe = find_python()
        self.log.emit(f"[INFO] Launching: {script_path}")
        self.log.emit(f"[INFO] Python: {python_exe}")
        cmd = [python_exe, script_path, "--config", config_json_path, "--out", self._version_folder]

        import pathlib
        project_root = str(pathlib.Path(__file__).resolve().parents[2])
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

        popen_kwargs = dict(
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        self._proc = subprocess.Popen(cmd, **popen_kwargs)

        # Step 5 — Read stdout line by line
        # When cancelled we do NOT terminate here — the stop_requested sentinel
        # file was already written by cancel().  We keep draining stdout so the
        # script can finish its current epoch and exit cleanly.
        total_epochs = self._config["hyperparams"]["epochs"]
        diagnostics = _TrainingDiagnostics(total_epochs)
        for raw_line in self._proc.stdout:
            line = _ANSI_RE.sub("", raw_line).rstrip()
            if not line:
                continue
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
                mask_map50 = parsed.get("mask_map50")
                if mask_map50 is not None:
                    self.log.emit(f"  Mask mAP50: {mask_map50:.4f}")

                # Extra metrics — only logged if they were enabled and present
                t_prec    = parsed.get("train_precision")
                v_prec    = parsed.get("val_precision")
                t_pi_prec = parsed.get("train_per_image_precision")
                v_pi_prec = parsed.get("val_per_image_precision")
                t_rec     = parsed.get("train_recall")
                v_rec     = parsed.get("val_recall")
                t_pi_rec  = parsed.get("train_per_image_recall")
                v_pi_rec  = parsed.get("val_per_image_recall")
                t_f1      = parsed.get("train_f1")
                v_f1      = parsed.get("val_f1")
                t_pi_f1   = parsed.get("train_per_image_f1")
                v_pi_f1   = parsed.get("val_per_image_f1")
                if t_prec is not None:
                    self.log.emit(
                        f"  Precision   train: {t_prec:.4f}  val: {v_prec:.4f}  |  "
                        f"per-img  train: {t_pi_prec:.4f}  val: {v_pi_prec:.4f}"
                    )
                if t_rec is not None:
                    self.log.emit(
                        f"  Recall      train: {t_rec:.4f}  val: {v_rec:.4f}  |  "
                        f"per-img  train: {t_pi_rec:.4f}  val: {v_pi_rec:.4f}"
                    )
                if t_f1 is not None:
                    self.log.emit(
                        f"  F1 / Dice   train: {t_f1:.4f}  val: {v_f1:.4f}  |  "
                        f"per-img  train: {t_pi_f1:.4f}  val: {v_pi_f1:.4f}"
                    )

                # LR visibility — highlight drops
                current_lr = parsed.get("learning_rate")
                if current_lr is not None:
                    if self._prev_lr is not None and current_lr < self._prev_lr * 0.99:
                        self.log.emit(
                            f"  [LR] Learning rate reduced:  {self._prev_lr:.6f}  →  {current_lr:.6f}"
                        )
                    self._prev_lr = current_lr

                # ── Training health diagnostics ──
                diagnostics.update(epoch, t_loss, v_loss, t_iou, v_iou)
                for diag_msg in diagnostics.diagnose(epoch, t_loss, v_loss, t_iou, v_iou):
                    self.log.emit(diag_msg)
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
                    final_val_loss   = self._final_metrics.get("final_val_loss"),
                    best_val_iou     = self._final_metrics.get("best_val_iou"),
                    final_train_iou  = self._final_metrics.get("final_train_iou"),
                    final_val_iou    = self._final_metrics.get("final_val_iou"),
                )
            if os.path.isfile(os.path.join(self._version_folder, "best.pt")):
                writer.update_artifacts(weights="best.pt")
            writer.mark_partial()
            self._mirror_to_local(project, version_id)
            self.log.emit("[PARTIAL] Training stopped early — checkpoint preserved.")
            return version_id, self._version_folder, True

        elif self._cancelled:
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
                    final_val_loss   = self._final_metrics.get("final_val_loss"),
                    best_val_iou     = self._final_metrics.get("best_val_iou"),
                    final_train_iou  = self._final_metrics.get("final_train_iou"),
                    final_val_iou    = self._final_metrics.get("final_val_iou"),
                )
            writer.update_artifacts(weights="best.pt")
            self.progress.emit(100)
            writer.finalize()
            try:
                from mlops.registry.utils import write_project_csv
                write_project_csv(self._registry_root, project)
            except Exception:
                pass
            self._mirror_to_local(project, version_id)
            self.log.emit(f"[DONE] Version {version_id} saved to registry.")
            return version_id, self._version_folder, False

    # ------------------------------------------------------------------
    # Local mirror
    # ------------------------------------------------------------------

    def _mirror_to_local(self, project: str, version_id: str):
        """Copy the completed version folder to local_root (if configured)."""
        try:
            local_root = RegistrySettings().get_local_root()
        except Exception:
            return
        if not local_root:
            return
        # Resolve the destination path using the same hierarchy as the source
        parts = project.replace("\\", "/").split("/")
        dest = os.path.join(local_root, *parts, version_id)
        if os.path.exists(dest):
            # Already mirrored (e.g. from a previous interrupted run) — skip
            return
        try:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copytree(self._version_folder, dest)
            self.log.emit(f"[MIRROR] Local copy saved: {dest}")
        except Exception as exc:
            self.log.emit(f"[WARN] Local mirror failed: {exc}")

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


def _patch_dataset_info(dataset_folder: str, new_abs_path: str):
    """Rewrite dataset_info.json inside dataset_folder with the updated absolute path."""
    info_path = os.path.join(dataset_folder, "dataset_info.json")
    if not os.path.isfile(info_path):
        return
    try:
        with open(info_path, "r", encoding="utf-8") as fh:
            info = json.load(fh)
        info["output_folder"] = new_abs_path.replace("\\", "/")
        info["path"] = new_abs_path.replace("\\", "/")
        _write_atomic(info_path, info)
    except Exception:
        pass


def _patch_data_yaml(dataset_folder: str, new_abs_path: str):
    """Rewrite the 'path:' line in data.yaml to point at the new location."""
    yaml_path = os.path.join(dataset_folder, "data.yaml")
    if not os.path.isfile(yaml_path):
        return
    try:
        with open(yaml_path, "r", encoding="utf-8") as fh:
            content = fh.read()
        abs_fwd = new_abs_path.replace("\\", "/")
        new_lines = []
        for line in content.splitlines():
            new_lines.append(f"path: {abs_fwd}" if line.startswith("path:") else line)
        tmp = yaml_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write("\n".join(new_lines) + "\n")
        os.replace(tmp, yaml_path)
    except Exception:
        pass
