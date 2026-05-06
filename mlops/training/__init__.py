"""
mlops/training/__init__.py

Public API for the Phase 3 training package.
TrainWorker is imported lazily so environments without PyQt5 can still import config_builder.
"""

from .config_builder import build_training_config, parse_metric_line, run_preflight


def __getattr__(name):
    if name == "TrainWorker":
        from .train_worker import TrainWorker
        return TrainWorker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["build_training_config", "parse_metric_line", "run_preflight", "TrainWorker"]
