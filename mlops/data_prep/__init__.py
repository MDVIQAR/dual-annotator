"""
mlops/data_prep/__init__.py

Public API for the Phase 2 data preparation package.
DataPrepWorker is imported lazily so that environments without PyQt5
(e.g. test runners) can still import DataPreparator without errors.
"""

from .preparator import DataPreparator


def __getattr__(name):
    if name == "DataPrepWorker":
        from .prep_worker import DataPrepWorker
        return DataPrepWorker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["DataPreparator", "DataPrepWorker"]
