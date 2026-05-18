"""
mlops/registry/__init__.py

Public API entry point for the DualAnnotator Registry package.
Exposes the three core components: settings persistence, manifest I/O, and registry scanning.
"""

from .settings_manager import RegistrySettings
from .manifest import ManifestWriter, ManifestReader, generate_version_id
from .scanner import RegistryScanner
from . import projects_config

__all__ = [
    "RegistrySettings",
    "ManifestWriter",
    "ManifestReader",
    "generate_version_id",
    "RegistryScanner",
    "projects_config",
]
