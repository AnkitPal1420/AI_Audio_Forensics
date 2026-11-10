"""Shared path, configuration, and logging helpers."""

import json
import logging
from pathlib import Path
from typing import Any, Dict


def project_root() -> Path:
    """Return the project root independently of the caller's working directory."""
    return Path(__file__).resolve().parent.parent


def dataset_directories() -> Dict[str, Path]:
    """Return the project's real and synthetic audio directories."""
    dataset_root = project_root() / "dataset"
    return {
        "real": dataset_root / "real",
        "fake": dataset_root / "fake",
    }


def preprocessed_directories() -> Dict[str, Path]:
    """Return the project's preprocessed real and synthetic audio directories."""
    prep_root = project_root() / "dataset" / "preprocessed"
    return {
        "real": prep_root / "real",
        "fake": prep_root / "fake",
    }


def load_config() -> Dict[str, Any]:
    """Load the JSON project configuration."""
    config_file = project_root() / "config" / "config.json"
    with config_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure concise console logging for project scripts."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
