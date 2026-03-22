"""
PRISM configuration package
===========================

Exposes stable paths to YAML configs under ``config/``. Runtime tuning (API keys,
ports) lives in :mod:`config.settings` and is loaded from ``.env`` via
``python-dotenv``.

Typical usage::

    from config import MCDA_CONFIG_PATH
    # pass MCDA_CONFIG_PATH to ProjectRanker or yaml.safe_load
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Configuration file paths
CONFIG_DIR = PROJECT_ROOT / "config"
MODEL_CONFIG_PATH = CONFIG_DIR / "model_config.yaml"
MCDA_CONFIG_PATH = CONFIG_DIR / "mcda_config.yaml"
UI_CONFIG_PATH = CONFIG_DIR / "ui_config.yaml"
