"""
PRISM Configuration Module

Provides centralized configuration management for all PRISM components.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Configuration file paths
CONFIG_DIR = PROJECT_ROOT / "config"
MODEL_CONFIG_PATH = CONFIG_DIR / "model_config.yaml"
LLM_CONFIG_PATH = CONFIG_DIR / "llm_config.yaml"
MCDA_CONFIG_PATH = CONFIG_DIR / "mcda_config.yaml"
UI_CONFIG_PATH = CONFIG_DIR / "ui_config.yaml"
