"""
config.py — Centralised config loader.
All modules import from here; never read config.json directly.
"""
import json
import os
import threading
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

CONFIG_PATH = DATA_DIR / "config.json"
TRADE_HISTORY_PATH = DATA_DIR / "trade_history.json"
MODEL_DATA_PATH = DATA_DIR / "model_data.json"
LESSONS_PATH = DATA_DIR / "lessons.json"
PERFORMANCE_PATH = DATA_DIR / "performance_history.json"
STOP_FILE_PATH = BASE_DIR / "STOP"

_lock = threading.RLock()


def load_config() -> dict:
    """Load config from disk. Thread-safe."""
    with _lock:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)


def save_config(cfg: dict) -> None:
    """Persist config changes to disk. Thread-safe."""
    with _lock:
        with open(CONFIG_PATH, "w") as f:
            json.dump(cfg, f, indent=2)


def get(key_path: str, default: Any = None) -> Any:
    """
    Dot-notation accessor.
    Example: get('risk.max_drawdown_pct') -> 8.0
    """
    cfg = load_config()
    keys = key_path.split(".")
    val = cfg
    try:
        for k in keys:
            val = val[k]
        return val
    except (KeyError, TypeError):
        return default


def set_value(key_path: str, value: Any) -> None:
    """
    Dot-notation setter. Persists immediately.
    Example: set_value('bankroll.current_usd', 9500.0)
    """
    with _lock:
        cfg = load_config()
        keys = key_path.split(".")
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
        save_config(cfg)


def is_kill_switch_active() -> bool:
    """Returns True if STOP file exists in project root."""
    return STOP_FILE_PATH.exists()


def ensure_dirs() -> None:
    """Create all required directories if missing."""
    for d in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
