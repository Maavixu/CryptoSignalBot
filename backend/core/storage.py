"""
storage.py — Append-only, thread-safe JSON file operations.
All writes go through here. Never truncate trade_history.json.
"""
import json
import threading
import time
from pathlib import Path
from typing import Any, List, Optional
from datetime import datetime

from .config import (
    TRADE_HISTORY_PATH,
    MODEL_DATA_PATH,
    LESSONS_PATH,
    PERFORMANCE_PATH,
)

_file_locks: dict[str, threading.RLock] = {
    "trade_history": threading.RLock(),
    "model_data": threading.RLock(),
    "lessons": threading.RLock(),
    "performance": threading.RLock(),
}


def _read_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    # Write to temp file then rename — atomic on POSIX
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    tmp.replace(path)


# ── Trade History ────────────────────────────────────────────────────────────

def append_trade(trade: dict) -> None:
    """Append a new trade record. Never truncates existing records."""
    if "id" not in trade:
        trade["id"] = f"trade_{int(time.time() * 1000)}"
    if "created_at" not in trade:
        trade["created_at"] = datetime.utcnow().isoformat()

    with _file_locks["trade_history"]:
        records = _read_json(TRADE_HISTORY_PATH)
        records.append(trade)
        _write_json(TRADE_HISTORY_PATH, records)


def update_trade(trade_id: str, updates: dict) -> bool:
    """Update fields on an existing trade by id. Returns True if found."""
    with _file_locks["trade_history"]:
        records = _read_json(TRADE_HISTORY_PATH)
        for i, t in enumerate(records):
            if t.get("id") == trade_id:
                records[i].update(updates)
                records[i]["updated_at"] = datetime.utcnow().isoformat()
                _write_json(TRADE_HISTORY_PATH, records)
                return True
    return False


def get_all_trades() -> List[dict]:
    with _file_locks["trade_history"]:
        return _read_json(TRADE_HISTORY_PATH)


def get_open_trades() -> List[dict]:
    return [t for t in get_all_trades() if t.get("status") == "open"]


def get_trade_by_id(trade_id: str) -> Optional[dict]:
    for t in get_all_trades():
        if t.get("id") == trade_id:
            return t
    return None


# ── Model Data ───────────────────────────────────────────────────────────────

def append_training_sample(features: List[float], outcome: int, trade_id: str) -> None:
    """Append one feature vector + binary outcome for model training."""
    with _file_locks["model_data"]:
        data = _read_json(MODEL_DATA_PATH)
        data["features"].append(features)
        data["outcomes"].append(outcome)
        data["trade_ids"].append(trade_id)
        _write_json(MODEL_DATA_PATH, data)


def get_model_data() -> dict:
    with _file_locks["model_data"]:
        return _read_json(MODEL_DATA_PATH)


def update_model_meta(last_trained: str, brier_score: Optional[float] = None) -> None:
    with _file_locks["model_data"]:
        data = _read_json(MODEL_DATA_PATH)
        data["last_trained"] = last_trained
        data["n_trades_at_last_train"] = len(data["features"])
        if brier_score is not None:
            data["brier_scores"].append({
                "score": brier_score,
                "timestamp": last_trained,
                "n_samples": len(data["features"]),
            })
        _write_json(MODEL_DATA_PATH, data)


# ── Lessons ──────────────────────────────────────────────────────────────────

def append_lesson(lesson: dict) -> None:
    with _file_locks["lessons"]:
        data = _read_json(LESSONS_PATH)
        lesson["recorded_at"] = datetime.utcnow().isoformat()
        data["lessons"].append(lesson)
        data["last_updated"] = datetime.utcnow().isoformat()
        _write_json(LESSONS_PATH, data)


def get_lessons() -> dict:
    with _file_locks["lessons"]:
        return _read_json(LESSONS_PATH)


def downrank_market(symbol: str, reason: str, duration_hours: int = 24) -> None:
    with _file_locks["lessons"]:
        data = _read_json(LESSONS_PATH)
        data["downranked_markets"][symbol] = {
            "reason": reason,
            "until": (
                datetime.utcnow().timestamp() + duration_hours * 3600
            ),
        }
        data["last_updated"] = datetime.utcnow().isoformat()
        _write_json(LESSONS_PATH, data)


def get_downranked_markets() -> dict:
    lessons = get_lessons()
    now = datetime.utcnow().timestamp()
    # Filter expired downranks
    return {
        sym: info
        for sym, info in lessons.get("downranked_markets", {}).items()
        if info.get("until", 0) > now
    }


# ── Performance ──────────────────────────────────────────────────────────────

def append_daily_performance(record: dict) -> None:
    with _file_locks["performance"]:
        data = _read_json(PERFORMANCE_PATH)
        record["date"] = datetime.utcnow().date().isoformat()
        data["daily"].append(record)
        data["last_updated"] = datetime.utcnow().isoformat()
        _write_json(PERFORMANCE_PATH, data)


def get_performance_history() -> dict:
    with _file_locks["performance"]:
        return _read_json(PERFORMANCE_PATH)
