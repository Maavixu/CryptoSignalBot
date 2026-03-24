"""
logger.py — Structured logger that writes to file AND broadcasts
to the SSE queue so the frontend sees logs in real time.
"""
import logging
import queue
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import LOGS_DIR

# ── SSE broadcast queue (frontend consumes this) ─────────────────────────────
_sse_queue: queue.Queue = queue.Queue(maxsize=2000)


def broadcast(
    message: str,
    level: str = "INFO",
    step: Optional[str] = None,
    data: Optional[dict] = None,
) -> None:
    """
    Push a structured log event to the SSE queue.
    Non-blocking — drops oldest if queue is full.
    """
    event = {
        "ts": datetime.utcnow().isoformat(),
        "level": level,
        "step": step,
        "message": message,
        "data": data or {},
    }
    try:
        _sse_queue.put_nowait(event)
    except queue.Full:
        # Drop oldest to make room
        try:
            _sse_queue.get_nowait()
            _sse_queue.put_nowait(event)
        except queue.Empty:
            pass


def get_sse_queue() -> queue.Queue:
    return _sse_queue


# ── File logger setup ─────────────────────────────────────────────────────────

def _setup_file_logger() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"bot_{datetime.utcnow().strftime('%Y%m%d')}.log"

    logger = logging.getLogger("crypto_bot")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        # Console handler (for development)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger


_logger = _setup_file_logger()


# ── Public API ────────────────────────────────────────────────────────────────

class BotLogger:
    """
    Thin wrapper that logs to file AND broadcasts to SSE queue.
    Usage:
        log = BotLogger("scanner")
        log.info("Scan started", data={"pairs": 100})
    """

    def __init__(self, step: str):
        self.step = step
        self._log = logging.getLogger(f"crypto_bot.{step}")

    def info(self, msg: str, data: Optional[dict] = None) -> None:
        self._log.info(f"[{self.step.upper()}] {msg}")
        broadcast(msg, level="INFO", step=self.step, data=data)

    def warning(self, msg: str, data: Optional[dict] = None) -> None:
        self._log.warning(f"[{self.step.upper()}] {msg}")
        broadcast(msg, level="WARNING", step=self.step, data=data)

    def error(self, msg: str, data: Optional[dict] = None) -> None:
        self._log.error(f"[{self.step.upper()}] {msg}")
        broadcast(msg, level="ERROR", step=self.step, data=data)

    def debug(self, msg: str, data: Optional[dict] = None) -> None:
        self._log.debug(f"[{self.step.upper()}] {msg}")
        # Debug events go to file only — don't flood SSE


def get_logger(step: str) -> BotLogger:
    return BotLogger(step)
