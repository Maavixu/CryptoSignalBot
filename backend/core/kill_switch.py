"""
kill_switch.py — STOP file detector.
Any module that touches orders must call check() first.
"""
from .config import is_kill_switch_active, STOP_FILE_PATH
from .logger import get_logger

_log = get_logger("kill_switch")


class KillSwitchTriggered(Exception):
    """Raised when STOP file is detected."""
    pass


def check() -> None:
    """
    Call before every order or trade execution.
    Raises KillSwitchTriggered if STOP file exists.
    """
    if is_kill_switch_active():
        _log.error(
            "STOP file detected — all new orders halted.",
            data={"stop_file": str(STOP_FILE_PATH)},
        )
        raise KillSwitchTriggered(
            f"Kill switch active: {STOP_FILE_PATH} exists. "
            "Delete the STOP file to resume trading."
        )


def activate() -> None:
    """Create the STOP file programmatically (e.g. from frontend button)."""
    STOP_FILE_PATH.touch()
    _log.error("Kill switch ACTIVATED — STOP file created.")


def deactivate() -> None:
    """Remove the STOP file to resume trading."""
    if STOP_FILE_PATH.exists():
        STOP_FILE_PATH.unlink()
        _log.info("Kill switch DEACTIVATED — STOP file removed.")


def status() -> dict:
    active = is_kill_switch_active()
    return {
        "active": active,
        "stop_file": str(STOP_FILE_PATH),
        "message": "Trading HALTED" if active else "Trading allowed",
    }
