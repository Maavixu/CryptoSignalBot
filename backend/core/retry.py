"""
retry.py — Exponential backoff retry decorator for external API calls.
Wraps async functions with configurable retry logic and circuit-breaker state.
"""
import asyncio
import functools
import time
from typing import Callable, Optional, Tuple, Type
from .logger import get_logger

_log = get_logger("system")

# Circuit breaker state per service
_circuit: dict = {}   # service_name -> {failures, last_failure, open_until}
CIRCUIT_THRESHOLD = 5          # open circuit after this many failures
CIRCUIT_RESET_SECONDS = 120    # try again after 2 minutes


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    service: Optional[str] = None,
):
    """
    Async retry decorator with exponential backoff.
    Usage:
        @with_retry(max_attempts=3, base_delay=1.0, service='binance')
        async def fetch_data(): ...
    """
    def decorator(fn: Callable):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            svc = service or fn.__name__

            # Circuit breaker check
            if _is_circuit_open(svc):
                _log.warning(f"Circuit breaker open for {svc} — skipping call")
                raise RuntimeError(f"Circuit breaker open: {svc}")

            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    result = await fn(*args, **kwargs)
                    _record_success(svc)
                    return result
                except exceptions as e:
                    last_exc = e
                    _record_failure(svc)
                    if attempt < max_attempts:
                        delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                        _log.warning(
                            f"{svc} attempt {attempt}/{max_attempts} failed: {e}. "
                            f"Retrying in {delay:.1f}s"
                        )
                        await asyncio.sleep(delay)
                    else:
                        _log.error(f"{svc} all {max_attempts} attempts failed: {e}")

            raise last_exc

        return wrapper
    return decorator


def _is_circuit_open(service: str) -> bool:
    state = _circuit.get(service, {})
    open_until = state.get("open_until", 0)
    return time.time() < open_until


def _record_failure(service: str):
    state = _circuit.setdefault(service, {"failures": 0, "last_failure": 0, "open_until": 0})
    state["failures"] += 1
    state["last_failure"] = time.time()
    if state["failures"] >= CIRCUIT_THRESHOLD:
        state["open_until"] = time.time() + CIRCUIT_RESET_SECONDS
        _log.error(
            f"Circuit breaker OPENED for {service} after {state['failures']} failures. "
            f"Pausing for {CIRCUIT_RESET_SECONDS}s"
        )


def _record_success(service: str):
    if service in _circuit:
        _circuit[service] = {"failures": 0, "last_failure": 0, "open_until": 0}


def circuit_status() -> dict:
    now = time.time()
    return {
        svc: {
            "failures": s["failures"],
            "open": now < s.get("open_until", 0),
            "resets_in": max(0, s.get("open_until", 0) - now),
        }
        for svc, s in _circuit.items()
    }
