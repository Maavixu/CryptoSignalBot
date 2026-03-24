"""
health.py — Deep health check across all bot subsystems.
Called by GET /health/deep and monitored by the scheduler.
"""
import time
from datetime import datetime
from typing import Dict

from .config import (
    load_config, DATA_DIR, MODELS_DIR, LOGS_DIR,
    TRADE_HISTORY_PATH, MODEL_DATA_PATH, LESSONS_PATH,
    PERFORMANCE_PATH, CONFIG_PATH,
)
from .logger import get_logger
from .kill_switch import status as ks_status
from .storage import get_all_trades, get_open_trades, get_model_data
from .retry import circuit_status

_log = get_logger("system")


def run_health_check() -> Dict:
    """
    Runs all subsystem checks. Returns structured health report.
    Any FAIL means the bot should not trade.
    """
    checks = {}
    overall = "healthy"
    start = time.time()

    # ── Config ────────────────────────────────────────────────────────────
    try:
        cfg = load_config()
        required_keys = ["bot", "scanner", "research", "prediction", "risk", "trade", "bankroll"]
        missing = [k for k in required_keys if k not in cfg]
        if missing:
            checks["config"] = {"status": "FAIL", "reason": f"Missing keys: {missing}"}
        else:
            checks["config"] = {"status": "OK", "mode": cfg["bot"]["mode"]}
    except Exception as e:
        checks["config"] = {"status": "FAIL", "reason": str(e)}
        overall = "degraded"

    # ── Data files ────────────────────────────────────────────────────────
    file_checks = {
        "trade_history": TRADE_HISTORY_PATH,
        "model_data": MODEL_DATA_PATH,
        "lessons": LESSONS_PATH,
        "performance": PERFORMANCE_PATH,
        "config_file": CONFIG_PATH,
    }
    for name, path in file_checks.items():
        if path.exists():
            size = path.stat().st_size
            checks[name] = {"status": "OK", "size_bytes": size}
        else:
            checks[name] = {"status": "WARN", "reason": "File missing — will be created on write"}

    # ── Model ─────────────────────────────────────────────────────────────
    try:
        from .model import get_model
        model = get_model()
        md = get_model_data()
        n_samples = len(md.get("features", []))
        checks["model"] = {
            "status": "OK" if model.is_trained else "WARN",
            "trained": model.is_trained,
            "n_samples": n_samples,
            "n_samples_at_last_train": model.n_samples,
            "reason": None if model.is_trained else f"Not trained yet ({n_samples} samples, need 10+)",
        }
    except Exception as e:
        checks["model"] = {"status": "FAIL", "reason": str(e)}
        overall = "degraded"

    # ── Exchange adapter ──────────────────────────────────────────────────
    try:
        from .exchange import get_adapter
        adapter = get_adapter()
        checks["exchange"] = {
            "status": "OK",
            "exchange_id": adapter.exchange_id,
            "mock_mode": adapter.use_mock,
        }
    except Exception as e:
        checks["exchange"] = {"status": "FAIL", "reason": str(e)}
        overall = "degraded"

    # ── Portfolio ─────────────────────────────────────────────────────────
    try:
        from .risk import get_portfolio_summary
        portfolio = get_portfolio_summary()
        risk_cfg = load_config()["risk"]

        dd_warn = portfolio["drawdown_pct"] >= risk_cfg["max_drawdown_pct"] * 0.8
        checks["portfolio"] = {
            "status": "WARN" if dd_warn else "OK",
            **portfolio,
        }
        if portfolio["drawdown_pct"] >= risk_cfg["max_drawdown_pct"]:
            overall = "degraded"
    except Exception as e:
        checks["portfolio"] = {"status": "FAIL", "reason": str(e)}

    # ── Kill switch ───────────────────────────────────────────────────────
    ks = ks_status()
    checks["kill_switch"] = {
        "status": "WARN" if ks["active"] else "OK",
        **ks,
    }

    # ── Circuit breakers ──────────────────────────────────────────────────
    circuits = circuit_status()
    open_circuits = [k for k, v in circuits.items() if v.get("open")]
    checks["circuit_breakers"] = {
        "status": "WARN" if open_circuits else "OK",
        "open_circuits": open_circuits,
        "all_circuits": circuits,
    }
    if open_circuits:
        overall = "degraded"

    # ── Disk space (warn if < 100MB free) ─────────────────────────────────
    try:
        import shutil
        free_bytes = shutil.disk_usage(DATA_DIR).free
        free_mb = free_bytes / 1024 / 1024
        checks["disk"] = {
            "status": "WARN" if free_mb < 100 else "OK",
            "free_mb": round(free_mb, 1),
        }
    except Exception:
        checks["disk"] = {"status": "UNKNOWN"}

    # Set overall
    if any(v.get("status") == "FAIL" for v in checks.values()):
        overall = "critical"
    elif any(v.get("status") == "WARN" for v in checks.values()):
        overall = "degraded" if overall != "critical" else overall

    return {
        "overall": overall,
        "timestamp": datetime.utcnow().isoformat(),
        "duration_ms": round((time.time() - start) * 1000, 1),
        "checks": checks,
    }
