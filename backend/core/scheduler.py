"""
scheduler.py — APScheduler configuration.
Runs the scanner every N minutes and monitors open trades every few seconds.
Must be started from the FastAPI lifespan handler.
"""
import asyncio
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED

from .config import load_config
from .logger import get_logger
from .kill_switch import is_kill_switch_active

_log = get_logger("scheduler")

# Global state shared with the API
_scan_result_cache = {"result": None, "running": False}
_research_cache: dict = {}   # symbol -> ResearchBrief
_scheduler: BackgroundScheduler = None


def get_scan_cache() -> dict:
    return _scan_result_cache


def get_research_cache() -> dict:
    return _research_cache


def _job_listener(event):
    if event.exception:
        _log.error(
            f"Scheduled job '{event.job_id}' raised an exception",
            data={"error": str(event.exception)},
        )
    else:
        _log.debug(f"Scheduled job '{event.job_id}' completed")


def _scan_job():
    """Called by scheduler every N minutes."""
    from .scanner import run_scan_sync
    from .researcher import run_research_batch
    import asyncio

    if is_kill_switch_active():
        _log.warning("Scan skipped — kill switch active")
        return
    _log.info("Scheduled scan triggered")
    _scan_result_cache["running"] = True
    result = run_scan_sync()
    _scan_result_cache["result"] = result
    _scan_result_cache["running"] = False
    if result:
        _log.info(
            f"Scan cached: {result.markets_passed_filter} tradable markets",
            data={"top_5": [s.symbol for s in result.top_opportunities[:5]]},
        )
        # Auto-run research on top 10 opportunities after each scan
        top = result.top_opportunities[:10]
        if top:
            _log.info(f"Auto-research triggered for {len(top)} markets")
            try:
                briefs = asyncio.run(run_research_batch(top, max_concurrent=3))
                _research_cache.update(briefs)
                _log.info(
                    f"Research cache updated: {len(briefs)} briefs",
                    data={"symbols": list(briefs.keys())},
                )
            except Exception as e:
                _log.error(f"Auto-research failed: {e}")


def _trade_monitor_job():
    """Called every few seconds to check TP/SL on open trades."""
    from .storage import get_open_trades
    from .execution import check_open_trades_prices
    from .exchange import get_adapter
    import asyncio

    if is_kill_switch_active():
        return
    open_trades = get_open_trades()
    if not open_trades:
        return

    _log.debug(f"Monitoring {len(open_trades)} open trades")

    # Fetch current prices for all open symbols
    symbols = list({t["symbol"] for t in open_trades})
    try:
        adapter = get_adapter()
        loop = asyncio.new_event_loop()
        tickers = loop.run_until_complete(adapter.fetch_tickers(symbols))
        loop.close()
        current_prices = {sym: t["last"] for sym, t in tickers.items() if "last" in t}
    except Exception as e:
        _log.error(f"Trade monitor: price fetch failed: {e}")
        return

    closed = check_open_trades_prices(current_prices)
    if closed:
        _log.info(
            f"Trade monitor: {len(closed)} trade(s) closed",
            data={"closed_ids": [t["id"] for t in closed]},
        )


def _nightly_job():
    """Runs at midnight UTC to consolidate daily performance."""
    _log.info("Nightly consolidation job running")
    try:
        from ..core.storage import (
            get_all_trades, append_daily_performance
        )
        import statistics
        from datetime import datetime, date

        today = date.today().isoformat()
        trades_today = [
            t for t in get_all_trades()
            if t.get("closed_at", "")[:10] == today
        ]
        if not trades_today:
            _log.info("No trades today — skipping consolidation")
            return

        pnl_list = [t.get("pnl_usd", 0) for t in trades_today]
        wins = [p for p in pnl_list if p > 0]
        win_rate = len(wins) / len(pnl_list) if pnl_list else 0
        total_pnl = sum(pnl_list)

        append_daily_performance({
            "n_trades": len(trades_today),
            "win_rate": round(win_rate, 4),
            "total_pnl_usd": round(total_pnl, 2),
            "avg_pnl_usd": round(statistics.mean(pnl_list), 2) if pnl_list else 0,
        })
        _log.info(
            f"Daily consolidation: {len(trades_today)} trades, "
            f"win rate {win_rate:.1%}, PnL ${total_pnl:.2f}"
        )
    except Exception as e:
        _log.error(f"Nightly job failed: {e}")


def start_scheduler() -> BackgroundScheduler:
    global _scheduler
    if _scheduler and _scheduler.running:
        return _scheduler

    cfg = load_config()
    scan_interval = cfg["scanner"]["interval_minutes"]
    monitor_interval = cfg["trade"]["monitor_interval_seconds"]

    _scheduler = BackgroundScheduler(
        job_defaults={"coalesce": True, "max_instances": 1},
        timezone="UTC",
    )
    _scheduler.add_listener(_job_listener, EVENT_JOB_ERROR | EVENT_JOB_EXECUTED)

    # Market scanner
    _scheduler.add_job(
        _scan_job,
        trigger=IntervalTrigger(minutes=scan_interval),
        id="market_scan",
        name=f"Market scanner (every {scan_interval}m)",
        replace_existing=True,
    )

    # Open trade monitor
    _scheduler.add_job(
        _trade_monitor_job,
        trigger=IntervalTrigger(seconds=monitor_interval),
        id="trade_monitor",
        name=f"Trade monitor (every {monitor_interval}s)",
        replace_existing=True,
    )

    # Nightly consolidation at midnight UTC
    from apscheduler.triggers.cron import CronTrigger
    _scheduler.add_job(
        _nightly_job,
        trigger=CronTrigger(hour=0, minute=0, timezone="UTC"),
        id="nightly_consolidation",
        name="Nightly consolidation",
        replace_existing=True,
    )

    _scheduler.start()
    _log.info(
        "Scheduler started",
        data={
            "scan_interval_minutes": scan_interval,
            "monitor_interval_seconds": monitor_interval,
        },
    )

    # Run an immediate scan on startup
    _scheduler.add_job(
        _scan_job,
        id="startup_scan",
        name="Startup scan",
        replace_existing=True,
    )

    return _scheduler


def stop_scheduler():
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        _log.info("Scheduler stopped")
