"""
main.py — FastAPI application entry point.
Handles lifespan (scheduler start/stop), CORS, and route registration.
"""
import json
import asyncio
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .core.config import load_config, ensure_dirs
from .core.logger import get_logger, get_sse_queue
from .core.scheduler import start_scheduler, stop_scheduler, get_scan_cache, get_research_cache
from .core.kill_switch import activate, deactivate, status as ks_status
from .core.storage import (
    get_all_trades,
    get_open_trades,
    get_performance_history,
    get_model_data,
)
from .core.scanner import run_scan

_log = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Startup / shutdown lifecycle."""
    ensure_dirs()
    _log.info("CryptoSignalBot starting up")
    start_scheduler()
    yield
    _log.info("CryptoSignalBot shutting down")
    stop_scheduler()


app = FastAPI(
    title="CryptoSignalBot API",
    version="1.0.0",
    description="Five-step AI trading signal pipeline",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── /status ───────────────────────────────────────────────────────────────────

@app.get("/status")
async def get_status():
    """Bot status: running state, drawdown, kill switch, scan cache."""
    cfg = load_config()
    scan_cache = get_scan_cache()
    scan_result = scan_cache.get("result")

    from .core.risk import get_portfolio_summary
    portfolio = get_portfolio_summary()

    return {
        "status": "running",
        "mode": cfg["bot"]["mode"],
        "bot": cfg["bot"],
        "kill_switch": ks_status(),
        "scanner": {
            "running": scan_cache.get("running", False),
            "last_scan": scan_result.timestamp if scan_result else None,
            "markets_found": scan_result.markets_passed_filter if scan_result else 0,
            "top_opportunities": (
                [s.symbol for s in scan_result.top_opportunities[:5]]
                if scan_result else []
            ),
        },
        "portfolio": portfolio,
    }


# ── /scan ─────────────────────────────────────────────────────────────────────

@app.get("/scan")
async def trigger_scan():
    """Manually trigger a full market scan (Step 1)."""
    _log.info("Manual scan triggered via API")
    result = await run_scan()
    return {
        "timestamp": result.timestamp,
        "exchange": result.exchange,
        "total_fetched": result.total_markets_fetched,
        "passed_filter": result.markets_passed_filter,
        "anomalies": result.markets_flagged_anomaly,
        "duration_seconds": result.duration_seconds,
        "top_opportunities": [
            {
                "symbol": s.symbol,
                "score": s.opportunity_score,
                "price": s.last_price,
                "volume_24h_usd": round(s.volume_24h_usd),
                "change_24h_pct": round(s.price_change_pct_24h, 2),
                "spread_pct": round(s.spread_pct, 4),
                "anomalies": s.anomaly_flags,
            }
            for s in result.top_opportunities
        ],
    }


# ── /research ─────────────────────────────────────────────────────────────────

@app.get("/research/{symbol:path}")
async def research_symbol(symbol: str):
    """Run or return cached Step 2 Research brief for a specific symbol."""
    from .core.researcher import run_research
    from .core.scanner import run_scan

    # Normalise: BTC -> BTC/USDT
    if "/" not in symbol:
        symbol = f"{symbol.upper()}/USDT"
    else:
        symbol = symbol.upper()

    _log.info(f"Research requested: {symbol}")

    # Try to get price change from scan cache
    price_change = 0.0
    scan_cache = get_scan_cache()
    if scan_cache.get("result"):
        for snap in scan_cache["result"].all_tradable:
            if snap.symbol == symbol:
                price_change = snap.price_change_pct_24h
                break

    brief = await run_research(symbol, price_change_24h_pct=price_change)

    # Cache it
    get_research_cache()[symbol] = brief

    return {
        "symbol": brief.symbol,
        "timestamp": brief.timestamp,
        "consensus": {
            "label": brief.consensus_label,
            "compound": brief.consensus_compound,
            "confidence": brief.consensus_confidence,
        },
        "sources": [
            {
                "source": s.source,
                "label": s.label,
                "compound": s.compound,
                "confidence": s.confidence,
                "items": s.item_count,
                "snippets": s.top_snippets[:3],
            }
            for s in brief.sources
        ],
        "price_change_24h_pct": brief.price_change_24h_pct,
        "sentiment_price_gap": brief.sentiment_price_gap,
        "gap_signal": brief.gap_signal,
        "source_agreement": brief.source_agreement,
        "divergent_sources": brief.divergent_sources,
        "lesson_penalty": brief.lesson_penalty,
        "adjusted_compound": brief.adjusted_compound,
        "total_items": brief.total_items_analysed,
        "research_quality": brief.research_quality,
        "duration_seconds": brief.duration_seconds,
    }


@app.get("/research")
async def research_all_cached():
    """Return all cached research briefs."""
    cache = get_research_cache()
    return {
        "total": len(cache),
        "briefs": {
            sym: {
                "label": b.consensus_label,
                "compound": b.consensus_compound,
                "confidence": b.consensus_confidence,
                "gap_signal": b.gap_signal,
                "quality": b.research_quality,
                "timestamp": b.timestamp,
            }
            for sym, b in cache.items()
        },
    }


@app.post("/research/batch")
async def research_batch():
    """Run research on all current top scanner opportunities."""
    from .core.researcher import run_research_batch
    scan_cache = get_scan_cache()
    if not scan_cache.get("result"):
        return JSONResponse(status_code=400, content={"error": "No scan results. Run /scan first."})

    top = scan_cache["result"].top_opportunities[:10]
    briefs = await run_research_batch(top, max_concurrent=3)
    get_research_cache().update(briefs)

    return {
        "researched": len(briefs),
        "symbols": list(briefs.keys()),
        "summary": {
            sym: {"label": b.consensus_label, "compound": b.consensus_compound}
            for sym, b in briefs.items()
        },
    }



@app.get("/history")
async def get_history():
    """Full trade history — no pagination, no artificial limits."""
    trades = get_all_trades()
    return {"total": len(trades), "trades": trades}


# ── /logs (SSE) ───────────────────────────────────────────────────────────────

@app.get("/logs")
async def stream_logs():
    """Server-Sent Events stream of real-time log events."""
    q = get_sse_queue()

    async def event_generator():
        yield "data: {\"message\": \"Log stream connected\", \"level\": \"INFO\"}\n\n"
        while True:
            try:
                # Non-blocking check
                event = q.get_nowait()
                yield f"data: {json.dumps(event)}\n\n"
            except Exception:
                await asyncio.sleep(0.1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── /kill-switch ──────────────────────────────────────────────────────────────

@app.post("/kill-switch/activate")
async def activate_kill_switch():
    activate()
    return {"status": "activated", "message": "All new orders halted"}


@app.post("/kill-switch/deactivate")
async def deactivate_kill_switch():
    deactivate()
    return {"status": "deactivated", "message": "Trading resumed"}


@app.get("/kill-switch")
async def kill_switch_status():
    return ks_status()


# ── /config ───────────────────────────────────────────────────────────────────

@app.get("/config")
async def get_config():
    return load_config()


# ── /performance ──────────────────────────────────────────────────────────────

@app.get("/performance")
async def get_performance():
    history = get_performance_history()
    trades = get_all_trades()
    closed = [t for t in trades if t.get("status") != "open"]

    wins = [t for t in closed if t.get("outcome") == "win"]
    losses = [t for t in closed if t.get("outcome") == "loss"]
    pnl_list = [t.get("pnl_usd", 0) for t in closed]
    total_pnl = sum(pnl_list)

    win_rate = len(wins) / len(closed) if closed else 0.0

    # Sharpe (simplified: mean/std of daily returns)
    daily = history.get("daily", [])
    daily_pnls = [d.get("total_pnl_usd", 0) for d in daily]
    import statistics
    sharpe = 0.0
    if len(daily_pnls) > 1:
        mean_r = statistics.mean(daily_pnls)
        std_r = statistics.stdev(daily_pnls)
        sharpe = (mean_r / std_r * (365 ** 0.5)) if std_r > 0 else 0.0

    return {
        "total_trades": len(closed),
        "win_rate": round(win_rate, 4),
        "total_pnl_usd": round(total_pnl, 2),
        "sharpe_ratio": round(sharpe, 3),
        "wins": len(wins),
        "losses": len(losses),
        "daily_history": daily,
    }


# ── /signal ───────────────────────────────────────────────────────────────────

_signal_cache: dict = {"result": None, "running": False, "last_run": None}
_signal_rate_limit: float = 0.0   # timestamp of last /signal call

@app.get("/signal")
async def get_signal(symbol: str = None):
    """
    Run the full 5-step pipeline for a symbol (or best scanner pick).
    Rate limited to 1 call per 30 seconds. Returns cached result if recent.
    """
    global _signal_rate_limit
    from .core.scanner import run_scan
    from .core.researcher import run_research
    from .core.predictor import run_prediction

    # Rate limit
    now = time.time()
    if now - _signal_rate_limit < 30 and _signal_cache["result"] and not symbol:
        cached = _signal_cache["result"]
        cached["cached"] = True
        return cached
    _signal_rate_limit = now
    _signal_cache["running"] = True

    try:
        # Step 1: Scan (use cache if available and recent)
        scan_cache = get_scan_cache()
        if scan_cache.get("result") and not symbol:
            scan_result = scan_cache["result"]
        else:
            scan_result = await run_scan()

        # Pick symbol
        if symbol:
            sym = symbol.upper()
            if "/" not in sym:
                sym += "/USDT"
            market = next(
                (s for s in scan_result.all_tradable if s.symbol == sym),
                scan_result.top_opportunities[0] if scan_result.top_opportunities else None
            )
        else:
            market = scan_result.top_opportunities[0] if scan_result.top_opportunities else None

        if not market:
            return JSONResponse(status_code=404, content={"error": "No tradable markets found"})

        # Step 2: Research
        research_cache = get_research_cache()
        brief = research_cache.get(market.symbol)
        if not brief:
            brief = await run_research(market.symbol, market.price_change_pct_24h)
            research_cache[market.symbol] = brief

        # Step 3: Predict
        pred = await run_prediction(market.symbol, market, brief)
        if not pred:
            return JSONResponse(status_code=422, content={"error": "Prediction failed — insufficient data"})

        result = {
            "symbol": market.symbol,
            "direction": pred.direction,
            "has_edge": pred.has_edge,
            "p_win": pred.p_win,
            "p_market": pred.p_market,
            "edge_pct": pred.edge_pct,
            "confidence": pred.confidence,
            "xgb_prob": pred.xgb_prob,
            "mlp_prob": pred.mlp_prob,
            "sentiment": {
                "label": brief.consensus_label,
                "compound": brief.consensus_compound,
                "gap_signal": brief.gap_signal,
            },
            "market": {
                "price": market.last_price,
                "volume_24h_usd": market.volume_24h_usd,
                "change_24h_pct": market.price_change_pct_24h,
                "spread_pct": market.spread_pct,
                "anomalies": market.anomaly_flags,
                "opportunity_score": market.opportunity_score,
            },
            "top_features": pred.feature_importances,
            "timestamp": pred.timestamp,
            "cached": False,
        }
        _signal_cache["result"] = result
        _signal_cache["last_run"] = pred.timestamp
        return result

    finally:
        _signal_cache["running"] = False


@app.get("/signal/status")
async def signal_status():
    from .core.predictor import compute_rolling_brier_score, get_prediction_log
    model = None
    try:
        from .core.model import get_model
        model = get_model()
    except Exception:
        pass
    return {
        "running": _signal_cache.get("running", False),
        "last_run": _signal_cache.get("last_run"),
        "model_trained": model.is_trained if model else False,
        "model_samples": model.n_samples if model else 0,
        "rolling_brier_score": compute_rolling_brier_score(),
        "total_predictions": len(get_prediction_log()),
    }


# ── /trade ────────────────────────────────────────────────────────────────────

@app.post("/trade")
async def place_trade(body: dict = None):
    """
    Run full Steps 1-4 pipeline and execute if approved.
    Optional body: {"symbol": "BTC/USDT"} to target a specific pair.
    """
    from .core.scanner import run_scan
    from .core.researcher import run_research
    from .core.predictor import run_prediction
    from .core.risk import validate_trade
    from .core.execution import execute_trade
    from .core.exchange import get_adapter

    symbol = (body or {}).get("symbol")

    # Step 1: Scan
    scan_cache = get_scan_cache()
    scan_result = scan_cache.get("result") or await run_scan()
    market = None
    if symbol:
        sym = symbol.upper()
        if "/" not in sym:
            sym += "/USDT"
        market = next((s for s in scan_result.all_tradable if s.symbol == sym), None)
    if not market:
        market = scan_result.top_opportunities[0] if scan_result.top_opportunities else None
    if not market:
        return JSONResponse(status_code=404, content={"error": "No tradable market found"})

    # Step 2: Research
    research_cache = get_research_cache()
    brief = research_cache.get(market.symbol) or \
            await run_research(market.symbol, market.price_change_pct_24h)

    # Step 3: Predict
    pred = await run_prediction(market.symbol, market, brief)
    if not pred or not pred.has_edge or pred.direction == "SKIP":
        return JSONResponse(status_code=422, content={
            "error": "No edge or SKIP signal",
            "edge_pct": pred.edge_pct if pred else None,
            "direction": pred.direction if pred else None,
        })

    # Fetch OHLCV for VaR
    adapter = get_adapter()
    ohlcv = await adapter.fetch_ohlcv(market.symbol, "1h", limit=30)

    # Step 4: Risk
    risk = validate_trade(
        symbol=market.symbol,
        direction=pred.direction,
        p_win=pred.p_win,
        p_market=pred.p_market,
        edge_pct=pred.edge_pct,
        entry_price=market.last_price,
        ohlcv=ohlcv,
    )

    if not risk.approved:
        return JSONResponse(status_code=422, content={
            "error": "Risk check failed",
            "reason": risk.rejection_reason,
            "checks": [{"name": c.name, "passed": c.passed, "reason": c.reason}
                       for c in risk.checks],
        })

    # Execute
    result = await execute_trade(
        symbol=market.symbol,
        direction=pred.direction,
        risk_result=risk,
        predicted_probability=pred.p_win,
        feature_vector=None,  # stored separately in model_data.json
    )

    if not result.success:
        return JSONResponse(status_code=422, content={
            "error": "Order execution failed",
            "reason": result.reason,
            "slippage_pct": result.slippage_pct,
        })

    return {
        "trade_id": result.trade_id,
        "symbol": market.symbol,
        "direction": pred.direction,
        "fill_price": result.fill_price,
        "slippage_pct": result.slippage_pct,
        "position_usd": risk.position.usd_size,
        "tp_price": risk.position.tp_price,
        "sl_price": risk.position.sl_price,
        "p_win": pred.p_win,
        "edge_pct": pred.edge_pct,
        "risk_checks": [{"name": c.name, "passed": c.passed} for c in risk.checks],
    }


@app.post("/trade/{trade_id}/close")
async def manual_close_trade(trade_id: str, body: dict = None):
    """Manually close an open trade at current price."""
    from .core.execution import close_trade
    from .core.exchange import get_adapter
    from .core.storage import get_trade_by_id

    trade = get_trade_by_id(trade_id)
    if not trade:
        return JSONResponse(status_code=404, content={"error": "Trade not found"})
    if trade.get("status") != "open":
        return JSONResponse(status_code=400, content={"error": "Trade already closed"})

    # Get current price
    adapter = get_adapter()
    tickers = await adapter.fetch_tickers([trade["symbol"]])
    ticker = tickers.get(trade["symbol"], {})
    current_price = ticker.get("last", trade["entry_price"])

    result = await close_trade(trade_id, current_price, "manual")
    return result


@app.get("/risk/validate")
async def validate_risk(symbol: str, direction: str = "BUY"):
    """Preview risk validation for a symbol without executing."""
    from .core.risk import validate_trade
    from .core.exchange import get_adapter
    from .core.config import load_config as _cfg

    cfg = _cfg()
    adapter = get_adapter()
    tickers = await adapter.fetch_tickers([symbol.upper()])
    ticker = tickers.get(symbol.upper(), {})
    entry_price = ticker.get("last", 100.0)

    ohlcv = await adapter.fetch_ohlcv(symbol.upper(), "1h", limit=30)
    risk = validate_trade(
        symbol=symbol.upper(),
        direction=direction.upper(),
        p_win=0.65,
        p_market=0.40,
        edge_pct=25.0,
        entry_price=entry_price,
        ohlcv=ohlcv,
    )
    return risk.to_dict()


@app.get("/portfolio")
async def get_portfolio():
    """Full portfolio risk summary."""
    from .core.risk import get_portfolio_summary
    summary = get_portfolio_summary()
    open_trades = get_open_trades()
    return {
        **summary,
        "open_trades": open_trades,
    }



@app.post("/train")
async def trigger_training():
    """Manually trigger full model retraining."""
    _log.info("Manual model retrain triggered via API")
    from .core.model import retrain
    result = retrain()
    return {"status": result.get("status"), "result": result}


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/health/deep")
async def health_deep():
    """Deep health check across all subsystems."""
    from .core.health import run_health_check
    return run_health_check()


@app.get("/circuits")
async def circuit_breakers():
    """Circuit breaker states for all external services."""
    from .core.retry import circuit_status
    return circuit_status()
