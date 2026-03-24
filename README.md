# CryptoSignalBot

A production-ready crypto trading signal bot with a five-step AI analysis pipeline, ensemble machine learning, and a real-time web dashboard. Runs in paper (simulated) mode out of the box — no exchange credentials required to start.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [The Five-Step Pipeline](#the-five-step-pipeline)
3. [File Reference](#file-reference)
4. [Local Setup](#local-setup)
5. [Configuration](#configuration)
6. [Running the Bot](#running-the-bot)
7. [Testing](#testing)
8. [Deploying to a VPS](#deploying-to-a-vps)
9. [Going Live](#going-live)
10. [API Reference](#api-reference)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (SPA)                        │
│         frontend/index.html  —  served statically        │
└───────────────────────┬─────────────────────────────────┘
                        │ HTTP / SSE
┌───────────────────────▼─────────────────────────────────┐
│                 FastAPI Backend                          │
│              backend/main.py  :8000                      │
└───┬───────────┬───────────┬──────────┬──────────────────┘
    │           │           │          │
  Step 1     Step 2      Step 3    Step 4+5
  Scan      Research    Predict   Risk + Compound
    │           │           │          │
Exchange    RSS/Reddit   XGBoost   Kelly/VaR
  API        /Twitter    + MLP     + Execution
                        Ensemble
```

All configuration lives in `data/config.json`. All persistent data is stored in flat JSON files under `data/`. Models are persisted as `.pkl` files under `models/`.

---

## The Five-Step Pipeline

Every signal and trade passes through all five steps sequentially. The output of each step is the input to the next.

### Step 1 — Scan (`backend/core/scanner.py`)

Connects to a crypto exchange (Binance by default, via the `ccxt` library) and fetches the top 100 USDT trading pairs. Filters by:

- Minimum 24h volume (default $1M USD)
- Minimum order book depth (default $50K USD)
- Maximum bid-ask spread (default 0.5%)

Flags anomalies: price moves > 10% in 1h, volume spikes > 5× the 7-day average. Scores each market with a composite opportunity score (volume + spread + momentum + depth) and returns the top 20 ranked opportunities. Runs automatically every 15 minutes via APScheduler.

### Step 2 — Research (`backend/core/researcher.py`)

For each top market, runs three parallel research agents:

- **RSS** (`sources/rss.py`): fetches CryptoPanic and CoinDesk RSS feeds
- **Reddit** (`sources/reddit.py`): fetches hot posts from crypto subreddits via PRAW
- **Twitter/X** (`sources/twitter.py`): fetches recent tweets (real API or mock)

All external content passes through the prompt injection sanitiser (`sanitiser.py`) before any analysis. Runs VADER sentiment analysis with a 60-term crypto lexicon on each item. Aggregates across sources with reliability weights (RSS=1.0, Reddit=0.75, Twitter=0.6). Computes a sentiment-price gap signal and applies lesson penalties from past bad predictions. Returns a `ResearchBrief` per market.

### Step 3 — Predict (`backend/core/predictor.py`)

Extracts a 25-feature vector from 1h OHLCV data plus the research brief (see `features.py` for the full list). Runs two independent classifiers:

- **XGBoost** (`xgboost.XGBClassifier`) — strong on tabular features
- **MLP** (`sklearn.MLPClassifier`) — neural net, different inductive bias

Both are wrapped in `CalibratedClassifierCV` so outputs are true probabilities. Averages their predictions for the ensemble P(win). Calculates edge:

```
edge = p_model - p_market
```

where `p_market` is the Kelly break-even probability for the configured reward:risk ratio, adjusted for realised volatility. Only proceeds if `edge >= 4%` (configurable). Logs every prediction for Brier score tracking.

### Step 4 — Risk (`backend/core/risk.py`)

Eight sequential checks before any order is placed:

| # | Check | Default threshold |
|---|-------|------------------|
| 1 | Edge threshold | ≥ 4% |
| 2 | Kelly position sizing | Quarter-Kelly, capped 0.5–10% bankroll |
| 3 | Total exposure limit | ≤ 50% of bankroll across all positions |
| 4 | Value at Risk (95%) | Daily VaR ≤ 5% of bankroll |
| 5 | Max drawdown guard | Current drawdown ≤ 8% |
| 6 | Daily loss limit | Today's losses ≤ 15% of bankroll |
| 7 | Concurrent positions | ≤ 15 open trades |
| 8 | Kill switch | `STOP` file must not exist |

If all pass, generates a `PositionSize` with entry, TP, and SL prices. Execution (`execution.py`) simulates a limit order fill with realistic slippage and aborts if slippage > 2%.

### Step 5 — Compound (`backend/core/compound.py`)

Fires automatically when any trade closes (TP hit, SL hit, or manual close):

1. **Failure classification**: bad_prediction / bad_timing / bad_execution / external_shock
2. **Lesson writing**: appends structured lesson to `data/lessons.json`
3. **Market downranking**: downranks a symbol for 24h after 3+ bad predictions
4. **Brier score update**: records the binary outcome against the stored prediction probability
5. **Model retraining**: triggers XGBoost + MLP retrain if dataset grew by ≥ 5 new samples
6. **Bankroll update**: adds PnL to `config.json` bankroll

---

## File Reference

```
crypto_bot/
│
├── run.py                          Entry point — starts uvicorn server
├── requirements.txt                All Python dependencies
├── README.md                       This file
│
├── backend/
│   ├── main.py                     FastAPI app, all route definitions, lifespan handler
│   ├── __init__.py
│   │
│   ├── api/
│   │   ├── routes.py               Stub route helpers (trade monitor hook)
│   │   └── __init__.py
│   │
│   └── core/
│       ├── config.py               Config loader — dot-notation get/set, path constants
│       ├── storage.py              Append-only thread-safe JSON file operations
│       ├── logger.py               Dual-output logger: file + SSE broadcast queue
│       ├── kill_switch.py          STOP file detector, activate/deactivate
│       ├── retry.py                Exponential backoff decorator + circuit breaker
│       ├── health.py               Deep health check across all 12 subsystems
│       ├── sanitiser.py            Prompt injection defence for all external text
│       │
│       ├── exchange.py             ccxt adapter — unified market data interface
│       ├── scanner.py              Step 1: market scan, filter, anomaly, score
│       ├── sentiment.py            VADER + crypto lexicon NLP classifier
│       ├── researcher.py           Step 2: parallel agents → ResearchBrief
│       ├── features.py             25-feature extractor (shared train + inference)
│       ├── model.py                XGBoost + MLP ensemble, calibration, persistence
│       ├── predictor.py            Step 3: OHLCV → features → ensemble → edge
│       ├── risk.py                 Step 4: 8 risk checks, Kelly sizer, VaR
│       ├── execution.py            Simulated order execution, trade lifecycle
│       ├── compound.py             Step 5: post-mortem, lessons, retrain
│       ├── scheduler.py            APScheduler: scan (15m), monitor (5s), nightly
│       │
│       └── sources/
│           ├── rss.py              RSS feed connector (feedparser + aiohttp)
│           ├── reddit.py           Reddit connector (PRAW + mock)
│           └── twitter.py          Twitter API v2 connector + mock
│
├── frontend/
│   └── index.html                  Single-page app — Dashboard / History / Health tabs
│
├── data/                           All persistent state (JSON files)
│   ├── config.json                 All tunable parameters — edit this to configure
│   ├── trade_history.json          Append-only trade log (never truncated)
│   ├── model_data.json             Feature vectors + outcomes for model training
│   ├── lessons.json                Post-mortem lessons, downranked markets
│   └── performance_history.json    Nightly consolidated performance metrics
│
├── models/                         Trained model files (auto-created)
│   ├── xgboost_model.pkl
│   ├── mlp_model.pkl
│   ├── scaler.pkl
│   └── model_meta.json
│
└── logs/                           Rotating daily log files (auto-created)
    └── bot_YYYYMMDD.log
```

### Key design decisions

**`features.py` is the single source of truth for feature engineering.** Both the training loop and live inference call the same `extract_features()` function. Never compute features in two places — that is the most common cause of silent model degradation in production.

**`storage.py` uses atomic writes.** Every JSON write goes to a `.tmp` file first, then is renamed to the target. On POSIX systems this is atomic — a crash mid-write cannot corrupt the file.

**`config.json` is the only place with magic numbers.** No thresholds are hardcoded in logic files. Everything reads from `load_config()`.

**The kill switch is checked at three points:** (1) before each scheduled scan job, (2) inside `risk.validate_trade()` as check #8, and (3) at the top of `execution.execute_trade()`. Creating a `STOP` file in the project root halts all new orders within seconds.

---

## Local Setup

### Prerequisites

- Python 3.11 or 3.12
- pip
- A modern browser (Chrome, Firefox, Safari)

### Install

```bash
# Clone or unzip the project
cd crypto_bot

# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### First run

```bash
python run.py
```

The server starts on `http://localhost:8000`. Open `frontend/index.html` directly in your browser (double-click the file, or use `open frontend/index.html` on macOS).

The bot starts in **paper mode** with mock exchange data — no credentials needed, no real money at risk.

### Serve the frontend via the backend (optional)

If you want the frontend served from the same port as the API (useful for VPS deployments):

```bash
# Add this to backend/main.py after the app definition:
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
```

Then visit `http://localhost:8000` instead of opening the HTML file directly.

---

## Configuration

All settings live in `data/config.json`. Edit this file to tune the bot. The server reads config on every request — no restart needed for most changes.

### Key settings

```json
{
  "bot": {
    "mode": "paper"          // "paper" = simulated, "live" = real orders
  },
  "scanner": {
    "interval_minutes": 15,              // how often to scan markets
    "min_volume_usd_24h": 1000000,       // $1M minimum 24h volume
    "max_spread_pct": 0.5,               // max bid-ask spread
    "opportunity_score_top_n": 20        // how many markets to research
  },
  "prediction": {
    "min_edge_pct": 4.0                  // minimum edge to trade (4%)
  },
  "risk": {
    "kelly_fraction": 0.25,              // quarter-Kelly (safer)
    "max_total_exposure_pct": 50.0,      // max % of bankroll in open positions
    "max_drawdown_pct": 8.0,             // halt trading at 8% drawdown
    "daily_loss_limit_pct": 15.0,        // halt for the day at 15% daily loss
    "max_concurrent_positions": 15       // max open trades at once
  },
  "trade": {
    "default_reward_risk_ratio": 2.0,    // 2:1 reward:risk (4% TP, 2% SL)
    "default_sl_pct": 2.0,
    "default_tp_pct": 4.0
  },
  "bankroll": {
    "initial_usd": 10000.0,              // starting capital
    "current_usd": 10000.0               // updated automatically after each trade
  },
  "exchange": {
    "id": "binance"                      // any ccxt-supported exchange
  },
  "reddit": {
    "client_id": "",                     // optional — leave blank for mock data
    "client_secret": ""
  },
  "twitter": {
    "bearer_token": ""                   // optional — leave blank for mock data
  }
}
```

---

## Running the Bot

### Start

```bash
python run.py
```

Options:

```bash
python run.py --port 8080          # custom port
python run.py --reload             # auto-reload on code changes (development)
python run.py --log-level debug    # verbose logging
```

### Stop gracefully

Press `Ctrl+C`. The scheduler shuts down cleanly.

### Emergency stop (kill switch)

To immediately halt all new orders without stopping the server:

```bash
touch STOP
```

Remove to resume:

```bash
rm STOP
```

The frontend also has a **KILL** button in the header that does the same thing.

### Trigger a trade manually

From the frontend, click **⚡ TRADE** in the header. This runs the full five-step pipeline and executes a trade if all risk checks pass. You will see a confirmation dialog before anything executes.

Or via the API:

```bash
curl -X POST http://localhost:8000/trade
curl -X POST http://localhost:8000/trade -H "Content-Type: application/json" -d '{"symbol":"BTC/USDT"}'
```

### Trigger a signal (analysis only, no execution)

```bash
curl http://localhost:8000/signal
curl http://localhost:8000/signal?symbol=ETH
```

---

## Testing

### Run the full test suite

```bash
cd crypto_bot
python -m pytest tests/ -v           # if you add a tests/ directory
```

### Manual integration tests (what was used during development)

Each phase has a standalone test you can run:

```bash
# Test the scanner
python -c "
import sys, asyncio; sys.path.insert(0, '.')
from backend.core.scanner import run_scan
result = asyncio.run(run_scan())
print(f'Markets: {result.markets_passed_filter}, Top: {result.top_opportunities[0].symbol}')
"

# Test the full pipeline (Steps 1–3)
python -c "
import sys, asyncio; sys.path.insert(0, '.')
from backend.core.scanner import run_scan
from backend.core.researcher import run_research
from backend.core.predictor import run_prediction

async def test():
    scan = await run_scan()
    market = scan.top_opportunities[0]
    brief = await run_research(market.symbol, market.price_change_pct_24h)
    pred = await run_prediction(market.symbol, market, brief)
    print(f'{pred.symbol}: {pred.direction} | p_win={pred.p_win:.3f} | edge={pred.edge_pct:+.1f}%')

asyncio.run(test())
"

# Test risk checks
python -c "
import sys; sys.path.insert(0, '.')
from backend.core.risk import validate_trade
result = validate_trade('BTC/USDT', 'BUY', p_win=0.65, p_market=0.38, edge_pct=27.0, entry_price=67000)
print('Approved:', result.approved)
for c in result.checks:
    print(f'  [{\"OK\" if c.passed else \"FAIL\"}] {c.name}: {c.reason}')
"

# Test the health check
python -c "
import sys; sys.path.insert(0, '.')
from backend.core.health import run_health_check
h = run_health_check()
print('Overall:', h['overall'])
for name, check in h['checks'].items():
    print(f'  {check[\"status\"]:4} {name}')
"
```

### Test via the API (with server running)

```bash
# Health
curl http://localhost:8000/health
curl http://localhost:8000/health/deep

# Status
curl http://localhost:8000/status

# Run scan
curl http://localhost:8000/scan | python3 -m json.tool

# Get signal for BTC
curl "http://localhost:8000/signal?symbol=BTC" | python3 -m json.tool

# Preview risk checks (no execution)
curl "http://localhost:8000/risk/validate?symbol=BTC/USDT&direction=BUY" | python3 -m json.tool

# View all trades
curl http://localhost:8000/history | python3 -m json.tool

# Manually trigger model retraining
curl -X POST http://localhost:8000/train | python3 -m json.tool

# Activate kill switch
curl -X POST http://localhost:8000/kill-switch/activate
# Deactivate
curl -X POST http://localhost:8000/kill-switch/deactivate
```

### What to verify after setup

1. `GET /health/deep` returns `"overall": "healthy"` with all 12 checks OK
2. `GET /scan` returns markets with opportunity scores
3. `GET /signal` returns a direction (BUY/SELL/SKIP) with edge percentage
4. The frontend log stream shows scan and research events in real time
5. `POST /trade` with paper mode places a simulated trade visible in the History tab

---

## Deploying to a VPS

### Recommended spec

- 1 vCPU, 2GB RAM minimum (4GB recommended for model training)
- Ubuntu 22.04 LTS
- 20GB SSD

### Providers that work well

DigitalOcean Droplet, Hetzner CX21, Linode Nanode 2GB, Vultr Regular.

### Step 1 — Server setup

```bash
# Connect to your VPS
ssh root@YOUR_SERVER_IP

# Update system
apt update && apt upgrade -y

# Install Python 3.11
apt install -y python3.11 python3.11-venv python3-pip git

# Create a non-root user
adduser botuser
usermod -aG sudo botuser
su - botuser
```

### Step 2 — Upload the project

**Option A: SCP (simplest)**

```bash
# From your local machine
scp -r crypto_bot botuser@YOUR_SERVER_IP:/home/botuser/
```

**Option B: Git**

```bash
# On your local machine — push to a private GitHub repo first
git init
git add .
git commit -m "initial"
git remote add origin git@github.com:yourname/crypto-bot.git
git push -u origin main

# On the VPS
git clone git@github.com:yourname/crypto-bot.git crypto_bot
```

### Step 3 — Install dependencies on VPS

```bash
cd /home/botuser/crypto_bot
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 4 — Configure

```bash
nano data/config.json
```

Set your bankroll, exchange, and any API keys. Keep `"mode": "paper"` until you have verified everything works.

### Step 5 — Run as a systemd service (auto-restart, survives reboots)

```bash
# Create the service file
sudo nano /etc/systemd/system/cryptobot.service
```

Paste this content (adjust paths if needed):

```ini
[Unit]
Description=CryptoSignalBot
After=network.target

[Service]
Type=simple
User=botuser
WorkingDirectory=/home/botuser/crypto_bot
Environment=PATH=/home/botuser/crypto_bot/.venv/bin
ExecStart=/home/botuser/crypto_bot/.venv/bin/python run.py --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable cryptobot
sudo systemctl start cryptobot

# Check it's running
sudo systemctl status cryptobot

# View logs
sudo journalctl -u cryptobot -f
```

### Step 6 — Open the firewall

```bash
sudo ufw allow 22/tcp      # SSH — do not skip this
sudo ufw allow 8000/tcp    # Bot API and frontend
sudo ufw enable
```

### Step 7 — (Optional) Nginx reverse proxy with HTTPS

Install nginx and certbot:

```bash
sudo apt install -y nginx certbot python3-certbot-nginx
```

Create `/etc/nginx/sites-available/cryptobot`:

```nginx
server {
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;           # required for SSE log stream
    }
}
```

Enable and get HTTPS certificate:

```bash
sudo ln -s /etc/nginx/sites-available/cryptobot /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
sudo certbot --nginx -d your-domain.com
```

After this, visit `https://your-domain.com` — the frontend and API are both served over HTTPS.

### Updating the bot on VPS

```bash
# If using git
cd /home/botuser/crypto_bot
git pull origin main
source .venv/bin/activate
pip install -r requirements.txt   # only if requirements changed
sudo systemctl restart cryptobot
```

---

## Going Live

When you are ready to trade with real money:

### 1. Get exchange API keys

On Binance (or your chosen exchange):
- Log in → API Management → Create API
- Enable **Spot Trading** only — do not enable withdrawals
- Whitelist your VPS IP address
- Copy the API key and secret

### 2. Update config.json

```json
{
  "bot": {
    "mode": "live"
  },
  "exchange": {
    "id": "binance",
    "api_key": "YOUR_API_KEY",
    "api_secret": "YOUR_API_SECRET",
    "testnet": false
  }
}
```

### 3. Update exchange.py to pass credentials

In `backend/core/exchange.py`, update the adapter initialisation:

```python
cfg = load_config()
exchange_cfg = cfg.get("exchange", {})
ExClass = getattr(ccxt_async, exchange_id)
self._exchange = ExClass({
    "apiKey": exchange_cfg.get("api_key", ""),
    "secret": exchange_cfg.get("api_secret", ""),
    "enableRateLimit": True,
    "timeout": 30000,
})
```

### 4. Start with small bankroll

Set `"initial_usd"` and `"current_usd"` to a small amount (e.g. $500) and trade for at least two weeks in paper mode first to verify the model is calibrated (Brier score < 0.20 is a reasonable target).

### 5. Optional API credentials for richer sentiment

**Reddit** (free):
- Go to https://www.reddit.com/prefs/apps
- Create an app → script type
- Copy client_id and client_secret into config.json

**Twitter/X** (requires Basic or Pro plan):
- Go to https://developer.twitter.com
- Create a project and app
- Copy Bearer Token into config.json

---

## API Reference

All endpoints are served at `http://localhost:8000` (or your VPS address).

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Simple health ping |
| GET | `/health/deep` | Full 12-subsystem health check |
| GET | `/status` | Bot state, portfolio, kill switch, last scan |
| GET | `/scan` | Trigger market scan (Step 1), returns ranked opportunities |
| GET | `/research/{symbol}` | Research brief for one symbol (Step 2) |
| POST | `/research/batch` | Research all current scanner top-10 |
| GET | `/signal` | Run Steps 1–3, return signal (rate-limited 30s) |
| GET | `/signal?symbol=BTC` | Signal for specific symbol |
| GET | `/signal/status` | Model state, sample count, rolling Brier score |
| POST | `/trade` | Run Steps 1–4 and execute if approved |
| POST | `/trade/{id}/close` | Manually close an open trade |
| GET | `/risk/validate?symbol=X&direction=BUY` | Preview risk checks without executing |
| GET | `/portfolio` | Bankroll, drawdown, exposure, open positions |
| GET | `/history` | Full trade history (no pagination limit) |
| GET | `/performance` | Win rate, Sharpe ratio, total P&L, Brier score |
| POST | `/train` | Manually trigger model retraining |
| GET | `/config` | Current configuration |
| GET | `/circuits` | Circuit breaker states for external services |
| GET | `/logs` | SSE stream of real-time log events |
| POST | `/kill-switch/activate` | Halt all new orders (creates STOP file) |
| POST | `/kill-switch/deactivate` | Resume trading (removes STOP file) |
| GET | `/kill-switch` | Kill switch status |

### SSE log stream format

`GET /logs` returns a persistent Server-Sent Events stream. Each event is JSON:

```json
{
  "ts": "2026-03-24T10:30:00.123456",
  "level": "INFO",
  "step": "scan",
  "message": "Scan complete: 18 tradable markets, 3 top opportunities",
  "data": { "duration_s": 0.21, "top_pairs": ["BTC/USDT", "ETH/USDT"] }
}
```

Connect with:

```javascript
const sse = new EventSource('http://localhost:8000/logs');
sse.onmessage = e => console.log(JSON.parse(e.data));
```

---

## Troubleshooting

**Bot starts but scan returns 0 markets**
Check `data/config.json` — `min_order_book_depth_usd` may be too high for mock data. Default is $50,000. The mock exchange generates realistic depths but randomises them — occasionally some pairs will be below threshold.

**Model shows "untrained" in the dashboard**
The model needs at least 10 completed trades to train. Run a few paper trades via `POST /trade` and close them. Once you have 10+ closed trades, `POST /train` will train the ensemble.

**SSE log stream disconnects frequently**
This is normal behind some proxies. The frontend reconnects automatically within 3 seconds. If using nginx, ensure `proxy_buffering off` is set.

**Exchange API errors in live mode**
Check that your IP is whitelisted on the exchange, and that the API key has Spot Trading enabled. Look at `GET /circuits` — if the exchange circuit breaker is open (after 5 consecutive failures), it will pause for 2 minutes before retrying.

**High memory usage**
The XGBoost and MLP models are loaded into memory at startup. On very small VPS instances (512MB RAM), this may cause issues. Use a 1GB+ instance, or reduce `n_estimators` in `model.py` from 100 to 50.
