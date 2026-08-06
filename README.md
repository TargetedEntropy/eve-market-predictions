# EVE Online Price Prediction Engine

A production-ready machine learning system for predicting EVE Online market prices using LSTM neural networks. Built with PyTorch, FastAPI, TimescaleDB, and completely free/open-source technologies.

## Features

- 🚀 **LSTM Neural Network** for price prediction with PyTorch
- 📊 **Real-time data collection** from ESI API (every 5 minutes)
- 📈 **Interactive Dashboard** with Dash/Plotly for data visualization
- 🔧 **Automated feature engineering** with 35 technical indicators plus order-book and temporal features
- 🎯 **FastAPI REST API** for serving predictions
- 📉 **Historical data import** from EVE Ref (configurable date range)
- 🤖 **Bulk prediction** for all tracked items (30-day forecasts)
- 🔍 **Drift detection** with Evidently AI, plus realized prediction-error scoring
- 🐳 **Docker deployment** with full stack (6 services)
- ⏰ **APScheduler** for automated data collection and model retraining
- 💾 **Time-series optimized** database with TimescaleDB (hypertables, compression, retention)
- 🔬 **MLflow integration** for experiment tracking and a registered model per item
- 🗄️ **Redis-backed ESI cache** honouring `ETag`/`Expires` so unchanged pages cost one 304

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  ESI API        │────▶│ Data         │────▶│ PostgreSQL      │
│  (EVE Online)   │     │ Collector    │     │ + TimescaleDB   │
└─────────────────┘     └──────────────┘     └─────────────────┘
                              │                      │
┌─────────────────┐           │                      │
│  EVE Ref        │───────────┘                      │
│  (Historical)   │                                  │
└─────────────────┘                                  │
                                                     ▼
                        ┌──────────────┐     ┌─────────────────┐
                        │ Feature      │────▶│ Feature Store   │
                        │ Engineering  │     │ + Redis Cache   │
                        └──────────────┘     └─────────────────┘
                              │                      │
                              ▼                      ▼
                        ┌──────────────┐     ┌─────────────────┐
                        │ LSTM Model   │◀────│ Training        │
                        │ (PyTorch)    │     │ Pipeline        │
                        └──────────────┘     └─────────────────┘
                              │                      │
                              ▼                      ▼
                  ┌──────────────────┐       ┌─────────────────┐
                  │ FastAPI Server   │       │ MLflow Tracking │
                  │ (Predictions)    │       │ (Experiments)   │
                  └──────────────────┘       └─────────────────┘
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
        ┌──────────────┐            ┌──────────────┐
        │  Dashboard   │            │   REST API   │
        │ (Dash/Plotly)│            │   Clients    │
        └──────────────┘            └──────────────┘
```

## Quick Start (Docker)

### 1. Prerequisites

- Docker & Docker Compose
- 4GB+ RAM
- 10GB+ disk space

### 2. Setup

```bash
# Clone repository
git clone <your-repo-url>
cd eve-discovery

# Copy environment file and configure
cp .env.example .env
# Edit .env: DB_PASSWORD and REDIS_PASSWORD are REQUIRED (compose refuses to
# start without them -- there is no "changeme" default)

# Start all services
docker compose up -d

# Check logs
docker compose logs -f
```

### 3. Initialize Database

```bash
# Create tables, hypertables, compression and retention policies.
# This fails loudly if TimescaleDB is missing; pass
# --allow-missing-timescaledb to run on plain PostgreSQL instead.
docker compose exec api python scripts/init_database.py

# Collect initial data (takes ~5-10 minutes)
docker compose exec collector python scripts/collect_data.py

# Train a model per tracked item, then forecast
docker compose exec collector python scripts/train_model.py --all
docker compose exec collector python scripts/predict_all_items.py
```

Until at least one model exists, `/health` returns **503** and `/predict`
returns **503** -- that is intentional, so orchestrator probes fail on a
non-functional API.

### 4. Access Services

- **Dashboard**: http://localhost:8050 (Interactive market visualization)
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs (Swagger UI)
- **MLflow UI**: http://localhost:5000 (Experiment tracking)
- **Health Check**: http://localhost:8000/health

All published ports bind to `127.0.0.1` by default; Postgres and Redis are not
published at all. To expose the API or dashboard on the network set
`API_BIND=0.0.0.0` / `DASHBOARD_BIND=0.0.0.0` -- neither service has
authentication, so put a reverse proxy in front of them first.

## Local Development Setup

### 1. System Requirements

- Python 3.11+
- PostgreSQL 15+ with TimescaleDB
- Redis 7+

### 2. Install Dependencies

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 3. Database Setup

```bash
# Install PostgreSQL and TimescaleDB
sudo apt install postgresql-15 timescaledb-2-postgresql-15

# Create database
sudo -u postgres createuser eve_user -P
sudo -u postgres createdb -O eve_user eve_markets

# Enable TimescaleDB
sudo -u postgres psql eve_markets -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"

# Initialize schema
python scripts/init_database.py
```

### 4. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

### 5. Collect Data

```bash
# Collect historical data and initial snapshot
python scripts/collect_data.py

# Start scheduler daemon for continuous collection
python scripts/scheduler_daemon.py
```

### 6. Run API Server

```bash
# Development
uvicorn src.api.main:app --reload

# Production
gunicorn src.api.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000
```

## Training a Model

Every item gets its **own** checkpoint plus a JSON sidecar holding the fitted
scaler, feature column order, architecture and validation metrics:

```
data/models/model_44992_10000002.pth    # state_dict only (loads with weights_only=True)
data/models/model_44992_10000002.json   # scaler params + metadata + metrics
```

```bash
# Train one item
python scripts/train_model.py --type-id 44992

# Train every tracked item
python scripts/train_model.py --all

# What the weekly job does: keep the old checkpoint unless the new one
# validates better
python scripts/train_model.py --all --promote-only-if-better

# Inspect what is on disk
python scripts/check_model_metrics.py
```

### Methodology

- **Chronological split** -- the most recent `--val-split` fraction is held
  out. A random split over overlapping lookback windows leaks future data.
- **Scaler fit on the training portion only**, then persisted with the model.
- **Baselines** (naive, 7-day moving average, linear trend) are scored on the
  same validation window and logged next to the model, so "the LSTM is better"
  is a measurement rather than an assumption.
- **Univariate**: the input is the last `LOOKBACK_WINDOW` daily average prices.
  The engineered feature set is computed and stored but not yet fed to the
  model; `feature_columns` in the metadata is the extension point.

## Dashboard Usage

The interactive dashboard provides real-time market visualization and analysis.

### Accessing the Dashboard

Open http://localhost:8050 in your web browser.

### Features

- **Price History Charts**: High/low/average price trends with interactive zoom
- **Volume Analysis**: Trading volume bar charts
- **Price Distribution**: Histogram showing price ranges
- **Prediction Overlay**: the latest stored forecast is drawn on the price
  chart, with realized prices marked once the evaluation job has scored them
- **Multi-Item Support**: 11 tracked items (PLEX, Skill Injectors, Minerals, etc.)
- **Regions**: whatever `COLLECTION_REGION_IDS` lists (default: The Forge only).
  The dropdown is built from that setting, so it can no longer offer a region
  that is never collected.
- **Time Range Selection**: 30 days, 90 days, 6 months, 1 year, all time
- **Auto-Refresh**: every `DASHBOARD_REFRESH_SECONDS` (default 900). Market
  history changes once a day; a 60s poll was pure overhead.
- **Dark Theme**: Professional Cyborg theme optimized for trading

### Tracked Items

- PLEX
- Large Skill Injector
- Skill Extractor
- Tritanium, Pyerite, Mexallon
- Isogen, Nocxium, Zydrine
- Megacyte, Morphite

## Bulk Predictions

Generate 30-day price forecasts for all tracked items:

```bash
# Forecast from the saved models (does NOT retrain)
docker compose exec collector python scripts/predict_all_items.py

# Retrain first, and also export JSON
docker compose exec collector python scripts/predict_all_items.py --train --json-dir data

# One item
docker compose exec collector python scripts/predict_item.py 40520 --days 30
```

Every prediction path writes rows to the `predictions` table (one row per
horizon day), which is what the dashboard overlay and the accuracy evaluation
read. JSON export is optional (`--json-dir`).

Each prediction includes:
- Current price
- Predicted price at the requested horizon
- Percentage change
- Day-by-day forecast
- The model version that produced it, and a confidence of `1 - validation MAPE`

> **Horizon caveat:** the model predicts one day ahead. Multi-day forecasts are
> autoregressive -- each prediction is fed back in as input -- so error
> compounds. A 30-day number is a trend indication, not a quote.

## Prediction Accuracy

```bash
# Fill in actual_price for elapsed horizons and refresh model_performance
docker compose exec collector python scripts/evaluate_predictions.py
```

This matches each elapsed prediction with the realized `market_history` price
for its `target_time`, then aggregates MAE / RMSE / MAPE / directional accuracy
per (item, region, model version, horizon) into the `model_performance` table.
The scheduler runs it daily, just after the history collection job.

### Import Historical Data

Import years of historical market data from EVE Ref:

```bash
# Last 90 days, tracked items, configured regions (the defaults)
docker compose exec collector python scripts/import_everef_data.py

# Explicit window
docker compose exec collector python scripts/import_everef_data.py \
    --start 2025-01-01 --end 2025-03-31

# Every region in the dumps (a lot more rows -- needed if you want the
# dashboard's Amarr/Dodixie options to have data)
docker compose exec collector python scripts/import_everef_data.py --days 30 --all-regions

# Every item
docker compose exec collector python scripts/import_everef_data.py --days 7 --all-items
```

Downloads are paced and retried; a day that is not published yet is skipped.
Rows are bulk-upserted rather than SELECT-then-insert per row.

## Model Drift Detection

Monitor model performance and detect data/prediction drift using Evidently AI:

```bash
# Check all items for drift
docker compose exec collector python scripts/check_drift.py

# View summary
docker compose exec collector cat data/drift_summary.json
```

The scheduler runs this daily; when `RETRAIN_ON_DRIFT=True` a drifted item is
retrained immediately (and only promoted if the new model validates better).

### What is Drift Detection?

Three distinct signals, which the docs used to conflate:
- **Data Drift**: input feature distributions changed (market conditions, game updates)
- **Target Drift**: the *price* distribution shifted (Evidently target drift)
- **Prediction error**: how wrong recently scored predictions actually were.
  This is the signal that justifies retraining, and it only appears once
  `scripts/evaluate_predictions.py` has filled in realized prices.

### Drift Detection Features

- **Reference Window**: 30-day baseline period for comparison
- **Current Window**: 7-day recent data to detect changes
- **Drift Threshold**: Configurable sensitivity (default: 50% of features)
- **HTML Reports**: Detailed visualizations for each item in `data/drift_reports/`
- **JSON Summary**: Machine-readable drift status in `data/drift_summary.json`
- **Alerting**: drift and job failures are logged, and POSTed to
  `ALERT_WEBHOOK_URL` (Slack-compatible payload) when one is configured

### Interpreting Results

When drift is detected:
1. Review HTML reports to identify which features drifted
2. Investigate EVE Online events (patches, expansions, meta shifts)
3. Consider retraining models with recent data
4. Update feature engineering if market dynamics changed

Example output:
```
Items with target drift: 5/11
- Large Skill Injector: Target drift detected ⚠️
- Tritanium: Target drift detected ⚠️
- Isogen: Target drift detected ⚠️
```

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### List Tracked Items

```bash
curl http://localhost:8000/items
```

### Predict Price

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "type_id": 44992,
    "region_id": 10000002,
    "horizon_days": 7
  }'
```

The horizon is in **days**, not hours: the model is trained on daily bars, so
an hourly horizon was never meaningful (the old API validated `horizon_hours`
and then ignored it).

Response:
```json
{
  "type_id": 44992,
  "item_name": "PLEX",
  "region_id": 10000002,
  "current_price": 3250000.0,
  "predicted_price": 3285000.0,
  "horizon_days": 7,
  "confidence": 0.9612,
  "timestamp": "2026-08-06T12:00:00",
  "model_version": "44992-10000002-20260806T020000",
  "daily": [
    {"day": 1, "date": "2026-08-07", "predicted_price": 3255000.0}
  ]
}
```

`confidence` is `1 - validation MAPE` for the serving model -- a measured
error rate, not a calibrated probability, and not the hardcoded `0.85` the API
used to return.

### List Servable Models

```bash
curl http://localhost:8000/models
```

### Status Codes

| Code | Meaning |
|------|---------|
| 200 | Prediction produced |
| 404 | Not enough price history for this item to forecast |
| 503 | No trained model for this `(type_id, region_id)` |
| 500 | Internal error -- details are logged server-side, never returned |

## Project Structure

```
eve-discovery/
├── src/
│   ├── api/              # FastAPI application
│   ├── collectors/       # ESI API clients & data collectors
│   ├── dashboard/        # Dash/Plotly interactive dashboard
│   ├── database/         # SQLAlchemy models & connections
│   ├── features/         # Feature engineering pipeline
│   ├── models/           # PyTorch LSTM model & training
│   ├── monitoring/       # Drift detection, prediction scoring, alerting
│   ├── utils/            # Time helpers and the Redis ESI cache
│   ├── constants.py      # TRACKED_ITEMS / REGIONS (single source of truth)
│   └── config.py         # Configuration management
│
├── scripts/
│   ├── init_database.py         # Tables, hypertables, compression/retention
│   ├── collect_data.py          # One-off data collection
│   ├── train_model.py           # Train per-item LSTM models
│   ├── predict_item.py          # Single item forecast (--train to retrain)
│   ├── predict_all_items.py     # Bulk forecast
│   ├── evaluate_predictions.py  # Score predictions, fill model_performance
│   ├── check_model_metrics.py   # Inspect saved model metadata
│   ├── import_everef_data.py    # Import historical data (argparse)
│   ├── check_drift.py           # Drift detection
│   └── scheduler_daemon.py      # Automated job daemon
│
├── tests/                # pytest suite (see pyproject.toml)
│
├── data/
│   ├── raw/                     # Downloaded EVE Ref dumps
│   ├── processed/               # Processed features
│   ├── models/                  # model_{type_id}_{region_id}.pth + .json
│   ├── drift_reports/           # HTML drift detection reports
│   ├── predictions_*.json       # Optional JSON exports
│   └── drift_summary.json       # Drift detection summary
│
├── requirements.txt      # Python dependencies
├── pyproject.toml        # pytest / black / ruff configuration
├── Dockerfile           # Application container
├── docker-compose.yml   # Full stack deployment (6 services)
├── .env.example         # Environment template
├── LICENSE              # MIT
└── README.md           # This file
```

## Configuration

Key environment variables in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+asyncpg://eve_user:password@localhost:5432/eve_markets` |
| `DB_PASSWORD` | Postgres password (**required** by docker-compose) | none |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/0` |
| `REDIS_PASSWORD` | Redis password (**required** by docker-compose) | none |
| `ESI_CACHE_ENABLED` | Cache ESI responses/ETags in Redis | `True` |
| `ESI_RATE_LIMIT` | Max ESI requests per period | `100` |
| `ESI_RATE_PERIOD` | Rate limit period (seconds) | `60` |
| `ESI_MAX_RETRIES` | Retries for 5xx/timeouts | `4` |
| `ESI_ERROR_LIMIT_FLOOR` | Pause all requests below this error budget | `20` |
| `ESI_USER_AGENT` | Identifies the app to CCP | see `.env.example` |
| `COLLECTION_REGION_IDS` | Regions to collect and show (comma separated) | `10000002` |
| `MODEL_DIR` | Directory of per-item checkpoints | `data/models` |
| `LOOKBACK_WINDOW` | LSTM lookback window (days) | `30` |
| `PREDICTION_HORIZON_DAYS` | Default forecast horizon | `30` |
| `API_CORS_ORIGINS` | Allowed CORS origins (comma separated) | `http://localhost:8050` |
| `API_CORS_ALLOW_CREDENTIALS` | Rejected together with `*` origins | `False` |
| `DASHBOARD_DEBUG` | Werkzeug debugger (never in a container) | `False` |
| `DASHBOARD_REFRESH_SECONDS` | Dashboard auto-refresh | `900` |
| `MLFLOW_TRACKING_URI` | MLflow server URL (`http://mlflow:5000` in Docker) | `http://localhost:5000` |
| `HISTORY_COLLECTION_HOUR` | UTC hour of the daily history job | `1` |
| `PREDICTION_HOUR` | UTC hour of the daily prediction job | `3` |
| `DRIFT_CHECK_HOUR` | UTC hour of the daily drift job | `4` |
| `RETRAINING_DAY` | `mon`-`sun`, a full day name, or `0`-`6` | `sun` |
| `RETRAIN_ON_DRIFT` | Retrain drifted items immediately | `True` |
| `ALERT_WEBHOOK_URL` | Optional Slack-compatible alert webhook | unset |
| `ORDER_SNAPSHOT_RETENTION_DAYS` | Drop raw snapshot chunks after N days | `30` |

All timestamps in the database are **naive UTC**; scheduler hours are UTC too.

## Scheduled Jobs

`scripts/scheduler_daemon.py` runs everything on a schedule. Every job uses
`max_instances=1` and `coalesce=True`, so a slow ESI response cannot stack runs.

| When (UTC) | Job |
|------------|-----|
| every `COLLECTION_INTERVAL_MINUTES` | Order book snapshots |
| every `FEATURE_COMPUTATION_INTERVAL_HOURS` | Feature computation |
| daily `HISTORY_COLLECTION_HOUR` | Market history + item metadata + liquidity tiers |
| daily `HISTORY_COLLECTION_HOUR + 1` | Reference prices, then prediction scoring |
| daily `PREDICTION_HOUR` | Generate and store predictions |
| daily `DRIFT_CHECK_HOUR` | Drift detection (retrains drifted items if enabled) |
| weekly `RETRAINING_DAY`/`RETRAINING_HOUR` | Retrain everything, promote only if better |

## Docker Commands

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f api
docker compose logs -f collector

# Stop services
docker compose down

# Rebuild after code changes
docker compose up --build -d

# Run commands in containers
docker compose exec api python scripts/collect_data.py
docker compose exec postgres psql -U eve_user eve_markets
```

## Docker Services

The application runs 6 containerized services:

| Service | Published on | Health check | Description |
|---------|--------------|--------------|-------------|
| **postgres** | not published | yes | PostgreSQL 15 + TimescaleDB for time-series data |
| **redis** | not published | yes | ESI response/ETag cache (password required) |
| **collector** | not published | no (no listener) | Scheduler daemon: collection, training, drift |
| **api** | `127.0.0.1:8000` | yes (`/health`) | FastAPI server for predictions |
| **mlflow** | `127.0.0.1:5000` | yes | MLflow tracking server (no auth -- loopback only) |
| **dashboard** | `127.0.0.1:8050` | yes | Dash/Plotly dashboard, served by gunicorn |

Every service restarts automatically. Five of the six have container health
checks; `collector` has none because it exposes no listener.

Postgres and Redis are reachable only on the compose network. `DB_PASSWORD` and
`REDIS_PASSWORD` have no defaults -- compose refuses to start without them.

## Monitoring

### MLflow

Track experiments and model performance:
```bash
# Access MLflow UI
open http://localhost:5000

# Log custom metrics
import mlflow
mlflow.log_metric("mae", 0.05)
```

### Database Queries

```sql
-- Check data collection status
SELECT type_id, COUNT(*), MAX(time)
FROM market_history
GROUP BY type_id;

-- View recent predictions
SELECT *
FROM predictions
ORDER BY prediction_time DESC
LIMIT 10;

-- How accurate has each model actually been?
SELECT type_id, model_version, horizon_days, n_samples, mape, directional_accuracy
FROM model_performance
ORDER BY time DESC, type_id
LIMIT 20;

-- Predictions still waiting for their horizon to elapse
SELECT COUNT(*) FROM predictions WHERE actual_price IS NULL;
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -r requirements.txt

# Run tests (pytest config lives in pyproject.toml; asyncio_mode=auto)
pytest

# With coverage
pytest --cov=src --cov-report=html
```

The suite runs without PostgreSQL, Redis, MLflow or network access: ESI is
stubbed with `httpx.MockTransport`, the database with recording fakes, and
models train on synthetic series. Tests needing a live database would be marked
`@pytest.mark.integration` (there are none yet).

### Code Formatting

```bash
# Format code
black src/ scripts/

# Lint
ruff check src/
```

## Troubleshooting

### Database Connection Issues

```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Test connection
psql -U eve_user -d eve_markets -c "SELECT version();"
```

### ESI API Rate Limiting

If you see "420 Error Limit Exceeded":
- Reduce `ESI_RATE_LIMIT` in `.env`
- Increase `COLLECTION_INTERVAL_MINUTES`

### Model Not Loading

```bash
# Which models does this instance have?
python scripts/check_model_metrics.py
curl http://localhost:8000/models

# Models are per item: data/models/model_{type_id}_{region_id}.pth
ls -lh data/models/

# Train the missing one
python scripts/train_model.py --type-id 44992
```

If the API reports `503` but the collector trained a model, check that both
containers mount `./data:/app/data` -- otherwise the checkpoint was written to
ephemeral container storage.

## Performance Optimization

### Database

`scripts/init_database.py` already applies these; they are listed here so you
know what exists:

```sql
-- Compression (applied automatically)
SELECT add_compression_policy('market_history',  INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('order_snapshots', INTERVAL '2 days', if_not_exists => TRUE);

-- Retention: order_snapshots is the highest-volume table
-- (full order book x 11 items x 288 runs/day)
SELECT add_retention_policy('order_snapshots', INTERVAL '30 days', if_not_exists => TRUE);
```

Both windows are configurable via `ORDER_SNAPSHOT_COMPRESS_AFTER_DAYS` and
`ORDER_SNAPSHOT_RETENTION_DAYS`.

### API

```python
# Connection pooling: already configured in src/database/connection.py
# ESI responses: cached in Redis with ETag revalidation (src/utils/cache.py)
# Model loading: memoized on the checkpoint's mtime, so a retrained model is
#                picked up without a restart
```

## Roadmap

### Completed ✅
- [x] Web dashboard with Dash/Plotly, including the prediction overlay
- [x] Historical data import from EVE Ref (configurable range, bulk upserts)
- [x] Bulk prediction for all tracked items, persisted to the database
- [x] Docker containerization (6 services)
- [x] LSTM training pipeline with chronological split and persisted scalers
- [x] Per-item models and an MLflow registered model per item
- [x] Real-time data collection from ESI with error-budget and ETag handling
- [x] Drift detection with Evidently, scheduled daily
- [x] Automated weekly retraining that only promotes better models
- [x] Prediction scoring and a `model_performance` table
- [x] Redis caching layer for ESI responses
- [x] Test suite

### In Progress 🚧
- [ ] Backtesting framework (no code yet; `vectorbt` was removed from
      requirements until there is)
- [ ] Prediction confidence intervals (today's `confidence` is
      `1 - validation MAPE`, not an interval)
- [ ] Multivariate model: the engineered features are stored but the model is
      still price-only

### Planned 📋
- [ ] Baseline/alternative models beyond naive, moving average and linear trend
      (ARIMA, XGBoost -- would add `statsmodels`/`xgboost`)
- [ ] Automated hyperparameter tuning
- [ ] Multi-region price correlation analysis
- [ ] Whale/large-order detection and order-depth levels (±0.5/1/2/5%)
- [ ] Patch-cycle and regional-hub features
- [ ] Data validation layer (gap detection, >3σ anomaly flags)
- [ ] Real-time WebSocket predictions
- [ ] Trading signal generation
- [ ] Portfolio optimization recommendations

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- **EVE Online** and **CCP Games** for the ESI API
- **EVE Ref** (data.everef.net) for historical market data archive
- **TimescaleDB** for time-series database excellence
- **PyTorch** for deep learning framework
- **FastAPI** for modern Python web framework
- **Plotly/Dash** for interactive data visualization
- **MLflow** for experiment tracking and model versioning
- **Evidently AI** for model drift detection and monitoring

## Support

For issues and questions:
- GitHub Issues: [https://github.com/TargetedEntropy/eve-market-predictions/issues]
- EVE Online API Documentation: https://developers.eveonline.com/
- TimescaleDB Docs: https://docs.timescale.com/

---

**Built with ❤️ for the EVE Online community**
