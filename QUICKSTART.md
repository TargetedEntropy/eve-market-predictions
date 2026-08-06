# Quick Start Guide

From nothing to a served prediction. See `README.md` for the full reference and
`SETUP_GUIDE.md` for a step-by-step walkthrough.

## Option 1: Docker (recommended)

```bash
cd eve-discovery
cp .env.example .env

# DB_PASSWORD and REDIS_PASSWORD are REQUIRED -- compose refuses to start
# without them (there is no "changeme" default).
nano .env

docker compose up -d

# Tables, hypertables, compression and retention policies
docker compose exec api python scripts/init_database.py

# Item metadata, market history, an order snapshot and reference prices
docker compose exec collector python scripts/collect_data.py

# One model per tracked item, then forecasts
docker compose exec collector python scripts/train_model.py --all
docker compose exec collector python scripts/predict_all_items.py
```

Then open:

| Service | URL |
|---------|-----|
| Dashboard | http://localhost:8050 |
| API docs | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |

All ports bind to `127.0.0.1`. Set `API_BIND=0.0.0.0` / `DASHBOARD_BIND=0.0.0.0`
to expose them -- neither service authenticates, so put a proxy in front first.

## Option 2: Local installation

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env   # point DATABASE_URL at your PostgreSQL

# Requires PostgreSQL 15+ with the TimescaleDB extension.
# Add --allow-missing-timescaledb to run on plain PostgreSQL (no hypertables,
# no compression, no retention).
python scripts/init_database.py

python scripts/collect_data.py
python scripts/train_model.py --all
python scripts/predict_all_items.py

uvicorn src.api.main:app --reload
```

## Verify the installation

```bash
python -c "
import fastapi, torch, pandas, ta, sqlalchemy, redis
print('packages OK')
"

python -c "
import pandas as pd, numpy as np
from src.features.technical_indicators import TechnicalIndicators, INDICATOR_COLUMNS
df = pd.DataFrame({'average': np.linspace(100, 200, 120)})
df['volume'] = 1000
out = TechnicalIndicators.compute_all(df)
print(f'technical indicators OK ({len(INDICATOR_COLUMNS)} columns)')
"

pytest -q
```

## Backfill historical data

ESI only serves about a year of daily history. For more, import EVE Ref's dumps:

```bash
python scripts/import_everef_data.py --days 365
python scripts/import_everef_data.py --start 2023-01-01 --end 2023-12-31
```

## First prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"type_id": 44992, "horizon_days": 7}'
```

`/health` and `/predict` return **503** until a model exists for the requested
item -- train one first.

## Troubleshooting

**`/health` returns 503 with `models_loaded: 0`**
No checkpoint in `MODEL_DIR`. Run `python scripts/train_model.py --all`. In
Docker, confirm both `api` and `collector` mount `./data:/app/data`.

**`TimescaleDBUnavailable` on init**
The extension is not installed on your PostgreSQL server. Install it, or re-run
with `--allow-missing-timescaledb` and accept plain tables.

**Compose refuses to start**
`DB_PASSWORD` or `REDIS_PASSWORD` is unset in `.env`. That is deliberate.

**`ValueError` from the scheduler on `RETRAINING_DAY`**
Fixed: full day names (`sunday`) are now normalized to `sun`. Anything that is
not a day at all is rejected at startup with a clear message.

**ESI 420 / error limit**
The client pauses all in-flight work until the error window resets and retries
with backoff. If it happens constantly, lower `ESI_RATE_LIMIT`.
