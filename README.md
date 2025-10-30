# EVE Online Price Prediction Engine

A production-ready machine learning system for predicting EVE Online market prices using LSTM neural networks. Built with PyTorch, FastAPI, TimescaleDB, and completely free/open-source technologies.

## Features

- 🚀 **LSTM Neural Network** for price prediction with PyTorch
- 📊 **Real-time data collection** from ESI API (every 5 minutes)
- 📈 **Interactive Dashboard** with Dash/Plotly for data visualization
- 🔧 **Automated feature engineering** with 50+ technical indicators
- 🎯 **FastAPI REST API** for serving predictions
- 📉 **Historical data import** from EVE Ref (2003-present)
- 🤖 **Bulk prediction** for all tracked items (30-day forecasts)
- 🔍 **Model drift detection** with Evidently AI for monitoring
- 🐳 **Docker deployment** with full stack (6 services)
- ⏰ **APScheduler** for automated data collection and model retraining
- 💾 **Time-series optimized** database with TimescaleDB
- 🔬 **MLflow integration** for experiment tracking and model versioning

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
# Edit .env and set DB_PASSWORD

# Start all services
docker compose up -d

# Check logs
docker compose logs -f
```

### 3. Initialize Database

```bash
# Run database initialization
docker compose exec api python scripts/init_database.py

# Collect initial data (takes ~5-10 minutes)
docker compose exec collector python scripts/collect_data.py
```

### 4. Access Services

- **Dashboard**: http://localhost:8050 (Interactive market visualization)
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs (Swagger UI)
- **MLflow UI**: http://localhost:5000 (Experiment tracking)
- **Health Check**: http://localhost:8000/health

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

### Prepare Training Data

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.models import MarketLSTM, MarketDataset, ModelTrainer

# Load your data (example)
# df = load_market_history()

# Prepare sequences
lookback = 30
sequences = []
targets = []

for i in range(lookback, len(df)):
    sequences.append(df[i-lookback:i].values)
    targets.append(df.iloc[i]['average'])

sequences = np.array(sequences)
targets = np.array(targets)

# Create dataset
dataset = MarketDataset(sequences, targets)

# Train model
trainer = ModelTrainer(
    input_size=sequences.shape[2],
    hidden_size=128,
    num_layers=2,
    dropout=0.2,
    learning_rate=0.001,
    epochs=50
)

metrics = trainer.train(dataset, val_split=0.2)
```

## Dashboard Usage

The interactive dashboard provides real-time market visualization and analysis.

### Accessing the Dashboard

Open http://localhost:8050 in your web browser.

### Features

- **Price History Charts**: High/low/average price trends with interactive zoom
- **Volume Analysis**: Trading volume bar charts
- **Price Distribution**: Histogram showing price ranges
- **Multi-Item Support**: 11 tracked items (PLEX, Skill Injectors, Minerals, etc.)
- **Multiple Regions**: The Forge (Jita), Domain (Amarr), Sinq Laison (Dodixie)
- **Time Range Selection**: 30 days, 90 days, 6 months, 1 year, all time
- **Auto-Refresh**: Updates every 60 seconds
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
# Run predictions for all 11 items
docker compose exec collector python scripts/predict_all_items.py

# Check results
docker compose exec collector cat data/predictions_summary.json
```

Each prediction includes:
- Current price
- Predicted price (30 days ahead)
- Percentage change
- Day-by-day forecast
- Model validation loss

### Import Historical Data

Import years of historical market data from EVE Ref:

```bash
# Import last 90 days for all items
docker compose exec collector python scripts/import_everef_data.py

# Data available from 2003 to present
# Automatically filtered for tracked items
```

## Model Drift Detection

Monitor model performance and detect data/prediction drift using Evidently AI:

```bash
# Check all items for drift
docker compose exec collector python scripts/check_drift.py

# View summary
docker compose exec collector cat data/drift_summary.json
```

### What is Drift Detection?

Model drift occurs when:
- **Data Drift**: Input feature distributions change over time (market conditions, game updates)
- **Target Drift**: Price prediction patterns shift (meta changes, economic events)

### Drift Detection Features

- **Reference Window**: 30-day baseline period for comparison
- **Current Window**: 7-day recent data to detect changes
- **Drift Threshold**: Configurable sensitivity (default: 50% of features)
- **HTML Reports**: Detailed visualizations for each item in `data/drift_reports/`
- **JSON Summary**: Machine-readable drift status in `data/drift_summary.json`

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
    "horizon_hours": 24
  }'
```

Response:
```json
{
  "type_id": 44992,
  "region_id": 10000002,
  "current_price": 3250000.0,
  "predicted_price": 3285000.0,
  "horizon_hours": 24,
  "confidence": 0.85,
  "timestamp": "2025-10-26T12:00:00.000Z",
  "model_version": "0.1.0"
}
```

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
│   ├── monitoring/       # Drift detection with Evidently AI
│   ├── utils/            # Utility functions
│   └── config.py         # Configuration management
│
├── scripts/
│   ├── init_database.py       # Database initialization
│   ├── collect_data.py        # Manual data collection
│   ├── train_model.py         # Train LSTM model
│   ├── predict_item.py        # Single item prediction
│   ├── predict_all_items.py   # Bulk prediction (all items)
│   ├── import_everef_data.py  # Import historical data
│   ├── check_drift.py         # Model drift detection
│   └── scheduler_daemon.py    # Automated collection daemon
│
├── data/
│   ├── raw/              # Raw collected data
│   ├── processed/        # Processed features
│   ├── models/           # Trained model files
│   ├── drift_reports/    # HTML drift detection reports
│   ├── predictions_*.json     # Prediction outputs
│   └── drift_summary.json     # Drift detection summary
│
├── requirements.txt      # Python dependencies
├── Dockerfile           # Application container
├── docker-compose.yml   # Full stack deployment (6 services)
├── .env.example         # Environment template
└── README.md           # This file
```

## Configuration

Key environment variables in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+asyncpg://eve_user:password@localhost:5432/eve_markets` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/0` |
| `ESI_RATE_LIMIT` | Max ESI requests per period | `100` |
| `ESI_RATE_PERIOD` | Rate limit period (seconds) | `60` |
| `MODEL_PATH` | Path to trained model | `data/models/best_model.pth` |
| `LOOKBACK_WINDOW` | LSTM lookback window (days) | `30` |
| `MLFLOW_TRACKING_URI` | MLflow server URL | `http://localhost:5000` |

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

| Service | Port | Description |
|---------|------|-------------|
| **postgres** | 5432 | PostgreSQL 15 + TimescaleDB for time-series data |
| **redis** | 6379 | Redis cache for rate limiting and caching |
| **collector** | - | Background daemon for data collection (ESI API) |
| **api** | 8000 | FastAPI server for predictions and health checks |
| **mlflow** | 5000 | MLflow tracking server for experiments |
| **dashboard** | 8050 | Dash/Plotly interactive visualization dashboard |

All services use automatic health checks and restart policies for reliability.

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
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install pytest pytest-asyncio pytest-cov

# Run tests
pytest

# With coverage
pytest --cov=src --cov-report=html
```

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
# Check model file exists
ls -lh data/models/best_model.pth

# Train a new model if needed
# (see Training a Model section)
```

## Performance Optimization

### Database

```sql
-- Add compression for old data
SELECT add_compression_policy('market_history', INTERVAL '7 days');

-- Create additional indexes
CREATE INDEX IF NOT EXISTS idx_predictions_type_time
ON predictions (type_id, prediction_time DESC);
```

### API

```python
# Use connection pooling
# Already configured in src/database/connection.py

# Enable Redis caching for predictions
# (requires additional implementation)
```

## Roadmap

### Completed ✅
- [x] Web dashboard with Dash/Plotly
- [x] Historical data import from EVE Ref
- [x] Bulk prediction for all tracked items
- [x] Interactive price charts and visualizations
- [x] Docker containerization (6 services)
- [x] LSTM model training pipeline
- [x] Real-time data collection from ESI
- [x] Model drift detection with Evidently

### In Progress 🚧
- [ ] Backtesting framework integration
- [ ] Prediction confidence intervals
- [ ] Automated drift detection scheduling

### Planned 📋
- [ ] Multi-region price correlation analysis
- [ ] Automated hyperparameter tuning
- [ ] Real-time WebSocket predictions
- [ ] Mobile app API endpoints
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
