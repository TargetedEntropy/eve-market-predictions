# Technology Stack - EVE Online Price Prediction Engine

> **Design document, not a status report.** This file describes the intended
> architecture. Substantial parts of it are still roadmap. Where the two
> disagree, **README.md describes what actually exists**; see
> **AUDIT_REMEDIATION.md** for the current implementation status.
>
> Known to be design-only at the time of writing:
> ARIMA/XGBoost and other alternative models, hyperparameter search
> (`ParameterGrid`), `vectorbt` backtesting and trading metrics,
> email/Slack/PagerDuty alerting beyond the single webhook hook,
> whale/large-order detection, ±0.5/1/2/5% order-depth levels, patch-cycle and
> regional-hub features, the data-validation layer (gap detection, >3σ anomaly
> flags), and the Grafana/Prometheus/ELK observability stack.
>
> Since implemented: the Redis caching layer, automated weekly retraining,
> prediction persistence, `actual_price` backfill, the `model_performance`
> table, prediction-error monitoring, and an MLflow registered model per item.

**Last Updated:** 2025-10-26
**Status:** Production-Ready Recommendations

This document identifies all technologies, libraries, tools, and methods required to implement the EVE Online Price Prediction Engine, with specific versions, installation instructions, comparisons, and best practices.

---

## Table of Contents

1. [Core Technologies](#1-core-technologies)
2. [Machine Learning & Deep Learning](#2-machine-learning--deep-learning)
3. [Data Collection & Async HTTP](#3-data-collection--async-http)
4. [Database Layer](#4-database-layer)
5. [Caching & Rate Limiting](#5-caching--rate-limiting)
6. [Feature Engineering](#6-feature-engineering)
7. [Data Preprocessing](#7-data-preprocessing)
8. [API Framework & Serving](#8-api-framework--serving)
9. [Backtesting Frameworks](#9-backtesting-frameworks)
10. [MLOps & Model Monitoring](#10-mlops--model-monitoring)
11. [Task Scheduling](#11-task-scheduling)
12. [Containerization & Deployment](#12-containerization--deployment)
13. [Visualization & Analysis](#13-visualization--analysis)
14. [Development Tools](#14-development-tools)
15. [Complete Installation Guide](#15-complete-installation-guide)

---

## 1. Core Technologies

### Python Version

**Recommended: Python 3.11**

**Rationale:**
- Mature ecosystem with full ML library support (PyTorch 2.x supports 3.11+)
- Performance improvements over 3.10 (up to 25% faster)
- Excellent asyncio support
- Stable and well-tested for production

**Installation:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev

# macOS (using Homebrew)
brew install python@3.11

# Verify installation
python3.11 --version
```

**Alternative: Python 3.12**
- Latest features, but some ML libraries may lag in support
- Use if bleeding-edge features are needed

### Virtual Environment

**Tool: venv (built-in)**

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Verify
which python
python --version
```

**Alternative: conda**
- Better for complex dependency management
- Heavier weight, slower environment creation
- Recommended if managing multiple Python/CUDA versions

```bash
conda create -n eve-predictor python=3.11
conda activate eve-predictor
```

---

## 2. Machine Learning & Deep Learning

### Primary Framework: PyTorch

**Version: 2.4+ (Latest stable as of 2025)**

**Why PyTorch:**
- ✅ Excellent for LSTM/RNN architectures
- ✅ Dynamic computational graphs (easier debugging)
- ✅ Strong community support for time series
- ✅ Native asyncio compatibility
- ✅ TorchScript for production deployment

**Installation:**

**CPU-only:**
```bash
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**GPU (CUDA 12.1):**
```bash
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Verify installation:**
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

### Alternative: TensorFlow/Keras

**Version: TensorFlow 2.17+**

**When to use:**
- Familiar with Keras API
- Need TensorFlow Lite for mobile deployment
- Prefer static graphs for production optimization

**Not recommended for this project:**
- PyTorch has better time series ecosystem
- More complex deployment pipeline

---

## 3. Data Collection & Async HTTP

### Primary: httpx

**Version: 0.27+**

**Why httpx:**
- ✅ Full asyncio support
- ✅ HTTP/2 support
- ✅ Clean API similar to requests
- ✅ Better timeout handling
- ✅ Connection pooling out of the box

**Installation:**
```bash
pip install httpx[http2]==0.27.0
```

**Example Usage:**
```python
import httpx
import asyncio

async def fetch_market_data(region_id: int, type_id: int):
    async with httpx.AsyncClient(timeout=10.0) as client:
        url = f"https://esi.evetech.net/latest/markets/{region_id}/history/"
        params = {'type_id': type_id}

        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()

# Run
data = asyncio.run(fetch_market_data(10000002, 34))
```

### Alternative: aiohttp

**Version: 3.10+**

**Pros:**
- Slightly faster for high-concurrency scenarios
- More mature asyncio library
- Built-in websocket support

**Cons:**
- More complex API
- Less intuitive error handling

**Installation:**
```bash
pip install aiohttp==3.10.0
```

**Use case:** Choose if you need WebSocket support or have existing aiohttp codebase.

### Rate Limiting

**Library: asyncio-redis-rate-limit**

**Version: 0.1.0+**

**Installation:**
```bash
pip install asyncio-redis-rate-limit
```

**Example:**
```python
from asyncio_redis_rate_limit import RateLimiter
import redis.asyncio as aioredis

redis = await aioredis.from_url("redis://localhost")
limiter = RateLimiter(redis, limit=100, period=60)  # 100 requests/60 seconds

async def rate_limited_request():
    async with limiter:
        # Make API call
        return await fetch_market_data(10000002, 34)
```

**Alternative: aiometer**

**Installation:**
```bash
pip install aiometer
```

**Example:**
```python
import aiometer

async def fetch_all_items(item_ids):
    results = await aiometer.run_on_each(
        fetch_market_data,
        item_ids,
        max_per_second=20,  # Rate limit
        max_at_once=10      # Concurrency limit
    )
    return results
```

**Comparison:**
- **asyncio-redis-rate-limit:** Distributed rate limiting across multiple processes
- **aiometer:** In-memory, single-process rate limiting (simpler setup)

**Recommendation:** Use **aiometer** for development, **asyncio-redis-rate-limit** for production.

---

## 4. Database Layer

### PostgreSQL

**Version: 15+**

**Installation (Ubuntu):**
```bash
sudo apt update
sudo apt install postgresql-15 postgresql-contrib-15
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

**Create database:**
```bash
sudo -u postgres createuser eve_user -P  # Set password
sudo -u postgres createdb -O eve_user eve_markets
```

### TimescaleDB Extension

**Version: 2.17+**

**Why TimescaleDB:**
- ✅ Built on PostgreSQL (SQL familiarity)
- ✅ 10-100x faster for time series queries
- ✅ Automatic partitioning (hypertables)
- ✅ Advanced compression (save 90%+ storage)
- ✅ Continuous aggregates for real-time analytics

**Installation:**
```bash
# Add TimescaleDB repository
sudo sh -c "echo 'deb https://packagecloud.io/timescale/timescaledb/ubuntu/ $(lsb_release -c -s) main' > /etc/apt/sources.list.d/timescaledb.list"
wget --quiet -O - https://packagecloud.io/timescale/timescaledb/gpgkey | sudo apt-key add -

# Install
sudo apt update
sudo apt install timescaledb-2-postgresql-15

# Configure
sudo timescaledb-tune

# Restart PostgreSQL
sudo systemctl restart postgresql
```

**Enable in database:**
```sql
-- Connect to database
psql -U eve_user -d eve_markets

-- Enable extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create hypertable
CREATE TABLE market_history (
    time TIMESTAMPTZ NOT NULL,
    region_id INTEGER NOT NULL,
    type_id INTEGER NOT NULL,
    average NUMERIC(20,2),
    volume BIGINT,
    PRIMARY KEY (time, region_id, type_id)
);

SELECT create_hypertable('market_history', 'time');

-- Enable compression
ALTER TABLE market_history SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'region_id, type_id'
);

-- Auto-compress data older than 7 days
SELECT add_compression_policy('market_history', INTERVAL '7 days');
```

### SQLAlchemy (Async)

**Version: 2.0+**

**Installation:**
```bash
pip install sqlalchemy[asyncio]==2.0.25
pip install asyncpg==0.29.0  # PostgreSQL async driver
```

**Setup:**
```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, Numeric, DateTime

# Connection string
DATABASE_URL = "postgresql+asyncpg://eve_user:password@localhost:5432/eve_markets"

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_size=20,              # Connection pool size
    max_overflow=10,           # Max connections above pool_size
    pool_pre_ping=True,        # Verify connections before using
    pool_recycle=3600          # Recycle connections every hour
)

# Session factory
async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Base model
Base = declarative_base()

# Example model
class MarketHistory(Base):
    __tablename__ = 'market_history'

    time = Column(DateTime, primary_key=True)
    region_id = Column(Integer, primary_key=True)
    type_id = Column(Integer, primary_key=True)
    average = Column(Numeric(20, 2))
    volume = Column(Integer)

# Usage
async def insert_market_data(data):
    async with async_session() as session:
        async with session.begin():
            session.add(MarketHistory(**data))
            await session.commit()
```

**Key Features:**
- **AsyncEngine:** Uses asyncio-compatible connection pool (`AsyncAdaptedQueuePool`)
- **Connection Pooling:** QueuePool manages connections efficiently
- **asyncpg driver:** Fastest PostgreSQL driver for Python

**Alternative: psycopg3**
- Newer driver, excellent async support
- Less mature than asyncpg
- Use if you need better PostgreSQL 15+ feature support

```bash
pip install psycopg[binary,pool]==3.1.16
# Connection string: postgresql+psycopg://...
```

---

## 5. Caching & Rate Limiting

### Redis

**Version: 7+**

**Installation (Ubuntu):**
```bash
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Test
redis-cli ping  # Should return PONG
```

**Docker (Alternative):**
```bash
docker run -d -p 6379:6379 --name redis redis:7-alpine
```

### Python Redis Client: redis-py (async)

**Version: 5.0+**

**Installation:**
```bash
pip install redis[hiredis]==5.0.1
```

**Example:**
```python
import redis.asyncio as aioredis
import json

class RedisCache:
    def __init__(self):
        self.redis = None

    async def connect(self):
        self.redis = await aioredis.from_url(
            "redis://localhost:6379",
            encoding="utf-8",
            decode_responses=True
        )

    async def get_market_data(self, key: str):
        data = await self.redis.get(key)
        return json.loads(data) if data else None

    async def set_market_data(self, key: str, value: dict, ttl: int = 300):
        await self.redis.setex(key, ttl, json.dumps(value))

    async def close(self):
        await self.redis.close()

# Usage
cache = RedisCache()
await cache.connect()

# Cache API response for 5 minutes
await cache.set_market_data("market:10000002:34", data, ttl=300)
cached = await cache.get_market_data("market:10000002:34")
```

### Alternative: aiocache

**Version: 0.12+**

**Pros:**
- Higher-level abstraction
- Decorator support for caching functions
- Multiple backends (Redis, Memcached, Memory)

**Installation:**
```bash
pip install aiocache[redis]==0.12.2
```

**Example:**
```python
from aiocache import caches, cached
from aiocache.serializers import JsonSerializer

caches.set_config({
    'default': {
        'cache': "aiocache.RedisCache",
        'endpoint': "localhost",
        'port': 6379,
        'serializer': {'class': "aiocache.serializers.JsonSerializer"}
    }
})

@cached(ttl=300, key="market:{region_id}:{type_id}")
async def fetch_market_data(region_id: int, type_id: int):
    # Automatically cached
    return await api_call(region_id, type_id)
```

**Recommendation:** Use **aiocache** for simplicity, **redis-py** for fine-grained control.

---

## 6. Feature Engineering

### Technical Indicators: pandas-ta

**Version: 0.3.14+**

**Why pandas-ta:**
- ✅ 150+ indicators (RSI, MACD, Bollinger Bands, etc.)
- ✅ Pure Python (no C dependencies like TA-Lib)
- ✅ Pandas DataFrame integration
- ✅ Easy installation
- ✅ Actively maintained

**Installation:**
```bash
pip install pandas-ta==0.3.14b0
```

**Example Usage:**
```python
import pandas as pd
import pandas_ta as ta

# Load price data
df = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})

# Calculate indicators
df['RSI'] = ta.rsi(df['close'], length=14)
df['SMA_20'] = ta.sma(df['close'], length=20)
df['EMA_12'] = ta.ema(df['close'], length=12)

# MACD
macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
df = pd.concat([df, macd], axis=1)

# Bollinger Bands
bbands = ta.bbands(df['close'], length=20, std=2)
df = pd.concat([df, bbands], axis=1)

# Strategy - apply multiple indicators at once
strategy = ta.Strategy(
    name="EVE Market Strategy",
    ta=[
        {"kind": "rsi", "length": 14},
        {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
        {"kind": "sma", "length": 20},
        {"kind": "ema", "length": 12}
    ]
)

df.ta.strategy(strategy)
```

### Alternative: TA-Lib

**Version: 0.4.28+**

**Pros:**
- Industry standard
- Extremely fast (C implementation)
- 200+ indicators

**Cons:**
- Difficult installation (requires C compiler)
- Less Pythonic API

**Installation (challenging):**
```bash
# Ubuntu - install dependencies first
sudo apt install build-essential wget
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install

# Then install Python wrapper
pip install TA-Lib
```

**Example:**
```python
import talib
import numpy as np

close = np.array([...])

rsi = talib.RSI(close, timeperiod=14)
sma = talib.SMA(close, timeperiod=20)
macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
upper, middle, lower = talib.BBANDS(close, timeperiod=20)
```

**Recommendation:** Use **pandas-ta** unless you need absolute maximum performance.

### Custom Feature Engineering

**Libraries needed:**
```bash
pip install pandas==2.2.0
pip install numpy==1.26.3
pip install scipy==1.12.0
```

**Example custom features:**
```python
import pandas as pd
import numpy as np

def compute_returns(prices: pd.Series) -> pd.Series:
    """Calculate percentage returns"""
    return prices.pct_change()

def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Calculate log returns (better for ML)"""
    return np.log(prices / prices.shift(1))

def compute_volatility(returns: pd.Series, window: int = 14) -> pd.Series:
    """Rolling volatility"""
    return returns.rolling(window=window).std()

def compute_momentum(prices: pd.Series, periods: int = 10) -> pd.Series:
    """Price momentum"""
    return prices.diff(periods)

def encode_cyclic_time(hour: int) -> tuple:
    """Cyclic encoding for time features"""
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    return hour_sin, hour_cos
```

---

## 7. Data Preprocessing

### Scikit-learn

**Version: 1.4+**

**Installation:**
```bash
pip install scikit-learn==1.4.0
```

### Scalers for Time Series

**MinMaxScaler vs StandardScaler**

#### MinMaxScaler

**When to use:**
- Neural networks (LSTM) that expect input in [0,1] or [-1,1]
- Data without extreme outliers
- Need bounded output range

**Example:**
```python
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Fit on training data ONLY
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data.reshape(-1, 1))

# Transform test data using same scaler
test_scaled = scaler.transform(test_data.reshape(-1, 1))

# Inverse transform predictions
predictions = model.predict(test_scaled)
predictions_original = scaler.inverse_transform(predictions)
```

**Critical for time series:**
```python
# WRONG - causes data leakage
scaler.fit(all_data)

# CORRECT - only fit on training data
scaler.fit(train_data)
test_scaled = scaler.transform(test_data)
```

#### StandardScaler

**When to use:**
- Data with outliers (more robust)
- Linear models (regression, SVM)
- Want to preserve outlier information

**Example:**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data.reshape(-1, 1))
test_scaled = scaler.transform(test_data.reshape(-1, 1))

# Mean = 0, Std = 1
print(f"Mean: {train_scaled.mean():.6f}")  # ~0
print(f"Std: {train_scaled.std():.6f}")    # ~1
```

**Recommendation for LSTM:** Use **MinMaxScaler** with `feature_range=(0, 1)`

### Handling Multiple Features

```python
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Multiple features
features = ['price', 'volume', 'rsi', 'macd']
df_train = pd.DataFrame(...)  # Training data
df_test = pd.DataFrame(...)   # Test data

scaler = MinMaxScaler()
df_train_scaled = pd.DataFrame(
    scaler.fit_transform(df_train[features]),
    columns=features,
    index=df_train.index
)

df_test_scaled = pd.DataFrame(
    scaler.transform(df_test[features]),
    columns=features,
    index=df_test.index
)
```

### Alternative: RobustScaler

**When to use:** Data with many outliers

```python
from sklearn.preprocessing import RobustScaler

# Uses median and IQR (robust to outliers)
scaler = RobustScaler()
train_scaled = scaler.fit_transform(train_data.reshape(-1, 1))
```

---

## 8. API Framework & Serving

### FastAPI

**Version: 0.115+**

**Why FastAPI:**
- ✅ Async/await native support
- ✅ Automatic API documentation (Swagger/ReDoc)
- ✅ Pydantic validation
- ✅ Best performance among Python frameworks
- ✅ Easy dependency injection

**Installation:**
```bash
pip install fastapi==0.115.0
pip install uvicorn[standard]==0.32.0  # ASGI server
```

**Example API:**
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime

app = FastAPI(title="EVE Price Predictor")

class PredictionRequest(BaseModel):
    type_id: int
    region_id: int
    horizon_hours: int = 24

class PredictionResponse(BaseModel):
    type_id: int
    current_price: float
    predicted_price: float
    confidence: float
    timestamp: datetime

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(request: PredictionRequest):
    try:
        # Load model and make prediction
        prediction = await make_prediction(
            request.type_id,
            request.region_id,
            request.horizon_hours
        )
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

# Run with: uvicorn main:app --reload
```

### Production Deployment

**Option 1: Uvicorn with Workers**

```bash
# Single process
uvicorn main:app --host 0.0.0.0 --port 8000

# Multiple workers (production)
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Option 2: Gunicorn + Uvicorn Workers (Recommended)**

```bash
pip install gunicorn==21.2.0
```

```bash
# gunicorn with uvicorn workers
gunicorn main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile -
```

**Configuration file:** `gunicorn_conf.py`
```python
import multiprocessing

# Server socket
bind = "0.0.0.0:8000"

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"

# Timeouts
timeout = 120
keepalive = 5

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
```

Run: `gunicorn -c gunicorn_conf.py main:app`

### Alternative: Hypercorn

**For HTTP/2 support:**
```bash
pip install hypercorn==0.16.0
hypercorn main:app --bind 0.0.0.0:8000
```

---

## 9. Backtesting Frameworks

### Comparison: VectorBT vs Backtrader

| Feature | VectorBT | Backtrader |
|---------|----------|------------|
| **Speed** | ⚡ 100-1000x faster | 🐢 Slower (event-driven) |
| **Vectorization** | ✅ NumPy/Numba | ❌ Loop-based |
| **Learning Curve** | 📈 Steeper | 📉 Gentler |
| **Live Trading** | ❌ Backtest only | ✅ Supported |
| **Multi-strategy** | ✅ Thousands at once | ❌ One at a time |
| **Cost** | 💰 Free (limited), Pro ($$$) | 💚 Free (open source) |
| **Maintenance** | ⚠️ Free version unmaintained | ⚠️ Community fork only |

### VectorBT

**Version: 0.26+ (Free), VectorBT PRO (Paid)**

**When to use:**
- Need to test hundreds/thousands of parameter combinations
- Have large datasets (years of minute data)
- Focus on systematic/quantitative strategies
- Comfortable with NumPy/Pandas

**Installation:**
```bash
pip install vectorbt==0.26.0
```

**Example:**
```python
import vectorbt as vbt
import pandas as pd

# Load data
price_data = pd.Series([...], index=pd.DatetimeIndex([...]))

# Define entry/exit signals
fast_ma = price_data.rolling(window=10).mean()
slow_ma = price_data.rolling(window=30).mean()

entries = fast_ma > slow_ma
exits = fast_ma < slow_ma

# Run backtest
portfolio = vbt.Portfolio.from_signals(
    price_data,
    entries,
    exits,
    init_cash=1_000_000_000,  # 1B ISK
    fees=0.025,               # 2.5% broker fee
    slippage=0.001            # 0.1% slippage
)

# Performance metrics
print(portfolio.stats())
print(f"Total Return: {portfolio.total_return()}")
print(f"Sharpe Ratio: {portfolio.sharpe_ratio()}")
print(f"Max Drawdown: {portfolio.max_drawdown()}")

# Plot
portfolio.plot().show()
```

**Pros:**
- Extremely fast for large-scale backtesting
- Built-in performance metrics
- Excellent visualization

**Cons:**
- Free version no longer developed
- Pro version is expensive ($300-500/year)
- Harder to customize complex strategies

### Backtrader

**Version: 1.9.78+ (use backtrader2 fork)**

**When to use:**
- Need live trading integration
- Prefer object-oriented strategy design
- Want fine-grained control over execution
- Plan to connect to real brokers later

**Installation:**
```bash
# Original (unmaintained)
pip install backtrader

# Maintained fork
pip install backtrader2
```

**Example:**
```python
import backtrader as bt

class MAStrategy(bt.Strategy):
    params = (
        ('fast', 10),
        ('slow', 30),
    )

    def __init__(self):
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.params.fast)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.params.slow)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)

    def next(self):
        if self.crossover > 0:  # Fast crosses above slow
            self.buy()
        elif self.crossover < 0:  # Fast crosses below slow
            self.close()

# Run backtest
cerebro = bt.Cerebro()
cerebro.addstrategy(MAStrategy)

# Add data
data = bt.feeds.PandasData(dataname=df)
cerebro.adddata(data)

# Set initial cash and commission
cerebro.broker.setcash(1_000_000_000.0)
cerebro.broker.setcommission(commission=0.025)

# Run
cerebro.run()
cerebro.plot()
```

**Pros:**
- More intuitive for discretionary traders
- Live trading support
- Large ecosystem of indicators

**Cons:**
- Much slower (1000x vs VectorBT)
- No longer actively maintained (use fork)
- Can't test multiple strategies simultaneously

### Recommendation

**For EVE Price Prediction Engine:**

**Use VectorBT if:**
- Testing ML model predictions against buy/hold
- Need to optimize hyperparameters via backtest
- Working with large datasets

**Use Backtrader if:**
- Planning to implement live automated trading
- Need complex order types
- Prefer OOP strategy design

**Hybrid approach:** Use VectorBT for rapid strategy development, then implement final strategy in Backtrader for live trading.

### Alternative: bt (Flexible Backtesting Framework)

```bash
pip install bt==0.2.9
```

Simpler than both, good for portfolio-level backtesting.

---

## 10. MLOps & Model Monitoring

### Experiment Tracking: MLflow

**Version: 2.17+**

**Why MLflow:**
- ✅ Track experiments (hyperparameters, metrics, artifacts)
- ✅ Model versioning and registry
- ✅ PyTorch native integration
- ✅ Deployment agnostic
- ✅ Free and open source

**Installation:**
```bash
pip install mlflow==2.17.0
```

**Setup Tracking Server:**
```bash
# Local tracking
mlflow ui

# Access at: http://localhost:5000
```

**PyTorch Integration:**
```python
import mlflow
import mlflow.pytorch
import torch

# Auto-logging
mlflow.pytorch.autolog()

# Start experiment
with mlflow.start_run(run_name="LSTM-v1"):
    # Log parameters
    mlflow.log_param("hidden_size", 128)
    mlflow.log_param("num_layers", 2)
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("lookback_window", 30)

    # Train model
    model = train_model(...)

    # Log metrics
    mlflow.log_metric("train_loss", train_loss)
    mlflow.log_metric("val_loss", val_loss)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("directional_accuracy", dir_acc)

    # Log model
    mlflow.pytorch.log_model(model, "model")

    # Log artifacts
    mlflow.log_artifact("training_plot.png")

# Load model later
model_uri = "runs:/<run_id>/model"
loaded_model = mlflow.pytorch.load_model(model_uri)
```

**Model Registry:**
```python
# Register model
model_uri = "runs:/<run_id>/model"
mlflow.register_model(model_uri, "EVE-LSTM-Predictor")

# Transition to production
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="EVE-LSTM-Predictor",
    version=1,
    stage="Production"
)

# Load production model
model = mlflow.pyfunc.load_model("models:/EVE-LSTM-Predictor/Production")
```

### Drift Detection: Evidently

**Version: 0.4.34+**

**Why Evidently:**
- ✅ 20+ drift detection methods
- ✅ Interactive HTML reports
- ✅ Real-time monitoring dashboards
- ✅ Statistical tests (KS, Chi-square, etc.)
- ✅ Open source

**Installation:**
```bash
pip install evidently==0.4.34
```

**Example - Data Drift Report:**
```python
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import pandas as pd

# Reference data (training)
reference_data = pd.DataFrame(...)

# Current data (production)
current_data = pd.DataFrame(...)

# Define column mapping
column_mapping = ColumnMapping(
    target='price',
    numerical_features=['volume', 'rsi', 'macd', 'spread'],
    categorical_features=['hour_of_day', 'day_of_week']
)

# Generate drift report
report = Report(metrics=[DataDriftPreset()])
report.run(
    reference_data=reference_data,
    current_data=current_data,
    column_mapping=column_mapping
)

# Save report
report.save_html("drift_report.html")

# Get drift results
report_dict = report.as_dict()
drift_detected = report_dict['metrics'][0]['result']['dataset_drift']
print(f"Drift detected: {drift_detected}")
```

**Example - Model Performance Monitoring:**
```python
from evidently.metric_preset import RegressionPreset

# Add predictions to data
reference_data['prediction'] = model.predict(reference_data[features])
current_data['prediction'] = model.predict(current_data[features])

# Generate performance report
perf_report = Report(metrics=[RegressionPreset()])
perf_report.run(
    reference_data=reference_data,
    current_data=current_data,
    column_mapping=column_mapping
)

perf_report.save_html("performance_report.html")
```

**Test Suite (for CI/CD):**
```python
from evidently.test_suite import TestSuite
from evidently.tests import TestShareOfDriftedColumns

test_suite = TestSuite(tests=[
    TestShareOfDriftedColumns(lte=0.3)  # Max 30% columns drifted
])

test_suite.run(reference_data, current_data)

if test_suite.as_dict()['summary']['all_passed']:
    print("✅ All drift tests passed")
else:
    print("❌ Drift detected - investigate!")
```

### Alternative: NannyML

**For performance monitoring without ground truth:**
```bash
pip install nannyml
```

Good for detecting performance degradation before labels arrive.

---

## 11. Task Scheduling

### Comparison: APScheduler vs Celery

| Feature | APScheduler | Celery |
|---------|-------------|--------|
| **Setup Complexity** | 🟢 Simple | 🔴 Complex |
| **Message Broker** | ❌ Not required | ✅ Required (Redis/RabbitMQ) |
| **Distributed** | ❌ Single process | ✅ Multi-worker |
| **Async Support** | ✅ Native | ✅ Supported |
| **Scheduling Types** | ✅ Cron, interval, date | ✅ Cron, interval |
| **Best For** | Single-server apps | Distributed systems |

### APScheduler (Recommended)

**Version: 3.10+**

**When to use:**
- Single-server deployment
- Simple periodic tasks (data collection)
- Async application
- Don't need distributed task queue

**Installation:**
```bash
pip install apscheduler==3.10.4
```

**Example - Asyncio Integration:**
```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
import asyncio

scheduler = AsyncIOScheduler()

async def collect_market_data():
    """Collect market data every 5 minutes"""
    print("Collecting market data...")
    # Fetch from ESI API
    await fetch_all_market_orders()

async def compute_features():
    """Compute features every hour"""
    print("Computing features...")
    await compute_technical_indicators()

async def retrain_model():
    """Retrain model weekly"""
    print("Retraining model...")
    await train_lstm_model()

# Add jobs
scheduler.add_job(
    collect_market_data,
    IntervalTrigger(minutes=5),
    id='market_collector',
    name='Market Data Collection'
)

scheduler.add_job(
    compute_features,
    IntervalTrigger(hours=1),
    id='feature_computer',
    name='Feature Computation'
)

scheduler.add_job(
    retrain_model,
    'cron',
    day_of_week='sun',
    hour=2,
    id='model_retrainer',
    name='Weekly Model Retraining'
)

# Start scheduler
scheduler.start()

# Keep running
try:
    asyncio.get_event_loop().run_forever()
except (KeyboardInterrupt, SystemExit):
    scheduler.shutdown()
```

**Persistence (survive restarts):**
```python
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore

jobstores = {
    'default': SQLAlchemyJobStore(url='sqlite:///jobs.sqlite')
}

scheduler = AsyncIOScheduler(jobstores=jobstores)
```

### Celery

**Version: 5.4+**

**When to use:**
- Multi-server deployment
- Need distributed task queue
- Complex workflows with task dependencies
- Already using Redis/RabbitMQ

**Installation:**
```bash
pip install celery[redis]==5.4.0
```

**Example:**
```python
# celery_app.py
from celery import Celery
from celery.schedules import crontab

app = Celery('eve_predictor', broker='redis://localhost:6379/0')

app.conf.beat_schedule = {
    'collect-market-data': {
        'task': 'tasks.collect_market_data',
        'schedule': 300.0,  # Every 5 minutes
    },
    'retrain-model': {
        'task': 'tasks.retrain_model',
        'schedule': crontab(hour=2, day_of_week='sunday'),
    },
}

# tasks.py
from celery_app import app

@app.task
def collect_market_data():
    # Sync task (Celery doesn't handle async well)
    return fetch_data()

@app.task
def retrain_model():
    return train_model()
```

**Run:**
```bash
# Start worker
celery -A celery_app worker --loglevel=info

# Start beat scheduler (separate process)
celery -A celery_app beat --loglevel=info
```

**Cons:**
- More complex setup
- Two processes (worker + beat)
- Not ideal for async code

**Recommendation:** Use **APScheduler** for this project (simpler, async-native).

---

## 12. Containerization & Deployment

### Docker

**Version: 24+**

**Installation (Ubuntu):**
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt install docker-compose-plugin

# Verify
docker --version
docker compose version
```

### Dockerfile for PyTorch (CPU)

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Dockerfile for PyTorch (GPU/CUDA)

```dockerfile
# Use official PyTorch image with CUDA
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

# Install additional dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Alternative - NVIDIA CUDA base:**
```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA
RUN pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# ... rest of Dockerfile
```

### Docker Compose (Full Stack)

```yaml
# docker-compose.yml
version: '3.8'

services:
  # PostgreSQL + TimescaleDB
  postgres:
    image: timescale/timescaledb:latest-pg15
    environment:
      POSTGRES_USER: eve_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: eve_markets
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U eve_user"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5

  # Data Collector
  collector:
    build:
      context: .
      dockerfile: Dockerfile
    command: python collector.py
    environment:
      DATABASE_URL: postgresql+asyncpg://eve_user:${DB_PASSWORD}@postgres:5432/eve_markets
      REDIS_URL: redis://redis:6379
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped

  # API Server
  api:
    build:
      context: .
      dockerfile: Dockerfile
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql+asyncpg://eve_user:${DB_PASSWORD}@postgres:5432/eve_markets
      REDIS_URL: redis://redis:6379
      MODEL_PATH: /app/models/best_model.pth
    volumes:
      - ./models:/app/models
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped

  # MLflow Tracking Server
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.17.0
    command: mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root /mlflow/artifacts
    ports:
      - "5000:5000"
    volumes:
      - mlflow_data:/mlflow
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  mlflow_data:
```

**Environment file:** `.env`
```bash
DB_PASSWORD=your_secure_password
```

**Run:**
```bash
docker compose up -d
docker compose logs -f api
```

### GPU Support

**Install NVIDIA Container Toolkit:**
```bash
# Add repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

**Docker Compose with GPU:**
```yaml
services:
  api:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

**Test GPU:**
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

---

## 13. Visualization & Analysis

### Interactive Dashboards: Dash/Plotly

**Version: Dash 2.18+, Plotly 5.24+**

**Installation:**
```bash
pip install dash==2.18.0
pip install plotly==5.24.0
```

**Example Dashboard:**
```python
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd

app = Dash(__name__)

app.layout = html.Div([
    html.H1("EVE Market Prediction Dashboard"),

    dcc.Dropdown(
        id='item-selector',
        options=[
            {'label': 'PLEX', 'value': 44992},
            {'label': 'Skill Injector', 'value': 40519}
        ],
        value=44992
    ),

    dcc.Graph(id='price-chart'),
    dcc.Graph(id='prediction-chart'),

    dcc.Interval(
        id='interval-component',
        interval=60*1000,  # Update every minute
        n_intervals=0
    )
])

@app.callback(
    Output('price-chart', 'figure'),
    Input('item-selector', 'value'),
    Input('interval-component', 'n_intervals')
)
def update_price_chart(type_id, n):
    # Fetch latest data
    df = fetch_price_history(type_id)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['average'],
        mode='lines',
        name='Actual Price'
    ))

    fig.update_layout(
        title=f"Price History - Type ID {type_id}",
        xaxis_title="Date",
        yaxis_title="Price (ISK)",
        template="plotly_dark"
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
```

### Jupyter Lab

**Version: 4.2+**

**Installation:**
```bash
pip install jupyterlab==4.2.0
pip install ipywidgets==8.1.2  # Interactive widgets
```

**Run:**
```bash
jupyter lab
```

**Useful Extensions:**
```bash
# Variable inspector
pip install lckr-jupyterlab-variableinspector

# Code formatter
pip install jupyterlab-code-formatter black

# Git integration
pip install jupyterlab-git
```

### Static Plots: Matplotlib/Seaborn

```bash
pip install matplotlib==3.9.0
pip install seaborn==0.13.2
```

**Example:**
```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")

# Plot price vs prediction
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(dates, actual_prices, label='Actual', linewidth=2)
ax.plot(dates, predicted_prices, label='Predicted', linewidth=2, linestyle='--')
ax.fill_between(dates, lower_bound, upper_bound, alpha=0.2, label='Confidence Interval')
ax.set_xlabel('Date')
ax.set_ylabel('Price (ISK)')
ax.set_title('Price Prediction - PLEX')
ax.legend()
plt.savefig('prediction_plot.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## 14. Development Tools

### Code Quality

**Formatter: Black**
```bash
pip install black==24.10.0
black .
```

**Linter: Ruff (faster than Flake8)**
```bash
pip install ruff==0.7.0
ruff check .
ruff check --fix .
```

**Type Checking: mypy**
```bash
pip install mypy==1.13.0
mypy --install-types
mypy .
```

### Testing

**pytest**
```bash
pip install pytest==8.3.0
pip install pytest-asyncio==0.24.0  # For async tests
pip install pytest-cov==6.0.0       # Coverage
```

**Example test:**
```python
# test_predictor.py
import pytest
from predictor import make_prediction

@pytest.mark.asyncio
async def test_prediction():
    result = await make_prediction(type_id=34, region_id=10000002)
    assert result['predicted_price'] > 0
    assert 'confidence' in result

# Run tests
# pytest --cov=. --cov-report=html
```

### IDE Recommendations

1. **VS Code** (Recommended)
   - Extensions: Python, Pylance, Jupyter, Docker
   - Free, excellent Python support

2. **PyCharm Professional**
   - Best for data science (Scientific Mode)
   - Database tools built-in
   - Paid ($200/year, free for students)

3. **Jupyter Lab**
   - For exploratory analysis only

### Environment Management

**.env file management:**
```bash
pip install python-dotenv==1.0.1
```

```python
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
REDIS_URL = os.getenv("REDIS_URL")
```

---

## 15. Complete Installation Guide

### Step 1: System Setup (Ubuntu 22.04)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install build essentials
sudo apt install -y build-essential curl wget git

# Install Python 3.11
sudo apt install -y python3.11 python3.11-venv python3.11-dev

# Install PostgreSQL 15
sudo apt install -y postgresql-15 postgresql-contrib-15

# Install Redis
sudo apt install -y redis-server

# Install TimescaleDB
sudo sh -c "echo 'deb https://packagecloud.io/timescale/timescaledb/ubuntu/ $(lsb_release -c -s) main' > /etc/apt/sources.list.d/timescaledb.list"
wget --quiet -O - https://packagecloud.io/timescale/timescaledb/gpgkey | sudo apt-key add -
sudo apt update
sudo apt install -y timescaledb-2-postgresql-15
sudo timescaledb-tune --quiet --yes
sudo systemctl restart postgresql
```

### Step 2: Database Setup

```bash
# Create database user and database
sudo -u postgres psql << EOF
CREATE USER eve_user WITH PASSWORD 'your_secure_password';
CREATE DATABASE eve_markets OWNER eve_user;
\c eve_markets
CREATE EXTENSION IF NOT EXISTS timescaledb;
GRANT ALL PRIVILEGES ON DATABASE eve_markets TO eve_user;
EOF
```

### Step 3: Python Environment

```bash
# Create project directory
mkdir -p ~/eve-discovery
cd ~/eve-discovery

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Step 4: Install Python Packages

**Create `requirements.txt`:**
```txt
# Core
python-dotenv==1.0.1

# Web Framework
fastapi==0.115.0
uvicorn[standard]==0.32.0
gunicorn==21.2.0
pydantic==2.9.0

# Async HTTP
httpx[http2]==0.27.0
aiohttp==3.10.0

# Database
sqlalchemy[asyncio]==2.0.25
asyncpg==0.29.0
psycopg[binary,pool]==3.1.16

# Caching & Rate Limiting
redis[hiredis]==5.0.1
aiocache[redis]==0.12.2
asyncio-redis-rate-limit==0.1.0
aiometer==0.5.0

# ML & Deep Learning
torch==2.4.0
numpy==1.26.3
scipy==1.12.0
scikit-learn==1.4.0

# Data Processing
pandas==2.2.0
pandas-ta==0.3.14b0

# MLOps
mlflow==2.17.0
evidently==0.4.34

# Backtesting
vectorbt==0.26.0

# Task Scheduling
apscheduler==3.10.4

# Visualization
plotly==5.24.0
dash==2.18.0
matplotlib==3.9.0
seaborn==0.13.2
jupyterlab==4.2.0

# Development
pytest==8.3.0
pytest-asyncio==0.24.0
pytest-cov==6.0.0
black==24.10.0
ruff==0.7.0
mypy==1.13.0
```

**Install:**
```bash
pip install -r requirements.txt
```

### Step 5: Docker Setup (Optional but Recommended)

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt install -y docker-compose-plugin

# Log out and back in for group changes
```

### Step 6: Verify Installation

```bash
# Test Python
python --version  # Should be 3.11.x

# Test PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Test database connection
python -c "
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine

async def test():
    engine = create_async_engine('postgresql+asyncpg://eve_user:your_secure_password@localhost/eve_markets')
    async with engine.begin() as conn:
        result = await conn.execute('SELECT version()')
        print(result.scalar())

asyncio.run(test())
"

# Test Redis
redis-cli ping  # Should return PONG
```

### Step 7: Project Structure

```bash
eve-discovery/
├── .env                      # Environment variables
├── .gitignore
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── README.md
├── CLAUDE.md
├── PRICE_PREDICTION_ENGINE.md
├── TECHNOLOGY_STACK.md
│
├── src/
│   ├── __init__.py
│   ├── config.py            # Configuration management
│   ├── database/
│   │   ├── __init__.py
│   │   ├── models.py        # SQLAlchemy models
│   │   └── connection.py    # Database connection
│   ├── collectors/
│   │   ├── __init__.py
│   │   ├── esi_client.py    # ESI API client
│   │   └── market_collector.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── technical_indicators.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm_model.py
│   │   └── train.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py          # FastAPI app (routes live here)
│   ├── dashboard/
│   │   ├── __init__.py
│   │   └── app.py           # Dash/Plotly dashboard
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── drift_detector.py
│   │   ├── evaluation.py    # prediction scoring -> model_performance
│   │   └── alerting.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── timeutils.py     # naive-UTC helpers
│   │   └── cache.py         # Redis ESI response/ETag cache
│   ├── constants.py         # TRACKED_ITEMS / REGIONS
│   └── config.py
│
├── tests/                   # pytest suite
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_collector.py
│   ├── test_config.py
│   ├── test_constants_and_dashboard.py
│   ├── test_database.py
│   ├── test_esi_client.py
│   ├── test_evaluation.py
│   ├── test_feature_engineering.py
│   ├── test_models.py
│   ├── test_scheduler.py
│   ├── test_technical_indicators.py
│   ├── test_timeutils.py
│   └── test_upsert.py
│
├── scripts/
│   ├── init_database.py     # schema + hypertables + policies (no .sql file)
│   ├── collect_data.py
│   ├── import_everef_data.py
│   ├── train_model.py
│   ├── predict_item.py
│   ├── predict_all_items.py
│   ├── evaluate_predictions.py
│   ├── check_drift.py
│   ├── check_model_metrics.py
│   └── scheduler_daemon.py
│
├── pyproject.toml           # pytest / black / ruff config
│
└── data/                    # Gitignored
    ├── raw/
    ├── processed/
    ├── models/              # model_{type_id}_{region_id}.pth + .json
    └── drift_reports/
```

There are no Jupyter notebooks, no `src/api/routes.py`, no `src/utils/metrics.py`,
no `scripts/setup_database.sql` and no `scripts/backfill_data.py` -- the tree
above is what exists. GitHub Actions CI, Grafana and Prometheus/ELK are also
not set up; they remain roadmap items.

---

## Summary: Recommended Tech Stack

### Core Stack

| Component | Technology | Version | Why |
|-----------|-----------|---------|-----|
| **Language** | Python | 3.11 | Best ML ecosystem |
| **ML Framework** | PyTorch | 2.4+ | LSTM support |
| **Database** | PostgreSQL + TimescaleDB | 15 + 2.17 | Time-series optimization |
| **Caching** | Redis | 7+ | Fast in-memory cache |
| **API Framework** | FastAPI | 0.115+ | Async, fast, modern |
| **HTTP Client** | httpx | 0.27+ | Async support |
| **Feature Engineering** | pandas-ta | 0.3.14 | Easy to use |
| **Backtesting** | VectorBT | 0.26+ | Fast vectorized |
| **MLOps** | MLflow | 2.17+ | Experiment tracking |
| **Monitoring** | Evidently | 0.4.34+ | Drift detection |
| **Scheduling** | APScheduler | 3.10+ | Async compatible |
| **Visualization** | Dash/Plotly | Latest | Interactive dashboards |
| **Containerization** | Docker | 24+ | Deployment |

### Total Installation Time

- **Manual setup:** 2-3 hours
- **Docker setup:** 30 minutes
- **Development environment:** 1 hour

### Estimated Costs

**Open Source (Free):**
- All core technologies: $0

**Optional Paid:**
- VectorBT PRO: $300-500/year (optional, free version works)
- Cloud hosting: ~$525/month (AWS as per previous estimate)
- PyCharm Professional: $200/year (optional, VS Code is free)

**Total minimum cost: $0 for development, ~$525/month for production cloud hosting**

---

## Next Steps

1. **Set up development environment** using Step-by-Step guide above
2. **Initialize database schema** from PRICE_PREDICTION_ENGINE.md
3. **Implement ESI API client** with rate limiting
4. **Start data collection** (historical + real-time)
5. **Build feature engineering pipeline**
6. **Develop LSTM model** with MLflow tracking
7. **Create backtesting framework**
8. **Deploy API** with Docker Compose
9. **Set up monitoring** with Evidently

Refer to **PRICE_PREDICTION_ENGINE.md** for detailed implementation roadmap.

---

**Document Version:** 1.0
**Last Updated:** 2025-10-26
**Maintainer:** EVE Discovery Project

