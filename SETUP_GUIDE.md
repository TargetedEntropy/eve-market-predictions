# EVE Discovery - Complete Setup Guide

This guide walks you through setting up the EVE Online Price Prediction Engine from scratch.

## Choose Your Setup Method

### Option 1: Docker (Recommended) ⭐
**Time:** ~15 minutes | **Difficulty:** Easy

Best for: Quick start, testing, production deployment

### Option 2: Local Development
**Time:** ~45 minutes | **Difficulty:** Medium

Best for: Active development, customization

---

## Option 1: Docker Setup

### Prerequisites

1. **Install Docker & Docker Compose**

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt install docker-compose-plugin

# Log out and back in for group changes to take effect
```

2. **Verify Installation**

```bash
docker --version
docker compose version
```

### Step 1: Clone & Configure

```bash
# Clone repository
cd ~/
git clone <your-repo> eve-discovery
cd eve-discovery

# Create environment file
cp .env.example .env

# IMPORTANT: Edit .env and set a secure password
nano .env
# Change DB_PASSWORD=changeme to a strong password
```

### Step 2: Start Services

```bash
# Start all containers
docker compose up -d

# Check services are running
docker compose ps

# View logs
docker compose logs -f
```

Expected output:
```
NAME              STATUS          PORTS
eve-postgres      Up (healthy)    5432/tcp
eve-redis         Up (healthy)    6379/tcp
eve-api           Up (healthy)    127.0.0.1:8000->8000/tcp
eve-collector     Up
eve-mlflow        Up (healthy)    127.0.0.1:5000->5000/tcp
eve-dashboard     Up (healthy)    127.0.0.1:8050->8050/tcp
```

Six services. Postgres and Redis are not published to the host at all; the
other three bind to loopback (override with `API_BIND` / `DASHBOARD_BIND`).

### Step 3: Initialize Database

```bash
# Initialize database schema
docker compose exec api python scripts/init_database.py
```

Expected output:
```
INFO - Creating database engine...
INFO - TimescaleDB extension enabled
INFO - Created hypertable: market_history
INFO - Created hypertable: order_snapshots
INFO - Database initialization complete
```

### Step 4: Collect Initial Data

```bash
# Run data collection (takes 5-10 minutes)
docker compose exec collector python scripts/collect_data.py
```

This will:
1. Fetch item metadata from ESI
2. Download historical market data (90 days)
3. Collect current order book snapshot
4. Calculate liquidity tiers

### Step 5: Verify Setup

```bash
# Check API health
curl http://localhost:8000/health

# View tracked items
curl http://localhost:8000/items

# Access MLflow UI
open http://localhost:5000
```

### Step 6: Access Services

- **API Documentation**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8050
- **MLflow Tracking**: http://localhost:5000
- **Health Check**: http://localhost:8000/health (503 until a model is trained)

### Next Steps

The collector daemon is now running and will (all times UTC):
- Collect order snapshots every 5 minutes
- Compute features every hour
- Collect market history, item metadata and liquidity tiers daily (01:00)
- Collect ESI reference prices and score elapsed predictions daily (02:15/02:30)
- Generate and store predictions daily (03:00)
- Run drift detection daily (04:00), retraining drifted items when
  `RETRAIN_ON_DRIFT=True`
- Retrain every item weekly (Sundays at 02:00), promoting a new checkpoint only
  when it validates better than the incumbent

To train your first model, see the "Training Your First Model" section below.

---

## Option 2: Local Development Setup

### Prerequisites

#### 1. Install Python 3.11

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev

# Verify
python3.11 --version
```

#### 2. Install PostgreSQL 15

```bash
# Add PostgreSQL repository
sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -

# Install
sudo apt update
sudo apt install postgresql-15 postgresql-contrib-15

# Start service
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

#### 3. Install TimescaleDB

```bash
# Add TimescaleDB repository
sudo sh -c "echo 'deb https://packagecloud.io/timescale/timescaledb/ubuntu/ $(lsb_release -c -s) main' > /etc/apt/sources.list.d/timescaledb.list"
wget --quiet -O - https://packagecloud.io/timescale/timescaledb/gpgkey | sudo apt-key add -

# Install
sudo apt update
sudo apt install timescaledb-2-postgresql-15

# Configure
sudo timescaledb-tune --quiet --yes

# Restart PostgreSQL
sudo systemctl restart postgresql
```

#### 4. Install Redis

```bash
# Install Redis
sudo apt install redis-server

# Start service
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Test
redis-cli ping  # Should return PONG
```

### Step 1: Clone Repository

```bash
cd ~/
git clone <your-repo> eve-discovery
cd eve-discovery
```

### Step 2: Create Virtual Environment

```bash
# Create venv
python3.11 -m venv venv

# Activate
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Configure Database

```bash
# Create database user
sudo -u postgres psql << EOF
CREATE USER eve_user WITH PASSWORD 'your_secure_password';
CREATE DATABASE eve_markets OWNER eve_user;
\c eve_markets
CREATE EXTENSION IF NOT EXISTS timescaledb;
GRANT ALL PRIVILEGES ON DATABASE eve_markets TO eve_user;
EOF

# Test connection
psql -U eve_user -d eve_markets -c "SELECT version();"
```

### Step 4: Configure Application

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

Update these values:
```bash
DATABASE_URL=postgresql+asyncpg://eve_user:your_secure_password@localhost:5432/eve_markets
REDIS_URL=redis://localhost:6379/0
LOG_LEVEL=INFO
```

### Step 5: Initialize Database

```bash
# Run initialization script
python scripts/init_database.py
```

### Step 6: Collect Data

```bash
# Collect initial data
python scripts/collect_data.py
```

### Step 7: Run Services

**Terminal 1 - Scheduler (Data Collection):**
```bash
python scripts/scheduler_daemon.py
```

**Terminal 2 - API Server:**
```bash
uvicorn src.api.main:app --reload --port 8000
```

**Terminal 3 - MLflow (Optional):**
```bash
mlflow ui --port 5000
```

### Step 8: Verify Setup

```bash
# Check API
curl http://localhost:8000/health

# Check data collection
psql -U eve_user -d eve_markets -c "SELECT COUNT(*) FROM market_history;"
```

---

## Training Your First Model

`scripts/train_model.py` already exists -- do not recreate it. (An earlier
version of this guide pasted a full script body here, which overwrote the real
one if you followed along.)

### 1. Train

```bash
# One item
python scripts/train_model.py --type-id 44992      # PLEX

# Every tracked item
python scripts/train_model.py --all

# See all options
python scripts/train_model.py --help
```

Each item is saved separately as `data/models/model_{type_id}_{region_id}.pth`
with a JSON sidecar holding the fitted scaler, the feature column order,
architecture and validation metrics. Training one item never overwrites
another's checkpoint.

The output reports validation MAE/RMSE/MAPE in ISK plus three baselines (naive,
7-day moving average, linear trend) scored on the same window -- if the LSTM
does not beat naive, that is worth knowing.

### 2. Inspect and predict

```bash
python scripts/check_model_metrics.py                  # what is on disk
python scripts/predict_item.py 44992 --days 7          # forecast
python scripts/predict_all_items.py                    # all items
python scripts/evaluate_predictions.py                 # score elapsed forecasts
```

### 3. View Results in MLflow

Open http://localhost:5000 and view experiment metrics.

---

## Common Issues & Solutions

### Issue: "Connection refused" to PostgreSQL

**Solution:**
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Start if stopped
sudo systemctl start postgresql

# Check it's listening
sudo netstat -plnt | grep 5432
```

### Issue: "Extension timescaledb does not exist"

**Solution:**
```bash
# Enable extension manually
sudo -u postgres psql eve_markets -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"
```

### Issue: ESI API 420 Error (Rate Limited)

**Solution:**
Edit `.env`:
```bash
ESI_RATE_LIMIT=50  # Reduce from 100
ESI_RATE_PERIOD=60
```

### Issue: Docker containers won't start

**Solution:**
```bash
# Check logs
docker compose logs

# Restart clean
docker compose down
docker compose up -d

# Rebuild if needed
docker compose up --build -d
```

### Issue: Out of memory during training

**Solution:**
- Reduce batch size in trainer
- Reduce lookback window
- Use CPU instead of GPU
- Add swap space

---

## Monitoring & Maintenance

### Check Data Collection Status

```bash
# Via Docker
docker compose exec postgres psql -U eve_user eve_markets -c "
  SELECT
    type_id,
    COUNT(*) as records,
    MAX(time) as latest
  FROM market_history
  GROUP BY type_id;
"

# Local
psql -U eve_user -d eve_markets -c "..."
```

### View Logs

```bash
# Docker
docker compose logs -f api
docker compose logs -f collector

# Local (if running in terminals)
# Logs appear in respective terminal windows
```

### Backup Database

```bash
# Docker
docker compose exec postgres pg_dump -U eve_user eve_markets > backup.sql

# Local
pg_dump -U eve_user eve_markets > backup.sql
```

### Restore Database

```bash
# Docker
docker compose exec -T postgres psql -U eve_user eve_markets < backup.sql

# Local
psql -U eve_user eve_markets < backup.sql
```

---

## Next Steps

1. **Explore API**: Visit http://localhost:8000/docs
2. **Train models**: Create training scripts for different items
3. **Monitor**: Check MLflow UI for experiment results
4. **Customize**: Modify features, model architecture, etc.
5. **Deploy**: Use docker-compose for production deployment

## Getting Help

- Documentation: See CLAUDE.md, PRICE_PREDICTION_ENGINE.md, TECHNOLOGY_STACK.md
- Issues: Check logs first, then create GitHub issue
- Community: EVE Online Third-Party Developers forum

---

**Happy trading! o7**
