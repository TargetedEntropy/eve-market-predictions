# Quick Start Guide - Fixed Installation

## ✅ Working Installation (No Issues)

### Option 1: Docker (RECOMMENDED - Zero Issues!)

```bash
# 1. Setup
cd eve-discovery
cp .env.example .env
nano .env  # Change DB_PASSWORD

# 2. Start everything
docker compose up -d

# 3. Initialize
docker compose exec api python scripts/init_database.py
docker compose exec collector python scripts/collect_data.py

# 4. Done! Access at:
# http://localhost:8000/docs
```

### Option 2: Local Installation (Fixed)

```bash
# 1. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 2. Install dependencies (FIXED VERSION)
pip install --upgrade pip

# Install core packages
pip install -r requirements.txt

# 3. Setup database (requires PostgreSQL + TimescaleDB)
python scripts/init_database.py

# 4. Collect data
python scripts/collect_data.py

# 5. Run API
uvicorn src.api.main:app --reload
```

## 🔧 What Was Fixed

### Issue: pandas-ta not found
**Solution:** Replaced with `ta` library (more reliable)

### Changes Made:
1. **requirements.txt** - Now uses `ta==0.11.0` instead of pandas-ta
2. **src/features/technical_indicators_alt.py** - New implementation with fallback support
3. Works with either library automatically!

## 📦 Installation Options

### Minimal Working Installation

```bash
# Just the essentials
pip install fastapi uvicorn sqlalchemy asyncpg redis httpx
pip install torch pandas numpy scikit-learn
pip install ta  # Technical indicators (working package!)
pip install mlflow apscheduler
```

### Full Installation

```bash
# Everything
pip install -r requirements.txt
```

### From GitHub (pandas-ta if you prefer)

```bash
# Use pandas-ta from GitHub instead
pip install git+https://github.com/twopirllc/pandas-ta.git
```

## 🎯 What Works Now

✅ All packages install without errors
✅ Technical indicators work with `ta` library
✅ Backward compatible with pandas-ta if installed
✅ Docker works perfectly (no changes needed)

## 🚀 Test Your Installation

```bash
# Test imports
python -c "
import fastapi
import torch
import pandas
import ta
import sqlalchemy
print('✅ All packages working!')
"

# Test technical indicators
python -c "
from src.features.technical_indicators_alt import TechnicalIndicators
print('✅ Technical indicators working!')
"
```

## 📖 Updated Files

1. **requirements.txt** - Uses `ta` library now
2. **src/features/technical_indicators_alt.py** - New implementation
3. **Original files still work** - Backward compatible!

## 🐳 Docker (Still the Easiest)

Docker installation is **unchanged** and works perfectly:

```bash
docker compose up -d
```

That's it! No dependency issues with Docker.

## 💡 Tips

1. **Use Docker** if you want zero hassle
2. **Local installation** now works with fixed requirements
3. The `ta` library provides same indicators as pandas-ta
4. Code automatically detects which library is available

## 🆘 Still Having Issues?

```bash
# Nuclear option - fresh start
rm -rf venv
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Or just use Docker!
docker compose up -d
```

## ✨ You're Ready!

Start the system:
```bash
# Docker
docker compose up -d

# Or Local
python scripts/scheduler_daemon.py  # Terminal 1
uvicorn src.api.main:app --reload   # Terminal 2
```

Access at: **http://localhost:8000/docs**

Happy trading! 🚀
