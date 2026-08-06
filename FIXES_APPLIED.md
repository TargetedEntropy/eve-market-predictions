# Fixes Applied (2025 installation round)

> **Historical.** This file records an early round of installation fixes. It is
> not a statement about the current health of the system -- a later audit
> (`fixes.md`) found ~70 further defects, which are addressed in
> **AUDIT_REMEDIATION.md**. Read that first.
>
> One claim below was also wrong when written: the `RETRAINING_DAY` fix updated
> `src/config.py` and `.env.example` but not the on-disk `.env`, which still
> said `sunday` and still crash-looped the scheduler. That is fixed now, and
> the config layer normalizes full day names so the class of bug cannot recur.

---

## 🔧 Issues Fixed

### 1. ❌ pandas-ta Package Not Found
**Error:** `ERROR: No matching distribution found for pandas-ta`

**Fix Applied:**
- Replaced `pandas-ta` with `ta==0.11.0` library
- Updated `src/features/technical_indicators.py` with smart fallback
- Works with either library automatically

**Files Changed:**
- `requirements.txt`
- `src/features/technical_indicators.py`

---

### 2. ❌ Missing Item Import
**Error:** `ImportError: cannot import name 'Item' from 'src.database'`

**Fix Applied:**
- Added `Item` to exports in `src/database/__init__.py`

**Files Changed:**
- `src/database/__init__.py`

---

### 3. ❌ Pydantic Protected Namespace Warning
**Warning:** `Field "model_path" has conflict with protected namespace "model_"`

**Fix Applied:**
- Set `protected_namespaces=()` in Settings config
- Allows "model_" prefix in field names

**Files Changed:**
- `src/config.py`

---

### 4. ❌ Invalid Weekday Name
**Error:** `ValueError: Invalid weekday name "sunday"`

**Fix Applied (incomplete at the time -- see the note at the top):**
- Changed the *default* from `"sunday"` to `"sun"` (APScheduler format)
- Updated documentation with valid values
- **Missed:** the on-disk `.env` still contained `RETRAINING_DAY=sunday`, so
  the scheduler kept crashing locally.

**Completed later:** `src/config.py` now normalizes `sunday`/`Sunday`/`tues`/...
to APScheduler's three-letter form and rejects anything that is not a weekday,
`.env` was corrected, and `tests/test_config.py` covers it.

**Files Changed:**
- `src/config.py`
- `.env.example`, `.env`
- `scripts/scheduler_daemon.py`

---

## ✅ System Status

### All Containers Running:

```
NAME           STATUS          PORTS
eve-api        Up              0.0.0.0:8000->8000/tcp
eve-collector  Up              (scheduler running)
eve-mlflow     Up              0.0.0.0:5000->5000/tcp
eve-postgres   Up (healthy)    0.0.0.0:5432->5432/tcp
eve-redis      Up (healthy)    0.0.0.0:6379->6379/tcp
```

### Scheduler Jobs Active:

✅ Market Data Collection - Every 5 minutes
✅ Feature Computation - Every 1 hour
✅ Weekly Model Retraining - Sundays at 2 AM

---

## 🚀 Ready to Use!

### Access Your Services:

1. **API Documentation**: http://localhost:8000/docs
2. **MLflow Tracking**: http://localhost:5000
3. **Health Check**: http://localhost:8000/health

### Next Steps:

```bash
# 1. Initialize database
docker compose exec api python scripts/init_database.py

# 2. Collect initial data (5-10 minutes)
docker compose exec collector python scripts/collect_data.py

# 3. Verify data collection
curl http://localhost:8000/items

# 4. Check database
docker compose exec postgres psql -U eve_user eve_markets -c "SELECT COUNT(*) FROM market_history;"
```

---

## 📋 Configuration Values

Valid day names for `RETRAINING_DAY`:
- **Short names**: mon, tue, wed, thu, fri, sat, sun
- **Numbers**: 0-6 (Monday=0, Sunday=6)

Example `.env`:
```bash
RETRAINING_DAY=sun  # ✅ Correct
RETRAINING_DAY=sunday  # ❌ Won't work
RETRAINING_DAY=0  # ✅ Also works (Monday)
```

---

## 🧪 Verify Everything Works

### Test 1: Check Container Logs
```bash
docker compose logs collector --tail 20
# Should show: "Scheduler started" with 3 jobs added
```

### Test 2: Health Check
```bash
curl http://localhost:8000/health
# Should return: {"status":"healthy" or "degraded",...}
```

### Test 3: Database Connection
```bash
docker compose exec api python -c "
import asyncio
from src.database import init_db
asyncio.run(init_db())
print('✅ Database working!')
"
```

### Test 4: Technical Indicators
```bash
docker compose exec api python -c "
from src.features.technical_indicators import TechnicalIndicators
print('✅ Technical indicators working!')
"
```

---

## 💡 Tips

### View Real-Time Logs
```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f collector
docker compose logs -f api
```

### Restart Services
```bash
# Restart everything
docker compose restart

# Restart specific service
docker compose restart collector
```

### Stop Everything
```bash
docker compose down
```

### Complete Reset
```bash
# Stop and remove everything (including data!)
docker compose down -v

# Start fresh
docker compose up -d
```

---

## 📖 Documentation

All documentation is in the repository:

- **INSTALL_NOW.md** - Installation guide
- **README.md** - Complete usage guide
- **QUICKSTART.md** - Fast setup
- **SETUP_GUIDE.md** - Detailed setup
- **PROJECT_SUMMARY.md** - Project overview

---

## 🎉 Summary

**Status: ✅ FULLY OPERATIONAL**

All issues have been resolved and the system is running successfully:

✅ Dependencies installed correctly
✅ All containers running
✅ Scheduler active
✅ API responding
✅ Database connected
✅ Services healthy

**You're ready to start collecting data and training models!**

---

## 🆘 Need Help?

If you encounter any issues:

1. **Check logs**: `docker compose logs [service]`
2. **Review docs**: See documentation files
3. **Restart**: `docker compose restart`
4. **Full reset**: `docker compose down && docker compose up -d`

---

**Happy trading, Capsuleer! o7** 🚀📈
