# Installation Fixes & Common Issues

> **Superseded.** This file documented a 2025 troubleshooting round for the
> `pandas-ta` dependency. The project now uses the `ta` library exclusively --
> the `pandas_ta` code path has been deleted, so none of the workarounds below
> are needed. For current instructions see **QUICKSTART.md**, and for the state
> of the codebase see **AUDIT_REMEDIATION.md**. Kept for history only.

## pandas-ta Version Issue

### Problem
```
ERROR: No matching distribution found for pandas-ta==0.3.14b0
```

### Solutions

#### Option 1: Install without version constraint (Recommended)
```bash
pip install pandas-ta
```

#### Option 2: Install specific working version
```bash
# Try these in order until one works:
pip install "pandas-ta>=0.3.14b"
pip install pandas-ta==0.3.14b
pip install git+https://github.com/twopirllc/pandas-ta.git
```

#### Option 3: Updated requirements.txt
The requirements.txt has been updated to:
```
pandas-ta  # Latest version
```

### Why This Happens
pandas-ta uses beta versioning (e.g., 0.3.14b) which can have inconsistent naming on PyPI. Using the latest version without strict pinning is more reliable.

---

## PyTorch Version Issues

### Problem: CUDA version mismatch or CPU-only installation

### Solution: Install specific PyTorch version

**For CPU only:**
```bash
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
```

**For CUDA 11.8:**
```bash
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

### Check your installation:
```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Alternative Installation Method

If you encounter multiple dependency issues, try this staged installation:

```bash
# 1. Core dependencies first
pip install python-dotenv pydantic pydantic-settings

# 2. Web framework
pip install fastapi uvicorn[standard] gunicorn

# 3. Database (without version constraints)
pip install sqlalchemy asyncpg

# 4. HTTP & caching
pip install httpx redis aiometer

# 5. Data processing (without strict versions)
pip install pandas numpy scipy scikit-learn

# 6. ML frameworks
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install mlflow evidently

# 7. Technical indicators
pip install pandas-ta

# 8. Remaining packages
pip install vectorbt apscheduler plotly dash matplotlib seaborn
pip install pytest pytest-asyncio pytest-cov black ruff
```

---

## Docker Installation (No Issues)

If local installation is problematic, use Docker instead:

```bash
# No dependency issues with Docker!
docker compose up -d
```

Docker automatically handles all dependencies correctly.

---

## Minimal Requirements (Core Functionality Only)

If you want just the core functionality without visualization/testing:

```bash
# Create minimal requirements
cat > requirements-minimal.txt << EOF
python-dotenv==1.0.1
fastapi==0.115.0
uvicorn[standard]==0.32.0
pydantic==2.9.0
pydantic-settings==2.5.0
httpx[http2]==0.27.0
sqlalchemy[asyncio]==2.0.25
asyncpg==0.29.0
redis[hiredis]==5.0.1
aiometer==0.5.0
torch
numpy
pandas
pandas-ta
scikit-learn
mlflow
apscheduler
EOF

pip install -r requirements-minimal.txt
```

---

## Verification Commands

After installation, verify everything works:

```bash
# Test Python packages
python -c "import fastapi, torch, pandas, pandas_ta, mlflow; print('✓ All core packages installed')"

# Test database connection
python -c "from sqlalchemy.ext.asyncio import create_async_engine; print('✓ SQLAlchemy async working')"

# Test Redis
python -c "import redis; r = redis.Redis(); r.ping(); print('✓ Redis working')"
```

---

## Platform-Specific Issues

### macOS
```bash
# May need to install Xcode Command Line Tools
xcode-select --install

# If OpenMP issues with PyTorch
brew install libomp
```

### Windows
```bash
# Use Windows Subsystem for Linux (WSL2) for best experience
# Or use Docker Desktop
```

### Linux (Ubuntu/Debian)
```bash
# Install build dependencies
sudo apt update
sudo apt install -y build-essential python3-dev libpq-dev
```

---

## Still Having Issues?

1. **Try Docker** - It's the most reliable method
2. **Use a fresh virtual environment**:
   ```bash
   rm -rf venv
   python3.11 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   ```

3. **Check Python version**:
   ```bash
   python --version  # Should be 3.11+
   ```

4. **Clear pip cache**:
   ```bash
   pip cache purge
   ```

5. **Use Docker** (seriously, it just works!)

---

## Contact & Support

If issues persist:
- Check the error message carefully
- Search for the specific package error online
- Try Docker deployment instead
- Open an issue on GitHub with the full error log

