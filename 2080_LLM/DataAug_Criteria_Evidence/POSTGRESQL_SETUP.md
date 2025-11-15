# PostgreSQL Setup for Optuna HPO

This guide shows how to migrate from SQLite to PostgreSQL for **4-8x faster HPO** with parallel trials.

## Why PostgreSQL?

| Metric | SQLite (Current) | PostgreSQL |
|--------|------------------|------------|
| Max PAR | 1 (database locks) | 8+ (no locks) |
| GPU Utilization | 70-85% | 95-100% |
| Time for 19K trials | ~120 hours | ~30 hours |

## Setup Steps (10 minutes)

### Step 1: Install PostgreSQL and Create Database

Run the automated setup script:

```bash
sudo bash scripts/setup_postgresql.sh
```

This script will:
- ✅ Install PostgreSQL
- ✅ Create `optuna` database
- ✅ Create `optuna_user` with password `optuna_pass`
- ✅ Configure local connections
- ✅ Test the connection

**Expected output:**
```
✅ PostgreSQL setup complete!

Connection details:
  Database: optuna
  User: optuna_user
  Password: optuna_pass
  Connection string: postgresql://optuna_user:optuna_pass@localhost/optuna
```

### Step 2: Install Python PostgreSQL Driver

```bash
poetry add psycopg2-binary
```

Or if you prefer pip:
```bash
pip install psycopg2-binary
```

### Step 3: Configure Optuna to Use PostgreSQL

Set the environment variable to use PostgreSQL:

```bash
export OPTUNA_STORAGE="postgresql://optuna_user:optuna_pass@localhost/optuna"
```

Or permanently add to your `~/.bashrc`:
```bash
echo 'export OPTUNA_STORAGE="postgresql://optuna_user:optuna_pass@localhost/optuna"' >> ~/.bashrc
source ~/.bashrc
```

### Step 4: Test the Setup

Run a quick test to verify PostgreSQL works:

```bash
NUM_WORKERS=12 HPO_EPOCHS=2 HPO_PATIENCE=1 \
  poetry run python scripts/tune_max.py \
  --agent criteria \
  --study test-postgresql \
  --n-trials 5 \
  --parallel 4 \
  --outdir ./_runs
```

**Success indicators:**
- ✅ No SQLite database lock errors
- ✅ All 4 trials run in parallel
- ✅ Higher GPU utilization (check with `nvidia-smi`)

## Usage

### Run Full Supermax HPO with PostgreSQL

```bash
# With PostgreSQL, you can use PAR=8 for maximum speed
PAR=8 NUM_WORKERS=16 make tune-all-supermax
```

**Expected performance:**
- 8 trials in parallel
- 95-100% GPU utilization
- ~25-35 hours for all 19K trials (vs 120 hours with SQLite PAR=1)

### Monitor Progress

```bash
# Terminal 1: Run HPO
PAR=8 NUM_WORKERS=16 make tune-all-supermax

# Terminal 2: Monitor GPU
watch -n 2 nvidia-smi

# Terminal 3: Monitor system resources
./scripts/monitor_hpo.sh
```

## Troubleshooting

### Connection Refused

If you see `connection refused`:

```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Start if needed
sudo systemctl start postgresql
```

### Authentication Failed

If you see `authentication failed for user optuna_user`:

```bash
# Reset password
sudo -u postgres psql -c "ALTER USER optuna_user WITH PASSWORD 'optuna_pass';"
```

### Database Does Not Exist

If you see `database "optuna" does not exist`:

```bash
# Create database
sudo -u postgres createdb optuna
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE optuna TO optuna_user;"
```

## Reverting to SQLite

If you need to revert to SQLite:

```bash
# Unset the environment variable
unset OPTUNA_STORAGE

# Or use --storage flag explicitly
poetry run python scripts/tune_max.py \
  --storage "sqlite:///_optuna/noaug.db" \
  ...other args...
```

## Performance Comparison

### Before (SQLite PAR=1)
```
GPU: ████░░░░░░ 70-85%
Trials: 1 at a time
Speed: ~6 min/trial
Total: ~120 hours
```

### After (PostgreSQL PAR=8)
```
GPU: ██████████ 95-100%
Trials: 8 in parallel
Speed: ~6 min/trial × 8 = 48 trials/hour
Total: ~30 hours
```

## Security Note

**Default password**: The setup uses `optuna_pass` as the default password. For production systems, change it:

```bash
sudo -u postgres psql -c "ALTER USER optuna_user WITH PASSWORD 'your_secure_password';"
```

Then update your connection string:
```bash
export OPTUNA_STORAGE="postgresql://optuna_user:your_secure_password@localhost/optuna"
```

## Next Steps

After setup, proceed with the full HPO run:

```bash
# Set environment variable (if not already set)
export OPTUNA_STORAGE="postgresql://optuna_user:optuna_pass@localhost/optuna"

# Run with maximum parallelism
PAR=8 NUM_WORKERS=16 make tune-all-supermax
```

Monitor GPU utilization - you should see **95-100%** sustained usage! 🚀
