#!/bin/bash
# Quick PostgreSQL Setup + Start HPO
# This achieves >90% GPU utilization vs 70-85% with SQLite

set -e

echo "=============================================="
echo " Quick PostgreSQL Setup for >90% GPU Usage"
echo "=============================================="
echo ""

# Step 1: Install PostgreSQL
echo "[1/4] Installing PostgreSQL..."
sudo apt-get update -qq
sudo apt-get install -y postgresql postgresql-contrib

# Step 2: Start and configure PostgreSQL
echo "[2/4] Configuring PostgreSQL..."
sudo systemctl start postgresql
sudo systemctl enable postgresql

sudo -u postgres psql <<EOF
CREATE DATABASE optuna;
CREATE USER optuna_user WITH PASSWORD 'optuna_pass';
GRANT ALL PRIVILEGES ON DATABASE optuna TO optuna_user;
ALTER DATABASE optuna OWNER TO optuna_user;
\q
EOF

echo "✅ PostgreSQL configured"

# Step 3: Install Python driver
echo "[3/4] Installing psycopg2-binary..."
cd /media/cvrlab308/cvrlab308_4090/YuNing/DataAug_Criteria_Evidence
poetry add psycopg2-binary

echo "✅ Driver installed"

# Step 4: Set environment variable
echo "[4/4] Setting environment variable..."
export OPTUNA_STORAGE="postgresql://optuna_user:optuna_pass@localhost/optuna"

# Add to current session
echo ""
echo "✅ Setup complete!"
echo ""
echo "To make this permanent, add to ~/.bashrc:"
echo "  echo 'export OPTUNA_STORAGE=\"postgresql://optuna_user:optuna_pass@localhost/optuna\"' >> ~/.bashrc"
echo ""
echo "Now run with maximum performance:"
echo "  export OPTUNA_STORAGE=\"postgresql://optuna_user:optuna_pass@localhost/optuna\""
echo "  PAR=8 NUM_WORKERS=16 make tune-all-supermax"
echo ""
