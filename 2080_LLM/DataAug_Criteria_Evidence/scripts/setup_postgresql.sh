#!/bin/bash
# Setup PostgreSQL for Optuna HPO
# Run this script with: sudo bash scripts/setup_postgresql.sh

set -e

echo "===== PostgreSQL Setup for Optuna ====="
echo ""

# 1. Install PostgreSQL
echo "[1/5] Installing PostgreSQL..."
apt-get update -qq
apt-get install -y postgresql postgresql-contrib

# 2. Start PostgreSQL service
echo "[2/5] Starting PostgreSQL service..."
systemctl start postgresql
systemctl enable postgresql

# 3. Create database and user
echo "[3/5] Creating Optuna database and user..."
sudo -u postgres psql <<EOF
CREATE DATABASE optuna;
CREATE USER optuna_user WITH PASSWORD 'optuna_pass';
GRANT ALL PRIVILEGES ON DATABASE optuna TO optuna_user;
ALTER DATABASE optuna OWNER TO optuna_user;
\q
EOF

# 4. Configure PostgreSQL for local connections
echo "[4/5] Configuring PostgreSQL..."
PG_VERSION=$(sudo -u postgres psql -V | grep -oP '\d+' | head -1)
PG_HBA="/etc/postgresql/$PG_VERSION/main/pg_hba.conf"

# Backup original config
cp $PG_HBA ${PG_HBA}.backup

# Add local connection for optuna_user (if not already present)
if ! grep -q "local.*optuna.*optuna_user" $PG_HBA; then
    echo "local   optuna          optuna_user                             md5" >> $PG_HBA
fi

# Restart PostgreSQL to apply changes
systemctl restart postgresql

# 5. Test connection
echo "[5/5] Testing connection..."
if sudo -u postgres psql -U optuna_user -d optuna -c "SELECT 1;" > /dev/null 2>&1; then
    echo ""
    echo "✅ PostgreSQL setup complete!"
    echo ""
    echo "Connection details:"
    echo "  Database: optuna"
    echo "  User: optuna_user"
    echo "  Password: optuna_pass"
    echo "  Connection string: postgresql://optuna_user:optuna_pass@localhost/optuna"
    echo ""
else
    echo ""
    echo "⚠️  PostgreSQL installed but connection test failed."
    echo "   You may need to manually configure pg_hba.conf"
    echo ""
fi

echo "Next step: Install Python driver with 'poetry add psycopg2-binary'"
