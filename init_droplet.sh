#!/bin/bash

# init_droplet.sh
# Automates the setup and starting of the FastAPI Backend and PostgreSQL Database on the AMD MI300X Droplet.

set -e

echo "🚀 Starting Droplet Initialization (Backend + Database)..."

# 1. System Updates and Prerequisites
echo "📦 Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv curl tmux postgresql postgresql-contrib

# Install uv for faster python package installation
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# 2. Setup PostgreSQL Database
echo "🗄️ Setting up PostgreSQL..."
sudo -u postgres psql -c "CREATE DATABASE medical_etl;" || true
sudo -u postgres psql -c "CREATE USER etl_user WITH PASSWORD 'etl_password';" || true
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE medical_etl TO etl_user;" || true

# Allow remote connections for local agents to insert data
sudo sed -i "s/#listen_addresses = 'localhost'/listen_addresses = '*'/g" /etc/postgresql/*/main/postgresql.conf
sudo bash -c "echo 'host    medical_etl     etl_user        0.0.0.0/0               md5' >> /etc/postgresql/*/main/pg_hba.conf"
sudo systemctl restart postgresql

# 3. Setup Backend Virtual Environment
echo "🐍 Setting up Backend environment..."
uv venv backend/.venv
source backend/.venv/bin/activate
echo "⚙️ Installing Backend dependencies (with ROCm PyTorch)..."
uv pip install -r backend/requirements.txt --extra-index-url https://download.pytorch.org/whl/rocm5.7
deactivate

# 4. Environment Configuration
if [ ! -f .env ]; then
    echo "📝 Creating default .env file..."
    cat <<EOF > .env
# Backend Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# Database Configuration
DATABASE_URL=postgresql://etl_user:etl_password@localhost/medical_etl

# Model Cache
HF_HOME=./model_cache
EOF
else
    echo "✅ .env file already exists."
fi

# 5. Start Services using Tmux
echo "▶️ Starting backend in tmux session..."

# Start Backend
tmux new-session -d -s backend 'source backend/.venv/bin/activate && cd backend && python main.py --host 0.0.0.0 --port 8000'
echo "✓ Backend started in tmux session 'backend'"

echo "🎉 Initialization complete!"
echo "👉 View backend logs: tmux attach-session -t backend"
echo "👉 Your PostgreSQL DB is running on port 5432. Set up your local .env DATABASE_URL to point to this Droplet's IP."
