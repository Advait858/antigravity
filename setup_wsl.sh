#!/bin/bash
# ============================================================
# Antigravity ICP Setup Script for WSL
# Run this script inside WSL (Linux terminal)
# ============================================================

set -e  # Exit on error

PROJECT_DIR="/mnt/c/Users/ADVAIT/.gemini/antigravity/scratch/antigravity"

echo "========================================"
echo "  ANTIGRAVITY ICP SETUP"
echo "========================================"

# Step 1: Check if dfx is installed
if ! command -v dfx &> /dev/null; then
    echo ""
    echo "[1/5] Installing dfx (DFINITY SDK)..."
    sh -ci "$(curl -fsSL https://internetcomputer.org/install.sh)"
    
    # Source bashrc to get dfx in path
    source ~/.bashrc 2>/dev/null || source ~/.profile 2>/dev/null || true
    
    # Add to path if still not found
    export PATH="$HOME/.local/share/dfx/bin:$PATH"
    
    echo "✅ dfx installed!"
else
    echo "[1/5] dfx already installed: $(dfx --version)"
fi

# Step 2: Navigate to project
echo ""
echo "[2/5] Navigating to project..."
cd "$PROJECT_DIR"
echo "✅ In: $(pwd)"

# Step 3: Stop any existing replica
echo ""
echo "[3/5] Stopping any existing replica..."
dfx stop 2>/dev/null || true
sleep 2

# Step 4: Start local replica
echo ""
echo "[4/5] Starting local replica..."
dfx start --clean --background
sleep 5

# Step 5: Deploy canister
echo ""
echo "[5/5] Deploying antigravity_bot canister..."
dfx deploy antigravity_bot

echo ""
echo "========================================"
echo "  DEPLOYMENT COMPLETE!"
echo "========================================"
echo ""
echo "Canister deployed. You can now:"
echo "  1. Open http://localhost:3000 in browser (React app)"
echo "  2. Query canister: dfx canister call antigravity get_health"
echo ""
echo "To trigger the agent manually:"
echo "  dfx canister call antigravity trigger_tick"
echo ""
