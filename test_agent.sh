#!/bin/bash
# Antigravity v5.0 - Load historical data and run analysis
source ~/.local/share/dfx/env
cd /mnt/c/Users/ADVAIT/.gemini/antigravity/scratch/antigravity

echo "========================================"
echo "ANTIGRAVITY v5.0 - LOADING HISTORICAL DATA"
echo "========================================"

echo ""
echo "1. Version:"
dfx canister call antigravity_bot get_version

echo ""
echo "2. Loading 1-year data for BTC..."
dfx canister call antigravity_bot 'fetch_ohlcv_data("bitcoin")'

echo ""
echo "3. Loading 1-year data for ETH..."
dfx canister call antigravity_bot 'fetch_ohlcv_data("ethereum")'

echo ""
echo "4. Loading 1-year data for SOL..."
dfx canister call antigravity_bot 'fetch_ohlcv_data("solana")'

echo ""
echo "5. Loading 1-year data for ICP..."
dfx canister call antigravity_bot 'fetch_ohlcv_data("internet-computer")'

echo ""
echo "6. Data Status:"
dfx canister call antigravity_bot get_data_status

echo ""
echo "7. Running Analysis..."
dfx canister call antigravity_bot run_analysis

echo ""
echo "8. Cointegrated Pairs:"
dfx canister call antigravity_bot get_cointegrated_pairs

echo ""
echo "9. Trading Signals:"
dfx canister call antigravity_bot get_trading_signals

echo ""
echo "10. State:"
dfx canister call antigravity_bot get_state

echo ""
echo "========================================"
echo "COMPLETE"
echo "========================================"
