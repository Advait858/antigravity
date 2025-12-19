#!/bin/bash
# Test script for Antigravity v3.0 Multi-Asset Agent
source ~/.local/share/dfx/env
cd /mnt/c/Users/ADVAIT/.gemini/antigravity/scratch/antigravity

echo "========================================"
echo "ANTIGRAVITY v3.0 MULTI-ASSET TEST"
echo "========================================"

echo ""
echo "1. Health Check:"
dfx canister call antigravity_bot get_health

echo ""
echo "2. Version:"
dfx canister call antigravity_bot get_version

echo ""
echo "3. Strategy Info:"
dfx canister call antigravity_bot get_strategy_info

echo ""
echo "4. Config (tracked assets):"
dfx canister call antigravity_bot get_config

echo ""
echo "5. Who Am I (Principal ID):"
dfx canister call antigravity_bot whoami

echo ""
echo "6. Register Wallet:"
dfx canister call antigravity_bot register_wallet

echo ""
echo "7. Fetch All Top 10 Prices:"
dfx canister call antigravity_bot fetch_all_prices

echo ""
echo "8. Get Top 10 Prices:"
dfx canister call antigravity_bot get_top_10_prices

echo ""
echo "9. Fetch and Analyze (with recommendations):"
dfx canister call antigravity_bot fetch_and_analyze

echo ""
echo "10. Get Recommendations:"
dfx canister call antigravity_bot get_recommendations

echo ""
echo "11. Get Portfolio Analysis:"
dfx canister call antigravity_bot get_portfolio_analysis

echo ""
echo "12. Get BTC History:"
dfx canister call antigravity_bot 'get_asset_history("BTC")'

echo ""
echo "13. Get State:"
dfx canister call antigravity_bot get_state

echo ""
echo "========================================"
echo "TEST COMPLETE"
echo "========================================"
