# Antigravity 🚀

**ICP Algo-Trading Agent using ADF Cointegration Model**

Built for the Zo House Hackathon - "AI Agents for Trading" Bounty

## Overview

Antigravity is a Web3 trading agent deployed on the Internet Computer (ICP) using the Kybra CDK. It implements statistical arbitrage using the Augmented Dickey-Fuller (ADF) cointegration model to identify trading opportunities between crypto pairs (e.g., BTC/ICP).

## Tech Stack

- **Blockchain:** Internet Computer (ICP)
- **Language:** Python 3.10+ (via Kybra CDK)
- **Libraries:** statsmodels, numpy, pandas, kybra

## Project Structure

```
antigravity/
├── dfx.json              # ICP canister configuration
├── kybra.json            # Kybra CDK config
├── requirements.txt      # Python dependencies
└── src/
    ├── main.py           # Canister entry point
    ├── strategy/
    │   ├── __init__.py
    │   └── cointegration.py  # ADF model & PairsTrader
    └── data/
        ├── __init__.py
        └── loader.py     # Mock data for testing
```

## Strategy

The bot uses **Pairs Trading with Cointegration Analysis**:

1. **Spread Calculation**: Uses OLS regression to find the hedge ratio between two assets
2. **Stationarity Test**: Applies the ADF test to determine if pairs are cointegrated
3. **Signal Generation**: Trades mean-reversion when spread deviates from equilibrium

## Development

### Prerequisites

- [dfx](https://internetcomputer.org/docs/current/developer-docs/setup/install) - ICP SDK
- Python 3.10+
- Kybra CDK

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start local ICP replica
dfx start --clean --background

# Deploy the canister
dfx deploy
```

### Canister Methods

- `get_health()` - Health check endpoint
- `get_version()` - Returns bot version
- `execute_strategy()` - Runs the cointegration trading strategy
- `get_strategy_info()` - Returns strategy description

## License

MIT License
