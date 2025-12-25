# Antigravity AI Trading Agent

**Hybrid AI Trading Agent for Internet Computer (ICP)**

A cointegration-based trading system that uses AI (Llama 3.3 70B) for decision-making and an ICP Motoko canister for secure on-chain execution.

## Architecture

```
+-------------------+     +------------------+     +------------------+
|   Binance API     | --> |  Python Agent    | --> |  ICP Canister    |
|   (Market Data)   |     |  (AI + Analysis) |     |  (Execution)     |
+-------------------+     +------------------+     +------------------+
                                   |
                          +--------v--------+
                          |   Groq API      |
                          |  (Llama 3.3)    |
                          +-----------------+
```

## Features

- **100,000+ Data Points**: Fetches 13 months of hourly OHLCV from Binance
- **45 Pair Analysis**: Cointegration testing on all asset pairs
- **Advanced Statistics**: Half-life, Hurst exponent, Z-score, Bollinger bands
- **AI Decision Making**: Groq (Llama 3.3 70B) evaluates opportunities
- **On-Chain Execution**: Motoko canister handles paper trading
- **Backtesting**: 12-month train / 1-month test with AI-driven trades

## Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/your-repo/antigravity.git
cd antigravity
```

### 2. Get API Keys
- **Groq (FREE)**: https://console.groq.com
- **ICP Cycles**: For mainnet deployment

### 3. Run the Trading Agent
```powershell
cd agent
pip install -r requirements.txt
$env:GROQ_API_KEY='your-key'
python trading_agent.py
```

### 4. Run Backtest
```powershell
$env:GROQ_API_KEY='your-key'
python backtest.py
```

### 5. Deploy ICP Canister (Local)
```bash
# In WSL
dfx start --background
dfx deploy trading_agent
```

## Project Structure

```
antigravity/
├── agent/                  # Python AI Agent
│   ├── trading_agent.py    # Live trading with cointegration
│   ├── backtest.py         # AI-driven backtesting engine
│   └── requirements.txt    # Python dependencies
├── src/
│   ├── backend/
│   │   └── main.mo         # ICP Motoko canister
│   └── react-app/          # React dashboard
├── dfx.json                # ICP configuration
└── README.md
```

## Cointegration Statistics

The agent analyzes all trading pairs with:

| Metric | Description |
|--------|-------------|
| Z-Score | Standard deviations from mean |
| Half-Life | Days to mean-revert 50% |
| Hurst Exponent | < 0.5 = mean-reverting |
| ADF p-value | < 0.05 = stationary |
| R-squared | Regression fit quality |
| Risk % | Composite risk score |

## ICP Canister Methods

```motoko
// Queries
get_portfolio() -> Text
get_trade_history() -> Text
get_logs() -> Text

// Updates
execute_trade(action, asset, amount, price, reasoning) -> Text
reset() -> Text
```

## Hackathon Submission

- **Deployed on ICP**: Motoko canister for secure execution
- **Open Source**: Full codebase available
- **Real AI Decision Making**: Groq Llama 3.3 70B integration
- **Backtested**: 13-month historical data analysis

## License

MIT License
