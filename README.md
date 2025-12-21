<p align="center">
  <img src="https://img.shields.io/badge/🚀-ANTIGRAVITY-black?style=for-the-badge&labelColor=blueviolet" alt="Antigravity">
</p>

<h1 align="center">Antigravity</h1>

<p align="center">
  <strong>🤖 Autonomous Multi-Asset Cointegration Trading Agent on ICP</strong>
</p>

<p align="center">
  <a href="https://internetcomputer.org"><img src="https://img.shields.io/badge/ICP-Internet%20Computer-blue?style=flat-square&logo=dfinity" alt="ICP"></a>
  <a href="https://demergent-labs.github.io/kybra/"><img src="https://img.shields.io/badge/CDK-Kybra%20Python-green?style=flat-square&logo=python" alt="Kybra"></a>
  <a href="#"><img src="https://img.shields.io/badge/Version-5.0.0-orange?style=flat-square" alt="Version"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Pairs_Analyzed-45-success?style=flat-square" alt="45 Pairs">
  <img src="https://img.shields.io/badge/Assets-Top_10_Crypto-blue?style=flat-square" alt="10 Assets">
  <img src="https://img.shields.io/badge/Analysis-Engle--Granger-purple?style=flat-square" alt="Engle-Granger">
</p>

---

> 🏆 **Zo House Hackathon Submission** - AI Agents for Trading Bounty

---

## ✨ What is Antigravity?

Antigravity is a **fully autonomous trading agent** deployed as a canister on the Internet Computer Protocol (ICP). It performs **statistical arbitrage** by analyzing **45 cryptocurrency pairs** using advanced cointegration analysis.

### 🎯 Key Features

| Feature | Description |
|---------|-------------|
| 🔗 **45 Pair Analysis** | All combinations of top 10 cryptocurrencies |
| 📊 **Engle-Granger Test** | Two-step cointegration with ADF on residuals |
| 📈 **Rolling Windows** | 30, 60, 90, 180-day cointegration stability |
| 💰 **Live Price Feeds** | HTTPS outcalls to CoinGecko API |
| 🎲 **Detailed Signals** | Entry ranges, targets, stop-loss, risk/reward |
| 👛 **Wallet Integration** | Principal-based user registration |
| ⚡ **On-Chain Logic** | 100% computation on ICP canister |

---

## 🪙 Tracked Assets

<p align="center">
  <img src="https://img.shields.io/badge/BTC-Bitcoin-orange?style=for-the-badge&logo=bitcoin" alt="BTC">
  <img src="https://img.shields.io/badge/ETH-Ethereum-blue?style=for-the-badge&logo=ethereum" alt="ETH">
  <img src="https://img.shields.io/badge/SOL-Solana-purple?style=for-the-badge" alt="SOL">
  <img src="https://img.shields.io/badge/XRP-Ripple-gray?style=for-the-badge" alt="XRP">
  <img src="https://img.shields.io/badge/DOGE-Dogecoin-yellow?style=for-the-badge" alt="DOGE">
</p>
<p align="center">
  <img src="https://img.shields.io/badge/ADA-Cardano-blue?style=for-the-badge" alt="ADA">
  <img src="https://img.shields.io/badge/AVAX-Avalanche-red?style=for-the-badge" alt="AVAX">
  <img src="https://img.shields.io/badge/DOT-Polkadot-pink?style=for-the-badge" alt="DOT">
  <img src="https://img.shields.io/badge/LINK-Chainlink-blue?style=for-the-badge" alt="LINK">
  <img src="https://img.shields.io/badge/ICP-Internet%20Computer-purple?style=for-the-badge" alt="ICP">
</p>

---

## 🔬 Strategy: Engle-Granger Cointegration

```
                         ┌──────────────────────────┐
                         │   📡 CoinGecko API       │
                         │   Live Price Feeds       │
                         └───────────┬──────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
              ┌─────▼─────┐    ┌─────▼─────┐    ┌─────▼─────┐
              │   BTC     │    │   ETH     │    │  ...×10   │
              │  OHLCV    │    │  OHLCV    │    │  Assets   │
              └─────┬─────┘    └─────┬─────┘    └─────┬─────┘
                    │                │                │
                    └────────────────┼────────────────┘
                                     │
                         ┌───────────▼───────────┐
                         │   45 PAIR ANALYSIS    │
                         │   C(10,2) = 45 pairs  │
                         └───────────┬───────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         │                           │                           │
   ┌─────▼─────┐              ┌──────▼──────┐             ┌──────▼──────┐
   │   OLS     │              │   ADF Test  │             │   Rolling   │
   │ Regression│              │  on Spread  │             │   Windows   │
   │ β = hedge │              │  p < 0.05?  │             │ 30/60/90/180│
   └─────┬─────┘              └──────┬──────┘             └──────┬──────┘
         │                           │                           │
         └───────────────────────────┼───────────────────────────┘
                                     │
                         ┌───────────▼───────────┐
                         │   SIGNAL GENERATION   │
                         │  Z-Score + Half-Life  │
                         └───────────┬───────────┘
                                     │
                         ┌───────────▼───────────┐
                         │  📊 TRADE SIGNALS     │
                         │  Entry/Target/Stop    │
                         └───────────────────────┘
```

### 📐 The Math

**Step 1: OLS Regression**
```
Y = α + β·X + ε
Spread = Price_A - β·Price_B - α
```

**Step 2: ADF Test on Residuals**
```
Δε_t = γ·ε_{t-1} + u_t
H₀: γ = 0 (non-stationary)
H₁: γ < 0 (stationary → cointegrated)
```

**Step 3: Half-Life of Mean Reversion**
```
t_{1/2} = -ln(2) / γ
```

---

## 📊 Signal Thresholds

| Signal | Z-Score | Action | Timing |
|--------|---------|--------|--------|
| 🟢 **LONG_SPREAD** | Z < -2σ | Buy A, Sell B | Based on half-life |
| 🔴 **SHORT_SPREAD** | Z > +2σ | Sell A, Buy B | Based on half-life |
| 🟡 **PREPARE** | 1.5σ < \|Z\| < 2σ | Watch closely | - |
| ⚪ **EXIT** | Z → 0 | Close position | Mean reversion |
| 🛑 **STOP_LOSS** | \|Z\| > 4σ | Emergency exit | Immediate |

### ⏱️ Timing Recommendations

| Half-Life | Timing | Description |
|-----------|--------|-------------|
| < 7 days | **NOW** | Execute immediately |
| 7-14 days | **2H** | Within 2 hours |
| 14-30 days | **2D** | Position over 1-2 days |
| > 30 days | **1W** | Longer-term trade |

---

## 🚀 Quick Start

### Prerequisites

- [dfx](https://internetcomputer.org/docs/current/developer-docs/setup/install) - ICP SDK
- Python 3.10+
- WSL (for Windows users)

### Installation

```bash
# Clone repository
git clone https://github.com/Advait858/antigravity.git
cd antigravity

# Install Kybra (in WSL)
pip install kybra
python -m kybra install-dfx-extension

# Start local replica
dfx start --clean --background

# Deploy canister
dfx deploy

# Load sample data and run analysis
dfx canister call antigravity_bot load_sample_data
```

---

## 📡 API Reference

### 🔍 Query Methods

```bash
# Health check
dfx canister call antigravity_bot get_health
# → ("System Operational")

# Version
dfx canister call antigravity_bot get_version
# → ("5.0.0-full-cointegration")

# Get all 45 pairs analysis
dfx canister call antigravity_bot get_all_pair_analysis

# Get cointegrated pairs only
dfx canister call antigravity_bot get_cointegrated_pairs

# Get trading signals with full details
dfx canister call antigravity_bot get_trading_signals

# Get specific pair analysis
dfx canister call antigravity_bot 'get_pair_detail("BTC/ETH")'

# Portfolio analysis
dfx canister call antigravity_bot get_portfolio_analysis

# Current state
dfx canister call antigravity_bot get_state
```

### 🔄 Update Methods

```bash
# Load sample data (100 days, all assets)
dfx canister call antigravity_bot load_sample_data

# Fetch 1-year historical data for an asset
dfx canister call antigravity_bot 'fetch_ohlcv_data("bitcoin")'

# Fetch current prices for all assets
dfx canister call antigravity_bot fetch_current_prices

# Fetch prices and run analysis
dfx canister call antigravity_bot fetch_and_analyze_all

# Run cointegration analysis
dfx canister call antigravity_bot run_analysis

# Register wallet
dfx canister call antigravity_bot register_wallet
```

---

## 📈 Example Output

### Trading Signal

```json
{
  "pair": "BTC/ETH",
  "action": "LONG_SPREAD",
  "direction": {"BTC": "BUY", "ETH": "SELL"},
  "timing": "NOW",
  "timing_description": "Execute immediately - fast mean reversion",
  "z_score": -2.34,
  "half_life_days": 5.2,
  "entry_range": {
    "BTC": [87500, 88500],
    "ETH": [2950, 3050]
  },
  "targets": {"BTC": 91000, "ETH": 2900},
  "stop_loss": {"BTC": 84000, "ETH": 3150},
  "potential_upside_pct": 3.5,
  "potential_risk_pct": 5.0,
  "risk_reward_ratio": 0.7,
  "confidence_level": "HIGH",
  "confidence_score": 75
}
```

### Cointegrated Pairs

```json
{
  "cointegrated_pairs": [
    {"pair": "SOL/LINK", "z_score": -2.1, "half_life": 8.3, "adf_p": 0.023},
    {"pair": "ETH/DOT", "z_score": 1.8, "half_life": 12.1, "adf_p": 0.041},
    {"pair": "BTC/ADA", "z_score": -1.6, "half_life": 15.7, "adf_p": 0.048}
  ],
  "count": 12
}
```

---

## 🏗️ Architecture

```
antigravity/
├── 📄 dfx.json              # ICP canister configuration
├── 📄 kybra.json            # Kybra Python settings
├── 📁 src/
│   ├── 🐍 main.py           # Main canister (1000+ lines)
│   │   ├── OHLCV data management
│   │   ├── Engle-Granger cointegration
│   │   ├── Rolling window analysis
│   │   ├── Signal generation
│   │   └── HTTPS outcalls
│   ├── 📁 engine/
│   │   ├── kalman.py        # Kalman Filter
│   │   └── adf.py           # ADF test
│   ├── 📁 risk/
│   │   ├── circuit_breaker.py
│   │   └── slippage.py
│   ├── 📁 strategy/
│   │   └── cointegration.py # PairsTrader class
│   └── 📁 data/
│       └── loader.py        # Data utilities
├── 📁 docs/
│   └── STRATEGY.md          # Strategy documentation
└── 🧪 test_agent.sh         # Test script
```

---

## 🏆 Hackathon Highlights

| Requirement | Status | Details |
|-------------|--------|---------|
| ✅ On-chain Agent | Complete | Canister on ICP |
| ✅ Autonomous Trading | Complete | Automated signals |
| ✅ Live Data | Complete | CoinGecko HTTPS outcalls |
| ✅ Multi-Asset | Complete | 10 assets, 45 pairs |
| ✅ Transparency | Complete | Full logging, query APIs |
| ✅ Risk Management | Complete | Stop-loss, confidence scores |
| ✅ Pure Python | Complete | WASM compatible |

---

## 📊 Performance Metrics

The Engle-Granger analysis provides:

- **Half-life**: How fast the spread mean-reverts
- **R-squared**: Regression fit quality
- **ADF p-value**: Cointegration strength
- **Z-score**: Current deviation from mean
- **Rolling stability**: Consistency across time windows

---

## 🔮 Future Roadmap

- [ ] DEX Integration (ICPSwap, Sonic)
- [ ] Timer-based auto-fetch (heartbeat)
- [ ] Historical trade performance tracking
- [ ] Web dashboard for monitoring
- [ ] Multi-timeframe analysis
- [ ] ML-enhanced signal filtering

---

## 🛡️ Risk Disclaimer

This is an experimental trading agent for educational and hackathon purposes. Cryptocurrency trading involves substantial risk of loss. Always conduct your own research and never invest more than you can afford to lose.

---

## 📜 License

MIT License - see [LICENSE](LICENSE)

---

## 👨‍💻 Author

**Advait** - Built for Zo House Hackathon 2024

---

<p align="center">
  <strong>🌌 "The only limit is gravity. We're going beyond." 🚀</strong>
</p>

<p align="center">
  <a href="https://github.com/Advait858/antigravity">
    <img src="https://img.shields.io/badge/⭐_Star_this_repo-black?style=for-the-badge" alt="Star">
  </a>
</p>
