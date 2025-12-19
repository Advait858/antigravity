# Antigravity 🚀

**Autonomous ICP Trading Agent with Statistical Arbitrage**

[![ICP](https://img.shields.io/badge/ICP-Internet%20Computer-blue)](https://internetcomputer.org)
[![Kybra](https://img.shields.io/badge/CDK-Kybra%20Python-green)](https://demergent-labs.github.io/kybra/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Zo House Hackathon Submission** - AI Agents for Trading Bounty ($500 USDC)

---

## 🎯 Overview

Antigravity is a fully autonomous trading agent deployed as a canister on the Internet Computer. It executes **statistical arbitrage** using pairs trading between BTC and ICP, powered by:

- **Kalman Filter** - Dynamic hedge ratio (β) estimation
- **ADF Cointegration Test** - Validates pair stationarity (p < 0.05)
- **Z-Score Signal Engine** - Entry/Exit/Stop-loss thresholds
- **Circuit Breaker** - Defensive halt on cointegration breakdown

## 🔬 Strategy: Engle-Granger Cointegration

```
                    ┌─────────────────┐
                    │  Price Feed API │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Kalman Filter  │
                    │  β = hedge ratio│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Spread = A - βB │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
       ┌──────▼──────┐ ┌─────▼─────┐ ┌──────▼──────┐
       │   ADF Test  │ │  Z-Score  │ │  Circuit    │
       │  p < 0.05?  │ │ = (S-μ)/σ │ │  Breaker    │
       └──────┬──────┘ └─────┬─────┘ └──────┬──────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────▼────────┐
                    │  Signal Engine  │
                    │  Long/Short/Exit│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Execution     │
                    └─────────────────┘
```

### Signal Thresholds

| Signal | Condition | Action |
|--------|-----------|--------|
| **ENTRY_LONG** | Z < -2σ | Buy spread (long A, short B) |
| **ENTRY_SHORT** | Z > +2σ | Sell spread (short A, long B) |
| **EXIT** | Z → 0 | Close position (mean reversion) |
| **STOP_LOSS** | \|Z\| > 4σ | Emergency exit + halt |

### Risk Management

- **Circuit Breaker** triggers on:
  - ADF failure (p ≥ 0.05 = cointegration broken)
  - 4σ deviation (extreme spread movement)
- **Slippage Model** with gas-adjusted profit thresholds
- **Cooldown Period** before resuming after halt

---

## 📁 Project Structure

```
antigravity/
├── dfx.json                 # ICP canister config
├── src/
│   ├── main.py              # Main canister (Kybra)
│   ├── engine/
│   │   ├── kalman.py        # Kalman Filter implementation
│   │   └── adf.py           # ADF stationarity test
│   ├── risk/
│   │   ├── circuit_breaker.py
│   │   └── slippage.py
│   └── data/
│       └── loader.py        # Mock price data
└── docs/
    └── STRATEGY.md
```

---

## 🛠️ Installation

### Prerequisites
- [dfx](https://internetcomputer.org/docs/current/developer-docs/setup/install) (ICP SDK)
- Python 3.10+
- WSL (for Windows)

### Setup

```bash
# Clone repository
git clone https://github.com/Advait858/antigravity.git
cd antigravity

# Install Kybra dfx extension (in WSL)
pip install kybra
python -m kybra install-dfx-extension

# Start local replica
dfx start --clean --background

# Deploy canister
dfx deploy
```

---

## 📡 Canister API

### Query Methods

```bash
# Health check
dfx canister call antigravity_bot get_health
# → ("System Operational")

# Get current state
dfx canister call antigravity_bot get_state
# → JSON with trading state, Z-score, hedge ratio, position

# Get strategy info
dfx canister call antigravity_bot get_strategy_info

# Get execution logs
dfx canister call antigravity_bot get_logs '(50)'

# Get analysis data
dfx canister call antigravity_bot get_analysis
```

### Update Methods

```bash
# Add price data (triggers strategy analysis)
dfx canister call antigravity_bot add_price_data '(42000.0, 12.5)'
# → Signal, Z-score, hedge ratio, ADF p-value

# Simulate with price series
dfx canister call antigravity_bot simulate_strategy '("[{\"a\":42000,\"b\":12},{\"a\":42100,\"b\":12.1}]")'

# Reset circuit breaker
dfx canister call antigravity_bot reset_breaker

# Update configuration
dfx canister call antigravity_bot update_config '("{\"entry_threshold\": 2.5}")'
```

---

## 🧮 Mathematical Foundations

### Kalman Filter (Hedge Ratio)

```
β_t = β_{t-1} + K_t × (y_t - x_t × β_{t-1})
K_t = P_{t-1} / (P_{t-1} + R)
P_t = (1 - K_t) × P_{t-1} + Q

Where:
  β = hedge ratio
  K = Kalman gain
  P = estimation covariance
  Q = process noise
  R = measurement noise
```

### ADF Test (Stationarity)

```
Δy_t = α + γ × y_{t-1} + ε_t

H₀: γ = 0 (unit root, non-stationary)
H₁: γ < 0 (stationary)

Reject H₀ if t-statistic < critical value (-2.86 at 5%)
```

### Z-Score (Signal)

```
Z = (spread - μ_spread) / σ_spread
spread = price_A - β × price_B
```

---

## 🏆 Hackathon Features

- ✅ **Pure Python** - No numpy/statsmodels (WASM compatible)
- ✅ **On-Chain Logic** - All computation in canister
- ✅ **Transparent Logging** - Full audit trail
- ✅ **Configurable** - Runtime parameter updates
- ✅ **Defensive** - Circuit breaker prevents losses
- ✅ **Tested** - Local replica deployment verified

---

## 📊 Example Output

```json
{
  "state": "scanning",
  "signal": "entry_long",
  "z_score": -2.34,
  "hedge_ratio": 3360.5,
  "spread": -0.0023,
  "adf_p": 0.032,
  "is_cointegrated": true
}
```

---

## 🔮 Future Enhancements

- [ ] HTTPS Outcalls for live price feeds (CoinGecko, Binance)
- [ ] DEX integration (ICPSwap, Sonic)
- [ ] Multi-pair support
- [ ] Machine learning signal enhancement
- [ ] Web dashboard for monitoring

---

## 📜 License

MIT License - see [LICENSE](LICENSE)

---

## 👨‍💻 Team

**Antigravity** - Built for Zo House Hackathon 2024

---

*"The only limit is gravity. We're going beyond."* 🌌
