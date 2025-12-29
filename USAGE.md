# Antigravity Usage Guide

**A step-by-step guide for judges and developers**

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Setup](#setup)
3. [Running the Trading Pipeline](#running-the-trading-pipeline)
4. [Oracle Canister Functions](#oracle-canister-functions)
5. [Trading Agent Functions](#trading-agent-functions)
6. [Python Modules](#python-modules)
7. [Demo Workflow](#demo-workflow)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Requirement | Version | Installation |
|-------------|---------|--------------|
| **WSL** | 2.0+ | `wsl --install` (Windows) |
| **dfx** | 0.24+ | `sh -ci "$(curl -fsSL https://internetcomputer.org/install.sh)"` |
| **Python** | 3.10+ | [python.org](https://python.org) |
| **Groq API Key** | Free | [console.groq.com](https://console.groq.com) |

---

## Setup

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/antigravity.git
cd antigravity
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Create `.env` in the project root:

```bash
# ============================================
# ANTIGRAVITY CONFIGURATION
# ============================================

# Groq AI API Key (get free at console.groq.com)
GROQ_API_KEY=your_groq_api_key_here

# ICP Mainnet Canister IDs (already deployed)
ORACLE_CANISTER_ID=xkflk-rqaaa-aaaaj-qoaua-cai
TRADING_CANISTER_ID=xnen6-4iaaa-aaaaj-qoauq-cai

# Network (ic = mainnet, local = local replica)
ICP_NETWORK=ic
DFX_NETWORK=ic
```

### 4. Verify WSL + dfx Setup

```bash
# In WSL
dfx --version
# Expected: dfx 0.24.0 or higher
```

---

## Running the Trading Pipeline

The main entry point is `agent/trading_pipeline.py`:

```bash
cd agent
python trading_pipeline.py
```

### Expected Output

```
======================================================================
🚀 ANTIGRAVITY TRADING PIPELINE v2.0
   On-Chain AI Trading Agent with HTTPS Outcalls
======================================================================

📋 CONFIGURATION
   Network: ic
   Oracle: xkflk-rqaaa-aaaaj-qoaua-cai
   Groq AI: ✅ Configured

1️⃣  ORACLE HEALTH CHECK
   ✅ Oracle is healthy

2️⃣  FETCHING LIVE PRICES (HTTPS Outcall)
   ✅ Prices fetched successfully!

   💰 CURRENT PRICES:
      BTC: $87,614.00 (+0.17%)
      ETH: $3,245.00 (-1.23%)
      ...

   📜 VERIFICATION ARTIFACT:
      URL: https://api.coingecko.com/api/v3/simple/price...
      Hash: 2847159263
      Status: 200

3️⃣  FETCHING HISTORICAL DATA (Binance)
   ✅ BTC: 365 days
   ✅ ETH: 365 days
   ...

4️⃣  COINTEGRATION ANALYSIS
   Found 5 opportunities:
   1. BTC-ETH: Z=1.84, Risk=23.5% ➡️
   2. SOL-AVAX: Z=-2.31, Risk=18.2% 📈
   ...

5️⃣  AI RECOMMENDATION (Groq Llama 3.3)
   🤖 RECOMMENDATION:
      Action: BUY
      Asset: SOL
      Confidence: 72%
      Reasoning: SOL-AVAX pair showing strong mean reversion...

======================================================================
✅ PIPELINE COMPLETE
======================================================================
```

---

## Oracle Canister Functions

Canister ID: `xkflk-rqaaa-aaaaj-qoaua-cai`

### get_health

Check if the oracle is operational.

```bash
dfx canister --network ic call xkflk-rqaaa-aaaaj-qoaua-cai get_health '()'
```

**Response:**
```
("{\"status\":\"healthy\",\"version\":\"2.6\",\"fetches\":142,\"cycles\":4823491827364}")
```

---

### fetch_prices

Fetch prices for specific assets via HTTPS outcall.

```bash
dfx canister --network ic call xkflk-rqaaa-aaaaj-qoaua-cai fetch_prices '(vec {"bitcoin"; "ethereum"})'
```

**Response:**
```
(
  variant {
    ok = record {
      artifact = record {
        request_url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd&include_24hr_change=true";
        request_timestamp = 1_703_856_000_000_000_000 : int;
        response_status = 200 : nat;
        payload_hash = "2847159263";
        payload_size = 156 : nat;
      };
      raw_json = "{\"bitcoin\":{\"usd\":87614,\"usd_24h_change\":0.17},\"ethereum\":{\"usd\":3245,\"usd_24h_change\":-1.23}}";
    }
  },
)
```

---

### fetch_top_prices

Fetch BTC, ETH, ICP prices.

```bash
dfx canister --network ic call xkflk-rqaaa-aaaaj-qoaua-cai fetch_top_prices '()'
```

---

### fetch_trading_prices

Fetch all 10 trading assets (BTC, ETH, ICP, SOL, XRP, ADA, DOGE, DOT, AVAX, LINK).

```bash
dfx canister --network ic call xkflk-rqaaa-aaaaj-qoaua-cai fetch_trading_prices '()'
```

---

### get_last_json

Get raw JSON from the last successful fetch.

```bash
dfx canister --network ic call xkflk-rqaaa-aaaaj-qoaua-cai get_last_json '()'
```

---

### get_cycles

Check the canister's cycle balance.

```bash
dfx canister --network ic call xkflk-rqaaa-aaaaj-qoaua-cai get_cycles '()'
```

---

## Trading Agent Functions

Canister ID: `xnen6-4iaaa-aaaaj-qoauq-cai`

### get_version

```bash
dfx canister --network ic call xnen6-4iaaa-aaaaj-qoauq-cai get_version '()'
```

**Response:** `"Antigravity v2.0 - Hybrid AI Trading Agent"`

---

### get_portfolio

Get current portfolio state.

```bash
dfx canister --network ic call xnen6-4iaaa-aaaaj-qoauq-cai get_portfolio '()'
```

**Response:**
```json
{
  "usd": 98500.00,
  "positions": [
    {"asset": "BTC", "amount": 0.05, "avg_price": 87000.00},
    {"asset": "ETH", "amount": 1.5, "avg_price": 3200.00}
  ]
}
```

---

### get_trade_history

Get all executed trades.

```bash
dfx canister --network ic call xnen6-4iaaa-aaaaj-qoauq-cai get_trade_history '()'
```

---

### execute_trade

Execute a trade.

```bash
dfx canister --network ic call xnen6-4iaaa-aaaaj-qoauq-cai execute_trade '("BUY", "BTC", 0.01, 87000.0, "Cointegration signal")'
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| action | Text | "BUY" or "SELL" |
| asset | Text | Asset symbol (e.g., "BTC") |
| amount | Float | Quantity to trade |
| price | Float | Current price in USD |
| reasoning | Text | Trade justification |

---

### reset

Reset portfolio to $100,000 starting balance.

```bash
dfx canister --network ic call xnen6-4iaaa-aaaaj-qoauq-cai reset '()'
```

---

## Python Modules

### trading_pipeline.py (601 lines)

Main orchestration script.

| Function | Description |
|----------|-------------|
| `run_full_pipeline()` | Execute complete trading cycle |
| `run_quick_test()` | Quick oracle connectivity test |
| `call_oracle_health()` | Check oracle status |
| `call_oracle_fetch_prices()` | Fetch prices from oracle |
| `fetch_historical_prices(asset, days)` | Get historical data from Binance |
| `analyze_pair(asset_a, asset_b, history)` | Cointegration analysis |
| `find_best_opportunities(history, top_n)` | Find top trading opportunities |
| `get_ai_recommendation(prices, artifact, opportunities)` | Get AI trading decision |

---

### math_core.py (385 lines, 22 functions)

Statistical analysis library.

| Category | Functions |
|----------|-----------|
| **Basic Stats** | `mean(data)`, `variance(data)`, `std_dev(data)`, `covariance(x, y)`, `correlation(x, y)` |
| **Regression** | `ols_regression(y, x) → OLSResult` |
| **Stationarity** | `adf_test(series) → {t_stat, p_value, is_stationary}` |
| **Mean Reversion** | `z_score(current, history, window)`, `calculate_half_life(spread)`, `calculate_hurst_exponent(series)` |
| **Spread** | `calculate_spread_stats(spread) → {mean, std, bands, percentile}` |
| **Risk** | `calculate_risk_score(z, p_value, r_squared)`, `get_sandwich_signal(stats, z)` |

---

### trading_agent.py (346 lines)

Alternative trading cycle runner.

| Function | Description |
|----------|-------------|
| `fetch_historical_data(asset, days)` | Get OHLCV from Binance |
| `fetch_all_historical()` | Fetch all 10 assets |
| `fetch_current_prices()` | Get live prices |
| `analyze_pair(asset_a, asset_b)` | Full pair analysis |
| `scan_all_pairs()` | Analyze all 45 pairs |
| `call_canister(method, args)` | Generic canister call |
| `get_ai_decision(prices, portfolio, signals)` | AI recommendation |
| `run_trading_cycle()` | Main loop |

---

### backtest.py (500+ lines)

Backtesting engine for strategy validation.

| Function | Description |
|----------|-------------|
| `load_historical_data()` | Load from CSV/Binance |
| `simulate_trading_day(date, history)` | Simulate one day |
| `run_backtest(start, end)` | Full backtest |
| `calculate_metrics(trades)` | Performance metrics |

---

## Demo Workflow

### For Judges: Quick Verification (5 minutes)

```bash
# 1. Check oracle health
dfx canister --network ic call xkflk-rqaaa-aaaaj-qoaua-cai get_health '()'

# 2. Verify HTTPS outcall works (REAL HTTP request!)
dfx canister --network ic call xkflk-rqaaa-aaaaj-qoaua-cai fetch_top_prices '()'

# 3. Run full Python pipeline
cd agent
python trading_pipeline.py
```

### Full Demonstration (15 minutes)

1. **Verify Oracle HTTPS Outcall**
   ```bash
   dfx canister --network ic call xkflk-rqaaa-aaaaj-qoaua-cai fetch_trading_prices '()'
   ```
   - Note: This makes a REAL HTTP request to CoinGecko from the ICP canister
   - Check the `verification_artifact` in the response

2. **Check Trading Agent State**
   ```bash
   dfx canister --network ic call xnen6-4iaaa-aaaaj-qoauq-cai get_portfolio '()'
   ```

3. **Run Full Trading Pipeline**
   ```bash
   cd agent
   python trading_pipeline.py
   ```

4. **Execute a Test Trade**
   ```bash
   dfx canister --network ic call xnen6-4iaaa-aaaaj-qoauq-cai execute_trade '("BUY", "ICP", 10.0, 12.5, "Demo trade")'
   ```

5. **View Trade History**
   ```bash
   dfx canister --network ic call xnen6-4iaaa-aaaaj-qoauq-cai get_trade_history '()'
   ```

---

## Troubleshooting

### "Groq AI: ❌ Not set"

The GROQ_API_KEY is not loaded from `.env`.

**Fix:** Ensure `.env` is in the project root (not in `agent/`) and contains:
```
GROQ_API_KEY=gsk_your_key_here
```

---

### "Oracle health check failed"

The oracle canister may be out of cycles.

**Check cycles:**
```bash
dfx canister --network ic call xkflk-rqaaa-aaaaj-qoaua-cai get_cycles '()'
```

---

### "HTTPS outcall failed"

CoinGecko may be rate-limiting. Wait 60 seconds and retry.

---

### WSL/dfx not found

Ensure dfx is installed and sourced:
```bash
source ~/.local/share/dfx/env
dfx --version
```

---

## Support

For issues or questions, please open a GitHub issue.

---

<div align="center">

**Antigravity** — On-Chain AI Trading for the Internet Computer

</div>
