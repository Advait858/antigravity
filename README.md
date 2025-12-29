# Antigravity: On-Chain AI Trading Agent

<div align="center">

**Autonomous Cryptocurrency Trading on the Internet Computer**

[![ICP Mainnet](https://img.shields.io/badge/ICP-Mainnet-blue?logo=internet-computer)](https://dashboard.internetcomputer.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Groq AI](https://img.shields.io/badge/AI-Groq%20Llama%203.3-orange)](https://groq.com)

[Quick Start](#-quick-start) • [Architecture](#-architecture) • [Mathematics](#-cointegration-mathematics) • [API Reference](#-api-reference)

</div>

---

## 🎯 Overview

**Antigravity** is an AI-powered cryptocurrency trading agent that operates entirely on-chain using the Internet Computer Protocol (ICP). It leverages:

- **ICP HTTPS Outcalls** — Real HTTP requests from Motoko canisters to external APIs
- **Statistical Arbitrage** — Cointegration-based pairs trading across 45 crypto pairs
- **AI Decision Engine** — Groq's Llama 3.3 70B for trade recommendations
- **Verification Artifacts** — Cryptographic proof of data authenticity

### Key Features

| Feature | Description |
|---------|-------------|
| 🌐 **On-Chain Oracle** | HTTPS outcalls to CoinGecko for real-time prices |
| 📊 **Cointegration Analysis** | Engle-Granger method with ADF stationarity testing |
| 🤖 **AI Recommendations** | Llama 3.3 70B analyzes opportunities |
| ✅ **Verification Artifacts** | Hash + timestamp for data authenticity |
| 🔒 **Fail-Closed Design** | Refuses to trade on fabricated/error data |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ICP MAINNET                                     │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                     Oracle Canister (oracle.mo)                        │ │
│  │                                                                         │ │
│  │   fetch_prices([assets])  ─────▶  CoinGecko API  ─────▶  Parse JSON    │ │
│  │           │                                                    │        │ │
│  │           └──────────  VerificationArtifact  ◀─────────────────┘        │ │
│  │                        • request_url                                    │ │
│  │                        • request_timestamp                              │ │
│  │                        • payload_hash                                   │ │
│  │                        • response_status                                │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                 Trading Agent Canister (main.mo)                       │ │
│  │                                                                         │ │
│  │   • execute_trade(action, asset, amount, price, reasoning)             │ │
│  │   • get_portfolio() → {usd: Float, positions: [...]}                   │ │
│  │   • get_trade_history() → [Trade]                                      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Python Agent Layer                                 │
│                                                                              │
│  ┌─────────────────┐    ┌──────────────────┐    ┌────────────────────────┐  │
│  │ trading_pipeline│───▶│   math_core.py   │───▶│   Groq AI (Llama 3.3) │  │
│  │      .py        │    │  22 functions    │    │   Trading Decision     │  │
│  └─────────────────┘    └──────────────────┘    └────────────────────────┘  │
│                                                                              │
│  Data Flow: Oracle ──▶ Historical (Binance) ──▶ Statistics ──▶ AI ──▶ Trade │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Mainnet Deployment

| Canister | ID | Purpose |
|----------|-------|---------|
| **Oracle** | `xkflk-rqaaa-aaaaj-qoaua-cai` | HTTPS outcalls, price fetching |
| **Trading Agent** | `xnen6-4iaaa-aaaaj-qoauq-cai` | Portfolio management, trade execution |

---

## 📐 Cointegration Mathematics

### Why Cointegration?

Traditional correlation breaks down in non-stationary time series (like crypto prices). Two assets may be correlated but drift apart over time. **Cointegration** identifies pairs that maintain a stable long-term equilibrium—when the spread deviates, it will revert.

### The Engle-Granger Two-Step Method

#### Step 1: OLS Regression

For assets A and B, estimate the hedging ratio:

```
A_t = α + β × B_t + ε_t
```

Where:
- `α` = Intercept (constant offset)
- `β` = Hedge ratio (how many units of B to hedge 1 unit of A)
- `ε_t` = Residual spread

**Implementation** (`math_core.py`):
```python
def ols_regression(y: list, x: list) -> OLSResult:
    """
    Ordinary Least Squares: Y = α + βX
    Returns: beta, alpha, r_squared, residuals
    """
    n = len(y)
    mean_x, mean_y = mean(x), mean(y)
    
    # β = Cov(X,Y) / Var(X)
    beta = covariance(x, y) / (variance(x) + EPSILON)
    
    # α = mean(Y) - β × mean(X)
    alpha = mean_y - beta * mean_x
    
    # Calculate residuals and R²
    residuals = [y[i] - (alpha + beta * x[i]) for i in range(n)]
    ss_res = sum(r**2 for r in residuals)
    ss_tot = sum((y[i] - mean_y)**2 for i in range(n))
    r_squared = 1 - (ss_res / (ss_tot + EPSILON))
    
    return OLSResult(beta, alpha, r_squared, residuals)
```

#### Step 2: Augmented Dickey-Fuller (ADF) Test

Test if the spread (residuals) is **stationary** (mean-reverting):

```
Δε_t = ρ × ε_{t-1} + u_t
```

- **H₀**: ρ = 0 (unit root, non-stationary)
- **H₁**: ρ < 0 (stationary, mean-reverting)

If the t-statistic < critical value (-2.86 at 5%), reject H₀ → pair is cointegrated.

**Implementation**:
```python
def adf_test(series: list) -> dict:
    """
    Augmented Dickey-Fuller Test for stationarity.
    Returns: t_stat, p_value, is_stationary
    """
    n = len(series)
    
    # Calculate first differences: Δy_t = y_t - y_{t-1}
    diff = [series[i] - series[i-1] for i in range(1, n)]
    lagged = series[:-1]
    
    # Regress Δy on y_{t-1}
    ols = ols_regression(diff, lagged)
    
    # t-statistic = β / SE(β)
    se_beta = std_dev(ols.residuals) / (std_dev(lagged) * math.sqrt(n-1) + EPSILON)
    t_stat = ols.beta / (se_beta + EPSILON)
    
    # Approximate p-value using critical values
    is_stationary = t_stat < ADF_CRITICAL_5PCT  # -2.86
    
    return {"t_stat": t_stat, "p_value": p_value, "is_stationary": is_stationary}
```

### Z-Score Signal Generation

Once cointegration is confirmed, calculate the **Z-Score** of the current spread:

```
Z = (spread_current - μ_spread) / σ_spread
```

Trading signals:
- **Z < -2.0**: LONG the spread (buy A, short B) — undervalued
- **Z > +2.0**: SHORT the spread (sell A, buy B) — overvalued
- **-2.0 ≤ Z ≤ +2.0**: HOLD — within normal range

### Half-Life of Mean Reversion

The half-life indicates how quickly the spread reverts to its mean:

```
Δspread_t = λ × spread_{t-1} + ε_t
Half-life = -ln(2) / ln(1 + λ)
```

**Interpretation**:
- Half-life of 5 days → Spread reverts 50% in ~5 trading periods
- Shorter half-life = faster reversion = more trading opportunities

### Hurst Exponent

Measures the persistence of the time series:

| Hurst (H) | Interpretation |
|-----------|----------------|
| H < 0.5 | Mean-reverting (anti-persistent) ✅ |
| H = 0.5 | Random walk |
| H > 0.5 | Trending (persistent) |

**Implementation**:
```python
def calculate_hurst_exponent(series: list, max_lag: int = 20) -> float:
    """
    Calculate Hurst Exponent using R/S analysis.
    H < 0.5 indicates mean reversion.
    """
    lags = range(2, min(max_lag, len(series) // 2))
    rs_values = []
    
    for lag in lags:
        # Calculate rescaled range R/S for this lag
        rs = calculate_rs(series, lag)
        rs_values.append((math.log(lag), math.log(rs)))
    
    # Hurst = slope of log(R/S) vs log(lag)
    hurst = linear_regression_slope(rs_values)
    return max(0.0, min(1.0, hurst))
```

### Risk Scoring

Combines all metrics into a 0-100 risk score:

```python
def calculate_risk_score(z_score_val, adf_p_value, r_squared) -> float:
    """
    Lower score = lower risk = better trade.
    
    Components:
    - Z-Score magnitude (distance from mean)
    - ADF p-value (stationarity confidence)
    - R² (regression fit quality)
    """
    z_component = max(0, 50 - abs(z_score_val) * 10)   # Prefer higher |Z|
    adf_component = adf_p_value * 30                    # Prefer lower p-value
    r2_component = (1 - r_squared) * 20                 # Prefer higher R²
    
    return z_component + adf_component + r2_component
```

---

## 🔌 HTTPS Outcalls Implementation

The Oracle canister makes **real HTTP requests** from the Internet Computer to external APIs:

```motoko
// oracle.mo - Core HTTPS Outcall
public func fetch_prices(assetIds: [Text]) : async Types.OracleResult {
    // Build URL
    let ids = Text.join(",", Iter.fromArray(assetIds));
    let url = "https://api.coingecko.com/api/v3/simple/price?ids=" # ids # 
              "&vs_currencies=usd&include_24hr_change=true";
    
    // Add cycles for HTTP request (consensus cost)
    Cycles.add<system>(50_000_000_000);  // 50B cycles
    
    // Make REAL HTTPS outcall via management canister
    let resp = await IC.http_request({
        url = url;
        max_response_bytes = ?Nat64.fromNat(10000);
        method = #get;
        headers = [{ name = "Accept"; value = "application/json" }];
        body = null;
        transform = ?{ function = transform; context = Blob.fromArray([]) };
    });
    
    // Return with verification artifact
    #ok({
        artifact = {
            request_url = url;
            request_timestamp = Time.now();
            response_status = resp.status;
            payload_hash = computeHash(body);
            payload_size = Text.size(body);
        };
        raw_json = body;
    })
}
```

### Verification Artifact

Every API call returns cryptographic proof:

| Field | Type | Description |
|-------|------|-------------|
| `request_url` | Text | Exact URL called |
| `request_timestamp` | Int | ICP timestamp (nanoseconds) |
| `response_status` | Nat | HTTP status code (200, 404, etc.) |
| `payload_hash` | Text | DJB2 hash of response body |
| `payload_size` | Nat | Response size in bytes |

This allows independent verification that data came from the claimed source at the claimed time.

---

## 📁 Project Structure

```
antigravity/
├── .env                        # Environment config (gitignored)
├── .gitignore                  # Git ignore rules
├── LICENSE                     # MIT License
├── README.md                   # This file
├── USAGE.md                    # How-to guide for judges
├── dfx.json                    # ICP canister configuration
├── canister_ids.json           # Mainnet canister IDs
├── requirements.txt            # Python dependencies
├── setup_wsl.sh                # WSL setup script
│
├── src/backend/                # Motoko Canisters
│   ├── oracle.mo               # HTTPS outcalls, price oracle
│   ├── main.mo                 # Trading agent, portfolio mgmt
│   └── Types.mo                # Shared type definitions
│
├── agent/                      # Python Trading Agent
│   ├── trading_pipeline.py     # Main pipeline (601 lines)
│   ├── math_core.py            # Statistics library (385 lines, 22 functions)
│   ├── trading_agent.py        # Trading cycle runner
│   ├── backtest.py             # Backtesting engine
│   └── requirements.txt        # Agent-specific dependencies
```

---

## 📚 API Reference

### Oracle Canister

| Function | Signature | Description |
|----------|-----------|-------------|
| `get_health` | `() → Text` | Returns JSON health status |
| `fetch_prices` | `([Text]) → OracleResult` | Fetch prices for given asset IDs |
| `fetch_top_prices` | `() → OracleResult` | Fetch BTC, ETH, ICP |
| `fetch_trading_prices` | `() → OracleResult` | Fetch all 10 trading assets |
| `get_last_json` | `() → Text` | Raw JSON from last fetch |
| `get_cycles` | `() → Nat` | Current cycle balance |

### Trading Agent Canister

| Function | Signature | Description |
|----------|-----------|-------------|
| `get_version` | `() → Text` | Agent version string |
| `get_portfolio` | `() → Text` | JSON portfolio state |
| `get_trade_history` | `() → Text` | JSON array of all trades |
| `execute_trade` | `(Text, Text, Float, Float, Text) → Text` | Execute a trade |
| `reset` | `() → Text` | Reset to $100k starting balance |
| `get_cycles` | `() → Nat` | Current cycle balance |

### Python Math Core (22 Functions)

| Category | Functions |
|----------|-----------|
| **Basic Statistics** | `mean`, `variance`, `std_dev`, `covariance`, `correlation` |
| **Regression** | `ols_regression` → `OLSResult(beta, alpha, r_squared, residuals)` |
| **Stationarity** | `adf_test` → `{t_stat, p_value, is_stationary}` |
| **Mean Reversion** | `z_score`, `calculate_half_life`, `calculate_hurst_exponent` |
| **Spread Analysis** | `calculate_spread_stats` → `{mean, std, bands, percentile, volatility}` |
| **Risk** | `calculate_risk_score`, `get_sandwich_signal` |

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/antigravity.git
cd antigravity

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 4. Run the trading pipeline
cd agent
python trading_pipeline.py
```

See [USAGE.md](USAGE.md) for detailed setup and verification instructions.

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

<div align="center">

**Built for the Internet Computer • Powered by AI • Secured by Math**

</div>
