# 📊 Antigravity Trading Strategy

## Engle-Granger Cointegration with Multi-Asset Pairs Trading

---

## 🎯 Overview

Antigravity implements a **statistical arbitrage strategy** based on the Engle-Granger two-step cointegration methodology. The agent analyzes **45 cryptocurrency pairs** (all combinations of 10 assets) to identify mean-reverting relationships.

---

## 📐 Mathematical Foundation

### 1. Cointegration Theory

Two time series Y and X are **cointegrated** if:
- Both are individually non-stationary (I(1))
- A linear combination (the spread) is stationary (I(0))

```
Spread = Y - β·X - α
```

When the spread is stationary, it tends to revert to its mean, creating trading opportunities.

### 2. Engle-Granger Two-Step Method

**Step 1: OLS Regression**

Estimate the cointegrating relationship:
```
Y_t = α + β·X_t + ε_t
```

Where:
- `Y_t` = Price of asset A
- `X_t` = Price of asset B
- `β` = Hedge ratio (how many units of B per unit of A)
- `α` = Constant (intercept)
- `ε_t` = Residuals (the spread)

**Step 2: ADF Test on Residuals**

Test if residuals are stationary:
```
Δε_t = γ·ε_{t-1} + u_t

H₀: γ = 0 (unit root → non-stationary → NOT cointegrated)
H₁: γ < 0 (stationary → cointegrated)
```

Critical values (5% significance):
- t-statistic < -2.86 → Reject H₀ → **Cointegrated**

### 3. Half-Life of Mean Reversion

The half-life tells us how quickly the spread reverts to its mean:

```
Half-life = -ln(2) / γ
```

| Half-Life | Trading Horizon | Strategy |
|-----------|-----------------|----------|
| < 7 days | Intraday/Daily | Aggressive entry |
| 7-14 days | Swing trade | Standard entry |
| 14-30 days | Position trade | Patient entry |
| > 30 days | Long-term | Consider skipping |

### 4. Z-Score Signal Generation

The Z-score measures how many standard deviations the spread is from its mean:

```
Z = (Spread - μ) / σ
```

Where:
- `μ` = Mean of spread over lookback window
- `σ` = Standard deviation of spread

---

## 🚦 Trading Signals

### Entry Conditions

| Signal | Condition | Position |
|--------|-----------|----------|
| **LONG_SPREAD** | Z < -2 | Long A, Short B |
| **SHORT_SPREAD** | Z > +2 | Short A, Long B |

### Exit Conditions

| Signal | Condition | Action |
|--------|-----------|--------|
| **EXIT** | |Z| < 0.5 | Close position (mean reversion complete) |
| **STOP_LOSS** | |Z| > 4 | Emergency exit (cointegration may be broken) |

### Trade Logic

**LONG_SPREAD (Z < -2)**
- Spread is abnormally LOW
- Expect Asset A to outperform Asset B
- Buy A, Sell B
- Profit when spread returns to mean

**SHORT_SPREAD (Z > +2)**
- Spread is abnormally HIGH
- Expect Asset B to outperform Asset A
- Sell A, Buy B
- Profit when spread returns to mean

---

## 📈 Rolling Window Analysis

The agent tests cointegration over multiple time windows:

| Window | Purpose |
|--------|---------|
| 30 days | Short-term relationship |
| 60 days | Medium-term stability |
| 90 days | Quarterly trend |
| 180 days | Long-term reliability |

A pair is considered **highly reliable** if cointegrated in multiple windows.

---

## 🎯 Confidence Scoring

Each trade signal includes a confidence score (0-100):

| Factor | Points | Condition |
|--------|--------|-----------|
| Cointegration | +30 | ADF p < 0.05 |
| Strong Z-score | +20 | |Z| > 2.5 |
| High R² | +20 | R² > 0.7 |
| Fast half-life | +15 | < 14 days |
| Multiple windows | +15 | 2+ windows cointegrated |

**Confidence Levels:**
- **HIGH**: ≥ 70 points
- **MEDIUM**: 50-69 points
- **LOW**: < 50 points

---

## 💰 Position Sizing

Recommended position sizing based on confidence:

| Confidence | Position Size |
|------------|---------------|
| HIGH | 100% of allocation |
| MEDIUM | 50% of allocation |
| LOW | 25% of allocation |

---

## ⚠️ Risk Management

### Stop-Loss Rules

1. **Z-Score Stop**: Exit if |Z| > 4σ
2. **Time Stop**: Exit if half-life exceeded by 2x
3. **Cointegration Break**: Exit if p-value rises above 0.10

### Circuit Breaker Triggers

- ADF p-value ≥ 0.10 (cointegration weakening)
- 3 consecutive losing trades
- Maximum drawdown exceeded

---

## 📊 Example Trade

### Setup: BTC/ETH LONG_SPREAD

```
Current State:
- BTC: $88,000
- ETH: $3,000
- Hedge Ratio (β): 29.33
- Spread: $88,000 - 29.33 × $3,000 = $10
- Mean Spread: $50
- Std Dev: $15
- Z-Score: (10 - 50) / 15 = -2.67

Signal: LONG_SPREAD (Z < -2)
Confidence: HIGH (75 points)
Half-life: 8 days
```

### Execution

```
Entry:
  Long 1 BTC @ $88,000
  Short 29.33 ETH @ $3,000 = $88,000

Expected Exit (after 8 days):
  Spread reverts to mean ($50)
  BTC rises or ETH falls (or both)

Profit Target:
  Spread moves from $10 → $50 = +$40
  Return ≈ +$40 / $88,000 ≈ 0.045% per leg
```

---

## 🔧 Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MIN_DATA_POINTS` | 10 | Minimum history for analysis |
| `ENTRY_THRESHOLD` | 2.0 | Z-score for entry |
| `EXIT_THRESHOLD` | 0.5 | Z-score for exit |
| `STOP_LOSS_THRESHOLD` | 4.0 | Emergency exit Z-score |
| `COINTEGRATION_P_THRESHOLD` | 0.05 | ADF p-value cutoff |
| `ROLLING_WINDOWS` | [30, 60, 90, 180] | Analysis windows |

---

## 📚 References

1. Engle, R. F., & Granger, C. W. J. (1987). "Co-Integration and Error Correction: Representation, Estimation, and Testing"
2. Vidyamurthy, G. (2004). "Pairs Trading: Quantitative Methods and Analysis"
3. Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). "Pairs Trading: Performance of a Relative-Value Arbitrage Rule"

---

*Strategy implemented in Antigravity v5.0.0-full-cointegration*
