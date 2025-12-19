# Trading Strategy Documentation

## Engle-Granger Cointegration for Pairs Trading

### 1. Theoretical Foundation

Cointegration occurs when two non-stationary time series maintain a stable long-term relationship. For pairs trading:

- **Asset A** (BTC): Non-stationary random walk
- **Asset B** (ICP): Non-stationary random walk
- **Spread** = A - β × B: Stationary (mean-reverting)

The hedge ratio β makes the spread stationary, meaning deviations from the mean are temporary and will revert.

### 2. Kalman Filter for Dynamic Hedge Ratio

Unlike static OLS regression, we use a Kalman filter to track the time-varying hedge ratio:

```python
# State-space model
State:       β_t (hedge ratio)
Observation: price_A = β × price_B + noise

# Kalman equations
Predict:  β_pred = β_{t-1}
          P_pred = P_{t-1} + Q
          
Update:   K = P_pred × x / (x² × P_pred + R)
          β = β_pred + K × (y - x × β_pred)
          P = (1 - K × x) × P_pred
```

**Parameters:**
- Q (process noise): 0.0001 - How much β can change per period
- R (measurement noise): 0.001 - Observation uncertainty

### 3. ADF Test for Stationarity

The Augmented Dickey-Fuller test checks if the spread is stationary:

```
Δspread_t = γ × spread_{t-1} + ε_t

H₀: γ = 0 → unit root (non-stationary)
H₁: γ < 0 → stationary (mean-reverting)
```

**Decision Rule:**
- p < 0.05: Reject H₀ → Spread is stationary → Safe to trade
- p ≥ 0.05: Cannot reject H₀ → Cointegration broken → HALT

### 4. Z-Score Signal Generation

The Z-score normalizes the spread to identify trading opportunities:

```
Z = (spread - mean(spread)) / std(spread)
```

**Trading Signals:**

| Z-Score | Interpretation | Action |
|---------|---------------|--------|
| Z < -2 | Spread undervalued | LONG spread (buy A, sell B) |
| Z > +2 | Spread overvalued | SHORT spread (sell A, buy B) |
| Z → 0 | Mean reversion complete | EXIT position |
| \|Z\| > 4 | Extreme deviation | STOP-LOSS + HALT |

### 5. Risk Management

#### Circuit Breaker Triggers:
1. **ADF Failure**: p-value ≥ 0.05 (cointegration broken)
2. **4σ Stop-Loss**: Spread deviation too extreme
3. **Consecutive Losses**: >5 losing trades
4. **Max Drawdown**: >10% portfolio loss

#### Recovery:
- Cooldown period (5 minutes)
- Half-open testing before full resume
- Manual override available

### 6. Execution Considerations

#### Slippage Model:
```
slippage = base_bps + size_impact × sqrt(trade_size / volume)
```

#### Profit Threshold:
```
min_profit = 2 × (slippage_cost + gas_fee + dex_fee)
```

Only execute if expected profit > min_profit.

### 7. State Machine

```
IDLE → (data collected) → SCANNING
SCANNING → (Z < -2) → LONG_SPREAD
SCANNING → (Z > +2) → SHORT_SPREAD
LONG_SPREAD → (Z → 0) → SCANNING
SHORT_SPREAD → (Z → 0) → SCANNING
ANY → (circuit breaker) → HALTED
HALTED → (manual reset) → SCANNING
```

---

## Implementation Notes

### Pure Python (No NumPy)

All math is implemented in pure Python for WASM compatibility:
- Matrix operations via nested lists
- OLS regression via normal equations
- Statistics via iterative calculations

### Logging

Every significant event is logged:
- Price updates
- Signal generation
- Position changes
- Circuit breaker activations

Access logs via `get_logs()` query method.
