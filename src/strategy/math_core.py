"""
Math Core - Pure NumPy Econometrics Engine
WASM-safe implementation without statsmodels dependency

Implements:
- OLS Regression for hedge ratio
- Augmented Dickey-Fuller (ADF) test
- Z-score calculation
- Signal generation
"""

import math
from typing import Tuple, List, Optional

# ============================================================
# CONFIGURATION
# ============================================================

# Signal thresholds
ENTRY_THRESHOLD = 2.0      # |Z| > 2 for entry
EXIT_THRESHOLD = 0.5       # |Z| < 0.5 for exit
STOP_LOSS_THRESHOLD = 4.0  # |Z| > 4 for stop-loss
ADF_SIGNIFICANCE = 0.05    # p-value threshold

# ============================================================
# NUMERICAL UTILITIES
# ============================================================

def mean(arr: List[float]) -> float:
    """Calculate arithmetic mean"""
    if not arr:
        return 0.0
    return sum(arr) / len(arr)


def variance(arr: List[float]) -> float:
    """Calculate population variance"""
    if len(arr) < 2:
        return 0.0
    m = mean(arr)
    return sum((x - m) ** 2 for x in arr) / len(arr)


def std_dev(arr: List[float]) -> float:
    """Calculate population standard deviation"""
    return math.sqrt(variance(arr))


def covariance(x: List[float], y: List[float]) -> float:
    """Calculate covariance between two arrays"""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    mx, my = mean(x), mean(y)
    return sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / len(x)


def correlation(x: List[float], y: List[float]) -> float:
    """Calculate Pearson correlation coefficient"""
    sx, sy = std_dev(x), std_dev(y)
    if sx == 0 or sy == 0:
        return 0.0
    return covariance(x, y) / (sx * sy)


# ============================================================
# OLS REGRESSION (Pure Python)
# ============================================================

def ols_regression(y: List[float], x: List[float]) -> dict:
    """
    Perform Ordinary Least Squares regression: Y = alpha + beta * X + epsilon
    
    Returns:
        alpha: Intercept
        beta: Slope (hedge ratio for cointegration)
        r_squared: Coefficient of determination
        residuals: Regression residuals
    """
    n = len(y)
    if n != len(x) or n < 3:
        return {"error": "Insufficient data", "beta": 0, "alpha": 0, "r_squared": 0, "residuals": []}
    
    # Calculate means
    x_mean = mean(x)
    y_mean = mean(y)
    
    # Calculate beta (slope)
    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    denominator = sum((xi - x_mean) ** 2 for xi in x)
    
    if denominator == 0:
        return {"error": "Zero variance in X", "beta": 0, "alpha": 0, "r_squared": 0, "residuals": []}
    
    beta = numerator / denominator
    alpha = y_mean - beta * x_mean
    
    # Calculate residuals
    residuals = [yi - (alpha + beta * xi) for yi, xi in zip(y, x)]
    
    # Calculate R-squared
    ss_res = sum(r ** 2 for r in residuals)
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        "alpha": alpha,
        "beta": beta,
        "r_squared": r_squared,
        "residuals": residuals,
        "n": n
    }


# ============================================================
# ADF TEST (Simplified Pure Python)
# ============================================================

def adf_test(series: List[float], max_lag: int = 1) -> dict:
    """
    Simplified Augmented Dickey-Fuller test for stationarity.
    
    Tests: Δy_t = αy_{t-1} + ε_t
    H0: Series has unit root (non-stationary)
    H1: Series is stationary
    
    Returns t-statistic and approximate p-value.
    
    Note: This is a simplified implementation. For production,
    use statsmodels or implement with proper critical values.
    """
    n = len(series)
    if n < 10:
        return {"error": "Need at least 10 data points", "statistic": 0, "p_value": 1.0}
    
    # Calculate first differences: Δy_t = y_t - y_{t-1}
    diff = [series[i] - series[i-1] for i in range(1, n)]
    
    # Lagged levels: y_{t-1}
    lagged = series[:-1]
    
    # Regress diff on lagged
    regression = ols_regression(diff, lagged)
    
    if "error" in regression:
        return {"error": regression["error"], "statistic": 0, "p_value": 1.0}
    
    # ADF t-statistic
    beta = regression["beta"]
    residuals = regression["residuals"]
    
    # Standard error of beta
    ss_res = sum(r ** 2 for r in residuals) / (len(residuals) - 2)
    ss_x = sum((x - mean(lagged)) ** 2 for x in lagged)
    
    if ss_x == 0:
        return {"error": "Zero variance", "statistic": 0, "p_value": 1.0}
    
    se_beta = math.sqrt(ss_res / ss_x)
    
    if se_beta == 0:
        return {"error": "Zero standard error", "statistic": 0, "p_value": 1.0}
    
    t_stat = beta / se_beta
    
    # Approximate p-value using MacKinnon critical values
    # Critical values (approximate): 1%: -3.43, 5%: -2.86, 10%: -2.57
    if t_stat < -3.43:
        p_value = 0.01
    elif t_stat < -2.86:
        p_value = 0.05
    elif t_stat < -2.57:
        p_value = 0.10
    elif t_stat < -1.94:
        p_value = 0.30
    else:
        p_value = 0.50 + min(0.5, abs(t_stat) * 0.1)
    
    return {
        "statistic": t_stat,
        "p_value": p_value,
        "critical_1pct": -3.43,
        "critical_5pct": -2.86,
        "critical_10pct": -2.57,
        "is_stationary": p_value < ADF_SIGNIFICANCE,
        "n_observations": n
    }


# ============================================================
# HALF-LIFE CALCULATION
# ============================================================

def calculate_half_life(spread: List[float]) -> float:
    """
    Calculate mean-reversion half-life using AR(1) process.
    
    Model: S_t = φ * S_{t-1} + ε
    Half-life: t_½ = -ln(2) / ln(φ)
    """
    if len(spread) < 10:
        return float('inf')
    
    # Lag the spread
    lagged = spread[:-1]
    current = spread[1:]
    
    # Regress current on lagged
    regression = ols_regression(current, lagged)
    
    phi = regression.get("beta", 1.0)
    
    # phi must be between 0 and 1 for mean reversion
    if phi <= 0 or phi >= 1:
        return float('inf')
    
    half_life = -math.log(2) / math.log(phi)
    
    return max(1, min(365, half_life))  # Clamp between 1-365 days


# ============================================================
# Z-SCORE CALCULATION
# ============================================================

def calculate_zscore(spread: List[float], lookback: int = 60) -> float:
    """
    Calculate Z-score of current spread value.
    
    Z = (S_current - μ) / σ
    """
    if len(spread) < 2:
        return 0.0
    
    recent = spread[-lookback:] if len(spread) >= lookback else spread
    
    mu = mean(recent)
    sigma = std_dev(recent)
    
    if sigma == 0:
        return 0.0
    
    current = spread[-1]
    return (current - mu) / sigma


# ============================================================
# COINTEGRATION ANALYSIS
# ============================================================

def analyze_cointegration(prices_a: List[float], prices_b: List[float]) -> dict:
    """
    Full Engle-Granger cointegration analysis.
    
    Pipeline:
    1. OLS regression to find hedge ratio (β)
    2. Calculate spread: S = P_A - β * P_B
    3. ADF test on spread for stationarity
    4. If stationary, calculate Z-score and half-life
    5. Generate trading signal
    """
    n = min(len(prices_a), len(prices_b))
    
    if n < 30:
        return {
            "cointegrated": False,
            "error": "Insufficient data (need 30+ periods)"
        }
    
    # Synchronize arrays
    pa = prices_a[-n:]
    pb = prices_b[-n:]
    
    # Step 1: OLS Regression
    ols = ols_regression(pa, pb)
    hedge_ratio = ols["beta"]
    r_squared = ols["r_squared"]
    
    # Step 2: Calculate Spread
    spread = [a - hedge_ratio * b for a, b in zip(pa, pb)]
    
    # Step 3: ADF Test
    adf = adf_test(spread)
    is_cointegrated = adf.get("is_stationary", False)
    
    # Step 4: Z-Score and Half-Life
    z_score = calculate_zscore(spread)
    half_life = calculate_half_life(spread) if is_cointegrated else float('inf')
    
    # Step 5: Generate Signal
    signal = generate_signal(z_score, is_cointegrated, adf["p_value"])
    
    # Calculate spread statistics
    spread_mean = mean(spread)
    spread_std = std_dev(spread)
    
    return {
        "cointegrated": is_cointegrated,
        "hedge_ratio": round(hedge_ratio, 6),
        "r_squared": round(r_squared, 4),
        "adf_statistic": round(adf.get("statistic", 0), 4),
        "adf_p_value": round(adf.get("p_value", 1), 4),
        "z_score": round(z_score, 4),
        "half_life_days": round(half_life, 2) if half_life != float('inf') else None,
        "spread_mean": round(spread_mean, 6),
        "spread_std": round(spread_std, 6),
        "correlation": round(correlation(pa, pb), 4),
        "signal": signal,
        "n_observations": n
    }


# ============================================================
# SIGNAL GENERATION
# ============================================================

def generate_signal(z_score: float, is_cointegrated: bool, p_value: float) -> dict:
    """
    Generate trading signal based on analysis.
    
    Signals:
    - LONG_SPREAD: Z < -2 (buy A, sell B)
    - SHORT_SPREAD: Z > +2 (sell A, buy B)
    - EXIT: |Z| < 0.5 (close position)
    - HOLD: Wait for better entry
    """
    if not is_cointegrated or p_value > ADF_SIGNIFICANCE:
        return {
            "action": "NO_TRADE",
            "reason": "Not cointegrated (p > 0.05)"
        }
    
    # Calculate confidence score (0-100)
    confidence = 0
    confidence += 30 if p_value < 0.05 else 0
    confidence += 20 if abs(z_score) > 2.5 else (10 if abs(z_score) > 2.0 else 0)
    confidence += 15 if p_value < 0.01 else (10 if p_value < 0.03 else 0)
    confidence += 15  # Base for cointegration
    
    if abs(z_score) > STOP_LOSS_THRESHOLD:
        return {
            "action": "STOP_LOSS",
            "reason": f"|Z| = {abs(z_score):.2f} > 4.0",
            "confidence": confidence
        }
    
    if abs(z_score) < EXIT_THRESHOLD:
        return {
            "action": "EXIT",
            "reason": f"|Z| = {abs(z_score):.2f} < 0.5 (mean reverted)",
            "confidence": confidence
        }
    
    if z_score < -ENTRY_THRESHOLD:
        return {
            "action": "LONG_SPREAD",
            "reason": f"Z = {z_score:.2f} < -2.0 (spread undervalued)",
            "confidence": min(100, confidence + 20),
            "timing": "NOW"
        }
    
    if z_score > ENTRY_THRESHOLD:
        return {
            "action": "SHORT_SPREAD",
            "reason": f"Z = {z_score:.2f} > +2.0 (spread overvalued)",
            "confidence": min(100, confidence + 20),
            "timing": "NOW"
        }
    
    return {
        "action": "HOLD",
        "reason": f"Z = {z_score:.2f} (waiting for entry threshold)",
        "confidence": confidence
    }


# ============================================================
# PAIR SCANNER
# ============================================================

def scan_pair(prices_a: List[float], prices_b: List[float], 
              pair_name: str, current_price_a: float, current_price_b: float) -> dict:
    """
    Complete analysis of a single pair.
    Returns actionable trading signal with all relevant metrics.
    """
    analysis = analyze_cointegration(prices_a, prices_b)
    
    if not analysis.get("cointegrated"):
        return {
            "pair": pair_name,
            "tradeable": False,
            "reason": analysis.get("error", "Not cointegrated")
        }
    
    signal = analysis["signal"]
    
    # Calculate potential P&L
    target_z = 0  # Mean reversion target
    current_z = analysis["z_score"]
    expected_move_pct = abs(current_z - target_z) * analysis["spread_std"] / current_price_a * 100
    
    return {
        "pair": pair_name,
        "tradeable": signal["action"] in ["LONG_SPREAD", "SHORT_SPREAD"],
        "action": signal["action"],
        "z_score": analysis["z_score"],
        "adf_p_value": analysis["adf_p_value"],
        "hedge_ratio": analysis["hedge_ratio"],
        "half_life_days": analysis["half_life_days"],
        "confidence": signal.get("confidence", 0),
        "reason": signal["reason"],
        "current_prices": {"a": current_price_a, "b": current_price_b},
        "expected_move_pct": round(expected_move_pct, 2),
        "timing": signal.get("timing", "WAIT")
    }
