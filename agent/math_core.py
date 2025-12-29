"""
Antigravity Math Core
Shared statistical functions for cointegration analysis

This module contains all mathematical functions used by both
trading_agent.py and backtest.py to eliminate code duplication.
"""

import math

# ============================================================
# CONSTANTS
# ============================================================

EPSILON = 1e-10
ADF_CRITICAL_5PCT = -2.86


# ============================================================
# BASIC STATISTICS
# ============================================================

def mean(data: list) -> float:
    """Calculate arithmetic mean of a list of numbers."""
    if not data:
        return 0.0
    return sum(data) / len(data)


def variance(data: list) -> float:
    """Calculate sample variance (n-1 denominator)."""
    n = len(data)
    if n < 2:
        return 0.0
    mu = mean(data)
    return sum((x - mu) ** 2 for x in data) / (n - 1)


def std_dev(data: list) -> float:
    """Calculate sample standard deviation."""
    return math.sqrt(variance(data))


def covariance(x: list, y: list) -> float:
    """Calculate sample covariance between two lists."""
    n = len(x)
    if n != len(y) or n < 2:
        return 0.0
    mu_x, mu_y = mean(x), mean(y)
    return sum((xi - mu_x) * (yi - mu_y) for xi, yi in zip(x, y)) / (n - 1)


def correlation(x: list, y: list) -> float:
    """Calculate Pearson correlation coefficient."""
    sx, sy = std_dev(x), std_dev(y)
    if sx < EPSILON or sy < EPSILON:
        return 0.0
    return covariance(x, y) / (sx * sy)


# ============================================================
# OLS REGRESSION
# ============================================================

class OLSResult:
    """Result of OLS regression: Y = alpha + beta * X"""
    
    def __init__(self, beta: float, alpha: float, r_squared: float, residuals: list):
        self.beta = beta
        self.alpha = alpha
        self.r_squared = r_squared
        self.residuals = residuals
    
    def __repr__(self):
        return f"OLSResult(beta={self.beta:.4f}, alpha={self.alpha:.4f}, r_squared={self.r_squared:.4f})"


def ols_regression(y: list, x: list) -> OLSResult:
    """
    Ordinary Least Squares Regression: Y = alpha + beta * X
    
    Args:
        y: Dependent variable (list of floats)
        x: Independent variable (list of floats)
    
    Returns:
        OLSResult with beta, alpha, r_squared, and residuals
    """
    n = len(y)
    if n != len(x) or n < 2:
        return OLSResult(0, 0, 0, [])
    
    x_mean, y_mean = mean(x), mean(y)
    
    # Calculate beta (slope)
    num = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
    den = sum((x[i] - x_mean) ** 2 for i in range(n))
    
    if den < EPSILON:
        return OLSResult(0, 0, 0, [])
    
    beta = num / den
    alpha = y_mean - beta * x_mean
    
    # Calculate residuals and R-squared
    residuals = [y[i] - (alpha + beta * x[i]) for i in range(n)]
    ss_res = sum(r ** 2 for r in residuals)
    ss_tot = sum((y[i] - y_mean) ** 2 for i in range(n))
    r_sq = 1 - (ss_res / ss_tot) if ss_tot > EPSILON else 0
    
    return OLSResult(beta, alpha, r_sq, residuals)


# ============================================================
# STATIONARITY TESTS
# ============================================================

def adf_test(series: list) -> dict:
    """
    Augmented Dickey-Fuller Test for stationarity.
    
    Tests if a time series has a unit root (is non-stationary).
    
    Args:
        series: Time series data (list of floats)
    
    Returns:
        dict with t_stat, p_value, and is_stationary
    """
    n = len(series)
    if n < 10:
        return {"t_stat": 0, "p_value": 1.0, "is_stationary": False}
    
    # Calculate first differences
    dy = [series[i] - series[i - 1] for i in range(1, n)]
    x_lag = [series[i - 1] for i in range(1, n)]
    
    # Regress differences on lagged values
    res = ols_regression(dy, x_lag)
    gamma = res.beta
    
    # Calculate t-statistic
    sum_res_sq = sum(r * r for r in res.residuals)
    mean_x = mean(x_lag)
    sum_sq_x = sum((v - mean_x) ** 2 for v in x_lag)
    
    if sum_sq_x < EPSILON:
        return {"t_stat": 0, "p_value": 1.0, "is_stationary": False}
    
    sigma_sq = sum_res_sq / max(len(x_lag) - 2, 1)
    se = math.sqrt(sigma_sq / sum_sq_x) if sigma_sq > 0 else 0
    t_stat = gamma / se if se > EPSILON else 0
    
    # Approximate p-value based on critical values
    if t_stat < -3.43:
        p_value = 0.01
    elif t_stat < -2.86:
        p_value = 0.05
    elif t_stat < -2.57:
        p_value = 0.10
    else:
        p_value = 1.0
    
    return {
        "t_stat": t_stat,
        "p_value": p_value,
        "is_stationary": p_value < 0.05
    }


# ============================================================
# Z-SCORE AND MEAN REVERSION
# ============================================================

def z_score(current: float, history: list, window: int = 30) -> float:
    """
    Calculate Z-Score of current value relative to recent history.
    
    Args:
        current: Current value
        history: Historical values
        window: Lookback window size
    
    Returns:
        Z-score (number of standard deviations from mean)
    """
    if len(history) < 2:
        return 0
    
    data = history[-window:]
    mu, sigma = mean(data), std_dev(data)
    
    return (current - mu) / sigma if sigma > EPSILON else 0


def calculate_half_life(spread: list) -> float:
    """
    Calculate half-life of mean reversion.
    
    The half-life indicates how many periods it takes for the spread
    to revert halfway to its mean.
    
    Args:
        spread: Spread time series
    
    Returns:
        Half-life in periods (capped at 999 if no mean reversion)
    """
    if len(spread) < 10:
        return 999
    
    # Regress spread(t) - spread(t-1) on spread(t-1)
    dy = [spread[i] - spread[i - 1] for i in range(1, len(spread))]
    y_lag = spread[:-1]
    
    res = ols_regression(dy, y_lag)
    
    # If beta >= 0, no mean reversion
    if res.beta >= 0 or abs(res.beta) < EPSILON:
        return 999
    
    half_life = -math.log(2) / res.beta
    return max(1, min(half_life, 999))


def calculate_hurst_exponent(series: list, max_lag: int = 20) -> float:
    """
    Calculate Hurst Exponent.
    
    H < 0.5: Mean-reverting (anti-persistent)
    H = 0.5: Random walk
    H > 0.5: Trending (persistent)
    
    Args:
        series: Time series data
        max_lag: Maximum lag to consider
    
    Returns:
        Hurst exponent (0 to 1)
    """
    n = len(series)
    if n < max_lag * 2:
        return 0.5
    
    lags = range(2, min(max_lag, n // 4))
    tau = []
    rs = []
    
    for lag in lags:
        # Calculate variance of lagged differences
        diffs = [series[i] - series[i - lag] for i in range(lag, n)]
        if len(diffs) < 2:
            continue
        tau.append(lag)
        rs.append(std_dev(diffs))
    
    if len(tau) < 3:
        return 0.5
    
    # Regression: log(R/S) = H * log(tau)
    log_tau = [math.log(t) for t in tau]
    log_rs = [math.log(r) if r > EPSILON else 0 for r in rs]
    
    res = ols_regression(log_rs, log_tau)
    return max(0, min(res.beta, 1))


# ============================================================
# SPREAD STATISTICS
# ============================================================

def calculate_spread_stats(spread: list) -> dict:
    """
    Calculate comprehensive spread statistics.
    
    Args:
        spread: Spread time series
    
    Returns:
        dict with mean, std, current, bands, percentile, volatility
    """
    if len(spread) < 30:
        return {}
    
    mu = mean(spread)
    sigma = std_dev(spread)
    current = spread[-1]
    
    # Bollinger Bands (2 std dev)
    upper_band = mu + 2 * sigma
    lower_band = mu - 2 * sigma
    
    # Spread percentile
    sorted_spread = sorted(spread)
    percentile = sum(1 for s in sorted_spread if s <= current) / len(spread) * 100
    
    # Volatility (rolling 30-day)
    recent_vol = std_dev(spread[-30:]) if len(spread) >= 30 else sigma
    
    # Max drawdown from mean
    max_deviation = max(abs(s - mu) for s in spread)
    
    return {
        "mean": round(mu, 4),
        "std": round(sigma, 4),
        "current": round(current, 4),
        "upper_band": round(upper_band, 4),
        "lower_band": round(lower_band, 4),
        "percentile": round(percentile, 1),
        "volatility_30d": round(recent_vol, 4),
        "max_deviation": round(max_deviation, 4)
    }


# ============================================================
# RISK SCORING
# ============================================================

def calculate_risk_score(z_score_val: float, adf_p_value: float, r_squared: float) -> float:
    """
    Calculate risk percentage (0-100) based on statistical metrics.
    
    Lower score = lower risk = better trade.
    
    Args:
        z_score_val: Current Z-score
        adf_p_value: ADF test p-value
        r_squared: R-squared from regression
    
    Returns:
        Risk percentage (0-100)
    """
    # Lower z-score = lower risk (closer to mean)
    z_risk = min(abs(z_score_val) * 20, 50)  # 0-50 from z-score
    
    # Lower p-value = lower risk (more stationary)
    p_risk = adf_p_value * 30  # 0-30 from p-value
    
    # Higher R² = lower risk (better fit)
    r_risk = (1 - r_squared) * 20  # 0-20 from R²
    
    return round(z_risk + p_risk + r_risk, 1)


def get_sandwich_signal(spread_stats: dict, z: float) -> dict:
    """
    Determine sandwich trading strategy based on spread position.
    
    Args:
        spread_stats: Output from calculate_spread_stats
        z: Current Z-score
    
    Returns:
        dict with strategy, entry, confidence, target or None
    """
    if not spread_stats:
        return None
    
    pct = spread_stats.get("percentile", 50)
    
    # Sandwich strategy: trade when spread is at extremes
    if pct < 10:  # Bottom 10% - spread very low
        return {
            "strategy": "SANDWICH_LONG",
            "entry": "BUY first asset (spread will expand)",
            "confidence": "HIGH" if pct < 5 else "MEDIUM",
            "target": "Mean reversion to 50th percentile"
        }
    elif pct > 90:  # Top 10% - spread very high
        return {
            "strategy": "SANDWICH_SHORT",
            "entry": "SELL first asset (spread will contract)",
            "confidence": "HIGH" if pct > 95 else "MEDIUM",
            "target": "Mean reversion to 50th percentile"
        }
    elif 45 <= pct <= 55:  # Near mean - wait
        return {
            "strategy": "NEUTRAL",
            "entry": "No trade - spread near equilibrium",
            "confidence": "LOW",
            "target": "Wait for deviation"
        }
    
    return None
