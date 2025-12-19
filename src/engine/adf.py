"""
Antigravity Engine - Augmented Dickey-Fuller Test
Pure Python implementation for stationarity testing.
No external dependencies - WASM compatible for ICP canisters.

Mathematical Foundation:
The ADF test checks for unit root in a time series.
- Null hypothesis: Series has a unit root (non-stationary)
- Alternative: Series is stationary

Test equation: Δy_t = α + γ*y_{t-1} + Σ(δ_i*Δy_{t-i}) + ε_t

We reject null (series is stationary) if t-statistic < critical value.
For p < 0.05, critical value ≈ -2.86 (for n > 100)
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class ADFResult:
    """Result container for ADF test."""
    test_statistic: float
    p_value: float  # Approximate
    critical_values: dict  # {1%: x, 5%: y, 10%: z}
    is_stationary: bool
    n_observations: int
    lags_used: int


# Critical values for ADF test (MacKinnon, 1994)
# Format: n_samples -> {significance: critical_value}
ADF_CRITICAL_VALUES = {
    25: {"1%": -3.75, "5%": -3.00, "10%": -2.63},
    50: {"1%": -3.58, "5%": -2.93, "10%": -2.60},
    100: {"1%": -3.51, "5%": -2.89, "10%": -2.58},
    250: {"1%": -3.46, "5%": -2.88, "10%": -2.57},
    500: {"1%": -3.44, "5%": -2.87, "10%": -2.57},
    1000: {"1%": -3.43, "5%": -2.86, "10%": -2.57},
}


def get_critical_values(n: int) -> dict:
    """
    Get interpolated critical values for sample size n.
    """
    sizes = sorted(ADF_CRITICAL_VALUES.keys())
    
    if n <= sizes[0]:
        return ADF_CRITICAL_VALUES[sizes[0]]
    if n >= sizes[-1]:
        return ADF_CRITICAL_VALUES[sizes[-1]]
    
    # Find surrounding sizes and interpolate
    for i in range(len(sizes) - 1):
        if sizes[i] <= n <= sizes[i + 1]:
            lower, upper = sizes[i], sizes[i + 1]
            ratio = (n - lower) / (upper - lower)
            
            result = {}
            for key in ["1%", "5%", "10%"]:
                val_low = ADF_CRITICAL_VALUES[lower][key]
                val_high = ADF_CRITICAL_VALUES[upper][key]
                result[key] = val_low + ratio * (val_high - val_low)
            return result
    
    return ADF_CRITICAL_VALUES[sizes[-1]]


def approximate_p_value(t_stat: float, n: int) -> float:
    """
    Approximate p-value from t-statistic using critical value interpolation.
    This is a simplified approximation.
    """
    cv = get_critical_values(n)
    
    if t_stat < cv["1%"]:
        return 0.005  # Very significant
    elif t_stat < cv["5%"]:
        # Interpolate between 1% and 5%
        ratio = (t_stat - cv["1%"]) / (cv["5%"] - cv["1%"])
        return 0.01 + ratio * 0.04
    elif t_stat < cv["10%"]:
        # Interpolate between 5% and 10%
        ratio = (t_stat - cv["5%"]) / (cv["10%"] - cv["5%"])
        return 0.05 + ratio * 0.05
    else:
        # Not significant
        # Rough approximation for large p-values
        return min(0.5, 0.10 + (t_stat - cv["10%"]) * 0.1)


def ols_regression(y: List[float], X: List[List[float]]) -> Tuple[List[float], float]:
    """
    Simple OLS regression: y = Xβ + ε
    Returns (coefficients, standard_error_of_first_coef)
    
    Uses normal equations: β = (X'X)^(-1) X'y
    """
    n = len(y)
    k = len(X[0]) if X else 0
    
    if n == 0 or k == 0:
        return [], 0.0
    
    # Compute X'X
    XtX = [[0.0] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            XtX[i][j] = sum(X[t][i] * X[t][j] for t in range(n))
    
    # Compute X'y
    Xty = [sum(X[t][i] * y[t] for t in range(n)) for i in range(k)]
    
    # Solve via Gaussian elimination
    # Augment XtX with Xty
    aug = [XtX[i] + [Xty[i]] for i in range(k)]
    
    # Forward elimination
    for i in range(k):
        # Find pivot
        max_row = i
        for r in range(i + 1, k):
            if abs(aug[r][i]) > abs(aug[max_row][i]):
                max_row = r
        aug[i], aug[max_row] = aug[max_row], aug[i]
        
        if abs(aug[i][i]) < 1e-10:
            continue  # Singular
            
        for r in range(i + 1, k):
            factor = aug[r][i] / aug[i][i]
            for c in range(i, k + 1):
                aug[r][c] -= factor * aug[i][c]
    
    # Back substitution
    beta = [0.0] * k
    for i in range(k - 1, -1, -1):
        if abs(aug[i][i]) < 1e-10:
            continue
        beta[i] = aug[i][k]
        for j in range(i + 1, k):
            beta[i] -= aug[i][j] * beta[j]
        beta[i] /= aug[i][i]
    
    # Calculate residuals and standard error
    residuals = [y[t] - sum(X[t][i] * beta[i] for i in range(k)) for t in range(n)]
    sse = sum(r * r for r in residuals)
    mse = sse / max(n - k, 1)
    
    # Standard error of first coefficient (γ in ADF)
    # SE(β_0) = sqrt(MSE * (X'X)^(-1)[0,0])
    # Approximate by inverting XtX
    try:
        # Simple 2x2 or use first diagonal element
        if k >= 1 and abs(XtX[0][0]) > 1e-10:
            se_gamma = math.sqrt(mse / XtX[0][0])
        else:
            se_gamma = 1.0
    except:
        se_gamma = 1.0
    
    return beta, se_gamma


def adf_test(series: List[float], max_lags: int = 1) -> ADFResult:
    """
    Perform Augmented Dickey-Fuller test for stationarity.
    
    Args:
        series: Time series data (list of floats)
        max_lags: Number of lagged differences to include
        
    Returns:
        ADFResult with test statistic, p-value, and is_stationary flag
    """
    n = len(series)
    
    if n < 10:
        return ADFResult(
            test_statistic=0.0,
            p_value=1.0,
            critical_values={"1%": -3.5, "5%": -2.9, "10%": -2.6},
            is_stationary=False,
            n_observations=n,
            lags_used=0
        )
    
    # Calculate first differences
    diff = [series[i] - series[i-1] for i in range(1, n)]
    
    # Determine effective lags (ensure enough observations)
    effective_lags = min(max_lags, len(diff) // 4)
    
    # Build regression: Δy_t = α + γ*y_{t-1} + Σ(δ_i*Δy_{t-i}) + ε_t
    # Start index to ensure we have lag values
    start = effective_lags + 1
    
    y_reg = []  # Δy_t
    X_reg = []  # [y_{t-1}, Δy_{t-1}, ..., Δy_{t-lag}, 1(constant)]
    
    for t in range(start, len(diff)):
        y_reg.append(diff[t])
        
        row = [series[t]]  # y_{t-1} (level, not diff)
        
        # Add lagged differences
        for lag in range(1, effective_lags + 1):
            row.append(diff[t - lag])
        
        row.append(1.0)  # Constant term
        X_reg.append(row)
    
    if len(y_reg) < 5:
        return ADFResult(
            test_statistic=0.0,
            p_value=1.0,
            critical_values=get_critical_values(n),
            is_stationary=False,
            n_observations=n,
            lags_used=effective_lags
        )
    
    # Run regression
    beta, se_gamma = ols_regression(y_reg, X_reg)
    
    if not beta or se_gamma == 0:
        return ADFResult(
            test_statistic=0.0,
            p_value=1.0,
            critical_values=get_critical_values(n),
            is_stationary=False,
            n_observations=n,
            lags_used=effective_lags
        )
    
    # gamma is the coefficient on y_{t-1}
    gamma = beta[0]
    
    # Calculate t-statistic
    t_stat = gamma / se_gamma if se_gamma != 0 else 0.0
    
    # Get critical values and p-value
    critical_values = get_critical_values(n)
    p_value = approximate_p_value(t_stat, n)
    
    # Reject null (stationary) if t_stat < critical value at 5%
    is_stationary = t_stat < critical_values["5%"]
    
    return ADFResult(
        test_statistic=t_stat,
        p_value=p_value,
        critical_values=critical_values,
        is_stationary=is_stationary,
        n_observations=n,
        lags_used=effective_lags
    )


def test_adf():
    """Test ADF implementation with known series."""
    import random
    random.seed(42)
    
    # Test 1: Random walk (non-stationary)
    rw = [0.0]
    for _ in range(199):
        rw.append(rw[-1] + random.gauss(0, 1))
    
    result_rw = adf_test(rw)
    print("Random Walk (should be non-stationary):")
    print(f"  t-stat: {result_rw.test_statistic:.4f}")
    print(f"  p-value: {result_rw.p_value:.4f}")
    print(f"  Stationary: {result_rw.is_stationary}")
    
    # Test 2: Stationary series (mean-reverting)
    stationary = []
    val = 0.0
    for _ in range(200):
        val = 0.5 * val + random.gauss(0, 1)  # AR(1) with phi < 1
        stationary.append(val)
    
    result_stat = adf_test(stationary)
    print("\nAR(1) Process (should be stationary):")
    print(f"  t-stat: {result_stat.test_statistic:.4f}")
    print(f"  p-value: {result_stat.p_value:.4f}")
    print(f"  Stationary: {result_stat.is_stationary}")
    
    return result_rw, result_stat


if __name__ == "__main__":
    test_adf()
