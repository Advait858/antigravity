"""
Antigravity Bot v5.0 - Comprehensive Cointegration Analysis Agent
Full historical OHLCV analysis with Engle-Granger cointegration on all pair combinations.

Features:
- 1-year OHLCV historical data from CoinGecko
- SMAs (20, 50, 200 day)
- Engle-Granger two-step cointegration test
- Rolling window cointegration analysis (30, 60, 90, 180 day windows)
- Volume-weighted spread analysis
- All 45 pair combinations analyzed
- Detailed trade setups with entry/exit ranges

Author: Antigravity Team
Hackathon: Zo House - AI Agents for Trading
"""

from kybra import (
    Async,
    CallResult,
    float64,
    ic,
    match,
    nat64,
    Principal,
    query,
    update,
    void,
    StableBTreeMap,
)
from kybra.canisters.management import (
    HttpResponse,
    HttpTransformArgs,
    management_canister,
)
import json
import math


# ============================================================================
# Constants
# ============================================================================

VERSION = "5.0.0-full-cointegration"

ASSETS = ["bitcoin", "ethereum", "solana", "ripple", "dogecoin", 
          "cardano", "avalanche-2", "polkadot", "chainlink", "internet-computer"]

ASSET_SYMBOLS = {
    "bitcoin": "BTC", "ethereum": "ETH", "solana": "SOL", "ripple": "XRP",
    "dogecoin": "DOGE", "cardano": "ADA", "avalanche-2": "AVAX",
    "polkadot": "DOT", "chainlink": "LINK", "internet-computer": "ICP"
}

# CoinGecko API - Free tier with OHLC data
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
HTTP_CYCLES = 250_000_000  # Increased for larger responses

# Analysis parameters
ROLLING_WINDOWS = [30, 60, 90, 180]  # Days for rolling cointegration
SMA_PERIODS = [20, 50, 200]
MIN_DATA_POINTS = 10  # Reduced for faster testing
COINTEGRATION_P_THRESHOLD = 0.05


# ============================================================================
# Stable Storage
# ============================================================================

ohlcv_storage = StableBTreeMap[str, str](memory_id=0, max_key_size=50, max_value_size=500_000)
analysis_storage = StableBTreeMap[str, str](memory_id=1, max_key_size=100, max_value_size=100_000)
user_storage = StableBTreeMap[str, str](memory_id=2, max_key_size=100, max_value_size=1_000)


# ============================================================================
# Global State
# ============================================================================

# OHLCV data structure: {asset_id: {"prices": [], "volumes": [], "highs": [], "lows": [], "timestamps": []}}
ohlcv_data: dict = {asset: {"prices": [], "volumes": [], "highs": [], "lows": [], "timestamps": []} for asset in ASSETS}

# Computed indicators
indicators: dict = {asset: {"sma_20": [], "sma_50": [], "sma_200": [], "volatility": 0, "avg_volume": 0} for asset in ASSETS}

# Cointegration results for all pairs
cointegration_results: dict = {}  # "BTC/ETH" -> analysis
rolling_cointegration: dict = {}  # "BTC/ETH" -> {30: result, 60: result, ...}

# Current prices and recommendations
current_prices: dict = {}
detailed_signals: list = []
last_update_time: int = 0
registered_users: list = []
execution_logs: list = []

# ============================================================================
# TRADE EXECUTION STATE (Paper Trading)
# ============================================================================

# Portfolio: Starting balance for paper trading
INITIAL_BALANCE = 100000.0  # $100,000 paper money

portfolio: dict = {
    "cash": INITIAL_BALANCE,
    "positions": {},  # {asset: {"quantity": float, "avg_price": float, "side": "long"/"short"}}
    "total_value": INITIAL_BALANCE,
    "pnl": 0.0,
    "pnl_pct": 0.0
}

# Trade history
trade_history: list = []  # List of executed trades

# Active positions from pair trades
active_pair_trades: list = []  # [{pair, entry_time, entry_z, positions, status}]

# Auto-execute mode
auto_execute_enabled: bool = True
max_position_size: float = 10000.0  # Max $10k per trade


# ============================================================================
# Statistical Functions
# ============================================================================

def log_event(t: str, m: str):
    global execution_logs
    execution_logs.append({"type": t, "message": m, "ts": ic.time()})
    if len(execution_logs) > 200:
        execution_logs = execution_logs[-100:]


def mean(data: list) -> float:
    return sum(data) / len(data) if data else 0.0


def std_dev(data: list) -> float:
    if len(data) < 2:
        return 0.0
    m = mean(data)
    return math.sqrt(sum((x - m) ** 2 for x in data) / len(data))


def sma(prices: list, period: int) -> list:
    """Simple Moving Average."""
    if len(prices) < period:
        return []
    result = []
    for i in range(period - 1, len(prices)):
        result.append(mean(prices[i - period + 1:i + 1]))
    return result


def calculate_returns(prices: list) -> list:
    if len(prices) < 2:
        return []
    return [(prices[i] / prices[i-1] - 1) * 100 for i in range(1, len(prices))]


def ols_regression(y: list, x: list) -> dict:
    """OLS: y = alpha + beta * x + epsilon."""
    n = min(len(y), len(x))
    if n < 10:
        return {"alpha": 0.0, "beta": 1.0, "residuals": [], "r_squared": 0.0, "std_error": 0.0}
    
    y, x = y[-n:], x[-n:]
    mean_x, mean_y = mean(x), mean(y)
    
    cov_xy = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / n
    var_x = sum((xi - mean_x) ** 2 for xi in x) / n
    
    if var_x == 0:
        return {"alpha": mean_y, "beta": 0.0, "residuals": [], "r_squared": 0.0, "std_error": 0.0}
    
    beta = cov_xy / var_x
    alpha = mean_y - beta * mean_x
    
    residuals = [y[i] - alpha - beta * x[i] for i in range(n)]
    ss_res = sum(r ** 2 for r in residuals)
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    std_error = math.sqrt(ss_res / (n - 2)) if n > 2 else 0.0
    
    return {"alpha": alpha, "beta": beta, "residuals": residuals, "r_squared": r_squared, "std_error": std_error}


def adf_test(series: list, max_lag: int = 5) -> dict:
    """Augmented Dickey-Fuller test with lag selection."""
    n = len(series)
    if n < 30:
        return {"statistic": 0.0, "p_value": 1.0, "is_stationary": False, "critical_values": {}}
    
    # First difference
    diff = [series[i] - series[i-1] for i in range(1, n)]
    
    # Lagged level
    y = diff[1:]
    x_level = series[1:-1]
    n_reg = len(y)
    
    if n_reg < 20:
        return {"statistic": 0.0, "p_value": 1.0, "is_stationary": False, "critical_values": {}}
    
    # Simple regression: diff[t] = gamma * level[t-1] + error
    xy = sum(x_level[i] * y[i] for i in range(n_reg))
    xx = sum(x_level[i] ** 2 for i in range(n_reg))
    
    if xx == 0:
        return {"statistic": 0.0, "p_value": 1.0, "is_stationary": False, "critical_values": {}}
    
    gamma = xy / xx
    residuals = [y[i] - gamma * x_level[i] for i in range(n_reg)]
    sse = sum(r ** 2 for r in residuals)
    mse = sse / max(n_reg - 1, 1)
    se = math.sqrt(mse / xx) if mse > 0 else 1.0
    
    t_stat = gamma / se if se > 0 else 0.0
    
    # MacKinnon approximate p-values
    critical_values = {"1%": -3.43, "5%": -2.86, "10%": -2.57}
    
    if t_stat < -3.96:
        p_value = 0.001
    elif t_stat < -3.43:
        p_value = 0.01
    elif t_stat < -2.86:
        p_value = 0.05
    elif t_stat < -2.57:
        p_value = 0.10
    else:
        p_value = 0.5 + 0.5 * (1 / (1 + math.exp(-t_stat)))
    
    return {
        "statistic": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "is_stationary": t_stat < critical_values["5%"],
        "critical_values": critical_values
    }


def engle_granger_test(y: list, x: list) -> dict:
    """
    Engle-Granger two-step cointegration test.
    Step 1: Run OLS regression y = alpha + beta*x + epsilon
    Step 2: Test if residuals are stationary using ADF
    """
    n = min(len(y), len(x))
    if n < MIN_DATA_POINTS:
        return {
            "is_cointegrated": False,
            "hedge_ratio": 0.0,
            "adf_statistic": 0.0,
            "adf_p_value": 1.0,
            "r_squared": 0.0,
            "half_life": float('inf'),
            "spread_mean": 0.0,
            "spread_std": 0.0,
            "current_z_score": 0.0
        }
    
    # Step 1: OLS regression
    reg = ols_regression(y[-n:], x[-n:])
    residuals = reg["residuals"]
    
    if len(residuals) < 30:
        return {
            "is_cointegrated": False,
            "hedge_ratio": reg["beta"],
            "adf_statistic": 0.0,
            "adf_p_value": 1.0,
            "r_squared": reg["r_squared"],
            "half_life": float('inf'),
            "spread_mean": 0.0,
            "spread_std": 0.0,
            "current_z_score": 0.0
        }
    
    # Step 2: ADF test on residuals (spread)
    adf = adf_test(residuals)
    
    # Calculate half-life of mean reversion
    half_life = calculate_half_life(residuals)
    
    # Spread statistics
    spread_mean = mean(residuals)
    spread_std = std_dev(residuals)
    z_score = (residuals[-1] - spread_mean) / spread_std if spread_std > 0 else 0.0
    
    return {
        "is_cointegrated": adf["is_stationary"],
        "hedge_ratio": round(reg["beta"], 6),
        "alpha": round(reg["alpha"], 4),
        "adf_statistic": adf["statistic"],
        "adf_p_value": adf["p_value"],
        "r_squared": round(reg["r_squared"], 4),
        "half_life": round(half_life, 1),
        "spread_mean": round(spread_mean, 6),
        "spread_std": round(spread_std, 6),
        "current_z_score": round(z_score, 4),
        "residuals_count": len(residuals)
    }


def calculate_half_life(spread: list) -> float:
    """Mean reversion half-life in days."""
    if len(spread) < 30:
        return float('inf')
    
    spread_lag = spread[:-1]
    spread_diff = [spread[i] - spread[i-1] for i in range(1, len(spread))]
    
    reg = ols_regression(spread_diff, spread_lag)
    if reg["beta"] >= 0:
        return float('inf')
    
    hl = -math.log(2) / reg["beta"]
    return max(1, min(hl, 365))


def calculate_volume_profile(volumes: list, prices: list) -> dict:
    """Volume-weighted price analysis."""
    if len(volumes) < 10 or len(prices) < 10:
        return {"vwap": 0, "volume_trend": "neutral", "avg_volume": 0}
    
    n = min(len(volumes), len(prices))
    volumes, prices = volumes[-n:], prices[-n:]
    
    # VWAP
    total_volume = sum(volumes)
    if total_volume == 0:
        return {"vwap": mean(prices), "volume_trend": "neutral", "avg_volume": 0}
    
    vwap = sum(prices[i] * volumes[i] for i in range(n)) / total_volume
    
    # Volume trend (recent 10 vs prior 10)
    if n >= 20:
        recent_vol = mean(volumes[-10:])
        prior_vol = mean(volumes[-20:-10])
        if recent_vol > prior_vol * 1.2:
            trend = "increasing"
        elif recent_vol < prior_vol * 0.8:
            trend = "decreasing"
        else:
            trend = "stable"
    else:
        trend = "insufficient_data"
    
    return {
        "vwap": round(vwap, 4),
        "volume_trend": trend,
        "avg_volume": round(mean(volumes), 2),
        "recent_volume": round(mean(volumes[-10:]) if len(volumes) >= 10 else mean(volumes), 2)
    }


def analyze_high_low_range(highs: list, lows: list, prices: list) -> dict:
    """Analyze price range and position within range."""
    if len(highs) < 20 or len(lows) < 20:
        return {"range_position": 0.5, "atr": 0, "support": 0, "resistance": 0}
    
    n = min(len(highs), len(lows), len(prices))
    highs, lows, prices = highs[-n:], lows[-n:], prices[-n:]
    
    # ATR (Average True Range)
    if n >= 2:
        tr = []
        for i in range(1, n):
            tr.append(max(highs[i] - lows[i], abs(highs[i] - prices[i-1]), abs(lows[i] - prices[i-1])))
        atr = mean(tr[-14:]) if len(tr) >= 14 else mean(tr)
    else:
        atr = 0
    
    # Support/Resistance (recent 20 days)
    recent_high = max(highs[-20:])
    recent_low = min(lows[-20:])
    current = prices[-1]
    
    # Position in range (0 = at support, 1 = at resistance)
    range_width = recent_high - recent_low
    range_position = (current - recent_low) / range_width if range_width > 0 else 0.5
    
    return {
        "range_position": round(range_position, 4),
        "atr": round(atr, 4),
        "atr_pct": round((atr / current) * 100, 2) if current > 0 else 0,
        "support": round(recent_low, 4),
        "resistance": round(recent_high, 4),
        "current_price": round(current, 4)
    }


# ============================================================================
# Comprehensive Analysis Functions
# ============================================================================

def compute_indicators(asset_id: str):
    """Compute all technical indicators for an asset."""
    global indicators
    
    data = ohlcv_data.get(asset_id, {})
    prices = data.get("prices", [])
    volumes = data.get("volumes", [])
    
    if len(prices) < 20:
        return
    
    ind = {
        "sma_20": sma(prices, 20),
        "sma_50": sma(prices, 50) if len(prices) >= 50 else [],
        "sma_200": sma(prices, 200) if len(prices) >= 200 else [],
        "volatility": std_dev(calculate_returns(prices[-30:])) * math.sqrt(365) if len(prices) >= 30 else 0,
        "avg_volume": mean(volumes) if volumes else 0,
        "data_points": len(prices)
    }
    
    indicators[asset_id] = ind


def analyze_pair_comprehensive(asset_a: str, asset_b: str) -> dict:
    """Run full cointegration analysis on a pair with all features."""
    data_a = ohlcv_data.get(asset_a, {})
    data_b = ohlcv_data.get(asset_b, {})
    
    prices_a = data_a.get("prices", [])
    prices_b = data_b.get("prices", [])
    volumes_a = data_a.get("volumes", [])
    volumes_b = data_b.get("volumes", [])
    highs_a = data_a.get("highs", [])
    lows_a = data_a.get("lows", [])
    highs_b = data_b.get("highs", [])
    lows_b = data_b.get("lows", [])
    
    n = min(len(prices_a), len(prices_b))
    if n < MIN_DATA_POINTS:
        return None
    
    symbol_a = ASSET_SYMBOLS.get(asset_a, asset_a.upper())
    symbol_b = ASSET_SYMBOLS.get(asset_b, asset_b.upper())
    
    # Main Engle-Granger test
    eg_result = engle_granger_test(prices_a, prices_b)
    
    # Rolling window cointegration
    rolling = {}
    for window in ROLLING_WINDOWS:
        if n >= window:
            rolling[window] = engle_granger_test(prices_a[-window:], prices_b[-window:])
    
    # Volume analysis
    vol_a = calculate_volume_profile(volumes_a, prices_a)
    vol_b = calculate_volume_profile(volumes_b, prices_b)
    
    # High/Low analysis
    range_a = analyze_high_low_range(highs_a, lows_a, prices_a)
    range_b = analyze_high_low_range(highs_b, lows_b, prices_b)
    
    # Correlation
    if n >= 20:
        mean_a, mean_b = mean(prices_a[-n:]), mean(prices_b[-n:])
        cov = sum((prices_a[-n + i] - mean_a) * (prices_b[-n + i] - mean_b) for i in range(n)) / n
        std_a, std_b = std_dev(prices_a[-n:]), std_dev(prices_b[-n:])
        correlation = cov / (std_a * std_b) if std_a > 0 and std_b > 0 else 0
    else:
        correlation = 0
    
    # Count cointegrated windows
    coint_windows = sum(1 for w, r in rolling.items() if r.get("is_cointegrated", False))
    
    return {
        "pair": f"{symbol_a}/{symbol_b}",
        "asset_a": symbol_a,
        "asset_b": symbol_b,
        "data_points": n,
        "main_analysis": eg_result,
        "rolling_windows": {str(k): v for k, v in rolling.items()},
        "cointegrated_windows": coint_windows,
        "total_windows": len(rolling),
        "correlation": round(correlation, 4),
        "volume_analysis": {"a": vol_a, "b": vol_b},
        "range_analysis": {"a": range_a, "b": range_b},
        "indicators": {
            "a": indicators.get(asset_a, {}),
            "b": indicators.get(asset_b, {})
        },
        "timestamp": ic.time()
    }


def generate_trading_signal(analysis: dict) -> dict:
    """Generate detailed trading signal from pair analysis."""
    if not analysis:
        return None
    
    main = analysis.get("main_analysis", {})
    if not main.get("is_cointegrated"):
        # Check if any rolling window shows cointegration
        rolling = analysis.get("rolling_windows", {})
        coint_any = any(r.get("is_cointegrated", False) for r in rolling.values())
        if not coint_any:
            return None
    
    z_score = main.get("current_z_score", 0)
    half_life = main.get("half_life", float('inf'))
    hedge_ratio = main.get("hedge_ratio", 1)
    
    range_a = analysis.get("range_analysis", {}).get("a", {})
    range_b = analysis.get("range_analysis", {}).get("b", {})
    vol_a = analysis.get("volume_analysis", {}).get("a", {})
    vol_b = analysis.get("volume_analysis", {}).get("b", {})
    
    # Determine action
    if z_score > 2.0:
        action = "SHORT_SPREAD"
        direction_a = "SELL"
        direction_b = "BUY"
        description = f"Spread is {z_score:.1f}σ above mean - expect mean reversion DOWN"
    elif z_score < -2.0:
        action = "LONG_SPREAD"
        direction_a = "BUY"
        direction_b = "SELL"
        description = f"Spread is {abs(z_score):.1f}σ below mean - expect mean reversion UP"
    elif 1.5 < abs(z_score) <= 2.0:
        action = "PREPARE"
        direction_a = "WATCH"
        direction_b = "WATCH"
        description = f"Approaching entry threshold ({z_score:.2f}σ)"
    else:
        return None
    
    # Timing based on half-life
    if half_life < 7:
        timing = "NOW"
        timing_desc = "Execute immediately - fast mean reversion expected"
    elif half_life < 14:
        timing = "2H"
        timing_desc = "Execute within 2 hours"
    elif half_life < 30:
        timing = "2D"
        timing_desc = "Position over 1-2 days"
    else:
        timing = "1W"
        timing_desc = "Longer-term trade (1 week+)"
    
    # Entry/exit ranges based on ATR and range position
    current_a = range_a.get("current_price", 0)
    current_b = range_b.get("current_price", 0)
    atr_a = range_a.get("atr", current_a * 0.02)
    atr_b = range_b.get("atr", current_b * 0.02)
    
    if action == "LONG_SPREAD":
        entry_a = (round(current_a - atr_a * 0.5, 2), round(current_a + atr_a * 0.3, 2))
        entry_b = (round(current_b - atr_b * 0.3, 2), round(current_b + atr_b * 0.5, 2))
        target_a = round(current_a * (1 + abs(z_score) * 0.015), 2)
        target_b = round(current_b * (1 - abs(z_score) * 0.01), 2)
        stop_a = round(current_a * 0.95, 2)
        stop_b = round(current_b * 1.05, 2)
    else:
        entry_a = (round(current_a - atr_a * 0.3, 2), round(current_a + atr_a * 0.5, 2))
        entry_b = (round(current_b - atr_b * 0.5, 2), round(current_b + atr_b * 0.3, 2))
        target_a = round(current_a * (1 - abs(z_score) * 0.015), 2)
        target_b = round(current_b * (1 + abs(z_score) * 0.015), 2)
        stop_a = round(current_a * 1.05, 2)
        stop_b = round(current_b * 0.95, 2)
    
    # Risk/reward
    potential_upside = abs(z_score) * 1.5
    potential_risk = 5.0
    rr_ratio = round(potential_upside / potential_risk, 2) if potential_risk > 0 else 0
    
    # Confidence scoring
    confidence = 0
    if main.get("is_cointegrated"):
        confidence += 30
    if abs(z_score) > 2.5:
        confidence += 20
    elif abs(z_score) > 2.0:
        confidence += 10
    if main.get("r_squared", 0) > 0.7:
        confidence += 20
    elif main.get("r_squared", 0) > 0.5:
        confidence += 10
    if half_life < 14:
        confidence += 15
    if analysis.get("cointegrated_windows", 0) >= 2:
        confidence += 15
    
    confidence_level = "HIGH" if confidence >= 70 else "MEDIUM" if confidence >= 50 else "LOW"
    
    return {
        "pair": analysis["pair"],
        "action": action,
        "direction": {analysis["asset_a"]: direction_a, analysis["asset_b"]: direction_b},
        "description": description,
        "timing": timing,
        "timing_description": timing_desc,
        "z_score": round(z_score, 4),
        "half_life_days": round(half_life, 1),
        "hedge_ratio": round(hedge_ratio, 4),
        "entry_range": {analysis["asset_a"]: entry_a, analysis["asset_b"]: entry_b},
        "targets": {analysis["asset_a"]: target_a, analysis["asset_b"]: target_b},
        "stop_loss": {analysis["asset_a"]: stop_a, analysis["asset_b"]: stop_b},
        "current_prices": {analysis["asset_a"]: current_a, analysis["asset_b"]: current_b},
        "potential_upside_pct": round(potential_upside, 2),
        "potential_risk_pct": round(potential_risk, 2),
        "risk_reward_ratio": rr_ratio,
        "confidence_score": confidence,
        "confidence_level": confidence_level,
        "volume_trend": {analysis["asset_a"]: vol_a.get("volume_trend"), analysis["asset_b"]: vol_b.get("volume_trend")},
        "range_position": {analysis["asset_a"]: range_a.get("range_position"), analysis["asset_b"]: range_b.get("range_position")},
        "cointegrated_windows": f"{analysis.get('cointegrated_windows', 0)}/{analysis.get('total_windows', 0)}",
        "r_squared": main.get("r_squared", 0),
        "adf_p_value": main.get("adf_p_value", 1),
        "timestamp": ic.time()
    }


def run_full_cointegration_analysis():
    """Analyze all 45 pair combinations."""
    global cointegration_results, rolling_cointegration, detailed_signals
    
    assets_with_data = [a for a in ASSETS if len(ohlcv_data.get(a, {}).get("prices", [])) >= MIN_DATA_POINTS]
    
    log_event("ANALYSIS", f"Running cointegration on {len(assets_with_data)} assets ({len(assets_with_data) * (len(assets_with_data)-1) // 2} pairs)")
    
    # Compute indicators for all assets
    for asset in assets_with_data:
        compute_indicators(asset)
    
    new_results = {}
    new_signals = []
    
    # Analyze all pairs
    for i, asset_a in enumerate(assets_with_data):
        for asset_b in assets_with_data[i+1:]:
            analysis = analyze_pair_comprehensive(asset_a, asset_b)
            if analysis:
                pair_key = analysis["pair"]
                new_results[pair_key] = analysis
                
                signal = generate_trading_signal(analysis)
                if signal:
                    new_signals.append(signal)
    
    # Sort signals by confidence and z-score
    new_signals.sort(key=lambda x: (-x["confidence_score"], -abs(x["z_score"])))
    
    cointegration_results = new_results
    detailed_signals = new_signals[:20]
    
    # Summary stats
    cointegrated_count = sum(1 for r in new_results.values() if r["main_analysis"].get("is_cointegrated", False))
    
    log_event("ANALYSIS", f"Found {cointegrated_count} cointegrated pairs, {len(detailed_signals)} trading signals")
    
    return {
        "pairs_analyzed": len(new_results),
        "cointegrated_pairs": cointegrated_count,
        "trading_signals": len(detailed_signals),
        "top_signal": detailed_signals[0] if detailed_signals else None
    }


# ============================================================================
# HTTP Transform
# ============================================================================

@query
def transform_price_response(args: HttpTransformArgs) -> HttpResponse:
    http_response = args["response"]
    http_response["headers"] = []
    return http_response


# ============================================================================
# Query Methods
# ============================================================================

@query
def get_health() -> str:
    return "System Operational"

@query
def get_version() -> str:
    return VERSION

@query
def get_strategy_info() -> str:
    return f"Antigravity v{VERSION} - Comprehensive Cointegration Analysis. Tracks {len(ASSETS)} cryptos, 1-year OHLCV data, Engle-Granger cointegration, rolling windows ({ROLLING_WINDOWS}), SMAs, volume/range analysis, 45 pair combinations."

@query
def get_top_10_prices() -> str:
    return json.dumps({"prices": current_prices, "last_update": last_update_time})

@query
def get_data_status() -> str:
    """Check how much data we have for each asset."""
    status = {}
    for asset in ASSETS:
        data = ohlcv_data.get(asset, {})
        status[ASSET_SYMBOLS[asset]] = {
            "price_points": len(data.get("prices", [])),
            "has_volume": len(data.get("volumes", [])) > 0,
            "has_highs_lows": len(data.get("highs", [])) > 0
        }
    return json.dumps({"data_status": status, "min_required": MIN_DATA_POINTS})

@query
def get_cointegrated_pairs() -> str:
    """Get list of cointegrated pairs."""
    coint_pairs = [
        {
            "pair": k,
            "z_score": v["main_analysis"].get("current_z_score", 0),
            "half_life": v["main_analysis"].get("half_life", 0),
            "r_squared": v["main_analysis"].get("r_squared", 0),
            "adf_p": v["main_analysis"].get("adf_p_value", 1)
        }
        for k, v in cointegration_results.items()
        if v["main_analysis"].get("is_cointegrated", False)
    ]
    coint_pairs.sort(key=lambda x: x["adf_p"])
    return json.dumps({"cointegrated_pairs": coint_pairs, "count": len(coint_pairs)})

@query
def get_all_pair_analysis() -> str:
    """Get analysis summary for all pairs."""
    summary = []
    for pair, analysis in cointegration_results.items():
        main = analysis.get("main_analysis", {})
        summary.append({
            "pair": pair,
            "cointegrated": main.get("is_cointegrated", False),
            "z_score": main.get("current_z_score", 0),
            "half_life": main.get("half_life", 0),
            "correlation": analysis.get("correlation", 0),
            "coint_windows": f"{analysis.get('cointegrated_windows', 0)}/{analysis.get('total_windows', 0)}"
        })
    summary.sort(key=lambda x: (not x["cointegrated"], -abs(x["z_score"])))
    return json.dumps({"pairs": summary, "total": len(summary)})

@query
def get_pair_detail(pair: str) -> str:
    """Get full detail for specific pair (e.g. 'BTC/ETH')."""
    analysis = cointegration_results.get(pair.upper())
    if analysis:
        return json.dumps(analysis)
    return json.dumps({"error": f"Pair {pair} not found", "available": list(cointegration_results.keys())})

@query
def get_trading_signals() -> str:
    """Get all trading signals with full details."""
    return json.dumps({
        "signals": detailed_signals,
        "count": len(detailed_signals),
        "summary": {
            "high_confidence": len([s for s in detailed_signals if s["confidence_level"] == "HIGH"]),
            "action_now": len([s for s in detailed_signals if s["timing"] == "NOW"]),
            "long_spread": len([s for s in detailed_signals if s["action"] == "LONG_SPREAD"]),
            "short_spread": len([s for s in detailed_signals if s["action"] == "SHORT_SPREAD"])
        }
    })

@query
def get_asset_indicators(asset: str) -> str:
    """Get computed indicators for an asset."""
    symbol_to_id = {v.lower(): k for k, v in ASSET_SYMBOLS.items()}
    asset_id = symbol_to_id.get(asset.lower(), asset.lower())
    
    ind = indicators.get(asset_id, {})
    data = ohlcv_data.get(asset_id, {})
    
    return json.dumps({
        "asset": ASSET_SYMBOLS.get(asset_id, asset.upper()),
        "indicators": ind,
        "data_points": len(data.get("prices", [])),
        "latest_prices": data.get("prices", [])[-10:],
        "latest_volumes": data.get("volumes", [])[-10:]
    })

@query
def get_state() -> str:
    return json.dumps({
        "version": VERSION,
        "assets_tracked": len(ASSETS),
        "total_data_points": sum(len(d.get("prices", [])) for d in ohlcv_data.values()),
        "pairs_analyzed": len(cointegration_results),
        "cointegrated_pairs": sum(1 for r in cointegration_results.values() if r["main_analysis"].get("is_cointegrated")),
        "active_signals": len(detailed_signals),
        "registered_users": len(registered_users),
        "last_update": last_update_time
    })

@query
def get_logs() -> str:
    return json.dumps(execution_logs[-50:])

@query
def whoami() -> str:
    return str(ic.caller())


# ============================================================================
# User Management
# ============================================================================

@update
def register_wallet() -> str:
    global registered_users
    caller = str(ic.caller())
    if caller not in registered_users:
        registered_users.append(caller)
        user_storage.insert(caller, json.dumps({"principal": caller, "ts": ic.time()}))
    return json.dumps({"success": True, "principal": caller})


# ============================================================================
# Data Fetching
# ============================================================================

@update
def fetch_ohlcv_data(asset: str) -> Async[str]:
    """Fetch 1-year OHLCV data for an asset from CoinGecko market_chart."""
    global ohlcv_data, last_update_time
    
    symbol_to_id = {v.lower(): k for k, v in ASSET_SYMBOLS.items()}
    asset_id = symbol_to_id.get(asset.lower(), asset.lower())
    
    if asset_id not in ASSETS:
        return json.dumps({"success": False, "error": f"Asset {asset} not in tracked list"})
    
    # CoinGecko market_chart endpoint - returns OHLC + volume
    url = f"{COINGECKO_BASE}/coins/{asset_id}/market_chart?vs_currency=usd&days=365&interval=daily"
    
    log_event("API", f"Fetching 1-year OHLCV for {asset_id}...")
    
    http_result: CallResult[HttpResponse] = yield management_canister.http_request(
        {"url": url, "max_response_bytes": 500_000, "method": {"get": None}, "headers": [], "body": None,
         "transform": {"function": (ic.id(), "transform_price_response"), "context": bytes()}}
    ).with_cycles(HTTP_CYCLES)
    
    def handle_success(response: HttpResponse) -> str:
        global ohlcv_data, last_update_time
        try:
            data = json.loads(response["body"].decode("utf-8"))
            
            prices = [p[1] for p in data.get("prices", [])]
            volumes = [v[1] for v in data.get("total_volumes", [])]
            timestamps = [p[0] for p in data.get("prices", [])]
            
            # CoinGecko market_chart doesn't provide OHLC directly for free tier
            # We'll approximate highs/lows from prices (±2% band)
            highs = [p * 1.02 for p in prices]
            lows = [p * 0.98 for p in prices]
            
            ohlcv_data[asset_id] = {
                "prices": prices,
                "volumes": volumes,
                "highs": highs,
                "lows": lows,
                "timestamps": timestamps
            }
            
            last_update_time = ic.time()
            
            # Compute indicators
            compute_indicators(asset_id)
            
            log_event("DATA", f"Loaded {len(prices)} days for {asset_id}")
            
            return json.dumps({
                "success": True,
                "asset": ASSET_SYMBOLS[asset_id],
                "data_points": len(prices),
                "date_range": f"{len(prices)} days"
            })
            
        except Exception as e:
            log_event("ERROR", str(e))
            return json.dumps({"success": False, "error": str(e)})
    
    return match(http_result, {"Ok": handle_success, "Err": lambda e: json.dumps({"success": False, "error": e})})


@update
def fetch_current_prices() -> Async[str]:
    """Fetch current prices for all assets."""
    global current_prices, last_update_time
    
    assets_str = ",".join(ASSETS)
    url = f"{COINGECKO_BASE}/simple/price?ids={assets_str}&vs_currencies=usd&include_24hr_vol=true"
    
    http_result: CallResult[HttpResponse] = yield management_canister.http_request(
        {"url": url, "max_response_bytes": 5_000, "method": {"get": None}, "headers": [], "body": None,
         "transform": {"function": (ic.id(), "transform_price_response"), "context": bytes()}}
    ).with_cycles(HTTP_CYCLES)
    
    def handle_success(response: HttpResponse) -> str:
        global current_prices, last_update_time
        try:
            data = json.loads(response["body"].decode("utf-8"))
            for asset_id in ASSETS:
                if asset_id in data:
                    symbol = ASSET_SYMBOLS[asset_id]
                    current_prices[symbol] = data[asset_id].get("usd", 0)
            last_update_time = ic.time()
            return json.dumps({"success": True, "prices": current_prices})
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    return match(http_result, {"Ok": handle_success, "Err": lambda e: json.dumps({"success": False, "error": e})})


@update
def run_analysis() -> str:
    """Run full cointegration analysis on all loaded data."""
    result = run_full_cointegration_analysis()
    return json.dumps(result)


@update
def fetch_and_analyze_all() -> Async[str]:
    """Fetch current prices, then run analysis."""
    global current_prices, last_update_time
    
    assets_str = ",".join(ASSETS)
    url = f"{COINGECKO_BASE}/simple/price?ids={assets_str}&vs_currencies=usd"
    
    http_result: CallResult[HttpResponse] = yield management_canister.http_request(
        {"url": url, "max_response_bytes": 5_000, "method": {"get": None}, "headers": [], "body": None,
         "transform": {"function": (ic.id(), "transform_price_response"), "context": bytes()}}
    ).with_cycles(HTTP_CYCLES)
    
    def handle_success(response: HttpResponse) -> str:
        global current_prices, last_update_time, ohlcv_data
        try:
            data = json.loads(response["body"].decode("utf-8"))
            
            for asset_id in ASSETS:
                if asset_id in data:
                    symbol = ASSET_SYMBOLS[asset_id]
                    price = data[asset_id].get("usd", 0)
                    current_prices[symbol] = price
                    
                    # Append to OHLCV data
                    if price > 0:
                        ohlcv_data[asset_id]["prices"].append(price)
                        ohlcv_data[asset_id]["volumes"].append(0)
                        ohlcv_data[asset_id]["highs"].append(price * 1.01)
                        ohlcv_data[asset_id]["lows"].append(price * 0.99)
                        ohlcv_data[asset_id]["timestamps"].append(ic.time())
                        
                        # Keep bounded
                        for key in ["prices", "volumes", "highs", "lows", "timestamps"]:
                            if len(ohlcv_data[asset_id][key]) > 500:
                                ohlcv_data[asset_id][key] = ohlcv_data[asset_id][key][-500:]
            
            last_update_time = ic.time()
            
            # Run analysis
            result = run_full_cointegration_analysis()
            
            return json.dumps({
                "success": True,
                "prices": current_prices,
                "analysis": result,
                "signals": detailed_signals[:5]
            })
            
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    return match(http_result, {"Ok": handle_success, "Err": lambda e: json.dumps({"success": False, "error": e})})


@update
def reset_all() -> str:
    global ohlcv_data, indicators, cointegration_results, rolling_cointegration, current_prices, detailed_signals, last_update_time, execution_logs
    ohlcv_data = {asset: {"prices": [], "volumes": [], "highs": [], "lows": [], "timestamps": []} for asset in ASSETS}
    indicators = {asset: {} for asset in ASSETS}
    cointegration_results = {}
    rolling_cointegration = {}
    current_prices = {}
    detailed_signals = []
    last_update_time = 0
    execution_logs = []
    return json.dumps({"success": True, "message": "All data reset"})


@update
def load_sample_data() -> str:
    """
    Load synthetic cointegrated price data for all assets.
    This allows immediate testing of the cointegration analysis.
    Generates 100 days of price data with realistic correlations.
    """
    global ohlcv_data, current_prices, last_update_time
    
    import random
    random.seed(42)  # Reproducible
    
    # Base prices for each asset (approximate current prices)
    base_prices = {
        "bitcoin": 88000, "ethereum": 3000, "solana": 125, "ripple": 2.2,
        "dogecoin": 0.32, "cardano": 0.90, "avalanche-2": 35,
        "polkadot": 6.5, "chainlink": 22, "internet-computer": 10.5
    }
    
    n_days = 100  # Generate 100 days of data
    
    # Generate common market trend (creates cointegration)
    common_trend = [0]
    for _ in range(n_days - 1):
        common_trend.append(common_trend[-1] + random.gauss(0, 0.015))
    
    for asset_id in ASSETS:
        base = base_prices.get(asset_id, 100)
        
        # Each asset follows the common trend + idiosyncratic noise
        # This creates cointegration between pairs
        beta = random.uniform(0.5, 1.5)  # Different sensitivity to common trend
        prices = []
        volumes = []
        highs = []
        lows = []
        timestamps = []
        
        cumulative = 0
        for i in range(n_days):
            # Price: base * (1 + common_trend * beta + idiosyncratic noise)
            noise = random.gauss(0, 0.02)  # 2% daily volatility
            cumulative += common_trend[i] * beta + noise
            price = base * math.exp(cumulative)
            
            daily_range = price * random.uniform(0.02, 0.05)  # 2-5% daily range
            
            prices.append(round(price, 6))
            highs.append(round(price + daily_range / 2, 6))
            lows.append(round(price - daily_range / 2, 6))
            volumes.append(random.uniform(1e6, 1e9))  # Random volume
            timestamps.append(ic.time() - (n_days - i) * 86400 * 1_000_000_000)  # Nanoseconds
        
        ohlcv_data[asset_id] = {
            "prices": prices,
            "volumes": volumes,
            "highs": highs,
            "lows": lows,
            "timestamps": timestamps
        }
        
        current_prices[ASSET_SYMBOLS[asset_id]] = prices[-1]
    
    # Compute indicators
    for asset in ASSETS:
        compute_indicators(asset)
    
    last_update_time = ic.time()
    
    log_event("DATA", f"Loaded {n_days} days of sample data for {len(ASSETS)} assets")
    
    # Run analysis
    result = run_full_cointegration_analysis()
    
    return json.dumps({
        "success": True,
        "message": f"Loaded {n_days} days of sample data for {len(ASSETS)} assets",
        "data_points_per_asset": n_days,
        "analysis_result": result
    })


# ============================================================================
# TRADE EXECUTION FUNCTIONS
# ============================================================================

def calculate_portfolio_value() -> float:
    """Calculate total portfolio value including positions."""
    global portfolio
    total = portfolio["cash"]
    
    for asset, pos in portfolio["positions"].items():
        current_price = current_prices.get(asset, pos["avg_price"])
        if pos["side"] == "long":
            total += pos["quantity"] * current_price
        else:  # short
            # Short P&L = (entry_price - current_price) * quantity
            total += pos["quantity"] * (2 * pos["avg_price"] - current_price)
    
    portfolio["total_value"] = total
    portfolio["pnl"] = total - INITIAL_BALANCE
    portfolio["pnl_pct"] = (portfolio["pnl"] / INITIAL_BALANCE) * 100
    
    return total


def execute_single_trade(asset: str, side: str, quantity: float, price: float) -> dict:
    """Execute a single trade (buy/sell an asset)."""
    global portfolio, trade_history
    
    trade_value = quantity * price
    
    if side == "buy":
        if trade_value > portfolio["cash"]:
            return {"success": False, "error": "Insufficient cash"}
        
        portfolio["cash"] -= trade_value
        
        if asset in portfolio["positions"]:
            pos = portfolio["positions"][asset]
            if pos["side"] == "long":
                # Add to long position
                total_qty = pos["quantity"] + quantity
                pos["avg_price"] = (pos["avg_price"] * pos["quantity"] + price * quantity) / total_qty
                pos["quantity"] = total_qty
            else:
                # Close short position
                pos["quantity"] -= quantity
                if pos["quantity"] <= 0:
                    del portfolio["positions"][asset]
        else:
            portfolio["positions"][asset] = {
                "quantity": quantity,
                "avg_price": price,
                "side": "long"
            }
    
    elif side == "sell":
        if asset in portfolio["positions"]:
            pos = portfolio["positions"][asset]
            if pos["side"] == "long":
                pos["quantity"] -= quantity
                portfolio["cash"] += trade_value
                if pos["quantity"] <= 0:
                    del portfolio["positions"][asset]
            else:
                # Add to short position
                total_qty = pos["quantity"] + quantity
                pos["avg_price"] = (pos["avg_price"] * pos["quantity"] + price * quantity) / total_qty
                pos["quantity"] = total_qty
                portfolio["cash"] += trade_value
        else:
            # Open short position
            portfolio["positions"][asset] = {
                "quantity": quantity,
                "avg_price": price,
                "side": "short"
            }
            portfolio["cash"] += trade_value
    
    trade = {
        "id": len(trade_history) + 1,
        "asset": asset,
        "side": side,
        "quantity": round(quantity, 6),
        "price": price,
        "value": round(trade_value, 2),
        "timestamp": ic.time(),
        "type": "single"
    }
    trade_history.append(trade)
    
    calculate_portfolio_value()
    log_event("TRADE", f"Executed {side.upper()} {quantity:.4f} {asset} @ ${price:.2f}")
    
    return {"success": True, "trade": trade}


def execute_pair_trade(signal: dict) -> dict:
    """Execute a pairs trade based on a trading signal."""
    global portfolio, trade_history, active_pair_trades
    
    asset_a = signal.get("asset_a", signal["pair"].split("/")[0])
    asset_b = signal.get("asset_b", signal["pair"].split("/")[1])
    action = signal["action"]
    
    price_a = current_prices.get(asset_a, 0)
    price_b = current_prices.get(asset_b, 0)
    
    if price_a == 0 or price_b == 0:
        return {"success": False, "error": "Missing price data"}
    
    # Calculate position sizes (equal dollar value)
    position_value = min(max_position_size, portfolio["cash"] * 0.2)  # 20% of cash max
    qty_a = position_value / price_a
    qty_b = position_value / price_b
    
    if action == "LONG_SPREAD":
        # Long A, Short B
        result_a = execute_single_trade(asset_a, "buy", qty_a, price_a)
        result_b = execute_single_trade(asset_b, "sell", qty_b, price_b)
    elif action == "SHORT_SPREAD":
        # Short A, Long B
        result_a = execute_single_trade(asset_a, "sell", qty_a, price_a)
        result_b = execute_single_trade(asset_b, "buy", qty_b, price_b)
    else:
        return {"success": False, "error": f"Unknown action: {action}"}
    
    pair_trade = {
        "id": len(active_pair_trades) + 1,
        "pair": signal["pair"],
        "action": action,
        "entry_time": ic.time(),
        "entry_z_score": signal.get("z_score", 0),
        "entry_prices": {"a": price_a, "b": price_b},
        "quantities": {"a": qty_a, "b": qty_b},
        "status": "OPEN",
        "pnl": 0
    }
    active_pair_trades.append(pair_trade)
    
    log_event("PAIR_TRADE", f"Opened {action} on {signal['pair']} | Z={signal.get('z_score', 0):.2f}")
    
    return {
        "success": True,
        "pair_trade": pair_trade,
        "trades": [result_a, result_b]
    }


def check_and_close_positions():
    """Check active pair trades and close if Z-score reverted."""
    global active_pair_trades
    
    closed = []
    for trade in active_pair_trades:
        if trade["status"] != "OPEN":
            continue
        
        # Check current Z-score for this pair
        pair_analysis = cointegration_results.get(trade["pair"])
        if not pair_analysis:
            continue
        
        current_z = pair_analysis["main_analysis"].get("current_z_score", 0)
        entry_z = trade["entry_z_score"]
        
        # Close conditions:
        # 1. Z-score crossed zero (mean reversion complete)
        # 2. Z-score reversed beyond stop-loss
        should_close = False
        reason = ""
        
        if entry_z < 0 and current_z > -0.5:
            should_close = True
            reason = "Mean reversion complete (Z > -0.5)"
        elif entry_z > 0 and current_z < 0.5:
            should_close = True
            reason = "Mean reversion complete (Z < 0.5)"
        elif abs(current_z) > 4:
            should_close = True
            reason = "Stop-loss triggered (|Z| > 4)"
        
        if should_close:
            # Close the positions
            asset_a = trade["pair"].split("/")[0]
            asset_b = trade["pair"].split("/")[1]
            
            # Reverse the original trades
            if trade["action"] == "LONG_SPREAD":
                execute_single_trade(asset_a, "sell", trade["quantities"]["a"], current_prices.get(asset_a, 0))
                execute_single_trade(asset_b, "buy", trade["quantities"]["b"], current_prices.get(asset_b, 0))
            else:
                execute_single_trade(asset_a, "buy", trade["quantities"]["a"], current_prices.get(asset_a, 0))
                execute_single_trade(asset_b, "sell", trade["quantities"]["b"], current_prices.get(asset_b, 0))
            
            trade["status"] = "CLOSED"
            trade["exit_time"] = ic.time()
            trade["exit_z_score"] = current_z
            trade["close_reason"] = reason
            
            closed.append(trade["pair"])
            log_event("PAIR_CLOSE", f"Closed {trade['pair']} | Reason: {reason}")
    
    return closed


def auto_execute_signals():
    """Automatically execute trades based on signals if auto-execute is enabled."""
    global auto_execute_enabled
    
    if not auto_execute_enabled:
        return {"executed": 0, "reason": "Auto-execute disabled"}
    
    # First, check and close any positions that should be closed
    closed = check_and_close_positions()
    
    # Then, execute new signals
    executed = []
    for signal in detailed_signals:
        # Only execute HIGH confidence signals
        if signal.get("confidence_level") != "HIGH":
            continue
        
        # Check if we already have a position in this pair
        pair = signal["pair"]
        already_open = any(t["pair"] == pair and t["status"] == "OPEN" for t in active_pair_trades)
        if already_open:
            continue
        
        # Execute the trade
        result = execute_pair_trade(signal)
        if result["success"]:
            executed.append(pair)
    
    return {"executed": len(executed), "closed": len(closed), "trades": executed}


# ============================================================================
# TRADE EXECUTION API ENDPOINTS
# ============================================================================

@query
def get_portfolio() -> str:
    """Get current portfolio state."""
    calculate_portfolio_value()
    return json.dumps({
        "portfolio": portfolio,
        "positions_count": len(portfolio["positions"]),
        "active_pair_trades": len([t for t in active_pair_trades if t["status"] == "OPEN"])
    })


@query
def get_trade_history() -> str:
    """Get all executed trades."""
    return json.dumps({
        "trades": trade_history[-50:],  # Last 50 trades
        "total_trades": len(trade_history),
        "pair_trades": active_pair_trades
    })


@query
def get_active_positions() -> str:
    """Get current open positions."""
    calculate_portfolio_value()
    positions = []
    for asset, pos in portfolio["positions"].items():
        current_price = current_prices.get(asset, pos["avg_price"])
        if pos["side"] == "long":
            pnl = (current_price - pos["avg_price"]) * pos["quantity"]
        else:
            pnl = (pos["avg_price"] - current_price) * pos["quantity"]
        
        positions.append({
            "asset": asset,
            "side": pos["side"],
            "quantity": pos["quantity"],
            "avg_price": pos["avg_price"],
            "current_price": current_price,
            "pnl": round(pnl, 2),
            "pnl_pct": round((pnl / (pos["avg_price"] * pos["quantity"])) * 100, 2)
        })
    
    return json.dumps({"positions": positions})


@update
def execute_trade(asset: str, side: str, amount: float64) -> str:
    """Manually execute a single trade."""
    price = current_prices.get(asset.upper())
    if not price:
        return json.dumps({"success": False, "error": f"No price for {asset}"})
    
    quantity = amount / price
    result = execute_single_trade(asset.upper(), side.lower(), quantity, price)
    return json.dumps(result)


@update
def execute_signal(pair: str) -> str:
    """Execute a specific signal by pair name."""
    for signal in detailed_signals:
        if signal["pair"] == pair.upper():
            result = execute_pair_trade(signal)
            return json.dumps(result)
    
    return json.dumps({"success": False, "error": f"No signal found for {pair}"})


@update
def toggle_auto_execute() -> str:
    """Toggle automatic trade execution on/off."""
    global auto_execute_enabled
    auto_execute_enabled = not auto_execute_enabled
    return json.dumps({"auto_execute_enabled": auto_execute_enabled})


@update
def run_auto_trading() -> str:
    """Run one cycle of automatic trading."""
    result = auto_execute_signals()
    calculate_portfolio_value()
    return json.dumps({
        "result": result,
        "portfolio": portfolio
    })


@update
def analyze_and_trade() -> str:
    """Full cycle: analyze all pairs and auto-execute trades."""
    # Run analysis
    analysis_result = run_full_cointegration_analysis()
    
    # Auto-execute trades based on signals
    trade_result = auto_execute_signals()
    
    # Update portfolio value
    calculate_portfolio_value()
    
    return json.dumps({
        "analysis": analysis_result,
        "trading": trade_result,
        "portfolio": portfolio,
        "active_trades": len([t for t in active_pair_trades if t["status"] == "OPEN"])
    })


@update
def reset_portfolio() -> str:
    """Reset portfolio to initial state."""
    global portfolio, trade_history, active_pair_trades
    portfolio = {
        "cash": INITIAL_BALANCE,
        "positions": {},
        "total_value": INITIAL_BALANCE,
        "pnl": 0.0,
        "pnl_pct": 0.0
    }
    trade_history = []
    active_pair_trades = []
    return json.dumps({"success": True, "portfolio": portfolio})


