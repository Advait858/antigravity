"""
Antigravity Bot v4.0 - Advanced ICP Trading Agent
Multi-asset statistical arbitrage with 1-year historical analysis and detailed recommendations.

Features:
- Top 10 crypto price monitoring with 1-year historical data
- Advanced cointegration analysis on 365-day windows
- Detailed trade recommendations with timing (NOW/2H/2D)
- Entry/exit price ranges with risk/reward projections
- Wallet connection via Internet Identity
- HTTPS Outcalls for live + historical data

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
# Constants & Configuration
# ============================================================================

VERSION = "4.0.0-advanced-analysis"

# Top 10 crypto assets to track
ASSETS = ["bitcoin", "ethereum", "solana", "ripple", "dogecoin", 
          "cardano", "avalanche-2", "polkadot", "chainlink", "internet-computer"]

ASSET_SYMBOLS = {
    "bitcoin": "BTC", "ethereum": "ETH", "solana": "SOL", "ripple": "XRP",
    "dogecoin": "DOGE", "cardano": "ADA", "avalanche-2": "AVAX",
    "polkadot": "DOT", "chainlink": "LINK", "internet-computer": "ICP"
}

# CoinGecko API endpoints
COINGECKO_SIMPLE_URL = "https://api.coingecko.com/api/v3/simple/price"
COINGECKO_HISTORY_URL = "https://api.coingecko.com/api/v3/coins"
HTTP_CYCLES = 200_000_000

# Trading thresholds
ENTRY_THRESHOLD = 2.0
EXIT_THRESHOLD = 0.5
STOP_LOSS_THRESHOLD = 4.0
ADF_P_THRESHOLD = 0.05
MIN_HISTORY = 30
LOOKBACK_DAYS = 365  # 1 year of data

# Recommendation timing
TIMING_NOW = "NOW"
TIMING_2H = "2H"
TIMING_2D = "2D"
TIMING_1W = "1W"


# ============================================================================
# Stable Storage
# ============================================================================

price_storage = StableBTreeMap[str, str](memory_id=0, max_key_size=50, max_value_size=100_000)
user_storage = StableBTreeMap[str, str](memory_id=1, max_key_size=100, max_value_size=1_000)
analysis_storage = StableBTreeMap[str, str](memory_id=2, max_key_size=50, max_value_size=50_000)


# ============================================================================
# Global State
# ============================================================================

current_prices: dict = {}
price_history: dict = {asset: [] for asset in ASSETS}
historical_data: dict = {asset: [] for asset in ASSETS}  # 1-year data
last_update_time: int = 0
recommendations: list = []
detailed_recommendations: list = []
pair_analysis: dict = {}
registered_users: list = []
execution_logs: list = []


# ============================================================================
# Statistical Helper Functions
# ============================================================================

def log_event(event_type: str, message: str):
    global execution_logs
    log_entry = {"type": event_type, "message": message, "timestamp": ic.time()}
    execution_logs.append(log_entry)
    if len(execution_logs) > 200:
        execution_logs = execution_logs[-100:]


def mean(data: list) -> float:
    return sum(data) / len(data) if data else 0.0


def std_dev(data: list) -> float:
    if len(data) < 2:
        return 0.0
    m = mean(data)
    variance = sum((x - m) ** 2 for x in data) / len(data)
    return math.sqrt(variance)


def percentile(data: list, p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[int(f)] * (c - k) + sorted_data[int(c)] * (k - f)


def calculate_returns(prices: list) -> list:
    """Calculate percentage returns from prices."""
    if len(prices) < 2:
        return []
    return [(prices[i] / prices[i-1] - 1) * 100 for i in range(1, len(prices))]


def calculate_volatility(prices: list, window: int = 30) -> float:
    """Calculate annualized volatility."""
    returns = calculate_returns(prices[-window:] if len(prices) > window else prices)
    if not returns:
        return 0.0
    return std_dev(returns) * math.sqrt(365)


def calculate_sharpe_ratio(returns: list, risk_free_rate: float = 0.05) -> float:
    """Calculate Sharpe ratio."""
    if not returns or len(returns) < 2:
        return 0.0
    avg_return = mean(returns) * 365  # Annualize
    vol = std_dev(returns) * math.sqrt(365)
    if vol == 0:
        return 0.0
    return (avg_return - risk_free_rate) / vol


def calculate_max_drawdown(prices: list) -> float:
    """Calculate maximum drawdown percentage."""
    if len(prices) < 2:
        return 0.0
    peak = prices[0]
    max_dd = 0.0
    for price in prices:
        if price > peak:
            peak = price
        dd = (peak - price) / peak * 100
        if dd > max_dd:
            max_dd = dd
    return max_dd


def calculate_z_score(values: list) -> float:
    if len(values) < 20:
        return 0.0
    m = mean(values[-50:])
    s = std_dev(values[-50:])
    if s == 0:
        return 0.0
    return (values[-1] - m) / s


def ols_regression(y: list, x: list) -> dict:
    """Simple OLS regression: y = alpha + beta * x."""
    n = min(len(y), len(x))
    if n < 10:
        return {"alpha": 0.0, "beta": 1.0, "r_squared": 0.0}
    
    y = y[-n:]
    x = x[-n:]
    
    mean_x = mean(x)
    mean_y = mean(y)
    
    cov_xy = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / n
    var_x = sum((xi - mean_x) ** 2 for xi in x) / n
    
    if var_x == 0:
        return {"alpha": mean_y, "beta": 0.0, "r_squared": 0.0}
    
    beta = cov_xy / var_x
    alpha = mean_y - beta * mean_x
    
    # R-squared
    y_pred = [alpha + beta * xi for xi in x]
    ss_res = sum((y[i] - y_pred[i]) ** 2 for i in range(n))
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return {"alpha": alpha, "beta": beta, "r_squared": r_squared}


def adf_test(series: list) -> dict:
    """Augmented Dickey-Fuller test."""
    n = len(series)
    if n < 30:
        return {"p_value": 1.0, "is_stationary": False, "test_stat": 0.0}
    
    diff = [series[i] - series[i-1] for i in range(1, n)]
    y = diff[1:]
    x = series[1:-1]
    
    n_reg = len(y)
    if n_reg < 10:
        return {"p_value": 1.0, "is_stationary": False, "test_stat": 0.0}
    
    xy = sum(x[i] * y[i] for i in range(n_reg))
    xx = sum(x[i] * x[i] for i in range(n_reg))
    
    if xx == 0:
        return {"p_value": 1.0, "is_stationary": False, "test_stat": 0.0}
    
    gamma = xy / xx
    residuals = [y[i] - gamma * x[i] for i in range(n_reg)]
    sse = sum(r * r for r in residuals)
    mse = sse / max(n_reg - 1, 1)
    se = math.sqrt(mse / xx) if mse > 0 else 1.0
    t_stat = gamma / se if se > 0 else 0.0
    
    if t_stat < -3.5:
        p_value = 0.01
    elif t_stat < -2.86:
        p_value = 0.05
    elif t_stat < -2.5:
        p_value = 0.10
    else:
        p_value = 0.5
    
    return {"p_value": p_value, "is_stationary": t_stat < -2.86, "test_stat": t_stat}


def calculate_half_life(spread: list) -> float:
    """Calculate mean reversion half-life in days."""
    if len(spread) < 30:
        return float('inf')
    
    spread_lag = spread[:-1]
    spread_diff = [spread[i] - spread[i-1] for i in range(1, len(spread))]
    
    reg = ols_regression(spread_diff, spread_lag)
    if reg["beta"] >= 0:
        return float('inf')
    
    half_life = -math.log(2) / reg["beta"]
    return max(1, min(half_life, 365))  # Clamp between 1 and 365 days


def calculate_spread(prices_a: list, prices_b: list) -> tuple:
    """Calculate hedge ratio and spread using OLS."""
    n = min(len(prices_a), len(prices_b))
    if n < 20:
        return [], 1.0, 0.0
    
    a = prices_a[-n:]
    b = prices_b[-n:]
    
    reg = ols_regression(a, b)
    spread = [a[i] - reg["beta"] * b[i] - reg["alpha"] for i in range(n)]
    
    return spread, reg["beta"], reg["r_squared"]


def calculate_correlation(prices_a: list, prices_b: list) -> float:
    n = min(len(prices_a), len(prices_b))
    if n < 10:
        return 0.0
    
    a = prices_a[-n:]
    b = prices_b[-n:]
    
    mean_a, mean_b = mean(a), mean(b)
    cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n)) / n
    std_a, std_b = std_dev(a), std_dev(b)
    
    if std_a == 0 or std_b == 0:
        return 0.0
    return cov / (std_a * std_b)


# ============================================================================
# Advanced Analysis Functions
# ============================================================================

def analyze_pair_detailed(asset_a: str, asset_b: str) -> dict:
    """Perform comprehensive cointegration analysis on a pair."""
    prices_a = historical_data.get(asset_a, []) or price_history.get(asset_a, [])
    prices_b = historical_data.get(asset_b, []) or price_history.get(asset_b, [])
    
    if len(prices_a) < MIN_HISTORY or len(prices_b) < MIN_HISTORY:
        return None
    
    symbol_a = ASSET_SYMBOLS.get(asset_a, asset_a.upper())
    symbol_b = ASSET_SYMBOLS.get(asset_b, asset_b.upper())
    
    # Calculate spread and hedge ratio
    spread, hedge_ratio, r_squared = calculate_spread(prices_a, prices_b)
    if not spread:
        return None
    
    # ADF test for cointegration
    adf = adf_test(spread)
    
    # Z-score and statistics
    z_score = calculate_z_score(spread)
    spread_mean = mean(spread)
    spread_std = std_dev(spread)
    
    # Correlation
    correlation = calculate_correlation(prices_a, prices_b)
    
    # Half-life of mean reversion
    half_life = calculate_half_life(spread)
    
    # Current prices
    current_a = prices_a[-1] if prices_a else 0
    current_b = prices_b[-1] if prices_b else 0
    
    # Price stats
    vol_a = calculate_volatility(prices_a)
    vol_b = calculate_volatility(prices_b)
    max_dd_a = calculate_max_drawdown(prices_a)
    max_dd_b = calculate_max_drawdown(prices_b)
    
    # Returns
    returns_a = calculate_returns(prices_a)
    returns_b = calculate_returns(prices_b)
    
    return {
        "pair": f"{symbol_a}/{symbol_b}",
        "asset_a": symbol_a,
        "asset_b": symbol_b,
        "is_cointegrated": adf["is_stationary"],
        "adf_statistic": round(adf["test_stat"], 4),
        "adf_p_value": round(adf["p_value"], 4),
        "hedge_ratio": round(hedge_ratio, 6),
        "r_squared": round(r_squared, 4),
        "correlation": round(correlation, 4),
        "z_score": round(z_score, 4),
        "spread_mean": round(spread_mean, 4),
        "spread_std": round(spread_std, 4),
        "half_life_days": round(half_life, 1),
        "current_prices": {"a": current_a, "b": current_b},
        "volatility": {"a": round(vol_a, 2), "b": round(vol_b, 2)},
        "max_drawdown": {"a": round(max_dd_a, 2), "b": round(max_dd_b, 2)},
        "data_points": min(len(prices_a), len(prices_b)),
        "timestamp": ic.time()
    }


def generate_detailed_recommendation(analysis: dict) -> dict:
    """Generate detailed trade recommendation with timing and ranges."""
    if not analysis or not analysis.get("is_cointegrated"):
        return None
    
    z_score = analysis["z_score"]
    half_life = analysis["half_life_days"]
    spread_std = analysis["spread_std"]
    hedge_ratio = analysis["hedge_ratio"]
    current_a = analysis["current_prices"]["a"]
    current_b = analysis["current_prices"]["b"]
    vol_a = analysis["volatility"]["a"]
    vol_b = analysis["volatility"]["b"]
    
    # Skip if z-score not significant
    if abs(z_score) < 1.5:
        return None
    
    # Determine action and timing
    if z_score > ENTRY_THRESHOLD:
        action = "SHORT_SPREAD"
        description = f"Short {analysis['asset_a']}, Long {analysis['asset_b']}"
        expected_move = "Spread expected to DECREASE (mean reversion)"
    elif z_score < -ENTRY_THRESHOLD:
        action = "LONG_SPREAD"
        description = f"Long {analysis['asset_a']}, Short {analysis['asset_b']}"
        expected_move = "Spread expected to INCREASE (mean reversion)"
    elif abs(z_score) > 1.5:
        action = "WATCH"
        description = "Approaching entry threshold"
        expected_move = "Wait for stronger signal"
    else:
        return None
    
    # Timing based on half-life
    if half_life < 5:
        timing = TIMING_NOW
        timing_description = "Enter immediately - fast mean reversion"
    elif half_life < 14:
        timing = TIMING_2H
        timing_description = "Enter within 2 hours for optimal positioning"
    elif half_life < 30:
        timing = TIMING_2D
        timing_description = "Position over 1-2 days"
    else:
        timing = TIMING_1W
        timing_description = "Longer-term trade, position over 1 week"
    
    # Calculate entry/exit ranges
    daily_move_a = current_a * (vol_a / 100 / math.sqrt(365))
    daily_move_b = current_b * (vol_b / 100 / math.sqrt(365))
    
    if action == "LONG_SPREAD":
        # Long A, Short B - expect A to rise relative to B
        entry_range_a = (round(current_a * 0.99, 2), round(current_a * 1.01, 2))
        entry_range_b = (round(current_b * 0.99, 2), round(current_b * 1.01, 2))
        target_a = round(current_a * (1 + abs(z_score) * 0.02), 2)
        target_b = round(current_b * (1 - abs(z_score) * 0.01), 2)
        stop_a = round(current_a * 0.95, 2)
        stop_b = round(current_b * 1.05, 2)
    else:
        # Short A, Long B - expect B to rise relative to A
        entry_range_a = (round(current_a * 0.99, 2), round(current_a * 1.01, 2))
        entry_range_b = (round(current_b * 0.99, 2), round(current_b * 1.01, 2))
        target_a = round(current_a * (1 - abs(z_score) * 0.02), 2)
        target_b = round(current_b * (1 + abs(z_score) * 0.02), 2)
        stop_a = round(current_a * 1.05, 2)
        stop_b = round(current_b * 0.95, 2)
    
    # Risk/Reward calculation
    potential_upside = abs(z_score) * 2  # Approximate % gain at mean reversion
    potential_risk = 5.0  # Stop loss at 5%
    risk_reward_ratio = round(potential_upside / potential_risk, 2) if potential_risk > 0 else 0
    
    # Confidence scoring
    confidence_score = 0
    if abs(z_score) > 2.5:
        confidence_score += 30
    elif abs(z_score) > 2.0:
        confidence_score += 20
    else:
        confidence_score += 10
    
    if analysis["correlation"] > 0.7:
        confidence_score += 25
    elif analysis["correlation"] > 0.5:
        confidence_score += 15
    
    if analysis["r_squared"] > 0.7:
        confidence_score += 25
    elif analysis["r_squared"] > 0.5:
        confidence_score += 15
    
    if half_life < 14:
        confidence_score += 20
    elif half_life < 30:
        confidence_score += 10
    
    confidence_level = "HIGH" if confidence_score >= 70 else "MEDIUM" if confidence_score >= 50 else "LOW"
    
    return {
        "pair": analysis["pair"],
        "action": action,
        "description": description,
        "timing": timing,
        "timing_description": timing_description,
        "expected_move": expected_move,
        "z_score": round(z_score, 4),
        "half_life_days": round(half_life, 1),
        "entry_ranges": {
            analysis["asset_a"]: entry_range_a,
            analysis["asset_b"]: entry_range_b
        },
        "targets": {
            analysis["asset_a"]: target_a,
            analysis["asset_b"]: target_b
        },
        "stop_loss": {
            analysis["asset_a"]: stop_a,
            analysis["asset_b"]: stop_b
        },
        "potential_upside_pct": round(potential_upside, 2),
        "potential_risk_pct": round(potential_risk, 2),
        "risk_reward_ratio": risk_reward_ratio,
        "confidence_score": confidence_score,
        "confidence_level": confidence_level,
        "hedge_ratio": round(hedge_ratio, 4),
        "correlation": round(analysis["correlation"], 4),
        "current_prices": analysis["current_prices"],
        "volatility": analysis["volatility"],
        "timestamp": ic.time()
    }


def run_full_analysis() -> dict:
    """Run comprehensive analysis on all pairs."""
    global pair_analysis, detailed_recommendations
    
    new_analysis = {}
    new_recommendations = []
    
    assets_with_data = [a for a in ASSETS if len(historical_data.get(a, []) or price_history.get(a, [])) >= MIN_HISTORY]
    
    log_event("ANALYSIS", f"Analyzing {len(assets_with_data)} assets with sufficient data")
    
    # Analyze each pair
    for i, asset_a in enumerate(assets_with_data):
        for asset_b in assets_with_data[i+1:]:
            analysis = analyze_pair_detailed(asset_a, asset_b)
            if analysis:
                pair_key = f"{ASSET_SYMBOLS[asset_a]}/{ASSET_SYMBOLS[asset_b]}"
                new_analysis[pair_key] = analysis
                
                rec = generate_detailed_recommendation(analysis)
                if rec:
                    new_recommendations.append(rec)
    
    # Sort recommendations by confidence and z-score
    new_recommendations.sort(key=lambda x: (-x["confidence_score"], -abs(x["z_score"])))
    
    pair_analysis = new_analysis
    detailed_recommendations = new_recommendations[:15]  # Top 15 recommendations
    
    log_event("ANALYSIS", f"Generated {len(detailed_recommendations)} trade recommendations")
    
    return {
        "pairs_analyzed": len(new_analysis),
        "cointegrated_pairs": len([a for a in new_analysis.values() if a["is_cointegrated"]]),
        "recommendations": len(detailed_recommendations),
        "top_recommendation": detailed_recommendations[0] if detailed_recommendations else None
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
    return f"Antigravity v{VERSION} - Advanced Multi-Asset Statistical Arbitrage. Tracks {len(ASSETS)} cryptos, 1-year historical analysis, detailed trade recommendations with timing/ranges/risk:reward."

@query
def get_top_10_prices() -> str:
    return json.dumps({"prices": current_prices, "last_update": last_update_time, "assets_tracked": len(ASSETS)})

@query
def get_detailed_recommendations() -> str:
    """Get detailed trade recommendations with all parameters."""
    return json.dumps({
        "recommendations": detailed_recommendations,
        "count": len(detailed_recommendations),
        "summary": {
            "high_confidence": len([r for r in detailed_recommendations if r["confidence_level"] == "HIGH"]),
            "medium_confidence": len([r for r in detailed_recommendations if r["confidence_level"] == "MEDIUM"]),
            "action_now": len([r for r in detailed_recommendations if r["timing"] == TIMING_NOW]),
            "action_2h": len([r for r in detailed_recommendations if r["timing"] == TIMING_2H])
        },
        "last_analysis": last_update_time
    })

@query
def get_pair_analysis(pair: str) -> str:
    """Get detailed analysis for a specific pair (e.g., 'BTC/ETH')."""
    analysis = pair_analysis.get(pair.upper())
    if analysis:
        return json.dumps(analysis)
    return json.dumps({"error": f"Pair {pair} not found", "available_pairs": list(pair_analysis.keys())})

@query
def get_all_pair_stats() -> str:
    """Get summary statistics for all analyzed pairs."""
    stats = []
    for pair, analysis in pair_analysis.items():
        stats.append({
            "pair": pair,
            "cointegrated": analysis["is_cointegrated"],
            "z_score": analysis["z_score"],
            "correlation": analysis["correlation"],
            "half_life": analysis["half_life_days"]
        })
    stats.sort(key=lambda x: (not x["cointegrated"], -abs(x["z_score"])))
    return json.dumps({"pairs": stats, "total": len(stats)})

@query
def get_portfolio_analysis() -> str:
    """Get comprehensive portfolio analysis."""
    analysis = {"assets": {}, "top_correlations": [], "cointegrated_pairs": [], "market_stats": {}}
    
    total_mcap = 0
    for asset in ASSETS:
        prices = historical_data.get(asset, []) or price_history.get(asset, [])
        if len(prices) >= 10:
            symbol = ASSET_SYMBOLS[asset]
            current = prices[-1]
            returns = calculate_returns(prices)
            
            analysis["assets"][symbol] = {
                "price": current,
                "z_score": round(calculate_z_score(prices), 4),
                "volatility": round(calculate_volatility(prices), 2),
                "max_drawdown": round(calculate_max_drawdown(prices), 2),
                "return_30d": round(((prices[-1] / prices[-30]) - 1) * 100, 2) if len(prices) >= 30 else 0,
                "sharpe_ratio": round(calculate_sharpe_ratio(returns), 2)
            }
    
    # Top correlations
    analysis["top_correlations"] = [
        {"pair": k, "correlation": v["correlation"]} 
        for k, v in sorted(pair_analysis.items(), key=lambda x: -abs(x[1]["correlation"]))[:10]
    ]
    
    # Cointegrated pairs
    analysis["cointegrated_pairs"] = [
        k for k, v in pair_analysis.items() if v["is_cointegrated"]
    ]
    
    analysis["market_stats"] = {
        "total_pairs_analyzed": len(pair_analysis),
        "cointegrated_count": len(analysis["cointegrated_pairs"]),
        "avg_correlation": round(mean([v["correlation"] for v in pair_analysis.values()]), 4) if pair_analysis else 0
    }
    
    return json.dumps(analysis)

@query
def get_asset_history(asset: str) -> str:
    asset_lower = asset.lower()
    symbol_to_id = {v.lower(): k for k, v in ASSET_SYMBOLS.items()}
    asset_id = symbol_to_id.get(asset_lower, asset_lower)
    
    hist = historical_data.get(asset_id, []) or price_history.get(asset_id, [])
    if not hist:
        return json.dumps({"error": f"Asset {asset} not found"})
    
    return json.dumps({
        "asset": ASSET_SYMBOLS.get(asset_id, asset_id.upper()),
        "prices": hist[-100:],
        "total_points": len(hist),
        "current": hist[-1] if hist else None,
        "high": max(hist) if hist else None,
        "low": min(hist) if hist else None,
        "avg": round(mean(hist), 4) if hist else None
    })

@query
def get_state() -> str:
    return json.dumps({
        "version": VERSION,
        "assets_tracked": len(ASSETS),
        "historical_data_points": sum(len(v) for v in historical_data.values()),
        "pairs_analyzed": len(pair_analysis),
        "active_recommendations": len(detailed_recommendations),
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
        user_storage.insert(caller, json.dumps({"principal": caller, "registered_at": ic.time()}))
        log_event("USER", f"New user registered")
    return json.dumps({"success": True, "principal": caller})


# ============================================================================
# Price Fetching
# ============================================================================

@update
def fetch_all_prices() -> Async[str]:
    """Fetch current prices for all assets."""
    global current_prices, last_update_time, price_history
    
    assets_str = ",".join(ASSETS)
    url = f"{COINGECKO_SIMPLE_URL}?ids={assets_str}&vs_currencies=usd"
    
    log_event("API", "Fetching prices...")
    
    http_result: CallResult[HttpResponse] = yield management_canister.http_request(
        {"url": url, "max_response_bytes": 10_000, "method": {"get": None}, "headers": [], "body": None,
         "transform": {"function": (ic.id(), "transform_price_response"), "context": bytes()}}
    ).with_cycles(HTTP_CYCLES)
    
    def handle_success(response: HttpResponse) -> str:
        global current_prices, last_update_time, price_history
        try:
            data = json.loads(response["body"].decode("utf-8"))
            new_prices = {}
            for asset_id in ASSETS:
                if asset_id in data and "usd" in data[asset_id]:
                    price = data[asset_id]["usd"]
                    symbol = ASSET_SYMBOLS[asset_id]
                    new_prices[symbol] = price
                    price_history[asset_id].append(price)
                    if len(price_history[asset_id]) > 500:
                        price_history[asset_id] = price_history[asset_id][-500:]
            
            current_prices = new_prices
            last_update_time = ic.time()
            return json.dumps({"success": True, "prices": new_prices})
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    return match(http_result, {"Ok": handle_success, "Err": lambda e: json.dumps({"success": False, "error": e})})


@update
def fetch_historical_data(asset: str) -> Async[str]:
    """Fetch 1-year historical data for a specific asset."""
    global historical_data
    
    asset_lower = asset.lower()
    symbol_to_id = {v.lower(): k for k, v in ASSET_SYMBOLS.items()}
    asset_id = symbol_to_id.get(asset_lower, asset_lower)
    
    if asset_id not in ASSETS:
        return json.dumps({"success": False, "error": f"Asset {asset} not supported"})
    
    url = f"{COINGECKO_HISTORY_URL}/{asset_id}/market_chart?vs_currency=usd&days=365"
    
    log_event("API", f"Fetching 1-year history for {asset_id}...")
    
    http_result: CallResult[HttpResponse] = yield management_canister.http_request(
        {"url": url, "max_response_bytes": 500_000, "method": {"get": None}, "headers": [], "body": None,
         "transform": {"function": (ic.id(), "transform_price_response"), "context": bytes()}}
    ).with_cycles(HTTP_CYCLES * 2)
    
    def handle_success(response: HttpResponse) -> str:
        global historical_data
        try:
            data = json.loads(response["body"].decode("utf-8"))
            prices = [p[1] for p in data.get("prices", [])]
            historical_data[asset_id] = prices
            log_event("DATA", f"Loaded {len(prices)} historical points for {asset_id}")
            return json.dumps({"success": True, "asset": ASSET_SYMBOLS[asset_id], "data_points": len(prices)})
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    return match(http_result, {"Ok": handle_success, "Err": lambda e: json.dumps({"success": False, "error": e})})


@update
def fetch_and_analyze() -> Async[str]:
    """Fetch prices and run full analysis."""
    global current_prices, last_update_time, price_history
    
    assets_str = ",".join(ASSETS)
    url = f"{COINGECKO_SIMPLE_URL}?ids={assets_str}&vs_currencies=usd"
    
    log_event("AUTO", "Fetch and analyze cycle...")
    
    http_result: CallResult[HttpResponse] = yield management_canister.http_request(
        {"url": url, "max_response_bytes": 10_000, "method": {"get": None}, "headers": [], "body": None,
         "transform": {"function": (ic.id(), "transform_price_response"), "context": bytes()}}
    ).with_cycles(HTTP_CYCLES)
    
    def handle_success(response: HttpResponse) -> str:
        global current_prices, last_update_time, price_history
        try:
            data = json.loads(response["body"].decode("utf-8"))
            for asset_id in ASSETS:
                if asset_id in data and "usd" in data[asset_id]:
                    price = data[asset_id]["usd"]
                    current_prices[ASSET_SYMBOLS[asset_id]] = price
                    price_history[asset_id].append(price)
            
            last_update_time = ic.time()
            
            # Run analysis
            result = run_full_analysis()
            
            return json.dumps({
                "success": True,
                "prices": current_prices,
                "analysis_summary": result,
                "recommendations": detailed_recommendations[:5]
            })
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    return match(http_result, {"Ok": handle_success, "Err": lambda e: json.dumps({"success": False, "error": e})})


@update
def reset_all() -> str:
    global current_prices, price_history, historical_data, last_update_time, recommendations, detailed_recommendations, pair_analysis, execution_logs
    current_prices = {}
    price_history = {asset: [] for asset in ASSETS}
    historical_data = {asset: [] for asset in ASSETS}
    last_update_time = 0
    recommendations = []
    detailed_recommendations = []
    pair_analysis = {}
    execution_logs = []
    return json.dumps({"success": True})
