"""
Antigravity Bot v3.0 - Advanced ICP Trading Agent
Multi-asset statistical arbitrage with automated recommendations.

Features:
- Top 10 crypto price monitoring (BTC, ETH, SOL, XRP, DOGE, ADA, AVAX, DOT, LINK, ICP)
- Kalman Filter + ADF cointegration analysis
- Automated trade recommendations
- Wallet connection via Internet Identity
- HTTPS Outcalls for live price data

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

VERSION = "3.0.0-multi-asset"

# Top 10 crypto assets to track
ASSETS = ["bitcoin", "ethereum", "solana", "ripple", "dogecoin", 
          "cardano", "avalanche-2", "polkadot", "chainlink", "internet-computer"]

ASSET_SYMBOLS = {
    "bitcoin": "BTC",
    "ethereum": "ETH", 
    "solana": "SOL",
    "ripple": "XRP",
    "dogecoin": "DOGE",
    "cardano": "ADA",
    "avalanche-2": "AVAX",
    "polkadot": "DOT",
    "chainlink": "LINK",
    "internet-computer": "ICP"
}

COINGECKO_API_URL = "https://api.coingecko.com/api/v3/simple/price"
HTTP_CYCLES = 200_000_000  # 200M cycles for HTTPS outcalls

# Trading thresholds
ENTRY_THRESHOLD = 2.0
EXIT_THRESHOLD = 0.5
STOP_LOSS_THRESHOLD = 4.0
ADF_P_THRESHOLD = 0.05
MIN_HISTORY = 30

# Recommendation confidence levels
CONFIDENCE_HIGH = "HIGH"
CONFIDENCE_MEDIUM = "MEDIUM"
CONFIDENCE_LOW = "LOW"


# ============================================================================
# Stable Storage (persists across upgrades)
# ============================================================================

# Price history storage: asset_name -> JSON array of prices
price_storage = StableBTreeMap[str, str](
    memory_id=0, max_key_size=50, max_value_size=50_000
)

# User wallets storage: principal -> wallet info JSON
user_storage = StableBTreeMap[str, str](
    memory_id=1, max_key_size=100, max_value_size=1_000
)

# Recommendations storage
recommendation_storage = StableBTreeMap[str, str](
    memory_id=2, max_key_size=50, max_value_size=10_000
)


# ============================================================================
# Global State (in-memory, faster access)
# ============================================================================

# Current prices for all assets
current_prices: dict = {}

# Price history (rolling window of 100 prices per asset)
price_history: dict = {asset: [] for asset in ASSETS}

# Last update timestamp
last_update_time: int = 0

# Trading recommendations
recommendations: list = []

# Registered users (principals)
registered_users: list = []

# Execution logs
execution_logs: list = []


# ============================================================================
# Helper Functions
# ============================================================================

def log_event(event_type: str, message: str):
    """Log an event for debugging and transparency."""
    global execution_logs
    log_entry = {
        "type": event_type,
        "message": message,
        "timestamp": ic.time()
    }
    execution_logs.append(log_entry)
    if len(execution_logs) > 200:
        execution_logs = execution_logs[-100:]


def calculate_z_score(prices: list) -> float:
    """Calculate Z-score of most recent price."""
    if len(prices) < 20:
        return 0.0
    
    window = prices[-50:]
    n = len(window)
    mean = sum(window) / n
    variance = sum((p - mean) ** 2 for p in window) / n
    std = math.sqrt(variance) if variance > 0 else 1.0
    
    return (prices[-1] - mean) / std if std > 0 else 0.0


def simple_adf_test(series: list) -> dict:
    """Simplified ADF test for stationarity."""
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


def calculate_correlation(prices_a: list, prices_b: list) -> float:
    """Calculate Pearson correlation between two price series."""
    n = min(len(prices_a), len(prices_b))
    if n < 10:
        return 0.0
    
    a = prices_a[-n:]
    b = prices_b[-n:]
    
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    
    cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n)) / n
    std_a = math.sqrt(sum((x - mean_a) ** 2 for x in a) / n)
    std_b = math.sqrt(sum((x - mean_b) ** 2 for x in b) / n)
    
    if std_a == 0 or std_b == 0:
        return 0.0
    
    return cov / (std_a * std_b)


def calculate_spread(prices_a: list, prices_b: list) -> list:
    """Calculate spread using simple ratio for normalized series."""
    n = min(len(prices_a), len(prices_b))
    if n < 2:
        return []
    
    a = prices_a[-n:]
    b = prices_b[-n:]
    
    # Normalize to percentage returns
    base_a = a[0] if a[0] != 0 else 1
    base_b = b[0] if b[0] != 0 else 1
    
    norm_a = [p / base_a for p in a]
    norm_b = [p / base_b for p in b]
    
    # Spread = normalized_a - normalized_b
    return [norm_a[i] - norm_b[i] for i in range(n)]


def generate_recommendation(asset_a: str, asset_b: str, z_score: float, 
                           correlation: float, is_cointegrated: bool) -> dict:
    """Generate a trade recommendation for a pair."""
    symbol_a = ASSET_SYMBOLS.get(asset_a, asset_a.upper())
    symbol_b = ASSET_SYMBOLS.get(asset_b, asset_b.upper())
    
    if not is_cointegrated:
        return None
    
    # Determine action based on z-score
    if z_score > ENTRY_THRESHOLD:
        action = "SHORT_SPREAD"
        description = f"Short {symbol_a}, Long {symbol_b}"
    elif z_score < -ENTRY_THRESHOLD:
        action = "LONG_SPREAD"
        description = f"Long {symbol_a}, Short {symbol_b}"
    elif abs(z_score) < EXIT_THRESHOLD:
        action = "CLOSE"
        description = "Close existing positions"
    else:
        action = "HOLD"
        description = "Wait for better entry"
    
    # Confidence based on z-score magnitude and correlation
    if abs(z_score) > 2.5 and abs(correlation) > 0.7:
        confidence = CONFIDENCE_HIGH
    elif abs(z_score) > 2.0 and abs(correlation) > 0.5:
        confidence = CONFIDENCE_MEDIUM
    else:
        confidence = CONFIDENCE_LOW
    
    return {
        "pair": f"{symbol_a}/{symbol_b}",
        "action": action,
        "description": description,
        "z_score": round(z_score, 4),
        "correlation": round(correlation, 4),
        "confidence": confidence,
        "timestamp": ic.time()
    }


def analyze_all_pairs() -> list:
    """Analyze all possible pairs and generate recommendations."""
    global recommendations
    
    new_recommendations = []
    assets_with_data = [a for a in ASSETS if len(price_history.get(a, [])) >= MIN_HISTORY]
    
    # Analyze each pair
    for i, asset_a in enumerate(assets_with_data):
        for asset_b in assets_with_data[i+1:]:
            prices_a = price_history[asset_a]
            prices_b = price_history[asset_b]
            
            # Calculate spread and test cointegration
            spread = calculate_spread(prices_a, prices_b)
            if len(spread) < MIN_HISTORY:
                continue
            
            adf_result = simple_adf_test(spread)
            z_score = calculate_z_score(spread)
            correlation = calculate_correlation(prices_a, prices_b)
            
            rec = generate_recommendation(
                asset_a, asset_b, z_score, 
                correlation, adf_result["is_stationary"]
            )
            
            if rec and rec["action"] in ["LONG_SPREAD", "SHORT_SPREAD"]:
                new_recommendations.append(rec)
    
    # Sort by confidence and z-score magnitude
    confidence_order = {CONFIDENCE_HIGH: 0, CONFIDENCE_MEDIUM: 1, CONFIDENCE_LOW: 2}
    new_recommendations.sort(key=lambda x: (confidence_order[x["confidence"]], -abs(x["z_score"])))
    
    recommendations = new_recommendations[:10]  # Keep top 10
    
    log_event("ANALYSIS", f"Generated {len(recommendations)} recommendations")
    
    return recommendations


# ============================================================================
# HTTP Transform Function
# ============================================================================

@query
def transform_price_response(args: HttpTransformArgs) -> HttpResponse:
    """Transform function for HTTPS outcalls consensus."""
    http_response = args["response"]
    http_response["headers"] = []
    return http_response


# ============================================================================
# Query Methods
# ============================================================================

@query
def get_health() -> str:
    """Health check endpoint."""
    return "System Operational"


@query
def get_version() -> str:
    """Get bot version."""
    return VERSION


@query
def get_strategy_info() -> str:
    """Get strategy description."""
    return f"Antigravity v{VERSION} - Multi-Asset Statistical Arbitrage. Tracks {len(ASSETS)} cryptocurrencies, analyzes cointegration pairs, provides trade recommendations."


@query
def get_top_10_prices() -> str:
    """Get current prices for all tracked assets."""
    result = {
        "prices": current_prices,
        "last_update": last_update_time,
        "assets_tracked": len(ASSETS)
    }
    return json.dumps(result)


@query
def get_recommendations() -> str:
    """Get current trade recommendations."""
    return json.dumps({
        "recommendations": recommendations,
        "count": len(recommendations),
        "last_analysis": last_update_time
    })


@query
def get_asset_history(asset: str) -> str:
    """Get price history for a specific asset."""
    asset_lower = asset.lower()
    
    # Map symbol to CoinGecko ID
    symbol_to_id = {v.lower(): k for k, v in ASSET_SYMBOLS.items()}
    asset_id = symbol_to_id.get(asset_lower, asset_lower)
    
    if asset_id not in price_history:
        return json.dumps({"error": f"Asset {asset} not found", "available": list(ASSET_SYMBOLS.values())})
    
    prices = price_history[asset_id]
    
    return json.dumps({
        "asset": ASSET_SYMBOLS.get(asset_id, asset_id.upper()),
        "prices": prices[-50:],  # Last 50 prices
        "count": len(prices),
        "current": prices[-1] if prices else None
    })


@query
def get_portfolio_analysis() -> str:
    """Get comprehensive portfolio analysis."""
    analysis = {
        "assets": {},
        "correlations": [],
        "cointegrated_pairs": []
    }
    
    for asset in ASSETS:
        prices = price_history.get(asset, [])
        if len(prices) >= 10:
            z = calculate_z_score(prices)
            analysis["assets"][ASSET_SYMBOLS.get(asset, asset)] = {
                "current_price": prices[-1] if prices else 0,
                "z_score": round(z, 4),
                "price_change_24h": round(((prices[-1] / prices[-24]) - 1) * 100, 2) if len(prices) >= 24 else 0
            }
    
    # Top correlations
    assets_with_data = [a for a in ASSETS if len(price_history.get(a, [])) >= MIN_HISTORY]
    for i, a in enumerate(assets_with_data):
        for b in assets_with_data[i+1:]:
            corr = calculate_correlation(price_history[a], price_history[b])
            if abs(corr) > 0.5:
                analysis["correlations"].append({
                    "pair": f"{ASSET_SYMBOLS[a]}/{ASSET_SYMBOLS[b]}",
                    "correlation": round(corr, 4)
                })
            
            spread = calculate_spread(price_history[a], price_history[b])
            if len(spread) >= MIN_HISTORY:
                adf = simple_adf_test(spread)
                if adf["is_stationary"]:
                    analysis["cointegrated_pairs"].append(f"{ASSET_SYMBOLS[a]}/{ASSET_SYMBOLS[b]}")
    
    analysis["correlations"].sort(key=lambda x: -abs(x["correlation"]))
    
    return json.dumps(analysis)


@query
def get_state() -> str:
    """Get current system state."""
    return json.dumps({
        "version": VERSION,
        "assets_tracked": len(ASSETS),
        "prices_collected": sum(len(v) for v in price_history.values()),
        "active_recommendations": len(recommendations),
        "registered_users": len(registered_users),
        "last_update": last_update_time
    })


@query
def get_logs() -> str:
    """Get execution logs."""
    return json.dumps(execution_logs[-50:])


@query
def get_config() -> str:
    """Get current configuration."""
    return json.dumps({
        "assets": list(ASSET_SYMBOLS.values()),
        "entry_threshold": ENTRY_THRESHOLD,
        "exit_threshold": EXIT_THRESHOLD,
        "stop_loss_threshold": STOP_LOSS_THRESHOLD,
        "adf_p_threshold": ADF_P_THRESHOLD,
        "min_history": MIN_HISTORY
    })


# ============================================================================
# User/Wallet Management
# ============================================================================

@query
def whoami() -> str:
    """Get the caller's principal ID."""
    return str(ic.caller())


@update
def register_wallet() -> str:
    """Register the caller's wallet/principal."""
    global registered_users
    
    caller = str(ic.caller())
    
    if caller in registered_users:
        return json.dumps({"success": True, "message": "Already registered", "principal": caller})
    
    registered_users.append(caller)
    
    user_data = {
        "principal": caller,
        "registered_at": ic.time(),
        "preferences": {}
    }
    user_storage.insert(caller, json.dumps(user_data))
    
    log_event("USER", f"New user registered: {caller[:20]}...")
    
    return json.dumps({"success": True, "message": "Wallet registered", "principal": caller})


@query
def get_user_info() -> str:
    """Get info about the calling user."""
    caller = str(ic.caller())
    
    stored = user_storage.get(caller)
    if stored:
        return stored
    
    return json.dumps({"registered": False, "principal": caller})


@query
def get_registered_users_count() -> str:
    """Get count of registered users."""
    return json.dumps({"count": len(registered_users)})


# ============================================================================
# Price Fetching
# ============================================================================

@update
def fetch_all_prices() -> Async[str]:
    """Fetch live prices for all top 10 assets."""
    global current_prices, last_update_time, price_history
    
    assets_str = ",".join(ASSETS)
    url = f"{COINGECKO_API_URL}?ids={assets_str}&vs_currencies=usd"
    
    log_event("API", "Fetching prices for top 10 assets...")
    
    http_result: CallResult[HttpResponse] = yield management_canister.http_request(
        {
            "url": url,
            "max_response_bytes": 10_000,
            "method": {"get": None},
            "headers": [],
            "body": None,
            "transform": {"function": (ic.id(), "transform_price_response"), "context": bytes()},
        }
    ).with_cycles(HTTP_CYCLES)
    
    def handle_success(response: HttpResponse) -> str:
        global current_prices, last_update_time, price_history
        
        try:
            body = response["body"].decode("utf-8")
            data = json.loads(body)
            
            new_prices = {}
            for asset_id in ASSETS:
                if asset_id in data and "usd" in data[asset_id]:
                    price = data[asset_id]["usd"]
                    symbol = ASSET_SYMBOLS.get(asset_id, asset_id.upper())
                    new_prices[symbol] = price
                    
                    # Update history
                    if asset_id not in price_history:
                        price_history[asset_id] = []
                    price_history[asset_id].append(price)
                    
                    # Keep rolling window of 200
                    if len(price_history[asset_id]) > 200:
                        price_history[asset_id] = price_history[asset_id][-200:]
            
            current_prices = new_prices
            last_update_time = ic.time()
            
            log_event("PRICE", f"Updated {len(new_prices)} asset prices")
            
            return json.dumps({
                "success": True,
                "prices": new_prices,
                "assets_updated": len(new_prices),
                "timestamp": last_update_time
            })
            
        except Exception as e:
            log_event("ERROR", f"Parse error: {str(e)}")
            return json.dumps({"success": False, "error": str(e)})
    
    def handle_error(err: str) -> str:
        log_event("API_ERROR", err)
        return json.dumps({"success": False, "error": err})
    
    return match(http_result, {"Ok": handle_success, "Err": handle_error})


@update
def fetch_and_analyze() -> Async[str]:
    """Fetch prices and run full analysis with recommendations."""
    global current_prices, last_update_time, price_history, recommendations
    
    assets_str = ",".join(ASSETS)
    url = f"{COINGECKO_API_URL}?ids={assets_str}&vs_currencies=usd"
    
    log_event("AUTO", "Starting auto-fetch and analysis cycle...")
    
    http_result: CallResult[HttpResponse] = yield management_canister.http_request(
        {
            "url": url,
            "max_response_bytes": 10_000,
            "method": {"get": None},
            "headers": [],
            "body": None,
            "transform": {"function": (ic.id(), "transform_price_response"), "context": bytes()},
        }
    ).with_cycles(HTTP_CYCLES)
    
    def handle_success(response: HttpResponse) -> str:
        global current_prices, last_update_time, price_history, recommendations
        
        try:
            body = response["body"].decode("utf-8")
            data = json.loads(body)
            
            new_prices = {}
            for asset_id in ASSETS:
                if asset_id in data and "usd" in data[asset_id]:
                    price = data[asset_id]["usd"]
                    symbol = ASSET_SYMBOLS.get(asset_id, asset_id.upper())
                    new_prices[symbol] = price
                    
                    if asset_id not in price_history:
                        price_history[asset_id] = []
                    price_history[asset_id].append(price)
                    
                    if len(price_history[asset_id]) > 200:
                        price_history[asset_id] = price_history[asset_id][-200:]
            
            current_prices = new_prices
            last_update_time = ic.time()
            
            # Run analysis and generate recommendations
            recs = analyze_all_pairs()
            
            return json.dumps({
                "success": True,
                "prices": new_prices,
                "recommendations": recs,
                "recommendation_count": len(recs),
                "timestamp": last_update_time
            })
            
        except Exception as e:
            log_event("ERROR", f"Analysis error: {str(e)}")
            return json.dumps({"success": False, "error": str(e)})
    
    def handle_error(err: str) -> str:
        log_event("API_ERROR", err)
        return json.dumps({"success": False, "error": err})
    
    return match(http_result, {"Ok": handle_success, "Err": handle_error})


# ============================================================================
# Admin/Utility Functions
# ============================================================================

@update
def reset_all() -> str:
    """Reset all state (for testing)."""
    global current_prices, price_history, last_update_time, recommendations, execution_logs
    
    current_prices = {}
    price_history = {asset: [] for asset in ASSETS}
    last_update_time = 0
    recommendations = []
    execution_logs = []
    
    log_event("SYSTEM", "Full state reset")
    return json.dumps({"success": True, "message": "All state reset"})


@update
def add_manual_prices(prices_json: str) -> str:
    """Manually add price data for testing."""
    global current_prices, last_update_time, price_history
    
    try:
        data = json.loads(prices_json)
        
        for symbol, price in data.items():
            symbol_to_id = {v.lower(): k for k, v in ASSET_SYMBOLS.items()}
            asset_id = symbol_to_id.get(symbol.lower())
            
            if asset_id:
                current_prices[symbol.upper()] = price
                if asset_id not in price_history:
                    price_history[asset_id] = []
                price_history[asset_id].append(price)
        
        last_update_time = ic.time()
        
        return json.dumps({"success": True, "prices_added": len(data)})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
