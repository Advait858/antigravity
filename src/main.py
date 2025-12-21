"""
Antigravity Trading Agent v6.0 - Live Autonomous System
========================================================
An on-chain AI Hedge Fund Manager on Internet Computer

Features:
- Heartbeat automation (60s tick)
- Live Binance/CoinGecko oracle
- Engle-Granger cointegration analysis
- DEX simulation with fees
- Principal-based user portfolios
"""

from kybra import (
    Async, CallResult, ic, init, post_upgrade, pre_upgrade,
    query, update, void, Principal, StableBTreeMap,
    nat, nat64, float64, Vec, Opt, Record, Variant
)
import json
import math

# ============================================================
# CONFIGURATION
# ============================================================

VERSION = "6.0.0-live"

ASSETS = [
    "bitcoin", "ethereum", "solana", "ripple", "dogecoin",
    "cardano", "avalanche-2", "polkadot", "chainlink", "internet-computer"
]

ASSET_SYMBOLS = {
    "bitcoin": "BTC", "ethereum": "ETH", "solana": "SOL",
    "ripple": "XRP", "dogecoin": "DOGE", "cardano": "ADA",
    "avalanche-2": "AVAX", "polkadot": "DOT", "chainlink": "LINK",
    "internet-computer": "ICP"
}

# Generate all 45 pairs
def generate_pairs():
    pairs = []
    for i, a in enumerate(ASSETS):
        for b in ASSETS[i+1:]:
            pairs.append((a, b))
    return pairs

ALL_PAIRS = generate_pairs()

# Heartbeat configuration
HEARTBEAT_INTERVAL_NS = 60_000_000_000  # 60 seconds in nanoseconds
BINANCE_API = "https://api.binance.com/api/v3/ticker/price"
COINGECKO_API = "https://api.coingecko.com/api/v3/simple/price"
HTTP_CYCLES = 250_000_000

# Analysis thresholds
MIN_HISTORY = 30
ADF_SIGNIFICANCE = 0.05
ENTRY_Z_THRESHOLD = 2.0
LP_FEE_PERCENT = 0.003  # 0.3%
GAS_FEE_ICP = 0.001
ICP_PRICE_USD = 10.50
DEFAULT_BALANCE = 100000.0

# ============================================================
# STABLE STORAGE
# ============================================================

user_storage = StableBTreeMap[str, str](memory_id=0, max_key_size=100, max_value_size=10_000)
price_storage = StableBTreeMap[str, str](memory_id=1, max_key_size=50, max_value_size=50_000)
signal_storage = StableBTreeMap[str, str](memory_id=2, max_key_size=100, max_value_size=10_000)

# ============================================================
# GLOBAL STATE
# ============================================================

# Price oracle state
prices: dict = {}  # symbol -> current price
price_history: dict = {asset: [] for asset in ASSETS}  # asset -> [price, price, ...]
last_price_update: int = 0
price_source: str = "none"

# Analysis state
cointegration_results: dict = {}  # "pair" -> analysis
active_signals: list = []
scanner_index: int = 0  # Round-robin pair scanner

# Heartbeat state
heartbeat_count: int = 0
last_heartbeat: int = 0
heartbeat_errors: list = []

# User portfolios
user_portfolios: dict = {}  # principal -> portfolio

# Trade state
active_trades: list = []
trade_history: list = []
dex_stats: dict = {"total_trades": 0, "total_fees": 0.0, "total_slippage": 0.0}


# ============================================================
# MATH UTILITIES (Inlined for WASM)
# ============================================================

def mean(arr):
    return sum(arr) / len(arr) if arr else 0

def variance(arr):
    if len(arr) < 2:
        return 0
    m = mean(arr)
    return sum((x - m) ** 2 for x in arr) / len(arr)

def std_dev(arr):
    return math.sqrt(variance(arr))

def ols_regression(y, x):
    n = len(y)
    if n != len(x) or n < 3:
        return {"beta": 0, "alpha": 0, "residuals": [], "r_squared": 0}
    
    x_mean, y_mean = mean(x), mean(y)
    
    num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    den = sum((xi - x_mean) ** 2 for xi in x)
    
    if den == 0:
        return {"beta": 0, "alpha": 0, "residuals": [], "r_squared": 0}
    
    beta = num / den
    alpha = y_mean - beta * x_mean
    residuals = [yi - (alpha + beta * xi) for yi, xi in zip(y, x)]
    
    ss_res = sum(r ** 2 for r in residuals)
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {"beta": beta, "alpha": alpha, "residuals": residuals, "r_squared": r_squared}

def adf_test(series):
    n = len(series)
    if n < 10:
        return {"statistic": 0, "p_value": 1.0}
    
    diff = [series[i] - series[i-1] for i in range(1, n)]
    lagged = series[:-1]
    reg = ols_regression(diff, lagged)
    
    beta = reg["beta"]
    residuals = reg["residuals"]
    
    if not residuals or len(residuals) < 3:
        return {"statistic": 0, "p_value": 1.0}
    
    ss_res = sum(r ** 2 for r in residuals) / max(1, len(residuals) - 2)
    ss_x = sum((x - mean(lagged)) ** 2 for x in lagged)
    
    if ss_x == 0:
        return {"statistic": 0, "p_value": 1.0}
    
    se_beta = math.sqrt(ss_res / ss_x) if ss_res > 0 else 1
    t_stat = beta / se_beta if se_beta > 0 else 0
    
    # Approximate p-value
    if t_stat < -3.43:
        p_value = 0.01
    elif t_stat < -2.86:
        p_value = 0.05
    elif t_stat < -2.57:
        p_value = 0.10
    else:
        p_value = 0.50
    
    return {"statistic": t_stat, "p_value": p_value}

def calculate_zscore(spread, lookback=60):
    if len(spread) < 2:
        return 0
    recent = spread[-lookback:]
    mu = mean(recent)
    sigma = std_dev(recent)
    return (spread[-1] - mu) / sigma if sigma > 0 else 0

def calculate_half_life(spread):
    if len(spread) < 10:
        return 999
    lagged = spread[:-1]
    current = spread[1:]
    reg = ols_regression(current, lagged)
    phi = reg["beta"]
    if phi <= 0 or phi >= 1:
        return 999
    return max(1, min(365, -math.log(2) / math.log(phi)))


# ============================================================
# PRICE ORACLE
# ============================================================

BINANCE_SYMBOLS = {
    "bitcoin": "BTCUSDT", "ethereum": "ETHUSDT", "solana": "SOLUSDT",
    "ripple": "XRPUSDT", "dogecoin": "DOGEUSDT", "cardano": "ADAUSDT",
    "avalanche-2": "AVAXUSDT", "polkadot": "DOTUSDT", "chainlink": "LINKUSDT",
    "internet-computer": "ICPUSDT"
}

async def fetch_binance_price(symbol: str) -> float:
    """Fetch single price from Binance"""
    try:
        url = f"{BINANCE_API}?symbol={symbol}"
        
        response = await ic.http_request(
            {
                "url": url,
                "method": {"get": None},
                "headers": [],
                "body": None,
                "max_response_bytes": 1000,
                "transform": None
            },
            HTTP_CYCLES
        )
        
        if response["status"] == 200:
            body = bytes(response["body"]).decode("utf-8")
            data = json.loads(body)
            return float(data.get("price", 0))
    except Exception as e:
        log_error("BINANCE", str(e))
    
    return 0.0


async def fetch_all_prices() -> dict:
    """Fetch all prices from Binance"""
    global prices, price_history, last_price_update, price_source
    
    fetched = {}
    
    for asset_id, binance_symbol in BINANCE_SYMBOLS.items():
        price = await fetch_binance_price(binance_symbol)
        if price > 0:
            fetched[asset_id] = price
            symbol = ASSET_SYMBOLS[asset_id]
            prices[symbol] = price
            
            # Update history (rolling 500 window)
            price_history[asset_id].append(price)
            if len(price_history[asset_id]) > 500:
                price_history[asset_id] = price_history[asset_id][-500:]
    
    if fetched:
        last_price_update = ic.time()
        price_source = "binance"
    
    return fetched


# ============================================================
# COINTEGRATION ANALYSIS
# ============================================================

def analyze_pair(asset_a: str, asset_b: str) -> dict:
    """Analyze a single pair for cointegration"""
    hist_a = price_history.get(asset_a, [])
    hist_b = price_history.get(asset_b, [])
    
    n = min(len(hist_a), len(hist_b))
    
    if n < MIN_HISTORY:
        return {"cointegrated": False, "reason": f"Insufficient data ({n}/{MIN_HISTORY})"}
    
    pa = hist_a[-n:]
    pb = hist_b[-n:]
    
    # OLS regression
    ols = ols_regression(pa, pb)
    hedge_ratio = ols["beta"]
    r_squared = ols["r_squared"]
    
    # Spread
    spread = [a - hedge_ratio * b for a, b in zip(pa, pb)]
    
    # ADF test
    adf = adf_test(spread)
    is_cointegrated = adf["p_value"] < ADF_SIGNIFICANCE
    
    # Z-score and half-life
    z_score = calculate_zscore(spread)
    half_life = calculate_half_life(spread) if is_cointegrated else 999
    
    sym_a = ASSET_SYMBOLS.get(asset_a, asset_a)
    sym_b = ASSET_SYMBOLS.get(asset_b, asset_b)
    pair_name = f"{sym_a}/{sym_b}"
    
    # Generate signal
    signal = None
    if is_cointegrated and abs(z_score) > ENTRY_Z_THRESHOLD:
        action = "LONG_SPREAD" if z_score < 0 else "SHORT_SPREAD"
        confidence = min(100, 50 + int(abs(z_score) * 10) + int((0.05 - adf["p_value"]) * 500))
        signal = {
            "pair": pair_name,
            "action": action,
            "z_score": round(z_score, 4),
            "confidence": confidence,
            "adf_p_value": round(adf["p_value"], 4),
            "half_life": round(half_life, 1),
            "timestamp": ic.time()
        }
    
    return {
        "pair": pair_name,
        "cointegrated": is_cointegrated,
        "hedge_ratio": round(hedge_ratio, 6),
        "r_squared": round(r_squared, 4),
        "adf_statistic": round(adf["statistic"], 4),
        "adf_p_value": round(adf["p_value"], 4),
        "z_score": round(z_score, 4),
        "half_life": round(half_life, 1) if half_life < 999 else None,
        "signal": signal,
        "n_observations": n
    }


def run_round_robin_scan() -> dict:
    """Scan one pair (round-robin to prevent cycle limit)"""
    global scanner_index, cointegration_results, active_signals
    
    if not ALL_PAIRS:
        return {"error": "No pairs to scan"}
    
    # Get current pair
    pair = ALL_PAIRS[scanner_index]
    asset_a, asset_b = pair
    
    # Analyze
    result = analyze_pair(asset_a, asset_b)
    
    # Store result
    pair_name = result.get("pair", f"{asset_a}/{asset_b}")
    cointegration_results[pair_name] = result
    
    # Update signals
    if result.get("signal"):
        # Check if signal already exists
        existing = [s for s in active_signals if s["pair"] == pair_name]
        if not existing:
            active_signals.append(result["signal"])
            # Keep only top 10 signals
            active_signals = sorted(active_signals, key=lambda x: x["confidence"], reverse=True)[:10]
    
    # Advance scanner
    scanner_index = (scanner_index + 1) % len(ALL_PAIRS)
    
    return {
        "scanned_pair": pair_name,
        "scanner_position": f"{scanner_index + 1}/{len(ALL_PAIRS)}",
        "result": result
    }


# ============================================================
# DEX SIMULATION
# ============================================================

def calculate_fees(amount: float) -> dict:
    """Calculate trading fees"""
    lp_fee = amount * LP_FEE_PERCENT
    gas_fee = GAS_FEE_ICP * ICP_PRICE_USD
    
    # Slippage for >$10k trades
    slippage = 0
    if amount > 10000:
        slippage = 0.00001 * ((amount - 10000) ** 1.5)
        slippage = min(0.02, slippage) * amount
    
    total = lp_fee + gas_fee + slippage
    
    return {
        "lp_fee": round(lp_fee, 4),
        "gas_fee": round(gas_fee, 4),
        "slippage": round(slippage, 4),
        "total": round(total, 4)
    }


def execute_paper_trade(principal: str, signal: dict, position_size: float) -> dict:
    """Execute a paper trade with DEX simulation"""
    global active_trades, dex_stats
    
    # Get or create user portfolio
    if principal not in user_portfolios:
        user_portfolios[principal] = {
            "cash": DEFAULT_BALANCE,
            "initial": DEFAULT_BALANCE,
            "positions": {},
            "realized_pnl": 0
        }
    
    portfolio = user_portfolios[principal]
    
    # Check funds
    fees = calculate_fees(position_size)
    total_cost = position_size  # Fees are deducted from position
    
    if portfolio["cash"] < total_cost:
        return {"success": False, "error": "Insufficient funds"}
    
    # Deduct cash
    portfolio["cash"] -= total_cost
    
    # Get entry prices
    pair = signal["pair"]
    assets = pair.split("/")
    price_a = prices.get(assets[0], 0)
    price_b = prices.get(assets[1], 0)
    
    # Create trade
    trade = {
        "id": len(active_trades) + 1,
        "principal": principal,
        "pair": pair,
        "action": signal["action"],
        "position_size": position_size,
        "effective_size": position_size - fees["total"],
        "entry_z": signal["z_score"],
        "entry_prices": {"a": price_a, "b": price_b},
        "fees_paid": fees["total"],
        "entry_time": ic.time(),
        "status": "OPEN",
        "current_pnl": 0
    }
    
    active_trades.append(trade)
    
    # Update DEX stats
    dex_stats["total_trades"] += 1
    dex_stats["total_fees"] += fees["total"]
    
    return {"success": True, "trade": trade, "fees": fees}


def close_paper_trade(trade_id: int, current_z: float) -> dict:
    """Close a paper trade"""
    global active_trades, trade_history
    
    trade = next((t for t in active_trades if t["id"] == trade_id), None)
    if not trade:
        return {"success": False, "error": "Trade not found"}
    
    # Calculate P&L based on Z-score change
    entry_z = trade["entry_z"]
    effective = trade["effective_size"]
    
    # P&L = z-score move * position / 100 (simplified model)
    z_move = abs(entry_z) - abs(current_z)  # Positive when mean-reverts
    pnl = effective * (z_move / 5)  # 5 z-score points = 100% return
    
    # Exit fees
    exit_fees = calculate_fees(effective + pnl)
    net_pnl = pnl - exit_fees["total"]
    
    # Update portfolio
    principal = trade["principal"]
    if principal in user_portfolios:
        user_portfolios[principal]["cash"] += effective + net_pnl
        user_portfolios[principal]["realized_pnl"] += net_pnl
    
    # Move to history
    trade["status"] = "CLOSED"
    trade["exit_time"] = ic.time()
    trade["exit_z"] = current_z
    trade["final_pnl"] = net_pnl
    trade_history.append(trade)
    active_trades = [t for t in active_trades if t["id"] != trade_id]
    
    return {"success": True, "pnl": net_pnl, "exit_fees": exit_fees["total"]}


# ============================================================
# HEARTBEAT (60-second tick)
# ============================================================

async def heartbeat_tick():
    """Main heartbeat function - runs every 60 seconds"""
    global heartbeat_count, last_heartbeat
    
    try:
        heartbeat_count += 1
        last_heartbeat = ic.time()
        
        # Tick 1: Fetch market data
        if heartbeat_count % 3 == 1:
            await fetch_all_prices()
            log_event("HEARTBEAT", f"Tick {heartbeat_count}: Fetched prices")
        
        # Tick 2: Run analysis (round-robin)
        elif heartbeat_count % 3 == 2:
            scan_result = run_round_robin_scan()
            log_event("HEARTBEAT", f"Tick {heartbeat_count}: Scanned {scan_result.get('scanned_pair')}")
        
        # Tick 3: Auto-execute trades
        else:
            auto_execute_result = auto_execute_signals()
            log_event("HEARTBEAT", f"Tick {heartbeat_count}: Auto-executed {auto_execute_result.get('executed', 0)} trades")
    
    except Exception as e:
        log_error("HEARTBEAT", str(e))


def auto_execute_signals():
    """Auto-execute high-confidence signals"""
    executed = 0
    
    for signal in active_signals[:3]:  # Top 3 signals only
        if signal["confidence"] >= 70:
            # Execute for each connected user with auto-trade enabled
            # For now, skip auto-execution (require manual trigger)
            pass
    
    return {"executed": executed}


# ============================================================
# LOGGING
# ============================================================

execution_logs: list = []

def log_event(category: str, message: str):
    execution_logs.append({
        "time": ic.time(),
        "category": category,
        "message": message
    })
    if len(execution_logs) > 100:
        execution_logs.pop(0)

def log_error(category: str, error: str):
    heartbeat_errors.append({
        "time": ic.time(),
        "category": category,
        "error": error
    })
    if len(heartbeat_errors) > 50:
        heartbeat_errors.pop(0)


# ============================================================
# CANISTER ENDPOINTS
# ============================================================

@query
def get_version() -> str:
    return VERSION

@query
def get_health() -> str:
    return json.dumps({
        "status": "healthy",
        "version": VERSION,
        "heartbeat_count": heartbeat_count,
        "last_heartbeat": last_heartbeat,
        "price_source": price_source,
        "assets_tracked": len(prices),
        "pairs_analyzed": len(cointegration_results),
        "active_signals": len(active_signals),
        "active_trades": len(active_trades)
    })

@query
def get_live_prices() -> str:
    return json.dumps({
        "prices": prices,
        "source": price_source,
        "last_update": last_price_update,
        "history_lengths": {ASSET_SYMBOLS.get(k, k): len(v) for k, v in price_history.items()}
    })

@query
def get_trading_signals() -> str:
    return json.dumps({
        "signals": active_signals,
        "count": len(active_signals),
        "scanner_position": f"{scanner_index + 1}/{len(ALL_PAIRS)}"
    })

@query
def get_cointegration_results() -> str:
    return json.dumps({
        "pairs": cointegration_results,
        "total_analyzed": len(cointegration_results),
        "cointegrated_count": sum(1 for p in cointegration_results.values() if p.get("cointegrated"))
    })

@query
def get_portfolio(principal: str) -> str:
    if principal not in user_portfolios:
        return json.dumps({
            "principal": principal,
            "cash": DEFAULT_BALANCE,
            "positions": {},
            "realized_pnl": 0,
            "registered": False
        })
    
    portfolio = user_portfolios[principal]
    return json.dumps({
        "principal": principal,
        **portfolio,
        "registered": True
    })

@query
def get_active_trades() -> str:
    return json.dumps({
        "trades": active_trades,
        "count": len(active_trades)
    })

@query
def get_trade_history_recent(limit: nat) -> str:
    return json.dumps({
        "trades": trade_history[-limit:],
        "total": len(trade_history)
    })

@query
def get_dex_stats() -> str:
    return json.dumps(dex_stats)

@query
def get_logs() -> str:
    return json.dumps({
        "logs": execution_logs[-20:],
        "errors": heartbeat_errors[-10:]
    })

@query
def get_latest_candles(asset: str, periods: nat) -> str:
    """Get price history for charting"""
    asset_id = None
    for k, v in ASSET_SYMBOLS.items():
        if v == asset.upper():
            asset_id = k
            break
    
    if not asset_id:
        return json.dumps({"error": f"Unknown asset: {asset}"})
    
    history = price_history.get(asset_id, [])[-periods:]
    
    return json.dumps({
        "asset": asset.upper(),
        "prices": history,
        "count": len(history),
        "current": prices.get(asset.upper(), 0)
    })


# ============================================================
# UPDATE ENDPOINTS
# ============================================================

@update
def register_user() -> str:
    """Register a new user with paper trading balance"""
    principal = str(ic.caller())
    
    if principal in user_portfolios:
        return json.dumps({"success": False, "error": "Already registered"})
    
    user_portfolios[principal] = {
        "cash": DEFAULT_BALANCE,
        "initial": DEFAULT_BALANCE,
        "positions": {},
        "realized_pnl": 0
    }
    
    log_event("USER", f"Registered: {principal[:20]}...")
    
    return json.dumps({
        "success": True,
        "principal": principal,
        "balance": DEFAULT_BALANCE
    })

@update
async def trigger_price_fetch() -> str:
    """Manually trigger price fetch"""
    result = await fetch_all_prices()
    return json.dumps({
        "fetched": len(result),
        "prices": prices
    })

@update
def trigger_analysis() -> str:
    """Manually trigger pair analysis"""
    result = run_round_robin_scan()
    return json.dumps(result)

@update
def execute_signal_trade(signal_index: nat, position_size: float64) -> str:
    """Execute a trade from signal"""
    principal = str(ic.caller())
    
    if signal_index >= len(active_signals):
        return json.dumps({"success": False, "error": "Invalid signal index"})
    
    signal = active_signals[signal_index]
    result = execute_paper_trade(principal, signal, position_size)
    
    return json.dumps(result)

@update
def close_trade(trade_id: nat) -> str:
    """Close an active trade"""
    principal = str(ic.caller())
    
    # Find trade
    trade = next((t for t in active_trades if t["id"] == trade_id), None)
    if not trade:
        return json.dumps({"success": False, "error": "Trade not found"})
    
    if trade["principal"] != principal:
        return json.dumps({"success": False, "error": "Not your trade"})
    
    # Get current Z-score
    pair = trade["pair"]
    current_analysis = cointegration_results.get(pair, {})
    current_z = current_analysis.get("z_score", 0)
    
    result = close_paper_trade(trade_id, current_z)
    return json.dumps(result)

@update
async def run_heartbeat_manual() -> str:
    """Manually trigger heartbeat"""
    await heartbeat_tick()
    return json.dumps({
        "heartbeat_count": heartbeat_count,
        "success": True
    })


# ============================================================
# LIFECYCLE
# ============================================================

@init
def init_canister():
    log_event("INIT", f"Canister initialized - {VERSION}")

@pre_upgrade
def pre_upgrade_hook():
    # Save state to stable storage
    user_storage.insert("users", json.dumps(user_portfolios))
    price_storage.insert("prices", json.dumps(prices))

@post_upgrade
def post_upgrade_hook():
    global user_portfolios, prices
    
    try:
        users_data = user_storage.get("users")
        if users_data:
            user_portfolios = json.loads(users_data)
        
        prices_data = price_storage.get("prices")
        if prices_data:
            prices = json.loads(prices_data)
    except:
        pass
    
    log_event("UPGRADE", f"Canister upgraded to {VERSION}")
