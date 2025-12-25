"""
Antigravity AI Trading Agent - Full Version
Includes: Cointegration Analysis, OLS Regression, ADF Tests, Z-Score Signals
Uses Groq (Llama 3.3) for AI analysis + ICP Canister for execution
"""

import os
import json
import requests
import subprocess
import math
from datetime import datetime
from collections import defaultdict

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CANISTER_ID = os.getenv("CANISTER_ID", "uxrrr-q7777-77774-qaaaq-cai")

# ============================================================
# MATH CORE (From original canister)
# ============================================================

EPSILON = 1e-10
ADF_CRITICAL_5PCT = -2.86

def mean(data):
    if not data: return 0.0
    return sum(data) / len(data)

def variance(data):
    n = len(data)
    if n < 2: return 0.0
    mu = mean(data)
    return sum((x - mu) ** 2 for x in data) / (n - 1)

def std_dev(data):
    return math.sqrt(variance(data))

def covariance(x, y):
    n = len(x)
    if n != len(y) or n < 2: return 0.0
    mu_x, mu_y = mean(x), mean(y)
    return sum((xi - mu_x) * (yi - mu_y) for xi, yi in zip(x, y)) / (n - 1)

def correlation(x, y):
    sx, sy = std_dev(x), std_dev(y)
    if sx < EPSILON or sy < EPSILON: return 0.0
    return covariance(x, y) / (sx * sy)

class OLSResult:
    def __init__(self, beta, alpha, r_squared, residuals):
        self.beta = beta
        self.alpha = alpha
        self.r_squared = r_squared
        self.residuals = residuals

def ols_regression(y, x):
    """OLS Regression: Y = alpha + beta*X"""
    n = len(y)
    if n != len(x) or n < 2:
        return OLSResult(0, 0, 0, [])
    
    x_mean, y_mean = mean(x), mean(y)
    num = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
    den = sum((x[i] - x_mean) ** 2 for i in range(n))
    
    if den < EPSILON:
        return OLSResult(0, 0, 0, [])
    
    beta = num / den
    alpha = y_mean - beta * x_mean
    
    residuals = [y[i] - (alpha + beta * x[i]) for i in range(n)]
    ss_res = sum(r**2 for r in residuals)
    ss_tot = sum((y[i] - y_mean)**2 for i in range(n))
    r_sq = 1 - (ss_res / ss_tot) if ss_tot > EPSILON else 0
    
    return OLSResult(beta, alpha, r_sq, residuals)

def adf_test(series):
    """Augmented Dickey-Fuller Test for stationarity"""
    n = len(series)
    if n < 10:
        return {"t_stat": 0, "p_value": 1.0, "is_stationary": False}
    
    dy = [series[i] - series[i-1] for i in range(1, n)]
    x_lag = [series[i-1] for i in range(1, n)]
    
    res = ols_regression(dy, x_lag)
    gamma = res.beta
    
    sum_res_sq = sum(r*r for r in res.residuals)
    mean_x = mean(x_lag)
    sum_sq_x = sum((v - mean_x)**2 for v in x_lag)
    
    if sum_sq_x < EPSILON:
        return {"t_stat": 0, "p_value": 1.0, "is_stationary": False}
    
    sigma_sq = sum_res_sq / max(len(x_lag) - 2, 1)
    se = math.sqrt(sigma_sq / sum_sq_x) if sigma_sq > 0 else 0
    t_stat = gamma / se if se > EPSILON else 0
    
    # Approximate p-value
    p_value = 0.01 if t_stat < -3.43 else (0.05 if t_stat < -2.86 else (0.10 if t_stat < -2.57 else 1.0))
    
    return {"t_stat": t_stat, "p_value": p_value, "is_stationary": p_value < 0.05}

def z_score(current, history, window=30):
    """Calculate Z-Score"""
    if len(history) < 2: return 0
    data = history[-window:]
    mu, sigma = mean(data), std_dev(data)
    return (current - mu) / sigma if sigma > EPSILON else 0

# ============================================================
# ADVANCED COINTEGRATION STATISTICS
# ============================================================

def calculate_half_life(spread):
    """Calculate half-life of mean reversion (in days)"""
    if len(spread) < 10:
        return 999
    
    # Regress spread(t) - spread(t-1) on spread(t-1)
    dy = [spread[i] - spread[i-1] for i in range(1, len(spread))]
    y_lag = spread[:-1]
    
    res = ols_regression(dy, y_lag)
    
    if res.beta >= 0 or abs(res.beta) < EPSILON:
        return 999  # No mean reversion
    
    half_life = -math.log(2) / res.beta
    return max(1, min(half_life, 999))

def calculate_hurst_exponent(series, max_lag=20):
    """Calculate Hurst Exponent (H < 0.5 = mean-reverting, H > 0.5 = trending)"""
    n = len(series)
    if n < max_lag * 2:
        return 0.5
    
    lags = range(2, min(max_lag, n // 4))
    tau = []
    rs = []
    
    for lag in lags:
        # Calculate variance of lagged differences
        diffs = [series[i] - series[i-lag] for i in range(lag, n)]
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

def calculate_spread_stats(spread):
    """Calculate comprehensive spread statistics"""
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

def get_sandwich_signal(spread_stats, z):
    """Determine sandwich trading strategy based on spread position"""
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

# ============================================================
# MARKET DATA - BINANCE HISTORICAL OHLCV
# ============================================================

ASSETS = ["BTC", "ETH", "ICP", "SOL", "XRP", "AVAX", "DOT", "LINK", "ADA", "DOGE"]
PAIRS = [(ASSETS[i], ASSETS[j]) for i in range(len(ASSETS)) for j in range(i+1, len(ASSETS))]

# Price history storage
price_history = {}

def fetch_historical_data(asset, days=365):
    """Fetch 1 year of daily OHLCV data from Binance"""
    print(f"   Fetching {days} days of {asset} data...")
    try:
        # Binance klines endpoint
        url = f"https://api.binance.com/api/v3/klines"
        params = {
            "symbol": f"{asset}USDT",
            "interval": "1d",  # Daily candles
            "limit": min(days, 1000)  # Max 1000 per request
        }
        resp = requests.get(url, params=params, timeout=30)
        
        if resp.status_code == 200:
            klines = resp.json()
            # Extract close prices
            closes = [float(k[4]) for k in klines]  # Index 4 = close price
            volumes = [float(k[5]) for k in klines]  # Index 5 = volume
            return {
                "closes": closes,
                "volumes": volumes,
                "count": len(closes)
            }
    except Exception as e:
        print(f"   Error fetching {asset}: {e}")
    return {"closes": [], "volumes": [], "count": 0}

def fetch_all_historical():
    """Fetch historical data for all assets"""
    print("\n📊 FETCHING 1 YEAR OF HISTORICAL DATA...")
    for asset in ASSETS:
        data = fetch_historical_data(asset)
        price_history[asset] = data["closes"]
        print(f"   ✅ {asset}: {data['count']} daily candles")
    
    # Calculate total data points
    total = sum(len(v) for v in price_history.values())
    print(f"\n   📈 Total: {total} data points across {len(ASSETS)} assets")
    print(f"   📊 Pairs to analyze: {len(PAIRS)}")

def fetch_current_prices():
    """Fetch current prices from Binance"""
    prices = {}
    for asset in ASSETS:
        try:
            resp = requests.get(
                f"https://api.binance.com/api/v3/ticker/price?symbol={asset}USDT",
                timeout=10
            )
            if resp.status_code == 200:
                prices[asset] = float(resp.json()["price"])
        except:
            pass
    return prices

# ============================================================
# COINTEGRATION ANALYSIS
# ============================================================

def calculate_risk_score(z_score, adf_p_value, r_squared):
    """Calculate risk percentage (0-100) based on statistical metrics"""
    # Lower z-score = lower risk (closer to mean)
    z_risk = min(abs(z_score) * 20, 50)  # 0-50 from z-score
    
    # Lower p-value = lower risk (more stationary)
    p_risk = adf_p_value * 30  # 0-30 from p-value
    
    # Higher R² = lower risk (better fit)
    r_risk = (1 - r_squared) * 20  # 0-20 from R²
    
    return round(z_risk + p_risk + r_risk, 1)

def analyze_pair(asset_a, asset_b):
    """Analyze a pair for cointegration with full statistics suite"""
    hist_a = price_history.get(asset_a, [])
    hist_b = price_history.get(asset_b, [])
    
    n = min(len(hist_a), len(hist_b))
    if n < 30:
        return None
    
    pa = hist_a[-n:]
    pb = hist_b[-n:]
    
    # OLS: pa = alpha + beta * pb
    ols = ols_regression(pa, pb)
    
    # Calculate spread
    spread = [pa[i] - (ols.alpha + ols.beta * pb[i]) for i in range(n)]
    
    # ADF test on spread
    adf = adf_test(spread)
    
    # Z-Score of current spread
    current_z = z_score(spread[-1], spread, 30)
    
    # Correlation
    corr = correlation(pa, pb)
    
    # ADVANCED STATS
    half_life = calculate_half_life(spread)
    hurst = calculate_hurst_exponent(spread)
    spread_stats = calculate_spread_stats(spread)
    sandwich = get_sandwich_signal(spread_stats, current_z)
    
    # Calculate risk score
    risk = calculate_risk_score(current_z, adf["p_value"], ols.r_squared)
    
    # Trading signal (only if stationary AND z-score extreme)
    signal = None
    if adf["is_stationary"]:
        if current_z < -2.0:
            signal = "LONG_SPREAD"
        elif current_z > 2.0:
            signal = "SHORT_SPREAD"
    
    return {
        "pair": f"{asset_a}-{asset_b}",
        "signal": signal,
        "z_score": round(current_z, 2),
        "adf_p_value": round(adf["p_value"], 4),
        "adf_t_stat": round(adf["t_stat"], 2),
        "is_stationary": adf["is_stationary"],
        "correlation": round(corr, 3),
        "beta": round(ols.beta, 4),
        "r_squared": round(ols.r_squared, 3),
        "risk_pct": risk,
        # Advanced stats
        "half_life_days": round(half_life, 1),
        "hurst_exponent": round(hurst, 3),
        "spread_percentile": spread_stats.get("percentile", 50),
        "spread_volatility": spread_stats.get("volatility_30d", 0),
        "bollinger_upper": spread_stats.get("upper_band", 0),
        "bollinger_lower": spread_stats.get("lower_band", 0),
        "sandwich_strategy": sandwich
    }

def scan_all_pairs():
    """Scan all 45 pairs and return analysis results"""
    all_results = []
    strong_signals = []
    
    for asset_a, asset_b in PAIRS:
        result = analyze_pair(asset_a, asset_b)
        if result:
            all_results.append(result)
            if result["signal"]:  # Strong cointegration signal
                strong_signals.append(result)
    
    # Sort all results by risk (lowest first)
    all_results.sort(key=lambda x: x["risk_pct"])
    
    # Top 5 lowest risk (regardless of signal strength)
    top_5_low_risk = all_results[:5]
    
    # Strong signals sorted by |z-score| (strongest first)
    strong_signals.sort(key=lambda x: abs(x["z_score"]), reverse=True)
    
    return {
        "top_5_low_risk": top_5_low_risk,
        "strong_signals": strong_signals[:5],
        "total_analyzed": len(all_results),
        "all_results": all_results
    }

# ============================================================
# ICP CANISTER
# ============================================================

def call_canister(method, args=""):
    try:
        cmd = (
            f'export DFX_WARNING=-mainnet_plaintext_identity; '
            f'source ~/.local/share/dfx/env 2>/dev/null; '
            f'cd /mnt/c/Users/ADVAIT/.gemini/antigravity/scratch/antigravity && '
            f'dfx ping local >/dev/null 2>&1 || (dfx start --background && sleep 3); '
            f'dfx canister call trading_agent {method} {args} 2>&1'
        )
        result = subprocess.run(
            ["wsl", "bash", "--noprofile", "--norc", "-c", cmd],
            capture_output=True, text=True, timeout=45
        )
        return result.stdout.strip()
    except:
        return "Error calling canister"

def get_portfolio():
    return call_canister("get_portfolio")

def execute_trade(action, asset, amount, price, reasoning):
    reasoning = reasoning.replace('"', "'")[:80]
    args = f'\'("{action}", "{asset}", {amount}: float64, {price}: float64, "{reasoning}")\''
    return call_canister("execute_trade", args)

# ============================================================
# AI DECISION (Groq)
# ============================================================

def get_ai_decision(prices, portfolio, signals):
    """Get AI trading decision based on cointegration signals"""
    if not GROQ_API_KEY:
        return {"should_trade": False, "reasoning": "No API key"}
    
    prompt = f"""You are Antigravity, an AI trading agent using COINTEGRATION ANALYSIS.

MARKET PRICES:
{json.dumps(prices, indent=2)}

PORTFOLIO:
{portfolio}

COINTEGRATION SIGNALS (These pairs are stationary - mean-reverting!):
{json.dumps(signals, indent=2) if signals else "No signals currently"}

SIGNAL INTERPRETATION:
- LONG_SPREAD: The spread is too low (Z < -2). Buy the first asset, it will go up relative to second.
- SHORT_SPREAD: The spread is too high (Z > 2). Sell the first asset, it will go down relative to second.
- Higher |Z-Score| = Stronger signal
- Lower ADF p-value = More confident in cointegration

RULES:
- If there are cointegration signals, trade the most confident one
- If no signals, HOLD and wait
- Max 20% of portfolio per trade
- Explain your statistical reasoning

Respond with JSON only:
{{
    "should_trade": true/false,
    "action": "BUY" or "SELL",
    "asset": "BTC" etc,
    "amount": number,
    "price": current price,
    "reasoning": "brief statistical explanation"
}}"""

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "temperature": 0.3},
            timeout=30
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return json.dumps({"should_trade": False, "reasoning": str(e)})

# ============================================================
# MAIN
# ============================================================

def run_trading_cycle():
    print("\n" + "="*70)
    print("🚀 ANTIGRAVITY AI TRADING AGENT - Cointegration Edition")
    print("   10 Assets • 45 Pairs • 1 Year History • Risk Analysis")
    print("="*70)
    
    # 1. Fetch 1 year of historical data
    fetch_all_historical()
    
    # 2. Fetch current prices
    print("\n💹 CURRENT PRICES:")
    prices = fetch_current_prices()
    for asset, price in prices.items():
        print(f"   {asset}: ${price:,.2f}")
    
    # 3. Run cointegration analysis on all 45 pairs
    print(f"\n🔬 COINTEGRATION ANALYSIS ({len(PAIRS)} pairs)...")
    results = scan_all_pairs()
    
    # 4. Display TOP 5 LOWEST RISK trades with advanced stats
    print(f"\n📊 TOP 5 LOWEST RISK TRADES (of {results['total_analyzed']} analyzed):")
    print("   " + "-"*65)
    for i, t in enumerate(results["top_5_low_risk"], 1):
        stat = "✅" if t["is_stationary"] else "⚠️"
        h_icon = "🔄" if t.get("hurst_exponent", 0.5) < 0.5 else "📈"
        print(f"   {i}. {t['pair']} {stat}")
        print(f"      Risk: {t['risk_pct']}% | Z: {t['z_score']} | R²: {t['r_squared']} | β: {t['beta']}")
        print(f"      Half-Life: {t.get('half_life_days', 'N/A')}d | Hurst: {t.get('hurst_exponent', 'N/A')} {h_icon} | Pct: {t.get('spread_percentile', 'N/A')}%")
        if t.get("sandwich_strategy"):
            sw = t["sandwich_strategy"]
            print(f"      🥪 Sandwich: {sw.get('strategy', 'N/A')} ({sw.get('confidence', 'N/A')}) - {sw.get('entry', '')[:40]}")
    
    # 5. Display STRONG SIGNALS (if any)
    if results["strong_signals"]:
        print(f"\n🎯 STRONG COINTEGRATION SIGNALS ({len(results['strong_signals'])} found):")
        for sig in results["strong_signals"]:
            direction = "📈 LONG" if sig["signal"] == "LONG_SPREAD" else "📉 SHORT"
            print(f"   {direction} {sig['pair']}: Z={sig['z_score']}, Risk={sig['risk_pct']}%, ADF p={sig['adf_p_value']}")
    else:
        print(f"\n⚠️ No strong cointegration signals (Z > 2 or Z < -2) currently.")
    
    # 6. Get portfolio from canister
    print("\n💰 CANISTER PORTFOLIO:")
    portfolio = get_portfolio()
    print(f"   {portfolio}")
    
    # 7. AI Decision
    print("\n🤖 AI TRADING DECISION (Llama 3.3 70B)...")
    decision = get_ai_decision(prices, portfolio, results)
    print(f"   {decision}")
    
    try:
        dec = json.loads(decision)
        if dec.get("should_trade"):
            print(f"\n✅ EXECUTING TRADE:")
            print(f"   {dec['action']} {dec['amount']} {dec['asset']} @ ${dec['price']}")
            print(f"   Reasoning: {dec['reasoning']}")
            result = execute_trade(dec["action"], dec["asset"], dec["amount"], dec["price"], dec["reasoning"])
            print(f"   📝 Result: {result}")
        else:
            print(f"\n⏸️ HOLDING: {dec.get('reasoning', 'No clear opportunity')}")
    except:
        print("   Could not parse AI response")
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    run_trading_cycle()
