"""
Antigravity AI Trading Agent - Full Version
Includes: Cointegration Analysis, OLS Regression, ADF Tests, Z-Score Signals
Uses Groq (Llama 3.3) for AI analysis + ICP Canister for execution
"""

import os
import json
import requests
import subprocess
from datetime import datetime
from collections import defaultdict

# Import shared math functions
from math_core import (
    EPSILON, mean, variance, std_dev, covariance, correlation,
    OLSResult, ols_regression, adf_test, z_score,
    calculate_half_life, calculate_hurst_exponent,
    calculate_spread_stats, calculate_risk_score, get_sandwich_signal
)

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CANISTER_ID = os.getenv("CANISTER_ID", "uxrrr-q7777-77774-qaaaq-cai")
ICP_NETWORK = os.getenv("ICP_NETWORK", "local")  # 'local' or 'ic' for mainnet

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
