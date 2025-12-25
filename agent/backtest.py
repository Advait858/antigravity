"""
Antigravity Backtesting Engine
Fetches 13 months of hourly data, analyzes 12, backtests on month 13
Uses cointegration strategies and paper portfolio from canister
"""

import os
import json
import requests
import subprocess
import math
from datetime import datetime, timedelta
from collections import defaultdict

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EPSILON = 1e-10

# Assets to trade
ASSETS = ["BTC", "ETH", "ICP", "SOL", "XRP", "AVAX", "DOT", "LINK", "ADA", "DOGE"]
PAIRS = [(ASSETS[i], ASSETS[j]) for i in range(len(ASSETS)) for j in range(i+1, len(ASSETS))]

# ============================================================
# MATH FUNCTIONS (copied from trading_agent.py)
# ============================================================

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

class OLSResult:
    def __init__(self, beta, alpha, r_squared, residuals):
        self.beta = beta
        self.alpha = alpha
        self.r_squared = r_squared
        self.residuals = residuals

def ols_regression(y, x):
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
    p_value = 0.01 if t_stat < -3.43 else (0.05 if t_stat < -2.86 else (0.10 if t_stat < -2.57 else 1.0))
    return {"t_stat": t_stat, "p_value": p_value, "is_stationary": p_value < 0.05}

def z_score(current, history, window=30):
    if len(history) < 2: return 0
    data = history[-window:]
    mu, sigma = mean(data), std_dev(data)
    return (current - mu) / sigma if sigma > EPSILON else 0

def calculate_half_life(spread):
    if len(spread) < 10: return 999
    dy = [spread[i] - spread[i-1] for i in range(1, len(spread))]
    y_lag = spread[:-1]
    res = ols_regression(dy, y_lag)
    if res.beta >= 0 or abs(res.beta) < EPSILON: return 999
    half_life = -math.log(2) / res.beta
    return max(1, min(half_life, 999))

# ============================================================
# DATA FETCHING - 13 MONTHS HOURLY
# ============================================================

hourly_data = {}  # {asset: {"closes": [], "highs": [], "lows": [], "volumes": [], "timestamps": []}}

def fetch_hourly_klines(asset, months=13):
    """Fetch hourly klines for specified months (paginated)"""
    print(f"   Fetching {months} months of hourly {asset} data...")
    
    all_klines = []
    end_time = int(datetime.now().timestamp() * 1000)
    hours_needed = months * 30 * 24  # ~9360 hours for 13 months
    
    # Binance returns max 1000 klines per request
    while len(all_klines) < hours_needed:
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                "symbol": f"{asset}USDT",
                "interval": "1h",
                "limit": 1000,
                "endTime": end_time
            }
            resp = requests.get(url, params=params, timeout=30)
            
            if resp.status_code == 200:
                klines = resp.json()
                if not klines:
                    break
                all_klines = klines + all_klines
                end_time = klines[0][0] - 1  # Start of next batch
            else:
                break
        except Exception as e:
            print(f"      Error: {e}")
            break
    
    # Parse klines
    closes = [float(k[4]) for k in all_klines]
    highs = [float(k[2]) for k in all_klines]
    lows = [float(k[3]) for k in all_klines]
    volumes = [float(k[5]) for k in all_klines]
    timestamps = [k[0] for k in all_klines]
    
    return {
        "closes": closes,
        "highs": highs,
        "lows": lows,
        "volumes": volumes,
        "timestamps": timestamps,
        "count": len(closes)
    }

def fetch_all_hourly_data():
    """Fetch 13 months of hourly data for all assets"""
    print("\n📊 FETCHING 13 MONTHS OF HOURLY DATA...")
    for asset in ASSETS:
        data = fetch_hourly_klines(asset, 13)
        hourly_data[asset] = data
        print(f"   ✅ {asset}: {data['count']} hourly candles")
    
    total = sum(d["count"] for d in hourly_data.values())
    print(f"\n   📈 Total: {total:,} hourly data points")
    print(f"   📊 Pairs to analyze: {len(PAIRS)}")

# ============================================================
# SPLIT DATA: 12 MONTHS TRAIN, 1 MONTH TEST
# ============================================================

def split_data():
    """Split data into 12 months training and 1 month testing"""
    train_data = {}
    test_data = {}
    
    for asset, data in hourly_data.items():
        n = len(data["closes"])
        # Last month = ~720 hours
        test_start = n - 720
        
        train_data[asset] = data["closes"][:test_start]
        test_data[asset] = data["closes"][test_start:]
    
    return train_data, test_data

# ============================================================
# COINTEGRATION ANALYSIS
# ============================================================

def analyze_pair_for_backtest(asset_a, asset_b, price_data):
    """Analyze a pair using training data"""
    pa = price_data.get(asset_a, [])
    pb = price_data.get(asset_b, [])
    
    n = min(len(pa), len(pb))
    if n < 100:
        return None
    
    pa = pa[-n:]
    pb = pb[-n:]
    
    ols = ols_regression(pa, pb)
    spread = [pa[i] - (ols.alpha + ols.beta * pb[i]) for i in range(n)]
    adf = adf_test(spread)
    
    half_life = calculate_half_life(spread)
    current_z = z_score(spread[-1], spread, 100)
    
    # Return all pairs, filter later
    return {
        "pair": f"{asset_a}-{asset_b}",
        "alpha": ols.alpha,
        "beta": ols.beta,
        "r_squared": ols.r_squared,
        "half_life": half_life,
        "adf_p": adf["p_value"],
        "is_stationary": adf["is_stationary"],
        "current_z": current_z,
        "spread_mean": mean(spread),
        "spread_std": std_dev(spread)
    }

def find_cointegrated_pairs(train_data):
    """Find all cointegrated pairs from training data"""
    pairs = []
    for asset_a, asset_b in PAIRS:
        result = analyze_pair_for_backtest(asset_a, asset_b, train_data)
        if result:
            # Relaxed criteria for hourly data
            # Half-life < 500 hours (~3 weeks) and some mean reversion
            if result["half_life"] < 500 and result["r_squared"] > 0.3:
                pairs.append(result)
    
    pairs.sort(key=lambda x: x["half_life"])
    return pairs

# ============================================================
# BACKTESTING ENGINE
# ============================================================

class Backtest:
    def __init__(self, initial_capital=100000):
        self.capital = initial_capital
        self.positions = {}  # {asset: {"amount": x, "entry_price": y}}
        self.trades = []
        self.equity_curve = [initial_capital]
        
    def execute_trade(self, action, asset, amount, price, reason):
        """Execute a trade in backtest"""
        trade_value = amount * price
        
        if action == "BUY":
            if trade_value > self.capital:
                return False, "Insufficient capital"
            self.capital -= trade_value
            if asset in self.positions:
                self.positions[asset]["amount"] += amount
            else:
                self.positions[asset] = {"amount": amount, "entry_price": price}
        
        elif action == "SELL":
            if asset not in self.positions or self.positions[asset]["amount"] < amount:
                return False, "Insufficient position"
            self.positions[asset]["amount"] -= amount
            self.capital += trade_value
            if self.positions[asset]["amount"] <= 0:
                del self.positions[asset]
        
        self.trades.append({
            "action": action,
            "asset": asset,
            "amount": amount,
            "price": price,
            "reason": reason
        })
        return True, "OK"
    
    def get_portfolio_value(self, current_prices):
        """Calculate total portfolio value"""
        total = self.capital
        for asset, pos in self.positions.items():
            if asset in current_prices:
                total += pos["amount"] * current_prices[asset]
        return total
    
    def record_equity(self, current_prices):
        """Record equity curve point"""
        self.equity_curve.append(self.get_portfolio_value(current_prices))

def run_backtest(cointegrated_pairs, test_data, train_data, entry_z=1.0, exit_z=0.2):
    """Run backtest with AI agent picking trades - minimum 5 trades per day"""
    print("\n🔄 RUNNING AI-DRIVEN BACKTEST ON MONTH 13...")
    print("   Strategy: AI picks trades, min 5/day, lowest risk priority")
    
    bt = Backtest(100000)
    
    # Get test period length
    n = min(len(test_data[asset]) for asset in ASSETS)
    print(f"   Test period: {n} hourly candles ({n//24} days)")
    
    # Track open trades and daily stats
    open_trades = {}
    daily_trade_count = 0
    prev_day = -1
    
    for hour in range(100, n):
        current_day = hour // 24
        
        # Reset daily trade count on new day
        if current_day != prev_day:
            daily_trade_count = 0
            prev_day = current_day
        
        # Current prices
        prices = {asset: test_data[asset][hour] for asset in ASSETS}
        
        # Every 4 hours: AI evaluates and picks trades
        if hour % 4 == 0:
            # Calculate current state for all pairs
            pair_opportunities = []
            
            for pair_info in cointegrated_pairs:
                pair = pair_info["pair"]
                asset_a, asset_b = pair.split("-")
                
                pa = test_data[asset_a][hour]
                pb = test_data[asset_b][hour]
                spread = pa - (pair_info["alpha"] + pair_info["beta"] * pb)
                current_z = (spread - pair_info["spread_mean"]) / pair_info["spread_std"]
                
                # Risk score
                risk = abs(current_z) * 10 + (1 - pair_info["r_squared"]) * 50 + pair_info["half_life"] / 10
                
                pair_opportunities.append({
                    "pair": pair,
                    "asset_a": asset_a,
                    "asset_b": asset_b,
                    "z": current_z,
                    "risk": risk,
                    "pa": pa,
                    "pb": pb,
                    "half_life": pair_info["half_life"],
                    "r_squared": pair_info["r_squared"],
                    "info": pair_info
                })
            
            # Sort by lowest risk
            pair_opportunities.sort(key=lambda x: x["risk"])
            
            # AI DECISION: Get recommendation from Groq
            ai_trades = get_ai_backtest_decision(
                pair_opportunities[:10],  # Top 10 opportunities
                bt.capital,
                len(bt.positions),
                daily_trade_count,
                current_day
            )
            
            # Execute AI-recommended trades
            for trade in ai_trades:
                if trade["action"] == "BUY":
                    opp = next((o for o in pair_opportunities if o["pair"] == trade["pair"]), None)
                    if opp and bt.capital > 5000:
                        asset = opp["asset_a"] if trade.get("direction") == "long" else opp["asset_b"]
                        price = opp["pa"] if asset == opp["asset_a"] else opp["pb"]
                        trade_amt = min(bt.capital * 0.12, 12000) / price
                        success, _ = bt.execute_trade("BUY", asset, trade_amt, price,
                            f"AI: {trade.get('reason', 'Trade')} Z={opp['z']:.2f}")
                        if success:
                            open_trades[trade["pair"]] = {
                                "direction": trade.get("direction", "long"),
                                "entry_hour": hour,
                                "asset": asset
                            }
                            daily_trade_count += 1
            
            # Force minimum 5 trades per day if not met
            if hour % 24 == 20 and daily_trade_count < 5:
                needed = 5 - daily_trade_count
                for opp in pair_opportunities[:needed]:
                    if opp["pair"] not in open_trades and bt.capital > 5000:
                        # Pick direction based on z-score
                        if opp["z"] < 0:
                            asset = opp["asset_a"]
                            price = opp["pa"]
                            direction = "long"
                        else:
                            asset = opp["asset_b"]
                            price = opp["pb"]
                            direction = "short"
                        
                        trade_amt = min(bt.capital * 0.10, 10000) / price
                        success, _ = bt.execute_trade("BUY", asset, trade_amt, price,
                            f"Force: {opp['pair']} Z={opp['z']:.2f} Risk={opp['risk']:.1f}")
                        if success:
                            open_trades[opp["pair"]] = {
                                "direction": direction,
                                "entry_hour": hour,
                                "asset": asset
                            }
                            daily_trade_count += 1
            
            # EXIT logic
            for pair, trade in list(open_trades.items()):
                opp = next((o for o in pair_opportunities if o["pair"] == pair), None)
                if not opp:
                    continue
                
                should_exit = False
                exit_reason = ""
                
                if trade["direction"] == "long" and opp["z"] > -exit_z:
                    should_exit = True
                    exit_reason = f"Exit long {pair} Z={opp['z']:.2f}"
                elif trade["direction"] == "short" and opp["z"] < exit_z:
                    should_exit = True
                    exit_reason = f"Exit short {pair} Z={opp['z']:.2f}"
                elif hour - trade["entry_hour"] > 72:  # 3-day timeout
                    should_exit = True
                    exit_reason = f"Timeout {pair} after {hour - trade['entry_hour']}h"
                
                if should_exit:
                    asset = trade["asset"]
                    if asset in bt.positions:
                        bt.execute_trade("SELL", asset, bt.positions[asset]["amount"], prices[asset], exit_reason)
                        del open_trades[pair]
        
        # Daily equity recording
        if hour % 24 == 0:
            bt.record_equity(prices)
    
    # Close all positions
    final_prices = {asset: test_data[asset][-1] for asset in ASSETS}
    for pair, trade in list(open_trades.items()):
        asset = trade["asset"]
        if asset in bt.positions:
            bt.execute_trade("SELL", asset, bt.positions[asset]["amount"], final_prices[asset], "End")
    
    bt.record_equity(final_prices)
    return bt

def get_ai_backtest_decision(opportunities, capital, open_positions, daily_trades, day):
    """Get AI agent's trade recommendations"""
    
    # If no GROQ key, use rule-based fallback
    if not GROQ_API_KEY:
        return get_rule_based_trades(opportunities, capital, daily_trades)
    
    prompt = f"""You are Antigravity AI trading agent running a BACKTEST.
Day {day}, Capital: ${capital:,.0f}, Open Positions: {open_positions}, Daily Trades: {daily_trades}

TOP 10 TRADING OPPORTUNITIES (lowest risk first):
"""
    for i, opp in enumerate(opportunities, 1):
        prompt += f"{i}. {opp['pair']}: Z={opp['z']:.2f}, Risk={opp['risk']:.1f}, Half-Life={opp['half_life']:.0f}h, R²={opp['r_squared']:.2f}\n"
    
    prompt += """
RULES:
- You MUST pick at least 2 trades from this list
- Lower risk = safer trade
- Z < -1: LONG spread (buy first asset)
- Z > 1: SHORT spread (buy second asset)
- Explain your picks briefly

Return JSON array:
[{"action": "BUY", "pair": "ASSET1-ASSET2", "direction": "long" or "short", "reason": "brief"}]
"""
    
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "temperature": 0.3},
            timeout=10
        )
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            # Extract JSON array
            import re
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                return json.loads(match.group())
    except:
        pass
    
    # Fallback to rule-based
    return get_rule_based_trades(opportunities, capital, daily_trades)

def get_rule_based_trades(opportunities, capital, daily_trades):
    """Rule-based trade selection when AI unavailable"""
    trades = []
    for opp in opportunities[:3]:  # Top 3 lowest risk
        if abs(opp["z"]) > 0.8:
            direction = "long" if opp["z"] < 0 else "short"
            trades.append({
                "action": "BUY",
                "pair": opp["pair"],
                "direction": direction,
                "reason": f"Rule: Z={opp['z']:.2f}"
            })
    return trades

def print_backtest_results(bt):
    """Print backtest results"""
    initial = bt.equity_curve[0]
    final = bt.equity_curve[-1]
    pnl = final - initial
    pnl_pct = (pnl / initial) * 100
    
    print("\n" + "="*60)
    print("📊 BACKTEST RESULTS (Month 13)")
    print("="*60)
    print(f"   Initial Capital: ${initial:,.2f}")
    print(f"   Final Capital:   ${final:,.2f}")
    print(f"   P&L:             ${pnl:,.2f} ({pnl_pct:+.2f}%)")
    print(f"   Total Trades:    {len(bt.trades)}")
    
    # Calculate max drawdown
    peak = initial
    max_dd = 0
    for eq in bt.equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd:
            max_dd = dd
    
    print(f"   Max Drawdown:    {max_dd:.2f}%")
    
    # Win rate
    wins = sum(1 for i in range(0, len(bt.trades)-1, 2) 
               if i+1 < len(bt.trades) and bt.trades[i]["action"] == "BUY" and bt.trades[i+1]["action"] == "SELL")
    if len(bt.trades) > 0:
        print(f"   Trade Count:     {len(bt.trades)}")
    
    print("\n   📈 Sample Trades:")
    for trade in bt.trades[:10]:
        print(f"      {trade['action']} {trade['amount']:.4f} {trade['asset']} @ ${trade['price']:.2f} - {trade['reason']}")
    
    print("="*60)

# ============================================================
# MAIN
# ============================================================

def run_full_backtest():
    print("\n" + "="*70)
    print("🚀 ANTIGRAVITY BACKTESTING ENGINE")
    print("   13 Months Hourly Data | 12 Month Train | 1 Month Test")
    print("="*70)
    
    # 1. Fetch all hourly data
    fetch_all_hourly_data()
    
    # 2. Split into train/test
    print("\n📊 SPLITTING DATA...")
    train_data, test_data = split_data()
    print(f"   Training: {len(train_data['BTC']):,} hourly candles (~12 months)")
    print(f"   Testing:  {len(test_data['BTC']):,} hourly candles (~1 month)")
    
    # 3. Find cointegrated pairs
    print("\n🔬 FINDING COINTEGRATED PAIRS FROM TRAINING DATA...")
    coint_pairs = find_cointegrated_pairs(train_data)
    print(f"   Found {len(coint_pairs)} cointegrated pairs")
    
    for i, p in enumerate(coint_pairs[:5], 1):
        print(f"   {i}. {p['pair']}: Half-Life={p['half_life']:.1f}h, R²={p['r_squared']:.3f}")
    
    # 4. Run backtest on month 13
    bt = run_backtest(coint_pairs, test_data, train_data)
    
    # 5. Print results
    print_backtest_results(bt)
    
    print("\n✅ Backtest complete!")
    print("="*70 + "\n")
    
    return bt

if __name__ == "__main__":
    run_full_backtest()
