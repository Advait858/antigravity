"""
Antigravity Backtest Module
Proves the CADF cointegration strategy works on dummy data.

This module demonstrates:
1. OLS Hedge Ratio calculation
2. ADF stationarity testing (p < 0.05 = cointegrated)
3. Z-Score signal generation (|Z| > 2.0 = trade)
4. Full backtest with P&L tracking

Run this to validate the strategy math before ICP deployment.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Import our strategy
from strategy.cointegration import PairsTrader


class Position(Enum):
    """Trading position states."""
    FLAT = "flat"
    LONG_SPREAD = "long_spread"    # Long A, Short B
    SHORT_SPREAD = "short_spread"  # Short A, Long B


@dataclass
class Trade:
    """Record of a completed trade."""
    entry_idx: int
    exit_idx: int
    position: str
    entry_z: float
    exit_z: float
    entry_spread: float
    exit_spread: float
    pnl: float
    return_pct: float


@dataclass
class BacktestResult:
    """Complete backtest results."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_trade_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    trades: List[Trade]
    equity_curve: List[float]


class Backtester:
    """
    Backtest engine for the CADF cointegration strategy.
    
    Simulates trading based on:
    - Entry: |Z-Score| > entry_threshold (default 2.0)
    - Exit: |Z-Score| < exit_threshold (default 0.5)
    - Stop-Loss: |Z-Score| > stop_loss_threshold (default 4.0)
    """
    
    def __init__(
        self,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        stop_loss_threshold: float = 4.0,
        trade_size: float = 1000.0,  # USD per trade
        transaction_cost_bps: float = 30.0  # 0.3% per trade
    ):
        """
        Initialize backtester.
        
        Args:
            entry_threshold: Z-score for entry (default 2.0)
            exit_threshold: Z-score for exit (default 0.5)
            stop_loss_threshold: Z-score for stop-loss (default 4.0)
            trade_size: Notional trade size in USD
            transaction_cost_bps: Transaction cost in basis points
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.trade_size = trade_size
        self.transaction_cost_pct = transaction_cost_bps / 10000
        
        self.trader = PairsTrader()
    
    def run(
        self,
        series_a: np.ndarray,
        series_b: np.ndarray,
        lookback_window: int = 60
    ) -> BacktestResult:
        """
        Run backtest on price series.
        
        Uses rolling window for OLS and ADF calculations.
        
        Args:
            series_a: First price series (e.g., BTC)
            series_b: Second price series (e.g., ICP)
            lookback_window: Rolling window for calculations
            
        Returns:
            BacktestResult with all metrics
        """
        n = len(series_a)
        
        if n < lookback_window + 20:
            raise ValueError(f"Need at least {lookback_window + 20} data points")
        
        # Track state
        position = Position.FLAT
        entry_idx = 0
        entry_z = 0.0
        entry_spread = 0.0
        
        trades: List[Trade] = []
        equity = [0.0]  # Starting with 0 P&L
        
        print(f"\n{'='*60}")
        print("BACKTEST RUNNING")
        print(f"{'='*60}")
        print(f"Data points: {n}")
        print(f"Lookback window: {lookback_window}")
        print(f"Entry threshold: ±{self.entry_threshold}σ")
        print(f"Exit threshold: ±{self.exit_threshold}σ")
        print(f"Stop-loss: ±{self.stop_loss_threshold}σ")
        print(f"{'='*60}\n")
        
        # Rolling backtest
        for i in range(lookback_window, n):
            # Get rolling window
            window_a = series_a[i - lookback_window:i]
            window_b = series_b[i - lookback_window:i]
            
            # Calculate spread
            spread = self.trader.calculate_spread(window_a, window_b)
            
            # Current spread value (last in window)
            current_spread = spread[-1]
            
            # Get Z-score
            z_scores = self.trader.get_z_score(spread)
            z = z_scores[-1]
            
            # Check cointegration periodically (every 20 bars)
            if i % 20 == 0:
                adf_result = self.trader.check_stationarity(spread)
                is_coint = adf_result['is_cointegrated']
                if not is_coint and position != Position.FLAT:
                    # Cointegration broke - close position
                    print(f"[{i}] ⚠️ Cointegration broken (p={adf_result['p_value']:.4f})")
            
            # Trading logic
            if position == Position.FLAT:
                # Look for entry
                if z < -self.entry_threshold:
                    position = Position.LONG_SPREAD
                    entry_idx = i
                    entry_z = z
                    entry_spread = current_spread
                    print(f"[{i}] 📈 LONG SPREAD @ Z={z:.2f}")
                    
                elif z > self.entry_threshold:
                    position = Position.SHORT_SPREAD
                    entry_idx = i
                    entry_z = z
                    entry_spread = current_spread
                    print(f"[{i}] 📉 SHORT SPREAD @ Z={z:.2f}")
            
            else:
                # In position - check for exit
                should_exit = False
                exit_reason = ""
                
                # Mean reversion exit
                if abs(z) < self.exit_threshold:
                    should_exit = True
                    exit_reason = "Mean reversion"
                
                # Stop-loss
                elif abs(z) > self.stop_loss_threshold:
                    should_exit = True
                    exit_reason = "Stop-loss"
                
                if should_exit:
                    # Calculate P&L
                    spread_move = current_spread - entry_spread
                    
                    if position == Position.LONG_SPREAD:
                        pnl = spread_move * (self.trade_size / abs(entry_spread)) if entry_spread != 0 else 0
                    else:  # SHORT_SPREAD
                        pnl = -spread_move * (self.trade_size / abs(entry_spread)) if entry_spread != 0 else 0
                    
                    # Subtract transaction costs (round-trip)
                    pnl -= self.trade_size * self.transaction_cost_pct * 2
                    
                    trade = Trade(
                        entry_idx=entry_idx,
                        exit_idx=i,
                        position=position.value,
                        entry_z=entry_z,
                        exit_z=z,
                        entry_spread=entry_spread,
                        exit_spread=current_spread,
                        pnl=pnl,
                        return_pct=(pnl / self.trade_size) * 100
                    )
                    trades.append(trade)
                    
                    emoji = "✅" if pnl > 0 else "❌"
                    print(f"[{i}] {emoji} EXIT ({exit_reason}) @ Z={z:.2f}, PnL=${pnl:.2f}")
                    
                    position = Position.FLAT
            
            # Update equity curve
            equity.append(equity[-1] + (trades[-1].pnl if trades and trades[-1].exit_idx == i else 0))
        
        # Calculate metrics
        result = self._calculate_metrics(trades, equity)
        
        self._print_summary(result)
        
        return result
    
    def _calculate_metrics(self, trades: List[Trade], equity: List[float]) -> BacktestResult:
        """Calculate performance metrics."""
        if not trades:
            return BacktestResult(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                avg_trade_pnl=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                trades=[],
                equity_curve=equity
            )
        
        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]
        
        total_pnl = sum(t.pnl for t in trades)
        
        # Calculate max drawdown
        peak = equity[0]
        max_dd = 0.0
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        
        # Calculate Sharpe (annualized, assuming daily returns)
        returns = [trades[i].return_pct for i in range(len(trades))]
        if len(returns) > 1:
            avg_ret = np.mean(returns)
            std_ret = np.std(returns)
            sharpe = (avg_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0
        else:
            sharpe = 0.0
        
        return BacktestResult(
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=len(winning) / len(trades) * 100 if trades else 0,
            total_pnl=total_pnl,
            avg_trade_pnl=total_pnl / len(trades) if trades else 0,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            trades=trades,
            equity_curve=equity
        )
    
    def _print_summary(self, result: BacktestResult) -> None:
        """Print backtest summary."""
        print(f"\n{'='*60}")
        print("BACKTEST RESULTS")
        print(f"{'='*60}")
        print(f"Total Trades:    {result.total_trades}")
        print(f"Winning Trades:  {result.winning_trades}")
        print(f"Losing Trades:   {result.losing_trades}")
        print(f"Win Rate:        {result.win_rate:.1f}%")
        print(f"{'='*60}")
        print(f"Total P&L:       ${result.total_pnl:.2f}")
        print(f"Avg Trade P&L:   ${result.avg_trade_pnl:.2f}")
        print(f"Max Drawdown:    ${result.max_drawdown:.2f}")
        print(f"Sharpe Ratio:    {result.sharpe_ratio:.2f}")
        print(f"{'='*60}")


def generate_cointegrated_series(
    n: int = 500,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic cointegrated price series for testing.
    
    Creates two assets that follow a common stochastic trend
    with mean-reverting spread - ideal for pairs trading.
    
    Args:
        n: Number of data points
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (series_a, series_b)
    """
    np.random.seed(seed)
    
    # Common stochastic trend (random walk)
    trend = np.cumsum(np.random.normal(0, 1, n))
    
    # Asset A: BTC-like (high price, follows trend)
    base_a = 42000
    series_a = base_a + 100 * trend + np.cumsum(np.random.normal(0, 50, n))
    
    # Asset B: ICP-like (low price, follows same trend with different beta)
    base_b = 12.5
    beta_true = 3360  # BTC/ICP ratio
    series_b = base_b + (100 / beta_true) * trend + np.cumsum(np.random.normal(0, 0.05, n))
    
    # Add some mean-reverting noise to make spread stationary
    spread_noise = np.zeros(n)
    for i in range(1, n):
        spread_noise[i] = 0.9 * spread_noise[i-1] + np.random.normal(0, 0.2)
    
    series_b += spread_noise
    
    return series_a, series_b


def main():
    """Run the backtest demonstration."""
    print("\n" + "=" * 60)
    print("ANTIGRAVITY BACKTEST - CADF Strategy Validation")
    print("=" * 60)
    
    # Generate test data
    print("\n📊 Generating cointegrated test data...")
    series_a, series_b = generate_cointegrated_series(n=500, seed=42)
    
    print(f"  Series A (BTC): {len(series_a)} points, range ${series_a.min():.0f} - ${series_a.max():.0f}")
    print(f"  Series B (ICP): {len(series_b)} points, range ${series_b.min():.2f} - ${series_b.max():.2f}")
    
    # Quick cointegration check
    print("\n🔬 Quick cointegration check (full series)...")
    trader = PairsTrader()
    analysis = trader.analyze_pair(series_a, series_b)
    
    print(f"  ADF Statistic: {analysis['adf_statistic']:.4f}")
    print(f"  P-Value: {analysis['adf_p_value']:.6f}")
    print(f"  Cointegrated: {analysis['is_cointegrated']}")
    print(f"  Hedge Ratio (β): {analysis['hedge_ratio']:.2f}")
    
    if not analysis['is_cointegrated']:
        print("\n⚠️ WARNING: Series not cointegrated. Backtest may perform poorly.")
    
    # Run backtest
    print("\n🚀 Running backtest...")
    backtester = Backtester(
        entry_threshold=2.0,
        exit_threshold=0.5,
        stop_loss_threshold=4.0,
        trade_size=1000.0,
        transaction_cost_bps=30.0
    )
    
    result = backtester.run(series_a, series_b, lookback_window=60)
    
    # Print trade details
    if result.trades:
        print("\n📋 TRADE LOG")
        print("-" * 80)
        for i, t in enumerate(result.trades[:10], 1):  # Show first 10 trades
            print(f"  {i}. {t.position.upper():15} | Entry Z: {t.entry_z:+.2f} → Exit Z: {t.exit_z:+.2f} | PnL: ${t.pnl:+.2f}")
        if len(result.trades) > 10:
            print(f"  ... and {len(result.trades) - 10} more trades")
    
    print("\n✅ Backtest complete!")
    return result


if __name__ == "__main__":
    main()
