"""
Cointegration Strategy Module
Implements the ADF (Augmented Dickey-Fuller) cointegration model 
for statistical arbitrage between crypto pairs.

Uses statsmodels for proper ADF testing - the "Gospel" CADF logic:
Step A: Calculate Rolling OLS Regression (β) on historical windows to find the Hedge Ratio.
Step B: Run statsmodels.adfuller on the residuals (Spread).
Step C: If p < 0.05, the pair is valid. If Z-Score of Spread > 2.0, execute trade.
"""

import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from typing import Dict, Any, List, Union


class PairsTrader:
    """
    Pairs Trading strategy using cointegration analysis.
    
    This class implements statistical arbitrage by:
    1. Calculating the spread between two price series using OLS regression
    2. Testing stationarity of the spread using the ADF test
    3. Generating trading signals when pairs are cointegrated
    
    The "Scanner & Sniper" Logic:
    - Scanner: Check if pairs are cointegrated (ADF p < 0.05)
    - Sniper: Execute trade when Z-Score > 2.0 or < -2.0
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize the PairsTrader.
        
        Args:
            significance_level: P-value threshold for cointegration test.
                               Default is 0.05 (95% confidence).
        """
        self.significance_level = significance_level
        self.beta = None  # Hedge ratio from OLS regression
        self.alpha = None  # Intercept from OLS regression
        self.spread_mean = None
        self.spread_std = None
    
    def calculate_spread(
        self, 
        series_a: Union[List[float], np.ndarray], 
        series_b: Union[List[float], np.ndarray]
    ) -> np.ndarray:
        """
        Calculate the spread between two price series using OLS regression.
        
        Step A of CADF: Rolling OLS Regression to find Hedge Ratio (β)
        
        The spread is calculated as: spread = series_a - beta * series_b
        where beta is the hedge ratio derived from OLS regression.
        
        Args:
            series_a: First price series (e.g., BTC prices)
            series_b: Second price series (e.g., ICP prices)
            
        Returns:
            numpy array containing the spread values (residuals)
        
        Raises:
            ValueError: If series have different lengths or are too short
        """
        # Convert to numpy arrays
        series_a = np.array(series_a)
        series_b = np.array(series_b)
        
        if len(series_a) != len(series_b):
            raise ValueError("Both series must have the same length")
        
        if len(series_a) < 20:
            raise ValueError("Series must have at least 20 data points for reliable analysis")
        
        # Perform OLS regression: series_a = alpha + beta * series_b + epsilon
        series_b_const = add_constant(series_b)
        model = OLS(series_a, series_b_const)
        results = model.fit()
        
        # Extract beta (hedge ratio) and alpha (intercept)
        self.alpha = results.params[0]
        self.beta = results.params[1]
        
        # Calculate spread (residuals)
        spread = series_a - (self.alpha + self.beta * series_b)
        
        # Store spread statistics
        self.spread_mean = np.mean(spread)
        self.spread_std = np.std(spread)
        
        return spread
    
    def check_stationarity(self, spread: Union[List[float], np.ndarray]) -> Dict[str, Any]:
        """
        Check if the spread is stationary using the Augmented Dickey-Fuller test.
        
        Step B of CADF: Run statsmodels.adfuller on the residuals (Spread)
        
        A stationary spread indicates cointegration between the two series,
        meaning they tend to move together over time and any deviations
        are temporary - perfect for mean-reversion trading.
        
        Args:
            spread: The spread series to test for stationarity
            
        Returns:
            Dictionary containing:
                - p_value: The p-value from the ADF test
                - adf_statistic: The ADF test statistic
                - critical_values: Critical values at 1%, 5%, 10% levels
                - is_cointegrated: Boolean indicating if pairs are cointegrated
                - significance_level: The significance level used for testing
        """
        spread = np.array(spread)
        
        # Perform ADF test using statsmodels
        adf_result = adfuller(spread, autolag='AIC')
        
        adf_statistic = adf_result[0]
        p_value = adf_result[1]
        used_lag = adf_result[2]
        n_obs = adf_result[3]
        critical_values = adf_result[4]
        
        # Step C: If p < 0.05, the pair is valid (cointegrated)
        is_cointegrated = p_value < self.significance_level
        
        return {
            'p_value': p_value,
            'adf_statistic': adf_statistic,
            'critical_values': critical_values,
            'used_lag': used_lag,
            'n_observations': n_obs,
            'is_cointegrated': is_cointegrated,
            'significance_level': self.significance_level
        }
    
    def get_z_score(self, spread: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Calculate the z-score of the spread for signal generation.
        
        Z = (spread - mean) / std
        
        Args:
            spread: The spread series
            
        Returns:
            Z-score values indicating how many standard deviations
            the spread is from its mean
        """
        spread = np.array(spread)
        
        if self.spread_mean is None or self.spread_std is None:
            self.spread_mean = np.mean(spread)
            self.spread_std = np.std(spread)
        
        if self.spread_std == 0:
            return np.zeros_like(spread)
        
        z_score = (spread - self.spread_mean) / self.spread_std
        return z_score
    
    def generate_signal(
        self, 
        spread: Union[List[float], np.ndarray],
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        stop_loss_threshold: float = 4.0
    ) -> Dict[str, Any]:
        """
        Generate trading signals based on z-score of the spread.
        
        Step C of CADF: If Z-Score of Spread > 2.0, execute trade.
        
        Signal Logic:
            - LONG_SPREAD when Z < -entry_threshold (spread too low, expect rise)
            - SHORT_SPREAD when Z > entry_threshold (spread too high, expect fall)
            - EXIT when |Z| < exit_threshold (spread normalized)
            - STOP_LOSS when |Z| > stop_loss_threshold (extreme deviation)
            - HOLD otherwise
        
        Args:
            spread: The spread series
            entry_threshold: Z-score threshold for entry (default: 2.0)
            exit_threshold: Z-score threshold for exit (default: 0.5)
            stop_loss_threshold: Z-score threshold for stop-loss (default: 4.0)
            
        Returns:
            Dictionary containing signal information
        """
        z_scores = self.get_z_score(spread)
        current_z = z_scores[-1]  # Most recent z-score
        
        signal = "HOLD"
        
        # Check stop-loss first
        if abs(current_z) >= stop_loss_threshold:
            signal = "STOP_LOSS"
        elif current_z > entry_threshold:
            signal = "SHORT_SPREAD"  # Spread is too high, expect mean reversion
        elif current_z < -entry_threshold:
            signal = "LONG_SPREAD"   # Spread is too low, expect mean reversion
        elif abs(current_z) < exit_threshold:
            signal = "EXIT"  # Close positions when spread normalizes
        
        return {
            'signal': signal,
            'current_z_score': float(current_z),
            'entry_threshold': entry_threshold,
            'exit_threshold': exit_threshold,
            'stop_loss_threshold': stop_loss_threshold,
            'hedge_ratio': float(self.beta) if self.beta is not None else None,
            'alpha': float(self.alpha) if self.alpha is not None else None
        }
    
    def analyze_pair(
        self,
        series_a: Union[List[float], np.ndarray],
        series_b: Union[List[float], np.ndarray]
    ) -> Dict[str, Any]:
        """
        Complete analysis pipeline for a trading pair.
        
        Implements the full "Scanner & Sniper" logic:
        1. Calculate spread using OLS
        2. Test for cointegration using ADF
        3. Generate trading signal if cointegrated
        
        Args:
            series_a: First price series
            series_b: Second price series
            
        Returns:
            Complete analysis results
        """
        # Step A: Calculate spread
        spread = self.calculate_spread(series_a, series_b)
        
        # Step B: Check stationarity
        adf_result = self.check_stationarity(spread)
        
        # Step C: Generate signal if cointegrated
        signal_result = self.generate_signal(spread)
        
        return {
            'is_cointegrated': adf_result['is_cointegrated'],
            'adf_p_value': adf_result['p_value'],
            'adf_statistic': adf_result['adf_statistic'],
            'hedge_ratio': float(self.beta),
            'alpha': float(self.alpha),
            'spread_mean': float(self.spread_mean),
            'spread_std': float(self.spread_std),
            'current_z_score': signal_result['current_z_score'],
            'signal': signal_result['signal'],
            'spread': spread.tolist()
        }


def test_pairs_trader():
    """Test the PairsTrader class with synthetic cointegrated data."""
    np.random.seed(42)
    
    print("=" * 60)
    print("CADF Cointegration Strategy Test")
    print("=" * 60)
    
    # Generate cointegrated series
    n = 200
    
    # Common stochastic trend (random walk)
    common_trend = np.cumsum(np.random.normal(0, 1, n))
    
    # Series A: Follows trend with noise
    series_a = 100 + 2 * common_trend + np.random.normal(0, 2, n)
    
    # Series B: Follows same trend with different scaling
    series_b = 50 + common_trend + np.random.normal(0, 1, n)
    
    # Test the trader
    trader = PairsTrader()
    
    print("\n[Step A] OLS Regression - Calculating Hedge Ratio (β)")
    spread = trader.calculate_spread(series_a, series_b)
    print(f"  • Beta (hedge ratio): {trader.beta:.4f}")
    print(f"  • Alpha (intercept): {trader.alpha:.4f}")
    print(f"  • Spread mean: {trader.spread_mean:.4f}")
    print(f"  • Spread std: {trader.spread_std:.4f}")
    
    print("\n[Step B] ADF Test - Checking Stationarity")
    result = trader.check_stationarity(spread)
    print(f"  • ADF statistic: {result['adf_statistic']:.4f}")
    print(f"  • P-value: {result['p_value']:.6f}")
    print(f"  • Critical values: {result['critical_values']}")
    print(f"  • Is cointegrated (p < 0.05): {result['is_cointegrated']}")
    
    print("\n[Step C] Signal Generation - Z-Score Analysis")
    signal = trader.generate_signal(spread)
    print(f"  • Current Z-score: {signal['current_z_score']:.4f}")
    print(f"  • Signal: {signal['signal']}")
    
    print("\n" + "=" * 60)
    if result['is_cointegrated']:
        print("✅ PAIR IS COINTEGRATED - Valid for pairs trading!")
    else:
        print("❌ PAIR IS NOT COINTEGRATED - Not suitable for pairs trading")
    print("=" * 60)
    
    return trader, spread, result


if __name__ == "__main__":
    test_pairs_trader()
