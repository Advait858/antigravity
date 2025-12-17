"""
Cointegration Strategy Module
Implements the ADF (Augmented Dickey-Fuller) cointegration model 
for statistical arbitrage between crypto pairs.
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
        self.spread_mean = None
        self.spread_std = None
    
    def calculate_spread(
        self, 
        series_a: Union[List[float], np.ndarray], 
        series_b: Union[List[float], np.ndarray]
    ) -> np.ndarray:
        """
        Calculate the spread between two price series using OLS regression.
        
        The spread is calculated as: spread = series_a - beta * series_b
        where beta is the hedge ratio derived from OLS regression.
        
        Args:
            series_a: First price series (e.g., BTC prices)
            series_b: Second price series (e.g., ICP prices)
            
        Returns:
            numpy array containing the spread values
        
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
        
        # Extract beta (hedge ratio)
        self.beta = results.params[1]
        alpha = results.params[0]
        
        # Calculate spread
        spread = series_a - (alpha + self.beta * series_b)
        
        # Store spread statistics
        self.spread_mean = np.mean(spread)
        self.spread_std = np.std(spread)
        
        return spread
    
    def check_stationarity(self, spread: Union[List[float], np.ndarray]) -> Dict[str, Any]:
        """
        Check if the spread is stationary using the Augmented Dickey-Fuller test.
        
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
        
        # Perform ADF test
        adf_result = adfuller(spread, autolag='AIC')
        
        adf_statistic = adf_result[0]
        p_value = adf_result[1]
        used_lag = adf_result[2]
        n_obs = adf_result[3]
        critical_values = adf_result[4]
        
        # Determine if cointegrated based on p-value
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
        
        z_score = (spread - self.spread_mean) / self.spread_std
        return z_score
    
    def generate_signal(
        self, 
        spread: Union[List[float], np.ndarray],
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Generate trading signals based on z-score of the spread.
        
        Args:
            spread: The spread series
            entry_threshold: Z-score threshold for entry (default: 2.0)
            exit_threshold: Z-score threshold for exit (default: 0.5)
            
        Returns:
            Dictionary containing signal information
        """
        z_scores = self.get_z_score(spread)
        current_z = z_scores[-1]  # Most recent z-score
        
        signal = "HOLD"
        if current_z > entry_threshold:
            signal = "SHORT_SPREAD"  # Spread is too high, expect mean reversion
        elif current_z < -entry_threshold:
            signal = "LONG_SPREAD"   # Spread is too low, expect mean reversion
        elif abs(current_z) < exit_threshold:
            signal = "EXIT"  # Close positions when spread normalizes
        
        return {
            'signal': signal,
            'current_z_score': current_z,
            'entry_threshold': entry_threshold,
            'exit_threshold': exit_threshold,
            'hedge_ratio': self.beta
        }
