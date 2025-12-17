"""
Data Loader Module
Provides mock data fetching functionality for testing the trading strategy.
In production, this would be replaced with calls to the ICP Management Canister
or external HTTPS outcalls to fetch real price data.
"""

import numpy as np
from typing import List, Optional


def fetch_candles(
    symbol: str, 
    num_candles: int = 100,
    seed: Optional[int] = None
) -> List[float]:
    """
    Fetch mock candlestick price data for a given symbol.
    
    This function generates realistic-looking price data using a 
    geometric random walk with cointegration properties for testing
    the pairs trading strategy.
    
    Args:
        symbol: Trading pair symbol (e.g., "BTC", "ICP")
        num_candles: Number of price points to generate (default: 100)
        seed: Random seed for reproducibility (optional)
        
    Returns:
        List of closing prices for the requested symbol
    
    Note:
        In production, this should be replaced with actual API calls
        using the ICP Management Canister's HTTPS outcall feature.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Define base prices and volatilities for different assets
    asset_configs = {
        "BTC": {"base_price": 42000.0, "volatility": 0.02, "drift": 0.0001},
        "ICP": {"base_price": 12.5, "volatility": 0.03, "drift": 0.0002},
        "ETH": {"base_price": 2200.0, "volatility": 0.025, "drift": 0.00015},
        "SOL": {"base_price": 95.0, "volatility": 0.035, "drift": 0.0002},
    }
    
    # Get config or use defaults
    config = asset_configs.get(symbol.upper(), {
        "base_price": 100.0,
        "volatility": 0.02,
        "drift": 0.0001
    })
    
    base_price = config["base_price"]
    volatility = config["volatility"]
    drift = config["drift"]
    
    # Generate geometric random walk
    returns = np.random.normal(drift, volatility, num_candles)
    
    # Add a cointegration component for related assets
    # This makes BTC and ICP move somewhat together (for realistic testing)
    if symbol.upper() in ["ICP", "ETH", "SOL"]:
        # Add correlation with a "market factor"
        np.random.seed(42 if seed is None else seed)  # Use consistent seed for market factor
        market_factor = np.random.normal(0, volatility * 0.5, num_candles)
        returns = returns * 0.6 + market_factor * 0.4
    
    # Convert returns to prices
    prices = [base_price]
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))
    
    return prices


def fetch_candles_cointegrated(
    symbol_a: str = "BTC",
    symbol_b: str = "ICP",
    num_candles: int = 100,
    seed: int = 42,
    cointegration_strength: float = 0.8
) -> tuple:
    """
    Fetch mock price data for two cointegrated assets.
    
    This is specifically designed for testing the cointegration strategy
    by generating two price series that are statistically cointegrated.
    
    Args:
        symbol_a: First symbol (e.g., "BTC")
        symbol_b: Second symbol (e.g., "ICP")
        num_candles: Number of price points
        seed: Random seed for reproducibility
        cointegration_strength: How strongly the assets are cointegrated (0-1)
        
    Returns:
        Tuple of (prices_a, prices_b) as lists of floats
    """
    np.random.seed(seed)
    
    asset_configs = {
        "BTC": {"base_price": 42000.0, "scale": 1.0},
        "ICP": {"base_price": 12.5, "scale": 1/3360},  # ICP ~ BTC/3360
    }
    
    config_a = asset_configs.get(symbol_a.upper(), {"base_price": 100.0, "scale": 1.0})
    config_b = asset_configs.get(symbol_b.upper(), {"base_price": 10.0, "scale": 0.1})
    
    # Generate a common random walk (the cointegrating relationship)
    common_trend = np.cumsum(np.random.normal(0, 0.02, num_candles))
    
    # Generate idiosyncratic noise for each asset
    noise_a = np.cumsum(np.random.normal(0, 0.01, num_candles))
    noise_b = np.cumsum(np.random.normal(0, 0.01, num_candles))
    
    # Combine: prices follow common trend + some noise
    prices_a = config_a["base_price"] * np.exp(
        cointegration_strength * common_trend + 
        (1 - cointegration_strength) * noise_a
    )
    
    prices_b = config_b["base_price"] * np.exp(
        cointegration_strength * common_trend + 
        (1 - cointegration_strength) * noise_b
    )
    
    return list(prices_a), list(prices_b)


# Convenience function for quick testing
def get_sample_data() -> dict:
    """
    Get sample price data for quick testing.
    
    Returns:
        Dictionary with BTC and ICP sample price series
    """
    btc, icp = fetch_candles_cointegrated("BTC", "ICP", num_candles=100, seed=42)
    return {
        "BTC": btc,
        "ICP": icp,
        "timestamp": "2024-12-17T00:00:00Z",
        "source": "mock_data"
    }
