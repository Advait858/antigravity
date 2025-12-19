"""Data module for Antigravity trading bot."""

from .loader import fetch_candles, fetch_candles_cointegrated, get_sample_data

__all__ = ["fetch_candles", "fetch_candles_cointegrated", "get_sample_data"]
