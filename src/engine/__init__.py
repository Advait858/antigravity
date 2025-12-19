"""
Antigravity Engine Module
Pure Python implementations for pairs trading.
"""

from .kalman import KalmanFilter, KalmanState
from .adf import adf_test, ADFResult

__all__ = [
    "KalmanFilter",
    "KalmanState", 
    "adf_test",
    "ADFResult"
]
