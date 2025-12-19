"""
Antigravity Engine - Kalman Filter Implementation
Pure Python implementation for dynamic hedge ratio estimation.
No external dependencies - WASM compatible for ICP canisters.

Mathematical Foundation:
- State: β (hedge ratio)
- Measurement: spread = price_A - β * price_B
- Process noise: Q (how much β can change)
- Measurement noise: R (observation uncertainty)

Kalman Update Equations:
- Prediction: β_pred = β_prev, P_pred = P_prev + Q
- Update: K = P_pred / (P_pred + R)
- New state: β = β_pred + K * (y - x * β_pred)
- New variance: P = (1 - K) * P_pred
"""

from typing import Tuple, List, Optional
from dataclasses import dataclass
import math


@dataclass
class KalmanState:
    """State container for Kalman Filter."""
    beta: float  # Hedge ratio estimate
    P: float     # Estimation error covariance
    Q: float     # Process noise covariance
    R: float     # Measurement noise covariance
    

class KalmanFilter:
    """
    Kalman Filter for dynamic hedge ratio estimation in pairs trading.
    
    The hedge ratio β represents: spread = price_A - β * price_B
    A Kalman filter adaptively estimates β as market conditions change.
    
    Usage:
        kf = KalmanFilter()
        for price_a, price_b in prices:
            beta, spread = kf.update(price_a, price_b)
    """
    
    def __init__(
        self,
        initial_beta: float = 1.0,
        initial_P: float = 1.0,
        Q: float = 0.0001,  # Process noise - how much beta can change
        R: float = 0.001    # Measurement noise - observation uncertainty
    ):
        """
        Initialize Kalman Filter.
        
        Args:
            initial_beta: Initial hedge ratio guess
            initial_P: Initial estimation uncertainty
            Q: Process noise (higher = more adaptive, but noisier)
            R: Measurement noise (higher = smoother, but slower)
        """
        self.state = KalmanState(
            beta=initial_beta,
            P=initial_P,
            Q=Q,
            R=R
        )
        self.history: List[Tuple[float, float, float]] = []  # (beta, spread, variance)
        
    def update(self, price_a: float, price_b: float) -> Tuple[float, float]:
        """
        Update Kalman filter with new price observations.
        
        Args:
            price_a: Price of asset A (e.g., BTC)
            price_b: Price of asset B (e.g., ICP)
            
        Returns:
            Tuple of (updated_beta, spread)
        """
        if price_b == 0:
            return self.state.beta, 0.0
            
        # Prediction step
        beta_pred = self.state.beta
        P_pred = self.state.P + self.state.Q
        
        # Measurement: y = price_a, x = price_b
        # Model: y = beta * x + noise
        y = price_a
        x = price_b
        
        # Innovation (measurement residual)
        innovation = y - beta_pred * x
        
        # Innovation covariance: S = x² * P + R
        S = x * x * P_pred + self.state.R
        
        # Kalman gain
        if S != 0:
            K = (P_pred * x) / S
        else:
            K = 0
        
        # Update state
        self.state.beta = beta_pred + K * innovation
        self.state.P = (1 - K * x) * P_pred
        
        # Calculate spread with updated beta
        spread = price_a - self.state.beta * price_b
        
        # Store history for analysis
        self.history.append((self.state.beta, spread, self.state.P))
        
        # Keep history bounded
        if len(self.history) > 1000:
            self.history = self.history[-500:]
            
        return self.state.beta, spread
    
    def get_spread_statistics(self, window: int = 50) -> Tuple[float, float]:
        """
        Calculate rolling mean and std of spread.
        
        Args:
            window: Lookback period for statistics
            
        Returns:
            Tuple of (mean, std)
        """
        if len(self.history) < 2:
            return 0.0, 1.0
            
        spreads = [h[1] for h in self.history[-window:]]
        n = len(spreads)
        
        mean = sum(spreads) / n
        variance = sum((s - mean) ** 2 for s in spreads) / n
        std = math.sqrt(variance) if variance > 0 else 1.0
        
        return mean, std
    
    def get_z_score(self, window: int = 50) -> float:
        """
        Calculate current Z-score of the spread.
        
        Z = (spread - mean) / std
        
        Returns:
            Current Z-score
        """
        if not self.history:
            return 0.0
            
        current_spread = self.history[-1][1]
        mean, std = self.get_spread_statistics(window)
        
        if std == 0:
            return 0.0
            
        return (current_spread - mean) / std
    
    def reset(self) -> None:
        """Reset filter to initial state."""
        self.state.beta = 1.0
        self.state.P = 1.0
        self.history.clear()
        
    def get_beta(self) -> float:
        """Get current hedge ratio estimate."""
        return self.state.beta
    
    def get_estimation_confidence(self) -> float:
        """
        Get confidence in current beta estimate (0-1).
        Lower P = higher confidence.
        """
        # Normalize P to 0-1 range (assuming P typically < 1)
        confidence = 1.0 - min(self.state.P, 1.0)
        return max(0.0, confidence)


def test_kalman_filter():
    """Test the Kalman filter with synthetic data."""
    kf = KalmanFilter()
    
    # Simulate cointegrated prices
    # price_b follows random walk, price_a = 2 * price_b + noise
    import random
    random.seed(42)
    
    price_b = 100.0
    true_beta = 2.0
    
    results = []
    for i in range(100):
        price_b += random.gauss(0, 1)  # Random walk
        noise = random.gauss(0, 0.5)
        price_a = true_beta * price_b + noise
        
        beta, spread = kf.update(price_a, price_b)
        z_score = kf.get_z_score()
        
        results.append({
            "step": i,
            "true_beta": true_beta,
            "estimated_beta": beta,
            "spread": spread,
            "z_score": z_score
        })
    
    # Check convergence
    final_beta = results[-1]["estimated_beta"]
    print(f"True beta: {true_beta}, Estimated: {final_beta:.4f}")
    print(f"Beta error: {abs(true_beta - final_beta):.4f}")
    
    return results


if __name__ == "__main__":
    test_kalman_filter()
