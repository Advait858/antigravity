"""
Antigravity Risk - Circuit Breaker Implementation
Defensive halt mechanism for pairs trading safety.

Triggers:
1. ADF Failure - Cointegration breaks (p >= 0.05)
2. Stop-Loss - Spread deviation exceeds 4σ
3. Consecutive Losses - Too many losing trades
4. Drawdown Limit - Portfolio drawdown exceeds threshold
"""

from typing import Optional, List
from dataclasses import dataclass
from enum import Enum
import time


class BreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Trading halted
    HALF_OPEN = "half_open"  # Testing if conditions improved


class BreakerTrigger(Enum):
    """Reasons for circuit breaker activation."""
    NONE = "none"
    ADF_FAILURE = "adf_failure"
    STOP_LOSS_4_SIGMA = "stop_loss_4_sigma"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    MAX_DRAWDOWN = "max_drawdown"
    MANUAL = "manual"


@dataclass
class BreakerStatus:
    """Current circuit breaker status."""
    state: BreakerState
    trigger: BreakerTrigger
    triggered_at: float  # Timestamp
    cooldown_remaining: float  # Seconds until can test half-open
    message: str


class CircuitBreaker:
    """
    Circuit breaker for pairs trading safety.
    
    Protects against:
    - Cointegration breakdown (ADF failure)
    - Extreme spread deviations (>4σ)
    - Consecutive losing trades
    - Maximum drawdown limits
    
    States:
    - CLOSED: Normal trading allowed
    - OPEN: Trading halted, waiting for cooldown
    - HALF_OPEN: Testing conditions before full recovery
    """
    
    def __init__(
        self,
        adf_p_threshold: float = 0.05,
        stop_loss_sigma: float = 4.0,
        max_consecutive_losses: int = 5,
        max_drawdown_pct: float = 0.10,
        cooldown_seconds: float = 300.0,  # 5 minutes
        half_open_test_trades: int = 3
    ):
        """
        Initialize circuit breaker.
        
        Args:
            adf_p_threshold: Max p-value for ADF test (default 0.05)
            stop_loss_sigma: Sigma level for stop-loss (default 4.0)
            max_consecutive_losses: Max losing trades before halt
            max_drawdown_pct: Max drawdown percentage (0-1)
            cooldown_seconds: Seconds before testing recovery
            half_open_test_trades: Trades to test in half-open state
        """
        self.adf_p_threshold = adf_p_threshold
        self.stop_loss_sigma = stop_loss_sigma
        self.max_consecutive_losses = max_consecutive_losses
        self.max_drawdown_pct = max_drawdown_pct
        self.cooldown_seconds = cooldown_seconds
        self.half_open_test_trades = half_open_test_trades
        
        # State
        self.state = BreakerState.CLOSED
        self.trigger = BreakerTrigger.NONE
        self.triggered_at: Optional[float] = None
        
        # Metrics
        self.consecutive_losses = 0
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.half_open_successes = 0
        
        # History
        self.trigger_history: List[dict] = []
    
    def check_and_update(
        self,
        adf_p_value: float,
        z_score: float,
        trade_pnl: Optional[float] = None,
        current_equity: Optional[float] = None
    ) -> BreakerStatus:
        """
        Check conditions and update circuit breaker state.
        
        Args:
            adf_p_value: Current ADF test p-value
            z_score: Current spread z-score
            trade_pnl: P&L of most recent trade (if any)
            current_equity: Current portfolio equity
            
        Returns:
            BreakerStatus with current state and details
        """
        current_time = time.time()
        
        # Update equity tracking
        if current_equity is not None:
            self.current_equity = current_equity
            if current_equity > self.peak_equity:
                self.peak_equity = current_equity
        
        # Update consecutive losses
        if trade_pnl is not None:
            if trade_pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
        
        # If currently OPEN, check if cooldown passed
        if self.state == BreakerState.OPEN:
            if self.triggered_at is not None:
                elapsed = current_time - self.triggered_at
                if elapsed >= self.cooldown_seconds:
                    self.state = BreakerState.HALF_OPEN
                    self.half_open_successes = 0
        
        # If HALF_OPEN, check conditions for full recovery
        if self.state == BreakerState.HALF_OPEN:
            # Check if conditions are now good
            conditions_ok = (
                adf_p_value < self.adf_p_threshold and
                abs(z_score) < self.stop_loss_sigma
            )
            
            if conditions_ok:
                self.half_open_successes += 1
                if self.half_open_successes >= self.half_open_test_trades:
                    self._close_breaker()
            else:
                # Conditions failed, back to OPEN
                self._open_breaker(BreakerTrigger.ADF_FAILURE if adf_p_value >= self.adf_p_threshold else BreakerTrigger.STOP_LOSS_4_SIGMA)
        
        # If CLOSED, check all trigger conditions
        if self.state == BreakerState.CLOSED:
            # Check ADF failure
            if adf_p_value >= self.adf_p_threshold:
                self._open_breaker(BreakerTrigger.ADF_FAILURE)
            
            # Check 4σ stop-loss
            elif abs(z_score) >= self.stop_loss_sigma:
                self._open_breaker(BreakerTrigger.STOP_LOSS_4_SIGMA)
            
            # Check consecutive losses
            elif self.consecutive_losses >= self.max_consecutive_losses:
                self._open_breaker(BreakerTrigger.CONSECUTIVE_LOSSES)
            
            # Check max drawdown
            elif self.peak_equity > 0 and self.current_equity > 0:
                drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
                if drawdown >= self.max_drawdown_pct:
                    self._open_breaker(BreakerTrigger.MAX_DRAWDOWN)
        
        return self.get_status()
    
    def _open_breaker(self, trigger: BreakerTrigger) -> None:
        """Open the circuit breaker (halt trading)."""
        self.state = BreakerState.OPEN
        self.trigger = trigger
        self.triggered_at = time.time()
        
        self.trigger_history.append({
            "trigger": trigger.value,
            "timestamp": self.triggered_at,
            "consecutive_losses": self.consecutive_losses,
            "peak_equity": self.peak_equity,
            "current_equity": self.current_equity
        })
        
        # Keep history bounded
        if len(self.trigger_history) > 100:
            self.trigger_history = self.trigger_history[-50:]
    
    def _close_breaker(self) -> None:
        """Close the circuit breaker (resume trading)."""
        self.state = BreakerState.CLOSED
        self.trigger = BreakerTrigger.NONE
        self.triggered_at = None
        self.consecutive_losses = 0
        self.half_open_successes = 0
    
    def force_halt(self, reason: str = "Manual halt") -> BreakerStatus:
        """Force trading halt manually."""
        self._open_breaker(BreakerTrigger.MANUAL)
        return self.get_status()
    
    def force_resume(self) -> BreakerStatus:
        """Force resume trading (use with caution)."""
        self._close_breaker()
        return self.get_status()
    
    def is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed."""
        return self.state == BreakerState.CLOSED
    
    def get_status(self) -> BreakerStatus:
        """Get current circuit breaker status."""
        cooldown_remaining = 0.0
        if self.state == BreakerState.OPEN and self.triggered_at is not None:
            elapsed = time.time() - self.triggered_at
            cooldown_remaining = max(0.0, self.cooldown_seconds - elapsed)
        
        messages = {
            BreakerTrigger.NONE: "Trading allowed",
            BreakerTrigger.ADF_FAILURE: "HALT: Cointegration broken (ADF p >= 0.05)",
            BreakerTrigger.STOP_LOSS_4_SIGMA: "HALT: Stop-loss triggered (|Z| >= 4σ)",
            BreakerTrigger.CONSECUTIVE_LOSSES: f"HALT: {self.max_consecutive_losses} consecutive losses",
            BreakerTrigger.MAX_DRAWDOWN: f"HALT: Max drawdown exceeded ({self.max_drawdown_pct*100:.1f}%)",
            BreakerTrigger.MANUAL: "HALT: Manual override"
        }
        
        return BreakerStatus(
            state=self.state,
            trigger=self.trigger,
            triggered_at=self.triggered_at or 0.0,
            cooldown_remaining=cooldown_remaining,
            message=messages.get(self.trigger, "Unknown state")
        )


def test_circuit_breaker():
    """Test circuit breaker functionality."""
    cb = CircuitBreaker(cooldown_seconds=1.0)  # Short cooldown for testing
    
    print("Test 1: Normal conditions")
    status = cb.check_and_update(adf_p_value=0.01, z_score=1.5)
    print(f"  State: {status.state.value}, Message: {status.message}")
    
    print("\nTest 2: ADF failure")
    status = cb.check_and_update(adf_p_value=0.10, z_score=1.5)
    print(f"  State: {status.state.value}, Message: {status.message}")
    
    print("\nTest 3: Wait for cooldown...")
    import time
    time.sleep(1.5)
    status = cb.check_and_update(adf_p_value=0.01, z_score=1.0)
    print(f"  State: {status.state.value}, Message: {status.message}")
    
    print("\nTest 4: 4σ stop-loss")
    cb.force_resume()
    status = cb.check_and_update(adf_p_value=0.01, z_score=4.5)
    print(f"  State: {status.state.value}, Message: {status.message}")
    
    return cb


if __name__ == "__main__":
    test_circuit_breaker()
