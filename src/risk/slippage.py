"""
Antigravity Risk - Slippage and Cost Modeling
Pre-trade checks for realistic execution.

Models:
- Slippage based on trade size and liquidity
- Gas/fee costs for ICP transactions
- Minimum profit threshold after costs
"""

from typing import Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class SlippageEstimate:
    """Estimated slippage for a trade."""
    expected_slippage_pct: float
    worst_case_slippage_pct: float
    estimated_fill_price: float
    liquidity_score: float  # 0-1, higher = more liquid


@dataclass
class TradeCosts:
    """Total estimated costs for a trade."""
    slippage_cost: float
    gas_fee: float
    dex_fee: float
    total_cost: float
    min_profit_required: float


class SlippageModel:
    """
    Slippage model for pairs trading execution.
    
    Estimates slippage based on:
    - Trade size relative to available liquidity
    - Market depth approximation
    - Historical volatility
    """
    
    def __init__(
        self,
        base_slippage_bps: float = 10.0,  # Base slippage in basis points
        size_impact_factor: float = 0.5,   # How much size affects slippage
        volatility_multiplier: float = 2.0  # Multiplier for volatile markets
    ):
        """
        Initialize slippage model.
        
        Args:
            base_slippage_bps: Base slippage in basis points (1bp = 0.01%)
            size_impact_factor: Coefficient for size impact
            volatility_multiplier: Multiplier for high volatility
        """
        self.base_slippage_bps = base_slippage_bps
        self.size_impact_factor = size_impact_factor
        self.volatility_multiplier = volatility_multiplier
    
    def estimate_slippage(
        self,
        trade_size_usd: float,
        daily_volume_usd: float,
        volatility: float = 0.02,  # Daily volatility
        is_sell: bool = False
    ) -> SlippageEstimate:
        """
        Estimate slippage for a trade.
        
        Args:
            trade_size_usd: Size of trade in USD
            daily_volume_usd: 24h trading volume in USD
            volatility: Daily volatility (e.g., 0.02 = 2%)
            is_sell: True if selling (higher slippage in less liquid markets)
            
        Returns:
            SlippageEstimate with expected and worst-case slippage
        """
        if daily_volume_usd <= 0:
            return SlippageEstimate(
                expected_slippage_pct=1.0,  # 1% if no volume data
                worst_case_slippage_pct=5.0,
                estimated_fill_price=0.0,
                liquidity_score=0.0
            )
        
        # Size as fraction of daily volume
        size_fraction = trade_size_usd / daily_volume_usd
        
        # Base slippage (convert bps to percentage)
        base_slip = self.base_slippage_bps / 100.0
        
        # Size impact: slippage increases with square root of size fraction
        size_impact = self.size_impact_factor * math.sqrt(size_fraction) * 100
        
        # Volatility adjustment
        vol_adjustment = 1.0 + (volatility - 0.02) * self.volatility_multiplier
        vol_adjustment = max(0.5, min(3.0, vol_adjustment))  # Clamp
        
        # Expected slippage
        expected_slippage = (base_slip + size_impact) * vol_adjustment
        
        # Sells typically have higher slippage
        if is_sell:
            expected_slippage *= 1.1
        
        # Worst case: 2x expected
        worst_case = expected_slippage * 2.0
        
        # Liquidity score based on how much of volume we're taking
        liquidity_score = max(0.0, 1.0 - size_fraction * 10)
        
        return SlippageEstimate(
            expected_slippage_pct=expected_slippage,
            worst_case_slippage_pct=worst_case,
            estimated_fill_price=0.0,  # Would need current price
            liquidity_score=liquidity_score
        )
    
    def calculate_fill_price(
        self,
        current_price: float,
        slippage_pct: float,
        is_buy: bool
    ) -> float:
        """Calculate expected fill price after slippage."""
        if is_buy:
            return current_price * (1 + slippage_pct / 100.0)
        else:
            return current_price * (1 - slippage_pct / 100.0)


class CostCalculator:
    """
    Calculate total trade costs including slippage, gas, and DEX fees.
    """
    
    def __init__(
        self,
        icp_gas_fee: float = 0.0001,  # ICP per transaction
        dex_fee_bps: float = 30.0,     # DEX fee in basis points (0.3%)
        min_profit_multiplier: float = 2.0  # Require 2x costs for profit
    ):
        """
        Initialize cost calculator.
        
        Args:
            icp_gas_fee: ICP cost per transaction
            dex_fee_bps: DEX fee in basis points
            min_profit_multiplier: Required profit as multiple of costs
        """
        self.icp_gas_fee = icp_gas_fee
        self.dex_fee_bps = dex_fee_bps
        self.min_profit_multiplier = min_profit_multiplier
        self.slippage_model = SlippageModel()
    
    def calculate_costs(
        self,
        trade_size_usd: float,
        daily_volume_usd: float,
        icp_price_usd: float,
        volatility: float = 0.02
    ) -> TradeCosts:
        """
        Calculate total costs for a round-trip trade.
        
        Args:
            trade_size_usd: Trade size in USD
            daily_volume_usd: 24h volume in USD
            icp_price_usd: Current ICP price for gas calculation
            volatility: Daily volatility
            
        Returns:
            TradeCosts with all cost breakdowns
        """
        # Estimate slippage (for both entry and exit)
        slip_est = self.slippage_model.estimate_slippage(
            trade_size_usd, daily_volume_usd, volatility
        )
        
        # Slippage cost for round-trip (entry + exit)
        slippage_cost = trade_size_usd * (slip_est.expected_slippage_pct / 100.0) * 2
        
        # Gas fees (2 transactions: entry + exit)
        gas_fee = self.icp_gas_fee * icp_price_usd * 2
        
        # DEX fees for round-trip
        dex_fee = trade_size_usd * (self.dex_fee_bps / 10000.0) * 2
        
        # Total cost
        total_cost = slippage_cost + gas_fee + dex_fee
        
        # Minimum required profit
        min_profit = total_cost * self.min_profit_multiplier
        
        return TradeCosts(
            slippage_cost=slippage_cost,
            gas_fee=gas_fee,
            dex_fee=dex_fee,
            total_cost=total_cost,
            min_profit_required=min_profit
        )
    
    def is_trade_profitable(
        self,
        expected_profit_usd: float,
        trade_size_usd: float,
        daily_volume_usd: float,
        icp_price_usd: float
    ) -> Tuple[bool, str]:
        """
        Check if a trade is expected to be profitable after costs.
        
        Returns:
            Tuple of (is_profitable, reason_message)
        """
        costs = self.calculate_costs(
            trade_size_usd, daily_volume_usd, icp_price_usd
        )
        
        if expected_profit_usd < costs.min_profit_required:
            return False, f"Expected profit ${expected_profit_usd:.2f} < min required ${costs.min_profit_required:.2f}"
        
        net_profit = expected_profit_usd - costs.total_cost
        roi = (net_profit / trade_size_usd) * 100
        
        return True, f"Net profit: ${net_profit:.2f} (ROI: {roi:.2f}%)"


def test_slippage_model():
    """Test slippage and cost calculations."""
    calc = CostCalculator()
    
    print("Test 1: Small trade in liquid market")
    costs = calc.calculate_costs(
        trade_size_usd=1000,
        daily_volume_usd=10_000_000,
        icp_price_usd=12.0
    )
    print(f"  Slippage: ${costs.slippage_cost:.2f}")
    print(f"  Gas: ${costs.gas_fee:.4f}")
    print(f"  DEX fee: ${costs.dex_fee:.2f}")
    print(f"  Total: ${costs.total_cost:.2f}")
    print(f"  Min profit needed: ${costs.min_profit_required:.2f}")
    
    print("\nTest 2: Large trade in illiquid market")
    costs = calc.calculate_costs(
        trade_size_usd=10000,
        daily_volume_usd=100_000,
        icp_price_usd=12.0
    )
    print(f"  Slippage: ${costs.slippage_cost:.2f}")
    print(f"  Total: ${costs.total_cost:.2f}")
    print(f"  Min profit needed: ${costs.min_profit_required:.2f}")
    
    print("\nTest 3: Profitability check")
    is_profitable, msg = calc.is_trade_profitable(
        expected_profit_usd=50,
        trade_size_usd=1000,
        daily_volume_usd=10_000_000,
        icp_price_usd=12.0
    )
    print(f"  Profitable: {is_profitable}")
    print(f"  Reason: {msg}")
    
    return calc


if __name__ == "__main__":
    test_slippage_model()
