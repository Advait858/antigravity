"""
DEX Simulation Engine
Simulates realistic DEX trading with fees and slippage

Features:
- 0.3% LP fee simulation
- 0.001 ICP gas fee
- Quadratic slippage for large trades
- ICPSwap adapter interface (ready for real integration)
"""

import math
from typing import Optional

# ============================================================
# CONFIGURATION
# ============================================================

# Fee structure (matches ICPSwap/Sonic)
LP_FEE_PERCENT = 0.3        # 0.3% liquidity provider fee
GAS_FEE_ICP = 0.001         # 0.001 ICP per transaction
ICP_PRICE_USD = 10.50       # Approximate ICP price for gas calculation

# Slippage model parameters
SLIPPAGE_THRESHOLD = 10000  # Apply slippage for trades > $10,000
SLIPPAGE_COEFFICIENT = 0.00001  # k in quadratic model

# Paper trading defaults
DEFAULT_BALANCE = 100000.0  # $100,000 USDC starting balance


# ============================================================
# FEE CALCULATION
# ============================================================

def calculate_lp_fee(trade_amount: float) -> float:
    """Calculate LP fee (0.3% of trade amount)"""
    return trade_amount * (LP_FEE_PERCENT / 100)


def calculate_gas_fee_usd() -> float:
    """Calculate gas fee in USD"""
    return GAS_FEE_ICP * ICP_PRICE_USD


def calculate_slippage(trade_amount: float, is_buy: bool) -> float:
    """
    Calculate slippage using quadratic model.
    
    For trades > $10,000:
    Price_exec = Price_spot * (1 + k * Volume)
    
    Returns slippage as a decimal (e.g., 0.002 = 0.2%)
    """
    if trade_amount <= SLIPPAGE_THRESHOLD:
        return 0.0
    
    excess = trade_amount - SLIPPAGE_THRESHOLD
    slippage = SLIPPAGE_COEFFICIENT * (excess ** 1.5)
    
    # Cap at 2%
    return min(0.02, slippage)


def calculate_total_fees(trade_amount: float) -> dict:
    """Calculate all fees for a trade"""
    lp_fee = calculate_lp_fee(trade_amount)
    gas_fee = calculate_gas_fee_usd()
    slippage_pct = calculate_slippage(trade_amount, True)
    slippage_cost = trade_amount * slippage_pct
    
    return {
        "lp_fee": round(lp_fee, 4),
        "gas_fee": round(gas_fee, 4),
        "slippage_pct": round(slippage_pct * 100, 4),
        "slippage_cost": round(slippage_cost, 4),
        "total_fees": round(lp_fee + gas_fee + slippage_cost, 4),
        "effective_amount": round(trade_amount - lp_fee - gas_fee - slippage_cost, 4)
    }


# ============================================================
# DEX TRADE EXECUTION
# ============================================================

class DEXSimulator:
    """Simulates DEX trading with realistic costs"""
    
    def __init__(self):
        self.trade_count = 0
        self.total_fees_paid = 0.0
        self.total_slippage = 0.0
    
    def execute_swap(
        self,
        from_token: str,
        to_token: str,
        amount: float,
        spot_price: float
    ) -> dict:
        """
        Execute a simulated swap.
        
        Args:
            from_token: Token selling (e.g., "USDC")
            to_token: Token buying (e.g., "BTC")
            amount: Amount in from_token
            spot_price: Current spot price of to_token in from_token
        
        Returns:
            Trade execution result with fees and filled amount
        """
        # Calculate fees
        fees = calculate_total_fees(amount)
        
        # Calculate execution price with slippage
        slippage = fees["slippage_pct"] / 100
        exec_price = spot_price * (1 + slippage)
        
        # Amount after fees
        effective_amount = fees["effective_amount"]
        
        # Tokens received
        tokens_received = effective_amount / exec_price if exec_price > 0 else 0
        
        self.trade_count += 1
        self.total_fees_paid += fees["total_fees"]
        self.total_slippage += fees["slippage_cost"]
        
        return {
            "success": True,
            "trade_id": self.trade_count,
            "from_token": from_token,
            "to_token": to_token,
            "amount_in": amount,
            "amount_out": round(tokens_received, 8),
            "spot_price": spot_price,
            "exec_price": round(exec_price, 8),
            "fees": fees,
            "timestamp": None  # Set by caller with ic.time()
        }
    
    def execute_pair_trade(
        self,
        asset_a: str,
        asset_b: str,
        price_a: float,
        price_b: float,
        position_size: float,
        action: str  # "LONG_SPREAD" or "SHORT_SPREAD"
    ) -> dict:
        """
        Execute a pairs trade (two simultaneous swaps).
        
        LONG_SPREAD: Buy A, Sell B
        SHORT_SPREAD: Sell A, Buy B
        
        Position is split 50/50 between the two legs.
        """
        leg_size = position_size / 2
        
        if action == "LONG_SPREAD":
            # Leg 1: Buy A with USDC
            leg1 = self.execute_swap("USDC", asset_a, leg_size, price_a)
            # Leg 2: Sell B for USDC (short simulation)
            leg2 = self.execute_swap(asset_b, "USDC", leg_size / price_b, price_b)
        else:
            # Leg 1: Sell A for USDC (short simulation)
            leg1 = self.execute_swap(asset_a, "USDC", leg_size / price_a, price_a)
            # Leg 2: Buy B with USDC
            leg2 = self.execute_swap("USDC", asset_b, leg_size, price_b)
        
        total_fees = leg1["fees"]["total_fees"] + leg2["fees"]["total_fees"]
        
        return {
            "success": True,
            "action": action,
            "pair": f"{asset_a}/{asset_b}",
            "position_size": position_size,
            "leg1": leg1,
            "leg2": leg2,
            "total_fees": round(total_fees, 4),
            "effective_position": round(position_size - total_fees, 4)
        }
    
    def get_stats(self) -> dict:
        return {
            "total_trades": self.trade_count,
            "total_fees_paid": round(self.total_fees_paid, 2),
            "total_slippage": round(self.total_slippage, 2),
            "avg_fee_per_trade": round(self.total_fees_paid / self.trade_count, 2) if self.trade_count > 0 else 0
        }


# ============================================================
# USER PORTFOLIO
# ============================================================

class UserPortfolio:
    """Manages user paper trading portfolio"""
    
    def __init__(self, principal: str, initial_balance: float = DEFAULT_BALANCE):
        self.principal = principal
        self.cash = initial_balance
        self.initial_balance = initial_balance
        self.positions = {}  # {asset: {"qty": float, "avg_price": float, "side": str}}
        self.trade_history = []
        self.realized_pnl = 0.0
    
    def can_afford(self, amount: float) -> bool:
        return self.cash >= amount
    
    def open_position(self, trade_result: dict) -> dict:
        """Record a new position from trade execution"""
        if trade_result["position_size"] > self.cash:
            return {"success": False, "error": "Insufficient funds"}
        
        # Deduct cash (including fees)
        cost = trade_result["position_size"]
        self.cash -= cost
        
        # Record position
        pair = trade_result["pair"]
        self.positions[pair] = {
            "action": trade_result["action"],
            "entry_size": trade_result["effective_position"],
            "fees_paid": trade_result["total_fees"],
            "entry_time": None,
            "current_pnl": 0.0
        }
        
        self.trade_history.append({
            "type": "OPEN",
            "pair": pair,
            "size": trade_result["position_size"],
            "fees": trade_result["total_fees"]
        })
        
        return {"success": True, "remaining_cash": self.cash}
    
    def close_position(self, pair: str, exit_pnl: float) -> dict:
        """Close a position and realize P&L"""
        if pair not in self.positions:
            return {"success": False, "error": "No open position"}
        
        position = self.positions[pair]
        
        # Return capital + P&L (minus exit fees)
        exit_fees = calculate_total_fees(position["entry_size"])["total_fees"]
        net_pnl = exit_pnl - exit_fees
        
        self.cash += position["entry_size"] + net_pnl
        self.realized_pnl += net_pnl
        
        del self.positions[pair]
        
        self.trade_history.append({
            "type": "CLOSE",
            "pair": pair,
            "pnl": net_pnl,
            "fees": exit_fees
        })
        
        return {"success": True, "net_pnl": net_pnl, "cash": self.cash}
    
    def get_summary(self) -> dict:
        return {
            "principal": self.principal,
            "cash": round(self.cash, 2),
            "initial_balance": self.initial_balance,
            "open_positions": len(self.positions),
            "realized_pnl": round(self.realized_pnl, 2),
            "total_value": round(self.cash + sum(p["entry_size"] for p in self.positions.values()), 2)
        }


# ============================================================
# ICPSWAP ADAPTER INTERFACE (Stub for Real Integration)
# ============================================================

class ICPSwapAdapter:
    """
    Interface for real ICPSwap integration.
    
    IMPORTANT: This is a stub. For production:
    1. Import the ICPSwap canister interface
    2. Use ic.call() for inter-canister calls
    3. Handle ICRC-1 token approvals
    
    ICPSwap Canister IDs (Mainnet):
    - Swap Router: 5qhvy-7qaaa-aaaag-qbzra-cai
    - Quote: 5qkw5-5yaaa-aaaag-qbzsa-cai
    """
    
    def __init__(self, canister_id: str = "5qhvy-7qaaa-aaaag-qbzra-cai"):
        self.canister_id = canister_id
        self.is_live = False  # Set to True when real integration is ready
    
    async def get_quote(self, token_in: str, token_out: str, amount: int) -> dict:
        """
        Get swap quote from ICPSwap.
        
        TODO: Implement real inter-canister call:
        ```python
        from kybra import ic
        
        result = await ic.call(
            self.canister_id,
            "quote",
            {"token_in": token_in, "token_out": token_out, "amount": amount}
        )
        ```
        """
        # Stub: Return simulated quote
        return {
            "amount_out": amount * 0.997,  # 0.3% fee
            "price_impact": 0.001,
            "route": [token_in, token_out]
        }
    
    async def execute_swap(self, token_in: str, token_out: str, amount: int, min_amount_out: int) -> dict:
        """
        Execute real swap on ICPSwap.
        
        TODO: Implement real inter-canister call:
        ```python
        # Step 1: Approve token spend
        await ic.call(token_in_canister, "icrc2_approve", {...})
        
        # Step 2: Execute swap
        result = await ic.call(
            self.canister_id,
            "swap",
            {
                "token_in": token_in,
                "token_out": token_out,
                "amount_in": amount,
                "amount_out_min": min_amount_out
            }
        )
        ```
        """
        # Stub: Return simulated execution
        return {
            "success": False,
            "error": "Real ICPSwap integration not enabled",
            "simulated": True
        }


# ============================================================
# GLOBAL INSTANCES
# ============================================================

dex_simulator = DEXSimulator()
icpswap_adapter = ICPSwapAdapter()
