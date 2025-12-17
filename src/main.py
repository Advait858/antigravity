"""
Antigravity Bot - ICP Algo-Trading Agent
Main canister entry point for the Kybra CDK
"""

from kybra import query, update, void


class AntigravityBot:
    """
    Main trading bot class for the Antigravity cointegration strategy.
    Deployed as a canister on the Internet Computer.
    """
    
    def __init__(self):
        self.status = "initialized"
    
    @staticmethod
    @query
    def get_health() -> str:
        """
        Health check endpoint to verify the canister is operational.
        Returns a status message.
        """
        return "System Operational"
    
    @staticmethod
    @query
    def get_version() -> str:
        """
        Returns the current version of the Antigravity Bot.
        """
        return "0.1.0"


# Canister entry points
@query
def get_health() -> str:
    """Query method to check system health."""
    return AntigravityBot.get_health()


@query
def get_version() -> str:
    """Query method to get bot version."""
    return AntigravityBot.get_version()


@update
def execute_strategy() -> str:
    """
    Update method to execute the cointegration trading strategy.
    This is a placeholder that will be connected to the strategy logic.
    
    Note: Strategy imports are done locally to avoid namespace issues
    during Kybra compilation.
    """
    try:
        # Import strategy modules only when needed
        from strategy.cointegration import PairsTrader
        from data.loader import fetch_candles
        
        # Fetch mock data for testing
        btc_data = fetch_candles("BTC")
        icp_data = fetch_candles("ICP")
        
        # Initialize the pairs trader
        trader = PairsTrader()
        
        # Calculate spread between BTC and ICP
        spread = trader.calculate_spread(btc_data, icp_data)
        
        # Check if the spread is stationary (cointegrated)
        result = trader.check_stationarity(spread)
        
        return f"Strategy executed. Cointegration p-value: {result['p_value']:.4f}, Is Cointegrated: {result['is_cointegrated']}"
    
    except Exception as e:
        return f"Strategy execution failed: {str(e)}"


@query
def get_strategy_info() -> str:
    """Query method to get information about the trading strategy."""
    return "ADF Cointegration Model - Statistical Arbitrage between crypto pairs (BTC/ICP)"
