"""
Oracle Module - Live Market Data Ingestion
Fetches real-time prices from Binance and CoinGecko

Sources:
- Primary: Binance Public API (/api/v3/ticker/price)
- Fallback: CoinGecko Simple Price API
"""

from kybra import ic, update, query
import json

# ============================================================
# CONFIGURATION
# ============================================================

BINANCE_API = "https://api.binance.com/api/v3/ticker/price"
COINGECKO_API = "https://api.coingecko.com/api/v3/simple/price"

# Asset symbol mapping (Internal -> Exchange)
BINANCE_SYMBOLS = {
    "bitcoin": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "solana": "SOLUSDT",
    "ripple": "XRPUSDT",
    "dogecoin": "DOGEUSDT",
    "cardano": "ADAUSDT",
    "avalanche-2": "AVAXUSDT",
    "polkadot": "DOTUSDT",
    "chainlink": "LINKUSDT",
    "internet-computer": "ICPUSDT"
}

COINGECKO_IDS = [
    "bitcoin", "ethereum", "solana", "ripple", "dogecoin",
    "cardano", "avalanche-2", "polkadot", "chainlink", "internet-computer"
]

# HTTP Cycles for outcalls
HTTP_CYCLES = 250_000_000

# ============================================================
# PRICE ORACLE STATE
# ============================================================

class PriceOracle:
    """Manages live price data with rolling window"""
    
    def __init__(self, max_history: int = 500):
        self.max_history = max_history
        self.prices: dict = {}  # symbol -> current price
        self.history: dict = {}  # symbol -> [price1, price2, ...]
        self.timestamps: list = []
        self.last_source: str = "none"
        self.last_update: int = 0
        self.fetch_count: int = 0
        self.error_count: int = 0
    
    def update_price(self, symbol: str, price: float, timestamp: int):
        """Update price and maintain rolling window"""
        self.prices[symbol] = price
        
        if symbol not in self.history:
            self.history[symbol] = []
        
        self.history[symbol].append(price)
        
        # Maintain rolling window - drop old data
        if len(self.history[symbol]) > self.max_history:
            self.history[symbol] = self.history[symbol][-self.max_history:]
        
        self.last_update = timestamp
    
    def get_price(self, symbol: str) -> float:
        return self.prices.get(symbol, 0.0)
    
    def get_history(self, symbol: str, periods: int = 100) -> list:
        return self.history.get(symbol, [])[-periods:]
    
    def get_all_prices(self) -> dict:
        return self.prices.copy()
    
    def to_dict(self) -> dict:
        return {
            "prices": self.prices,
            "last_source": self.last_source,
            "last_update": self.last_update,
            "fetch_count": self.fetch_count,
            "history_lengths": {k: len(v) for k, v in self.history.items()}
        }

# Global oracle instance
oracle = PriceOracle()


# ============================================================
# HTTP OUTCALL HELPERS
# ============================================================

async def http_get(url: str) -> dict:
    """Make HTTPS outcall to external API"""
    from kybra import Async
    
    request = {
        "url": url,
        "method": {"get": None},
        "headers": [],
        "body": None,
        "max_response_bytes": 10000,
        "transform": None
    }
    
    try:
        response = await ic.http_request(request, HTTP_CYCLES)
        
        if response["status"] >= 200 and response["status"] < 300:
            body = bytes(response["body"]).decode("utf-8")
            return {"success": True, "data": json.loads(body)}
        else:
            return {"success": False, "error": f"HTTP {response['status']}"}
    
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================
# PRICE FETCHERS
# ============================================================

async def fetch_from_binance() -> dict:
    """Fetch all prices from Binance"""
    results = {}
    
    for asset_id, symbol in BINANCE_SYMBOLS.items():
        url = f"{BINANCE_API}?symbol={symbol}"
        response = await http_get(url)
        
        if response["success"] and "price" in response["data"]:
            results[asset_id] = float(response["data"]["price"])
    
    return results


async def fetch_from_coingecko() -> dict:
    """Fetch all prices from CoinGecko (fallback)"""
    ids = ",".join(COINGECKO_IDS)
    url = f"{COINGECKO_API}?ids={ids}&vs_currencies=usd"
    
    response = await http_get(url)
    
    if not response["success"]:
        return {}
    
    results = {}
    for asset_id in COINGECKO_IDS:
        if asset_id in response["data"] and "usd" in response["data"][asset_id]:
            results[asset_id] = response["data"][asset_id]["usd"]
    
    return results


# ============================================================
# MAIN ORACLE FUNCTIONS
# ============================================================

async def fetch_market_data() -> dict:
    """
    Fetch live market data from primary source, fallback if needed.
    Updates the global oracle state.
    """
    global oracle
    
    timestamp = ic.time()
    
    # Try Binance first
    prices = await fetch_from_binance()
    source = "binance"
    
    # Fallback to CoinGecko if Binance failed
    if len(prices) < 5:
        prices = await fetch_from_coingecko()
        source = "coingecko"
    
    if not prices:
        oracle.error_count += 1
        return {"success": False, "error": "All sources failed"}
    
    # Update oracle state
    for symbol, price in prices.items():
        oracle.update_price(symbol, price, timestamp)
    
    oracle.last_source = source
    oracle.fetch_count += 1
    
    return {
        "success": True,
        "source": source,
        "prices_fetched": len(prices),
        "timestamp": timestamp
    }


def get_price_arrays(asset_a: str, asset_b: str, periods: int = 100) -> tuple:
    """
    Get synchronized price arrays for two assets.
    Returns (prices_a, prices_b) as lists.
    """
    history_a = oracle.get_history(asset_a, periods)
    history_b = oracle.get_history(asset_b, periods)
    
    # Ensure same length
    min_len = min(len(history_a), len(history_b))
    
    return (history_a[-min_len:], history_b[-min_len:]) if min_len > 0 else ([], [])


# ============================================================
# CANISTER ENDPOINTS
# ============================================================

@query
def get_oracle_status() -> str:
    """Get current oracle state"""
    return json.dumps(oracle.to_dict())


@query
def get_live_prices() -> str:
    """Get all current prices"""
    return json.dumps({
        "prices": oracle.get_all_prices(),
        "source": oracle.last_source,
        "last_update": oracle.last_update
    })


@query
def get_price_history(asset: str, periods: int) -> str:
    """Get price history for an asset"""
    return json.dumps({
        "asset": asset,
        "history": oracle.get_history(asset, periods),
        "current": oracle.get_price(asset)
    })
