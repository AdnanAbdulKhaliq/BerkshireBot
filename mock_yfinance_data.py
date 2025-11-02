"""
Mock Yahoo Finance data for AAPL, TSLA, and NVDA
This module provides mock historical price data and analyst information
to bypass Yahoo Finance API issues.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Mock historical price data (2 years of daily data)
MOCK_PRICE_DATA = {
    "AAPL": {
        "current_price": 178.50,
        "start_price": 145.30,
        "volatility": 0.25,  # 25% annualized volatility
        "drift": 0.15,  # 15% annualized return
    },
    "TSLA": {
        "current_price": 242.80,
        "start_price": 185.60,
        "volatility": 0.45,  # 45% annualized volatility
        "drift": 0.20,  # 20% annualized return
    },
    "NVDA": {
        "current_price": 485.20,
        "start_price": 185.40,
        "volatility": 0.50,  # 50% annualized volatility
        "drift": 0.75,  # 75% annualized return
    },
}

# Mock analyst data
MOCK_ANALYST_DATA = {
    "AAPL": {
        "recommendationKey": "buy",
        "numberOfAnalystOpinions": 42,
        "targetHighPrice": 215.00,
        "targetLowPrice": 165.00,
        "targetMeanPrice": 195.50,
        "targetMedianPrice": 195.00,
        "currentPrice": 178.50,
        "recommendations": pd.DataFrame({
            "period": ["0m", "-1m", "-2m", "-3m"],
            "strongBuy": [18, 17, 16, 15],
            "buy": [15, 16, 17, 18],
            "hold": [7, 7, 7, 8],
            "sell": [2, 2, 2, 1],
            "strongSell": [0, 0, 0, 0],
        }),
    },
    "TSLA": {
        "recommendationKey": "hold",
        "numberOfAnalystOpinions": 38,
        "targetHighPrice": 350.00,
        "targetLowPrice": 180.00,
        "targetMeanPrice": 255.00,
        "targetMedianPrice": 250.00,
        "currentPrice": 242.80,
        "recommendations": pd.DataFrame({
            "period": ["0m", "-1m", "-2m", "-3m"],
            "strongBuy": [8, 7, 8, 9],
            "buy": [12, 13, 12, 11],
            "hold": [14, 14, 15, 15],
            "sell": [3, 3, 2, 2],
            "strongSell": [1, 1, 1, 1],
        }),
    },
    "NVDA": {
        "recommendationKey": "buy",
        "numberOfAnalystOpinions": 45,
        "targetHighPrice": 720.00,
        "targetLowPrice": 420.00,
        "targetMeanPrice": 595.00,
        "targetMedianPrice": 600.00,
        "currentPrice": 485.20,
        "recommendations": pd.DataFrame({
            "period": ["0m", "-1m", "-2m", "-3m"],
            "strongBuy": [22, 21, 20, 19],
            "buy": [16, 17, 18, 19],
            "hold": [6, 6, 6, 6],
            "sell": [1, 1, 1, 1],
            "strongSell": [0, 0, 0, 0],
        }),
    },
}


def generate_mock_price_history(ticker: str, days: int = 504) -> pd.DataFrame:
    """
    Generate mock historical price data for a ticker.
    
    Args:
        ticker: Stock ticker symbol (AAPL, TSLA, or NVDA)
        days: Number of days of historical data (default: 504 = ~2 years)
    
    Returns:
        DataFrame with columns: Close, Open, High, Low, Volume
    """
    if ticker not in MOCK_PRICE_DATA:
        raise ValueError(f"Mock data not available for ticker: {ticker}")
    
    data = MOCK_PRICE_DATA[ticker]
    
    # Generate dates
    end_date = datetime.now()
    dates = [end_date - timedelta(days=i) for i in range(days)]
    dates.reverse()
    
    # Generate price path using geometric Brownian motion
    np.random.seed(hash(ticker) % 2**32)  # Deterministic but different per ticker
    
    dt = 1/252  # Daily time step (252 trading days per year)
    mu = data["drift"]
    sigma = data["volatility"]
    
    # Generate returns
    returns = np.random.normal(
        (mu - 0.5 * sigma**2) * dt,
        sigma * np.sqrt(dt),
        days
    )
    
    # Generate price path
    price_multipliers = np.exp(np.cumsum(returns))
    
    # Scale to match start and end prices
    scale_factor = (data["current_price"] - data["start_price"]) / (price_multipliers[-1] - price_multipliers[0])
    prices = data["start_price"] + (price_multipliers - price_multipliers[0]) * scale_factor
    
    # Ensure final price matches current price
    prices = prices * (data["current_price"] / prices[-1])
    
    # Generate OHLC data
    close_prices = prices
    open_prices = close_prices * (1 + np.random.normal(0, 0.005, days))
    high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(np.random.normal(0, 0.01, days)))
    low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(np.random.normal(0, 0.01, days)))
    volumes = np.random.lognormal(17, 0.5, days).astype(int)  # Realistic volume distribution
    
    # Create DataFrame
    df = pd.DataFrame({
        'Close': close_prices,
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Volume': volumes
    }, index=dates)
    
    df.index.name = 'Date'
    
    return df


def get_mock_analyst_data(ticker: str) -> dict:
    """
    Get mock analyst data for a ticker.
    
    Args:
        ticker: Stock ticker symbol (AAPL, TSLA, or NVDA)
    
    Returns:
        Dictionary with analyst information
    """
    if ticker not in MOCK_ANALYST_DATA:
        raise ValueError(f"Mock analyst data not available for ticker: {ticker}")
    
    return MOCK_ANALYST_DATA[ticker].copy()


class MockTicker:
    """Mock yfinance Ticker object"""
    
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        if self.ticker not in MOCK_PRICE_DATA:
            raise ValueError(f"Mock data not available for ticker: {self.ticker}")
    
    def history(self, period: str = "2y", auto_adjust: bool = True):
        """Return mock historical data"""
        # Convert period to days
        if period == "1y":
            days = 252
        elif period == "2y":
            days = 504
        elif period == "5y":
            days = 1260
        else:
            days = 504  # Default to 2 years
        
        return generate_mock_price_history(self.ticker, days)
    
    @property
    def info(self):
        """Return mock ticker info"""
        analyst_data = get_mock_analyst_data(self.ticker)
        return {
            "symbol": self.ticker,
            "currentPrice": analyst_data["currentPrice"],
            "targetMeanPrice": analyst_data["targetMeanPrice"],
            "targetMedianPrice": analyst_data["targetMedianPrice"],
            "targetHighPrice": analyst_data["targetHighPrice"],
            "targetLowPrice": analyst_data["targetLowPrice"],
            "recommendationKey": analyst_data["recommendationKey"],
            "numberOfAnalystOpinions": analyst_data["numberOfAnalystOpinions"],
        }
    
    @property
    def recommendations(self):
        """Return mock recommendations"""
        analyst_data = get_mock_analyst_data(self.ticker)
        return analyst_data["recommendations"]
    
    @property
    def upgrades_downgrades(self):
        """Return mock upgrades/downgrades data"""
        from datetime import datetime, timedelta
        
        # Generate some recent upgrades/downgrades with dates as index
        dates = [datetime.now() - timedelta(days=i*7) for i in range(4)]
        
        if self.ticker == "AAPL":
            data = {
                "Firm": ["Morgan Stanley", "Goldman Sachs", "JP Morgan", "Bank of America"],
                "ToGrade": ["Overweight", "Buy", "Overweight", "Buy"],
                "FromGrade": ["Equal-Weight", "Neutral", "Equal-Weight", "Neutral"],
                "Action": ["up", "up", "up", "up"],
            }
        elif self.ticker == "TSLA":
            data = {
                "Firm": ["Morgan Stanley", "Goldman Sachs", "UBS", "Barclays"],
                "ToGrade": ["Equal-Weight", "Neutral", "Hold", "Equal-Weight"],
                "FromGrade": ["Overweight", "Buy", "Buy", "Overweight"],
                "Action": ["down", "down", "down", "down"],
            }
        else:  # NVDA
            data = {
                "Firm": ["Morgan Stanley", "Goldman Sachs", "JP Morgan", "Citi"],
                "ToGrade": ["Overweight", "Buy", "Overweight", "Buy"],
                "FromGrade": ["Equal-Weight", "Neutral", "Neutral", "Neutral"],
                "Action": ["up", "up", "up", "up"],
            }
        
        df = pd.DataFrame(data, index=dates)
        df.index.name = "GradeDate"
        return df


def use_mock_data() -> bool:
    """
    Determine if mock data should be used.
    Set environment variable USE_MOCK_YFINANCE=1 to force mock data.
    """
    import os
    return os.getenv("USE_MOCK_YFINANCE", "1") == "1"


# For easy importing
__all__ = [
    "MockTicker",
    "generate_mock_price_history",
    "get_mock_analyst_data",
    "use_mock_data",
    "MOCK_PRICE_DATA",
    "MOCK_ANALYST_DATA",
]
