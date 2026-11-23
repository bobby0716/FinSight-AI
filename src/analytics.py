"""
FinSight AI - Quantitative Technical Analysis Module
Calculates Moving Averages (SMA-20, SMA-50), RSI, Daily Returns, and Volatility.
"""

import pandas as pd
import numpy as np


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes standard quant indicators on sorted historical price DataFrame.
    """
    if df.empty or len(df) < 5:
        return df

    data = df.copy().sort_values(by="trade_date", ascending=True)

    # Convert close_price to float
    data["close_price"] = data["close_price"].astype(float)

    # 1. Moving Averages
    data["SMA_20"] = data["close_price"].rolling(window=20).mean()
    data["SMA_50"] = data["close_price"].rolling(window=50).mean()

    # 2. Daily Percentage Return & Volatility
    data["Daily_Return_%"] = data["close_price"].pct_change() * 100
    data["Volatility_20D"] = data["Daily_Return_%"].rolling(window=20).std()

    # 3. Relative Strength Index (RSI - 14 Days)
    delta = data["close_price"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data["RSI_14"] = 100 - (100 / (1 + rs))

    return data