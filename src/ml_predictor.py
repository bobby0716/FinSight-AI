"""
FinSight AI - Machine Learning Market Trend Predictor
Trains rolling feature model using Scikit-Learn to forecast next-day trend direction and target price.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from src.analytics import compute_technical_indicators


def train_and_forecast_stock(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Trains ML models on historical prices and generates next-day forecast.
    """
    if len(df) < 60:
        raise ValueError("Insufficient historical data (at least 60 trading days required for ML).")

    data = compute_technical_indicators(df).dropna().copy()
    
    # Feature Engineering (Lags)
    data["Lag_1"] = data["close_price"].shift(1)
    data["Lag_2"] = data["close_price"].shift(2)
    data["Lag_3"] = data["close_price"].shift(3)
    data["Return_Lag1"] = data["Daily_Return_%"].shift(1)
    
    # Target 1 (Direction): 1 if tomorrow's close > today's close, else 0
    data["Target_Direction"] = (data["close_price"].shift(-1) > data["close_price"]).astype(int)
    # Target 2 (Price): Tomorrow's close price
    data["Target_Price"] = data["close_price"].shift(-1)

    # Features for training
    features = ["Lag_1", "Lag_2", "Lag_3", "Return_Lag1", "SMA_20", "SMA_50", "RSI_14"]
    clean_data = data.dropna()

    X = clean_data[features]
    y_dir = clean_data["Target_Direction"]
    y_price = clean_data["Target_Price"]

    # Train Classification Model (Direction: Up/Down)
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X, y_dir)

    # Train Regression Model (Target Price)
    reg = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    reg.fit(X, y_price)

    # Predict Next Day using latest row
    latest_features = data[features].iloc[[-1]]
    pred_direction = clf.predict(latest_features)[0]
    pred_prob = clf.predict_proba(latest_features)[0][pred_direction]
    pred_price = reg.predict(latest_features)[0]
    current_price = float(data["close_price"].iloc[-1])

    expected_change_pct = ((pred_price - current_price) / current_price) * 100

    return {
        "current_price": round(current_price, 2),
        "predicted_price": round(float(pred_price), 2),
        "expected_change_pct": round(float(expected_change_pct), 2),
        "trend_direction": "BULLISH 📈" if pred_direction == 1 else "BEARISH 📉",
        "confidence_score": round(float(pred_prob) * 100, 1),
        "latest_rsi": round(float(data["RSI_14"].iloc[-1]), 1),
    }