"""
FinSight AI - Data Ingestion Pipeline
Fetches historical stock prices from Yahoo Finance and populates MySQL.
"""

from datetime import datetime, date
import logging
import pandas as pd
import yfinance as yf
from database import create_tables, register_stock, save_stock_data, log_import

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

STOCKS = {
    "AAPL": {"name": "Apple Inc.", "exchange": "NASDAQ", "currency": "USD"},
    "MSFT": {"name": "Microsoft Corporation", "exchange": "NASDAQ", "currency": "USD"},
    "GOOGL": {"name": "Alphabet Inc.", "exchange": "NASDAQ", "currency": "USD"},
    "NVDA": {"name": "NVIDIA Corporation", "exchange": "NASDAQ", "currency": "USD"},
    "TSLA": {"name": "Tesla, Inc.", "exchange": "NASDAQ", "currency": "USD"},
    "005930.KS": {"name": "Samsung Electronics", "exchange": "Korea Exchange", "currency": "KRW"},
}


def fetch_stock(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    exclusive_end = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    data = ticker.history(start=start_date, end=exclusive_end, interval="1d", auto_adjust=False, actions=False)
    if data.empty:
        raise ValueError(f"No data returned for {symbol}. Verify symbol and network connectivity.")

    data = data.reset_index()
    if "Date" not in data.columns and "Datetime" in data.columns:
        data = data.rename(columns={"Datetime": "Date"})

    data["Date"] = pd.to_datetime(data["Date"], utc=True).dt.date
    start_day = pd.Timestamp(start_date).date()
    end_day = pd.Timestamp(end_date).date()
    data = data[(data["Date"] >= start_day) & (data["Date"] <= end_day)]

    data = data.rename(columns={
        "Date": "trade_date", "Open": "open_price", "High": "high_price",
        "Low": "low_price", "Close": "close_price", "Adj Close": "adj_close", "Volume": "volume"
    })
    data["symbol"] = symbol
    return data[["symbol", "trade_date", "open_price", "high_price", "low_price", "close_price", "adj_close", "volume"]]


def import_stock(symbol: str, start_date: str, end_date: str):
    started_at = datetime.utcnow()
    stock_meta = STOCKS.get(symbol, {"name": symbol, "exchange": "Unknown", "currency": "USD"})
    register_stock(symbol, stock_meta["name"], stock_meta["exchange"], stock_meta["currency"])

    try:
        df = fetch_stock(symbol, start_date, end_date)
        rows_written = save_stock_data(df)
        finished_at = datetime.utcnow()
        log_import(symbol, started_at, finished_at, "SUCCESS", len(df), rows_written)
        return {"status": "SUCCESS", "rows_received": len(df), "rows_written": rows_written, "message": f"Successfully imported {rows_written} records for {symbol}."}
    except Exception as e:
        finished_at = datetime.utcnow()
        log_import(symbol, started_at, finished_at, "FAILED", 0, 0, str(e))
        return {"status": "FAILED", "rows_received": 0, "rows_written": 0, "message": str(e)}


if __name__ == "__main__":
    create_tables()
    print("[INFO] Testing ingestion for AAPL (last 30 days)...")
    res = import_stock("AAPL", "2024-01-01", date.today().isoformat())
    print(res)