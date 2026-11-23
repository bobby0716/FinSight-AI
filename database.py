"""
FinSight AI - Database Connection & Management Engine
SQLAlchemy ORM connection pool with automated schema creation and upsert logic.
"""

import os
import pymysql
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_NAME = os.getenv("DB_NAME", "finsight_ai")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")


def ensure_database_exists():
    """Connects to MySQL server and creates finsight_ai if not present."""
    conn = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        autocommit=True,
    )
    with conn.cursor() as cursor:
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME} CHARACTER SET utf8mb4;")
    conn.close()


ensure_database_exists()

# Safe SQLAlchemy connection URL
connection_url = URL.create(
    drivername="mysql+pymysql",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME,
    query={"charset": "utf8mb4"},
)

engine = create_engine(
    connection_url,
    pool_pre_ping=True,
    pool_recycle=1800,
)


def test_connection():
    """Test MySQL connectivity."""
    with engine.connect() as connection:
        return connection.execute(text("SELECT VERSION()")).scalar_one()


def create_tables():
    """Initializes tables for stocks, daily prices, and ingestion logs."""
    statements = [
        """
        CREATE TABLE IF NOT EXISTS stocks (
            symbol VARCHAR(20) PRIMARY KEY,
            company_name VARCHAR(100) NOT NULL,
            exchange_name VARCHAR(50),
            currency VARCHAR(10),
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS stock_prices (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            trade_date DATE NOT NULL,
            open_price DECIMAL(18, 6),
            high_price DECIMAL(18, 6),
            low_price DECIMAL(18, 6),
            close_price DECIMAL(18, 6),
            adj_close DECIMAL(18, 6),
            volume BIGINT,
            imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            CONSTRAINT uq_stock_trade_date UNIQUE (symbol, trade_date),
            CONSTRAINT fk_stock_prices_symbol FOREIGN KEY (symbol)
                REFERENCES stocks(symbol) ON UPDATE CASCADE ON DELETE RESTRICT,
            INDEX idx_stock_prices_date (trade_date)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS import_logs (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            started_at DATETIME NOT NULL,
            finished_at DATETIME,
            status ENUM('SUCCESS', 'PARTIAL', 'FAILED') NOT NULL,
            rows_received INT DEFAULT 0,
            rows_written INT DEFAULT 0,
            error_message TEXT,
            INDEX idx_import_logs_symbol (symbol),
            INDEX idx_import_logs_started (started_at)
        );
        """
    ]

    with engine.begin() as connection:
        for stmt in statements:
            connection.execute(text(stmt))


def register_stock(symbol, company_name, exchange_name=None, currency=None):
    sql = text("""
        INSERT INTO stocks (symbol, company_name, exchange_name, currency, is_active)
        VALUES (:symbol, :company_name, :exchange_name, :currency, TRUE)
        ON DUPLICATE KEY UPDATE
            company_name = VALUES(company_name),
            exchange_name = VALUES(exchange_name),
            currency = VALUES(currency),
            is_active = TRUE;
    """)
    with engine.begin() as conn:
        conn.execute(sql, {"symbol": symbol, "company_name": company_name, "exchange_name": exchange_name, "currency": currency})


def save_stock_data(data_frame):
    if data_frame.empty:
        return 0
    sql = text("""
        INSERT INTO stock_prices (
            symbol, trade_date, open_price, high_price, low_price, close_price, adj_close, volume
        ) VALUES (
            :symbol, :trade_date, :open_price, :high_price, :low_price, :close_price, :adj_close, :volume
        ) ON DUPLICATE KEY UPDATE
            open_price = VALUES(open_price),
            high_price = VALUES(high_price),
            low_price = VALUES(low_price),
            close_price = VALUES(close_price),
            adj_close = VALUES(adj_close),
            volume = VALUES(volume);
    """)
    records = data_frame.to_dict(orient="records")
    with engine.begin() as conn:
        conn.execute(sql, records)
    return len(records)


def log_import(symbol, started_at, finished_at, status, rows_received, rows_written, error_message=None):
    sql = text("""
        INSERT INTO import_logs (
            symbol, started_at, finished_at, status, rows_received, rows_written, error_message
        ) VALUES (
            :symbol, :started_at, :finished_at, :status, :rows_received, :rows_written, :error_message
        );
    """)
    with engine.begin() as conn:
        conn.execute(sql, {
            "symbol": symbol, "started_at": started_at, "finished_at": finished_at,
            "status": status, "rows_received": rows_received, "rows_written": rows_written,
            "error_message": error_message,
        })


def get_stock_data(symbol=None, start_date=None, end_date=None, limit=5000):
    filters = []
    params = {"limit": limit}
    if symbol:
        filters.append("symbol = :symbol")
        params["symbol"] = symbol
    if start_date:
        filters.append("trade_date >= :start_date")
        params["start_date"] = start_date
    if end_date:
        filters.append("trade_date <= :end_date")
        params["end_date"] = end_date

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    query = text(f"SELECT * FROM stock_prices {where_clause} ORDER BY trade_date ASC LIMIT :limit;")
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params=params)


def get_stock_list():
    with engine.connect() as conn:
        return pd.read_sql(text("SELECT * FROM stocks WHERE is_active = TRUE ORDER BY symbol;"), conn)


def get_summary():
    with engine.connect() as conn:
        return pd.read_sql(text("SELECT COUNT(*) AS total_records, COUNT(DISTINCT symbol) AS total_stocks, MAX(trade_date) AS latest_trading_date FROM stock_prices;"), conn)


def get_latest_dates():
    with engine.connect() as conn:
        return pd.read_sql(text("SELECT s.symbol, s.company_name, MAX(sp.trade_date) AS latest_date, COUNT(sp.id) as record_count FROM stocks s LEFT JOIN stock_prices sp ON s.symbol = sp.symbol GROUP BY s.symbol, s.company_name;"), conn)


def get_import_logs(limit=50):
    with engine.connect() as conn:
        return pd.read_sql(text("SELECT * FROM import_logs ORDER BY started_at DESC LIMIT :limit;"), conn, params={"limit": limit})


def get_table_counts():
    with engine.connect() as conn:
        s_count = conn.execute(text("SELECT COUNT(*) FROM stocks;")).scalar_one()
        p_count = conn.execute(text("SELECT COUNT(*) FROM stock_prices;")).scalar_one()
        l_count = conn.execute(text("SELECT COUNT(*) FROM import_logs;")).scalar_one()
    return {"stocks": s_count, "stock_prices": p_count, "import_logs": l_count}