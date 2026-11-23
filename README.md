### FinSight AI

An enterprise financial data engineering and quantitative intelligence platform that extracts historical equity data from Yahoo Finance, loads it into a normalized MySQL database via SQLAlchemy, computes quantitative indicators (SMA-20, SMA-50, RSI-14), and predicts market trends using Scikit-Learn Machine Learning.

## System Architecture

              EXTERNAL DATA SOURCE
               [Yahoo Finance API]
                        │
                        ▼
              [fetch_stock_data.py]
          (ETL Pipeline & Normalization)
                        │
                        ▼
          ┌───────────────────────────┐
          │    MySQL Database Engine  │
          │       (finsight_ai)       │
          ├───────────────────────────┤
          │ - stocks (Metadata)       │
          │ - stock_prices (OHLCV)    │
          │ - import_logs (Audit)     │
          └─────────────┬─────────────┘
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼

         QUANTITATIVE ENGINE STREAMLIT INTERACTION ┌─────────────────────────┐ ┌─────────────────────────┐ │ - SMA-20 / SMA-50 │ │ - Interactive Candlestick│ │ - Relative Strength (RSI│──►│ - Bullish/Bearish Badges│ │ - Random Forest ML Model│ │ - CSV Export Explorer │ └─────────────────────────┘ └─────────────────────────┘

         
---
##  Features

- ** Automated Financial ETL:** Fetches daily OHLCV equity prices via `yfinance`, normalizes timestamps, and performs idempotent upserts into MySQL.

- ** Interactive Candlestick & Technical Charts:** High-performance Plotly charts featuring **SMA-20, SMA-50 overlays, and RSI-14 oscillators**.

- ** Machine Learning Trend Prediction:** Uses Random Forest classifiers and Gradient Boosting regressors trained on rolling price lags to predict next-day price targets and directional momentum (**Bullish  vs. Bearish **).

- **🔒 Enterprise Security:** Enforces strict zero-credential exposure via `.env` and `.
gitignore`.

- ** Automated Unit Testing:** Includes `unittest` suite testing database connection pools, quantitative formulas, and ML training pipelines.

---

## Project Structure

```text
FinSight-AI/
├── src/
│   ├── __init__.py
│   ├── analytics.py          # Quant indicators (SMA, RSI, Volatility)
│   └── ml_predictor.py       # Scikit-Learn trend forecaster
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py      # Automated integration tests
├── database.py               # SQLAlchemy ORM connection pool & DDL
├── fetch_stock_data.py       # Yahoo Finance automated ingestion script
├── app.py                    # Streamlit quantitative trading dashboard
├── .env.example              # Public database configuration template
├── .gitignore                # Protects secrets from version control
├── requirements.txt          # Pinned dependencies
└── README.md                 # Project documentation

### Quickstart Guide

1. Configure Environment

copy .env.example .env

Update .env with your local MySQL credentials:

ini
DB_HOST=localhost
DB_PORT=3306
DB_NAME=finsight_ai
DB_USER=root
DB_PASSWORD=your_mysql_password

2. Install Dependencies

pip install -r requirements.txt

3. Run Automated Tests

python -m unittest tests.test_pipeline -v

4. Launch the Dashboard

streamlit run app.py