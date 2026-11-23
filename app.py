"""
FinSight AI - Financial Analytics & Machine Learning Dashboard
Combines MySQL ETL, Quant Technical Indicators, and Scikit-Learn Predictive Modeling.
"""

from datetime import date, timedelta
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from database import (
    create_tables, test_connection, get_stock_list, get_stock_data,
    get_summary, get_latest_dates, get_import_logs, get_table_counts
)
from fetch_stock_data import STOCKS, import_stock
from src.analytics import compute_technical_indicators
from src.ml_predictor import train_and_forecast_stock

st.set_page_config(page_title="FinSight AI - Financial Analytics", page_icon="📈", layout="wide")

# Theme styling
st.markdown("""
    <style>
    .metric-box { background-color: #1e1e24; padding: 15px; border-radius: 8px; border-left: 5px solid #00d084; }
    .bullish { color: #00d084; font-weight: bold; }
    .bearish { color: #cf2e2e; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# Database check
try:
    create_tables()
    mysql_version = test_connection()
except Exception as error:
    st.error("Unable to connect to MySQL. Verify your .env credentials.")
    st.code(str(error))
    st.stop()

# Sidebar
st.sidebar.title("📈 FinSight AI")
st.sidebar.caption("Financial Data Engineering & Predictive Intelligence")
st.sidebar.success(f"MySQL Connected (v{mysql_version.split('-')[0]})")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Market Overview", "📈 Technical Analysis", "🤖 ML Trend Forecast", "📥 Ingestion Center", "🗄️ Database Explorer"]
)

# ------------------------------------------------------------------------------
# 1. MARKET OVERVIEW
# ------------------------------------------------------------------------------
if page == "📊 Market Overview":
    st.header("Executive Market & Portfolio Overview")
    summary = get_summary().iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Historical Records", f"{int(summary['total_records']):,}")
    c2.metric("Tracked Companies", int(summary["total_stocks"]))
    latest_dt = summary["latest_trading_date"]
    c3.metric("Latest Market Date", str(latest_dt) if pd.notna(latest_dt) else "No data")

    st.markdown("---")
    st.subheader("Registered Ticker Status")
    latest_dates = get_latest_dates()
    if latest_dates.empty or int(summary['total_records']) == 0:
        st.info("No stock records in database yet. Head over to '📥 Ingestion Center' to fetch historical prices.")
    else:
        st.dataframe(latest_dates, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# 2. TECHNICAL ANALYSIS & CHARTS
# ------------------------------------------------------------------------------
elif page == "📈 Technical Analysis":
    st.header("Technical Indicators & Candlestick Analysis")
    stocks_df = get_stock_list()

    if stocks_df.empty:
        st.warning("No stocks registered yet. Import data first.")
    else:
        symbols = stocks_df["symbol"].tolist()
        sel_symbol = st.selectbox("Select Ticker Symbol", symbols)
        
        col1, col2 = st.columns(2)
        start_d = col1.date_input("Start Date", value=date.today() - timedelta(days=365))
        end_d = col2.date_input("End Date", value=date.today())

        df_raw = get_stock_data(symbol=sel_symbol, start_date=start_d.isoformat(), end_date=end_d.isoformat())

        if df_raw.empty:
            st.warning(f"No price records found for {sel_symbol} in this date range.")
        else:
            df_tech = compute_technical_indicators(df_raw)

            # Candlestick + SMA + RSI Chart
            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                vertical_spacing=0.08, subplot_titles=(f"{sel_symbol} Price & Moving Averages", "Relative Strength Index (RSI 14)"),
                row_heights=[0.75, 0.25]
            )

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df_tech["trade_date"], open=df_tech["open_price"], high=df_tech["high_price"],
                low=df_tech["low_price"], close=df_tech["close_price"], name="OHLC"
            ), row=1, col=1)

            # SMAs
            fig.add_trace(go.Scatter(x=df_tech["trade_date"], y=df_tech["SMA_20"], line=dict(color="#f39c12", width=1.5), name="SMA 20"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_tech["trade_date"], y=df_tech["SMA_50"], line=dict(color="#3498db", width=1.5), name="SMA 50"), row=1, col=1)

            # RSI
            fig.add_trace(go.Scatter(x=df_tech["trade_date"], y=df_tech["RSI_14"], line=dict(color="#9b59b6", width=1.5), name="RSI (14)"), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

            fig.update_layout(xaxis_rangeslider_visible=False, height=650, margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------
# 3. ML TREND FORECAST
# ------------------------------------------------------------------------------
elif page == "🤖 ML Trend Forecast":
    st.header("Machine Learning Next-Day Trend Forecasting")
    st.caption("Trains a predictive model using Scikit-Learn Random Forest on rolling price lags and momentum features.")
    
    stocks_df = get_stock_list()
    if stocks_df.empty:
        st.warning("Import stock data first.")
    else:
        sel_symbol = st.selectbox("Select Asset to Forecast", stocks_df["symbol"].tolist(), key="ml_sym")
        df_raw = get_stock_data(symbol=sel_symbol, limit=2000)

        if len(df_raw) < 60:
            st.error(f"Insufficient history ({len(df_raw)} records). At least 60 trading days required to train the model.")
        else:
            if st.button("Train ML Model & Generate Forecast", type="primary"):
                with st.spinner("Engineering lag features and fitting Random Forest model..."):
                    try:
                        forecast = train_and_forecast_stock(df_raw)
                        st.success(f"Forecast Ready for **{sel_symbol}**!")

                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Latest Close", f"${forecast['current_price']:.2f}")
                        m2.metric("Predicted Target Price", f"${forecast['predicted_price']:.2f}", delta=f"{forecast['expected_change_pct']:.2f}%")
                        m3.metric("Projected Trend", forecast["trend_direction"])
                        m4.metric("Model Confidence", f"{forecast['confidence_score']:.1f}%")

                        st.info(f"**Quantitative Note:** RSI is currently at **{forecast['latest_rsi']}**. Model evaluated with 100 decision trees.")
                    except Exception as e:
                        st.error(f"Prediction Error: {e}")

# ------------------------------------------------------------------------------
# 4. INGESTION CENTER
# ------------------------------------------------------------------------------
elif page == "📥 Ingestion Center":
    st.header("Automated Market Data Ingestion Pipeline")
    st.write("Fetch daily market data from Yahoo Finance and upsert into MySQL.")

    stock_options = {f"{info['name']} ({symbol})": symbol for symbol, info in STOCKS.items()}
    selected_stock = st.selectbox("Select Stock", list(stock_options.keys()))
    symbol = stock_options[selected_stock]

    today = date.today()
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start Date", value=today - timedelta(days=365 * 3), max_value=today)
    end_date = col2.date_input("End Date", value=today, min_value=start_date, max_value=today)

    if st.button("Fetch and Upsert Data into MySQL", type="primary"):
        with st.spinner(f"Ingesting historical daily prices for {symbol}..."):
            result = import_stock(symbol=symbol, start_date=start_date.isoformat(), end_date=end_date.isoformat())
            if result["status"] == "SUCCESS":
                st.success(f"✅ {result['message']}")
            else:
                st.error(f"❌ {result['message']}")

# ------------------------------------------------------------------------------
# 5. DATABASE EXPLORER
# ------------------------------------------------------------------------------
elif page == "🗄️ Database Explorer":
    st.header("Database Records & Export")
    stocks_df = get_stock_list()

    if not stocks_df.empty:
        symbol_choice = st.selectbox("Filter Ticker", ["All"] + stocks_df["symbol"].tolist())
        filter_sym = None if symbol_choice == "All" else symbol_choice
        data = get_stock_data(symbol=filter_sym, limit=1000)
        st.write(f"Showing {len(data):,} records from MySQL.")
        st.dataframe(data, use_container_width=True, hide_index=True)
        if not data.empty:
            st.download_button("Download as CSV", data=data.to_csv(index=False).encode("utf-8"), file_name="finsight_stock_data.csv", mime="text/csv")