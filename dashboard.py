# dashboard.py
# Phase 3 (Part B): Simple interactive dashboard for MA backtest (fixed)

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st

st.set_page_config(page_title="Finance Dashboard", layout="wide")

# Use plain titles (no markdown emojis/links) to dodge Safari's autolink issue
st.title("Finance Dashboard - Moving Average Backtest")

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

with st.sidebar:
    st.header("Controls")
    ticker = st.selectbox("Stock", TICKERS, index=0)
    start = st.date_input("Start date", pd.to_datetime("2022-01-01"))
    end   = st.date_input("End date",   pd.to_datetime("2025-01-01"))
    short = st.slider("Short MA window", 5, 100, 50, step=1)
    long  = st.slider("Long MA window",  20, 300, 200, step=5)
    st.text("Long must be greater than Short.")

# Guards
if long <= short:
    st.warning("Choose a Long window larger than Short.")
    st.stop()
if pd.to_datetime(end) <= pd.to_datetime(start):
    st.warning("End date must be after Start date.")
    st.stop()

# --- Data download (force consistent shapes) ---
raw = yf.download([ticker], start=start, end=end, auto_adjust=True)

if raw is None or raw.empty:
    st.error("No data returned. Try a different date range or ticker.")
    st.stop()

# MultiIndex vs single-index handling
if isinstance(raw.columns, pd.MultiIndex):
    if ("Close", ticker) not in raw.columns:
        st.error("Unexpected data format from Yahoo Finance.")
        st.stop()
    close = raw[("Close", ticker)].astype(float)
else:
    if "Close" not in raw.columns:
        st.error("No 'Close' prices in returned data.")
        st.stop()
    close = raw["Close"].astype(float)

# --- Indicators ---
short_ma = close.rolling(short, min_periods=1).mean()
long_ma  = close.rolling(long,  min_periods=1).mean()

df_ma = pd.DataFrame({
    "Close": close,
    f"MA{short}": short_ma,
    f"MA{long}": long_ma
}).dropna(how="all")

if df_ma.empty:
    st.error("Not enough data to compute moving averages. Increase the date range.")
    st.stop()

# --- Strategy ---
signal = (short_ma > long_ma).astype(int).shift(1).fillna(0)
daily_ret = close.pct_change().fillna(0.0)
strat_ret = daily_ret * signal

buyhold_curve = (1 + daily_ret).cumprod()
strat_curve   = (1 + strat_ret).cumprod()

# --- Charts ---
col1, col2 = st.columns(2)
with col1:
    st.subheader(f"{ticker} Price and Moving Averages")
    st.line_chart(df_ma)

with col2:
    st.subheader("Strategy vs Buy and Hold")
    curves = pd.DataFrame({
        "Buy and Hold": buyhold_curve,
        "MA Strategy": strat_curve
    }).dropna(how="all")
    st.line_chart(curves)

# --- Metrics (no markdown formatting, plain text only) ---
def ann_vol(r): 
    return float(r.std() * np.sqrt(252))
def sharpe(r):
    v = ann_vol(r)
    return float((r.mean() * 252) / v) if v > 0 else 0.0
def cagr(curve):
    n = len(curve)
    if n == 0 or curve.iloc[-1] <= 0:
        return 0.0
    return float(curve.iloc[-1] ** (252 / max(n, 1)) - 1)

metrics = {
    "CAGR Strategy": cagr(strat_curve),
    "CAGR Buy and Hold": cagr(buyhold_curve),
    "Vol Strategy": ann_vol(strat_ret),
    "Vol Buy and Hold": ann_vol(daily_ret),
    "Sharpe Strategy": sharpe(strat_ret),
    "Sharpe Buy and Hold": sharpe(daily_ret),
}

# Show as a simple table, values formatted in Python to avoid markdown quirks
fmt = {}
for k, v in metrics.items():
    if "CAGR" in k or "Vol" in k:
        fmt[k] = f"{v:.2%}"
    else:
        fmt[k] = f"{v:.2f}"

st.subheader("Key Metrics")
st.table(pd.Series(fmt))
