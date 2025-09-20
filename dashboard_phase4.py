# dashboard_phase4.py
# Phase 4: Multi-stock portfolio backtesting in Streamlit
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st

st.set_page_config(page_title="Finance Dashboard - Phase 4", layout="wide")
st.title("Multi-Stock Portfolio Backtesting")

ALL_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"]

# ---- sidebar controls ----
with st.sidebar:
    st.header("Controls")
    tickers = st.multiselect("Select 2+ stocks", ALL_TICKERS, default=["AAPL","MSFT","GOOGL"])
    start = st.date_input("Start date", pd.to_datetime("2022-01-01"))
    end   = st.date_input("End date",   pd.to_datetime("2025-01-01"))
    strategy = st.selectbox("Strategy", ["Moving Average", "Momentum", "Mean Reversion"])

    if strategy == "Moving Average":
        short = st.slider("Short MA", 5, 100, 50, 1)
        long  = st.slider("Long MA",  20, 300, 200, 5)
        if long <= short:
            st.warning("Long MA must be greater than Short MA.")
            st.stop()
    elif strategy == "Momentum":
        lookback = st.slider("Momentum lookback (days)", 5, 120, 20, 1)
    else:  # Mean Reversion (Bollinger)
        bb_window = st.slider("Bollinger window", 10, 60, 20, 1)
        bb_std    = st.slider("Std dev (bands)", 1.0, 3.0, 2.0, 0.1)

# ---- guards ----
if len(tickers) < 2:
    st.warning("Please select at least 2 stocks.")
    st.stop()
if pd.to_datetime(end) <= pd.to_datetime(start):
    st.warning("End date must be after Start date.")
    st.stop()

# ---- data download (adjusted Close) ----
raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
if raw is None or raw.empty:
    st.error("No data returned. Try a different date range or other tickers.")
    st.stop()

# Handle multi-index vs single-index columns
if isinstance(raw.columns, pd.MultiIndex):
    if "Close" not in raw.columns.get_level_values(0):
        st.error("Unexpected data format: 'Close' not found.")
        st.stop()
    close = raw["Close"].copy()
else:
    # Safety: convert to 2D with one column named after the ticker
    # (shouldn't happen here since we enforce 2+)
    close = raw[["Close"]].copy()
# Keep only selected tickers and drop fully-empty cols
close = close[[c for c in close.columns if c in tickers]].dropna(how="all")
if close.shape[1] < 2:
    st.error("Not enough usable price series after cleaning.")
    st.stop()

# ---- helpers ----
def portfolio_equal_weight_returns(prices: pd.DataFrame):
    daily = prices.pct_change().fillna(0.0)
    port_ret = daily.mean(axis=1)  # equal weight each day
    return daily, port_ret

def moving_average_signals(prices: pd.Series, short_w: int, long_w: int):
    s = prices.rolling(short_w, min_periods=1).mean()
    l = prices.rolling(long_w,  min_periods=1).mean()
    return (s > l).astype(int).shift(1).fillna(0)

def momentum_signals(prices: pd.Series, lb: int):
    mom = prices.pct_change(lb)
    return (mom > 0).astype(int).shift(1).fillna(0)

def mean_reversion_signals(prices: pd.Series, w: int, nstd: float):
    ma = prices.rolling(w, min_periods=1).mean()
    sd = prices.rolling(w, min_periods=1).std().fillna(0.0)
    lower = ma - nstd*sd
    return (prices < lower).astype(int).shift(1).fillna(0)  # buy when below lower band

def build_portfolio_curve(prices: pd.DataFrame, strat: str):
    signals = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for c in prices.columns:
        if strat == "Moving Average":
            signals[c] = moving_average_signals(prices[c], short, long)
        elif strat == "Momentum":
            signals[c] = momentum_signals(prices[c], lookback)
        else:
            signals[c] = mean_reversion_signals(prices[c], bb_window, bb_std)

    daily = prices.pct_change().fillna(0.0)
    # returns only when active; otherwise 0
    active = signals.replace(0, np.nan)
    strat_daily = (daily * signals).where(active.notna(), 0)

    # average across active signals; if none active that day, 0
    active_count = active.notna().sum(axis=1).replace(0, np.nan)
    strat_port_ret = (strat_daily.sum(axis=1) / active_count).fillna(0.0)

    # equal-weight buy & hold
    _, bh_port_ret = portfolio_equal_weight_returns(prices)

    strat_curve = (1 + strat_port_ret).cumprod()
    bh_curve    = (1 + bh_port_ret).cumprod()
    return strat_port_ret, bh_port_ret, strat_curve, bh_curve

def ann_vol(r):  return float(r.std() * np.sqrt(252))
def sharpe(r):
    v = ann_vol(r)
    return float((r.mean()*252) / v) if v > 0 else 0.0
def cagr(curve):
    n = len(curve)
    if n == 0 or curve.iloc[-1] <= 0:
        return 0.0
    return float(curve.iloc[-1] ** (252 / max(n, 1)) - 1)
def max_dd(curve):
    peak = curve.cummax()
    dd = (curve/peak) - 1.0
    return float(dd.min()) if len(dd) else 0.0

# ---- run backtest ----
strat_port_ret, bh_port_ret, strat_curve, bh_curve = build_portfolio_curve(close, strategy)

# ---- charts ----
st.subheader("Prices (selected stocks)")
st.line_chart(close)

st.subheader("Portfolio: Strategy vs Buy & Hold")
curves = pd.DataFrame({
    "Buy & Hold (Equal Weight)": bh_curve,
    f"{strategy} Strategy": strat_curve
})
st.line_chart(curves)

# ---- metrics ----
metrics = {
    "CAGR Strategy": cagr(strat_curve),
    "CAGR Buy & Hold": cagr(bh_curve),
    "Vol Strategy": ann_vol(strat_port_ret),
    "Vol Buy & Hold": ann_vol(bh_port_ret),
    "Sharpe Strategy": sharpe(strat_port_ret),
    "Sharpe Buy & Hold": sharpe(bh_port_ret),
    "MaxDD Strategy": max_dd(strat_curve),
    "MaxDD Buy & Hold": max_dd(bh_curve),
}
fmt = {k: (f"{v:.2%}" if "CAGR" in k or "Vol" in k or "MaxDD" in k else f"{v:.2f}") for k,v in metrics.items()}

st.subheader("Key Portfolio Metrics")
st.table(pd.Series(fmt))

st.caption("Notes: Equal-weight portfolio. Strategy applies per-stock signals then averages active positions daily. Buy & Hold is equal-weight, daily rebalanced.")
