# dashboard_phase5.py
# Clean, working Phase 5: multi-stock portfolio + alerts + simple optimizer + ML + Excel report
import io
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Finance Dashboard - Phase 5", layout="wide")
st.title("Finance Dashboard - Phase 5")

# ---------------- Sidebar ----------------
ALL_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"]
with st.sidebar:
    st.header("Controls")
    tickers = st.multiselect("Select 2+ stocks", ALL_TICKERS, default=["AAPL","MSFT","GOOGL"])
    start = st.date_input("Start date", pd.to_datetime("2022-01-01"))
    end   = st.date_input("End date",   pd.to_datetime("2025-01-01"))
    strategy = st.selectbox("Strategy", ["Moving Average", "Momentum", "Mean Reversion"])
    if strategy == "Moving Average":
        short = st.slider("Short MA", 5, 100, 50, 1)
        long  = st.slider("Long MA",  20, 300, 200, 5)
    elif strategy == "Momentum":
        lookback = st.slider("Momentum lookback (days)", 5, 120, 20, 1)
    else:
        bb_window = st.slider("Bollinger window", 10, 60, 20, 1)
        bb_std    = st.slider("Std dev (bands)", 1.0, 3.0, 2.0, 0.1)

    st.divider()
    st.subheader("Alerts")
    alert_dd   = st.slider("Alert if portfolio drawdown (%) exceeds", 1, 30, 10, 1)
    alert_move = st.slider("Alert if any stock moves > (%) in a day", 1, 15, 5, 1)

    st.divider()
    run_opt = st.checkbox("Run MA optimizer (Sharpe)", value=False)
    short_min = st.slider("Short min", 5, 50, 10, 1)
    long_max  = st.slider("Long max", 60, 300, 200, 5)

# ---------------- Guards ----------------
if len(tickers) < 2:
    st.warning("Pick at least 2 stocks.")
    st.stop()
if pd.to_datetime(end) <= pd.to_datetime(start):
    st.warning("End date must be after Start date.")
    st.stop()
if strategy == "Moving Average" and long <= short:
    st.warning("For MA, Long must be > Short.")
    st.stop()
if run_opt and long_max <= short_min:
    st.warning("Optimizer: Long max must be > Short min.")
    st.stop()

# ---------------- Data ----------------
raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
if raw is None or raw.empty:
    st.error("No data returned. Try different dates/tickers.")
    st.stop()

# Handle MultiIndex vs single-index columns robustly
if isinstance(raw.columns, pd.MultiIndex):
    if "Close" not in raw.columns.get_level_values(0):
        st.error("Bad data format: 'Close' not found.")
        st.stop()
    close = raw["Close"].copy()
else:
    close = raw[["Close"]].copy()
# Keep selected
close = close[[c for c in close.columns if c in tickers]].dropna(how="all")
if close.shape[1] < 2:
    st.error("Not enough usable price series after cleaning.")
    st.stop()

# ---------------- Helpers ----------------
def moving_average_signals(prices: pd.Series, s: int, l: int):
    fast = prices.rolling(s, min_periods=1).mean()
    slow = prices.rolling(l, min_periods=1).mean()
    return (fast > slow).astype(int).shift(1).fillna(0)

def momentum_signals(prices: pd.Series, lb: int):
    mom = prices.pct_change(lb)
    return (mom > 0).astype(int).shift(1).fillna(0)

def mean_reversion_signals(prices: pd.Series, w: int, nstd: float):
    ma = prices.rolling(w, min_periods=1).mean()
    sd = prices.rolling(w, min_periods=1).std().fillna(0.0)
    lower = ma - nstd * sd
    return (prices < lower).astype(int).shift(1).fillna(0)

def build_signals(prices: pd.DataFrame, strat: str):
    sigs = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for c in prices.columns:
        if strat == "Moving Average":
            sigs[c] = moving_average_signals(prices[c], short, long)
        elif strat == "Momentum":
            sigs[c] = momentum_signals(prices[c], lookback)
        else:
            sigs[c] = mean_reversion_signals(prices[c], bb_window, bb_std)
    return sigs

def portfolio_equal_weight_returns(prices: pd.DataFrame):
    daily = prices.pct_change().fillna(0.0)
    port = daily.mean(axis=1)  # equal weight each day
    return daily, port

def build_portfolio(prices: pd.DataFrame, strat: str):
    sigs = build_signals(prices, strat)
    daily = prices.pct_change().fillna(0.0)
    # apply signals per stock; inactive -> 0
    active = sigs.replace(0, np.nan)
    strat_daily = (daily * sigs).where(active.notna(), 0)
    # average across active signals; if none active, 0
    active_count = active.notna().sum(axis=1).replace(0, np.nan)
    strat_port = (strat_daily.sum(axis=1) / active_count).fillna(0.0)
    # equal-weight buy&hold
    _, bh_port = portfolio_equal_weight_returns(prices)
    strat_curve = (1 + strat_port).cumprod()
    bh_curve    = (1 + bh_port).cumprod()
    return sigs, strat_port, bh_port, strat_curve, bh_curve

def ann_vol(r):  return float(r.std() * np.sqrt(252))
def sharpe(r):
    v = ann_vol(r)
    return float((r.mean()*252)/v) if v > 0 else 0.0
def cagr(curve):
    n = len(curve)
    return float(curve.iloc[-1] ** (252/max(n,1)) - 1) if n and curve.iloc[-1] > 0 else 0.0
def max_dd(curve):
    peak = curve.cummax()
    dd = (curve/peak) - 1.0
    return float(dd.min()) if len(dd) else 0.0

# ---------------- Run backtest ----------------
signals, strat_port, bh_port, strat_curve, bh_curve = build_portfolio(close, strategy)

# ---------------- Charts ----------------
st.subheader("Prices (selected stocks)")
st.line_chart(close)

st.subheader("Portfolio: Strategy vs Buy & Hold")
curves = pd.DataFrame({
    "Buy & Hold (Equal Weight)": bh_curve,
    f"{strategy} Strategy": strat_curve
})
st.line_chart(curves)

# ---------------- Metrics ----------------
metrics = {
    "CAGR Strategy": cagr(strat_curve),
    "CAGR Buy & Hold": cagr(bh_curve),
    "Vol Strategy": ann_vol(strat_port),
    "Vol Buy & Hold": ann_vol(bh_port),
    "Sharpe Strategy": sharpe(strat_port),
    "Sharpe Buy & Hold": sharpe(bh_port),
    "MaxDD Strategy": max_dd(strat_curve),
    "MaxDD Buy & Hold": max_dd(bh_curve),
}
fmt = {k: (f"{v:.2%}" if "CAGR" in k or "Vol" in k or "MaxDD" in k else f"{v:.2f}") for k,v in metrics.items()}
st.subheader("Key Portfolio Metrics")
st.table(pd.Series(fmt))

# ---------------- Alerts ----------------
with st.expander("Alerts"):
    dd_series = strat_curve / strat_curve.cummax() - 1.0
    dd = dd_series.min() if len(dd_series) else 0.0
    if dd <= -alert_dd/100.0:
        st.error(f"Drawdown alert: {dd:.2%} ≤ -{alert_dd}%")
    else:
        st.success(f"Drawdown OK: {dd:.2%}")

    daily_moves = close.pct_change().iloc[-1].abs()*100 if len(close) > 1 else pd.Series(dtype=float)
    big = daily_moves[daily_moves > alert_move]
    if not big.empty:
        st.warning("Large daily moves: " + ", ".join([f"{k}: {v:.2f}%" for k,v in big.items()]))
    else:
        st.info("No single-day moves above threshold.")

# ---------------- Simple MA Optimizer (optional) ----------------
if strategy == "Moving Average" and run_opt:
    st.subheader("Optimizer (Sharpe) for Moving Average")
    best = {"short": None, "long": None, "sharpe": -1e9}
    # modest grid to keep it fast
    for s in range(short_min, min(60, long_max)):
        for l in range(max(s+5, s+1), long_max+1):
            sigs = pd.DataFrame({c: moving_average_signals(close[c], s, l) for c in close.columns})
            daily = close.pct_change().fillna(0.0)
            active = sigs.replace(0, np.nan)
            strat_daily = (daily * sigs).where(active.notna(), 0)
            active_count = active.notna().sum(axis=1).replace(0, np.nan)
            port = (strat_daily.sum(axis=1) / active_count).fillna(0.0)
            sr = sharpe(port)
            if sr > best["sharpe"]:
                best = {"short": s, "long": l, "sharpe": sr}
    st.write(f"Best Short={best['short']}, Long={best['long']}, Sharpe={best['sharpe']:.2f}")

# ---------------- ML predictions (per stock) ----------------
st.subheader("ML: Predicted Next-Day Returns (per stock)")

def predict_next_return(series: pd.Series):
    r1 = series.pct_change().fillna(0.0)
    feats = pd.DataFrame({
        "ret1": r1,
        "ret2": r1.shift(1),
        "ret3": r1.shift(2),
        "roll_mean_5": r1.rolling(5, min_periods=1).mean(),
        "roll_std_5": r1.rolling(5, min_periods=1).std().fillna(0.0),
        "roll_mean_10": r1.rolling(10, min_periods=1).mean(),
        "roll_std_10": r1.rolling(10, min_periods=1).std().fillna(0.0),
    }).dropna()
    if len(feats) < 60:
        return None
    X = feats.iloc[:-1].values
    y = r1.shift(-1).dropna().values[:len(X)]
    if len(y) != len(X):
        return None
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    pred = float(model.predict([feats.iloc[-1].values])[0])
    return pred

rows = []
for c in close.columns:
    p = predict_next_return(close[c])
    rows.append({"Ticker": c, "Predicted Next Ret": "N/A" if p is None else f"{p*100:.2f}%"})
st.table(pd.DataFrame(rows))

# ---------------- Download Excel Report ----------------
st.subheader("Download Excel Report")
def build_excel_bytes():
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        close.to_excel(w, sheet_name="Prices")
        curves.to_excel(w, sheet_name="Portfolio Curves")
        pd.Series(metrics).to_frame("Value").to_excel(w, sheet_name="Metrics")
        signals.to_excel(w, sheet_name="Signals")
    out.seek(0)
    return out.getvalue()

st.download_button(
    label="Download Excel Report",
    data=build_excel_bytes(),
    file_name="finance_dashboard_phase5_report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
