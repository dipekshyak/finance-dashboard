# dashboard_phase5.py
# Phase 5: Multi-stock portfolio backtesting + ML predictions + optimizer + alerts + Excel report

import io
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Finance Dashboard - Phase 5", layout="wide")
st.title("Finance Dashboard - Phase 5")

ALL_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"]

# ---------------- Sidebar controls ----------------
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

    st.divider()
    st.subheader("Alerts")
    alert_dd = st.slider("Alert if portfolio drawdown (%) exceeds", 1, 30, 10, 1)
    alert_move = st.slider("Alert if any stock moves > (%) in a day", 1, 15, 5, 1)

    st.divider()
    st.subheader("Optimizer (for MA)")
    do_opt = st.checkbox("Find best Short/Long (Sharpe)", value=False)
    short_grid = st.slider("Short range (min)", 5, 50, 10, 1)
    long_grid  = st.slider("Long range (max)", 60, 300, 200, 5)

# ---------------- Guards ----------------
if len(tickers) < 2:
    st.warning("Please select at least 2 stocks.")
    st.stop()
if pd.to_datetime(end) <= pd.to_datetime(start):
    st.warning("End date must be after Start date.")
    st.stop()
if do_opt and long_grid <= short_grid:
    st.warning("Optimizer: Long max must be > Short min.")
    st.stop()

# ---------------- Data download (adjusted Close) ----------------
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
    close = raw[["Close"]].copy()  # fallback for single ticker (shouldn't happen here)

# Keep only selected tickers and drop fully-empty cols
close = close[[c for c in close.columns if c in tickers]].dropna(how="all")
if close.shape[1] < 2:
    st.error("Not enough usable price series after cleaning.")
    st.stop()

# ---------------- Helper functions ----------------
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

def build_signals_for_all(prices: pd.DataFrame, strat: str):
    signals = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for c in prices.columns:
        if strat == "Moving Average":
            signals[c] = moving_average_signals(prices[c], short, long)
        elif strat == "Momentum":
            signals[c] = momentum_signals(prices[c], lookback)
        else:
            signals[c] = mean_reversion_signals(prices[c], bb_window, bb_std)
    return signals

def build_portfolio_curve(prices: pd.DataFrame, strat: str):
    signals = build_signals_for_all(prices, strat)
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
    return signals, strat_port_ret, bh_port_ret, strat_curve, bh_curve

# ---------------- Optimizer (for MA only) ----------------
def grid_search_ma(prices: pd.DataFrame, short_min: int, long_max: int):
    best = {"short": None, "long": None, "sharpe": -1e9}
    for s in range(short_min, min(60, long_max)):     # keep search modest
        for l in range(max(s+5, s+1), long_max+1):    # long must be > short
            # build portfolio for these s/l
            daily = prices.pct_change().fillna(0.0)
            sigs = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
            for c in prices.columns:
                sigs[c] = moving_average_signals(prices[c], s, l)
            active = sigs.replace(0, np.nan)
            strat_daily = (daily * sigs).where(active.notna(), 0)
            active_count = active.notna().sum(axis=1).replace(0, np.nan)
            strat_ret = (strat_daily.sum(axis=1) / active_count).fillna(0.0)
            sr = sharpe(strat_ret)
            if sr > best["sharpe"]:
                best = {"short": s, "long": l, "sharpe": sr}
    return best

# ---------------- ML Predictions (per stock) ----------------
def make_features(ser: pd.Series) -> pd.DataFrame:
    r1 = ser.pct_change().fillna(0.0)
    feats = pd.DataFrame({
        "ret1": r1,
        "ret2": r1.shift(1),
        "ret3": r1.shift(2),
        "roll_mean_5": r1.rolling(5, min_periods=1).mean(),
        "roll_std_5": r1.rolling(5, min_periods=1).std().fillna(0.0),
        "roll_mean_10": r1.rolling(10, min_periods=1).mean(),
        "roll_std_10": r1.rolling(10, min_periods=1).std().fillna(0.0),
    })
    feats["target_next_ret"] = r1.shift(-1)
    feats = feats.dropna()
    return feats

def train_predict_next_return(ser: pd.Series):
    feats = make_features(ser)
    if feats.empty or len(feats) < 50:
        return None, None, None
    X = feats.drop(columns=["target_next_ret"]).values
    y = feats["target_next_ret"].values
    # simple split: last 20% as "test-like"
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test   = X[split:], y[split:]
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    # predict next-day return using latest row
    latest_row = feats.drop(columns=["target_next_ret"]).iloc[[-1]].values
    next_pred = float(model.predict(latest_row)[0])
    # simple score: in-sample R^2 on the test-like split
    score = float(model.score(X_test, y_test)) if len(X_test) > 0 else None
    return next_pred, score, len(X)

# ---------------- Run backtest ----------------
signals, strat_port_ret, bh_port_ret, strat_curve, bh_curve = build_portfolio_curve(close, strategy)

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

# ---------------- Alerts ----------------
with st.expander("Alerts"):
    # portfolio drawdown
    dd = (strat_curve / strat_curve.cummax() - 1.0).min()
    if dd <= -alert_dd/100.0:
        st.error(f"Alert: Strategy drawdown {dd:.2%} ≤ -{alert_dd}% threshold.")
    else:
        st.success(f"Strategy drawdown OK: {dd:.2%}")

    # any single-day big move
    daily_moves = close.pct_change().iloc[-1].abs() * 100
    big = daily_moves[daily_moves > alert_move]
    if not big.empty:
        st.warning("Large daily moves: " + ", ".join([f"{k}: {v:.2f}%" for k,v in big.items()]))
    else:
        st.info("No single-day moves above threshold.")

# ---------------- Optimizer (MA only) ----------------
best_params = None
if do_opt and strategy == "Moving Average":
    st.subheader("Optimizer (Sharpe) for Moving Average")
    with st.spinner("Searching Short/Long..."):
        best_params = grid_search_ma(close, short_grid, long_grid)
    st.write(f"Best Short={best_params['short']}, Long={best_params['long']}, Sharpe={best_params['sharpe']:.2f}")

# ---------------- ML predictions per stock ----------------
st.subheader("ML: Predicted Next-Day Returns (per stock)")
rows = []
for c in close.columns:
    pred, score, nobs = train_predict_next_return(close[c])
    if pred is None:
        rows.append({"Ticker": c, "Predicted Next Ret": "N/A", "Model R2": "N/A", "Obs": nobs or 0})
    else:
        rows.append({"Ticker": c, "Predicted Next Ret": f"{pred*100:.2f}%", "Model R2": f"{(score or 0.0):.2f}", "Obs": nobs})
pred_df = pd.DataFrame(rows)
st.table(pred_df)

# ---------------- Download Excel report ----------------
st.subheader("Download Report (Excel)")
def build_excel_bytes():
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        close.to_excel(writer, sheet_name="Prices")
        curves.to_excel(writer, sheet_name="Portfolio Curves")
        pd.Series(metrics).to_frame("Value").to_excel(writer, sheet_name="Metrics")
        if best_params:
            pd.DataFrame([best_params]).to_excel(writer, sheet_name="Optimizer", index=False)
        pred_df.to_excel(writer, sheet_name="ML Predictions", index=False)
        # also include signals (can be large)
        signals.to_excel(writer, sheet_name="Signals")
    output.seek(0)
    return output.getvalue()

excel_bytes = build_excel_bytes()
st.download_button(
    label="Download Excel Report",
    data=excel_bytes,
    file_name="finance_dashboard_phase5_report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.caption("Report contains: Prices, Portfolio curves, Metrics, (optional) Optimizer result, ML predictions, and Signals.")
