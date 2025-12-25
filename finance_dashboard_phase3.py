# finance_dashboard_phase3.py
# Phase 3 (Part A): Moving-Average (MA) backtest for one stock

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# ----- setup -----
if not os.path.exists("charts"):
    os.makedirs("charts")

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
START = "2022-01-01"
END   = "2025-01-01"

def backtest_ma(close, short=50, long=200):
    """Return a dict with curves + metrics for an MA crossover strategy."""
    short_ma = close.rolling(short).mean()
    long_ma  = close.rolling(long).mean()

    # signal: 1 when short>long else 0  (use yesterday's signal to trade today)
    signal = (short_ma > long_ma).astype(int).shift(1).fillna(0)

    daily_ret = close.pct_change().fillna(0.0)
    strat_ret = daily_ret * signal

    # equity curves
    buyhold_curve = (1 + daily_ret).cumprod()
    strat_curve   = (1 + strat_ret).cumprod()

    # metrics
    n = len(daily_ret)
    def cagr(curve):
        return curve.iloc[-1] ** (252 / max(n,1)) - 1
    def ann_vol(ret):
        return ret.std() * np.sqrt(252)
    def sharpe(ret):
        vol = ann_vol(ret)
        return (ret.mean() * 252) / vol if vol > 0 else 0.0
    def max_dd(curve):
        peak = curve.cummax()
        dd = (curve / peak) - 1
        return dd.min()

    metrics = {
        "CAGR_buyhold": cagr(buyhold_curve),
        "CAGR_strategy": cagr(strat_curve),
        "Vol_buyhold": ann_vol(daily_ret),
        "Vol_strategy": ann_vol(strat_ret),
        "Sharpe_buyhold": sharpe(daily_ret),
        "Sharpe_strategy": sharpe(strat_ret),
        "MaxDD_buyhold": max_dd(buyhold_curve),
        "MaxDD_strategy": max_dd(strat_curve),
    }

    return {
        "buyhold_curve": buyhold_curve,
        "strat_curve": strat_curve,
        "short_ma": short_ma,
        "long_ma": long_ma,
        "signal": signal,
        "metrics": metrics,
    }

def main():
    print("Available tickers:", ", ".join(TICKERS))
    ticker = input("Type one ticker exactly (e.g., AAPL): ").strip().upper()
    if ticker not in TICKERS:
        print("Not in list. Edit TICKERS at the top to add more.")
        return

    # download adjusted prices; use 'Close'
    df = yf.download(ticker, start=START, end=END, auto_adjust=True)
    close = df["Close"]

    # choose windows (kid-safe defaults)
    short = 50
    long = 200
    res = backtest_ma(close, short=short, long=long)

    # print metrics
    m = res["metrics"]
    print("\n=== Moving-Average Backtest ===")
    print(f"Ticker: {ticker} | Short={short} | Long={long}")
    print(f"CAGR  — Strategy: {m['CAGR_strategy']:.2%} | Buy&Hold: {m['CAGR_buyhold']:.2%}")
    print(f"Vol   — Strategy: {m['Vol_strategy']:.2%} | Buy&Hold: {m['Vol_buyhold']:.2%}")
    print(f"Sharpe— Strategy: {m['Sharpe_strategy']:.2f}  | Buy&Hold: {m['Sharpe_buyhold']:.2f}")
    print(f"MaxDD — Strategy: {m['MaxDD_strategy']:.2%} | Buy&Hold: {m['MaxDD_buyhold']:.2%}")

    # plot curves
    plt.figure(figsize=(12,6))
    res["buyhold_curve"].plot(label="Buy & Hold")
    res["strat_curve"].plot(label="MA Strategy")
    plt.title(f"{ticker}: Strategy vs Buy & Hold (Cumulative Growth)")
    plt.legend()
    plt.tight_layout()
    out = f"charts/{ticker}_ma_strategy.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved chart: {out}")

if __name__ == "__main__":
    main()
