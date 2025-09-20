# finance_dashboard_phase2.py
# Phase 2: Portfolio simulation & optimization (kid-friendly)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# ---------- setup ----------
if not os.path.exists("charts"):
    os.makedirs("charts")

# pick your tickers (>=2 please)
STOCKS = ["AAPL", "MSFT", "GOOGL"]   # you can add more later
START = "2022-01-01"
END   = "2025-01-01"

# ---------- data download ----------
# auto_adjust=True = prices are already adjusted; use the 'Close' level
raw = yf.download(STOCKS, start=START, end=END, auto_adjust=True)
close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw

# just in case you ever pass a single stock:
if isinstance(close, pd.Series):
    close = close.to_frame(name=STOCKS[0])

# ---------- basic metrics ----------
returns_daily = close.pct_change().dropna()
# annualized stats (252 trading days)
returns_annual = returns_daily.mean() * 252
cov_annual = returns_daily.cov() * 252
vol_annual = returns_daily.std() * np.sqrt(252)

# ---------- simulate portfolios ----------
def simulate_portfolios(annual_ret, cov_matrix, n_portfolios=5000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(annual_ret)
    all_weights = []
    rets = []
    vols = []
    sharpes = []
    for _ in range(n_portfolios):
        w = rng.random(n)
        w = w / w.sum()  # weights sum to 1
        all_weights.append(w)

        pr = float(np.dot(w, annual_ret.values))
        pv = float(np.sqrt(np.dot(w.T, np.dot(cov_matrix.values, w))))
        sr = pr / (pv + 1e-12)  # avoid divide-by-zero

        rets.append(pr); vols.append(pv); sharpes.append(sr)

    df = pd.DataFrame({"Return": rets, "Volatility": vols, "Sharpe": sharpes})
    return df, np.array(all_weights)

results, weights = simulate_portfolios(returns_annual, cov_annual, n_portfolios=5000)

# ---------- find optimal ----------
idx_max_sharpe = results["Sharpe"].idxmax()
idx_min_vol    = results["Volatility"].idxmin()

w_max_sharpe = weights[idx_max_sharpe]
w_min_vol    = weights[idx_min_vol]

# ---------- print summary ----------
def pretty_weights(tickers, w):
    pairs = [f"{t}: {w[i]*100:.1f}%" for i, t in enumerate(tickers)]
    return ", ".join(pairs)

print("\n=== Phase 2: Portfolio Optimization ===")
print(f"Max Sharpe — Return: {results.loc[idx_max_sharpe, 'Return']:.2%} | "
      f"Vol: {results.loc[idx_max_sharpe, 'Volatility']:.2%} | "
      f"Weights: {pretty_weights(close.columns.tolist(), w_max_sharpe)}")

print(f"Min Vol    — Return: {results.loc[idx_min_vol,    'Return']:.2%} | "
      f"Vol: {results.loc[idx_min_vol,    'Volatility']:.2%} | "
      f"Weights: {pretty_weights(close.columns.tolist(), w_min_vol)}")

# ---------- plot ----------
plt.figure(figsize=(12,6))
sc = plt.scatter(results["Volatility"], results["Return"],
                 c=results["Sharpe"], alpha=0.55, s=10)
plt.colorbar(sc, label="Sharpe")

# highlight the two optimal portfolios
plt.scatter(results.loc[idx_max_sharpe, "Volatility"],
            results.loc[idx_max_sharpe, "Return"],
            marker="*", s=300, label="Max Sharpe")
plt.scatter(results.loc[idx_min_vol, "Volatility"],
            results.loc[idx_min_vol, "Return"],
            marker="*", s=300, label="Min Vol")

plt.title("Simulated Portfolios (Risk vs Return)")
plt.xlabel("Volatility (Annualized)")
plt.ylabel("Return (Annualized)")
plt.legend()
plt.tight_layout()
plt.savefig("charts/portfolio_simulation.png")
plt.close()

print("✅ Saved: charts/portfolio_simulation.png")
