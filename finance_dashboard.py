# finance_dashboard.py
# Phase 1: Basic Finance Dashboard

import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import os

if not os.path.exists("charts"):
    os.makedirs("charts")

# Step 1: Choosing my stocks
stocks = ["AAPL", "MSFT", "GOOGL"]  # Apple, Microsoft, Google

# Step 2: Downloading stock data
data = yf.download(
    stocks,
    start="2022-01-01",
    end="2025-01-01",
    auto_adjust=False  # important!
)["Adj Close"]


# Step 3: Calculating daily returns
returns = data.pct_change().dropna()

# Step 4: Calculating cumulative returns
cumulative_returns = (1 + returns).cumprod()

# Step 5: Calculating volatility (standard deviation)
volatility = returns.std() * (252 ** 0.5)  # annualized volatility

# --- Plot 1: Stock Prices ---
plt.figure(figsize=(10,6))
for stock in stocks:
    plt.plot(data.index, data[stock], label=stock)
plt.legend()
plt.title("Stock Prices")
plt.savefig("charts/stock_prices.png")
plt.close()

# --- Plot 2: Cumulative Returns ---
plt.figure(figsize=(10,6))
for stock in stocks:
    plt.plot(cumulative_returns.index, cumulative_returns[stock], label=stock)
plt.legend()
plt.title("Cumulative Returns")
plt.savefig("charts/cumulative_returns.png")
plt.close()

# --- Plot 3: Volatility ---
plt.figure(figsize=(8,6))
volatility.plot(kind="bar")
plt.title("Annualized Volatility")
plt.savefig("charts/volatility.png")
plt.close()

print("✅ Analysis complete! Charts saved in 'charts' folder.")
