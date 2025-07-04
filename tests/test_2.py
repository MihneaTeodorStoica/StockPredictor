# === test_real_data.py ===
"""
Test a trained PPO trading model (`ppo_trader.zip`) on real NVDA data.
Adds: actual NVDA Close price overlaid, and action markers (Buy/Sell/Hold).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rl_trader import StockTradingEnv
import numpy as np

# --- Load real Tesla stock data ---
df = pd.read_csv("data/nvda.csv", parse_dates=["Date"]).sort_values("Date")

# --- Feature engineering (must match training format) ---
df["Daily_Return"] = df["Close"].pct_change().fillna(0)
df["Volatility"]   = (df["High"] - df["Low"]) / df["Open"]
df["MA_10"]        = df["Close"].rolling(10).mean().bfill()
df["MA_30"]        = df["Close"].rolling(30).mean().bfill()
df = df.dropna().reset_index(drop=True)

# --- Load PPO model ---
model = PPO.load("data/ppo_trader")

# --- Setup trading environment ---
window = 30  # match training
transaction_cost = 0.001  # match training
env_fn = lambda: StockTradingEnv(df, window=window, transaction_cost=transaction_cost)
venv = DummyVecEnv([env_fn])
venv = VecNormalize.load("data/vec_normalize.pkl", venv)
venv.training = False
venv.norm_reward = False
obs = venv.reset()
equity_curve = []
actions = []

# --- Run policy on real TSLA data ---
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, info = venv.step(action)
    val = info[0]["portfolio_value"] if isinstance(info, list) else info["portfolio_value"]
    equity_curve.append(val)
    actions.append(int(action[0]) if isinstance(action, np.ndarray) else int(action))
    if done:
        break

# --- Plot results ---
dates = list(df["Date"][window : window + len(equity_curve)])
equity = pd.Series(equity_curve, index=dates)
price = pd.Series(df["Close"][window : window + len(equity_curve)].values, index=dates)

plt.figure(figsize=(14,6))
plt.plot(dates, equity, label="Equity Curve ($)", linewidth=2)
plt.plot(dates, price, label="TSLA Close Price", linestyle="--", alpha=0.7)

# Markers
# for i, act in enumerate(actions):
    # color = {0: "gray", 1: "green", 2: "red"}.get(act, "black")
    # label = {0: "Hold", 1: "Buy", 2: "Sell"}[act]
    # plt.axvline(x=dates[i], color=color, linestyle=":", alpha=0.2)

plt.title("PPO Agent — Equity vs TSLA Price (Real Data)")
plt.ylabel("Portfolio Value / Price")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("data/equity_curve_real_2.png", dpi=300)
plt.show()

print(f"Final equity: ${equity.iloc[-1]:.4f}")
print("Saved: data/equity_curve_real_2.png")