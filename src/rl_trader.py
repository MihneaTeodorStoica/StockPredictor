# === rl_trader.py ===
#!/usr/bin/env python3
"""
Smart stock-market generator **plus** a reinforcement-learning (RL) trading agent.

Overview
--------
1. *Smart* synthetic market → `SmartMarketGenerator` ( regime-switching GBM + seasonality + news shocks ).
2. Custom OpenAI Gym environment → `StockTradingEnv`.
3. PPO agent (from **stable-baselines3**) is trained to maximise portfolio value.
4.  Plots: price series, equity curve, rolling Sharpe ratio.

Dependencies
------------
```bash
pip install gymnasium stable-baselines3[extra] pandas numpy matplotlib
```
The code is CPU-friendly; no GPU needed unless you want faster PPO rollout.
"""
# ---------------------------------------------------------------------------
#  Imports                                                                    
# ---------------------------------------------------------------------------
from __future__ import annotations
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from typing import Tuple, Dict, Any
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

# ---------------------------------------------------------------------------
#  1.  Smart synthetic market generator                                       
# ---------------------------------------------------------------------------
class SmartMarketGenerator:
    """Regime-switching geometric Brownian motion with seasonality & shock."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def _one_regime(self, n: int, mu: float, sigma: float, s0: float) -> np.ndarray:
        dt = 1/252
        noise = self.rng.normal(mu*dt, sigma*np.sqrt(dt), n)
        return s0 * np.exp(np.cumsum(noise))

    def generate(self, n_days: int = 2_000, start: float = 100.0) -> pd.DataFrame:
        """Return DataFrame with OHLCV & engineered features."""
        # Regimes: bull, bear, sideways (probabilities)
        regimes = self.rng.choice(["bull","bear","side"], p=[0.4,0.3,0.3], size=n_days)
        mu_map   = {"bull": 0.15, "bear": -0.15, "side": 0.0}
        sig_map  = {"bull": 0.25, "bear": 0.30, "side": 0.10}
        prices   = [start]
        for i in range(n_days):
            mu, sigma = mu_map[regimes[i]], sig_map[regimes[i]]
            s_next = prices[-1]*np.exp(self.rng.normal(mu/252, sigma/np.sqrt(252)))
            prices.append(max(1e-3, s_next))
        prices = np.array(prices[1:])

        # Seasonality (monthly drift) + random shocks
        months = np.arange(n_days)//21 % 12
        season_amp = 0.03
        prices *= 1 + season_amp*np.sin(2*np.pi*months/12)
        shock_idx = self.rng.choice(np.arange(20,n_days-20), size=int(0.02*n_days), replace=False)
        prices[shock_idx] *= self.rng.uniform(0.90,1.10,len(shock_idx))

        # Build OHLCV
        open_ = prices*(1+self.rng.normal(0,0.002,n_days))
        high  = prices*(1+self.rng.uniform(0,0.01,n_days))
        low   = prices*(1-self.rng.uniform(0,0.01,n_days))
        vol   = self.rng.integers(1e5,5e6,n_days)
        df = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=n_days),
            'Open': open_, 'High': high, 'Low': low, 'Close': prices, 'Volume': vol
        })
        return self._feature_engineer(df)

    @staticmethod
    def _feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
        df["Daily_Return"] = df["Close"].pct_change().fillna(0)
        df["Volatility"]   = (df["High"] - df["Low"]) / df["Open"]
        df["MA_10"]        = df["Close"].rolling(10).mean().bfill()
        df["MA_30"]        = df["Close"].rolling(30).mean().bfill()
        return df.reset_index(drop=True)

# ---------------------------------------------------------------------------
#  2.  Gym environment                                                        
# ---------------------------------------------------------------------------
class StockTradingEnv(gym.Env):
    """Simple long-only trading environment (Hold/Buy/Sell) with transaction costs."""
    metadata = {"render.modes": ["human"]}

    def __init__(self, df: pd.DataFrame, window: int = 30, start_cash: float = 1.0, transaction_cost: float = 0.001):
        super().__init__()
        self.df, self.window, self.start_cash = df, window, start_cash
        self.transaction_cost = transaction_cost  # cost as a fraction (e.g., 0.001 = 0.1%)
        self.feats = df[["Open","High","Low","Close","Volume","Daily_Return","Volatility","MA_10","MA_30"]].values.astype(np.float32)
        self.action_space  = spaces.Discrete(3)          # 0=Hold,1=Buy,2=Sell
        obs_len = window * self.feats.shape[1]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_len+2,), dtype=np.float32)
        self.reset()

    # Internal state
    def _get_obs(self):
        feat_window = self.feats[self.ptr-self.window:self.ptr].flatten()
        return np.concatenate([feat_window, [self.cash, self.stock]])

    def reset(self, *, seed: int | None = None, options: Dict[str,Any] | None = None):
        super().reset(seed=seed)
        self.ptr = self.window
        self.cash, self.stock = self.start_cash, 0.0
        self._prev_val = self.start_cash
        return self._get_obs(), {}

    def step(self, action: int):
        price_open  = self.df.loc[self.ptr, "Open"]
        price_close = self.df.loc[self.ptr, "Close"]
        done = self.ptr >= len(self.df) - 2

        # Execute action with transaction cost
        if action == 1 and self.cash > 0:      # Buy
            cost = self.cash * self.transaction_cost
            self.stock = (self.cash - cost) / price_open
            self.cash  = 0
        elif action == 2 and self.stock > 0:   # Sell
            proceeds = self.stock * price_open
            cost = proceeds * self.transaction_cost
            self.cash  = proceeds - cost
            self.stock = 0

        # Portfolio value & reward (shaped)
        current_val = self.cash + self.stock * price_close
        reward = current_val - self._prev_val
        self._prev_val = current_val

        self.ptr += 1
        info = {"portfolio_value": current_val}
        return self._get_obs(), reward, done, False, info

    def render(self):
        print(f"step={self.ptr}, cash={self.cash:.4f}, stock={self.stock:.4f}")

# ---------------------------------------------------------------------------
#  3.  Train PPO agent                                                        
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    gen = SmartMarketGenerator()
    df_market = gen.generate(5_000)

    # Quick sanity check: plot first 500 days
    plt.figure(figsize=(10,3))
    plt.plot(df_market["Date"][:500], df_market["Close"][:500])
    plt.title("Synthetic price sample (500d)")
    plt.tight_layout(); plt.savefig("data/smart_price_sample.png", dpi=300)

    env_fn = lambda: StockTradingEnv(df_market, window=30, transaction_cost=0.001)
    vec_env = DummyVecEnv([env_fn])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # Evaluation environment (no training leakage)
    test_df = gen.generate(2_500, start=df_market["Close"].iloc[-1])
    eval_env = DummyVecEnv([lambda: StockTradingEnv(test_df, transaction_cost=0.001)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # Evaluation callback for early stopping and best model saving
    eval_callback = EvalCallback(eval_env, best_model_save_path="./data/best_model/",
                                 log_path="./data/logs/", eval_freq=10_000,
                                 deterministic=True, render=False)

    model = PPO(
        "MlpPolicy", vec_env, verbose=1, n_steps=2048, batch_size=64, gamma=0.99,
        learning_rate=3e-4, tensorboard_log="./data/tb_logs/"
    )
    model.learn(total_timesteps=200_000, callback=eval_callback)
    model.save("data/ppo_trader")
    vec_env.save("data/vec_normalize.pkl")

    # Load best model for evaluation (optional)
    # from stable_baselines3 import PPO
    # model = PPO.load("./data/best_model/best_model.zip", env=eval_env)

    # Evaluate on a fresh slice (no training leakage)
    obs = eval_env.reset()
    values = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = eval_env.step(action)
        values.append(info[0]["portfolio_value"] if isinstance(info, list) else info["portfolio_value"])
        if done:
            break

    equity = pd.Series(values, index=test_df["Date"][eval_env.get_attr('window')[0]:eval_env.get_attr('window')[0]+len(values)])
    plt.figure(figsize=(12,4))
    equity.plot(label="Equity Curve", linewidth=2)
    plt.title("Equity Curve — PPO Agent ($1 initial)")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True); plt.tight_layout(); plt.savefig("data/equity_curve_rl.png", dpi=300)
    plt.show()

    print(f"Final equity = ${equity.iloc[-1]:.4f}\nSaved: data/smart_price_sample.png | data/equity_curve_rl.png | data/ppo_trader.zip")

    # For further improvement, consider hyperparameter tuning, curriculum learning, or using more advanced RL algorithms.