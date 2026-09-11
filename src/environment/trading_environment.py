"""
Gymnasium trading environment for R3T.
Supports 6 discrete allocation actions and computes risk-aware rewards.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class R3TTradingEnvironment(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, feature_cols: list, config=None, render_mode=None):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.config = config or {}
        self.render_mode = render_mode
        
        self.initial_capital = self.config.get("initial_capital", 100000.0)
        self.slippage_bps = self.config.get("slippage_bps", 5)
        self.brokerage_per_leg = self.config.get("brokerage_per_leg", 20.0)
        
        self.lambda_cost = self.config.get("lambda_cost", 0.5)
        self.lambda_dd = self.config.get("lambda_dd", 0.8)
        self.lambda_vol = self.config.get("lambda_vol", 0.2)
        self.lambda_turnover = self.config.get("lambda_turnover", 0.1)

        # Actions: 0=HOLD, 1=BUY_25PCT, 2=BUY_50PCT, 3=BUY_75PCT, 4=BUY_100PCT, 5=EXIT
        self.action_space = spaces.Discrete(6)
        self.target_allocations = [0.0, 0.25, 0.50, 0.75, 1.0, 0.0]
        
        # Observation space = feature_cols + 7 portfolio state vars
        obs_dim = len(feature_cols) + 7
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_capital
        self.shares_held = 0
        self.entry_price = 0.0
        self.peak_equity = self.initial_capital
        self.day_start_equity = self.initial_capital
        self.portfolio_history = [self.initial_capital]
        self.trade_history = []
        return self._get_observation(), {}

    def _compute_transaction_cost(self, trade_value: float) -> float:
        brokerage = min(self.brokerage_per_leg, trade_value * self.config.get("brokerage_pct", 0.0003))
        slippage = (self.slippage_bps / 10000.0) * trade_value
        return brokerage + slippage

    def step(self, action: int):
        current_price = float(self.df.iloc[self.current_step]["Close_raw"])
        regime = int(self.df.iloc[self.current_step].get("regime", 0))
        
        prev_equity = self.cash + (self.shares_held * current_price)
        
        if action == 5:
            target_alloc = 0.0 # EXIT
        elif action == 0:
            target_alloc = (self.shares_held * current_price) / prev_equity if prev_equity > 0 else 0.0 # HOLD maintains roughly current
        else:
            target_alloc = self.target_allocations[action]
            
        current_alloc = (self.shares_held * current_price) / prev_equity if prev_equity > 0 else 0.0
        
        trade_cost = 0.0
        allocation_change = 0.0
        
        # Execute trade to reach target allocation
        if action != 0 and abs(target_alloc - current_alloc) > 0.01:
            target_value = prev_equity * target_alloc
            current_value = self.shares_held * current_price
            value_to_trade = target_value - current_value
            
            if value_to_trade > 0: # Buy
                shares_to_buy = int(value_to_trade / current_price)
                if shares_to_buy > 0:
                    trade_val = shares_to_buy * current_price
                    cost = self._compute_transaction_cost(trade_val)
                    if self.cash >= trade_val + cost:
                        self.cash -= (trade_val + cost)
                        if self.shares_held == 0:
                            self.entry_price = current_price
                        else:
                            # average price
                            total_cost = (self.shares_held * self.entry_price) + trade_val
                            self.entry_price = total_cost / (self.shares_held + shares_to_buy)
                        self.shares_held += shares_to_buy
                        trade_cost = cost
                        allocation_change = (trade_val / prev_equity)
            elif value_to_trade < 0: # Sell
                shares_to_sell = min(self.shares_held, int(abs(value_to_trade) / current_price))
                # If action is EXIT (5), sell everything
                if action == 5:
                    shares_to_sell = self.shares_held
                    
                if shares_to_sell > 0:
                    trade_val = shares_to_sell * current_price
                    cost = self._compute_transaction_cost(trade_val)
                    self.cash += (trade_val - cost)
                    self.shares_held -= shares_to_sell
                    if self.shares_held == 0:
                        self.entry_price = 0.0
                    trade_cost = cost
                    allocation_change = (trade_val / prev_equity)

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        new_price = float(self.df.iloc[self.current_step]["Close_raw"])
        new_equity = self.cash + (self.shares_held * new_price)
        self.portfolio_history.append(new_equity)
        self.peak_equity = max(self.peak_equity, new_equity)
        
        # Risk-aware reward
        log_ret = np.log(new_equity / prev_equity) if prev_equity > 0 else 0.0
        drawdown = (self.peak_equity - new_equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        cost_ratio = trade_cost / self.initial_capital
        
        # Approx realized vol from recent portfolio history
        if len(self.portfolio_history) > 20:
            hist = np.array(self.portfolio_history[-21:])
            rets = np.log(hist[1:] / hist[:-1])
            realized_vol = np.std(rets) * np.sqrt(252)
        else:
            realized_vol = 0.0
            
        reward = (
            log_ret
            - self.lambda_cost * cost_ratio
            - self.lambda_dd * drawdown
            - self.lambda_vol * realized_vol
            - self.lambda_turnover * abs(allocation_change)
        )
        
        info = {
            "portfolio_value": new_equity,
            "cash": self.cash,
            "shares_held": self.shares_held,
            "current_price": new_price,
            "action_taken": action,
            "target_allocation": target_alloc,
            "trade_cost": trade_cost,
            "regime": regime
        }
        
        return self._get_observation(), float(reward), done, False, info

    def _get_observation(self) -> np.ndarray:
        row = self.df.iloc[self.current_step]
        market_feats = [float(row[c]) if c in row else 0.0 for c in self.feature_cols]
        
        curr_price = float(row["Close_raw"])
        equity = self.cash + (self.shares_held * curr_price)
        
        cash_ratio = np.clip(self.cash / self.initial_capital, 0, 1)
        pos_ratio = np.clip((self.shares_held * curr_price) / self.initial_capital, 0, 1)
        
        unrealized_pnl = 0.0
        if self.shares_held > 0 and self.entry_price > 0:
            unrealized_pnl = (curr_price - self.entry_price) / self.entry_price
        unrealized_pnl_ratio = np.clip(unrealized_pnl, -0.5, 0.5) + 0.5 # map to 0-1
        
        drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        daily_pnl = (equity - self.day_start_equity) / self.day_start_equity if self.day_start_equity > 0 else 0.0
        daily_pnl_ratio = np.clip(daily_pnl, -0.1, 0.1) * 5 + 0.5 # scale and center at 0.5
        
        exposure_ratio = (self.shares_held * curr_price) / equity if equity > 0 else 0.0
        cost_ratio = 0.0 # Just a placeholder since cost is realized instantly
        
        port_feats = [cash_ratio, pos_ratio, unrealized_pnl_ratio, drawdown, daily_pnl_ratio, exposure_ratio, cost_ratio]
        return np.array(market_feats + port_feats, dtype=np.float32)

    def render(self):
        print(f"Step: {self.current_step} | Eq: {self.cash + self.shares_held * self.df.iloc[self.current_step]['Close_raw']:.2f}")

