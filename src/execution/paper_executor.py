"""Paper executor module."""

import pandas as pd
from typing import Dict, Any
from .base_executor import AbstractExecutor

class PaperExecutor(AbstractExecutor):
    def __init__(self, config=None):
        self.config = config or {}
        self.initial_capital = self.config.get("initial_capital", 100000.0)
        self.slippage_bps = self.config.get("slippage_bps", 5)
        self.brokerage_per_leg = self.config.get("brokerage_per_leg", 20.0)
        self.brokerage_pct = self.config.get("brokerage_pct", 0.0003)
        
        self.target_allocations = [0.0, 0.25, 0.50, 0.75, 1.0, 0.0]
        self.reset(self.initial_capital)
        
    def reset(self, initial_capital: float):
        self.cash = initial_capital
        self.shares_held = 0
        self.entry_price = 0.0
        self.trade_log = []
        
    def _compute_cost(self, trade_value: float) -> float:
        brokerage = min(self.brokerage_per_leg, trade_value * self.brokerage_pct)
        slippage = (self.slippage_bps / 10000.0) * trade_value
        return brokerage + slippage

    def execute(self, action: int, current_price: float, state: Dict[str, Any] = None) -> Dict[str, Any]:
        prev_equity = self.cash + (self.shares_held * current_price)
        
        if action == 5:
            target_alloc = 0.0
        elif action == 0:
            target_alloc = (self.shares_held * current_price) / prev_equity if prev_equity > 0 else 0.0
        else:
            target_alloc = self.target_allocations[action]
            
        current_alloc = (self.shares_held * current_price) / prev_equity if prev_equity > 0 else 0.0
        
        trade_cost = 0.0
        shares_traded = 0
        side = "NONE"
        
        if action != 0 and abs(target_alloc - current_alloc) > 0.01:
            target_value = prev_equity * target_alloc
            current_value = self.shares_held * current_price
            value_to_trade = target_value - current_value
            
            if value_to_trade > 0: # Buy
                shares_to_buy = int(value_to_trade / current_price)
                if shares_to_buy > 0:
                    trade_val = shares_to_buy * current_price
                    cost = self._compute_cost(trade_val)
                    if self.cash >= trade_val + cost:
                        self.cash -= (trade_val + cost)
                        total_cost = (self.shares_held * self.entry_price) + trade_val
                        self.shares_held += shares_to_buy
                        self.entry_price = total_cost / self.shares_held
                        trade_cost = cost
                        shares_traded = shares_to_buy
                        side = "BUY"
                        
            elif value_to_trade < 0: # Sell
                shares_to_sell = min(self.shares_held, int(abs(value_to_trade) / current_price))
                if action == 5:
                    shares_to_sell = self.shares_held
                    
                if shares_to_sell > 0:
                    trade_val = shares_to_sell * current_price
                    cost = self._compute_cost(trade_val)
                    self.cash += (trade_val - cost)
                    self.shares_held -= shares_to_sell
                    if self.shares_held == 0:
                        self.entry_price = 0.0
                    trade_cost = cost
                    shares_traded = shares_to_sell
                    side = "SELL"
                    
        new_equity = self.cash + (self.shares_held * current_price)
        
        if shares_traded > 0:
            self.trade_log.append({
                "side": side,
                "shares": shares_traded,
                "price": current_price,
                "cost": trade_cost,
                "equity_after": new_equity
            })
            
        return {
            "portfolio_value": new_equity,
            "cash": self.cash,
            "shares_held": self.shares_held,
            "trade_cost": trade_cost
        }
        
    def get_portfolio_state(self) -> Dict[str, Any]:
        return {
            "cash": self.cash,
            "shares_held": self.shares_held,
            "entry_price": self.entry_price
        }
        
    def save_log(self, path: str):
        if self.trade_log:
            pd.DataFrame(self.trade_log).to_csv(path, index=False)

