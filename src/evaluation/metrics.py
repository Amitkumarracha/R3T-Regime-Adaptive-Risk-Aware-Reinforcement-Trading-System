"""
Evaluation metrics computation.
"""
import numpy as np
import pandas as pd

def compute_metrics(portfolio_history: list, risk_free_rate: float = 0.065, initial_capital: float = 100000.0) -> dict:
    if len(portfolio_history) < 2:
        return {}
        
    pf = np.array(portfolio_history)
    returns = (pf[1:] - pf[:-1]) / pf[:-1]
    
    total_return = (pf[-1] - initial_capital) / initial_capital
    
    # Approx annualized assuming daily steps
    days = len(returns)
    annualized_return = (1 + total_return) ** (252 / max(1, days)) - 1
    
    volatility = np.std(returns) * np.sqrt(252)
    
    # Sharpe
    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf
    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
    
    # Sortino
    downside = excess_returns[excess_returns < 0]
    downside_vol = np.std(downside) * np.sqrt(252) if len(downside) > 0 else 0
    sortino = np.mean(excess_returns) / downside_vol * np.sqrt(252) if downside_vol > 0 else 0
    
    # Max Drawdown
    peaks = np.maximum.accumulate(pf)
    drawdowns = (peaks - pf) / peaks
    max_drawdown = np.max(drawdowns)
    
    # Calmar
    calmar = annualized_return / max_drawdown if max_drawdown > 0 else 0
    
    # CVaR 95
    if len(returns) > 0:
        var_95 = np.percentile(returns, 5)
        cvar_95 = np.mean(returns[returns <= var_95])
    else:
        cvar_95 = 0.0
        
    # Win rate (days)
    win_rate = np.mean(returns > 0)
    
    return {
        "total_return_pct": total_return * 100,
        "annualized_return_pct": annualized_return * 100,
        "volatility_pct": volatility * 100,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown_pct": max_drawdown * 100,
        "calmar_ratio": calmar,
        "cvar_95": cvar_95 * 100,
        "win_rate_pct": win_rate * 100,
        "n_trades": len(portfolio_history) # Approximate
    }

