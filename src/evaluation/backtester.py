"""
Backtesting engine.
"""
from src.environment.trading_environment import R3TTradingEnvironment

def run_backtest(agent, test_df, feature_cols, config) -> list:
    env = R3TTradingEnvironment(test_df, feature_cols, config=config)
    obs, _ = env.reset()
    done = False
    
    portfolio_history = [config.get("initial_capital", 100000.0)]
    
    while not done:
        action = agent.select_action(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(action)
        portfolio_history.append(info["portfolio_value"])
        done = done or trunc
        
    return portfolio_history

