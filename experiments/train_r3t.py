#!/usr/bin/env python3
"""
experiments/train_r3t.py — Full R3T Training (Regime + Dueling Double DQN + Risk-Aware Reward)

This is the main training script for the R3T system.

Usage:
    python -m experiments.train_r3t
    python -m experiments.train_r3t --symbol TCS.NS --episodes 200
    python -m experiments.train_r3t --compare-all   # trains all 4 models and compares
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_data
from src.data.feature_engineering import prepare_dataset, chronological_split, FEATURE_COLS
from src.regime.detector import RegimeDetector
from src.rl.r3t_agent import R3TAgent
from src.rl.dqn import DQNAgent
from src.rl.double_dqn import DoubleDQNAgent
from src.environment.trading_environment import R3TTradingEnvironment
from src.risk.circuit_breaker import DailyCircuitBreaker
from src.evaluation.metrics import compute_metrics
from src.explainability.decision_explainer import DecisionExplainer


def parse_args():
    p = argparse.ArgumentParser(description="Train R3T trading agent")
    p.add_argument("--symbol",      default="TCS.NS",   help="NSE symbol (yfinance format)")
    p.add_argument("--episodes",    type=int, default=200, help="Training episodes")
    p.add_argument("--provider",    default="yfinance",  help="Data provider: yfinance or fyers")
    p.add_argument("--compare-all", action="store_true", help="Train all baselines and compare")
    p.add_argument("--save-model",  default="models/r3t_best.pt", help="Model save path")
    return p.parse_args()


def train_agent(agent, env_train, env_val, n_episodes, exec_cfg, model_name, save_path, circuit_breaker=None):
    """Generic training loop with validation checkpointing."""
    best_val_return = -np.inf
    episode_rewards = []

    for ep in tqdm(range(n_episodes), desc=f"Training {model_name}"):
        obs, _ = env_train.reset()
        if circuit_breaker:
            circuit_breaker.reset()
            circuit_breaker.set_day_start(exec_cfg["initial_capital"])

        total_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, truncated, info = env_train.step(action)

            if circuit_breaker:
                allowed, reason = circuit_breaker.check(info["portfolio_value"])
                if not allowed:
                    action = 0  # force HOLD (can't undo, but prevents future trades)

            agent.store_transition(obs, action, reward, next_obs, done or truncated)
            agent.learn()
            obs = next_obs
            total_reward += reward
            done = done or truncated

        episode_rewards.append(total_reward)

        # ── Validation checkpoint every 10 episodes ────────────────────────
        if (ep + 1) % 10 == 0:
            val_pf = _run_episode(agent, env_val, deterministic=True)
            val_ret = (val_pf[-1] - exec_cfg["initial_capital"]) / exec_cfg["initial_capital"] * 100
            avg_rew = np.mean(episode_rewards[-10:])
            tqdm.write(f"  Ep {ep+1}/{n_episodes} | avg_reward={avg_rew:.4f} | val_return={val_ret:.2f}%")

            if val_ret > best_val_return:
                best_val_return = val_ret
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                agent.save(save_path)

    return episode_rewards, best_val_return


def _run_episode(agent, env, deterministic=True):
    """Run one episode and return portfolio history."""
    obs, _ = env.reset()
    done = False
    portfolio = [env.config.get("initial_capital", 100000)]
    while not done:
        action = agent.select_action(obs, deterministic=deterministic)
        obs, _, done, trunc, info = env.step(action)
        portfolio.append(info["portfolio_value"])
        done = done or trunc
    return portfolio


def run_backtest_with_explanation(agent, test_df, feature_cols, exec_cfg, regime_det):
    """Run backtest on test set, collecting decisions + explanations."""
    env = R3TTradingEnvironment(test_df, feature_cols, config=exec_cfg)
    explainer = DecisionExplainer()
    circuit_breaker = DailyCircuitBreaker(exec_cfg.get("risk", {}))

    obs, _ = env.reset()
    circuit_breaker.reset()
    circuit_breaker.set_day_start(exec_cfg["initial_capital"])

    done = False
    portfolio_history = [exec_cfg["initial_capital"]]
    explanations = []
    ACTION_NAMES = {0:'HOLD', 1:'BUY_25PCT', 2:'BUY_50PCT', 3:'BUY_75PCT', 4:'BUY_100PCT', 5:'EXIT'}

    while not done:
        allowed, cb_reason = circuit_breaker.check(portfolio_history[-1])
        if not allowed:
            action = 0  # HOLD forced by circuit breaker
            confidence = 0.0
        else:
            action = agent.select_action(obs, deterministic=True)
            confidence = agent.get_confidence(obs)

        next_obs, reward, done, trunc, info = env.step(action)
        circuit_breaker.update_peak(info["portfolio_value"])

        portfolio_history.append(info["portfolio_value"])

        # Generate explanation for non-HOLD actions or periodically
        if action != 0 or len(explanations) % 20 == 0:
            step_idx = env.current_step - 1
            if 0 <= step_idx < len(test_df):
                row = test_df.iloc[step_idx]
                state_dict = {col: float(row[col]) for col in feature_cols if col in row.index}

                regime_id = int(row.get("regime", 0)) if "regime" in row.index else 0
                regime_names = {0:'SIDEWAYS', 1:'BULLISH', 2:'BEARISH', 3:'HIGH_VOL'}
                regime_name = regime_names.get(regime_id, 'UNKNOWN')

                expl = explainer.explain(
                    ticker="TCS",
                    action_name=ACTION_NAMES[action],
                    target_allocation=[0,0.25,0.5,0.75,1.0,0][action],
                    confidence=confidence,
                    regime_name=regime_name,
                    risk_score=30,
                    q_values={},
                    state_dict=state_dict,
                    risk_checks={"circuit_breaker": "PASS" if allowed else f"BLOCKED: {cb_reason}"},
                )
                explanations.append(expl)

        obs = next_obs
        done = done or trunc

    return portfolio_history, explanations


def compare_all_models(args, cfg, train_df_s, val_df_s, test_df_s, feature_cols, exec_cfg, regime_det):
    """Train all 4 models and produce comparison table."""
    from src.evaluation.backtester import run_backtest

    results = {}
    model_configs = {
        "VanillaDQN":    DQNAgent,
        "DoubleDQN":     DoubleDQNAgent,
        "DuelingDDQN":   R3TAgent,
        "R3T":           R3TAgent,
    }

    for model_name, AgentClass in model_configs.items():
        print(f"\n{'─'*50}")
        print(f"  Training: {model_name}")
        print(f"{'─'*50}")

        env_train = R3TTradingEnvironment(train_df_s, feature_cols, config=exec_cfg)
        env_val   = R3TTradingEnvironment(val_df_s,   feature_cols, config=exec_cfg)

        agent = AgentClass(
            obs_dim=env_train.observation_space.shape[0],
            action_dim=env_train.action_space.n,
            hidden_dim=cfg["model"]["hidden_dim"],
            gamma=cfg["model"]["gamma"],
            lr=cfg["model"]["learning_rate"],
            batch_size=cfg["model"]["batch_size"],
            buffer_capacity=cfg["model"]["replay_size"],
            target_update_freq=cfg["model"]["target_update_steps"],
            seed=cfg["model"]["seed"],
        )

        save_path = f"models/{model_name.lower()}_best.pt"
        train_agent(agent, env_train, env_val, args.episodes, exec_cfg, model_name, save_path)
        agent.load(save_path)

        portfolio_history = run_backtest(agent, test_df_s, feature_cols, config=exec_cfg)
        metrics = compute_metrics(portfolio_history, initial_capital=exec_cfg["initial_capital"])
        metrics["model"] = model_name
        results[model_name] = metrics

        with open(f"results/metrics/{model_name.lower()}.json", "w") as f:
            json.dump(metrics, f, indent=2)

    # Print comparison table
    print(f"\n\n{'='*90}")
    print(f"  MODEL COMPARISON — {args.symbol}")
    print(f"{'='*90}")
    header_cols = ["model", "total_return_pct", "sharpe_ratio", "sortino_ratio",
                   "max_drawdown_pct", "win_rate_pct", "n_trades"]
    header = f"{'Model':<20}" + "".join(f"{c:>18}" for c in header_cols[1:])
    print(header)
    print("─" * 90)

    for model_name, m in results.items():
        row = f"{model_name:<20}" + "".join(
            f"{m.get(c, float('nan')):>18.2f}" for c in header_cols[1:]
        )
        print(row)

    print(f"\n[SAVED] All results in results/metrics/")
    return results


def main():
    args = parse_args()

    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    np.random.seed(cfg["model"]["seed"])

    print(f"\n{'='*60}")
    print(f"  R3T — Regime-Adaptive Risk-Aware RL Trading")
    print(f"  Symbol: {args.symbol} | Episodes: {args.episodes}")
    print(f"{'='*60}\n")

    # ── Data pipeline ─────────────────────────────────────────────────────────
    print("[1/6] Loading data...")
    df_raw = load_data(args.symbol, provider=args.provider)
    print(f"  Loaded {len(df_raw)} bars for {args.symbol}")

    train_df, val_df, test_df = chronological_split(
        df_raw, cfg["data"]["train_ratio"], cfg["data"]["val_ratio"]
    )
    print(f"  Split → Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # ── Regime detection ──────────────────────────────────────────────────────
    print("[2/6] Detecting market regimes...")
    regime_det = RegimeDetector(
        cfg["regime"]["n_regimes"],
        cfg["regime"]["rolling_window"],
        cfg["regime"]["kmeans_seed"],
    )
    train_df = regime_det.fit_predict(train_df)
    val_df   = regime_det.predict(val_df)
    test_df  = regime_det.predict(test_df)
    regime_det.save("models/regime_detector.pkl")
    print(f"  Regime distribution in train: {train_df['regime'].value_counts().to_dict()}")

    # ── Feature preparation ───────────────────────────────────────────────────
    print("[3/6] Preparing features...")
    regime_cols  = ["regime_0","regime_1","regime_2","regime_3","regime_confidence"]
    feature_cols = [c for c in FEATURE_COLS if c in train_df.columns] + regime_cols

    train_df_s, scaler = prepare_dataset(train_df, feature_cols, fit_scaler=True)
    val_df_s,   _      = prepare_dataset(val_df,   feature_cols, scaler=scaler, fit_scaler=False)
    test_df_s,  _      = prepare_dataset(test_df,  feature_cols, scaler=scaler, fit_scaler=False)

    exec_cfg = {**cfg["execution"], **cfg["reward"], "risk": cfg["risk"]}

    if args.compare_all:
        os.makedirs("results/metrics", exist_ok=True)
        compare_all_models(args, cfg, train_df_s, val_df_s, test_df_s, feature_cols, exec_cfg, regime_det)
        return

    # ── Build R3T Agent ───────────────────────────────────────────────────────
    print("[4/6] Building R3T agent (Regime + Dueling Double DQN)...")
    env_train = R3TTradingEnvironment(train_df_s, feature_cols, config=exec_cfg)
    env_val   = R3TTradingEnvironment(val_df_s,   feature_cols, config=exec_cfg)

    agent = R3TAgent(
        obs_dim=env_train.observation_space.shape[0],
        action_dim=env_train.action_space.n,
        hidden_dim=cfg["model"]["hidden_dim"],
        gamma=cfg["model"]["gamma"],
        lr=cfg["model"]["learning_rate"],
        batch_size=cfg["model"]["batch_size"],
        buffer_capacity=cfg["model"]["replay_size"],
        target_update_freq=cfg["model"]["target_update_steps"],
        seed=cfg["model"]["seed"],
    )
    print(f"  Model: {agent.model_name}")
    print(f"  Obs dim: {env_train.observation_space.shape[0]}, Action dim: {env_train.action_space.n}")

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"[5/6] Training for {args.episodes} episodes...")
    circuit_breaker = DailyCircuitBreaker(cfg["risk"])
    t_start = time.time()

    rewards, best_val = train_agent(
        agent, env_train, env_val, args.episodes, exec_cfg,
        "R3T", args.save_model, circuit_breaker
    )
    print(f"  Training complete in {time.time()-t_start:.1f}s | best_val_return={best_val:.2f}%")

    # ── Backtest ──────────────────────────────────────────────────────────────
    print("[6/6] Backtesting on held-out test set...")
    agent.load(args.save_model)
    portfolio_history, explanations = run_backtest_with_explanation(
        agent, test_df_s, feature_cols, exec_cfg, regime_det
    )

    metrics = compute_metrics(portfolio_history, initial_capital=exec_cfg["initial_capital"])
    metrics["model"] = "R3T"
    metrics["symbol"] = args.symbol
    metrics["episodes"] = args.episodes

    # ── Results ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  R3T BACKTEST RESULTS")
    print(f"{'='*60}")
    key_metrics = [
        ("Total Return", "total_return_pct", "%"),
        ("Annualized Return", "annualized_return_pct", "%"),
        ("Sharpe Ratio", "sharpe_ratio", ""),
        ("Sortino Ratio", "sortino_ratio", ""),
        ("Max Drawdown", "max_drawdown_pct", "%"),
        ("Win Rate", "win_rate_pct", "%"),
        ("Trades", "n_trades", ""),
        ("Calmar Ratio", "calmar_ratio", ""),
        ("CVaR 95%", "cvar_95", "%"),
    ]
    for label, key, unit in key_metrics:
        val = metrics.get(key, float("nan"))
        if isinstance(val, float):
            print(f"  {label:25s}: {val:8.2f}{unit}")
        else:
            print(f"  {label:25s}: {val}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/trades", exist_ok=True)

    with open("results/metrics/r3t.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if explanations:
        with open("results/trades/r3t_explanations.json", "w") as f:
            json.dump(explanations[:20], f, indent=2)  # Save first 20 explanations as sample

    # Save portfolio history for dashboard
    import pandas as pd
    pd.DataFrame({"portfolio_value": portfolio_history}).to_csv(
        "results/r3t_equity_curve.csv", index=False
    )

    print("\n[SAVED] results/metrics/r3t.json")
    print("[SAVED] results/r3t_equity_curve.csv")

    if explanations:
        print("\n─── Sample Explanation ───")
        from src.explainability.decision_explainer import DecisionExplainer
        expl = DecisionExplainer()
        print(expl.format_human_readable(explanations[0]))

    print("\n✓ R3T training complete. Run dashboard: streamlit run src/app/dashboard.py")


if __name__ == "__main__":
    main()
