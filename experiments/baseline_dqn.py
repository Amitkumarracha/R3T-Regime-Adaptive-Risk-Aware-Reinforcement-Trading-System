#!/usr/bin/env python3
"""
experiments/baseline_dqn.py — Vanilla DQN Baseline (Baseline A)

Trains a vanilla DQN on the primary symbol and runs a backtest.
Results saved to results/metrics/baseline_dqn.json and results/trades/baseline_dqn.csv

Usage:
    python -m experiments.baseline_dqn
    python -m experiments.baseline_dqn --symbol INFY.NS --episodes 100
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml
from tqdm import tqdm

from src.data.loader import load_data
from src.data.feature_engineering import prepare_dataset, chronological_split, FEATURE_COLS
from src.regime.detector import RegimeDetector
from src.rl.dqn import DQNAgent
from src.environment.trading_environment import R3TTradingEnvironment
from src.evaluation.metrics import compute_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="TCS.NS")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--provider", default="yfinance")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    np.random.seed(cfg["model"]["seed"])

    print(f"\n{'='*60}")
    print(f"  BASELINE A: Vanilla DQN — {args.symbol}")
    print(f"{'='*60}\n")

    # ── Load & prepare data ───────────────────────────────────────────────────
    print("[1/5] Loading data...")
    df_raw = load_data(args.symbol, provider=args.provider)
    print(f"  Loaded {len(df_raw)} rows for {args.symbol}")

    train_df, val_df, test_df = chronological_split(
        df_raw,
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
    )
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # ── Regime detection ──────────────────────────────────────────────────────
    print("[2/5] Running regime detection...")
    regime_det = RegimeDetector(
        n_regimes=cfg["regime"]["n_regimes"],
        rolling_window=cfg["regime"]["rolling_window"],
        seed=cfg["regime"]["kmeans_seed"],
    )
    train_df = regime_det.fit_predict(train_df)
    val_df   = regime_det.predict(val_df)
    test_df  = regime_det.predict(test_df)

    # ── Prepare datasets (scale on train only) ─────────────────────────────
    regime_cols = ["regime_0", "regime_1", "regime_2", "regime_3", "regime_confidence"]
    feature_cols = [c for c in FEATURE_COLS if c in train_df.columns] + regime_cols

    train_df_s, scaler = prepare_dataset(train_df, feature_cols, fit_scaler=True)
    val_df_s,   _      = prepare_dataset(val_df,   feature_cols, scaler=scaler, fit_scaler=False)
    test_df_s,  _      = prepare_dataset(test_df,  feature_cols, scaler=scaler, fit_scaler=False)

    # ── Build environment & agent ─────────────────────────────────────────────
    exec_cfg = cfg["execution"]
    exec_cfg.update(cfg["reward"])

    env = R3TTradingEnvironment(train_df_s, feature_cols, config=exec_cfg)
    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=cfg["model"]["hidden_dim"],
        gamma=cfg["model"]["gamma"],
        lr=cfg["model"]["learning_rate"],
        batch_size=cfg["model"]["batch_size"],
        buffer_capacity=cfg["model"]["replay_size"],
        target_update_freq=cfg["model"]["target_update_steps"],
        seed=cfg["model"]["seed"],
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"[3/5] Training Vanilla DQN for {args.episodes} episodes...")
    episode_rewards = []
    best_val_return = -np.inf

    for ep in tqdm(range(args.episodes), desc="Training"):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            agent.store_transition(obs, action, reward, next_obs, done or truncated)
            agent.learn()
            obs = next_obs
            total_reward += reward

        episode_rewards.append(total_reward)

        # Validate every 10 episodes
        if (ep + 1) % 10 == 0:
            val_env = R3TTradingEnvironment(val_df_s, feature_cols, config=exec_cfg)
            val_obs, _ = val_env.reset()
            val_done = False
            val_portfolio = []
            while not val_done:
                val_action = agent.select_action(val_obs, deterministic=True)
                val_obs, _, val_done, val_trunc, val_info = val_env.step(val_action)
                val_portfolio.append(val_info["portfolio_value"])
                val_done = val_done or val_trunc
            val_return = (val_portfolio[-1] - exec_cfg["initial_capital"]) / exec_cfg["initial_capital"] * 100
            tqdm.write(f"  Ep {ep+1}/{args.episodes} | train_reward={total_reward:.4f} | val_return={val_return:.2f}%")
            if val_return > best_val_return:
                best_val_return = val_return
                os.makedirs("models", exist_ok=True)
                agent.save("models/baseline_dqn_best.pt")

    # ── Backtest on test set ──────────────────────────────────────────────────
    print("[4/5] Running backtest on test set...")
    agent.load("models/baseline_dqn_best.pt")
    test_env = R3TTradingEnvironment(test_df_s, feature_cols, config=exec_cfg)
    obs, _ = test_env.reset()
    done = False
    portfolio_history = [exec_cfg["initial_capital"]]
    trade_log = []

    while not done:
        action = agent.select_action(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        portfolio_history.append(info["portfolio_value"])
        if info.get("trade_cost", 0) > 0:
            trade_log.append(info)
        done = done or truncated

    # ── Metrics ───────────────────────────────────────────────────────────────
    print("[5/5] Computing metrics...")
    metrics = compute_metrics(portfolio_history, initial_capital=exec_cfg["initial_capital"])
    metrics["model"] = "VanillaDQN"
    metrics["symbol"] = args.symbol
    metrics["episodes"] = args.episodes

    print("\n─── Vanilla DQN Results ───")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")
        else:
            print(f"  {k:30s}: {v}")

    # ── Save results ──────────────────────────────────────────────────────────
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/trades", exist_ok=True)
    with open("results/metrics/baseline_dqn.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\n[SAVED] results/metrics/baseline_dqn.json")


if __name__ == "__main__":
    main()
