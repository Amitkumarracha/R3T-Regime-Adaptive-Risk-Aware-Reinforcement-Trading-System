#!/usr/bin/env python3
"""
experiments/dueling_ddqn.py — Dueling Double DQN Baseline (Baseline C)
Without regime features — to isolate the regime contribution.

Usage:
    python -m experiments.dueling_ddqn
"""

import argparse, json, os, sys, numpy as np, yaml
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_data
from src.data.feature_engineering import prepare_dataset, chronological_split, FEATURE_COLS
from src.regime.detector import RegimeDetector
from src.rl.r3t_agent import R3TAgent
from src.environment.trading_environment import R3TTradingEnvironment
from src.evaluation.metrics import compute_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",   default="TCS.NS")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--provider", default="yfinance")
    p.add_argument("--no-regime", action="store_true", help="Exclude regime features (pure dueling baseline)")
    return p.parse_args()


def main():
    args = parse_args()
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    np.random.seed(cfg["model"]["seed"])

    label = "Dueling DDQN (no regime)" if args.no_regime else "Dueling DDQN + Regime"
    print(f"\n{'='*60}")
    print(f"  BASELINE C: {label} — {args.symbol}")
    print(f"{'='*60}\n")

    df_raw = load_data(args.symbol, provider=args.provider)
    train_df, val_df, test_df = chronological_split(df_raw, cfg["data"]["train_ratio"], cfg["data"]["val_ratio"])

    regime_det = RegimeDetector(cfg["regime"]["n_regimes"], cfg["regime"]["rolling_window"], cfg["regime"]["kmeans_seed"])
    train_df = regime_det.fit_predict(train_df)
    val_df   = regime_det.predict(val_df)
    test_df  = regime_det.predict(test_df)

    regime_cols  = [] if args.no_regime else ["regime_0","regime_1","regime_2","regime_3","regime_confidence"]
    feature_cols = [c for c in FEATURE_COLS if c in train_df.columns] + regime_cols

    train_df_s, scaler = prepare_dataset(train_df, feature_cols, fit_scaler=True)
    val_df_s,   _      = prepare_dataset(val_df,   feature_cols, scaler=scaler, fit_scaler=False)
    test_df_s,  _      = prepare_dataset(test_df,  feature_cols, scaler=scaler, fit_scaler=False)

    exec_cfg = {**cfg["execution"], **cfg["reward"]}
    env = R3TTradingEnvironment(train_df_s, feature_cols, config=exec_cfg)

    agent = R3TAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dim=cfg["model"]["hidden_dim"],
        gamma=cfg["model"]["gamma"],
        lr=cfg["model"]["learning_rate"],
        batch_size=cfg["model"]["batch_size"],
        buffer_capacity=cfg["model"]["replay_size"],
        target_update_freq=cfg["model"]["target_update_steps"],
        seed=cfg["model"]["seed"],
    )

    best_val_return = -np.inf
    for ep in tqdm(range(args.episodes), desc="Training"):
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, trunc, info = env.step(action)
            agent.store_transition(obs, action, reward, next_obs, done or trunc)
            agent.learn()
            obs = next_obs
            done = done or trunc

        if (ep + 1) % 10 == 0:
            val_env = R3TTradingEnvironment(val_df_s, feature_cols, config=exec_cfg)
            vobs, _ = val_env.reset()
            vdone = False; vp = []
            while not vdone:
                va = agent.select_action(vobs, deterministic=True)
                vobs, _, vdone, vt, vi = val_env.step(va)
                vp.append(vi["portfolio_value"])
                vdone = vdone or vt
            vr = (vp[-1] - exec_cfg["initial_capital"]) / exec_cfg["initial_capital"] * 100
            tqdm.write(f"  Ep {ep+1} | val_return={vr:.2f}%")
            if vr > best_val_return:
                best_val_return = vr
                os.makedirs("models", exist_ok=True)
                fname = "models/dueling_ddqn_no_regime_best.pt" if args.no_regime else "models/dueling_ddqn_best.pt"
                agent.save(fname)

    fname = "models/dueling_ddqn_no_regime_best.pt" if args.no_regime else "models/dueling_ddqn_best.pt"
    agent.load(fname)
    test_env = R3TTradingEnvironment(test_df_s, feature_cols, config=exec_cfg)
    obs, _ = test_env.reset()
    done = False; ph = [exec_cfg["initial_capital"]]
    while not done:
        a = agent.select_action(obs, deterministic=True)
        obs, _, done, trunc, info = test_env.step(a)
        ph.append(info["portfolio_value"])
        done = done or trunc

    metrics = compute_metrics(ph, initial_capital=exec_cfg["initial_capital"])
    metrics["model"] = "DuelingDDQN_NoRegime" if args.no_regime else "DuelingDDQN"
    metrics["symbol"] = args.symbol

    print(f"\n─── {label} Results ───")
    for k, v in metrics.items():
        if isinstance(v, float): print(f"  {k:30s}: {v:.4f}")
        else: print(f"  {k:30s}: {v}")

    os.makedirs("results/metrics", exist_ok=True)
    fname_out = "results/metrics/dueling_ddqn_no_regime.json" if args.no_regime else "results/metrics/dueling_ddqn.json"
    with open(fname_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[SAVED] {fname_out}")


if __name__ == "__main__":
    main()
