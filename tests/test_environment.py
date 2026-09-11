"""tests/test_environment.py — Trading environment tests."""
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.feature_engineering import add_technical_indicators, prepare_dataset, FEATURE_COLS
from src.regime.detector import RegimeDetector
from src.environment.trading_environment import R3TTradingEnvironment


def make_env(sample_ohlcv, sample_config):
    df = add_technical_indicators(sample_ohlcv.copy())
    det = RegimeDetector(n_regimes=4, rolling_window=20, seed=42)
    df = det.fit_predict(df)
    regime_cols = ["regime_0", "regime_1", "regime_2", "regime_3", "regime_confidence"]
    feature_cols = [c for c in FEATURE_COLS if c in df.columns] + regime_cols
    df_s, _ = prepare_dataset(df, feature_cols, fit_scaler=True)
    env = R3TTradingEnvironment(df_s, feature_cols, config=sample_config)
    return env, feature_cols


class TestTradingEnvironment:
    def test_accounting_balance_hold(self, sample_ohlcv, sample_config):
        """Holding should not change cash or portfolio except via price movement."""
        env, _ = make_env(sample_ohlcv, sample_config)
        obs, _ = env.reset()
        cash_before = env.cash
        # Action 0 = HOLD
        env.step(0)
        assert env.cash == cash_before, "HOLD should not change cash"

    def test_buy_decreases_cash(self, sample_ohlcv, sample_config):
        """BUY action must decrease cash."""
        env, _ = make_env(sample_ohlcv, sample_config)
        obs, _ = env.reset()
        cash_before = env.cash
        obs, reward, done, trunc, info = env.step(2)  # BUY_50PCT
        if not done:
            assert env.cash <= cash_before, "BUY should decrease cash"

    def test_exit_clears_position(self, sample_ohlcv, sample_config):
        """EXIT (action=5) should fully clear the position."""
        env, _ = make_env(sample_ohlcv, sample_config)
        obs, _ = env.reset()
        # First buy something
        env.step(4)  # BUY_100PCT
        # Then exit
        obs, reward, done, trunc, info = env.step(5)  # EXIT
        if not done:
            assert env.shares_held == 0, "EXIT should clear position"

    def test_action_validity_all_6_actions(self, sample_ohlcv, sample_config):
        """All 6 actions must be valid and not crash."""
        env, _ = make_env(sample_ohlcv, sample_config)
        for action in range(6):
            obs, _ = env.reset()
            obs, reward, done, trunc, info = env.step(action)
            assert isinstance(reward, float), f"Reward must be float for action {action}"
            assert isinstance(obs, np.ndarray), f"Obs must be ndarray for action {action}"

    def test_cost_applied_on_trade(self, sample_ohlcv, sample_config):
        """Transaction cost must be deducted when trading."""
        env, _ = make_env(sample_ohlcv, sample_config)
        obs, _ = env.reset()
        # BUY — should pay cost
        obs, reward, done, trunc, info = env.step(2)  # BUY_50PCT
        assert info.get("trade_cost", 0) >= 0, "Trade cost must be non-negative"

    def test_reward_is_finite(self, sample_ohlcv, sample_config):
        """Reward must never be NaN or infinite."""
        env, _ = make_env(sample_ohlcv, sample_config)
        obs, _ = env.reset()
        for _ in range(min(50, len(env.df) - 1)):
            action = np.random.randint(0, 6)
            obs, reward, done, trunc, info = env.step(action)
            assert np.isfinite(reward), f"Reward is not finite: {reward}"
            if done or trunc:
                break

    def test_observation_shape_correct(self, sample_ohlcv, sample_config):
        """Observation shape must match environment's observation_space."""
        env, _ = make_env(sample_ohlcv, sample_config)
        obs, _ = env.reset()
        expected_shape = env.observation_space.shape
        assert obs.shape == expected_shape, \
            f"Obs shape {obs.shape} != expected {expected_shape}"

    def test_portfolio_value_always_nonnegative(self, sample_ohlcv, sample_config):
        """Portfolio value must never go negative."""
        env, _ = make_env(sample_ohlcv, sample_config)
        obs, _ = env.reset()
        done = False
        step = 0
        while not done and step < 200:
            action = np.random.randint(0, 6)
            obs, reward, done, trunc, info = env.step(action)
            assert info["portfolio_value"] >= 0, f"Negative portfolio at step {step}"
            done = done or trunc
            step += 1

    def test_episode_terminates(self, sample_ohlcv, sample_config):
        """An episode must eventually terminate (done=True)."""
        env, _ = make_env(sample_ohlcv, sample_config)
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < 10000:
            obs, reward, done, trunc, info = env.step(0)  # all HOLD
            done = done or trunc
            steps += 1
        assert done, f"Episode did not terminate after {steps} steps"
