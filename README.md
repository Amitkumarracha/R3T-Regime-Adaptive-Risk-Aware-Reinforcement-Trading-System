# R3T — Regime-Adaptive Risk-Aware Reinforcement Trading System

**An AI-driven, risk-aware financial market decision simulator for NSE trading that adapts to market regimes, optimizes risk-adjusted decisions under transaction costs, dynamically sizes positions, and provides explainable paper-trading decisions.**

## 1. Problem Statement
Traditional signal-based trading algorithms and naive RL systems often suffer from overtrading, where transaction costs can quickly turn apparent gross profits into net losses. Furthermore, markets are non-stationary; a model trained in a bullish regime may perform disastrously in a high-volatility or bearish regime. There is a need for an intelligent trading system that not only learns when to trade and how much to allocate, but also knows *when not to trade* by incorporating explicit risk controls, regime awareness, and explainable decision-making.

## 2. Key Features
- **Regime Detection**: Unsupervised K-Means clustering identifies market states (Bullish, Bearish, Sideways, High Volatility).
- **Explainable Decisions**: Every AI action is accompanied by a human-readable explanation comprising trend, momentum, regime, and risk factors.
- **Risk Engine**: Hard circuit breakers (daily loss limit, max portfolio drawdown) and ATR-based position sizing limit exposure dynamically.
- **Cost-Aware RL**: Rewards heavily penalize turnover, drawdown, and volatility to maximize risk-adjusted returns, not just raw profits.
- **Data Integration**: Built-in support for `yfinance` (backtesting) and `fyers_apiv3` (paper trading).

## 3. Architecture Diagram
```text
                         MARKET DATA
                              │
                              ▼
                    Feature Engineering
                              │
               ┌──────────────┴──────────────┐
               │                             │
               ▼                             ▼
       Market Regime Detector           Risk Engine
       Bull / Bear / Sideways        Volatility / DD /
       High Volatility               Exposure / Cost
               │                             │
               └──────────────┬──────────────┘
                              ▼
                     RL Decision Engine
                  Double + Dueling DQN
                              │
                    Action + Confidence
                              │
                              ▼
                       Pre-trade Gate
                         /         \
                     REJECT       APPROVE
                       │              │
                       ▼              ▼
                     HOLD        Paper Execute
                                      │
                                      ▼
                              Portfolio Monitor
                                      │
                                      ▼
                         Risk-adjusted Evaluation
                                      │
                                      ▼
                               Explainability
```

## 4. AI Algorithm
The core decision engine uses a **Dueling Double Deep Q-Network (D3QN)**. 
- **Double DQN**: Decouples action selection from action evaluation to reduce the overestimation bias common in standard DQN.
- **Dueling Architecture**: Separates the state-value stream from the advantage stream, allowing the agent to learn the value of being in a specific market regime independently from the effect of taking a specific allocation action.
- **Confidence Gate**: A softmax layer outputs prediction confidence. If confidence is low, the system vetoes high-allocation trades (e.g., reverting a 100% BUY to a 25% BUY or HOLD).

## 5. Risk-Control
The **Risk Engine** acts as a hard boundary outside the learned policy:
- **Daily Circuit Breaker**: Halts trading if daily loss exceeds a configured threshold (e.g., 1.5%).
- **Maximum Drawdown**: Halts trading if the portfolio drops below the maximum permitted drawdown from its peak.
- **Position Sizing**: Dynamically calculates the number of shares based on Average True Range (ATR) and a fixed risk-per-trade percentage (e.g., 0.5% of capital).

## 6. Explainability
An **Explainable AI (XAI)** module evaluates the state vector alongside the network's prediction. Every paper trade generates an explanation JSON detailing:
- The recognized market regime.
- Short-term trend (EMA slopes) and momentum (RSI/MACD).
- Prediction confidence percentage.
- Risk Engine approval status.

## 7. Installation
Create a virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 8. Configuration
System hyperparameters are defined in `configs/config.yaml`:
- **market**: `symbol` (e.g., `TCS.NS`), `interval` (e.g., `1d`)
- **model**: `learning_rate`, `gamma`, `batch_size`, `replay_size`
- **risk**: `max_daily_loss_pct`, `max_drawdown_pct`, `max_position_pct`
- **reward**: `lambda_cost`, `lambda_drawdown`, `lambda_volatility`, `lambda_turnover`

## 9. How to Run & Train
To train the full R3T system on a specific symbol:
```bash
python -m experiments.train_r3t --symbol TCS.NS --episodes 100
```
To run the ablation study comparing Vanilla DQN, Double DQN, Dueling DDQN, and the full R3T model:
```bash
python -m experiments.train_r3t --compare-all
```

## 10. How to Backtest
The training script automatically evaluates the agent on a chronologically held-out test set at the end of training. Alternatively, run the standalone evaluation suite:
```bash
python -m src.evaluation.backtester
```
*(Ensure a trained model exists in the `models/` directory first).*

## 11. Launch Dashboard
Visualize equity curves, drawdowns, AI actions, and explanations using Streamlit:
```bash
streamlit run src/app/dashboard.py
```

## 12. Example Result Screenshots
*(Insert screenshots of the Streamlit dashboard here once run locally)*
- `[Screenshot: Equity Curve vs Baseline]`
- `[Screenshot: AI Decision Explanation Panel]`

## 13. Limitations
- **Historical Data Non-Stationarity**: Past performance does not guarantee future results. Regimes change constantly.
- **Slippage & Market Impact**: The paper execution engine models slippage and brokerage, but extreme liquidity events or market impact of large orders are simplified.
- **Training Constraints**: The MVP is trained on limited data epochs; for a production model, a much longer walk-forward validation on tick data would be required.

## 14. Future Work
- **Live Execution Adapters**: Connect the `execution` module to live broker APIs (Fyers, Upstox).
- **Advanced Regimes**: Upgrade K-Means to Hidden Markov Models (HMM) or deep temporal clustering.
- **Portfolio Optimization**: Extend the environment from single-stock to multi-asset orchestration.
- **Cloud Deployment**: Containerize the inference engine and dashboard for AWS ECS/EC2 deployment.

## 15. Disclaimer
> ⚠️ **Important Financial Disclaimer**
> R3T is an educational, research, and software-engineering prototype. Backtested or simulated performance does not guarantee future returns. Real trading introduces additional risks including market gaps, liquidity constraints, spread, slippage, latency, model drift, API failures, execution failure, taxes/charges, and regulatory requirements. This software should be used for paper-trading and decision-support simulations only. Real-money deployment must only occur after extensive forward testing and verification of applicable broker and regulatory requirements.
