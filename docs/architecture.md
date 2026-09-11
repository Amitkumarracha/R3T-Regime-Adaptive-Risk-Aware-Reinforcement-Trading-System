# Architecture

The R3T system consists of the following core components:

1. **Data Layer**: Integrates yfinance and Fyers API for historical and live data. Includes a robust feature engineering pipeline (TA-Lib based) for computing 20+ features.
2. **Regime Detector**: Unsupervised KMeans model that clusters market data into regimes (Sideways, Bullish, Bearish, High Volatility).
3. **Environment**: A custom Gymnasium environment that tracks portfolio value, handles transaction costs, and computes risk-adjusted rewards based on user-defined lambda parameters.
4. **RL Agent**: Dueling Double DQN architecture with confidence gating to map state observations to 6 discrete allocation actions.
5. **Risk Engine**: Circuit breakers for daily loss and maximum drawdown, along with ATR-based position sizing.
6. **Execution Engine**: Paper trading engine that integrates with the RL agent to execute trades with realistic slippage and brokerage costs.
7. **Explainability Module**: Rule-based AI explanation system that parses RL decisions into human-readable text.

