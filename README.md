<!-- Redwood contribution: Project README documentation detailing the purpose, setup, and features of AlphaGuard -->
# 🛡️ AlphaGuard: Portfolio Risk Cockpit & Volatility Predictor

AlphaGuard is an advanced, production-grade quantitative risk management system and interactive financial analytics cockpit. It upgrades simple single-asset forecasting models into a multi-asset portfolio risk intelligence platform powered by **GARCH statistical modeling, Deep Learning (LSTM), Monte Carlo simulations, and news sentiment**.

---

## 🎯 What the Project is For

Traditional risk management often relies on static historical averages which fail to capture rapid changes in market conditions (volatility clustering). **AlphaGuard solves this by predicting tomorrow's risk dynamically.** 

It is built to:
1. **Forecast Asset Volatility**: Predict daily price volatility ranges using technical indicators combined with conditional statistical baselines (GARCH) and machine learning (LSTM/Random Forest).
2. **Quantify Tail Risk (VaR & ES)**: Compute portfolio-level Value at Risk (VaR) and Expected Shortfall (ES) across multiple assets using three distinct quantitative methodologies:
   * **Historical Simulation** (Non-parametric empirical distribution)
   * **Parametric Variance-Covariance** (Correlation matrix-driven analytics)
   * **Monte Carlo Simulation** (Geometric Brownian Motion paths generated via Cholesky decomposition)
3. **Backtest Volatility Strategies**: Simulate a dynamic **Volatility Targeting Strategy** that automatically de-leverages or allocations to cash when risk rises.
4. **Interactive Stress Testing**: Let users simulate market shocks (e.g., tech selloffs, interest rate spikes) and visualize the instantaneous drawdown impact on their capital.

---

## 💡 How It Helps

### 1. Capital Preservation & Drawdown Mitigation
By using the **Volatility Targeting Engine**, investors can scale down equity exposure when the predicted risk spike is high. In backtests, this has shown a significant reduction in Maximum Drawdown compared to a simple buy-and-hold portfolio, saving capital during market corrections.

### 2. Smarter Position Sizing
Traditional models assume asset returns are independent and normally distributed. AlphaGuard uses a covariance-driven correlation matrix and deep learning time-series memory (LSTM) to capture non-linear market dependencies, giving traders a much more accurate picture of their portfolio's real risk budget.

### 3. Tail-Risk Insights
Value at Risk (VaR) tells you the *minimum* loss you will face at a given confidence (e.g., 95% of the time, daily loss won't exceed 1.8%). Expected Shortfall (ES) answers the critical question: *"When things go horribly wrong (in the worst 5% of cases), what is the average expected loss?"* Knowing both numbers prevents catastrophic blowups.

### 4. Interactive Stress Testing
Traders can run hypothetical macro-economic shocks (e.g., *"What happens if NVDA drops 25% and SPY drops 5%?"*) and immediately see their projected net dollar losses and asset risk contributions.

---

## 🛠️ System Architecture & Code Modules

```
project-71-risk-management/
├── config.yaml             # Configurable parameters for tickers, training epochs, and risk budgets
├── requirements.txt        # Dependency list (streamlit, yfinance, pandas_ta, arch, tensorflow, etc.)
├── data_pipeline.py        # Fetches historical data, computes indicators, GARCH vol, and simulated news sentiment
├── models.py               # VolatilityPredictor class supporting LSTM and Random Forest training/inference
├── risk_math.py            # Portfolio return math, Historical/Parametric VaR & ES, and Monte Carlo engine
├── backtester.py           # Volatility Targeting trading simulator and VaR breach validation tracker
├── app.py                  # High-end Streamlit web dashboard interface
└── README.md               # Project overview & documentation
```

---

## 🚀 Getting Started

### 📋 Prerequisites
Ensure you have Python 3.10+ installed.

### 🔧 Installation
1. Clone the repository and navigate to the project directory:
   ```bash
   cd "project 71 risk management"
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

### 💻 Launching the Dashboard
Start the Streamlit application:
```bash
streamlit run app.py
```
This will spin up a local development server and open the interactive dashboard in your default web browser (typically at `http://localhost:8501`).
