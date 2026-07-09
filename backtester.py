# Redwood contribution: Volatility targeting backtesting simulation engine
import numpy as np
import pandas as pd

class VolatilityBacktester:
    def __init__(self, initial_capital=100000.0, risk_free_annual=0.04):
        self.initial_capital = initial_capital
        self.risk_free_daily = risk_free_annual / 252.0

    def run_volatility_targeting(self, portfolio_returns, predicted_volatility, target_vol_annual=0.15, max_leverage=1.0):
        """
        Simulates a volatility targeting strategy:
        Position size = Target Vol / Predicted Vol.
        Remaining capital goes into the risk-free cash rate.
        """
        # Convert annual target vol to daily target vol
        target_vol_daily = target_vol_annual / np.sqrt(252.0)
        
        # We shift predicted volatility by 1 day because we position at market close of day t
        # to get return of day t+1.
        pred_vol_shifted = pd.Series(predicted_volatility).shift(1)
        
        # Calculate leverage / weights: target_vol / predicted_vol
        leverage = target_vol_daily / pred_vol_shifted
        leverage = leverage.fillna(1.0) # Fallback for first element (NaN)
        
        # Apply leverage caps (e.g. no leverage > 1.0)
        leverage = np.clip(leverage, 0.0, max_leverage)
        
        # Calculate strategy returns
        # R_strat_t = Leverage_{t-1} * R_port_t + (1 - Leverage_{t-1}) * R_rf
        strategy_returns = leverage * portfolio_returns + (1.0 - leverage) * self.risk_free_daily
        
        # Alignment
        strategy_returns = strategy_returns.dropna()
        
        # Compute equity curves
        portfolio_equity = self.initial_capital * (1.0 + portfolio_returns).cumprod()
        strategy_equity = self.initial_capital * (1.0 + strategy_returns).cumprod()
        
        return {
            'strategy_returns': strategy_returns,
            'leverage_history': leverage,
            'portfolio_equity': portfolio_equity,
            'strategy_equity': strategy_equity
        }

    def evaluate_performance(self, returns):
        """Calculates annualized metrics: return, volatility, Sharpe, Sortino, Max Drawdown."""
        returns = pd.Series(returns).dropna()
        n_days = len(returns)
        
        if n_days == 0:
            return {}
            
        # Cumulative return
        cum_return = (1.0 + returns).prod() - 1.0
        
        # Annualized return
        ann_return = (1.0 + cum_return) ** (252.0 / n_days) - 1.0
        
        # Annualized volatility
        ann_vol = returns.std() * np.sqrt(252.0)
        
        # Sharpe ratio (excess return / vol)
        ann_rf = self.risk_free_daily * 252.0
        sharpe = (ann_return - ann_rf) / ann_vol if ann_vol > 0 else 0.0
        
        # Downside deviation for Sortino
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std() * np.sqrt(252.0)
        sortino = (ann_return - ann_rf) / downside_std if downside_std > 0 else 0.0
        
        # Maximum Drawdown
        equity = (1.0 + returns).cumprod()
        running_max = equity.cummax()
        drawdowns = (equity - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        return {
            'cumulative_return': float(cum_return),
            'annualized_return': float(ann_return),
            'annualized_volatility': float(ann_vol),
            'sharpe_ratio': float(sharpe),
            'sortino_ratio': float(sortino),
            'max_drawdown': float(max_drawdown)
        }

    def count_var_breaches(self, actual_returns, predicted_var, confidence_level=0.95):
        """Counts how many times actual loss exceeded predicted VaR."""
        # Align series
        df = pd.DataFrame({
            'returns': actual_returns,
            'predicted_var': predicted_var
        }).dropna()
        
        # Breach occurs when Return < -Predicted_VaR (or loss > VaR)
        df['breach'] = df['returns'] < -df['predicted_var']
        num_breaches = df['breach'].sum()
        total_days = len(df)
        
        breach_ratio = num_breaches / total_days if total_days > 0 else 0.0
        expected_ratio = 1.0 - confidence_level
        
        return {
            'num_breaches': int(num_breaches),
            'total_days': int(total_days),
            'breach_ratio': float(breach_ratio),
            'expected_ratio': float(expected_ratio),
            'breach_status': 'Acceptable' if breach_ratio <= expected_ratio * 1.5 else 'High Breaches (Underestimating Risk)'
        }
