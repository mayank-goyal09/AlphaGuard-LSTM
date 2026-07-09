# Redwood contribution: Portfolio returns calculation and VaR/ES risk mathematics engine
import numpy as np
import pandas as pd
from scipy.stats import norm

def calculate_portfolio_returns(returns_df, weights):
    """Calculates historical daily portfolio returns given weights and asset returns."""
    # Ensure returns match the weights order
    tickers = list(weights.keys())
    w = np.array([weights[t] for t in tickers])
    
    # Selected returns DataFrame (dropna to prevent NaNs in calculations)
    selected_returns = returns_df[tickers].dropna()
    
    # Portfolio returns: dot product of returns and weights
    portfolio_returns = selected_returns.dot(w)
    return portfolio_returns

def calculate_historical_var_es(portfolio_returns, confidence_level=0.95):
    """Computes Historical Value at Risk (VaR) and Expected Shortfall (ES)."""
    # Clean any remaining NaNs
    portfolio_returns = portfolio_returns.dropna()
    alpha = 1 - confidence_level
    # Historical VaR is the negative of the alpha-quantile of returns
    var = -np.percentile(portfolio_returns, alpha * 100)
    
    # Historical ES is the mean of losses exceeding the VaR
    exceedance_losses = portfolio_returns[portfolio_returns <= -var]
    es = -exceedance_losses.mean() if len(exceedance_losses) > 0 else var
    
    return float(var), float(es)

def calculate_parametric_var_es(returns_df, weights, confidence_level=0.95):
    """Computes Parametric (Variance-Covariance) Value at Risk (VaR) and Expected Shortfall (ES)."""
    tickers = list(weights.keys())
    w = np.array([weights[t] for t in tickers])
    
    # Calculate means and covariance matrix on clean returns
    clean_returns = returns_df[tickers].dropna()
    means = clean_returns.mean()
    cov_matrix = clean_returns.cov()
    
    # Portfolio mean and standard deviation
    portfolio_mean = np.dot(w, means)
    portfolio_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    
    # Z-score for the given confidence level
    z_score = norm.ppf(confidence_level)
    
    # VaR
    var = -(portfolio_mean - z_score * portfolio_vol)
    
    # ES under assumption of normally distributed returns
    # ES = - [mean - vol * (pdf(z) / (1 - alpha))]
    alpha = 1 - confidence_level
    pdf_z = norm.pdf(z_score)
    es = -(portfolio_mean - portfolio_vol * (pdf_z / alpha))
    
    return float(var), float(es), float(portfolio_vol)

def run_monte_carlo_simulation(returns_df, weights, init_portfolio_value=100000.0, 
                               simulations=5000, horizon_days=30, confidence_level=0.95):
    """
    Runs a Monte Carlo simulation of future portfolio values using Geometric Brownian Motion (GBM)
    with correlated assets via Cholesky decomposition.
    """
    tickers = list(weights.keys())
    w = np.array([weights[t] for t in tickers])
    num_assets = len(tickers)
    
    # Compute daily drift (means) and covariance on clean returns
    clean_returns = returns_df[tickers].dropna()
    means = clean_returns.mean().values
    cov_matrix = clean_returns.cov().values
    correlation_matrix = clean_returns.corr().values
    std_devs = clean_returns.std().values
    
    # Cholesky decomposition of the correlation matrix: L L^T = Corr
    # If the correlation matrix is not positive-definite due to numerical issues, use near-PD approximation
    try:
        L = np.linalg.cholesky(correlation_matrix)
    except np.linalg.LinAlgError:
        # Fallback: simple diagonal correlation (uncorrelated assets)
        print("Cholesky decomposition failed. Using uncorrelated assumptions.")
        L = np.eye(num_assets)
        
    # Standard deviations matrix for scaling correlated standard normals to actual covariances
    S = np.diag(std_devs)
    
    # Transform matrix to generate correlated returns: L_cov = S * L
    L_cov = np.dot(S, L)
    
    # Daily step simulation
    # portfolio_paths will hold shape (horizon_days + 1, simulations)
    portfolio_paths = np.zeros((horizon_days + 1, simulations))
    portfolio_paths[0, :] = init_portfolio_value
    
    # Pre-calculate asset price paths: shape (horizon_days + 1, num_assets, simulations)
    asset_paths = np.zeros((horizon_days + 1, num_assets, simulations))
    # Starting asset values based on their weights in the portfolio
    for i in range(num_assets):
        asset_paths[0, i, :] = init_portfolio_value * w[i]
        
    dt = 1.0 # 1 day step
    
    for t in range(1, horizon_days + 1):
        # Generate independent standard normals for each asset and simulation: shape (num_assets, simulations)
        Z = np.random.normal(0, 1, (num_assets, simulations))
        # Correlate them
        epsilon = np.dot(L_cov, Z) # shape (num_assets, simulations)
        
        # Apply Geometric Brownian Motion step
        for i in range(num_assets):
            drift = (means[i] - 0.5 * (std_devs[i]**2)) * dt
            diffusion = epsilon[i, :] * np.sqrt(dt)
            asset_paths[t, i, :] = asset_paths[t-1, i, :] * np.exp(drift + diffusion)
            
        # Sum asset paths to get portfolio paths at time t
        portfolio_paths[t, :] = np.sum(asset_paths[t, :, :], axis=0)
        
    # Calculate simulated portfolio returns over the full horizon
    final_values = portfolio_paths[-1, :]
    horizon_returns = (final_values - init_portfolio_value) / init_portfolio_value
    
    # Calculate Monte Carlo VaR & ES
    alpha = 1 - confidence_level
    mc_var = -np.percentile(horizon_returns, alpha * 100)
    
    exceedance_returns = horizon_returns[horizon_returns <= -mc_var]
    mc_es = -exceedance_returns.mean() if len(exceedance_returns) > 0 else mc_var
    
    return portfolio_paths, float(mc_var), float(mc_es)
