# Redwood contribution: Main Streamlit dashboard application for portfolio risk cockpit
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import yfinance as yf

# Import custom modules
from data_pipeline import load_config, fetch_portfolio_data, prepare_pipeline_for_ticker
from models import VolatilityPredictor
from risk_math import calculate_portfolio_returns, calculate_historical_var_es, calculate_parametric_var_es, run_monte_carlo_simulation
from backtester import VolatilityBacktester

# ----------------- PAGE CONFIG & THEME -----------------
st.set_page_config(
    page_title="AlphaGuard Risk Cockpit",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sleek Custom CSS for Premium Dark-Mode Aesthetics
st.markdown("""
<style>
    /* Main Background and Text */
    .stApp {
        background-color: #0e1117;
        color: #e2e8f0;
    }
    
    /* Metrics Cards Styling */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #38bdf8; /* sky-400 */
    }
    div[data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #94a3b8; /* slate-400 */
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Custom Card container */
    .metric-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        margin-bottom: 15px;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #f1f5f9;
        font-weight: 600;
    }
    
    /* Tab headers */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 6px 6px 0px 0px;
        color: #94a3b8;
        padding: 10px 16px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0284c7; /* sky-600 */
        color: white !important;
        border-color: #0284c7;
    }
    
    /* Status indicators */
    .acceptable-status {
        color: #10b981; /* emerald-500 */
        font-weight: bold;
    }
    .warning-status {
        color: #f59e0b; /* amber-500 */
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- HELPER DATA FUNCTIONS -----------------
@st.cache_data(ttl=3600)
def load_and_process_data(tickers, start_date):
    """Downloads and runs technical and GARCH features for the portfolio."""
    portfolio_raw = fetch_portfolio_data(tickers, start_date)
    portfolio_processed = {}
    for ticker, df in portfolio_raw.items():
        try:
            portfolio_processed[ticker] = prepare_pipeline_for_ticker(df, ticker)
        except Exception as e:
            st.error(f"Error processing ticker {ticker}: {e}")
    return portfolio_processed

# ----------------- APPLICATION HEADER -----------------
st.title("🛡️ AlphaGuard Portfolio Risk Cockpit")
st.markdown("##### Production-Grade Predictive Volatility Forecasting & Multi-Asset Portfolio Risk Analytics")
st.write("---")

# Load Configurations
config = load_config()

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.header("⚙️ Dashboard Controls")

# Portfolio Setup
tickers_list = st.sidebar.multiselect(
    "Select Portfolio Assets",
    options=["AAPL", "MSFT", "NVDA", "SPY", "QQQ", "GOOGL", "AMZN", "TSLA", "BTC-USD"],
    default=config['tickers']
)

# Portfolio Weight Allocation
st.sidebar.subheader("⚖️ Portfolio Weight Allocation")
weights = {}
if tickers_list:
    total_tickers = len(tickers_list)
    # Distribute weights evenly by default
    default_weight = round(100.0 / total_tickers, 1)
    
    cols = st.sidebar.columns(min(3, total_tickers))
    for idx, ticker in enumerate(tickers_list):
        col = cols[idx % 3]
        weights[ticker] = col.number_input(
            f"{ticker} (%)",
            min_value=0.0,
            max_value=100.0,
            value=default_weight,
            step=5.0
        )
    
    total_weight = sum(weights.values())
    st.sidebar.markdown(f"**Total Portfolio Weight:** `{total_weight:.1f}%`")
    if not np.isclose(total_weight, 100.0):
        st.sidebar.warning("⚠️ Total weight must equal 100%!")
else:
    st.sidebar.info("Select at least one ticker to begin.")

# Model & Parameter Selection
st.sidebar.write("---")
st.sidebar.subheader("🤖 Predictive Engine Settings")
model_choice = st.sidebar.selectbox(
    "Volatility Model Type",
    options=["LSTM (Deep Learning)", "Random Forest (Machine Learning)", "GARCH (Statistical Baseline)"]
)
model_choice_clean = "LSTM" if "LSTM" in model_choice else ("RandomForest" if "Random" in model_choice else "GARCH")

confidence_choice = st.sidebar.select_slider(
    "Risk Confidence Level",
    options=[0.95, 0.99],
    value=0.95
)

initial_capital = st.sidebar.number_input(
    "Initial Portfolio Value ($)",
    min_value=1000,
    max_value=10000000,
    value=int(config['backtest']['initial_capital']),
    step=10000
)

# Training triggers
train_model_btn = st.sidebar.button("🚀 Train & Run Forecasts")

# ----------------- MAIN PIPELINE RUNNER -----------------
if not tickers_list:
    st.info("👈 Please select portfolio assets in the sidebar to visualize risk analytics.")
else:
    # 1. Fetch & Process Data
    with st.spinner("Downloading market data and running GARCH estimations..."):
        portfolio = load_and_process_data(tickers_list, config['data']['start_date'])

    if not portfolio:
        st.error("No valid stock data returned. Check internet connection or tickers.")
        st.stop()

    # Align dates across assets
    common_dates = None
    for ticker, df in portfolio.items():
        if common_dates is None:
            common_dates = df.index
        else:
            common_dates = common_dates.intersection(df.index)
            
    # Filter portfolio to common dates
    portfolio_aligned = {}
    returns_df = pd.DataFrame(index=common_dates)
    for ticker, df in portfolio.items():
        portfolio_aligned[ticker] = df.loc[common_dates]
        returns_df[ticker] = portfolio_aligned[ticker]['Returns']

    # Normalize weights dictionary to fractions
    weights_fraction = {t: w / 100.0 for t, w in weights.items()}
    portfolio_returns = calculate_portfolio_returns(returns_df, weights_fraction)

    # 2. Volatility Forecasting Execution
    feature_cols = ['Open_Gap', 'RSI', 'ATR', 'BBP', 'Vol_Range', 'GARCH_Vol', 'Sentiment']
    predictions = {}
    
    # Check if models are trained/loaded or train on click
    for ticker in tickers_list:
        df = portfolio_aligned[ticker]
        predictor = VolatilityPredictor(ticker, sequence_length=config['model']['sequence_length'], model_type=model_choice_clean)
        
        # Train model if explicitly triggered OR if no model exists
        model_exists = predictor.load_saved_model()
        
        if train_model_btn or not model_exists:
            # Show progress indicator only for explicit training
            if train_model_btn:
                status_slot = st.empty()
                status_slot.info(f"Training {model_choice_clean} model for {ticker}...")
            
            # GARCH does not train a neural network, it directly uses GARCH estimation
            if model_choice_clean != "GARCH":
                predictor.train(df, feature_cols, target_col='Vol_Range', epochs=config['model']['epochs'])
            
            if train_model_btn:
                status_slot.success(f"✓ Model for {ticker} trained and saved successfully!")
                
        # Run prediction
        if model_choice_clean == "GARCH":
            # GARCH prediction is directly the GARCH volatility column
            predictions[ticker] = df['GARCH_Vol'].iloc[-1]
        else:
            try:
                predictions[ticker] = predictor.predict(df, feature_cols)
            except Exception as e:
                # Fallback to GARCH baseline if model fails to load
                predictions[ticker] = df['GARCH_Vol'].iloc[-1]
    
    # Calculate portfolio forecasted volatility (weighted average or covariance based)
    cov_matrix = returns_df.cov()
    w_vec = np.array([weights_fraction[t] for t in tickers_list])
    pred_vol_vector = np.array([predictions[t] for t in tickers_list])
    
    # Forecasted Portfolio Volatility using correlation matrix and predicted volatilities
    corr_matrix = returns_df.corr().values
    portfolio_predicted_vol = np.sqrt(np.dot(w_vec.T * pred_vol_vector.T, np.dot(corr_matrix, w_vec * pred_vol_vector)))

    # 3. Calculate Risk Engine Outputs
    # Parametric VaR
    param_var, param_es, portfolio_hist_vol = calculate_parametric_var_es(returns_df, weights_fraction, confidence_choice)
    
    # We can also compute a dynamic parametric VaR using the forecasted volatility instead of hist vol
    z_score = 1.64485 if confidence_choice == 0.95 else 2.32635
    dynamic_param_var = -(portfolio_returns.mean() - z_score * portfolio_predicted_vol)
    
    # Historical VaR
    hist_var, hist_es = calculate_historical_var_es(portfolio_returns, confidence_choice)
    
    # Monte Carlo VaR
    mc_paths, mc_var, mc_es = run_monte_carlo_simulation(
        returns_df, weights_fraction, 
        init_portfolio_value=initial_capital,
        simulations=config['risk']['monte_carlo_simulations'],
        horizon_days=30,
        confidence_level=confidence_choice
    )

    # ----------------- UI MAIN LAYOUT -----------------
    
    # Top Level KPI Row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 14px; color: #94a3b8; text-transform: uppercase;">Forecasted Portfolio VaR ({int(confidence_choice*100)}%)</div>
            <div style="font-size: 28px; font-weight: bold; color: #f43f5e;">-{dynamic_param_var*100:.2f}%</div>
            <div style="font-size: 12px; color: #64748b;">Potential Loss: ${initial_capital * dynamic_param_var:,.2f} on next trading day</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 14px; color: #94a3b8; text-transform: uppercase;">Expected Shortfall / Tail Risk</div>
            <div style="font-size: 28px; font-weight: bold; color: #fb7185;">-{param_es*100:.2f}%</div>
            <div style="font-size: 12px; color: #64748b;">Expected average loss if VaR is breached: ${initial_capital * param_es:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        # Diversification benefit: Difference between weighted sum of individual vols vs portfolio vol
        weighted_vol = np.sum([weights_fraction[t] * returns_df[t].std() for t in tickers_list])
        div_benefit = (weighted_vol - portfolio_hist_vol) * np.sqrt(252.0) * 100
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 14px; color: #94a3b8; text-transform: uppercase;">Diversification Benefit</div>
            <div style="font-size: 28px; font-weight: bold; color: #10b981;">+{div_benefit:.2f}%</div>
            <div style="font-size: 12px; color: #64748b;">Annualized volatility reduction due to asset correlations</div>
        </div>
        """, unsafe_allow_html=True)

    # Tabs Selection
    tab_vol, tab_var, tab_mc, tab_corr, tab_backtest, tab_stress = st.tabs([
        "📈 Volatility Forecasts", 
        "🛡️ VaR Math Comparison", 
        "🔮 Monte Carlo Projections", 
        "📊 Portfolio Analytics",
        "🔄 Risk Backtesting",
        "💥 Macro Stress Testing"
    ])

    # ----------------- TAB 1: VOLATILITY FORECASTS -----------------
    with tab_vol:
        st.subheader("Volatility Forecasts & Live News Sentiment")
        
        # Let's plot volatility history vs. forecasted next-day vol for a selected asset
        selected_ticker = st.selectbox("Select Asset to View Forecast Details", options=tickers_list)
        
        df_ticker = portfolio_aligned[selected_ticker]
        
        fig_vol = go.Figure()
        # Historical daily range
        fig_vol.add_trace(go.Scatter(
            x=df_ticker.index[-100:], 
            y=df_ticker['Vol_Range'].iloc[-100:] * 100,
            mode='lines',
            name='Daily Volatility Range',
            line=dict(color='#64748b', width=1.5)
        ))
        
        # GARCH Volatility
        fig_vol.add_trace(go.Scatter(
            x=df_ticker.index[-100:], 
            y=df_ticker['GARCH_Vol'].iloc[-100:] * 100,
            mode='lines',
            name='GARCH(1,1) Volatility',
            line=dict(color='#eab308', width=2)
        ))
        
        # Predicted vol point
        next_day = df_ticker.index[-1] + timedelta(days=1)
        fig_vol.add_trace(go.Scatter(
            x=[next_day],
            y=[predictions[selected_ticker] * 100],
            mode='markers+text',
            text=[f"Forecast: {predictions[selected_ticker]*100:.2f}%"],
            textposition="top center",
            name=f"Predicted Vol ({model_choice_clean})",
            marker=dict(color='#06b6d4', size=12, symbol='star')
        ))
        
        fig_vol.update_layout(
            title=f"Volatility Projections for {selected_ticker} (Last 100 Trading Days)",
            xaxis_title="Date",
            yaxis_title="Volatility %",
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_vol, use_container_width=True)
        
        # News Headlines & Live Sentiment Cockpit
        st.markdown(f"### 📰 Live News Sentiment: {selected_ticker}")
        try:
            live_news = yf.Ticker(selected_ticker).news
            if live_news:
                cols = st.columns(min(3, len(live_news)))
                for i, news_item in enumerate(live_news[:3]):
                    content = news_item.get('content', {})
                    title = content.get('title', news_item.get('title', ''))
                    publisher = content.get('provider', {}).get('displayName', news_item.get('publisher', 'News Source'))
                    
                    # Resolve URL from canonicalUrl or clickThroughUrl, falling back to top-level link
                    link = '#'
                    if isinstance(content, dict):
                        link = content.get('canonicalUrl', {}).get('url', content.get('clickThroughUrl', {}).get('url', news_item.get('link', '#')))
                    else:
                        link = news_item.get('link', '#')
                        
                    with cols[i]:
                        st.markdown(f"""
                        <div class="metric-card" style="height: 150px; overflow: hidden;">
                        <strong style="color: #38bdf8;">{publisher}</strong><br/>
                            <p style="font-size: 13px; margin-top: 5px;">{title}</p>
                            <a href="{link}" target="_blank" style="font-size: 12px; color: #10b981;">Read Article →</a>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No recent news headlines found for this ticker.")
        except Exception:
            st.info("Could not load recent news from Yahoo Finance.")

    # ----------------- TAB 2: VaR COMPARISON -----------------
    with tab_var:
        st.subheader("Value at Risk (VaR) Methodology Comparison")
        
        var_data = {
            'Method': ['Historical Simulation', 'Parametric (Var-Cov)', 'Monte Carlo Simulation'],
            'VaR %': [hist_var * 100, param_var * 100, mc_var * 100],
            'ES %': [hist_es * 100, param_es * 100, mc_es * 100],
            'Dollar VaR': [initial_capital * hist_var, initial_capital * param_var, initial_capital * mc_var],
            'Dollar ES': [initial_capital * hist_es, initial_capital * param_es, initial_capital * mc_es]
        }
        var_comparison_df = pd.DataFrame(var_data)
        
        # Display styled table
        st.dataframe(
            var_comparison_df.style.format({
                'VaR %': '{:.2f}%',
                'ES %': '{:.2f}%',
                'Dollar VaR': '${:,.2f}',
                'Dollar ES': '${:,.2f}'
            }),
            use_container_width=True
        )
        
        # Plotly Bar Chart
        fig_var_comp = go.Figure()
        fig_var_comp.add_trace(go.Bar(
            x=var_comparison_df['Method'],
            y=var_comparison_df['VaR %'],
            name=f'Value at Risk (VaR)',
            marker_color='#f43f5e'
        ))
        fig_var_comp.add_trace(go.Bar(
            x=var_comparison_df['Method'],
            y=var_comparison_df['ES %'],
            name='Expected Shortfall (ES / Tail Loss)',
            marker_color='#fda4af'
        ))
        fig_var_comp.update_layout(
            title="Comparison of Risk Metrics by Methodology",
            yaxis_title="Risk Percentage (%)",
            template="plotly_dark",
            barmode='group'
        )
        st.plotly_chart(fig_var_comp, use_container_width=True)

    # ----------------- TAB 3: MONTE CARLO PROJECTIONS -----------------
    with tab_mc:
        st.subheader("Geometric Brownian Motion (GBM) Portfolio Simulations")
        st.write("Below are the projected values for the portfolio over a 30-day trading horizon based on 5,000 simulated correlation-aligned paths.")
        
        # Process MC paths
        mc_steps = mc_paths.shape[0]
        t_steps = np.arange(mc_steps)
        
        # Calculate percentiles
        p_5 = np.percentile(mc_paths, 5, axis=1)
        p_50 = np.percentile(mc_paths, 50, axis=1)
        p_95 = np.percentile(mc_paths, 95, axis=1)
        
        fig_mc = go.Figure()
        # Plot a subset of paths (e.g. 50 paths for visual appeal)
        for i in range(min(50, mc_paths.shape[1])):
            fig_mc.add_trace(go.Scatter(
                x=t_steps,
                y=mc_paths[:, i],
                mode='lines',
                line=dict(width=0.5, color='rgba(94, 234, 212, 0.15)'),
                showlegend=False
            ))
            
        # Highlight Median, 5th, and 95th Percentile
        fig_mc.add_trace(go.Scatter(
            x=t_steps, y=p_95, mode='lines', name='95th Percentile (Optimistic)',
            line=dict(color='#10b981', width=3, dash='dash')
        ))
        fig_mc.add_trace(go.Scatter(
            x=t_steps, y=p_50, mode='lines', name='Median Projection',
            line=dict(color='#38bdf8', width=3)
        ))
        fig_mc.add_trace(go.Scatter(
            x=t_steps, y=p_5, mode='lines', name='5th Percentile (Pessimistic VaR)',
            line=dict(color='#f43f5e', width=3, dash='dash')
        ))
        
        fig_mc.update_layout(
            title=f"Monte Carlo Projections (30-Day Outlook on ${initial_capital:,.2f})",
            xaxis_title="Trading Days",
            yaxis_title="Portfolio Value ($)",
            template="plotly_dark"
        )
        st.plotly_chart(fig_mc, use_container_width=True)

    # ----------------- TAB 4: PORTFOLIO ANALYTICS -----------------
    with tab_corr:
        st.subheader("Asset Allocation and Core Correlation Analysis")
        
        col_pie, col_heat = st.columns(2)
        
        with col_pie:
            # Asset Allocations Pie Chart
            fig_pie = px.pie(
                values=list(weights.values()),
                names=list(weights.keys()),
                title="Current Asset Allocations",
                color_discrete_sequence=px.colors.sequential.Tealgrn,
                hole=0.4
            )
            fig_pie.update_layout(template="plotly_dark")
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_heat:
            # Correlation Matrix Heatmap
            fig_heat = px.imshow(
                returns_df.corr(),
                text_auto='.2f',
                title="Asset Return Correlations (Covariance Basis)",
                color_continuous_scale='RdBu_r',
                zmin=-1.0,
                zmax=1.0
            )
            fig_heat.update_layout(template="plotly_dark")
            st.plotly_chart(fig_heat, use_container_width=True)

    # ----------------- TAB 5: BACKTESTING Strategy -----------------
    with tab_backtest:
        st.subheader("Volatility Targeting Backtest Simulation")
        st.write("Dynamic Volatility Targeting holds cash when risk is high, and invests when risk is low. This strategy reduces drawdowns and matches long-term target volatility.")
        
        # Load backtest engine
        backtester = VolatilityBacktester(initial_capital=initial_capital, risk_free_annual=config['backtest']['risk_free_rate'])
        
        # Use rolling GARCH as a proxy for historical predicted volatility for the full backtest series
        # aligned with portfolio returns
        portfolio_vols_series = returns_df.rolling(window=21).std() * np.sqrt(252.0)
        # Dynamic portfolio historical volatility
        portfolio_combined_vols = []
        for d in returns_df.index:
            r_cov = returns_df.loc[:d].iloc[-252:].cov()
            p_vol = np.sqrt(np.dot(w_vec.T, np.dot(r_cov, w_vec)))
            portfolio_combined_vols.append(p_vol)
            
        portfolio_combined_vols = pd.Series(portfolio_combined_vols, index=returns_df.index)
        
        # Run Backtest
        bt_results = backtester.run_volatility_targeting(
            portfolio_returns, 
            portfolio_combined_vols,
            target_vol_annual=config['backtest']['target_volatility']
        )
        
        # Evaluate performance
        strat_performance = backtester.evaluate_performance(bt_results['strategy_returns'])
        bench_performance = backtester.evaluate_performance(portfolio_returns)
        
        # Breaches
        hist_var_series = portfolio_combined_vols * (1.64 / np.sqrt(252.0)) # approx VaR
        breach_stats = backtester.count_var_breaches(portfolio_returns, hist_var_series, confidence_level=0.95)
        
        # Plot equity curves
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(
            x=bt_results['strategy_equity'].index,
            y=bt_results['strategy_equity'].values,
            name='Volatility Targeted Strategy',
            line=dict(color='#10b981', width=2.5)
        ))
        fig_equity.add_trace(go.Scatter(
            x=bt_results['portfolio_equity'].index,
            y=bt_results['portfolio_equity'].values,
            name='Buy & Hold Portfolio',
            line=dict(color='#3b82f6', width=1.5, dash='dash')
        ))
        fig_equity.update_layout(
            title="Equity Curve: Volatility Targeting Strategy vs. Buy & Hold Portfolio",
            xaxis_title="Date",
            yaxis_title="Portfolio Wealth ($)",
            template="plotly_dark"
        )
        st.plotly_chart(fig_equity, use_container_width=True)
        
        # Performance comparison table
        perf_data = {
            'Metric': ['Cumulative Return', 'Annualized Return', 'Annualized Volatility', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown'],
            'Strategy': [
                f"{strat_performance['cumulative_return']*100:.2f}%",
                f"{strat_performance['annualized_return']*100:.2f}%",
                f"{strat_performance['annualized_volatility']*100:.2f}%",
                f"{strat_performance['sharpe_ratio']:.2f}",
                f"{strat_performance['sortino_ratio']:.2f}",
                f"{strat_performance['max_drawdown']*100:.2f}%"
            ],
            'Buy & Hold Benchmark': [
                f"{bench_performance['cumulative_return']*100:.2f}%",
                f"{bench_performance['annualized_return']*100:.2f}%",
                f"{bench_performance['annualized_volatility']*100:.2f}%",
                f"{bench_performance['sharpe_ratio']:.2f}",
                f"{bench_performance['sortino_ratio']:.2f}",
                f"{bench_performance['max_drawdown']*100:.2f}%"
            ]
        }
        st.dataframe(pd.DataFrame(perf_data), use_container_width=True)
        
        # VaR Breach report card
        st.markdown("### 🔍 Risk Engine Validation (VaR Backtesting)")
        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            st.metric("Total Backtesting Days", f"{breach_stats['total_days']}")
        with col_b2:
            st.metric("Observed VaR Breaches", f"{breach_stats['num_breaches']}", f"Expected: ~{int(breach_stats['total_days'] * breach_stats['expected_ratio'])}")
        with col_b3:
            status_class = "acceptable-status" if breach_stats['breach_ratio'] <= breach_stats['expected_ratio'] * 1.5 else "warning-status"
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 14px; color: #94a3b8; text-transform: uppercase;">Risk Model Validation Status</div>
                <div class="{status_class}">{breach_stats['breach_status']}</div>
                <div style="font-size: 12px; color: #64748b;">Breach Rate: {breach_stats['breach_ratio']*100:.2f}% vs. Expected {breach_stats['expected_ratio']*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

    # ----------------- TAB 6: STRESS TESTING -----------------
    with tab_stress:
        st.subheader("Macro Shock Stress Testing Simulator")
        st.write("Drag the sliders below to apply hypothetical shocks to specific assets and see the immediate impact on your portfolio value.")
        
        shock_weights = {}
        for ticker in tickers_list:
            shock_weights[ticker] = st.slider(f"Macro Shock to {ticker} (%)", min_value=-50, max_value=50, value=0, step=5)
            
        # Calculate impact
        portfolio_shock_impact = 0.0
        for ticker, shock_val in shock_weights.items():
            weight_frac = weights_fraction[ticker]
            portfolio_shock_impact += weight_frac * (shock_val / 100.0)
            
        shocked_portfolio_value = initial_capital * (1.0 + portfolio_shock_impact)
        dollar_change = shocked_portfolio_value - initial_capital
        
        st.write("---")
        st.markdown(f"### 💥 Stress Test Simulation Results")
        
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            color = "#10b981" if dollar_change >= 0 else "#f43f5e"
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 14px; color: #94a3b8; text-transform: uppercase;">Portfolio Shock Return</div>
                <div style="font-size: 28px; font-weight: bold; color: {color};">{portfolio_shock_impact*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_s2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 14px; color: #94a3b8; text-transform: uppercase;">Estimated Portfolio Value</div>
                <div style="font-size: 28px; font-weight: bold; color: #38bdf8;">${shocked_portfolio_value:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_s3:
            color = "#10b981" if dollar_change >= 0 else "#f43f5e"
            sign = "+" if dollar_change >= 0 else ""
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 14px; color: #94a3b8; text-transform: uppercase;">Net Dollar Shock Impact</div>
                <div style="font-size: 28px; font-weight: bold; color: {color};">{sign}${dollar_change:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Draw a bar chart of the asset shock contributions
        shock_contributions = {t: weights_fraction[t] * (shock_weights[t]) for t in tickers_list}
        fig_shocks = px.bar(
            x=list(shock_contributions.keys()),
            y=list(shock_contributions.values()),
            labels={'x': 'Asset', 'y': 'Portfolio Return Contribution (%)'},
            title="Portfolio Return Contribution by Asset Shock",
            color=list(shock_contributions.values()),
            color_continuous_scale=px.colors.diverging.RdYlGn
        )
        fig_shocks.update_layout(template="plotly_dark")
        st.plotly_chart(fig_shocks, use_container_width=True)
