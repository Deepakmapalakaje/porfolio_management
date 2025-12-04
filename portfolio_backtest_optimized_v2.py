import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from urllib.parse import quote
import warnings
import time
import scipy.stats as stats
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
import os
import sqlite3
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import itertools
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# RAG imports
from rag_prediction_simple import find_similar_scenarios

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# === CONFIGURATION ===
BASE_URL = "https://api.upstox.com/v3/historical-candle"
ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI3M0FIOUMiLCJqdGkiOiI2OTJiMGMxZGIzNGEzMTEzNGI4Njg3ZGQiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6dHJ1ZSwiaWF0IjoxNzY0NDI4ODI5LCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3NjQ0NTM2MDB9.FyfmIVYRBjw8p51FnR532XWaSaBSHOmptXy3Es_22JA"

# DATES
START_DATE = "2018-09-03"
END_DATE = "2025-12-02"

# STRATEGY PARAMS
LOOKBACK_WINDOW = 120
FORWARD_WINDOW = 60
WARMUP_DAYS = 120  # Additional days for indicator warmup
MIN_REQUIRED_DAYS = LOOKBACK_WINDOW + WARMUP_DAYS  # 240 days minimum

RISK_FREE_RATE = 0.065
TRANSACTION_COST = 0.001
INITIAL_CAPITAL = 100000.0
BENCHMARK_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"

# DATABASE
DATABASE_FILE = os.path.join("database", "portfolio_backtest_tcn.db")

# STOCK UNIVERSE (10 STOCKS)
STOCK_UNIVERSE = {
    "NSE_EQ|INE029A01011": "BPCL",
    "NSE_EQ|INE674K01013": "ABCAPITAL",
    "NSE_EQ|INE296A01032": "BAJFINANCE",
    "NSE_EQ|INE397D01024": "BHARTIARTL",
    "NSE_EQ|INE062A01020": "SBIN",
    "NSE_EQ|INE237A01028": "KOTAKBANK",
    "NSE_EQ|INE238A01034": "AXISBANK",
    "NSE_EQ|INE030A01027": "HINDUNILVR",
    "NSE_EQ|INE280A01028": "TITAN",
    "NSE_EQ|INE917I01010": "BAJAJ-AUTO"
}

# === DATABASE SETUP ===
def init_db():
    """Initialize SQLite database for storing backtest results"""
    os.makedirs(os.path.dirname(DATABASE_FILE), exist_ok=True)
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS backtest_results (
            id INTEGER PRIMARY KEY,
            rebalance_date TEXT,
            portfolio TEXT,
            tcn_prediction REAL,
            rag_ci_lower REAL,
            rag_ci_upper REAL,
            risk_reward_ratio REAL,
            actual_return REAL,
            exit_date TEXT,
            exit_reason TEXT,
            portfolio_entry_value REAL,
            portfolio_exit_value REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def store_backtest_result(rebalance_date, portfolio, tcn_pred, rag_ci_lower, rag_ci_upper, 
                         rr_ratio, actual_return, exit_date, exit_reason, entry_val, exit_val):
    """Store backtest results in database"""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        portfolio_str = ",".join([STOCK_UNIVERSE[k] for k in portfolio])
        
        cursor.execute('''
            INSERT INTO backtest_results 
            (rebalance_date, portfolio, tcn_prediction, rag_ci_lower, rag_ci_upper, 
             risk_reward_ratio, actual_return, exit_date, exit_reason, portfolio_entry_value, portfolio_exit_value)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (rebalance_date, portfolio_str, tcn_pred, rag_ci_lower, rag_ci_upper, 
              rr_ratio, actual_return, exit_date, exit_reason, entry_val, exit_val))
        
        conn.commit()
        conn.close()
        print(f"  [DB] Stored result for {portfolio_str}")
    except Exception as e:
        print(f"  [DB ERROR] {e}")

# === DATA FETCHING ===
def fetch_historical_data(instrument_key, from_date, to_date):
    """Fetch historical candle data for a given instrument and date range."""
    encoded_instrument_key = quote(instrument_key, safe='')
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}", "Accept": "application/json"}
    
    from_date_dt = pd.to_datetime(from_date)
    to_date_dt = pd.to_datetime(to_date)
    chunks = []
    current_start = from_date_dt
    
    while current_start < to_date_dt:
        current_end = min(current_start + pd.Timedelta(days=365), to_date_dt)
        chunks.append((current_start.strftime('%Y-%m-%d'), current_end.strftime('%Y-%m-%d')))
        current_start = current_end + pd.Timedelta(days=1)
    
    all_data = []
    for chunk_from, chunk_to in chunks:
        url = f"{BASE_URL}/{encoded_instrument_key}/days/1/{chunk_to}/{chunk_from}"
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json().get("data", {}).get("candles", [])
                all_data.extend(data)
            else:
                print(f"  Error fetching {instrument_key}: Status {response.status_code}")
            time.sleep(0.1)
        except Exception as e:
            print(f"  Error fetching data for {instrument_key}: {e}")
            continue
    
    if all_data:
        df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.index = df["timestamp"].dt.tz_localize(None)
        df.sort_index(inplace=True)
        df["returns"] = df["close"].pct_change()
        return df.dropna()
    return pd.DataFrame()

# === TECHNICAL INDICATORS ===
def calculate_indicators(df):
    """Calculate all technical indicators - EXACT copy from rebalanced_predictive_model.py"""
    df = df.copy()
    median_price = (df['high'] + df['low']) / 2
    
    def smma(series, window):
        return series.ewm(alpha=1/window, adjust=False).mean()
    
    # Alligator Indicator
    df['alligator_jaw'] = smma(median_price, 13).shift(8)
    df['alligator_teeth'] = smma(median_price, 8).shift(5)
    df['alligator_lips'] = smma(median_price, 5).shift(3)
    
    # Enhanced Alligator Features
    df['alligator_jaw_teeth_dist'] = df['alligator_jaw'] - df['alligator_teeth']
    df['alligator_teeth_lips_dist'] = df['alligator_teeth'] - df['alligator_lips']
    df['alligator_jaw_lips_dist'] = df['alligator_jaw'] - df['alligator_lips']
    
    def get_formation_code(row):
        jaw, teeth, lips = row['alligator_jaw'], row['alligator_teeth'], row['alligator_lips']
        tol = abs(row['close']) * 0.001
        if abs(jaw - teeth) < tol and abs(teeth - lips) < tol:
            return 8
        elif abs(jaw - teeth) < tol:
            return 5 if teeth > lips else 4
        elif abs(teeth - lips) < tol:
            return 7 if jaw > teeth else 6
        else:
            if jaw > teeth and teeth > lips:
                return 0
            elif jaw > teeth and teeth < lips:
                return 1
            elif jaw < teeth and teeth > lips:
                return 2
            elif jaw < teeth and teeth < lips:
                return 3
            return 0
    
    df['alligator_formation'] = df.apply(get_formation_code, axis=1)
    
    def get_position_state(row):
        price = row['close']
        jaw_above = row['alligator_jaw'] > price
        teeth_above = row['alligator_teeth'] > price
        lips_above = row['alligator_lips'] > price
        above_count = sum([jaw_above, teeth_above, lips_above])
        
        if above_count == 3:
            return 0
        elif above_count == 0:
            return 1
        elif above_count == 1 and jaw_above:
            return 3
        elif above_count == 1 and not jaw_above:
            return 4
        else:
            return 2
    
    df['alligator_position_state'] = df.apply(get_position_state, axis=1)
    df['alligator_jaw_teeth_converging'] = (df['alligator_jaw_teeth_dist'].diff() < 0).astype(int)
    df['alligator_teeth_lips_converging'] = (df['alligator_teeth_lips_dist'].diff() < 0).astype(int)
    df['alligator_jaw_teeth_dist_norm'] = df['alligator_jaw_teeth_dist'] / df['close']
    df['alligator_teeth_lips_dist_norm'] = df['alligator_teeth_lips_dist'] / df['close']
    df['alligator_jaw_lips_dist_norm'] = df['alligator_jaw_lips_dist'] / df['close']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    def get_rsi_state(row):
        rsi = row['rsi']
        if rsi < 20: return 0
        elif rsi < 30: return 1
        elif rsi < 40: return 2
        elif rsi < 60: return 3
        elif rsi < 70: return 4
        elif rsi < 80: return 5
        elif rsi > 80: return 6
        else: return 3
    
    df['rsi_state'] = df.apply(get_rsi_state, axis=1)
    df['rsi_normalized'] = (df['rsi'] - 50) / 50
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    def get_macd_state(row):
        macd, signal, hist = row['macd'], row['macd_signal'], row['macd_histogram']
        if macd > signal and macd > 0:
            return 2
        elif macd < signal and macd < 0:
            return 3
        elif macd > signal:
            return 4
        elif macd < signal:
            return 5
        else:
            return 1
    
    df['macd_state'] = df.apply(get_macd_state, axis=1)
    df['macd_normalized'] = df['macd'] / (abs(df['close']) + 1e-10)
    df['macd_signal_normalized'] = df['macd_signal'] / (abs(df['close']) + 1e-10)
    df['macd_histogram_normalized'] = df['macd_histogram'] / (abs(df['close']) + 1e-10)
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    df['bb_percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    
    def get_bb_state(row):
        price = row['close']
        upper, lower = row['bb_upper'], row['bb_lower']
        if price >= upper: return 2
        elif price <= lower: return 0
        else: return 1
    
    df['bb_state'] = df.apply(get_bb_state, axis=1)
    df['bb_std_normalized'] = df['bb_std'] / (abs(df['close']) + 1e-10)
    df['bb_bandwidth_normalized'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-10)
    
    # Volatility
    df['rolling_volatility'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
    df['volatility_normalized'] = df['rolling_volatility'] / df['close']
    
    # Clean up
    df.drop(['bb_percent_b'], axis=1, errors='ignore', inplace=True)
    return df.dropna()

# === HRP OPTIMIZATION ===
def optimize_weights_hrp(returns_df):
    """Hierarchical Risk Parity optimizer"""
    cov = returns_df.cov()
    corr = returns_df.corr()
    
    if cov.empty or cov.values.sum() == 0:
        return np.full(returns_df.shape[1], 1.0 / returns_df.shape[1])
    
    dist = np.sqrt((1 - corr) / 2)
    np.fill_diagonal(dist.values, 0)
    dist = dist.fillna(0)
    
    try:
        condensed_dist = squareform(dist.values)
        link = linkage(condensed_dist, method='single')
    except:
        return np.full(returns_df.shape[1], 1.0 / returns_df.shape[1])
    
    def get_quasi_diag(link):
        n = link.shape[0] + 1
        def seriation(Z, N, cur_index):
            if cur_index < N:
                return [cur_index]
            else:
                left = int(Z[cur_index - N, 0])
                right = int(Z[cur_index - N, 1])
                return seriation(Z, N, left) + seriation(Z, N, right)
        return seriation(link, n, 2 * n - 2)
    
    def get_cluster_var(cov, c_items):
        cov_slice = cov.iloc[c_items, c_items]
        w = 1 / np.diag(cov_slice)
        w /= w.sum()
        return np.dot(np.dot(w, cov_slice), w)
    
    def get_rec_bipart(cov, sort_ix):
        w = pd.Series(1.0, index=sort_ix)
        c_items = [sort_ix]
        while len(c_items) > 0:
            new_c_items = []
            for cluster in c_items:
                if len(cluster) > 1:
                    mid = len(cluster) // 2
                    c_items0 = cluster[:mid]
                    c_items1 = cluster[mid:]
                    c_var0 = get_cluster_var(cov, c_items0)
                    c_var1 = get_cluster_var(cov, c_items1)
                    alpha = 1 - c_var0 / (c_var0 + c_var1)
                    w[c_items0] *= alpha
                    w[c_items1] *= 1 - alpha
                    new_c_items.append(c_items0)
                    new_c_items.append(c_items1)
            c_items = new_c_items
        return w
    
    sort_ix = get_quasi_diag(link)
    weights_series = get_rec_bipart(cov, sort_ix)
    return weights_series.sort_index().values

# === TCN PREDICTION ===
def predict_portfolio_return(portfolio_df, model):
    """Predict portfolio return using TCN model"""
    try:
        # ✓ CORRECT: Calculate indicators on full data
        portfolio_with_indicators = calculate_indicators(portfolio_df.copy())
        
        # ✓ CORRECT: Check we have enough data (120 days for TCN)
        if len(portfolio_with_indicators) < 121:
            return None
        
        # ✓ CORRECT: Extract LAST 120 days only
        last_120 = portfolio_with_indicators.iloc[-120:]
        
        # EXACT 31 features
        feature_cols = [
            'close', 'high', 'low',
            'alligator_jaw', 'alligator_teeth', 'alligator_lips',
            'alligator_jaw_teeth_dist_norm', 'alligator_teeth_lips_dist_norm', 'alligator_jaw_lips_dist_norm',
            'alligator_formation', 'alligator_position_state',
            'alligator_jaw_teeth_converging', 'alligator_teeth_lips_converging',
            'rsi', 'rsi_state', 'rsi_normalized',
            'macd', 'macd_signal', 'macd_histogram', 'macd_state',
            'macd_normalized', 'macd_signal_normalized', 'macd_histogram_normalized',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_state',
            'bb_std_normalized', 'bb_bandwidth_normalized',
            'rolling_volatility', 'volatility_normalized'
        ]
        
        features = last_120[feature_cols].values
        
        # Normalize
        scaler = MinMaxScaler(feature_range=(0, 1))
        features_scaled = scaler.fit_transform(features)
        
        # Reshape: (1, 120, 31)
        X = features_scaled.reshape(1, 120, 31)
        
        # Predict
        prediction = model.predict(X, verbose=0)[0][0]
        return_pct = (prediction - 1) * 100
        
        return return_pct
    except Exception as e:
        print(f"  TCN Prediction Error: {e}")
        return None

# === RAG ANALYSIS ===
def get_rag_metrics(portfolio_df):
    """Get RAG metrics for risk assessment"""
    try:
        # ✓ CORRECT: Extract actual indicators from portfolio data
        portfolio_with_indicators = calculate_indicators(portfolio_df.copy())
        
        if len(portfolio_with_indicators) < 1:
            return None
        
        # Get latest indicator values
        latest = portfolio_with_indicators.iloc[-1]
        
        indicators = {
            'rsi': latest['rsi'],
            'volatility': latest['rolling_volatility'],
            'macd': latest['macd'],
            'bb_upper': latest['bb_upper'],
            'bb_lower': latest['bb_lower'],
            'close': latest['close']
        }
        
        # Find similar scenarios
        similar_scenarios = find_similar_scenarios(indicators, k=50)
        
        if not similar_scenarios:
            return None
        
        outcomes = [s['target_return_60d'] for s in similar_scenarios]
        mean_ret = np.mean(outcomes)
        std_ret = np.std(outcomes)
        
        # 95% Confidence Interval
        ci_lower = mean_ret - 1.96 * std_ret
        ci_upper = mean_ret + 1.96 * std_ret
        
        return {
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'mean': mean_ret,
            'std': std_ret,
            'median_return': np.median(outcomes),
            'success_rate': (sum(1 for o in outcomes if o > 0) / len(outcomes)) * 100,
            'scenarios_count': len(similar_scenarios)
        }
    except Exception as e:
        print(f"  RAG Error: {e}")
        return None

# === PERFORMANCE REPORTING ===
def generate_backtest_performance_report(daily_portfolio_values, daily_dates, benchmark_df, all_data, final_portfolio_weights, final_portfolio_combo):
    """
    Generate comprehensive performance report for backtest results.
    
    Args:
        daily_portfolio_values: List of daily portfolio values throughout backtest
        daily_dates: List of dates corresponding to portfolio values
        benchmark_df: Benchmark DataFrame with returns
        all_data: Dictionary of all stock data
        final_portfolio_weights: Final portfolio weights
        final_portfolio_combo: Final portfolio combination (stock keys)
    """
    print("\n" + "="*60)
    print("       DETAILED PERFORMANCE REPORT")
    print("="*60)
    
    # Create portfolio DataFrame
    portfolio_df = pd.DataFrame({
        'close': daily_portfolio_values
    }, index=daily_dates)
    portfolio_df['returns'] = portfolio_df['close'].pct_change()
    portfolio_df = portfolio_df.dropna()
    
    # Align with benchmark
    common_index = portfolio_df.index.intersection(benchmark_df.index)
    port_returns = portfolio_df.loc[common_index, 'returns']
    bench_returns = benchmark_df.loc[common_index, 'returns']
    
    if port_returns.empty or bench_returns.empty:
        print("Error: Insufficient overlapping data for report.")
        return
    
    # --- Helper Functions ---
    def get_cagr(returns, years):
        return (1 + returns).prod() ** (1 / years) - 1 if years > 0 else 0
    
    def get_drawdown(returns):
        cum_rets = (1 + returns).cumprod()
        peak = cum_rets.expanding(min_periods=1).max()
        dd = (cum_rets - peak) / peak
        return dd.min()
    
    def get_var_cvar(returns, confidence=0.95):
        sorted_rets = np.sort(returns)
        index = int((1 - confidence) * len(sorted_rets))
        var = abs(sorted_rets[index])
        cvar = abs(sorted_rets[:index].mean())
        return var, cvar
    
    # --- Risk & Performance Ratios ---
    excess_port = port_returns - RISK_FREE_RATE/252
    excess_bench = bench_returns - RISK_FREE_RATE/252
    
    std_port = port_returns.std() * np.sqrt(252)
    std_bench = bench_returns.std() * np.sqrt(252)
    
    sharpe_port = (excess_port.mean() / port_returns.std()) * np.sqrt(252) if port_returns.std() != 0 else 0
    sharpe_bench = (excess_bench.mean() / bench_returns.std()) * np.sqrt(252) if bench_returns.std() != 0 else 0
    
    downside_port = port_returns[port_returns < 0].std() * np.sqrt(252)
    sortino_port = (excess_port.mean() * 252) / downside_port if downside_port != 0 else 0
    
    downside_bench = bench_returns[bench_returns < 0].std() * np.sqrt(252)
    sortino_bench = (excess_bench.mean() * 252) / downside_bench if downside_bench != 0 else 0
    
    active_returns = port_returns - bench_returns
    tracking_error = active_returns.std() * np.sqrt(252)
    information_ratio = (active_returns.mean() * 252) / tracking_error if tracking_error != 0 else 0
    
    print(f"Risk & Performance Ratios")
    print(f"  Sharpe Ratio:            {sharpe_port:.2f}")
    print(f"  Sharpe Ratio (Bench):    {sharpe_bench:.2f}")
    print(f"  Sortino Ratio:           {sortino_port:.2f}")
    print(f"  Sortino Ratio (Bench):   {sortino_bench:.2f}")
    print(f"  Information Ratio:       {information_ratio:.2f}")
    
    # --- Drawdowns ---
    dd_port = get_drawdown(port_returns)
    dd_bench = get_drawdown(bench_returns)
    print(f"\nDrawdowns")
    print(f"  Max Drawdown (Port):     {dd_port*100:.2f}%")
    print(f"  Max Drawdown (Bench):    {dd_bench*100:.2f}%")
    
    # --- Returns ---
    print(f"\nReturns")
    periods = {
        '1-Day': 1, '5-Day': 5, '1-Month': 21, '3-Month': 63, '6-Month': 126
    }
    for name, days in periods.items():
        if len(port_returns) >= days:
            ret = (1 + port_returns.iloc[-days:]).prod() - 1
            print(f"  {name:<23}: {ret*100:.2f}%")
    
    # CAGRs
    total_days = (port_returns.index[-1] - port_returns.index[0]).days
    years = total_days / 365.25
    if years >= 1: print(f"  1Y CAGR:                 {get_cagr(port_returns.iloc[-252:], 1)*100:.2f}%")
    if years >= 3: print(f"  3Y CAGR:                 {get_cagr(port_returns.iloc[-756:], 3)*100:.2f}%")
    if years >= 5: print(f"  5Y CAGR:                 {get_cagr(port_returns.iloc[-1260:], 5)*100:.2f}%")
    
    # --- Benchmark Returns ---
    print(f"\nBenchmark Returns")
    for name, days in periods.items():
        if len(bench_returns) >= days:
            ret = (1 + bench_returns.iloc[-days:]).prod() - 1
            print(f"  Bench {name:<17}: {ret*100:.2f}%")
    
    if years >= 1: print(f"  Bench 1Y CAGR:           {get_cagr(bench_returns.iloc[-252:], 1)*100:.2f}%")
    if years >= 3: print(f"  Bench 3Y CAGR:           {get_cagr(bench_returns.iloc[-756:], 3)*100:.2f}%")
    if years >= 5: print(f"  Bench 5Y CAGR:           {get_cagr(bench_returns.iloc[-1260:], 5)*100:.2f}%")
    
    # --- Volatility & Tracking ---
    print(f"\nVolatility and Tracking")
    print(f"  Std Dev (Portfolio):     {std_port*100:.2f}%")
    print(f"  Std Dev (Benchmark):     {std_bench*100:.2f}%")
    print(f"  Tracking Error:          {tracking_error*100:.2f}%")
    
    # Rolling Volatility (last 20 days annualized)
    if len(port_returns) >= 20:
        rolling_vol = port_returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        print(f"  Rolling Std Dev (20d):   {rolling_vol*100:.2f}%")
    
    # --- Risk Sensitivity ---
    covariance = np.cov(port_returns, bench_returns)[0][1]
    beta = covariance / bench_returns.var() if bench_returns.var() != 0 else 0
    
    # Calculate individual stock betas and weighted beta
    if final_portfolio_weights is not None and final_portfolio_combo is not None:
        stock_betas = []
        for k in final_portfolio_combo:
            if k in all_data:
                s_ret = all_data[k].loc[common_index, 'returns']
                cov_stock = np.cov(s_ret, bench_returns)[0][1]
                beta_stock = cov_stock / bench_returns.var() if bench_returns.var() != 0 else 0
                stock_betas.append(beta_stock)
            else:
                stock_betas.append(0)
        
        weights_array = np.array([final_portfolio_weights[k] for k in final_portfolio_combo])
        weighted_beta = np.sum(np.array(stock_betas) * weights_array)
        
        print(f"\nRisk Sensitivity")
        print(f"  Beta (Portfolio):        {weighted_beta:.2f}")
        print(f"  Beta (General):          {beta:.2f}")
    else:
        print(f"\nRisk Sensitivity")
        print(f"  Beta (Portfolio):        {beta:.2f}")
    
    # --- Alpha and Related Stats ---
    annualized_return_port = (1 + port_returns.mean())**252 - 1
    annualized_return_bench = (1 + bench_returns.mean())**252 - 1
    jensens_alpha = annualized_return_port - (RISK_FREE_RATE + beta * (annualized_return_bench - RISK_FREE_RATE))
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(bench_returns, port_returns)
    
    print(f"\nAlpha and Related Stats")
    print(f"  Jensen's Alpha:          {jensens_alpha*100:.2f}%")
    print(f"  R-squared:               {r_value**2:.2f}")
    
    # Alpha Distribution (Residuals)
    residuals = port_returns - (RISK_FREE_RATE/252 + beta * (bench_returns - RISK_FREE_RATE/252))
    print(f"  Alpha Skewness:          {stats.skew(residuals):.2f}")
    print(f"  Alpha Kurtosis:          {stats.kurtosis(residuals):.2f}")
    
    # Mean Alpha on Stress Days (Benchmark < -2%)
    stress_mask = bench_returns < -0.02
    if stress_mask.any():
        stress_alpha = residuals[stress_mask].mean()
        print(f"  Mean Alpha (Stress):     {stress_alpha*100:.2f}%")
    else:
        print(f"  Mean Alpha (Stress):     N/A")
    
    # --- Value at Risk ---
    var_95, cvar_95 = get_var_cvar(port_returns, 0.95)
    print(f"\nValue at Risk (95%)")
    print(f"  1-Day VaR:               {var_95*100:.2f}%")
    print(f"  1-Day CVaR:              {cvar_95*100:.2f}%")
    print(f"  Annualized VaR:          {var_95 * np.sqrt(252) * 100:.2f}%")
    print(f"  Annualized CVaR:         {cvar_95 * np.sqrt(252) * 100:.2f}%")
    
    # --- Distribution Metrics ---
    print(f"\nDistribution Metrics")
    print(f"  Skewness:                {stats.skew(port_returns):.2f}")
    print(f"  Kurtosis:                {stats.kurtosis(port_returns):.2f}")
    
    # --- Diversification ---
    if final_portfolio_weights is not None and final_portfolio_combo is not None:
        stock_vols = []
        for k in final_portfolio_combo:
            if k in all_data:
                s_ret = all_data[k].loc[common_index, 'returns']
                stock_vols.append(s_ret.std() * np.sqrt(252))
            else:
                stock_vols.append(0)
        
        weighted_avg_vol = np.sum(np.array(stock_vols) * weights_array)
        
        if std_port > 0:
            div_ratio = weighted_avg_vol / std_port
            print(f"  Diversification Ratio:   {div_ratio:.2f}")
        else:
            print(f"  Diversification Ratio:   N/A")
    else:
        print(f"  Diversification Ratio:   N/A")
    
    # --- Correlation Matrix ---
    if final_portfolio_combo is not None:
        print(f"\nCorrelation Matrix (Pairwise Return Correlations)")
        returns_dict = {}
        for k in final_portfolio_combo:
            if k in all_data:
                symbol = STOCK_UNIVERSE[k]
                returns_dict[symbol] = all_data[k].loc[common_index, 'returns']
        
        if returns_dict:
            returns_df_all = pd.DataFrame(returns_dict)
            corr_matrix = returns_df_all.corr()
            print(corr_matrix.round(2))
        else:
            print("  N/A (Insufficient Data)")
    
    print("="*60)

# === PLOTTING ===
def plot_portfolio_forecast(daily_portfolio_values, daily_dates, final_portfolio_combo, model, aligned_data):
    """
    Plot the last 120 days of portfolio performance and predicted next 60 days with RAG boundaries.
    
    Args:
        daily_portfolio_values: List of daily portfolio values
        daily_dates: List of dates
        final_portfolio_combo: Final portfolio combination (stock keys)
        model: TCN model for prediction
        aligned_data: Dictionary of all stock data
    """
    if not daily_portfolio_values or len(daily_portfolio_values) < 120:
        print("Insufficient data for plotting (need at least 120 days)")
        return
    
    print("\nGenerating forecast plot...")
    
    # Get last 120 days of actual data
    last_120_values = daily_portfolio_values[-120:]
    last_120_dates = daily_dates[-120:]
    
    # Create portfolio DataFrame for the last 120 days
    portfolio_df = pd.DataFrame({
        'close': last_120_values
    }, index=last_120_dates)
    
    # Get current date and value
    current_date = last_120_dates[-1]
    current_value = last_120_values[-1]
    
    # Create weighted portfolio series for prediction
    if final_portfolio_combo is not None:
        # Get weights from the final portfolio
        # We need to reconstruct the portfolio series with OHLCV data
        port_series = pd.DataFrame(index=aligned_data[list(final_portfolio_combo)[0]].index)
        port_series['close'] = 0
        port_series['open'] = 0
        port_series['high'] = 0
        port_series['low'] = 0
        port_series['volume'] = 0
        
        # Calculate equal weights for final portfolio
        n_stocks = len(final_portfolio_combo)
        equal_weight = 1.0 / n_stocks
        
        for k in final_portfolio_combo:
            port_series['close'] += aligned_data[k]['close'] * equal_weight
            port_series['open'] += aligned_data[k]['open'] * equal_weight
            port_series['high'] += aligned_data[k]['high'] * equal_weight
            port_series['low'] += aligned_data[k]['low'] * equal_weight
            port_series['volume'] += aligned_data[k]['volume'] * equal_weight
        
        # Get data up to current date
        port_to_date = port_series.loc[:current_date]
        
        # Get TCN prediction
        tcn_pred = predict_portfolio_return(port_to_date, model)
        
        # Get RAG metrics
        rag_res = get_rag_metrics(port_to_date)
        
        if tcn_pred is not None and rag_res is not None:
            # Calculate predicted value
            predicted_return_factor = 1 + (tcn_pred / 100)
            predicted_value = current_value * predicted_return_factor
            
            # Calculate RAG boundaries
            rag_lower_factor = 1 + (rag_res['ci_lower'] / 100)
            rag_upper_factor = 1 + (rag_res['ci_upper'] / 100)
            rag_lower_value = current_value * rag_lower_factor
            rag_upper_value = current_value * rag_upper_factor
            
            # Generate future dates (60 trading days)
            future_dates = pd.bdate_range(start=current_date + timedelta(days=1), periods=60)
            
            # Create linear interpolation for forecast (simple visualization)
            forecast_dates = [current_date] + list(future_dates)
            forecast_values = np.linspace(current_value, predicted_value, 61)
            forecast_lower = np.linspace(current_value, rag_lower_value, 61)
            forecast_upper = np.linspace(current_value, rag_upper_value, 61)
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(14, 7))
            
            # Plot historical performance (last 120 days)
            ax.plot(last_120_dates, last_120_values, 
                   color='#2E86AB', linewidth=2, label='Historical Performance (120 days)', zorder=3)
            
            # Plot predicted performance (next 60 days)
            ax.plot(forecast_dates, forecast_values, 
                   color='#A23B72', linewidth=2, linestyle='--', label='TCN Prediction (60 days)', zorder=2)
            
            # Plot RAG confidence interval
            ax.fill_between(forecast_dates, forecast_lower, forecast_upper, 
                           color='#F18F01', alpha=0.3, label='RAG 95% Confidence Interval', zorder=1)
            
            # Add vertical line at current date
            ax.axvline(x=current_date, color='gray', linestyle=':', linewidth=1.5, 
                      label='Current Date', zorder=0)
            
            # Formatting
            ax.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax.set_ylabel('Portfolio Value (RS)', fontsize=12, fontweight='bold')
            ax.set_title('Portfolio Performance & 60-Day Forecast with RAG Boundaries', 
                        fontsize=14, fontweight='bold', pad=20)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            plt.xticks(rotation=45, ha='right')
            
            # Add grid
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            # Add legend
            ax.legend(loc='best', fontsize=10, framealpha=0.9)
            
            # Add annotations
            ax.annotate(f'Current: RS{current_value:,.0f}', 
                       xy=(current_date, current_value), 
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                       fontsize=9, fontweight='bold')
            
            ax.annotate(f'Predicted: RS{predicted_value:,.0f}\n({tcn_pred:+.2f}%)', 
                       xy=(forecast_dates[-1], predicted_value), 
                       xytext=(-80, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
                       fontsize=9, fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='black', lw=1))
            
            # Format y-axis with comma separator
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'RS{x:,.0f}'))
            
            # Tight layout
            plt.tight_layout()
            
            # Save plot
            plot_filename = 'portfolio_forecast_plot.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"[OK] Forecast plot saved as: {plot_filename}")
            
            # Show plot
            plt.show()
            
            # Print forecast summary
            print("\n" + "="*60)
            print("       FORECAST SUMMARY")
            print("="*60)
            print(f"Current Date:              {current_date.strftime('%Y-%m-%d')}")
            print(f"Current Portfolio Value:   RS{current_value:,.2f}")
            print(f"Forecast Period:           60 Trading Days")
            print(f"Forecast End Date:         {future_dates[-1].strftime('%Y-%m-%d')}")
            print(f"\nTCN Prediction:")
            print(f"  Predicted Return:        {tcn_pred:+.2f}%")
            print(f"  Predicted Value:         RS{predicted_value:,.2f}")
            print(f"\nRAG Analysis (95% CI):")
            print(f"  Lower Bound:             RS{rag_lower_value:,.2f} ({rag_res['ci_lower']:+.2f}%)")
            print(f"  Upper Bound:             RS{rag_upper_value:,.2f} ({rag_res['ci_upper']:+.2f}%)")
            print(f"  Historical Mean:         {rag_res['mean']:+.2f}%")
            print(f"  Historical Median:       {rag_res['median_return']:+.2f}%")
            print(f"  Success Rate:            {rag_res['success_rate']:.1f}%")
            print(f"  Scenarios Analyzed:      {rag_res['scenarios_count']}")
            print("="*60)
        else:
            print("Unable to generate forecast: TCN or RAG prediction failed")
    else:
        print("Unable to generate forecast: No final portfolio available")

# === MAIN BACKTEST ===
def main():
    print("=" * 70)
    print(" ADVANCED TCN PORTFOLIO BACKTEST WITH RAG - FIXED VERSION")
    print("=" * 70)
    
    # Initialize
    print("\nInitializing RAG system...")
    from rag_prediction_simple import load_rag_system
    if not load_rag_system():
        print("ERROR: RAG system failed to load. Exiting.")
        return
    
    init_db()
    
    # Load TCN Model
    print("\nLoading TCN Model...")
    try:
        model = load_model("generalized_lstm_model_improved_v2.keras")
        print("TCN Model loaded successfully.")
    except:
        print("ERROR: Could not load TCN model.")
        return
    
    # Fetch Data
    print(f"\nFetching data for {len(STOCK_UNIVERSE)} stocks...")
    all_data = {}
    for key, symbol in STOCK_UNIVERSE.items():
        print(f"  Fetching {symbol}...", end="")
        df = fetch_historical_data(key, START_DATE, END_DATE)
        if not df.empty:
            df = calculate_indicators(df)
            all_data[key] = df
            print(f" OK ({len(df)} rows)")
        else:
            print(" ✗")
    
    if len(all_data) < 5:
        print("ERROR: Not enough stocks fetched.")
        return
    
    # Fetch Benchmark
    print("\nFetching Benchmark...")
    benchmark_df = fetch_historical_data(BENCHMARK_INSTRUMENT_KEY, START_DATE, END_DATE)
    
    # Align Data
    print("\nAligning data...")
    common_index = sorted(list(set.intersection(*[set(df.index) for df in all_data.values()])))
    
    if benchmark_df is not None and not benchmark_df.empty:
        common_index = sorted(list(set(common_index).intersection(set(benchmark_df.index))))
    
    if not common_index:
        print("ERROR: No common dates found.")
        return
    
    print(f"  Common Period: {common_index[0].strftime('%Y-%m-%d')} to {common_index[-1].strftime('%Y-%m-%d')}")
    print(f"  Total Trading Days: {len(common_index)}")
    
    aligned_data = {k: v.loc[common_index] for k, v in all_data.items()}
    
    # Generate Combinations
    stock_keys = list(aligned_data.keys())
    combos = list(itertools.combinations(stock_keys, 5))
    print(f"\nGenerated {len(combos)} unique portfolios (C(10,5) = {len(combos)}).")
    
    # ✓ CORRECT: Start after minimum required days
    min_idx = MIN_REQUIRED_DAYS
    if min_idx >= len(common_index):
        print(f"ERROR: Not enough data ({len(common_index)} days < {MIN_REQUIRED_DAYS} minimum)")
        return
    
    current_date = common_index[min_idx]
    end_date = common_index[-1]
    
    print(f"\nStarting backtest from: {current_date.strftime('%Y-%m-%d')}")
    
    capital = INITIAL_CAPITAL
    cycle_count = 0
    
    # Track daily portfolio values for performance report
    daily_portfolio_values = []
    daily_dates = []
    final_portfolio_weights = None
    final_portfolio_combo = None
    
    while current_date < end_date - timedelta(days=60):
        cycle_count += 1
        
        print(f"\n{'='*70}")
        print(f"Cycle {cycle_count}: Rebalance Date: {current_date.strftime('%Y-%m-%d')}")
        print(f"Current Capital: RS{capital:,.2f}")
        print(f"{'='*70}")
        
        best_portfolio = None
        best_rr = -np.inf
        
        print(f"Analyzing {len(combos)} portfolios...")
        
        for combo in combos:
            combo_data = {k: aligned_data[k] for k in combo}
            
            # ✓ CORRECT: Use last 120 days before current_date for HRP weights
            current_idx = list(common_index).index(current_date)
            lookback_start_idx = max(0, current_idx - LOOKBACK_WINDOW)
            lookback_end_idx = current_idx
            
            lookback_index = common_index[lookback_start_idx:lookback_end_idx+1]
            
            returns_for_opt = pd.DataFrame({
                k: combo_data[k].loc[lookback_index, 'returns']
                for k in combo
            }).dropna()
            
            if len(returns_for_opt) < 10:
                continue
            
            weights_array = optimize_weights_hrp(returns_for_opt)
            weights = {k: weights_array[i] for i, k in enumerate(combo)}
            
            # Create portfolio series for TCN
            port_series = pd.DataFrame(index=aligned_data[combo[0]].index)
            port_series['close'] = 0
            port_series['open'] = 0
            port_series['high'] = 0
            port_series['low'] = 0
            port_series['volume'] = 0
            
            for k, w in weights.items():
                port_series['close'] += combo_data[k]['close'] * w
                port_series['open'] += combo_data[k]['open'] * w
                port_series['high'] += combo_data[k]['high'] * w
                port_series['low'] += combo_data[k]['low'] * w
                port_series['volume'] += combo_data[k]['volume'] * w
            
            # ✓ CORRECT: Use data UP TO current_date for prediction
            port_to_date = port_series.loc[:current_date]
            
            # TCN Prediction
            tcn_pred = predict_portfolio_return(port_to_date, model)
            if tcn_pred is None:
                continue
            
            # RAG Analysis
            rag_res = get_rag_metrics(port_to_date)
            if not rag_res:
                continue
            
            ci_lower = rag_res['ci_lower']
            ci_upper = rag_res['ci_upper']
            
            # ✓ CORRECT: Risk/Reward Calculation
            risk = abs(ci_lower)
            if risk < 0.1:  # Avoid division by very small numbers
                risk = 0.1
            
            rr_ratio = tcn_pred / risk
            
            if rr_ratio > best_rr:
                best_rr = rr_ratio
                best_portfolio = {
                    'combo': combo,
                    'weights': weights,
                    'tcn_pred': tcn_pred,
                    'rag_ci_lower': ci_lower,
                    'rag_ci_upper': ci_upper,
                    'rr': rr_ratio
                }
        
        # Execute Best Portfolio
        if best_portfolio:
            symbols = [STOCK_UNIVERSE[k] for k in best_portfolio['combo']]
            print(f"\nSelected Portfolio: {symbols}")
            print(f"  TCN Prediction: {best_portfolio['tcn_pred']:.2f}%")
            print(f"  RAG 95% CI: [{best_portfolio['rag_ci_lower']:.2f}%, {best_portfolio['rag_ci_upper']:.2f}%]")
            print(f"  Risk/Reward Ratio: {best_portfolio['rr']:.2f}")
            
            # Simulate next 60 days
            current_idx = list(common_index).index(current_date)
            holding_period = common_index[current_idx+1 : min(current_idx+61, len(common_index))]
            
            entry_val = capital
            entry_price_date = current_date
            exit_reason = "Completed 60 days"
            exit_date = holding_period[-1] if holding_period else current_date
            
            for day in holding_period:
                # Calculate portfolio value
                day_val = 0
                for k, w in best_portfolio['weights'].items():
                    entry_price = aligned_data[k].loc[entry_price_date, 'close']
                    current_price = aligned_data[k].loc[day, 'close']
                    ret = current_price / entry_price
                    day_val += (entry_val * w) * ret
                
                # Track daily values for performance report
                daily_portfolio_values.append(day_val)
                daily_dates.append(day)
                
                # Update capital and exit date (no early exit - hold full 60 days)
                capital = day_val
                exit_date = day
            
            actual_return_pct = (capital / entry_val - 1) * 100
            print(f"  {exit_reason}. Final Return: {actual_return_pct:.2f}%")
            
            # Store final portfolio info for performance report
            final_portfolio_weights = best_portfolio['weights']
            final_portfolio_combo = best_portfolio['combo']
            
            # Store result
            store_backtest_result(
                entry_price_date.strftime('%Y-%m-%d'),
                best_portfolio['combo'],
                best_portfolio['tcn_pred'],
                best_portfolio['rag_ci_lower'],
                best_portfolio['rag_ci_upper'],
                best_portfolio['rr'],
                actual_return_pct,
                exit_date.strftime('%Y-%m-%d'),
                exit_reason,
                entry_val,
                capital
            )
        else:
            print("  No valid portfolio found in this cycle.")
        
        # Move to next rebalance cycle
        current_idx = list(common_index).index(current_date)
        next_idx = min(current_idx + 60, len(common_index) - 1)
        current_date = common_index[next_idx]
    
    print(f"\n{'='*70}")
    print(f"Backtest Complete. Total Cycles: {cycle_count}")
    print(f"Final Capital: RS{capital:,.2f}")
    print(f"Total Return: {((capital / INITIAL_CAPITAL) - 1) * 100:.2f}%")
    print(f"Results saved to: {DATABASE_FILE}")
    print(f"{'='*70}")
    
    # Generate comprehensive performance report
    if daily_portfolio_values and benchmark_df is not None and not benchmark_df.empty:
        print("\n\nGenerating comprehensive performance report...")
        generate_backtest_performance_report(
            daily_portfolio_values,
            daily_dates,
            benchmark_df,
            aligned_data,
            final_portfolio_weights,
            final_portfolio_combo
        )
    
    # Generate forecast plot
    if daily_portfolio_values and final_portfolio_combo is not None:
        plot_portfolio_forecast(
            daily_portfolio_values,
            daily_dates,
            final_portfolio_combo,
            model,
            aligned_data
        )
    else:
        print("\nSkipping forecast plot: Insufficient data or no portfolio selected")


if __name__ == "__main__":
    main()
