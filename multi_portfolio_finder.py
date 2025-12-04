"""
Multi-Portfolio Finder with TCN + RAG Validation
=================================================
This script searches for the best 10 portfolios from a universe of 33 stocks.
It uses TCN predictions and RAG historical analysis to find portfolios with
consistent, moderate positive returns (not extreme gains, but reliable).

LLM is called ONLY ONCE at the end to rank and explain the top 10 portfolios.
"""

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
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import itertools
import random

# RAG-related imports
import google.generativeai as genai
from rag_prediction_simple import load_rag_system, find_similar_scenarios

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# === CONFIGURATION ===
ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI3M0FIOUMiLCJqdGkiOiI2OTJiMGMxZGIzNGEzMTEzNGI4Njg3ZGQiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6dHJ1ZSwiaWF0IjoxNzY0NDI4ODI5LCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3NjQ0NTM2MDB9.FyfmIVYRBjw8p51FnR532XWaSaBSHOmptXy3Es_22JA"
BASE_URL = "https://api.upstox.com/v3/historical-candle"
START_DATE = "2018-09-03"
END_DATE = "2025-12-04"
RISK_FREE_RATE = 0.065
TRANSACTION_COST = 0.001
INITIAL_CAPITAL = 100000.0
BENCHMARK_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"

# --- API Keys ---
GEMINI_API_KEY = ""

# --- Search Parameters ---
MAX_PORTFOLIOS_TO_FIND = 10      # Stop when we find this many good portfolios
PORTFOLIO_SIZE = 5               # Number of stocks per portfolio

# --- Selection Criteria ---
# We want moderate positive returns, not extreme ones
MIN_TCN_RETURN = 1.0      # Minimum TCN predicted return (%)
MAX_TCN_RETURN = 15.0     # Maximum TCN predicted return (%) - avoid unrealistic predictions
MIN_RAG_SUCCESS_RATE = 55.0  # Minimum historical success rate (%)
MIN_RAG_MEAN_RETURN = 0.5    # Minimum RAG historical mean return (%)
MAX_RAG_STD = 15.0           # Maximum RAG standard deviation (%) - prefer consistent returns

# === FULL STOCK UNIVERSE (33 stocks) ===
STOCK_UNIVERSE = {
    "NSE_EQ|INE732I01013": "ANGELONE",
    "NSE_EQ|INE296A01032": "BAJFINANCE",
    "NSE_EQ|INE044A01036": "SUNPHARMA",
    "NSE_EQ|INE674K01013": "ABCAPITAL",
    "NSE_EQ|INE028A01039": "BANKBARODA",
    "NSE_EQ|INE692A01016": "UNIONBANK",
    "NSE_EQ|INE476A01022": "CANBK",
    "NSE_EQ|INE372C01037": "PRECWIRE",
    "NSE_EQ|INE918Z01012": "KAYNES",
    "NSE_EQ|INE474Q01031": "MEDANTA",
    "NSE_EQ|INE034A01011": "ARVIND",
    "NSE_EQ|INE560A01023": "INDIAGLYCO",
    "NSE_EQ|INE029A01011": "BPCL",
    "NSE_EQ|INE118H01025": "BSE",
    "NSE_EQ|INE926K01017": "AHLEAST",
    "NSE_EQ|INE376G01013": "BIOCON",
    "NSE_EQ|INE849A01020": "TRENT",
    "NSE_EQ|INE364U01010": "IRFC",
    "NSE_EQ|INE133E01013": "TI",
    "NSE_EQ|INE397D01024": "BHARTIARTL",
    "NSE_EQ|INE931S01010": "ADANIENSOL",
    "NSE_EQ|INE199P01028": "GCSL",
    "NSE_EQ|INE075I01017": "HCG",
    "NSE_EQ|INE0L2Y01011": "ARHAM",
    "NSE_EQ|INE591G01025": "COFORGE",
    "NSE_EQ|INE262H01021": "PERSISTENT",
    "NSE_EQ|INE009A01021": "INFY",
    "NSE_EQ|INE062A01020": "SBIN",
    "NSE_EQ|INE237A01028": "KOTAKBANK",
    "NSE_EQ|INE238A01034": "AXISBANK",
    "NSE_EQ|INE030A01027": "HINDUNILVR",
    "NSE_EQ|INE280A01028": "TITAN",
    "NSE_EQ|INE917I01010": "BAJAJ-AUTO"
}

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
            response.raise_for_status()
            data = response.json().get("data", {}).get("candles", [])
            all_data.extend(data)
            time.sleep(0.1)
        except requests.exceptions.RequestException as e:
            continue

    if all_data:
        df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df.index = df.index.tz_localize(None)
        df.sort_index(inplace=True)
        df["returns"] = df["close"].pct_change()
        return df.dropna()
    return pd.DataFrame()

# === TECHNICAL INDICATORS ===
def calculate_indicators(df):
    """Calculate all technical indicators for TCN model"""
    df = df.copy()
    median_price = (df['high'] + df['low']) / 2
    
    def smma(series, window):
        return series.ewm(alpha=1/window, adjust=False).mean()

    # Alligator Indicator
    df['alligator_jaw'] = smma(median_price, 13).shift(8)
    df['alligator_teeth'] = smma(median_price, 8).shift(5)
    df['alligator_lips'] = smma(median_price, 5).shift(3)
    
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
            if jaw > teeth and teeth > lips: return 0
            elif jaw > teeth and teeth < lips: return 1
            elif jaw < teeth and teeth > lips: return 2
            elif jaw < teeth and teeth < lips: return 3
        return 0

    df['alligator_formation'] = df.apply(get_formation_code, axis=1)

    def get_position_state(row):
        price = row['close']
        jaw_above = row['alligator_jaw'] > price
        teeth_above = row['alligator_teeth'] > price
        lips_above = row['alligator_lips'] > price
        above_count = sum([jaw_above, teeth_above, lips_above])
        if above_count == 3: return 0
        elif above_count == 0: return 1
        elif above_count == 1 and jaw_above: return 3
        elif above_count == 1 and not jaw_above: return 4
        else: return 2

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
        macd, signal = row['macd'], row['macd_signal']
        if macd > 0 and macd > signal: return 2
        elif macd < 0 and macd < signal: return 3
        elif macd > signal: return 4
        elif macd < signal: return 5
        else: return 1

    df['macd_state'] = df.apply(get_macd_state, axis=1)
    df['macd_normalized'] = df['macd'] / abs(df['close'])
    df['macd_signal_normalized'] = df['macd_signal'] / abs(df['close'])
    df['macd_histogram_normalized'] = df['macd_histogram'] / abs(df['close'])

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)

    def get_bb_state(row):
        price = row['close']
        upper, lower = row['bb_upper'], row['bb_lower']
        if price <= lower: return 0
        elif price >= upper: return 2
        else: return 1

    df['bb_state'] = df.apply(get_bb_state, axis=1)
    df['bb_std_normalized'] = df['bb_std'] / abs(df['close'])
    df['bb_bandwidth_normalized'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

    # Volatility
    df['rolling_volatility'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
    df['volatility_normalized'] = df['rolling_volatility'] / df['close'].rolling(window=252).mean()

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
    """Predict 60-day return using TCN model"""
    try:
        if len(portfolio_df) < 121:
            return None
        
        last_120 = portfolio_df.iloc[-120:]
        
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
        scaler = MinMaxScaler(feature_range=(0, 1))
        features_scaled = scaler.fit_transform(features)
        X = features_scaled.reshape(1, 120, 31)
        
        prediction = model.predict(X, verbose=0)[0][0]
        return_pct = (prediction - 1) * 100
        
        return return_pct
    except Exception as e:
        return None

# === RAG ANALYSIS ===
def get_rag_metrics(portfolio_df):
    """Get RAG metrics for risk assessment"""
    try:
        if len(portfolio_df) < 1:
            return None
        
        latest = portfolio_df.iloc[-1]
        
        indicators = {
            'rsi': latest['rsi'],
            'volatility': latest['rolling_volatility'],
            'macd': latest['macd'],
            'alligator_jaw': latest['alligator_jaw'],
            'alligator_teeth': latest['alligator_teeth'],
            'alligator_lips': latest['alligator_lips']
        }
        
        similar_scenarios = find_similar_scenarios(indicators, k=50)
        
        if not similar_scenarios:
            return None
        
        outcomes = [s['target_return_60d'] for s in similar_scenarios]
        mean_ret = np.mean(outcomes) * 100
        std_ret = np.std(outcomes) * 100
        
        ci_lower = mean_ret - 1.96 * std_ret
        ci_upper = mean_ret + 1.96 * std_ret
        
        positive_count = sum(1 for o in outcomes if o > 0)
        success_rate = positive_count / len(outcomes) * 100
        
        return {
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'mean': mean_ret,
            'std': std_ret,
            'median_return': np.median(outcomes) * 100,
            'success_rate': success_rate,
            'scenarios_count': len(similar_scenarios),
            'min_return': np.min(outcomes) * 100,
            'max_return': np.max(outcomes) * 100
        }
    except Exception as e:
        return None

# === FINANCIAL METRICS ===
def calculate_sharpe_ratio(returns, risk_free_rate):
    if returns.empty or np.std(returns) == 0: return 0
    excess_returns = returns - (risk_free_rate / 252)
    return (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252)

def calculate_sortino_ratio(returns, risk_free_rate):
    if returns.empty: return 0
    excess_returns = returns - (risk_free_rate / 252)
    downside_returns = excess_returns[excess_returns < 0]
    downside_deviation = np.std(downside_returns)
    if downside_deviation == 0: return 0
    return (np.mean(excess_returns) / downside_deviation) * np.sqrt(252)

def calculate_max_drawdown(returns):
    if returns.empty: return 0
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

# === PORTFOLIO EVALUATION ===
def evaluate_portfolio(combo, aligned_data, model):
    """Evaluate a single portfolio combination"""
    try:
        symbols = [STOCK_UNIVERSE[k] for k in combo]
        
        # Get returns for HRP optimization (last 120 days)
        returns_df = pd.DataFrame({
            k: aligned_data[k]['returns'] for k in combo
        }).dropna()
        
        if len(returns_df) < 120:
            return None
        
        # Calculate HRP weights
        weights = optimize_weights_hrp(returns_df.iloc[-120:])
        
        # Create weighted portfolio series
        port_series = pd.DataFrame(index=aligned_data[combo[0]].index)
        port_series['close'] = sum(aligned_data[k]['close'] * w for k, w in zip(combo, weights))
        port_series['high'] = sum(aligned_data[k]['high'] * w for k, w in zip(combo, weights))
        port_series['low'] = sum(aligned_data[k]['low'] * w for k, w in zip(combo, weights))
        port_series['open'] = sum(aligned_data[k]['open'] * w for k, w in zip(combo, weights))
        port_series['volume'] = sum(aligned_data[k]['volume'] * w for k, w in zip(combo, weights))
        port_series['returns'] = port_series['close'].pct_change()
        port_series = port_series.dropna()
        
        # Calculate indicators
        port_with_indicators = calculate_indicators(port_series)
        
        if len(port_with_indicators) < 121:
            return None
        
        # Get TCN Prediction
        tcn_pred = predict_portfolio_return(port_with_indicators, model)
        if tcn_pred is None:
            return None
        
        # Get RAG Metrics
        rag_metrics = get_rag_metrics(port_with_indicators)
        if rag_metrics is None:
            return None
        
        # Calculate additional metrics
        port_returns = port_series['returns'].iloc[-252:]  # Last 1 year
        sharpe = calculate_sharpe_ratio(port_returns, RISK_FREE_RATE)
        sortino = calculate_sortino_ratio(port_returns, RISK_FREE_RATE)
        max_dd = calculate_max_drawdown(port_returns)
        volatility = port_returns.std() * np.sqrt(252) * 100
        
        # Current value
        current_value = port_series['close'].iloc[-1]
        
        result = {
            'combo': combo,
            'symbols': symbols,
            'weights': {STOCK_UNIVERSE[k]: w for k, w in zip(combo, weights)},
            'tcn_prediction': tcn_pred,
            'rag_mean': rag_metrics['mean'],
            'rag_std': rag_metrics['std'],
            'rag_ci_lower': rag_metrics['ci_lower'],
            'rag_ci_upper': rag_metrics['ci_upper'],
            'rag_success_rate': rag_metrics['success_rate'],
            'rag_median': rag_metrics['median_return'],
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd * 100,
            'volatility': volatility,
            'current_value': current_value,
            'scenarios_count': rag_metrics['scenarios_count']
        }
        
        return result
        
    except Exception as e:
        return None

# Global benchmark metrics (will be calculated during runtime)
BENCHMARK_SHARPE = None
BENCHMARK_SORTINO = None

def is_good_portfolio(result):
    """Check if portfolio meets our selection criteria for consistent moderate returns"""
    global BENCHMARK_SHARPE, BENCHMARK_SORTINO
    
    if result is None:
        return False
    
    tcn = result['tcn_prediction']
    rag_mean = result['rag_mean']
    rag_std = result['rag_std']
    success_rate = result['rag_success_rate']
    sharpe = result['sharpe_ratio']
    sortino = result['sortino_ratio']
    
    # Check criteria for consistent positive returns
    if tcn < MIN_TCN_RETURN or tcn > MAX_TCN_RETURN:
        return False
    if success_rate < MIN_RAG_SUCCESS_RATE:
        return False
    if rag_mean < MIN_RAG_MEAN_RETURN:
        return False
    if rag_std > MAX_RAG_STD:
        return False
    
    # NEW: Sharpe and Sortino must be greater than benchmark
    if BENCHMARK_SHARPE is not None and sharpe <= BENCHMARK_SHARPE:
        return False
    if BENCHMARK_SORTINO is not None and sortino <= BENCHMARK_SORTINO:
        return False
    
    # Additional: TCN and RAG should be aligned (both positive)
    if tcn > 0 and rag_mean > 0:
        return True
    
    return False

# === LLM RANKING ===
def rank_portfolios_with_llm(valid_portfolios):
    """Use LLM ONCE to rank and explain the top portfolios"""
    print("\n" + "="*70)
    print("       AI PORTFOLIO RANKING (Gemini)")
    print("="*70)
    
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        model_options = [
            'gemini-2.0-flash-exp',
            'gemini-1.5-flash',
        ]
        
        gemini_model = None
        for model_name in model_options:
            try:
                gemini_model = genai.GenerativeModel(model_name)
                print(f"Using model: {model_name}")
                break
            except:
                continue
        
        if not gemini_model:
            print("LLM not available. Using score-based ranking.")
            return score_based_ranking(valid_portfolios)
        
        # Build prompt with all portfolio data
        portfolio_details = ""
        for i, p in enumerate(valid_portfolios, 1):
            portfolio_details += f"""
Portfolio {i}: {', '.join(p['symbols'])}
  - TCN Prediction (60d): {p['tcn_prediction']:+.2f}%
  - RAG Historical Mean: {p['rag_mean']:+.2f}%
  - RAG Historical Median: {p['rag_median']:+.2f}%
  - RAG Std Deviation: {p['rag_std']:.2f}%
  - RAG 95% CI: [{p['rag_ci_lower']:.2f}%, {p['rag_ci_upper']:.2f}%]
  - Success Rate: {p['rag_success_rate']:.1f}%
  - Sharpe Ratio: {p['sharpe_ratio']:.2f}
  - Sortino Ratio: {p['sortino_ratio']:.2f}
  - Max Drawdown: {p['max_drawdown']:.2f}%
  - Volatility: {p['volatility']:.2f}%
  - Weights: {p['weights']}
"""
        
        prompt = f"""You are a quantitative portfolio analyst. Rank the following {len(valid_portfolios)} portfolios from BEST to WORST for the next 60 trading days.

IMPORTANT CRITERIA FOR RANKING:
1. We want CONSISTENT, MODERATE positive returns - NOT extreme high returns
2. High success rate (historical probability of positive outcomes) is VERY important
3. Lower standard deviation (less volatility) is preferred
4. TCN and RAG predictions should be ALIGNED (both positive, similar magnitude)
5. Good Sharpe and Sortino ratios indicate better risk-adjusted returns
6. Lower max drawdown is preferred

PORTFOLIO DATA:
{portfolio_details}

YOUR TASK:
1. Rank all {len(valid_portfolios)} portfolios from 1 (BEST) to {len(valid_portfolios)} (WORST)
2. For each portfolio, explain WHY it got that ranking in 2-3 sentences
3. Focus on CONSISTENCY and RELIABILITY, not maximum returns

OUTPUT FORMAT (use plain text, no markdown):
RANK 1: [Portfolio stocks]
Reason: [2-3 sentence explanation why this is the best choice]

RANK 2: [Portfolio stocks]
Reason: [explanation]

... and so on for all portfolios.

END with:
SUMMARY: [One paragraph summarizing the key factors that made the top 3 portfolios stand out]
"""
        
        response = gemini_model.generate_content(prompt)
        clean_text = response.text.replace('**', '').replace('* ', '- ')
        print(clean_text)
        
        return clean_text
        
    except Exception as e:
        print(f"LLM Error: {e}")
        print("Using score-based ranking instead.")
        return score_based_ranking(valid_portfolios)

def score_based_ranking(portfolios):
    """Fallback: Score-based ranking if LLM fails"""
    print("\n--- SCORE-BASED RANKING ---")
    
    for p in portfolios:
        # Calculate composite score
        # Higher success rate = better
        # Lower std = better (consistency)
        # Moderate positive TCN = better
        # Good Sharpe = better
        
        success_score = p['rag_success_rate'] / 100  # 0-1
        consistency_score = max(0, 1 - p['rag_std'] / 20)  # Lower std = higher score
        return_score = min(p['tcn_prediction'] / 10, 1) if p['tcn_prediction'] > 0 else 0
        sharpe_score = min(max(p['sharpe_ratio'], 0) / 2, 1)  # Cap at 2
        
        p['composite_score'] = (
            success_score * 0.35 +
            consistency_score * 0.25 +
            return_score * 0.20 +
            sharpe_score * 0.20
        )
    
    # Sort by composite score
    ranked = sorted(portfolios, key=lambda x: x['composite_score'], reverse=True)
    
    for i, p in enumerate(ranked, 1):
        print(f"\nRANK {i}: {', '.join(p['symbols'])}")
        print(f"  Composite Score: {p['composite_score']:.3f}")
        print(f"  TCN: {p['tcn_prediction']:+.2f}%, RAG Mean: {p['rag_mean']:+.2f}%")
        print(f"  Success Rate: {p['rag_success_rate']:.1f}%, Std: {p['rag_std']:.2f}%")
    
    return ranked

# === DISPLAY FUNCTIONS ===
def display_portfolio_details(p, rank):
    """Display full details for a portfolio"""
    print(f"\n{'='*70}")
    print(f"  RANK #{rank}: {', '.join(p['symbols'])}")
    print(f"{'='*70}")
    
    print(f"\n--- PORTFOLIO WEIGHTS (HRP Optimized) ---")
    for symbol, weight in p['weights'].items():
        print(f"  {symbol:<15}: {weight:.4f} ({weight*100:.2f}%)")
    
    print(f"\n--- TCN PREDICTION (Next 60 Days) ---")
    print(f"  Predicted Return:        {p['tcn_prediction']:+.2f}%")
    
    print(f"\n--- RAG HISTORICAL ANALYSIS ---")
    print(f"  Similar Scenarios Found: {p['scenarios_count']}")
    print(f"  Historical Mean Return:  {p['rag_mean']:+.2f}%")
    print(f"  Historical Median:       {p['rag_median']:+.2f}%")
    print(f"  Standard Deviation:      {p['rag_std']:.2f}%")
    print(f"  95% Confidence Interval: [{p['rag_ci_lower']:.2f}%, {p['rag_ci_upper']:.2f}%]")
    print(f"  Success Rate:            {p['rag_success_rate']:.1f}%")
    
    print(f"\n--- RISK METRICS (Last 1 Year) vs BENCHMARK ---")
    sharpe_diff = p['sharpe_ratio'] - BENCHMARK_SHARPE if BENCHMARK_SHARPE else 0
    sortino_diff = p['sortino_ratio'] - BENCHMARK_SORTINO if BENCHMARK_SORTINO else 0
    print(f"  Sharpe Ratio:            {p['sharpe_ratio']:.2f} (Benchmark: {BENCHMARK_SHARPE:.2f}, +{sharpe_diff:.2f})")
    print(f"  Sortino Ratio:           {p['sortino_ratio']:.2f} (Benchmark: {BENCHMARK_SORTINO:.2f}, +{sortino_diff:.2f})")
    print(f"  Max Drawdown:            {p['max_drawdown']:.2f}%")
    print(f"  Annualized Volatility:   {p['volatility']:.2f}%")
    
    print(f"\n--- ALIGNMENT CHECK ---")
    if abs(p['tcn_prediction'] - p['rag_mean']) < p['rag_std']:
        alignment = "ALIGNED - TCN prediction is within 1 std of historical mean"
    elif abs(p['tcn_prediction'] - p['rag_mean']) < 2 * p['rag_std']:
        alignment = "MODERATE - TCN prediction is within 2 std of historical mean"
    else:
        alignment = "DIVERGENT - TCN and RAG predictions differ significantly"
    print(f"  {alignment}")

def main():
    global BENCHMARK_SHARPE, BENCHMARK_SORTINO
    
    print("="*70)
    print("  MULTI-PORTFOLIO FINDER WITH TCN + RAG VALIDATION")
    print("="*70)
    print(f"\nSearching for {MAX_PORTFOLIOS_TO_FIND} best portfolios from {len(STOCK_UNIVERSE)} stocks...")
    print("Will search ALL combinations until 10 valid portfolios are found!")
    print(f"\nSelection Criteria:")
    print(f"  - TCN Return: {MIN_TCN_RETURN}% to {MAX_TCN_RETURN}%")
    print(f"  - RAG Success Rate: >= {MIN_RAG_SUCCESS_RATE}%")
    print(f"  - RAG Mean Return: >= {MIN_RAG_MEAN_RETURN}%")
    print(f"  - RAG Std Dev: <= {MAX_RAG_STD}%")
    print(f"  - Sharpe Ratio: > Benchmark (Nifty 50)")
    print(f"  - Sortino Ratio: > Benchmark (Nifty 50)")
    
    # Initialize RAG
    print("\nInitializing RAG system...")
    if not load_rag_system():
        print("ERROR: RAG system failed to load.")
        return
    
    # Load TCN Model
    print("Loading TCN model...")
    try:
        model = load_model("generalized_lstm_model_improved.keras")
        print("Model loaded successfully.")
    except:
        try:
            model = load_model("generalized_lstm_model_improved_v2.keras")
            print("Model v2 loaded successfully.")
        except:
            print("ERROR: Could not load TCN model.")
            return
    
    # Fetch all stock data
    print(f"\nFetching data for {len(STOCK_UNIVERSE)} stocks...")
    all_data = {}
    for key, symbol in STOCK_UNIVERSE.items():
        print(f"  Fetching {symbol}...", end="")
        df = fetch_historical_data(key, START_DATE, END_DATE)
        if not df.empty and len(df) > 300:
            all_data[key] = df
            print(f" OK ({len(df)} rows)")
        else:
            print(f" SKIP (insufficient data)")
    
    print(f"\nSuccessfully loaded {len(all_data)} stocks")
    
    if len(all_data) < 5:
        print("ERROR: Not enough stocks with data.")
        return
    
    # Align data
    print("\nAligning data to common dates...")
    common_index = sorted(list(set.intersection(*[set(df.index) for df in all_data.values()])))
    print(f"Common period: {common_index[0].strftime('%Y-%m-%d')} to {common_index[-1].strftime('%Y-%m-%d')}")
    print(f"Trading days: {len(common_index)}")
    
    aligned_data = {k: v.loc[common_index] for k, v in all_data.items()}
    
    # Fetch and calculate benchmark metrics
    print("\nFetching benchmark data (Nifty 50)...")
    benchmark_df = fetch_historical_data(BENCHMARK_INSTRUMENT_KEY, START_DATE, END_DATE)
    if not benchmark_df.empty:
        benchmark_df = benchmark_df.loc[benchmark_df.index.isin(common_index)]
        bench_returns = benchmark_df['returns'].iloc[-252:]  # Last 1 year
        BENCHMARK_SHARPE = calculate_sharpe_ratio(bench_returns, RISK_FREE_RATE)
        BENCHMARK_SORTINO = calculate_sortino_ratio(bench_returns, RISK_FREE_RATE)
        print(f"  Benchmark Sharpe Ratio:  {BENCHMARK_SHARPE:.2f}")
        print(f"  Benchmark Sortino Ratio: {BENCHMARK_SORTINO:.2f}")
        print("  Portfolios must BEAT these ratios to qualify!")
    else:
        print("  WARNING: Could not fetch benchmark. Skipping Sharpe/Sortino comparison.")
        BENCHMARK_SHARPE = None
        BENCHMARK_SORTINO = None
    
    # Generate random combinations
    stock_keys = list(aligned_data.keys())
    all_combos = list(itertools.combinations(stock_keys, PORTFOLIO_SIZE))
    random.shuffle(all_combos)  # Randomize to get diverse results
    
    print(f"\nTotal possible combinations: {len(all_combos)}")
    print("Will test ALL until 10 valid portfolios found!")
    
    # Search for valid portfolios
    valid_portfolios = []
    tested_count = 0
    
    print("\n" + "-"*70)
    print("SEARCHING FOR VALID PORTFOLIOS...")
    print("-"*70)
    
    for combo in all_combos:  # No limit - search until we find 10
        tested_count += 1
        symbols = [STOCK_UNIVERSE[k] for k in combo]
        
        print(f"\r[{tested_count}/{len(all_combos)}] Testing: {', '.join(symbols)[:50]}...", end="")
        
        result = evaluate_portfolio(combo, aligned_data, model)
        
        if is_good_portfolio(result):
            valid_portfolios.append(result)
            print(f"\n  [OK] VALID! TCN: {result['tcn_prediction']:+.2f}%, RAG: {result['rag_mean']:+.2f}%, Sharpe: {result['sharpe_ratio']:.2f} (>{BENCHMARK_SHARPE:.2f})")
            
            if len(valid_portfolios) >= MAX_PORTFOLIOS_TO_FIND:
                print(f"\n\nFound {MAX_PORTFOLIOS_TO_FIND} valid portfolios! Stopping search.")
                break
    
    print(f"\n\nSearch complete. Tested: {tested_count}, Valid: {len(valid_portfolios)}")
    
    if len(valid_portfolios) == 0:
        print("\nNo valid portfolios found with current criteria.")
        print("Try relaxing the selection criteria.")
        return
    
    # Sort by composite score for initial ordering
    for p in valid_portfolios:
        p['composite_score'] = (
            (p['rag_success_rate'] / 100) * 0.35 +
            max(0, 1 - p['rag_std'] / 20) * 0.25 +
            min(p['tcn_prediction'] / 10, 1) * 0.20 +
            min(max(p['sharpe_ratio'], 0) / 2, 1) * 0.20
        )
    
    valid_portfolios.sort(key=lambda x: x['composite_score'], reverse=True)
    
    # Display all valid portfolios with full details
    print("\n" + "="*70)
    print("       TOP {} PORTFOLIOS - FULL DETAILS".format(len(valid_portfolios)))
    print("="*70)
    
    for i, p in enumerate(valid_portfolios, 1):
        display_portfolio_details(p, i)
    
    # LLM Ranking (called ONCE)
    print("\n" + "="*70)
    print("       CALLING AI FOR FINAL RANKING...")
    print("="*70)
    
    rank_portfolios_with_llm(valid_portfolios)
    
    # Summary comparison table
    print("\n" + "="*70)
    print("       SUMMARY COMPARISON TABLE")
    print("="*70)
    print(f"\n{'Rank':<5} {'Portfolio':<40} {'TCN':>8} {'RAG Mean':>10} {'Success':>9} {'Sharpe':>8}")
    print("-"*80)
    
    for i, p in enumerate(valid_portfolios, 1):
        portfolio_str = ', '.join(p['symbols'][:3]) + ('...' if len(p['symbols']) > 3 else '')
        print(f"{i:<5} {portfolio_str:<40} {p['tcn_prediction']:>+7.2f}% {p['rag_mean']:>+9.2f}% {p['rag_success_rate']:>8.1f}% {p['sharpe_ratio']:>8.2f}")
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)

if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)
    main()
