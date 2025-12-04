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
import sqlite3
import json

# RAG-related imports
import google.generativeai as genai
from rag_prediction_simple import load_rag_system, find_similar_scenarios


# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Configuration ---
# Copied from analysis.py
ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI3M0FIOUMiLCJqdGkiOiI2OTJiMGMxZGIzNGEzMTEzNGI4Njg3ZGQiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6dHJ1ZSwiaWF0IjoxNzY0NDI4ODI5LCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3NjQ0NTM2MDB9.FyfmIVYRBjw8p51FnR532XWaSaBSHOmptXy3Es_22JA"
BASE_URL = "https://api.upstox.com/v3/historical-candle"
START_DATE = "2018-09-03"
END_DATE = "2025-12-04" # Matched with analysis.py
RISK_FREE_RATE = 0.065  # 6.5% Institutional Standard (restored to match analysis.py)
TRANSACTION_COST = 0.001  # 0.1% per rebalance
INITIAL_CAPITAL = 100000.0
BENCHMARK_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"
DATABASE_FILE = os.path.join("database", "portfolio_analysis.db")

# --- Risk Management ---
STOP_LOSS_THRESHOLD = 0.15  # 15% stop-loss (exit when drawdown exceeds this)

# --- API Keys ---
GEMINI_API_KEY = "AIzaSyDSe0VO0usxHXpRE4Jy3CKIz0RHXqrYTes"  # Get from: https://aistudio.google.com/app/apikey


# --- Portfolio Selection ---
CHOSEN_PORTFOLIO = {
    "NSE_EQ|INE029A01011": "BPCL",
    "NSE_EQ|INE674K01013": "ABCAPITAL",
    "NSE_EQ|INE918Z01012": "KAYNES",
    "NSE_EQ|INE237A01028": "KOTAKBANK",
    "NSE_EQ|INE397D01024":"BHARTIARTL",
    
}

# --- Data Fetching ---
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
            time.sleep(0.1) # Slight delay to be polite
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {instrument_key}: {e}")
            continue

    if all_data:
        df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df.index = df.index.tz_localize(None)
        df.sort_index(inplace=True) # Ensure sorted by date
        df["returns"] = df["close"].pct_change()
        return df.dropna()
    return pd.DataFrame()

# --- Financial Calculations ---
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

def optimize_weights(returns_df):
    """
    Hierarchical Risk Parity (HRP) Optimizer.
    Allocates weights based on hierarchical clustering and recursive bisection.
    Matches the optimization strategy from analysis.py
    """
    print("\n=== WEIGHT OPTIMIZATION PROCESS ===")

    # 1. Compute Covariance and Correlation
    cov = returns_df.cov()
    corr = returns_df.corr()

    # Handle constant returns (zero variance) to avoid NaNs
    if cov.empty or cov.values.sum() == 0:
        return np.full(returns_df.shape[1], 1.0 / returns_df.shape[1])

    print(f"Asset count: {len(returns_df.columns)}")
    print(f"Assets: {list(returns_df.columns)}")

    # 2. Hierarchical Clustering
    # Distance matrix based on correlation
    dist = np.sqrt((1 - corr) / 2)

    print(f"\nCorrelation Matrix:")
    print(corr.round(3))

    try:
        # Ensure diagonal is 0 for squareform checks and fill NaNs
        np.fill_diagonal(dist.values, 0)
        dist = dist.fillna(0)

        # Convert to condensed distance matrix
        condensed_dist = squareform(dist.values)

        # Single Linkage Clustering
        link = linkage(condensed_dist, method='single')


        print(f"\nDistance Matrix for Clustering:")
        print(dist.round(3))
    except Exception as e:
        print(f"Clustering failed: {e}")
        # Fallback to equal weights
        return np.full(returns_df.shape[1], 1.0 / returns_df.shape[1])
    
    # 3. Quasi-Diagonalization (Sort Indices)
    def get_quasi_diag(link):
        """Get the order of assets from hierarchical clustering"""
        n = link.shape[0] + 1
        def seriation(Z, N, cur_index):
            if cur_index < N:
                return [cur_index]
            else:
                left = int(Z[cur_index - N, 0])
                right = int(Z[cur_index - N, 1])
                return seriation(Z, N, left) + seriation(Z, N, right)
        # Start from the root (last merge)
        return seriation(link, n, 2 * n - 2)
    
    sort_ix = get_quasi_diag(link)
    print(f"\nQuasi-Diagonalization (Reordered Indices): {sort_ix}")
    print(f"Reordered assets: {[returns_df.columns[i] for i in sort_ix]}")
    
    # 4. Recursive Bisection
    def get_cluster_var(cov, c_items):
        cov_slice = cov.iloc[c_items, c_items]
        # Inverse variance weights
        w = 1 / np.diag(cov_slice)
        w /= w.sum()
        # Cluster variance
        cluster_var = np.dot(np.dot(w, cov_slice), w)
        print(f"  Cluster {c_items} variance: {cluster_var:.6f}")
        return cluster_var


    def get_rec_bipart(cov, sort_ix):
        print(f"\n--- RECURSIVE BISECTION START ---")
        print(f"Initial equal weights for all assets")
        w = pd.Series(1.0, index=sort_ix)
        print(f"Current weights: {w.values}")

        c_items = [sort_ix]
        level = 0

        while len(c_items) > 0:
            level += 1
            print(f"\nLEVEL {level}:")
            print(f"Clusters to process: {len(c_items)}")

            # Bisect each cluster into two sub-clusters
            new_c_items = []
            for cluster in c_items:
                if len(cluster) > 1:
                    # Split cluster in half
                    mid = len(cluster) // 2
                    c_items0 = cluster[:mid]
                    c_items1 = cluster[mid:]

                    print(f"  Splitting cluster {cluster}")
                    print(f"    Assets in cluster A: {[returns_df.columns[j] for j in c_items0]}")
                    print(f"    Assets in cluster B: {[returns_df.columns[j] for j in c_items1]}")

                    c_var0 = get_cluster_var(cov, c_items0)
                    c_var1 = get_cluster_var(cov, c_items1)

                    # Risk parity allocation (inverse variance weighting)
                    alpha = 1 - c_var0 / (c_var0 + c_var1)
                    print(f"    Alpha allocation: A={alpha:.3f}, B={1-alpha:.3f}")

                    # Apply weights to all items in each sub-cluster
                    w[c_items0] *= alpha
                    w[c_items1] *= 1 - alpha

                    print(f"    Updated weights: {w.values}")
                    
                    # Add sub-clusters to next level
                    new_c_items.append(c_items0)
                    new_c_items.append(c_items1)
            
            # Update clusters for next iteration
            c_items = new_c_items

        print(f"\nFINAL WEIGHTS (before reordering): {w.values}")
        return w

    # Calculate weights
    weights_series = get_rec_bipart(cov, sort_ix)

    # Return weights in original column order
    final_weights = weights_series.sort_index().values
    print(f"FINAL WEIGHTS (original order): {final_weights}")
    print("Weights by asset:")
    for i, asset in enumerate(returns_df.columns):
        print(f"  {asset}: {final_weights[i]:.4f}")

    return final_weights

# --- Technical Indicators ---
def calculate_indicators(df):
    df = df.copy()
    median_price = (df['high'] + df['low']) / 2
    def smma(series, window):
        return series.ewm(alpha=1/window, adjust=False).mean()

    # Alligator Indicator Lines
    df['alligator_jaw'] = smma(median_price, 13).shift(8)
    df['alligator_teeth'] = smma(median_price, 8).shift(5)
    df['alligator_lips'] = smma(median_price, 5).shift(3)

    # Enhanced Alligator Features for better pattern recognition
    # 1. Distances between lines
    df['alligator_jaw_teeth_dist'] = df['alligator_jaw'] - df['alligator_teeth']
    df['alligator_teeth_lips_dist'] = df['alligator_teeth'] - df['alligator_lips']
    df['alligator_jaw_lips_dist'] = df['alligator_jaw'] - df['alligator_lips']

    # 2. Relative positions (jaw vs teeth vs lips ordering)
    # Coding different line formations:
    # 0: Jaw > Teeth > Lips (bearish convergence)
    # 1: Jaw > Teeth < Lips (teeth below others)
    # 2: Jaw < Teeth > Lips (jaw below others)
    # 3: Jaw < Teeth < Lips (bullish convergence)
    # 4: Jaw = Teeth > Lips (equal jaw/teeth)
    # 5: Jaw = Teeth < Lips (equal jaw/teeth below lips)
    # 6: Jaw > Teeth = Lips (equal teeth/lips)
    # 7: Jaw < Teeth = Lips (equal teeth/lips below jaw)
    # 8: Jaw = Teeth = Lips (all equal - rare)

    def get_formation_code(row):
        jaw, teeth, lips = row['alligator_jaw'], row['alligator_teeth'], row['alligator_lips']

        # Small tolerance for "equal"
        tol = abs(row['close']) * 0.001  # 0.1% tolerance

        if abs(jaw - teeth) < tol and abs(teeth - lips) < tol:
            return 8  # All equal
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
        return 0  # Default

    df['alligator_formation'] = df.apply(get_formation_code, axis=1)

    # 3. Position states relative to price
    # 0: All lines above price
    # 1: All lines below price
    # 2: All lines mixed positions
    # 3: Jaw above, others below
    # 4: Jaw below, others above
    # 5: Teeth between jaw and lips relative to price

    def get_position_state(row):
        price = row['close']
        jaw_above = row['alligator_jaw'] > price
        teeth_above = row['alligator_teeth'] > price
        lips_above = row['alligator_lips'] > price

        above_count = sum([jaw_above, teeth_above, lips_above])

        if above_count == 3:
            return 0  # All above
        elif above_count == 0:
            return 1  # All below
        elif above_count == 1 and jaw_above:
            return 3  # Only jaw above
        elif above_count == 1 and not jaw_above:
            return 4  # Jaw below, others may vary
        else:
            return 2  # Mixed positions

    df['alligator_position_state'] = df.apply(get_position_state, axis=1)

    # 4. Convergence/Divergence patterns
    df['alligator_jaw_teeth_converging'] = (df['alligator_jaw_teeth_dist'].diff() < 0).astype(int)
    df['alligator_teeth_lips_converging'] = (df['alligator_teeth_lips_dist'].diff() < 0).astype(int)

    # 5. Normalize distances
    df['alligator_jaw_teeth_dist_norm'] = df['alligator_jaw_teeth_dist'] / df['close']
    df['alligator_teeth_lips_dist_norm'] = df['alligator_teeth_lips_dist'] / df['close']
    df['alligator_jaw_lips_dist_norm'] = df['alligator_jaw_lips_dist'] / df['close']

    # Enhanced RSI Features
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # RSI States (9 categories)
    # 0: Extreme oversold (<20)  1: Oversold (20-30)
    # 2: Weak oversold (30-40)  3: Neutral (40-60)
    # 4: Weak overbought (60-70) 5: Overbought (70-80)
    # 6: Extreme overbought (>80) 7: RSI rising momentum
    # 8: RSI falling momentum
    def get_rsi_state(row):
        rsi = row['rsi']
        prev_rsi = row['rsi'] if pd.notna(row.get('rsi_prev', row['rsi'])) else row['rsi']
        if rsi < 20: return 0
        elif rsi < 30: return 1
        elif rsi < 40: return 2
        elif rsi < 60: return 3
        elif rsi < 70: return 4
        elif rsi < 80: return 5
        elif rsi > 80: return 6
        elif rsi > prev_rsi: return 7  # Rising momentum
        else: return 8  # Falling momentum

    df['rsi_prev'] = df['rsi'].shift(1)  # For momentum
    df['rsi_state'] = df.apply(get_rsi_state, axis=1)
    df['rsi_normalized'] = (df['rsi'] - 50) / 50  # Normalize around neutral 50

    # Enhanced MACD Features
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']

    # MACD States (9 categories)
    # 0: MACD crosses Signal Up (bullish)  1: MACD crosses Signal Down (bearish)
    # 2: MACD > 0 (bullish)                3: MACD < 0 (bearish)
    # 4: MACD > Signal (rising)           5: MACD < Signal (falling)
    # 6: Zero line crossover up           7: Zero line crossover down
    # 8: Histogram increasing              9: Histogram decreasing

    def get_macd_state(row):
        macd, signal, hist = row['macd'], row['macd_signal'], row['macd_histogram']
        prev_macd = row.get('macd_prev', macd)
        prev_signal = row.get('macd_signal_prev', signal)
        prev_hist = row.get('macd_hist_prev', hist)

        # Primary signal: crossover
        if prev_macd <= prev_signal and macd > signal: return 0  # Bull crossover
        elif prev_macd >= prev_signal and macd < signal: return 1  # Bear crossover

        # Secondary: zero line
        elif prev_macd <= 0 and macd > 0: return 6  # Bullish zero cross
        elif prev_macd >= 0 and macd < 0: return 7  # Bearish zero cross

        # Tertiary: position and momentum
        elif macd > 0 and macd > signal: return 2  # Bullish trend
        elif macd < 0 and macd < signal: return 3  # Bearish trend
        elif macd > signal: return 4  # MACD above signal
        elif macd < signal: return 5  # MACD below signal
        elif hist > prev_hist: return 8  # Histogram increasing
        else: return 9  # Histogram decreasing

        # Keep previous values for crossover detection
        df['macd_prev'] = df['macd'].shift(1)
        df['macd_signal_prev'] = df['macd_signal'].shift(1)
        df['macd_hist_prev'] = df['macd_histogram'].shift(1)

    # Apply MACD state
    df[['macd_prev', 'macd_signal_prev', 'macd_hist_prev']] = df[['macd', 'macd_signal', 'macd_histogram']].shift(1)
    df['macd_state'] = df.apply(get_macd_state, axis=1)

    # Normalized MACD components
    df['macd_normalized'] = df['macd'] / abs(df['close'])  # Scale to price
    df['macd_signal_normalized'] = df['macd_signal'] / abs(df['close'])
    df['macd_histogram_normalized'] = df['macd_histogram'] / abs(df['close'])

    # Enhanced Bollinger Bands Features
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)

    # Bollinger Band States (9 categories)
    # 0: Price touches lower band (buy signal)    1: Price touches middle band
    # 2: Price touches upper band (sell signal)  3: Price above upper (overbought)
    # 4: Price below lower (oversold)            5: Price in upper half (bullish)
    # 6: Price in lower half (bearish)           7: Bands squeezing (low vol)
    # 8: Bands expanding (high vol)
    df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    def get_bb_state(row):
        price = row['close']
        upper, middle, lower = row['bb_upper'], row['bb_middle'], row['bb_lower']
        bandwidth = row['bb_bandwidth']
        percent_b = row['bb_percent_b']

        # Band touches
        if price <= lower: return 0  # Lower band touch
        elif price >= upper: return 2  # Upper band touch

        # Extremes beyond bands
        elif price > upper: return 3  # Above upper
        elif price < lower: return 4  # Below lower

        # Position within bands
        elif percent_b > 0.5: return 5  # Upper half of bands
        elif percent_b < 0.5: return 6  # Lower half of bands

        else: return 1  # Near middle

        # Additional volatility states (could be added)
        # if bandwidth < quantile: return 7  # Squeeze
        # elif bandwidth > quantile: return 8  # Expansion

    df['bb_state'] = df.apply(get_bb_state, axis=1)

    # Normalized Bollinger components
    df['bb_middle_normalized'] = (df['bb_middle'] - df['close'].rolling(window=20).mean()) / abs(df['close'])
    df['bb_upper_normalized'] = (df['bb_upper'] - df['close']) / abs(df['close'])
    df['bb_lower_normalized'] = (df['close'] - df['bb_lower']) / abs(df['close'])
    df['bb_std_normalized'] = df['bb_std'] / abs(df['close'])
    df['bb_bandwidth_normalized'] = df['bb_bandwidth']  # Already relative

    # Volatility features
    df['rolling_volatility'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
    df['volatility_normalized'] = df['rolling_volatility'] / df['close'].rolling(window=252).mean()  # Annualized relative vol

    # Clean up temporary columns
    df.drop(['rsi_prev', 'macd_prev', 'macd_signal_prev', 'macd_hist_prev', 'bb_bandwidth', 'bb_percent_b'],
            axis=1, errors='ignore', inplace=True)
    return df.dropna()

# --- Portfolio Simulation ---
def simulate_portfolio(all_stock_data):
    """
    Simulates the portfolio value day-by-day with 60-day rebalancing, transaction costs,
    and 15% stop-loss protection.
    
    Stop-Loss Logic:
    - Tracks entry value at each rebalance
    - If portfolio drops 15% from entry, exits all positions to cash
    - Waits until next scheduled rebalance to re-enter
    - Rebalances with optimized weights based on lookback period
    
    Returns a DataFrame with 'close', 'high', 'low' (simulated) for the portfolio,
    the final weights, and the next rebalance date.
    """
    print(f"Simulating portfolio performance with {STOP_LOSS_THRESHOLD*100:.0f}% stop-loss...")
    
    # Align all data to common index
    common_index = sorted(list(set.intersection(*[set(df.index) for df in all_stock_data.values()])))
    aligned_data = {k: v.loc[common_index] for k, v in all_stock_data.items()}
    
    portfolio_value_series = []
    portfolio_high_series = []
    portfolio_low_series = []
    
    current_capital = INITIAL_CAPITAL
    current_weights = np.array([1.0/len(CHOSEN_PORTFOLIO)] * len(CHOSEN_PORTFOLIO))
    units = np.zeros(len(CHOSEN_PORTFOLIO))
    
    # Stop-loss tracking
    entry_value = INITIAL_CAPITAL  # Track value at entry/rebalance
    in_position = False  # Whether we're currently holding positions
    stop_loss_triggered = False  # Whether stop-loss was triggered in current cycle
    
    # Initial purchase
    first_prices = np.array([aligned_data[k].iloc[0]['close'] for k in CHOSEN_PORTFOLIO.keys()])
    units = (current_capital * current_weights) / first_prices
    in_position = True
    entry_value = current_capital
    
    # Rebalancing schedule
    start_date = common_index[0]
    next_rebalance_date = start_date + timedelta(days=60)
    
    stop_loss_count = 0  # Track number of stop-loss events
    
    for i, date in enumerate(common_index):
        # Get current prices
        current_prices = np.array([aligned_data[k].iloc[i]['close'] for k in CHOSEN_PORTFOLIO.keys()])
        current_highs = np.array([aligned_data[k].iloc[i]['high'] for k in CHOSEN_PORTFOLIO.keys()])
        current_lows = np.array([aligned_data[k].iloc[i]['low'] for k in CHOSEN_PORTFOLIO.keys()])
        
        # Calculate Portfolio Value
        if in_position:
            daily_value = np.sum(units * current_prices)
            daily_high = np.sum(units * current_highs)
            daily_low = np.sum(units * current_lows)
            
            # Check for stop-loss trigger (15% drawdown from entry)
            drawdown = (entry_value - daily_value) / entry_value
            if drawdown >= STOP_LOSS_THRESHOLD:
                print(f"\n[!] STOP-LOSS TRIGGERED on {date.strftime('%Y-%m-%d')}")
                print(f"   Entry Value: Rs.{entry_value:,.2f}")
                print(f"   Current Value: Rs.{daily_value:,.2f}")
                print(f"   Drawdown: {drawdown*100:.2f}%")
                print(f"   Exiting positions and moving to cash until {next_rebalance_date.strftime('%Y-%m-%d')}")
                
                # Exit all positions - move to cash
                current_capital = daily_value
                units = np.zeros(len(CHOSEN_PORTFOLIO))
                in_position = False
                stop_loss_triggered = True
                stop_loss_count += 1
        else:
            # In cash - portfolio value remains constant
            daily_value = current_capital
            daily_high = current_capital
            daily_low = current_capital

        portfolio_value_series.append(daily_value)
        portfolio_high_series.append(daily_high)
        portfolio_low_series.append(daily_low)
        
        # Check for Rebalance
        if date >= next_rebalance_date:
            # Lookback for optimization (120 days to match analysis.py)
            lookback_start = date - timedelta(days=120)
            lookback_returns_list = []
            for k in CHOSEN_PORTFOLIO.keys():
                df = aligned_data[k]
                mask = (df.index >= lookback_start) & (df.index < date)
                lookback_returns_list.append(df.loc[mask, 'returns'])
            
            lookback_df = pd.concat(lookback_returns_list, axis=1).dropna()
            
            if len(lookback_df) > 10: # Ensure enough data
                new_weights = optimize_weights(lookback_df)
                
                # Get current capital (whether in position or cash)
                if in_position:
                    current_capital = daily_value
                
                # Apply Transaction Costs
                cost = current_capital * TRANSACTION_COST
                current_capital -= cost
                
                # Re-enter positions with new optimized weights
                units = (current_capital * new_weights) / current_prices
                current_weights = new_weights
                in_position = True
                entry_value = current_capital  # Reset entry value for new cycle
                stop_loss_triggered = False
                
                print(f"\n[*] REBALANCING on {date.strftime('%Y-%m-%d')}")
                print(f"   Portfolio Value: Rs.{current_capital:,.2f}")
                print(f"   New weights applied, next rebalance: {(date + timedelta(days=60)).strftime('%Y-%m-%d')}")
                
                next_rebalance_date = date + timedelta(days=60)
    
    portfolio_df = pd.DataFrame({
        'close': portfolio_value_series,
        'high': portfolio_high_series,
        'low': portfolio_low_series
    }, index=common_index)
    
    print(f"\n[SUMMARY] Stop-Loss Summary: Triggered {stop_loss_count} times during simulation")

    return portfolio_df, current_weights, next_rebalance_date

def simulate_equal_weight_portfolio(all_stock_data):
    """
    Simulates an equal-weighted portfolio without rebalancing.
    Returns a DataFrame with 'close', 'high', 'low' for the equal-weighted portfolio.
    """
    print("Simulating equal-weighted portfolio performance...")

    # Align all data to common index
    common_index = sorted(list(set.intersection(*[set(df.index) for df in all_stock_data.values()])))
    aligned_data = {k: v.loc[common_index] for k, v in all_stock_data.items()}

    portfolio_value_series = []
    portfolio_high_series = []
    portfolio_low_series = []

    current_capital = INITIAL_CAPITAL
    equal_weights = np.array([1.0/len(CHOSEN_PORTFOLIO)] * len(CHOSEN_PORTFOLIO))
    units = np.zeros(len(CHOSEN_PORTFOLIO))

    # Initial purchase with equal weights
    first_prices = np.array([aligned_data[k].iloc[0]['close'] for k in CHOSEN_PORTFOLIO.keys()])
    units = (current_capital * equal_weights) / first_prices

    for i, date in enumerate(common_index):
        # Get current prices
        current_prices = np.array([aligned_data[k].iloc[i]['close'] for k in CHOSEN_PORTFOLIO.keys()])
        current_highs = np.array([aligned_data[k].iloc[i]['high'] for k in CHOSEN_PORTFOLIO.keys()])
        current_lows = np.array([aligned_data[k].iloc[i]['low'] for k in CHOSEN_PORTFOLIO.keys()])

        # Calculate Portfolio Value
        daily_value = np.sum(units * current_prices)
        daily_high = np.sum(units * current_highs)
        daily_low = np.sum(units * current_lows)

        portfolio_value_series.append(daily_value)
        portfolio_high_series.append(daily_high)
        portfolio_low_series.append(daily_low)

        # No rebalancing for equal-weighted

    portfolio_df = pd.DataFrame({
        'close': portfolio_value_series,
        'high': portfolio_high_series,
        'low': portfolio_low_series
    }, index=common_index)

    return portfolio_df

# --- LSTM Model ---
def create_dataset(scaled_dataset, raw_prices, lookback, forward):
    X, y = [], []
    for i in range(len(scaled_dataset) - lookback - forward):
        X.append(scaled_dataset[i:(i + lookback)])
        # Target: Return Factor (Future Value / Current Value)
        # Use raw prices to calculate the actual return factor
        current_val = raw_prices[i + lookback - 1] 
        future_val = raw_prices[i + lookback + forward - 1]
        y.append(future_val / current_val) 
    return np.array(X), np.array(y)

# --- Database Functions ---
# --- Database Functions ---
def get_portfolio_signature():
    """Returns a unique signature for the current portfolio composition."""
    return ",".join(sorted(CHOSEN_PORTFOLIO.values()))

def init_db():
    """Initialize the database and handle schema migration."""
    if not os.path.exists("database"):
        os.makedirs("database")
    
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='portfolio_predictions'")
    table_exists = cursor.fetchone()
    
    if table_exists:
        # Check if portfolio_assets column exists
        cursor.execute("PRAGMA table_info(portfolio_predictions)")
        columns = [info[1] for info in cursor.fetchall()]
        
        if 'portfolio_assets' not in columns:
            print("Migrating database schema to include portfolio_assets...")
            # Migration needed
            cursor.execute("ALTER TABLE portfolio_predictions RENAME TO portfolio_predictions_old")
            
            # Create new table
            cursor.execute('''
            CREATE TABLE portfolio_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rebalance_date TEXT,
                expiry_date TEXT,
                portfolio_value_at_rebalance REAL,
                predicted_return_factor REAL,
                predicted_value_at_expiry REAL,
                weights_json TEXT,
                correlations_json TEXT,
                risk_metrics_json TEXT,
                portfolio_assets TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(rebalance_date, portfolio_assets)
            )
            ''')
            
            # Copy data
            cursor.execute("SELECT * FROM portfolio_predictions_old")
            rows = cursor.fetchall()
            
            for row in rows:
                # row structure depends on old schema: id(0), date(1), expiry(2), val(3), factor(4), pred_val(5), weights(6), corr(7), risk(8), created(9)
                weights_json = row[6]
                try:
                    weights = json.loads(weights_json)
                    assets = ",".join(sorted(weights.keys()))
                except:
                    assets = "UNKNOWN"
                
                try:
                    cursor.execute('''
                    INSERT INTO portfolio_predictions (
                        rebalance_date, expiry_date, portfolio_value_at_rebalance,
                        predicted_return_factor, predicted_value_at_expiry,
                        weights_json, correlations_json, risk_metrics_json,
                        portfolio_assets, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], assets, row[9]))
                except sqlite3.IntegrityError:
                    pass # Skip duplicates during migration
            
            # Drop old table
            cursor.execute("DROP TABLE portfolio_predictions_old")
            conn.commit()
            print("Database migration completed.")
    else:
        # Create new table directly
        cursor.execute('''
        CREATE TABLE portfolio_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rebalance_date TEXT,
            expiry_date TEXT,
            portfolio_value_at_rebalance REAL,
            predicted_return_factor REAL,
            predicted_value_at_expiry REAL,
            weights_json TEXT,
            correlations_json TEXT,
            risk_metrics_json TEXT,
            portfolio_assets TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(rebalance_date, portfolio_assets)
        )
        ''')
        conn.commit()
    
    conn.close()

def get_existing_prediction(rebalance_date, portfolio_signature):
    """Check if a prediction already exists for this rebalance date AND portfolio."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM portfolio_predictions WHERE rebalance_date = ? AND portfolio_assets = ?", 
                   (rebalance_date, portfolio_signature))
    row = cursor.fetchone()
    conn.close()
    return row

def save_prediction_to_db(data):
    """Save prediction and metrics to the database."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
        INSERT INTO portfolio_predictions (
            rebalance_date, expiry_date, portfolio_value_at_rebalance,
            predicted_return_factor, predicted_value_at_expiry,
            weights_json, correlations_json, risk_metrics_json,
            portfolio_assets
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['rebalance_date'],
            data['expiry_date'],
            data['portfolio_value_at_rebalance'],
            data['predicted_return_factor'],
            data['predicted_value_at_expiry'],
            json.dumps(data['weights']),
            json.dumps(data['correlations']),
            json.dumps(data['risk_metrics']),
            data['portfolio_assets']
        ))
        conn.commit()
        print(f"Successfully saved prediction for {data['rebalance_date']} ({data['portfolio_assets']}) to database.")
    except sqlite3.IntegrityError:
        print(f"Record for {data['rebalance_date']} with this portfolio already exists. Skipping.")
    except Exception as e:
        print(f"Error saving to database: {e}")
    finally:
        conn.close()

def calculate_metrics_at_date(portfolio_df, benchmark_df, risk_free_rate, all_stock_data, weights, date):
    """Calculate risk metrics up to a specific date."""
    # Slice data up to date
    mask_port = portfolio_df.index <= date
    mask_bench = benchmark_df.index <= date
    
    port_returns = portfolio_df.loc[mask_port, 'returns']
    bench_returns = benchmark_df.loc[mask_bench, 'returns']
    
    # Align
    common_index = port_returns.index.intersection(bench_returns.index)
    port_returns = port_returns.loc[common_index]
    bench_returns = bench_returns.loc[common_index]
    
    if port_returns.empty:
        return {}
        
    metrics = {}
    
    # Returns
    metrics['total_return'] = (1 + port_returns).prod() - 1
    
    # Sharpe
    excess_port = port_returns - risk_free_rate/252
    metrics['sharpe_ratio'] = (excess_port.mean() / port_returns.std()) * np.sqrt(252) if port_returns.std() != 0 else 0
    
    # Sortino
    downside_port = port_returns[port_returns < 0].std() * np.sqrt(252)
    metrics['sortino_ratio'] = (excess_port.mean() * 252) / downside_port if downside_port != 0 else 0
    
    # Max Drawdown
    cum_rets = (1 + port_returns).cumprod()
    peak = cum_rets.expanding(min_periods=1).max()
    dd = (cum_rets - peak) / peak
    metrics['max_drawdown'] = dd.min()
    
    # Beta
    if bench_returns.var() != 0:
        covariance = np.cov(port_returns, bench_returns)[0][1]
        metrics['beta'] = covariance / bench_returns.var()
    else:
        metrics['beta'] = 0
        
    # Volatility
    metrics['volatility'] = port_returns.std() * np.sqrt(252)
    
    # Correlations (at that point in time, using lookback)
    # Use last 120 days for correlation snapshot as per optimization logic
    lookback_start = date - timedelta(days=120)
    corr_data = {}
    for k, df in all_stock_data.items():
        mask = (df.index >= lookback_start) & (df.index <= date)
        if mask.any():
            corr_data[CHOSEN_PORTFOLIO[k]] = df.loc[mask, 'returns']
    
    if corr_data:
        corr_df = pd.DataFrame(corr_data)
        metrics['correlation_matrix'] = corr_df.corr().to_dict()
    else:
        metrics['correlation_matrix'] = {}
        
    return metrics

def generate_detailed_report(portfolio_df, benchmark_df, risk_free_rate):
    print("\n" + "="*60)
    print("       DETAILED PERFORMANCE REPORT")
    print("="*60)

    # Align data
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
    excess_port = port_returns - risk_free_rate/252
    excess_bench = bench_returns - risk_free_rate/252
    
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
    rolling_vol = port_returns.rolling(20).std().iloc[-1] * np.sqrt(252)
    print(f"  Rolling Std Dev (20d):   {rolling_vol*100:.2f}%")

    # --- Risk Sensitivity ---
    covariance = np.cov(port_returns, bench_returns)[0][1]
    beta = covariance / bench_returns.var()
    print(f"\nRisk Sensitivity")
    print(f"  Beta (Portfolio):        {beta:.2f}")
    
    # --- Alpha and Related Stats ---
    # Jensen's Alpha = Rp - (Rf + Beta * (Rm - Rf))
    # Annualized Alpha
    annualized_return_port = (1 + port_returns.mean())**252 - 1
    annualized_return_bench = (1 + bench_returns.mean())**252 - 1
    jensens_alpha = annualized_return_port - (risk_free_rate + beta * (annualized_return_bench - risk_free_rate))
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(bench_returns, port_returns)
    
    print(f"\nAlpha and Related Stats")
    print(f"  Jensen's Alpha:          {jensens_alpha*100:.2f}%")
    print(f"  R-squared:               {r_value**2:.2f}")
    
    # Alpha Distribution (Residuals)
    residuals = port_returns - (risk_free_rate/252 + beta * (bench_returns - risk_free_rate/252))
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
    print(f"  Diversification Ratio:   N/A (Multi-asset portfolio)")
    
    print("="*60)

# --- Main Function ---
def main():
    """Main execution function"""
    print("="*60)
    print("  REBALANCED PREDICTIVE PORTFOLIO MODEL")
    print("="*60)
    
    # 1. Fetch Data
    print("\nFetching stock data...")
    all_stock_data = {}
    for key, symbol in CHOSEN_PORTFOLIO.items():
        print(f"  Fetching {symbol}...")
        df = fetch_historical_data(key, START_DATE, END_DATE)
        if not df.empty:
            all_stock_data[key] = df
        else:
            print(f"  WARNING: No data for {symbol}")
    
    if len(all_stock_data) == 0:
        print("ERROR: No stock data fetched. Exiting.")
        return
    
    # 2. Simulate Portfolio
    portfolio_df, final_weights, next_rebalance_date = simulate_portfolio(all_stock_data)
    portfolio_df['returns'] = portfolio_df['close'].pct_change()
    portfolio_df = portfolio_df.dropna()

    # 3. Fetch Benchmark Data
    print("\nFetching benchmark data (Nifty 50)...")
    benchmark_df = fetch_historical_data(BENCHMARK_INSTRUMENT_KEY, START_DATE, END_DATE)
    if benchmark_df.empty:
        print("WARNING: Benchmark data not found. Using equal-weighted portfolio as fallback.")
        benchmark_df = simulate_equal_weight_portfolio(all_stock_data)
        benchmark_df['returns'] = benchmark_df['close'].pct_change()
        benchmark_df = benchmark_df.dropna()
    else:
        # Ensure benchmark has returns
        if 'returns' not in benchmark_df.columns:
             benchmark_df['returns'] = benchmark_df['close'].pct_change()
        benchmark_df = benchmark_df.dropna()

    # 4. Generate Report
    generate_detailed_report(portfolio_df, benchmark_df, RISK_FREE_RATE, all_stock_data, final_weights)

    # 5. Predict Next 60 Days
    print("\nPredicting next 60-day outcome...")
    
    # Prepare data for prediction
    # We need the last 120 days of data for the model
    # And we need to calculate indicators on it
    
    # Re-create the portfolio series for the last chunk to get indicators
    # We can just use the portfolio_df we already have
    
    # Calculate indicators on the portfolio_df
    portfolio_with_indicators = calculate_indicators(portfolio_df)
    
    # Get last 120 days
    last_120_days = portfolio_with_indicators.iloc[-120:]
    
    if len(last_120_days) < 120:
        print("Error: Not enough data for prediction.")
        return
        
    feature_cols = ['close', 'high', 'low', 'alligator_jaw', 'alligator_teeth', 'alligator_lips',
                    'alligator_jaw_teeth_dist_norm', 'alligator_teeth_lips_dist_norm', 'alligator_jaw_lips_dist_norm',
                    'alligator_formation', 'alligator_position_state', 'alligator_jaw_teeth_converging', 'alligator_teeth_lips_converging',
                    'rsi', 'rsi_state', 'rsi_normalized',
                    'macd', 'macd_signal', 'macd_histogram', 'macd_state', 'macd_normalized', 'macd_signal_normalized', 'macd_histogram_normalized',
                    'bb_upper', 'bb_middle', 'bb_lower', 'bb_state', 'bb_std_normalized', 'bb_bandwidth_normalized',
                    'rolling_volatility', 'volatility_normalized']
    
    features = last_120_days[feature_cols].values
    
    # Scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)
    
    # Reshape for LSTM with 31 features
    input_data = scaled_features.reshape(1, 120, 31)
    
    # Load Model
    try:
        model = load_model("generalized_lstm_model_improved_v2.keras")
        prediction = model.predict(input_data)
        predicted_return_factor = prediction[0][0]
        
        current_value = portfolio_df['close'].iloc[-1]
        predicted_value = current_value * predicted_return_factor
        pl = predicted_value - current_value
        pl_pct = (predicted_return_factor - 1) * 100
        
        print("\n" + "="*60)
        print("       PORTFOLIO PREDICTION REPORT")
        print("="*60)
        print(f"Current Date:           {portfolio_df.index[-1].strftime('%Y-%m-%d')}")
        print(f"Current Portfolio Value: RS{current_value:,.2f}")
        print(f"Predicted Return Factor: {predicted_return_factor:.4f}x")
        print(f"Predicted Value (60d):   RS{predicted_value:,.2f}")
        print(f"Projected P/L:           RS{pl:,.2f} ({pl_pct:.2f}%)")
        print("-" * 60)
        print("LATEST PORTFOLIO WEIGHTS:")
        for i, (key, symbol) in enumerate(CHOSEN_PORTFOLIO.items()):
            print(f"  {symbol:<15}: {final_weights[i]:.4f}")
        print("-" * 60)
        print(f"Weights Valid Until:     {next_rebalance_date.strftime('%Y-%m-%d')}")
        print(f"Next Rebalance In:       {(next_rebalance_date - portfolio_df.index[-1]).days} days")
        print("="*60)
        
    except Exception as e:
        print(f"Error loading model or predicting: {e}")
        print("Ensure 'generalized_lstm_model.keras' exists.")
def generate_detailed_report(portfolio_df, benchmark_df, risk_free_rate, all_stock_data, final_weights):
    print("\n" + "="*60)
    print("       DETAILED PERFORMANCE REPORT")
    print("="*60)

    # Align data
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
    excess_port = port_returns - risk_free_rate/252
    excess_bench = bench_returns - risk_free_rate/252
    
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
    rolling_vol = port_returns.rolling(20).std().iloc[-1] * np.sqrt(252)
    print(f"  Rolling Std Dev (20d):   {rolling_vol*100:.2f}%")

    # --- Risk Sensitivity ---
    covariance = np.cov(port_returns, bench_returns)[0][1]
    beta = covariance / bench_returns.var()

    # Calculate individual stock betas and weighted beta
    stock_betas = []
    for k in CHOSEN_PORTFOLIO.keys():
        if k in all_stock_data:
            s_ret = all_stock_data[k]['returns'].loc[common_index]
            cov_stock = np.cov(s_ret, bench_returns)[0][1]
            beta_stock = cov_stock / bench_returns.var() if bench_returns.var() != 0 else 0
            stock_betas.append(beta_stock)
        else:
            stock_betas.append(0)
    weighted_beta = np.sum(np.array(stock_betas) * final_weights)

    print(f"\nRisk Sensitivity")
    print(f"  Beta (Portfolio):        {weighted_beta:.2f}")
    print(f"  Beta (General):          {beta:.2f}")
    
    # --- Alpha and Related Stats ---
    # Jensen's Alpha = Rp - (Rf + Beta * (Rm - Rf))
    # Annualized Alpha
    annualized_return_port = (1 + port_returns.mean())**252 - 1
    annualized_return_bench = (1 + bench_returns.mean())**252 - 1
    jensens_alpha = annualized_return_port - (risk_free_rate + beta * (annualized_return_bench - risk_free_rate))
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(bench_returns, port_returns)
    
    print(f"\nAlpha and Related Stats")
    print(f"  Jensen's Alpha:          {jensens_alpha*100:.2f}%")
    print(f"  R-squared:               {r_value**2:.2f}")
    
    # Alpha Distribution (Residuals)
    residuals = port_returns - (risk_free_rate/252 + beta * (bench_returns - risk_free_rate/252))
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
    # Calculate Diversification Ratio
    # Ratio = (Weighted Avg Volatility of Assets) / (Portfolio Volatility)
    
    # 1. Get individual stock returns aligned with portfolio
    stock_vols = []
    for k in CHOSEN_PORTFOLIO.keys():
        if k in all_stock_data:
            s_ret = all_stock_data[k]['returns'].loc[common_index]
            stock_vols.append(s_ret.std() * np.sqrt(252))
        else:
            stock_vols.append(0)
            
    weighted_avg_vol = np.sum(np.array(stock_vols) * final_weights)
    
    if std_port > 0:
        div_ratio = weighted_avg_vol / std_port
        print(f"  Diversification Ratio:   {div_ratio:.2f}")
    else:
        print(f"  Diversification Ratio:   N/A")

    # --- Correlation Matrix ---
    print(f"\nCorrelation Matrix (Pairwise Return Correlations)")
    # Create DataFrame of all stock returns
    returns_dict = {}
    for k, symbol in CHOSEN_PORTFOLIO.items():
        if k in all_stock_data:
            returns_dict[symbol] = all_stock_data[k]['returns'].loc[common_index]
    
    if returns_dict:
        returns_df_all = pd.DataFrame(returns_dict)
        corr_matrix = returns_df_all.corr()
        print(corr_matrix.round(2))
    else:
        print("  N/A (Insufficient Data)")
    
    print("="*60)

# --- Main Function ---
def main():
    """Main execution function"""
    print("="*60)
    print("  REBALANCED PREDICTIVE PORTFOLIO MODEL")
    print("="*60)
    
    # 1. Fetch Data
    print("\nFetching stock data...")
    all_stock_data = {}
    for key, symbol in CHOSEN_PORTFOLIO.items():
        print(f"  Fetching {symbol}...")
        df = fetch_historical_data(key, START_DATE, END_DATE)
        if not df.empty:
            all_stock_data[key] = df
        else:
            print(f"  WARNING: No data for {symbol}")
    
    if len(all_stock_data) == 0:
        print("ERROR: No stock data fetched. Exiting.")
        return
    
    # 2. Simulate Portfolio
    portfolio_df, final_weights, next_rebalance_date = simulate_portfolio(all_stock_data)
    portfolio_df['returns'] = portfolio_df['close'].pct_change()
    portfolio_df = portfolio_df.dropna()

    # 2b. Simulate Equal-Weighted Portfolio
    equal_weight_df = simulate_equal_weight_portfolio(all_stock_data)
    equal_weight_df['returns'] = equal_weight_df['close'].pct_change()
    equal_weight_df = equal_weight_df.dropna()

    # 3. Fetch Benchmark
    print("\nFetching benchmark data...")
    benchmark_df = fetch_historical_data(BENCHMARK_INSTRUMENT_KEY, START_DATE, END_DATE)
    
    # 4. Prepare Data for LSTM
    print("\nPreparing data for LSTM...")
    portfolio_with_indicators = calculate_indicators(portfolio_df)
    
    # Features for LSTM (Must match the generalized model's features)
    feature_cols = ['close', 'high', 'low', 'alligator_jaw', 'alligator_teeth', 'alligator_lips',
                    'alligator_jaw_teeth_dist_norm', 'alligator_teeth_lips_dist_norm', 'alligator_jaw_lips_dist_norm',
                    'alligator_formation', 'alligator_position_state', 'alligator_jaw_teeth_converging', 'alligator_teeth_lips_converging',
                    'rsi', 'rsi_state', 'rsi_normalized',
                    'macd', 'macd_signal', 'macd_histogram', 'macd_state', 'macd_normalized', 'macd_signal_normalized', 'macd_histogram_normalized',
                    'bb_upper', 'bb_middle', 'bb_lower', 'bb_state', 'bb_std_normalized', 'bb_bandwidth_normalized',
                    'rolling_volatility', 'volatility_normalized']
    
    features = portfolio_with_indicators[feature_cols].values
    
    # Scale Features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(features)
    
    # 5. Load Pre-trained Generalized Model
    print("\nLoading pre-trained generalized model...")
    try:
        model = load_model('generalized_lstm_model_improved.keras')
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 6. Predict from Last Rebalance Date
    print("\n" + "="*60)
    print("       PREDICTION FROM LAST REBALANCE")
    print("="*60)
    
    # Initialize DB
    init_db()
    
    # Determine Last Rebalance Date
    # next_rebalance_date is the FUTURE rebalance date (calendar based from simulation).
    # The last one was 60 days prior.
    last_rebalance_date = next_rebalance_date - timedelta(days=60)
    
    # Ensure we have data for this date
    # Find closest available date in portfolio_df if exact date missing (e.g. weekend)
    if last_rebalance_date not in portfolio_df.index:
        # Find closest date before or on last_rebalance_date
        available_dates = portfolio_df.index[portfolio_df.index <= last_rebalance_date]
        if not available_dates.empty:
            last_rebalance_date = available_dates[-1]
        else:
            print(f"Error: Could not find valid data for rebalance date {last_rebalance_date}")
            return

    # RECALCULATE Expiry Date based on 60 TRADING DAYS (Business Days)
    # The model was trained on 60 trading days, so the prediction is valid for 60 business days later.
    from pandas.tseries.offsets import BusinessDay
    next_rebalance_date = last_rebalance_date + BusinessDay(60)

    print(f"Last Weight Change Date: {last_rebalance_date.strftime('%Y-%m-%d')}")
    print(f"Prediction Valid Until:  {next_rebalance_date.strftime('%Y-%m-%d')} (60 Trading Days)")
    # Get Portfolio Signature
    portfolio_signature = get_portfolio_signature()

    # Check if we already have this record
    existing = get_existing_prediction(last_rebalance_date.strftime('%Y-%m-%d'), portfolio_signature)
    
    if existing:
        # Check if the stored portfolio value matches the current simulation
        stored_rebalance_value = existing[3]
        
        # Allow a small floating point tolerance (e.g. 1.0)
        rebalance_value_check = portfolio_df.loc[last_rebalance_date, 'close']
        
        if abs(stored_rebalance_value - rebalance_value_check) > 1.0:
            print(f"Mismatch detected for this portfolio! Stored Value: {stored_rebalance_value:.2f}, Current Value: {rebalance_value_check:.2f}")
            print("Updating existing record...")
            
            # Delete old record for this specific portfolio
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM portfolio_predictions WHERE rebalance_date = ? AND portfolio_assets = ?", 
                           (last_rebalance_date.strftime('%Y-%m-%d'), portfolio_signature))
            conn.commit()
            conn.close()
            
            # Set existing to None to trigger regeneration
            existing = None
        else:
            print("Record for this portfolio cycle already exists. Fetching...")
            # Unpack existing record
            rebalance_value = existing[3]
            predicted_factor = existing[4]
            predicted_value = existing[5]
            
            predicted_pct = (predicted_factor - 1) * 100
            
            print(f"Portfolio Value on {last_rebalance_date.strftime('%Y-%m-%d')}: RS{rebalance_value:,.2f}")
            print(f"Predicted Return Factor: {predicted_factor:.4f}x")
            print(f"Predicted Return %:      {predicted_pct:+.2f}%")
            
            print(f"Stored Target Value:     RS{predicted_value:,.2f}")
            
            current_value = portfolio_df['close'].iloc[-1]
            current_date_str = portfolio_df.index[-1].strftime('%Y-%m-%d')
            print(f"Current Value ({current_date_str}): RS{current_value:,.2f}")

    if not existing:
        print("Generating new prediction for this cycle...")
        
        # Prepare data UP TO last_rebalance_date
        # We need 120 days ending at last_rebalance_date
        
        # Calculate indicators on the full history first (to ensure correct values)
        portfolio_with_indicators = calculate_indicators(portfolio_df)
        
        # Slice for the specific window
        mask = portfolio_with_indicators.index <= last_rebalance_date
        history_up_to_rebalance = portfolio_with_indicators.loc[mask]
        
        LOOKBACK = 120
        if len(history_up_to_rebalance) < LOOKBACK:
             print(f"Error: Insufficient data at rebalance date. Need {LOOKBACK}, got {len(history_up_to_rebalance)}")
             predicted_factor = 1.0 # Default to avoid crash
             rebalance_value = 0
             predicted_value = 0
        else:
            last_sequence_df = history_up_to_rebalance.iloc[-LOOKBACK:]
            
            # Extract features
            features = last_sequence_df[feature_cols].values
            
            # Scale
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_features = scaler.fit_transform(features)
            
            # Reshape
            input_data = scaled_features.reshape(1, LOOKBACK, 31)
            
            # Predict
            predicted_factor = model.predict(input_data, verbose=0)[0][0]
            
            # Calculate Percentage
            predicted_pct = (predicted_factor - 1) * 100
            
            # Clamp Negative Predictions to -2.5%
            if predicted_pct < -2.5:
                print(f"Note: Predicted return {predicted_pct:.2f}% capped at max negative -2.50%")
                predicted_pct = -2.5
                predicted_factor = 1 + (predicted_pct / 100)
            
            # Calculate Values
            rebalance_value = portfolio_df.loc[last_rebalance_date, 'close']
            predicted_value = rebalance_value * predicted_factor
            
            print(f"Portfolio Value on {last_rebalance_date.strftime('%Y-%m-%d')}: RS{rebalance_value:,.2f}")
            print(f"Predicted Factor:        {predicted_factor:.4f}x")
            print(f"Predicted Return %:      {predicted_pct:+.2f}%")
            
            print(f"Target Value (60d):      RS{predicted_value:,.2f}")
            
            # Calculate Metrics
            metrics = calculate_metrics_at_date(portfolio_df, benchmark_df, RISK_FREE_RATE, all_stock_data, final_weights, last_rebalance_date)
            
            # Prepare Data for DB
            db_record = {
                'rebalance_date': last_rebalance_date.strftime('%Y-%m-%d'),
                'expiry_date': next_rebalance_date.strftime('%Y-%m-%d'),
                'portfolio_value_at_rebalance': rebalance_value,
                'predicted_return_factor': float(predicted_factor),
                'predicted_value_at_expiry': float(predicted_value),
                'weights': dict(zip([CHOSEN_PORTFOLIO[k] for k in CHOSEN_PORTFOLIO], final_weights)),
                'correlations': metrics.get('correlation_matrix', {}),
                'risk_metrics': {k: v for k, v in metrics.items() if k != 'correlation_matrix'},
                'portfolio_assets': portfolio_signature
            }
            
            print(f"Saving new prediction record for {last_rebalance_date.strftime('%Y-%m-%d')}...")
            save_prediction_to_db(db_record)
            
            current_date = portfolio_df.index[-1]
            current_value = portfolio_df['close'].iloc[-1]
            print(f"Current Value ({current_date.strftime('%Y-%m-%d')}): RS{current_value:,.2f}")

            if current_date < next_rebalance_date:
                 print(f"Note: Current date ({current_date.strftime('%Y-%m-%d')}) < Expiry ({next_rebalance_date.strftime('%Y-%m-%d')}). Prediction is active.")


    # 8. Detailed Report (Standard - based on current data)
    if not benchmark_df.empty:
        generate_detailed_report(portfolio_df, benchmark_df, RISK_FREE_RATE, all_stock_data, final_weights)
    else:
        print("Skipping detailed report due to missing benchmark data.")

    # 9. Optimized vs Equal-Weighted Comparison
    if not benchmark_df.empty:
        print("\n" + "="*60)
        print("    OPTIMIZED VS EQUAL-WEIGHTED PORTFOLIO COMPARISON")
        print("="*60)

        eq_common_index = equal_weight_df.index.intersection(benchmark_df.index)
        eq_returns = equal_weight_df.loc[eq_common_index, 'returns']
        opt_common_index = portfolio_df.index.intersection(benchmark_df.index)
        opt_returns = portfolio_df.loc[opt_common_index, 'returns']

        opt_cagr = (1 + opt_returns.mean())**252 - 1
        opt_sharpe = calculate_sharpe_ratio(opt_returns, RISK_FREE_RATE)
        opt_dd = calculate_max_drawdown(opt_returns)

        eq_cagr = (1 + eq_returns.mean())**252 - 1
        eq_sharpe = calculate_sharpe_ratio(eq_returns, RISK_FREE_RATE)
        eq_dd = calculate_max_drawdown(eq_returns)

        print(f"Optimized Portfolio:")
        print(f"  CAGR:                 {opt_cagr*100:.2f}%")
        print(f"  Sharpe Ratio:         {opt_sharpe:.2f}")
        print(f"  Max Drawdown:         {opt_dd*100:.2f}%")

        print(f"\nEqual-Weighted Portfolio:")
        print(f"  CAGR:                 {eq_cagr*100:.2f}%")
        print(f"  Sharpe Ratio:         {eq_sharpe:.2f}")
        print(f"  Max Drawdown:         {eq_dd*100:.2f}%")

        print("\nDiscussion on Risk Metrics and Optimization Decisions:")
        print("Risk metrics such as volatility, Value at Risk (VaR), Expected Shortfall (CVaR), maximum drawdown, and Sharpe/Sortino ratios play a crucial role in portfolio optimization. Higher volatility and drawdowns indicate greater risk, which optimization algorithms like Hierarchical Risk Parity (HRP) aim to minimize by allocating more weight to low-risk assets during favorable correlation windows. The choice of rebalancing frequency (e.g., 60 days) balances transaction costs with the need to adapt to changing market conditions, while equal-weighted portfolios provide a naive benchmark that often underperforms optimized strategies by ignoring asset volatilities and correlations.")

    # 10. Time Series Forecast Plot
    print("="*60)
    
    # --- RAG ANALYSIS INTEGRATION ---
    try:
        print("\n" + "="*60)
        print("       RAG-ENHANCED PREDICTION ANALYSIS")
        print("="*60)
        print("Loading RAG system...")
        
        # Load RAG system
        if load_rag_system():
            # Get current indicators from portfolio
            current_indicators = {
                "rsi": float(portfolio_with_indicators['rsi'].iloc[-1]),
                "macd": float(portfolio_with_indicators['macd'].iloc[-1]),
                "volatility": float(portfolio_with_indicators['rolling_volatility'].iloc[-1]),
                "alligator_jaw": float(portfolio_with_indicators['alligator_jaw'].iloc[-1]),
                "alligator_teeth": float(portfolio_with_indicators['alligator_teeth'].iloc[-1]),
                "alligator_lips": float(portfolio_with_indicators['alligator_lips'].iloc[-1])
            }
            
            # Find similar scenarios with DYNAMIC k selection
            print(f"Searching 297,456 historical scenarios for similar conditions...")
            
            # First, get a large pool to analyze distances
            initial_pool = 100  # Get top 100 to analyze
            similar_scenarios_pool = find_similar_scenarios(current_indicators, k=initial_pool)
            
            # Calculate distances for the pool (using vector similarity)
            query_embedding = [
                current_indicators["rsi"] / 100.0,
                np.tanh(current_indicators["macd"] / 10000.0),
                current_indicators["volatility"],
                np.tanh(current_indicators["alligator_jaw"] / 1000000.0),
                np.tanh(current_indicators["alligator_teeth"] / 1000000.0),
                np.tanh(current_indicators["alligator_lips"] / 1000000.0),
            ]
            
            distances = []
            for scen in similar_scenarios_pool:
                scen_embedding = [
                    scen['indicators']['rsi'] / 100.0,
                    np.tanh(scen['indicators']['macd'] / 10000.0),
                    scen['indicators']['volatility'],
                    np.tanh(scen['indicators']['alligator_jaw'] / 1000000.0),
                    np.tanh(scen['indicators']['alligator_teeth'] / 1000000.0),
                    np.tanh(scen['indicators']['alligator_lips'] / 1000000.0),
                ]
                # Cosine distance
                dot_product = np.dot(query_embedding, scen_embedding)
                norm_query = np.linalg.norm(query_embedding)
                norm_scen = np.linalg.norm(scen_embedding)
                distance = 1 - (dot_product / (norm_query * norm_scen))
                distances.append(distance)
            
            # Dynamic k selection based on distance threshold
            # Strategy: Include scenarios within 1.5x the median distance of top 10
            top_10_median_distance = np.median(distances[:10])
            distance_threshold = top_10_median_distance * 1.5
            
            # Count how many scenarios are within threshold
            dynamic_k = sum(1 for d in distances if d <= distance_threshold)
            
            # Ensure k is within reasonable bounds
            dynamic_k = max(10, min(dynamic_k, 50))  # Between 10 and 50
            
            # Get final scenarios with dynamic k
            similar_scenarios = similar_scenarios_pool[:dynamic_k]
            
            print(f"[OK] Dynamically selected {dynamic_k} scenarios (distance threshold: {distance_threshold:.4f})")
            print(f"     Top 10 median distance: {top_10_median_distance:.4f}")
            print(f"     Most similar distance: {distances[0]:.4f}, Least similar: {distances[dynamic_k-1]:.4f}")
            
            # Calculate historical statistics
            outcomes = [s['target_return_60d'] for s in similar_scenarios]
            avg_return = np.mean(outcomes)
            std_return = np.std(outcomes)
            min_return = np.min(outcomes)
            max_return = np.max(outcomes)
            median_return = np.median(outcomes)
            positive_count = sum(1 for r in outcomes if r > 0)
            success_rate = positive_count / len(outcomes) * 100
            
            # Get TCN prediction percentage
            tcn_pct = (predicted_factor - 1) * 100
            
            # Display RAG statistics
            print("\n" + "-"*60)
            print(f"HISTORICAL CONTEXT ({dynamic_k} Similar Scenarios):")
            print("-"*60)
            print(f"TCN Prediction:           {tcn_pct:+.2f}%")
            print(f"Historical Average:       {avg_return:.2%}")
            print(f"Historical Median:        {median_return:.2%}")
            print(f"Historical Range:         {min_return:.2%} to {max_return:.2%}")
            print(f"Standard Deviation:       {std_return:.2%}")
            print(f"Success Rate:             {success_rate:.1f}% ({positive_count}/{dynamic_k} positive)")
            
            # Confidence interval
            conf_low = avg_return - 2*std_return
            conf_high = avg_return + 2*std_return
            print(f"95% Confidence Interval:  {conf_low:.2%} to {conf_high:.2%}")
            
            # Alignment check
            tcn_return = tcn_pct / 100
            if abs(tcn_return - avg_return) < std_return:
                alignment = "[ALIGNED]"
                alignment_desc = "TCN prediction is within 1 std dev of historical average"
            elif abs(tcn_return - avg_return) < 2*std_return:
                alignment = "[MODERATE]"
                alignment_desc = "TCN prediction is within 2 std dev of historical average"
            else:
                alignment = "[DIVERGENT]"
                alignment_desc = "TCN prediction differs significantly from historical average"
            
            print(f"TCN vs History:           {alignment}")
            print(f"                          {alignment_desc}")
            
            # Show top 3 similar scenarios
            print("\n" + "-"*60)
            print("TOP 3 MOST SIMILAR HISTORICAL SCENARIOS:")
            print("-"*60)
            for i, scen in enumerate(similar_scenarios[:3], 1):
                print(f"\n{i}. Date: {scen['date']}")
                print(f"   RSI: {scen['indicators']['rsi']:.1f}, MACD: {scen['indicators']['macd']:.2f}, Vol: {scen['indicators']['volatility']:.1%}")
                print(f"   Actual 60-day Return: {scen['target_return_pct']}")
                print(f"   Narrative: {scen['narrative'][:100]}...")
            
            # Gemini AI Analysis (optional, with fallback)
            try:
                print("\n" + "-"*60)
                print("AI DEEP ANALYSIS:")
                print("-"*60)
                
                # Uses GEMINI_API_KEY from configuration section at top of file
                genai.configure(api_key=GEMINI_API_KEY)
                
                # Use FREE TIER models only (to avoid quota issues)
                model_options = [
                    'gemini-2.0-flash-exp',          # Free experimental (best quality)
                    'gemini-flash-latest',           # Free latest flash
                    'gemini-2.5-flash',              # Free 2.5 flash
                    'gemini-2.0-flash',              # Free 2.0 flash
                    'gemini-1.5-flash',              # Free 1.5 flash (fallback)
                ]
                
                gemini_model = None
                last_error = None
                for model_name in model_options:
                    try:
                        gemini_model = genai.GenerativeModel(model_name)
                        print(f"Using FREE model: {model_name}")
                        break
                    except Exception as model_err:
                        last_error = str(model_err)
                        continue
                
                if not gemini_model:
                    raise Exception(f"No free Gemini models available. Last error: {last_error}")
                
                # Prepare detailed scenario analysis
                scenario_details = ""
                for i, scen in enumerate(similar_scenarios[:10], 1):
                    scenario_details += f"\n{i}. {scen['date']}: "
                    scenario_details += f"RSI={scen['indicators']['rsi']:.1f}, "
                    scenario_details += f"MACD={scen['indicators']['macd']:.0f}, "
                    scenario_details += f"Vol={scen['indicators']['volatility']:.1%} "
                    scenario_details += f"-> Return: {scen['target_return_pct']}"
                
                # Calculate additional insights
                positive_scenarios = [s for s in similar_scenarios if s['target_return_60d'] > 0]
                negative_scenarios = [s for s in similar_scenarios if s['target_return_60d'] <= 0]
                
                # Volatility clustering
                high_vol_scenarios = [s for s in similar_scenarios if s['indicators']['volatility'] > 0.15]
                low_vol_scenarios = [s for s in similar_scenarios if s['indicators']['volatility'] <= 0.15]
                
                # RSI regime analysis
                overbought_scenarios = [s for s in similar_scenarios if s['indicators']['rsi'] > 70]
                neutral_scenarios = [s for s in similar_scenarios if 30 <= s['indicators']['rsi'] <= 70]
                oversold_scenarios = [s for s in similar_scenarios if s['indicators']['rsi'] < 30]
                
                # Enhanced prompt with structured analysis
                prompt = f"""You are an expert quantitative analyst with deep knowledge of portfolio management, technical analysis, and market regimes.

**CURRENT SITUATION:**
- TCN Model Prediction: {tcn_pct:+.2f}%
- Prediction Horizon: 60 days
- Current Date: {portfolio_df.index[-1].strftime('%Y-%m-%d')}

**CURRENT MARKET INDICATORS:**
- RSI: {current_indicators['rsi']:.1f} ({'Overbought' if current_indicators['rsi'] > 70 else 'Oversold' if current_indicators['rsi'] < 30 else 'Neutral'})
- MACD: {current_indicators['macd']:.2f}
- Volatility: {current_indicators['volatility']:.1%}
- Alligator: Jaw={current_indicators['alligator_jaw']:.0f}, Teeth={current_indicators['alligator_teeth']:.0f}, Lips={current_indicators['alligator_lips']:.0f}

**HISTORICAL EVIDENCE ({dynamic_k} most similar scenarios from 297,456 total):**

Statistical Summary:
- Average Return: {avg_return:.2%}
- Median Return: {median_return:.2%}
- Std Deviation: {std_return:.2%}
- Range: {min_return:.2%} to {max_return:.2%}
- Success Rate: {success_rate:.1f}% ({positive_count}/{dynamic_k} positive)
- 95% Confidence Interval: {conf_low:.2%} to {conf_high:.2%}

Pattern Analysis:
- Positive Outcomes: {len(positive_scenarios)} scenarios (avg: {np.mean([s['target_return_60d'] for s in positive_scenarios]) if positive_scenarios else 0:.2%})
- Negative Outcomes: {len(negative_scenarios)} scenarios (avg: {np.mean([s['target_return_60d'] for s in negative_scenarios]) if negative_scenarios else 0:.2%})
- High Volatility (>15%): {len(high_vol_scenarios)} scenarios (avg return: {np.mean([s['target_return_60d'] for s in high_vol_scenarios]) if high_vol_scenarios else 0:.2%})
- Low Volatility (<=15%): {len(low_vol_scenarios)} scenarios (avg return: {np.mean([s['target_return_60d'] for s in low_vol_scenarios]) if low_vol_scenarios else 0:.2%})

RSI Regime Analysis:
- Overbought (>70): {len(overbought_scenarios)} scenarios (avg: {np.mean([s['target_return_60d'] for s in overbought_scenarios]) if overbought_scenarios else 0:.2%})
- Neutral (30-70): {len(neutral_scenarios)} scenarios (avg: {np.mean([s['target_return_60d'] for s in neutral_scenarios]) if neutral_scenarios else 0:.2%})
- Oversold (<30): {len(oversold_scenarios)} scenarios (avg: {np.mean([s['target_return_60d'] for s in oversold_scenarios]) if oversold_scenarios else 0:.2%})

Top 10 Similar Historical Cases:{scenario_details}

**YOUR TASK:**
Act as a Quantitative Risk Manager. Analyze the data and provide a strictly mathematical trading report. Avoid generic text. Focus on probabilities, expected values, and specific levels.

**1. STATISTICAL EXPECTATIONS:**
   - **Expected Value (EV):** Calculate the probability-weighted return. Formula: (Avg_Pos_Return * %Win) + (Avg_Neg_Return * %Loss).
   - **Kelly Criterion:** Estimate optimal fraction to bet (using Success Rate and Win/Loss ratio).
   - **Realism Check:** Compare TCN ({tcn_pct:+.2f}%) vs EV. Is TCN optimistic or conservative?

**2. TARGETS & LIMITS (Based on Data):**
   - **Primary Target:** {tcn_pct:+.2f}% (TCN Prediction)
   - **Secondary Target:** {max_return:.2f}% (Historical Max)
   - **Stop Loss Level:** Suggest a specific price/percentage based on the 2-std-dev lower bound ({conf_low:.2f}%).
   - **Invalidation Level:** At what return % does the thesis fail? (e.g., dropping below historical median).

**3. RISK OBSERVATION:**
   - **Downside Probability:** {len(negative_scenarios)/dynamic_k*100:.1f}%
   - **Tail Risk:** Probability of return < -5% based on distribution.
   - **Volatility Impact:** If volatility expands from {current_indicators['volatility']:.1%} to >15%, what is the historical impact on returns?

**4. TRADE SETUP (Mathematical):**
   - **Risk/Reward Ratio:** Calculate (Target - 0) / (0 - Stop Loss). Note: Entry is 0%. Use absolute values for denominator.
   - **Position Sizing:** Suggest % allocation based on Kelly and Conviction.
   - **Conclusion:** MATHEMATICALLY FAVORABLE or UNFAVORABLE?

**5. FINAL EXECUTIVE SUMMARY:**
   - Write a 2-3 sentence conclusion summarizing the entire analysis.
   - clearly state the final recommendation (Buy/Sell/Hold) and the key reason why.


**OUTPUT FORMAT:**
- DO NOT use markdown formatting (no asterisks *, no bold **).
- Use simple indentation for hierarchy.
- Use UPPERCASE for section headers.
- Keep descriptions concise and data-heavy.
"""
                
                response = gemini_model.generate_content(prompt)
                
                # Clean up any remaining markdown just in case
                clean_text = response.text.replace('**', '').replace('* ', '- ')
                print(clean_text)
            except Exception as e:
                print(f"AI analysis unavailable (Error: {str(e)[:100]})")
                print("\nUsing COMPREHENSIVE STATISTICAL ANALYSIS:\n")
                
                # Generate comprehensive fallback analysis
                print("1. MARKET REGIME IDENTIFICATION:")
                if current_indicators['rsi'] > 70:
                    print(f"   - Regime: OVERBOUGHT (RSI {current_indicators['rsi']:.1f})")
                    print(f"   - {len(overbought_scenarios)}/{dynamic_k} similar scenarios were overbought")
                    if overbought_scenarios:
                        avg_ob = np.mean([s['target_return_60d'] for s in overbought_scenarios])
                        print(f"   - Overbought scenarios averaged: {avg_ob:.2%}")
                elif current_indicators['rsi'] < 30:
                    print(f"   - Regime: OVERSOLD (RSI {current_indicators['rsi']:.1f})")
                    print(f"   - {len(oversold_scenarios)}/{dynamic_k} similar scenarios were oversold")
                    if oversold_scenarios:
                        avg_os = np.mean([s['target_return_60d'] for s in oversold_scenarios])
                        print(f"   - Oversold scenarios averaged: {avg_os:.2%}")
                else:
                    print(f"   - Regime: NEUTRAL (RSI {current_indicators['rsi']:.1f})")
                    print(f"   - {len(neutral_scenarios)}/{dynamic_k} similar scenarios were neutral")
                
                if current_indicators['volatility'] > 0.15:
                    print(f"   - Volatility: HIGH ({current_indicators['volatility']:.1%})")
                    if high_vol_scenarios:
                        avg_hv = np.mean([s['target_return_60d'] for s in high_vol_scenarios])
                        print(f"   - High volatility scenarios averaged: {avg_hv:.2%}")
                else:
                    print(f"   - Volatility: NORMAL ({current_indicators['volatility']:.1%})")
                    if low_vol_scenarios:
                        avg_lv = np.mean([s['target_return_60d'] for s in low_vol_scenarios])
                        print(f"   - Low volatility scenarios averaged: {avg_lv:.2%}")
                
                print("\n2. TCN PREDICTION VALIDATION:")
                print(f"   - TCN Prediction: {tcn_pct:+.2f}%")
                print(f"   - Historical Average: {avg_return:.2%}")
                print(f"   - Difference: {abs(tcn_pct/100 - avg_return):.2%}")
                print(f"   - Alignment: {alignment}")
                if alignment == "[ALIGNED]":
                    print(f"   - Confidence: HIGH - Within 1 standard deviation")
                elif alignment == "[MODERATE]":
                    print(f"   - Confidence: MODERATE - Within 2 standard deviations")
                else:
                    print(f"   - Confidence: LOW - Beyond 2 standard deviations")
                
                print("\n3. PATTERN RECOGNITION:")
                print(f"   - Positive Outcomes: {len(positive_scenarios)}/{dynamic_k} ({len(positive_scenarios)/dynamic_k*100:.1f}%)")
                if positive_scenarios:
                    avg_pos = np.mean([s['target_return_60d'] for s in positive_scenarios])
                    print(f"   - Average positive return: {avg_pos:.2%}")
                print(f"   - Negative Outcomes: {len(negative_scenarios)}/{dynamic_k} ({len(negative_scenarios)/dynamic_k*100:.1f}%)")
                if negative_scenarios:
                    avg_neg = np.mean([s['target_return_60d'] for s in negative_scenarios])
                    print(f"   - Average negative return: {avg_neg:.2%}")
                print(f"   - Most similar case: {similar_scenarios[0]['date']} ({similar_scenarios[0]['target_return_pct']})")
                
                print("\n4. RISK FACTORS:")
                print(f"   - Downside Risk: {len(negative_scenarios)}/{dynamic_k} scenarios were negative")
                print(f"   - Worst Case: {min_return:.2%}")
                print(f"   - 95% CI Lower Bound: {conf_low:.2%}")
                tail_risk = sum(1 for s in similar_scenarios if s['target_return_60d'] < conf_low) / dynamic_k * 100
                print(f"   - Tail Event Probability: {tail_risk:.1f}% (beyond 2 std dev)")
                print(f"   - Volatility Risk: Current {current_indicators['volatility']:.1%} vs Avg {np.mean([s['indicators']['volatility'] for s in similar_scenarios]):.1%}")
                
                print("\n5. OPPORTUNITY ASSESSMENT:")
                upside = max_return - (tcn_pct/100)
                downside = (tcn_pct/100) - min_return
                if downside > 0:
                    risk_reward = upside / downside
                    print(f"   - Risk-Reward Ratio: {risk_reward:.1f}:1")
                print(f"   - Upside Potential: {max_return:.2%} (best historical case)")
                print(f"   - Downside Risk: {min_return:.2%} (worst historical case)")
                print(f"   - Success Rate: {success_rate:.1f}%")
                if success_rate >= 70:
                    print(f"   - Conviction: HIGH")
                elif success_rate >= 50:
                    print(f"   - Conviction: MODERATE")
                else:
                    print(f"   - Conviction: LOW")
                
                print("\n6. ACTIONABLE RECOMMENDATIONS:")
                if alignment == "[ALIGNED]" and success_rate >= 60:
                    print(f"   - Action: HOLD / INCREASE exposure")
                    print(f"   - Position Sizing: Moderate to Aggressive")
                elif alignment == "[MODERATE]" or (50 <= success_rate < 60):
                    print(f"   - Action: HOLD current position")
                    print(f"   - Position Sizing: Conservative to Moderate")
                else:
                    print(f"   - Action: REDUCE / HEDGE exposure")
                    print(f"   - Position Sizing: Conservative only")
                
                stop_loss_pct = min_return * 1.2  # 20% beyond worst case
                take_profit_pct = avg_return * 1.5  # 50% beyond average
                print(f"   - Stop Loss: {stop_loss_pct:.2%} from entry")
                print(f"   - Take Profit: {take_profit_pct:.2%} from entry")
                print(f"   - Monitor: RSI, Volatility, and 30-day checkpoint")
            
            
            print("="*60)
        else:
            print("RAG system not available. Run rag_api_improved.py first to build vector DB.")
            print("="*60)
            
    except Exception as e:
        print(f"\nRAG analysis unavailable: {e}")
        print("Continuing without RAG analysis...")
        print("="*60)
    
    print("generating portfolio value forecast plot...")

    try:
        import matplotlib.pyplot as plt

        # Create date range for forecast (60 days from REBALANCE date)
        forecast_start_date = last_rebalance_date
        # Use BusinessDay to skip weekends (approximate market days)
        forecast_dates = pd.bdate_range(start=forecast_start_date, periods=61)[1:]  # Next 60 business days

        
        # Create forecast values using linear projection of predicted factor
        # Start from the value AT REBALANCE
        start_value = rebalance_value
        daily_factor = predicted_factor ** (1/60)  # Approximate daily factor
        forecast_values = [start_value * (daily_factor ** i) for i in range(1, 61)]
        
        # Create RAG historical range boundaries if available
        rag_min_forecast = None
        rag_max_forecast = None
        if 'min_return' in locals() and 'max_return' in locals():
            # Calculate min/max forecast based on historical range
            min_factor = 1 + min_return
            max_factor = 1 + max_return
            daily_min_factor = min_factor ** (1/60)
            daily_max_factor = max_factor ** (1/60)
            rag_min_forecast = [start_value * (daily_min_factor ** i) for i in range(1, 61)]
            rag_max_forecast = [start_value * (daily_max_factor ** i) for i in range(1, 61)]
        
        plt.figure(figsize=(16, 10))
        
        # Plot historical portfolio values (show only last 6 months for clarity)
        # Ensure we show up to the CURRENT date (last fetched data)
        hist_end = portfolio_df.index[-1]
        hist_start = hist_end - pd.DateOffset(months=6)
        hist_data = portfolio_df.loc[hist_start:hist_end]
        
        plt.plot(hist_data.index, hist_data['close'], 'b-', linewidth=2, label='Historical Portfolio Value')
        
        # Plot RAG historical range (if available)
        if rag_min_forecast and rag_max_forecast:
            plt.fill_between(forecast_dates, rag_min_forecast, rag_max_forecast, 
                           alpha=0.2, color='yellow', 
                           label=f'RAG Historical Range ({min_return:.1%} to {max_return:.1%})')
            plt.plot(forecast_dates, rag_min_forecast, 'y--', linewidth=1, alpha=0.5)
            plt.plot(forecast_dates, rag_max_forecast, 'y--', linewidth=1, alpha=0.5)
        
        # Plot TCN forecast (starting from rebalance date)
        plt.plot(forecast_dates, forecast_values, 'r--', linewidth=3, label=f'TCN Forecast ({(predicted_factor-1)*100:+.2f}%)', alpha=0.8)
        
        # Highlight key points: Rebalance Date and Forecast End
        plt.scatter([forecast_start_date, forecast_dates[-1]], [start_value, forecast_values[-1]], 
                   color=['orange', 'red'], s=100, zorder=5)
        
        # Add vertical line at Rebalance Date
        plt.axvline(x=forecast_start_date, color='orange', linestyle='--', alpha=0.7, label='Rebalance Date')
        
        # Add vertical line at Current Date (if different)
        if hist_end != forecast_start_date:
             plt.axvline(x=hist_end, color='gray', linestyle=':', alpha=0.7, label='Current Date')

        # Format plot
        plt.title(f'Portfolio Value: Historical vs Forecast with RAG Boundaries (from {forecast_start_date.strftime("%Y-%m-%d")})', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Portfolio Value (₹)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left', fontsize=11)
        
        # Format y-axis with rupee symbol
        import matplotlib.ticker as mticker
        def rupees(x, pos):
            return f'₹{x:,.0f}'
        plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(rupees))
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plot_filename = f'portfolio_forecast_{hist_end.strftime("%Y%m%d")}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"time series forecast plot saved as '{plot_filename}'")
        
        # Also save equal-weighted comparison if data exists
        if 'equal_weight_df' in locals():
            plt.figure(figsize=(16, 10))
            
            # Plot both portfolios historical data
            Portfolio_hist = portfolio_df.loc[hist_start:hist_end]
            Equal_hist = equal_weight_df.loc[hist_start:hist_end]
            
            plt.plot(Portfolio_hist.index, Portfolio_hist['close'], 'b-', linewidth=2, label='Optimized Portfolio')
            plt.plot(Equal_hist.index, Equal_hist['close'], 'g--', linewidth=2, label='Equal-Weighted Portfolio')
            plt.plot(forecast_dates, forecast_values, 'r:', linewidth=3, label='Optimized Forecast', alpha=0.7)
            
            plt.scatter([forecast_start_date], [start_value], color='orange', s=100, zorder=5)
            plt.axvline(x=forecast_start_date, color='orange', linestyle='--', alpha=0.7, label='Rebalance Date')
            
            plt.title('Portfolio Comparison: Optimized vs Equal-Weighted', fontsize=16, fontweight='bold')
            plt.xlabel('Date', fontsize=14)
            plt.ylabel('Portfolio Value (₹)', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper left', fontsize=12)
            plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(rupees))
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            comp_plot_filename = f'portfolio_comparison_{hist_end.strftime("%Y%m%d")}.png'
            plt.savefig(comp_plot_filename, dpi=300, bbox_inches='tight')
            print(f"portfolio comparison plot saved as '{comp_plot_filename}'")

        plt.show()

    except ImportError:
        print("matplotlib not available, skipping plot generation")
    except Exception as e:
        print(f"error generating plot: {e}")

    print("="*60)

if __name__ == "__main__":
    # Set seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    main()
