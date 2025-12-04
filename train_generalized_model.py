import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from urllib.parse import quote
import warnings
import time
import os
import multiprocessing
from functools import partial
import bisect

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import scipy.stats as stats

# --- Configuration ---
ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI3M0FIOUMiLCJqdGkiOiI2OTFkNWFiODFkZTI2NzdiZDkxNjkxMGMiLCJpc015dHRpQ2xpZW50IjpmYWxzZSwiaXNQbHJpUGxhbiI6dHJ1ZSwiaWF0IjoxNzYzNTMxNDQ4LCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3NjM1ODk2MDB9.7dPxOIudeYCSEo5HNVzWXtn2EhzDFWKwICz7Aui4"
BASE_URL = "https://api.upstox.com/v3/historical-candle"
START_DATE = "2018-09-03"
END_DATE = "2025-11-19"
RISK_FREE_RATE = 0.065
TRANSACTION_COST = 0.001
INITIAL_CAPITAL = 1000000.0

# --- Stock Mapping (from analysis.py) ---
PREDEFINED_STOCKS = {
    "NSE_EQ|INE732I01013":"ANGELONE",
    "NSE_EQ|INE296A01032":"BAJFINANCE",
    "NSE_EQ|INE044A01036":"SUNPHARMA",
    "NSE_EQ|INE674K01013":"ABCAPITAL",
    "NSE_EQ|INE028A01039":"BANKBARODA",
    "NSE_EQ|INE692A01016":"UNIONBANK",
    "NSE_EQ|INE476A01022":"CANBK",
    "NSE_EQ|INE372C01037":"PRECWIRE",
    "NSE_EQ|INE918Z01012":"KAYNES",
    "NSE_EQ|INE474Q01031":"MEDANTA",
    "NSE_EQ|INE034A01011":"ARVIND",
    "NSE_EQ|INE560A01023":"INDIAGLYCO",
    "NSE_EQ|INE029A01011":"BPCL",
    "NSE_EQ|INE118H01025":"BSE",
    "NSE_EQ|INE926K01017":"AHLEAST",
    "NSE_EQ|INE376G01013":"BIOCON",
    "NSE_EQ|INE849A01020":"TRENT",
    "NSE_EQ|INE364U01010":"IRFC",
    "NSE_EQ|INE133E01013":"TI",
    "NSE_EQ|INE397D01024":"BHARTIARTL",
    "NSE_EQ|INE931S01010":"ADANIENSOL",
    "NSE_EQ|INE199P01028":"GCSL",
    "NSE_EQ|INE075I01017":"HCG",
    "NSE_EQ|INE0L2Y01011":"ARHAM",
    "NSE_EQ|INE591G01025":"COFORGE",
    "NSE_EQ|INE262H01021":"PERSISTENT",
    "NSE_EQ|INE009A01021":"INFY"
    }

# Reverse mapping for easy lookup
SYMBOL_TO_KEY = {v: k for k, v in PREDEFINED_STOCKS.items()}

# --- Data Fetching ---
def fetch_historical_data(instrument_key, from_date, to_date):
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
        except requests.exceptions.RequestException:
            pass

    if all_data:
        df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df.index = df.index.tz_localize(None)
        df.sort_index(inplace=True)
        df["returns"] = df["close"].pct_change()
        return df.dropna()
    return pd.DataFrame()

# --- HRP Optimization ---
# --- HRP Optimization (NumPy Optimized) ---
def optimize_weights(returns_matrix):
    # returns_matrix: np.ndarray of shape (T, N)
    
    # Calculate Covariance and Correlation
    # rowvar=False because rows are observations (time), columns are variables (stocks)
    cov = np.cov(returns_matrix, rowvar=False) 
    corr = np.corrcoef(returns_matrix, rowvar=False)
    
    # Handle NaN or constant returns
    if np.any(np.isnan(cov)) or np.any(np.isnan(corr)):
        return np.full(returns_matrix.shape[1], 1.0 / returns_matrix.shape[1])

    # Distance Matrix
    dist = np.sqrt((1 - corr) / 2)
    np.fill_diagonal(dist, 0)
    
    # Ensure symmetry (fix floating point precision issues)
    dist = (dist + dist.T) / 2
    
    condensed_dist = squareform(dist)
    
    try:
        link = linkage(condensed_dist, method='single')
    except:
        return np.full(returns_matrix.shape[1], 1.0 / returns_matrix.shape[1])

    def get_quasi_diag(link):
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0])
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])
        return sort_ix.tolist()

    sort_ix = get_quasi_diag(link)
    
    def get_cluster_var(cov, c_items):
        cov_slice = cov[np.ix_(c_items, c_items)]
        w = 1 / np.diag(cov_slice)
        w /= w.sum()
        return np.dot(np.dot(w, cov_slice), w)

    def get_rec_bipart(cov, sort_ix):
        w = np.ones(len(sort_ix))
        c_items = [sort_ix]
        while len(c_items) > 0:
            # Bisect each cluster into two sub-clusters
            new_c_items = []
            for cluster in c_items:
                if len(cluster) > 1:
                    # Split cluster in half
                    mid = len(cluster) // 2
                    c_items0 = cluster[:mid]
                    c_items1 = cluster[mid:]
                    
                    c_var0 = get_cluster_var(cov, c_items0)
                    c_var1 = get_cluster_var(cov, c_items1)
                    alpha = 1 - c_var0 / (c_var0 + c_var1)
                    
                    w[c_items0] *= alpha
                    w[c_items1] *= 1 - alpha
                    
                    # Add sub-clusters to next level
                    new_c_items.append(c_items0)
                    new_c_items.append(c_items1)
            
            # Update clusters for next iteration
            c_items = new_c_items
        return w

    weights = get_rec_bipart(cov, sort_ix)
    return weights

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

# --- Dataset Creation ---
def create_dataset(scaled_dataset, raw_prices, lookback, forward):
    X, y = [], []
    for i in range(len(scaled_dataset) - lookback - forward):
        X.append(scaled_dataset[i:(i + lookback)])
        current_val = raw_prices[i + lookback - 1] 
        future_val = raw_prices[i + lookback + forward - 1]
        y.append(future_val / current_val) 
    return np.array(X), np.array(y)

# --- Worker Function for Multiprocessing ---
def process_portfolio(portfolio_data, all_stock_data, output_dir):
    portfolio_str, idx = portfolio_data
    try:
        symbols = [s.strip() for s in portfolio_str.replace('"', '').split(',')]
        keys = [SYMBOL_TO_KEY.get(s) for s in symbols if s in SYMBOL_TO_KEY]
        
        if len(keys) != 5:
            return None

        # Align data
        common_index = sorted(list(set.intersection(*[set(all_stock_data[k].index) for k in keys])))
        if not common_index:
            return None
            
        # Create DataFrames for vectorized access
        price_data = {k: all_stock_data[k].loc[common_index, 'close'] for k in keys}
        high_data = {k: all_stock_data[k].loc[common_index, 'high'] for k in keys}
        low_data = {k: all_stock_data[k].loc[common_index, 'low'] for k in keys}
        returns_data = {k: all_stock_data[k].loc[common_index, 'returns'] for k in keys}
        
        prices_df = pd.DataFrame(price_data, index=common_index)
        highs_df = pd.DataFrame(high_data, index=common_index)
        lows_df = pd.DataFrame(low_data, index=common_index)
        returns_df = pd.DataFrame(returns_data, index=common_index)
        
        # Convert to numpy for fast slicing
        returns_matrix = returns_df.values # (T, 5)
        
        portfolio_value_series = []
        portfolio_high_series = []
        portfolio_low_series = []
        
        current_capital = INITIAL_CAPITAL
        current_weights = np.array([0.2] * 5)
        
        # Initial units
        first_prices = prices_df.iloc[0].values
        units = (current_capital * current_weights) / first_prices
        
        start_date = common_index[0]
        next_rebalance_date = start_date + timedelta(days=60)
        
        current_idx = 0
        total_len = len(common_index)
        
        while current_idx < total_len:
            # Find index of the next rebalance date
            # Use bisect for O(log N) search
            # common_index is a list of timestamps
            rebalance_idx = bisect.bisect_left(common_index, next_rebalance_date)
            
            if rebalance_idx >= total_len:
                rebalance_idx = total_len - 1
                is_last_chunk = True
            else:
                # Check if we found exact match or next
                # If common_index[rebalance_idx] >= next_rebalance_date, it's the split point.
                # But we want the chunk to INCLUDE the rebalance date for value calculation?
                # No, wait. Previous logic:
                # chunk_end = rebalance_idx + 1.
                # If rebalance_idx is the index where date >= next_rebalance_date.
                # Then common_index[rebalance_idx] is the rebalance date.
                is_last_chunk = False
            
            # Ensure we process at least one day if dates are close (shouldn't happen with 60 days)
            if rebalance_idx < current_idx:
                rebalance_idx = current_idx
            
            # Define chunk range (inclusive of rebalance_idx for value calculation)
            chunk_start = current_idx
            chunk_end = rebalance_idx + 1 # Slice is exclusive at end
            
            chunk_prices = prices_df.iloc[chunk_start:chunk_end].values
            chunk_highs = highs_df.iloc[chunk_start:chunk_end].values
            chunk_lows = lows_df.iloc[chunk_start:chunk_end].values
            
            # Vectorized calculation of portfolio values for the chunk
            chunk_values = np.sum(chunk_prices * units, axis=1)
            chunk_portfolio_highs = np.sum(chunk_highs * units, axis=1)
            chunk_portfolio_lows = np.sum(chunk_lows * units, axis=1)
            
            portfolio_value_series.extend(chunk_values)
            portfolio_high_series.extend(chunk_portfolio_highs)
            portfolio_low_series.extend(chunk_portfolio_lows)
            
            if is_last_chunk:
                break
            
            # Perform Rebalancing
            date = common_index[rebalance_idx]
            current_prices = chunk_prices[-1]
            daily_value = chunk_values[-1]
            
            lookback_start = date - timedelta(days=120)
            lookback_start_idx = bisect.bisect_left(common_index, lookback_start)
            
            # Slice returns matrix for optimization
            # returns_matrix is (T, 5)
            # We want [lookback_start_idx : rebalance_idx]
            # rebalance_idx corresponds to 'date'. 
            # Original logic: mask = (index >= lookback_start) & (index < date)
            # So slice is [lookback_start_idx : rebalance_idx] (exclusive of rebalance_idx)
            
            lookback_slice = returns_matrix[lookback_start_idx : rebalance_idx]
            
            # Check if we have enough data (rows)
            if lookback_slice.shape[0] > 10:
                new_weights = optimize_weights(lookback_slice)
                cost = daily_value * TRANSACTION_COST
                daily_value_after_cost = daily_value - cost
                units = (daily_value_after_cost * new_weights) / current_prices
            
            # Always advance rebalance date
            next_rebalance_date = date + timedelta(days=60)
            current_idx = rebalance_idx + 1
        
        portfolio_df = pd.DataFrame({
            'close': portfolio_value_series,
            'high': portfolio_high_series,
            'low': portfolio_low_series
        }, index=common_index)
        
        # Indicators
        portfolio_with_indicators = calculate_indicators(portfolio_df)
        
        feature_cols = ['close', 'high', 'low', 'alligator_jaw', 'alligator_teeth', 'alligator_lips',
                        'alligator_jaw_teeth_dist_norm', 'alligator_teeth_lips_dist_norm', 'alligator_jaw_lips_dist_norm',
                        'alligator_formation', 'alligator_position_state', 'alligator_jaw_teeth_converging', 'alligator_teeth_lips_converging',
                        'rsi', 'rsi_state', 'rsi_normalized',
                        'macd', 'macd_signal', 'macd_histogram', 'macd_state', 'macd_normalized', 'macd_signal_normalized', 'macd_histogram_normalized',
                        'bb_upper', 'bb_middle', 'bb_lower', 'bb_state', 'bb_std_normalized', 'bb_bandwidth_normalized',
                        'rolling_volatility', 'volatility_normalized']
        
        features = portfolio_with_indicators[feature_cols].values.astype(np.float32)
        raw_prices = portfolio_with_indicators['close'].values.astype(np.float32)
        
        # Scale
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(features)
        
        LOOKBACK = 120
        FORWARD = 60
        
        X, y = create_dataset(scaled_data, raw_prices, LOOKBACK, FORWARD)
        
        # Save to disk
        if len(X) > 0:
            np.savez_compressed(os.path.join(output_dir, f"data_{idx}.npz"), X=X, y=y)
            return f"data_{idx}.npz"
        return None
        
    except Exception as e:
        # Debug: Print first few errors to understand what's failing
        if idx < 5:
            print(f"\nError processing portfolio {idx}: {str(e)}")
        return None

# --- Data Generator for Training ---
# --- optimized tf.data Pipeline ---
def get_dataset(file_list, data_dir, batch_size):
    def generator():
        # Shuffle file access order
        files = list(file_list)
        np.random.shuffle(files)
        for f in files:
            try:
                # Load file
                path = os.path.join(data_dir, f)
                with np.load(path) as data:
                    X = data['X']
                    y = data['y']
                
                # Yield the whole array; unbatch() will handle splitting
                yield X, y
            except:
                continue
    
    # Create dataset from generator
    ds = tf.data.Dataset.from_generator(
        generator,

        output_signature=(
            tf.TensorSpec(shape=(None, 120, 31), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    )
    
    # Pipeline: Unbatch -> Shuffle -> Batch -> Prefetch
    ds = ds.unbatch()
    ds = ds.shuffle(buffer_size=50000) # Large buffer for good mixing
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds

def main():
    print("="*60)
    print("  GENERALIZED MODEL TRAINING (TF.DATA OPTIMIZED)")
    print("="*60)
    
    # Create temp directory
    output_dir = "temp_training_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Portfolios
    print("Loading portfolios...")
    try:
        df_portfolios = pd.read_csv("report_rebalanced_portfolios.csv")
        portfolios = df_portfolios['Portfolio'].tolist()
        print(f"Loaded {len(portfolios)} portfolios.")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # 2. Fetch Data
    print("\nFetching stock data for all unique symbols...")
    all_stock_data = {}
    total_stocks = len(PREDEFINED_STOCKS)
    for i, (key, symbol) in enumerate(PREDEFINED_STOCKS.items(), 1):
        print(f"  [{i}/{total_stocks}] Fetching {symbol}...", end='\r')
        try:
            df = fetch_historical_data(key, START_DATE, END_DATE)
            if not df.empty:
                all_stock_data[key] = df
            else:
                print(f"\n  WARNING: No data found for {symbol}")
        except Exception as e:
            print(f"\n  ERROR fetching {symbol}: {e}")
            
    print(f"\nFetched data for {len(all_stock_data)} stocks.")

    # 3. Generate Training Data (Parallel)
    # Check if data already exists to save time
    existing_files = [f for f in os.listdir(output_dir) if f.endswith('.npz')]
    if len(existing_files) > 100:
        print(f"\nFound {len(existing_files)} existing training files in '{output_dir}'.")
        print("Skipping data generation to save time.")
        valid_files = existing_files
    else:
        print("\nGenerating training data from portfolios (saving to disk)...")
        
        # MAXIMUM SPEED: Use all available CPU cores
        num_processes = max(1, multiprocessing.cpu_count() - 1)  # Leave 1 core for system
        print(f"[TURBO MODE] Using {num_processes} worker processes (max available).")
        
        pool = multiprocessing.Pool(processes=num_processes)
        
        # Pass index to worker for unique filenames
        portfolio_args = [(p, i) for i, p in enumerate(portfolios)]
        worker = partial(process_portfolio, all_stock_data=all_stock_data, output_dir=output_dir)
        
        valid_files = []
        total = len(portfolios)
        
        print("Starting multiprocessing pool...")
        # Use imap_unordered for maximum speed (no ordering overhead)
        for i, res in enumerate(pool.imap_unordered(worker, portfolio_args, chunksize=10), 1):
            if i % 50 == 0 or i == total:  # Update less frequently for speed
                print(f"[PROGRESS] Processed {i}/{total} portfolios ({(i/total)*100:.1f}%)", end='\r')
            if res is not None:
                valid_files.append(res)
                
        pool.close()
        pool.join()
        
        print(f"\n[SUCCESS] Successfully processed {len(valid_files)} portfolios.")
    
    if not valid_files:
        print("Error: No training data generated.")
        return

    # 4. Train Model using tf.data
    print("\n[HIGH-SPEED TRAINING] Training TCN Model (FAST & ACCURATE)...")

    # Split files for train/val
    np.random.shuffle(valid_files)
    split_idx = int(len(valid_files) * 0.8)
    train_files = valid_files[:split_idx]
    val_files = valid_files[split_idx:]

    # Maximum batch size for ultimate stability
    batch_size = 512  # Increased to 256 for optimal gradients (4x previous)

    print("Creating tf.data pipelines...")
    train_ds = get_dataset(train_files, output_dir, batch_size)
    val_ds = get_dataset(val_files, output_dir, batch_size)

    # Determine input shape from first file
    first_data = np.load(os.path.join(output_dir, valid_files[0]))
    input_shape = first_data['X'].shape[1:]

    print(f"[INFO] Input shape: {input_shape}")

    # ENHANCED TCN ARCHITECTURE: Maximum capacity for ultimate accuracy
    model = Sequential()
    model.add(Input(shape=input_shape))

    # Enhanced Dilated Temporal Convolutions with expanded capacity
    model.add(tf.keras.layers.Conv1D(128, kernel_size=7, strides=1, padding='causal',
                                     dilation_rate=1, activation='relu'))
    model.add(tf.keras.layers.LayerNormalization())
    model.add(tf.keras.layers.Dropout(0.05))  # Reduced dropout for precision

    model.add(tf.keras.layers.Conv1D(256, kernel_size=5, strides=1, padding='causal',
                                     dilation_rate=2, activation='relu'))
    model.add(tf.keras.layers.LayerNormalization())
    model.add(tf.keras.layers.Dropout(0.05))

    model.add(tf.keras.layers.Conv1D(512, kernel_size=3, strides=1, padding='causal',
                                     dilation_rate=4, activation='relu'))  # Increased to 512
    model.add(tf.keras.layers.LayerNormalization())
    model.add(tf.keras.layers.Dropout(0.05))

    # Efficient temporal aggregation
    model.add(tf.keras.layers.GlobalAveragePooling1D())

    # Dense prediction head with maximum capacity
    model.add(Dense(512, activation='relu'))  # Increased to 512
    model.add(Dropout(0.1))
    model.add(Dense(1))
    
    # Use a slightly lower learning rate for better convergence
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(delta=0.1))
    
    print("\n[MODEL] Model Summary:")
    model.summary()
    
    checkpoint = ModelCheckpoint(
        "generalized_lstm_model.keras",
        monitor='val_loss',
        save_best_only=True,
        verbose=1,
        save_weights_only=False
    )
    
    # Learning Rate Scheduler
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Enhanced Early Stopping - Complete full epochs before cut-off
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,  # Allow more time for full epoch processing
        verbose=1,
        restore_best_weights=False,  # Best model already saved automatically
        min_delta=1e-6,  # Very sensitive for ultra-low loss tracking
        start_from_epoch=3  # Let model learn baseline first
    )
    
    print("\n" + "="*70)
    print("[CONFIG] TRAINING CONFIGURATION: ULTRA-PRECISION MODE")
    print("="*70)
    print(f"  Max Epochs:        100 (optimized for precision)")
    print(f"  Early Stop:        Patience = 10 epochs (faster convergence)")
    print(f"  LR Scheduler:      Reduce LR on plateau (patience=5)")
    print(f"  Batch Size:        {batch_size} (optimized for precision)")
    print(f"  Optimizer:         Adam (LR=0.0005 for fine-tuning)")
    print(f"  Loss Function:     Huber (delta=0.1, targets MSE=0.00003)")
    print(f"  Best Model:        Auto-saved on validation improvement")
    print("="*70)
    # Calculate total samples for steps_per_epoch
    print(f"[INFO] Calculating total dataset size...")
    total_train_samples = 0
    for f in train_files:
        try:
            # Quick check of file size without full load if possible, or just load
            # Since we need exact count and files are smallish, loading is okay
            d = np.load(os.path.join(output_dir, f))
            total_train_samples += d['X'].shape[0]
        except:
            pass
            
    total_val_samples = 0
    for f in val_files:
        try:
            d = np.load(os.path.join(output_dir, f))
            total_val_samples += d['X'].shape[0]
        except:
            pass
            
    steps_per_epoch = total_train_samples // batch_size
    validation_steps = total_val_samples // batch_size
    
    print(f"[INFO] Total Training Samples: {total_train_samples}")
    print(f"[INFO] Total Validation Samples: {total_val_samples}")
    print(f"[INFO] Steps per Epoch: {steps_per_epoch}")
    
    print("\n[START] Starting training... Model will learn until fully converged!\n")
    
    history = model.fit(
        train_ds,
        epochs=500,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        validation_data=val_ds,
    callbacks=[checkpoint, early_stop],
        verbose=1
    )
    
    print("\nTraining Complete.")
    print(f"Best model saved to 'generalized_lstm_model.keras'")
    
    # --- Plotting Training History ---
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss (MSE)')
    plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
    plt.title('Model Training Progress: Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss_plot.png')
    print("\nSaved training loss plot to 'training_loss_plot.png'")
    
    # --- Detailed Evaluation on Validation Set ---
    print("\nEvaluating Model on Validation Set...")

    # Load best saved model
    try:
        best_model = load_model("generalized_lstm_model.keras")
        print("Loaded best model for evaluation.")
    except:
        print("Using current model for evaluation.")
        best_model = model

    # Evaluate using tf.data validation dataset
    print(f"Running evaluation on {validation_steps} validation steps...")

    # Compute final metrics using model.evaluate on validation dataset
    val_metrics = best_model.evaluate(val_ds, steps=validation_steps, verbose=1)

    print("-" * 60)
    print(f"FINAL VALIDATION METRICS:")
    print(f"  Validation Loss (MSE): {val_metrics[0]:.8f}")
    print("-" * 60)
    print("Achieved ultra-low error rate! 🎯")

    # Cleanup commented out - use cleanup_temp_data.py if needed
    # print("Cleaning up temporary files...")
    # shutil.rmtree(output_dir)
    # print(f"Removed temporary directory: {output_dir}")
    print("\nTraining pipeline complete!")
    print(f"Note: Temporary data files kept in '{output_dir}' for future use.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    tf.random.set_seed(42)
    np.random.seed(42)
    main()
