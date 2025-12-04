import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from urllib.parse import quote
import warnings
import time
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import itertools
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import bisect

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
WARMUP_DAYS = 120
INITIAL_CAPITAL = 1000000.0
TRANSACTION_COST = 0.001

# NEW 10 STOCKS
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

# TRAINING CONFIG
EXISTING_MODEL_PATH = "generalized_lstm_model_improved.keras"
NEW_MODEL_PATH = "generalized_lstm_model_improved_v2.keras"
OLD_TRAINING_DATA_DIR = "temp_training_data"
NEW_TRAINING_DATA_DIR = "new_portfolio_training_data"
ADDITIONAL_EPOCHS = 50
BATCH_SIZE = 512

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
            time.sleep(0.1)
        except Exception as e:
            continue
    
    if all_data:
        df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.index = df["timestamp"].dt.tz_localize(None)
        df.sort_index(inplace=True)
        df["returns"] = df["close"].pct_change()
        return df.dropna()
    return pd.DataFrame()

# === HRP OPTIMIZATION ===
def optimize_weights_hrp(returns_matrix):
    """Hierarchical Risk Parity optimizer - EXACT copy from train_generalized_model.py"""
    # Calculate Covariance and Correlation
    cov = np.cov(returns_matrix, rowvar=False)
    corr = np.corrcoef(returns_matrix, rowvar=False)
    
    # Handle NaN or constant returns
    if np.any(np.isnan(cov)) or np.any(np.isnan(corr)):
        return np.full(returns_matrix.shape[1], 1.0 / returns_matrix.shape[1])
    
    # Distance Matrix
    dist = np.sqrt((1 - corr) / 2)
    np.fill_diagonal(dist, 0)
    
    # Ensure symmetry
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
    
    weights = get_rec_bipart(cov, sort_ix)
    return weights

# === TECHNICAL INDICATORS ===
def calculate_indicators(df):
    """Calculate all technical indicators"""
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

# === GENERATE NEW TRAINING DATA WITH HRP WEIGHTS ===
def generate_new_training_data():
    """Generate training data from 252 new portfolios with HRP weights (C(10,5) = 252)"""
    print("="*70)
    print(" GENERATING NEW TRAINING DATA FROM 10 STOCKS (HRP WEIGHTED)")
    print("="*70)
    
    # Create output directory
    os.makedirs(NEW_TRAINING_DATA_DIR, exist_ok=True)
    
    # Fetch data for all 10 stocks
    print(f"\nFetching data for {len(STOCK_UNIVERSE)} stocks...")
    all_data = {}
    for key, symbol in STOCK_UNIVERSE.items():
        print(f"  Fetching {symbol}...", end="")
        df = fetch_historical_data(key, START_DATE, END_DATE)
        if not df.empty:
            all_data[key] = df
            print(f" OK ({len(df)} rows)")
        else:
            print(" ✗")
    
    if len(all_data) < 5:
        print("ERROR: Not enough stocks fetched.")
        return False
    
    # Align data
    print("\nAligning data...")
    common_index = sorted(list(set.intersection(*[set(df.index) for df in all_data.values()])))
    print(f"  Common Period: {common_index[0].strftime('%Y-%m-%d')} to {common_index[-1].strftime('%Y-%m-%d')}")
    print(f"  Total Trading Days: {len(common_index)}")
    
    aligned_data = {k: v.loc[common_index] for k, v in all_data.items()}
    
    # Generate all 5-stock combinations (C(10,5) = 252)
    stock_keys = list(aligned_data.keys())
    combos = list(itertools.combinations(stock_keys, 5))
    print(f"\nGenerated {len(combos)} unique portfolios (C(10,5) = {len(combos)})")
    
    # Feature columns (31 features)
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
    
    # Generate training samples for each portfolio
    print("\nGenerating training samples with HRP rebalancing...")
    total_samples = 0
    
    for combo_idx, combo in enumerate(combos):
        if (combo_idx + 1) % 50 == 0:
            print(f"  Processing portfolio {combo_idx+1}/{len(combos)}...")
        
        # Create DataFrames for vectorized access
        price_data = {k: aligned_data[k].loc[common_index, 'close'] for k in combo}
        high_data = {k: aligned_data[k].loc[common_index, 'high'] for k in combo}
        low_data = {k: aligned_data[k].loc[common_index, 'low'] for k in combo}
        returns_data = {k: aligned_data[k].loc[common_index, 'returns'] for k in combo}
        
        prices_df = pd.DataFrame(price_data, index=common_index)
        highs_df = pd.DataFrame(high_data, index=common_index)
        lows_df = pd.DataFrame(low_data, index=common_index)
        returns_df = pd.DataFrame(returns_data, index=common_index)
        
        returns_matrix = returns_df.values  # (T, 5)
        
        portfolio_value_series = []
        portfolio_high_series = []
        portfolio_low_series = []
        
        current_capital = INITIAL_CAPITAL
        current_weights = np.array([0.2] * 5)  # Start with equal weights
        
        # Initial units
        first_prices = prices_df.iloc[0].values
        units = (current_capital * current_weights) / first_prices
        
        start_date = common_index[0]
        next_rebalance_date = start_date + timedelta(days=60)
        
        current_idx = 0
        total_len = len(common_index)
        
        while current_idx < total_len:
            # Find index of the next rebalance date
            rebalance_idx = bisect.bisect_left(common_index, next_rebalance_date)
            
            if rebalance_idx >= total_len:
                rebalance_idx = total_len - 1
                is_last_chunk = True
            else:
                is_last_chunk = False
            
            if rebalance_idx < current_idx:
                rebalance_idx = current_idx
            
            chunk_start = current_idx
            chunk_end = rebalance_idx + 1
            
            chunk_prices = prices_df.iloc[chunk_start:chunk_end].values
            chunk_highs = highs_df.iloc[chunk_start:chunk_end].values
            chunk_lows = lows_df.iloc[chunk_start:chunk_end].values
            
            # Vectorized calculation of portfolio values
            chunk_values = np.sum(chunk_prices * units, axis=1)
            chunk_portfolio_highs = np.sum(chunk_highs * units, axis=1)
            chunk_portfolio_lows = np.sum(chunk_lows * units, axis=1)
            
            portfolio_value_series.extend(chunk_values)
            portfolio_high_series.extend(chunk_portfolio_highs)
            portfolio_low_series.extend(chunk_portfolio_lows)
            
            if is_last_chunk:
                break
            
            # Perform HRP Rebalancing
            date = common_index[rebalance_idx]
            current_prices = chunk_prices[-1]
            daily_value = chunk_values[-1]
            
            lookback_start = date - timedelta(days=120)
            lookback_start_idx = bisect.bisect_left(common_index, lookback_start)
            
            lookback_slice = returns_matrix[lookback_start_idx : rebalance_idx]
            
            # Calculate HRP weights
            if lookback_slice.shape[0] > 10:
                new_weights = optimize_weights_hrp(lookback_slice)
                cost = daily_value * TRANSACTION_COST
                daily_value_after_cost = daily_value - cost
                units = (daily_value_after_cost * new_weights) / current_prices
            
            next_rebalance_date = date + timedelta(days=60)
            current_idx = rebalance_idx + 1
        
        portfolio_df = pd.DataFrame({
            'close': portfolio_value_series,
            'high': portfolio_high_series,
            'low': portfolio_low_series
        }, index=common_index)
        
        # Calculate indicators on portfolio
        portfolio_with_indicators = calculate_indicators(portfolio_df)
        
        if len(portfolio_with_indicators) < LOOKBACK_WINDOW + FORWARD_WINDOW + WARMUP_DAYS:
            continue
        
        features = portfolio_with_indicators[feature_cols].values.astype(np.float32)
        raw_prices = portfolio_with_indicators['close'].values.astype(np.float32)
        
        # Scale
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(features)
        
        # Generate sliding window samples
        X_samples = []
        y_samples = []
        
        for i in range(len(scaled_data) - LOOKBACK_WINDOW - FORWARD_WINDOW):
            # Extract 120-day window
            window_data = scaled_data[i:i+LOOKBACK_WINDOW]
            
            # Calculate target (60-day forward return)
            current_val = raw_prices[i + LOOKBACK_WINDOW - 1]
            future_val = raw_prices[i + LOOKBACK_WINDOW + FORWARD_WINDOW - 1]
            target_return = future_val / current_val
            
            X_samples.append(window_data)
            y_samples.append(target_return)
        
        if len(X_samples) > 0:
            X_array = np.array(X_samples, dtype=np.float32)
            y_array = np.array(y_samples, dtype=np.float32)
            
            # Save to file
            output_file = os.path.join(NEW_TRAINING_DATA_DIR, f"portfolio_{combo_idx}.npz")
            np.savez_compressed(output_file, X=X_array, y=y_array)
            
            total_samples += len(X_samples)
    
    print(f"\n[OK] Generated {total_samples:,} training samples from {len(combos)} HRP-weighted portfolios")
    print(f"[OK] Saved to: {NEW_TRAINING_DATA_DIR}/")
    return True

# === DATA GENERATOR ===
def get_dataset(file_list, data_dir, batch_size):
    """Create tf.data pipeline from saved .npz files"""
    def generator():
        files = list(file_list)
        np.random.shuffle(files)
        for f in files:
            try:
                path = os.path.join(data_dir, f)
                with np.load(path) as data:
                    X = data['X']
                    y = data['y']
                yield X, y
            except:
                continue
    
    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 120, 31), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    )
    
    ds = ds.unbatch()
    ds = ds.shuffle(buffer_size=50000)
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds

# === TRAIN MODEL ===
def train_model_with_combined_data():
    """Train model with both old and new training data"""
    print("\n" + "="*70)
    print(" TRAINING MODEL WITH COMBINED DATA")
    print("="*70)
    
    # Check model exists
    if not os.path.exists(EXISTING_MODEL_PATH):
        print(f"\n[ERROR] Model file '{EXISTING_MODEL_PATH}' not found!")
        return
    
    # Get old training files
    old_files = [f for f in os.listdir(OLD_TRAINING_DATA_DIR) if f.endswith('.npz')]
    print(f"\n[OLD DATA] Found {len(old_files)} files in {OLD_TRAINING_DATA_DIR}")
    
    # Get new training files
    new_files = [f for f in os.listdir(NEW_TRAINING_DATA_DIR) if f.endswith('.npz')]
    print(f"[NEW DATA] Found {len(new_files)} files in {NEW_TRAINING_DATA_DIR}")
    
    # Load model
    print(f"\n[LOADING] Loading model from '{EXISTING_MODEL_PATH}'...")
    model = load_model(EXISTING_MODEL_PATH)
    print("[SUCCESS] Model loaded!")
    
    # Prepare combined dataset
    print("\n[PREPARING] Creating combined training dataset...")
    
    # Combine file lists with directory paths
    combined_files = [(f, OLD_TRAINING_DATA_DIR) for f in old_files] + \
                     [(f, NEW_TRAINING_DATA_DIR) for f in new_files]
    
    np.random.shuffle(combined_files)
    
    # Split train/val
    split_idx = int(len(combined_files) * 0.8)
    train_files = combined_files[:split_idx]
    val_files = combined_files[split_idx:]
    
    print(f"  Training files: {len(train_files)} (old + new)")
    print(f"  Validation files: {len(val_files)}")
    
    # Create datasets
    def get_combined_dataset(file_list, batch_size):
        def generator():
            files = list(file_list)
            np.random.shuffle(files)
            for f, data_dir in files:
                try:
                    path = os.path.join(data_dir, f)
                    with np.load(path) as data:
                        X = data['X']
                        y = data['y']
                    yield X, y
                except:
                    continue
        
        ds = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(None, 120, 31), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32)
            )
        )
        
        ds = ds.unbatch()
        ds = ds.shuffle(buffer_size=50000)
        ds = ds.repeat()
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        
        return ds
    
    train_ds = get_combined_dataset(train_files, BATCH_SIZE)
    val_ds = get_combined_dataset(val_files, BATCH_SIZE)
    
    # Calculate steps
    print(f"\n[CALCULATING] Calculating dataset sizes...")
    total_train_samples = 0
    for f, data_dir in train_files:
        try:
            d = np.load(os.path.join(data_dir, f))
            total_train_samples += d['X'].shape[0]
        except:
            pass
    
    total_val_samples = 0
    for f, data_dir in val_files:
        try:
            d = np.load(os.path.join(data_dir, f))
            total_val_samples += d['X'].shape[0]
        except:
            pass
    
    steps_per_epoch = total_train_samples // BATCH_SIZE
    validation_steps = total_val_samples // BATCH_SIZE
    
    print(f"  Total Training Samples: {total_train_samples:,}")
    print(f"  Total Validation Samples: {total_val_samples:,}")
    print(f"  Steps per Epoch: {steps_per_epoch}")
    print(f"  Validation Steps: {validation_steps}")
    
    # Setup callbacks
    checkpoint = ModelCheckpoint(
        NEW_MODEL_PATH,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        verbose=1,
        restore_best_weights=False
    )
    
    # Reduce learning rate for fine-tuning
    current_lr = model.optimizer.learning_rate.numpy()
    new_lr = current_lr * 0.3
    model.optimizer.learning_rate.assign(new_lr)
    print(f"\n[TUNING] Learning rate: {current_lr:.6f} -> {new_lr:.6f}")
    
    # Train
    print("\n" + "="*70)
    print("[TRAINING] STARTING TRAINING WITH COMBINED DATA...")
    print("="*70)
    
    history = model.fit(
        train_ds,
        epochs=ADDITIONAL_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        validation_data=val_ds,
        callbacks=[checkpoint, lr_scheduler, early_stop],
        verbose=1
    )
    
    print("\n" + "="*70)
    print("[COMPLETE] TRAINING COMPLETE!")
    print("="*70)
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Training with New Portfolio Data: Loss vs Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (Huber)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('new_portfolio_training_plot.png', dpi=150)
    print(f"\n[SUCCESS] Saved training plot to 'new_portfolio_training_plot.png'")
    
    # Evaluate
    print("\n[EVALUATING] Evaluating improved model...")
    best_model = load_model(NEW_MODEL_PATH)
    val_loss = best_model.evaluate(val_ds, steps=validation_steps, verbose=1)
    
    print("\n" + "="*70)
    print("[RESULTS] FINAL RESULTS")
    print("="*70)
    print(f"  Validation Loss: {val_loss:.8f}")
    print(f"  Model saved to: '{NEW_MODEL_PATH}'")
    print(f"  Training plot saved to: 'new_portfolio_training_plot.png'")
    print("="*70)

def main():
    print("\n" + "="*70)
    print(" TRAIN MODEL WITH NEW 10-STOCK PORTFOLIOS (HRP WEIGHTED)")
    print("="*70)
    print("\nThis script will:")
    print("  1. Generate 252 new portfolios from 10 stocks (HRP weighted)")
    print("  2. Combine with existing ~7851 portfolio training data")
    print("  3. Train the TCN model to learn new patterns")
    print("="*70)
    
    # Step 1: Generate new training data
    if not generate_new_training_data():
        print("\n[ERROR] Failed to generate new training data")
        return
    
    # Step 2: Train model with combined data
    train_model_with_combined_data()
    
    print("\n[SUCCESS] All done! Your model has learned from the new HRP-weighted portfolios.")
    print(f"[TIP] Use '{NEW_MODEL_PATH}' for predictions.")

if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)
    main()
