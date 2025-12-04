import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from urllib.parse import quote
import warnings
import time
import os
import json
import bisect
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import scipy.stats as stats

# --- Configuration ---
# Using the same configuration as train_generalized_model.py
ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI3M0FIOUMiLCJqdGkiOiI2OTFkNWFiODFkZTI2NzdiZDkxNjkxMGMiLCJpc015dHRpQ2xpZW50IjpmYWxzZSwiaXNQbHJpUGxhbiI6dHJ1ZSwiaWF0IjoxNzYzNTMxNDQ4LCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3NjM1ODk2MDB9.7dPxOIudeYCSEo5HNVzWXtn2EhzDFWKwICz7Aui4"
BASE_URL = "https://api.upstox.com/v3/historical-candle"
START_DATE = "2018-09-03"
END_DATE = "2025-11-19"
RISK_FREE_RATE = 0.065
TRANSACTION_COST = 0.001
INITIAL_CAPITAL = 1000000.0

# --- Stock Mapping ---
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
def optimize_weights(returns_matrix):
    cov = np.cov(returns_matrix, rowvar=False) 
    corr = np.corrcoef(returns_matrix, rowvar=False)
    
    if np.any(np.isnan(cov)) or np.any(np.isnan(corr)):
        return np.full(returns_matrix.shape[1], 1.0 / returns_matrix.shape[1])

    dist = np.sqrt((1 - corr) / 2)
    np.fill_diagonal(dist, 0)
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

# --- Technical Indicators ---
def calculate_indicators(df):
    df = df.copy()
    median_price = (df['high'] + df['low']) / 2
    def smma(series, window):
        return series.ewm(alpha=1/window, adjust=False).mean()

    # Alligator
    df['alligator_jaw'] = smma(median_price, 13).shift(8)
    df['alligator_teeth'] = smma(median_price, 8).shift(5)
    df['alligator_lips'] = smma(median_price, 5).shift(3)
    
    df['alligator_jaw_teeth_dist'] = df['alligator_jaw'] - df['alligator_teeth']
    df['alligator_teeth_lips_dist'] = df['alligator_teeth'] - df['alligator_lips']
    
    def get_formation_code(row):
        jaw, teeth, lips = row['alligator_jaw'], row['alligator_teeth'], row['alligator_lips']
        tol = abs(row['close']) * 0.001
        if abs(jaw - teeth) < tol and abs(teeth - lips) < tol: return "All Equal"
        elif abs(jaw - teeth) < tol: return "Teeth > Lips" if teeth > lips else "Teeth = Jaw"
        elif abs(teeth - lips) < tol: return "Jaw > Teeth" if jaw > teeth else "Teeth = Lips"
        else:
            if jaw > teeth and teeth > lips: return "Bearish Convergence"
            elif jaw > teeth and teeth < lips: return "Teeth Below Others"
            elif jaw < teeth and teeth > lips: return "Jaw Below Others"
            elif jaw < teeth and teeth < lips: return "Bullish Convergence"
        return "Mixed"

    df['alligator_formation_desc'] = df.apply(get_formation_code, axis=1)

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    
    # Volatility
    df['rolling_volatility'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
    
    return df.dropna()

# --- Narrative Generator ---
def generate_narrative(row, hrp_weights, stock_names):
    # Create a text description of the environment
    
    # 1. Market State (RSI/MACD)
    rsi = row['rsi']
    rsi_desc = "Neutral"
    if rsi > 70: rsi_desc = "Overbought"
    elif rsi < 30: rsi_desc = "Oversold"
    elif rsi > 60: rsi_desc = "Bullish Momentum"
    elif rsi < 40: rsi_desc = "Bearish Momentum"
    
    macd_desc = "Neutral"
    if row['macd'] > row['macd_signal']:
        macd_desc = "Bullish (MACD > Signal)"
    else:
        macd_desc = "Bearish (MACD < Signal)"
        
    # 2. Alligator
    alli_desc = row['alligator_formation_desc']
    
    # 3. Volatility
    vol = row['rolling_volatility']
    vol_desc = f"{vol:.1%}"
    
    # 4. HRP Weights
    # Sort weights to find top allocations
    sorted_weights = sorted(zip(stock_names, hrp_weights), key=lambda x: x[1], reverse=True)
    top_allocations = ", ".join([f"{name} ({w:.1%})" for name, w in sorted_weights[:3]])
    
    narrative = (
        f"Market Environment: RSI is {rsi:.1f} ({rsi_desc}). "
        f"MACD is {macd_desc}. "
        f"Alligator Indicator shows a {alli_desc} pattern. "
        f"Annualized Volatility is {vol_desc}. "
        f"HRP Optimization allocated highest weights to: {top_allocations}."
    )
    return narrative

# --- Main Processing ---
def main():
    print("="*60)
    print("  CREATING RAG KNOWLEDGE BASE")
    print("="*60)
    
    # 1. Load Portfolios
    print("Loading portfolios...")
    try:
        df_portfolios = pd.read_csv("report_rebalanced_portfolios.csv")
        portfolios = df_portfolios['Portfolio'].tolist()
        print(f"Loaded {len(portfolios)} portfolios.")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # 2. Fetch Data (Reuse logic)
    print("\nFetching stock data...")
    all_stock_data = {}
    for i, (key, symbol) in enumerate(PREDEFINED_STOCKS.items(), 1):
        print(f"  [{i}/{len(PREDEFINED_STOCKS)}] Fetching {symbol}...", end='\r')
        try:
            df = fetch_historical_data(key, START_DATE, END_DATE)
            if not df.empty:
                all_stock_data[key] = df
        except:
            pass
            
    # 3. Process Portfolios & Generate Narratives
    rag_data = []
    
    print("\nProcessing portfolios and generating narratives...")
    
    # Process ALL portfolios
    for idx, portfolio_str in enumerate(portfolios): 
        try:
            symbols = [s.strip() for s in portfolio_str.replace('"', '').split(',')]
            keys = [SYMBOL_TO_KEY.get(s) for s in symbols if s in SYMBOL_TO_KEY]
            stock_names = [PREDEFINED_STOCKS[k] for k in keys]
            
            if len(keys) != 5: continue

            common_index = sorted(list(set.intersection(*[set(all_stock_data[k].index) for k in keys])))
            if not common_index: continue
            
            # Create Portfolio DataFrame
            price_data = {k: all_stock_data[k].loc[common_index, 'close'] for k in keys}
            returns_data = {k: all_stock_data[k].loc[common_index, 'returns'] for k in keys}
            
            prices_df = pd.DataFrame(price_data, index=common_index)
            returns_df = pd.DataFrame(returns_data, index=common_index)
            returns_matrix = returns_df.values
            
            # Simulate Portfolio
            current_capital = INITIAL_CAPITAL
            current_weights = np.array([0.2] * 5)
            units = (current_capital * current_weights) / prices_df.iloc[0].values
            
            start_date = common_index[0]
            next_rebalance_date = start_date + timedelta(days=60)
            
            current_idx = 0
            total_len = len(common_index)
            
            portfolio_values = []
            
            while current_idx < total_len:
                rebalance_idx = bisect.bisect_left(common_index, next_rebalance_date)
                if rebalance_idx >= total_len: rebalance_idx = total_len - 1
                
                # Calculate Value for this chunk
                chunk_prices = prices_df.iloc[current_idx:rebalance_idx+1].values
                chunk_values = np.sum(chunk_prices * units, axis=1)
                portfolio_values.extend(chunk_values)
                
                # REBALANCE POINT
                date = common_index[rebalance_idx]
                lookback_start = date - timedelta(days=120)
                lookback_start_idx = bisect.bisect_left(common_index, lookback_start)
                
                lookback_slice = returns_matrix[lookback_start_idx : rebalance_idx]
                
                if lookback_slice.shape[0] > 10:
                    # 1. Optimize Weights (HRP)
                    new_weights = optimize_weights(lookback_slice)
                    
                    # 2. Calculate Portfolio Indicators at this moment
                    # We need a temporary portfolio series up to this point to calc indicators
                    temp_portfolio_series = pd.Series(portfolio_values, index=common_index[:len(portfolio_values)])
                    temp_df = pd.DataFrame({'close': temp_portfolio_series, 'high': temp_portfolio_series, 'low': temp_portfolio_series}) # Mock high/low
                    temp_indicators = calculate_indicators(temp_df)
                    
                    if not temp_indicators.empty:
                        current_row = temp_indicators.iloc[-1]
                        
                        # 3. Generate Narrative
                        narrative = generate_narrative(current_row, new_weights, stock_names)
                        
                        # 4. Calculate Future Outcome (Target)
                        # Look ahead 60 days
                        future_date = date + timedelta(days=60)
                        future_idx = bisect.bisect_left(common_index, future_date)
                        
                        if future_idx < total_len:
                            current_val = chunk_values[-1]
                            # We need to estimate future value. 
                            # Since we rebalance here, we need to know the value of the NEW portfolio 60 days later.
                            # Update units first
                            cost = current_val * TRANSACTION_COST
                            val_after_cost = current_val - cost
                            new_units = (val_after_cost * new_weights) / prices_df.iloc[rebalance_idx].values
                            
                            future_prices = prices_df.iloc[future_idx].values
                            future_val = np.sum(future_prices * new_units)
                            
                            pct_return = (future_val - current_val) / current_val
                            
                            # Add to RAG Dataset
                            entry = {
                                "date": date.strftime('%Y-%m-%d'),
                                "portfolio_id": idx,
                                "narrative": narrative,
                                "indicators": {
                                    "rsi": float(current_row['rsi']),
                                    "macd": float(current_row['macd']),
                                    "volatility": float(current_row['rolling_volatility']),
                                    "alligator_jaw": float(current_row['alligator_jaw']),
                                    "alligator_teeth": float(current_row['alligator_teeth']),
                                    "alligator_lips": float(current_row['alligator_lips'])
                                },
                                "hrp_weights": {name: float(w) for name, w in zip(stock_names, new_weights)},
                                "target_return_60d": float(pct_return),
                                "target_return_pct": f"{pct_return:.2%}"
                            }
                            rag_data.append(entry)
                            
                            # Update global units for next loop
                            units = new_units
                    
                next_rebalance_date = date + timedelta(days=60)
                current_idx = rebalance_idx + 1
                
        except Exception as e:
            continue
            
        print(f"Processed Portfolio {idx}...", end='\r')

    # Save to JSON
    with open("rag_knowledge_base.json", "w") as f:
        json.dump(rag_data, f, indent=2)
        
    print(f"\n\n[SUCCESS] Created RAG Knowledge Base with {len(rag_data)} entries.")
    print("Saved to 'rag_knowledge_base.json'")

if __name__ == "__main__":
    main()
