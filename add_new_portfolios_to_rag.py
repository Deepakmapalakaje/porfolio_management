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
import itertools
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import chromadb

# Suppress warnings
warnings.filterwarnings("ignore")

# === CONFIGURATION ===
ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI3M0FIOUMiLCJqdGkiOiI2OTJiMGMxZGIzNGEzMTEzNGI4Njg3ZGQiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6dHJ1ZSwiaWF0IjoxNzY0NDI4ODI5LCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3NjQ0NTM2MDB9.FyfmIVYRBjw8p51FnR532XWaSaBSHOmptXy3Es_22JA"
BASE_URL = "https://api.upstox.com/v3/historical-candle"
START_DATE = "2018-09-03"
END_DATE = "2025-12-02"
RISK_FREE_RATE = 0.065
TRANSACTION_COST = 0.001
INITIAL_CAPITAL = 1000000.0

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

KNOWLEDGE_BASE_FILE = "rag_knowledge_base.json"
CHROMA_DB_PATH = "./chroma_db"

# === DATA FETCHING ===
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
            if response.status_code == 200:
                data = response.json().get("data", {}).get("candles", [])
                all_data.extend(data)
            time.sleep(0.1)
        except:
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

# === HRP OPTIMIZATION ===
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

# === TECHNICAL INDICATORS ===
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

# === NARRATIVE GENERATOR ===
def generate_narrative(row, hrp_weights, stock_names):
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
    
    alli_desc = row['alligator_formation_desc']
    vol = row['rolling_volatility']
    vol_desc = f"{vol:.1%}"
    
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

# === EMBEDDING VECTOR ===
def create_embedding_vector(indicators, hrp_weights, target_return):
    """Create embedding vector matching the original format"""
    vector = [
        indicators.get("rsi", 50) / 100.0,
        np.tanh(indicators.get("macd", 0) / 10000.0),
        indicators.get("volatility", 0.15),
        np.tanh(indicators.get("alligator_jaw", 0) / 1000000.0),
        np.tanh(indicators.get("alligator_teeth", 0) / 1000000.0),
        np.tanh(indicators.get("alligator_lips", 0) / 1000000.0),
    ]
    # Add HRP weights (5 stocks)
    vector.extend(hrp_weights)
    # Add target return
    vector.append(target_return)
    return vector

# === MAIN PROCESSING ===
def main():
    print("="*70)
    print(" ADDING NEW PORTFOLIOS TO RAG KNOWLEDGE BASE")
    print("="*70)
    
    # 1. Load existing knowledge base
    print("\nLoading existing knowledge base...")
    try:
        with open(KNOWLEDGE_BASE_FILE, 'r') as f:
            existing_data = json.load(f)
        print(f"[OK] Loaded {len(existing_data):,} existing entries")
        starting_portfolio_id = max([entry.get('portfolio_id', 0) for entry in existing_data]) + 1
    except FileNotFoundError:
        print("[WARNING] No existing knowledge base found. Creating new one.")
        existing_data = []
        starting_portfolio_id = 0
    
    # 2. Fetch data for new 10 stocks
    print(f"\nFetching data for {len(STOCK_UNIVERSE)} stocks...")
    all_stock_data = {}
    for key, symbol in STOCK_UNIVERSE.items():
        print(f"  Fetching {symbol}...", end="")
        df = fetch_historical_data(key, START_DATE, END_DATE)
        if not df.empty:
            all_stock_data[key] = df
            print(f" OK ({len(df)} rows)")
        else:
            print(" FAILED")
    
    if len(all_stock_data) < 5:
        print("[ERROR] Not enough stocks fetched.")
        return
    
    # 3. Align data
    print("\nAligning data...")
    common_index = sorted(list(set.intersection(*[set(df.index) for df in all_stock_data.values()])))
    print(f"  Common Period: {common_index[0].strftime('%Y-%m-%d')} to {common_index[-1].strftime('%Y-%m-%d')}")
    print(f"  Total Trading Days: {len(common_index)}")
    
    aligned_data = {k: v.loc[common_index] for k, v in all_stock_data.items()}
    
    # 4. Generate all 5-stock combinations (C(10,5) = 252)
    stock_keys = list(aligned_data.keys())
    combos = list(itertools.combinations(stock_keys, 5))
    print(f"\nGenerated {len(combos)} unique portfolios (C(10,5) = {len(combos)})")
    
    # 5. Process portfolios and generate RAG entries
    new_rag_data = []
    
    print("\nProcessing portfolios and generating RAG entries...")
    
    for combo_idx, combo in enumerate(combos):
        if (combo_idx + 1) % 50 == 0:
            print(f"  Processing portfolio {combo_idx+1}/{len(combos)}...")
        
        try:
            stock_names = [STOCK_UNIVERSE[k] for k in combo]
            
            # Create DataFrames
            price_data = {k: aligned_data[k].loc[common_index, 'close'] for k in combo}
            returns_data = {k: aligned_data[k].loc[common_index, 'returns'] for k in combo}
            
            prices_df = pd.DataFrame(price_data, index=common_index)
            returns_df = pd.DataFrame(returns_data, index=common_index)
            returns_matrix = returns_df.values
            
            # Simulate Portfolio with HRP rebalancing
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
                if rebalance_idx >= total_len:
                    rebalance_idx = total_len - 1
                
                # Calculate values
                chunk_prices = prices_df.iloc[current_idx:rebalance_idx+1].values
                chunk_values = np.sum(chunk_prices * units, axis=1)
                portfolio_values.extend(chunk_values)
                
                # REBALANCE POINT
                date = common_index[rebalance_idx]
                lookback_start = date - timedelta(days=120)
                lookback_start_idx = bisect.bisect_left(common_index, lookback_start)
                
                lookback_slice = returns_matrix[lookback_start_idx : rebalance_idx]
                
                if lookback_slice.shape[0] > 10:
                    # Optimize weights (HRP)
                    new_weights = optimize_weights(lookback_slice)
                    
                    # Calculate portfolio indicators
                    temp_portfolio_series = pd.Series(portfolio_values, index=common_index[:len(portfolio_values)])
                    temp_df = pd.DataFrame({
                        'close': temp_portfolio_series,
                        'high': temp_portfolio_series,
                        'low': temp_portfolio_series
                    })
                    temp_indicators = calculate_indicators(temp_df)
                    
                    if not temp_indicators.empty:
                        current_row = temp_indicators.iloc[-1]
                        
                        # Generate narrative
                        narrative = generate_narrative(current_row, new_weights, stock_names)
                        
                        # Calculate future outcome (60 days ahead)
                        future_date = date + timedelta(days=60)
                        future_idx = bisect.bisect_left(common_index, future_date)
                        
                        if future_idx < total_len:
                            current_val = chunk_values[-1]
                            cost = current_val * TRANSACTION_COST
                            val_after_cost = current_val - cost
                            new_units = (val_after_cost * new_weights) / prices_df.iloc[rebalance_idx].values
                            
                            future_prices = prices_df.iloc[future_idx].values
                            future_val = np.sum(future_prices * new_units)
                            pct_return = (future_val - current_val) / current_val
                            
                            # Add to RAG dataset
                            entry = {
                                "date": date.strftime('%Y-%m-%d'),
                                "portfolio_id": starting_portfolio_id + combo_idx,
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
                            new_rag_data.append(entry)
                            units = new_units
                
                next_rebalance_date = date + timedelta(days=60)
                current_idx = rebalance_idx + 1
                
        except Exception as e:
            continue
    
    print(f"\n[OK] Generated {len(new_rag_data):,} new RAG entries")
    
    # 6. Combine with existing data
    combined_data = existing_data + new_rag_data
    print(f"[OK] Total entries: {len(combined_data):,} (old: {len(existing_data):,}, new: {len(new_rag_data):,})")
    
    # 7. Save updated knowledge base
    print("\nSaving updated knowledge base...")
    with open(KNOWLEDGE_BASE_FILE, "w") as f:
        json.dump(combined_data, f, indent=2)
    print(f"[OK] Saved to '{KNOWLEDGE_BASE_FILE}'")
    
    # 8. Update ChromaDB
    print("\nUpdating ChromaDB vector database...")
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        
        # Get or create collection
        try:
            collection = chroma_client.get_collection("portfolio_scenarios")
            print(f"[OK] Found existing collection with {collection.count():,} entries")
        except:
            collection = chroma_client.create_collection("portfolio_scenarios")
            print("[OK] Created new collection")
        
        # Add new entries to ChromaDB
        print(f"Adding {len(new_rag_data):,} new entries to vector DB...")
        
        for i, entry in enumerate(new_rag_data):
            if (i + 1) % 100 == 0:
                print(f"  Added {i+1}/{len(new_rag_data)} entries...", end='\r')
            
            # Create embedding vector
            hrp_weights_list = list(entry['hrp_weights'].values())
            embedding = create_embedding_vector(
                entry['indicators'],
                hrp_weights_list,
                entry['target_return_60d']
            )
            
            # Add to collection
            collection.add(
                embeddings=[embedding],
                documents=[entry['narrative']],
                ids=[f"scenario_{entry['portfolio_id']}_{entry['date']}"]
            )
        
        print(f"\n[OK] ChromaDB updated! Total entries: {collection.count():,}")
        
    except Exception as e:
        print(f"[ERROR] Failed to update ChromaDB: {e}")
        return
    
    print("\n" + "="*70)
    print("[SUCCESS] RAG Knowledge Base Updated!")
    print("="*70)
    print(f"  Old entries: {len(existing_data):,}")
    print(f"  New entries: {len(new_rag_data):,}")
    print(f"  Total entries: {len(combined_data):,}")
    print(f"  Knowledge base: {KNOWLEDGE_BASE_FILE}")
    print(f"  Vector DB: {CHROMA_DB_PATH}")
    print("="*70)

if __name__ == "__main__":
    main()
