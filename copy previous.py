import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from scipy import stats
from urllib.parse import quote
from scipy.optimize import minimize
import itertools
import warnings
import multiprocessing
from functools import partial

# --- Configuration ---
ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI3M0FIOUMiLCJqdGkiOiI2OTFkNWFiODFkZTI2NzdiZDkxNjkxMGMiLCJpc015dHRpQ2xpZW50IjpmYWxzZSwiaXNQbHJpUGxhbiI6dHJ1ZSwiaWF0IjoxNzYzNTMxNDQ4LCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3NjM1ODk2MDB9.7dPxOIudeYCSEo5HNVzWXtn2EhzDFWKwVRwICz7Aui4"
BASE_URL = "https://api.upstox.com/v3/historical-candle"
START_DATE = "2018-09-03"
END_DATE = "2025-11-18"
RISK_FREE_RATE = 0.065  # 6.5% Institutional Standard
TRANSACTION_COST = 0.001  # 0.1% per rebalance
BENCHMARK_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"
BENCHMARK_TRADINGSYMBOL = "NIFTY 50"


# --- Data Fetching ---
def fetch_historical_data(instrument_key, from_date, to_date):
    """Fetch historical candle data for a given instrument and date range, chunked if necessary."""
    encoded_instrument_key = quote(instrument_key, safe='')
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Accept": "application/json"
    }

    from_date_dt = pd.to_datetime(from_date)
    to_date_dt = pd.to_datetime(to_date)

    max_chunk_days = 365
    chunks = []
    current_start = from_date_dt
    while current_start < to_date_dt:
        current_end = min(current_start + pd.Timedelta(days=max_chunk_days), to_date_dt)
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
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {instrument_key} ({chunk_from} to {chunk_to}): {e}")
            try:
                print(f"Response: {response.text}")
            except NameError:
                pass

    if all_data:
        df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df.index = df.index.tz_localize(None)
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


def calculate_std_dev(returns):
    if returns.empty: return 0
    return np.std(returns) * np.sqrt(252)


def get_metrics(returns, benchmark_returns, risk_free_rate):
    metrics = {}
    metrics['sharpe_ratio'] = calculate_sharpe_ratio(returns, risk_free_rate)
    metrics['sortino_ratio'] = calculate_sortino_ratio(returns, risk_free_rate)
    metrics['max_drawdown'] = calculate_max_drawdown(returns)
    metrics['std_dev'] = calculate_std_dev(returns)
    return metrics


# --- Portfolio Optimization Helper ---
def optimize_weights(returns_df):
    num_assets = returns_df.shape[1]
    # Objective: Maximize Sharpe Ratio (minimize negative Sharpe)
    # Note: We use 0.0 here for optimization stability, but report using RISK_FREE_RATE
    objective = lambda weights: -calculate_sharpe_ratio((returns_df * weights).sum(axis=1), 0.0)
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    initial_weights = np.array(num_assets * [1. / num_assets,])
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=cons)
        
    if result.success:
        return result.x
    else:
        return np.array(returns_df.shape[1] * [1. / returns_df.shape[1],])


# --- Core Evaluation Logic (Picklable) ---
def evaluate_combination(combination, all_stock_data, benchmark_metrics_overall):
    """
    Evaluates a single combination of stocks.
    Args:
        combination: Tuple of instrument_keys
        all_stock_data: Dictionary of DataFrames (shared memory or passed copy)
        benchmark_metrics_overall: Dict of benchmark metrics for comparison
    Returns:
        Dict with results or None if rejected
    """
    tradingsymbols = [all_stock_data[key]['symbol'] for key in combination]
    
    portfolio_full_returns = []
    current_date = datetime(2019, 1, 1)
    end_date_dt = datetime.strptime(END_DATE, '%Y-%m-%d')

    while current_date <= end_date_dt:
        rebalance_date = current_date
        lookback_start_date = rebalance_date - timedelta(days=60)
        forward_end_date = rebalance_date + timedelta(days=59)

        # 1. Lookback & Optimization
        lookback_returns_list = []
        for key in combination:
            stock_data = all_stock_data[key]['data']
            actual_lookback_start = max(stock_data.index.min(), lookback_start_date)
            lookback_mask = (stock_data.index >= actual_lookback_start) & (stock_data.index < rebalance_date)
            returns_slice = stock_data.loc[lookback_mask, 'returns']
            lookback_returns_list.append(returns_slice)
        
        lookback_returns_df = pd.concat(lookback_returns_list, axis=1).dropna()
        
        if lookback_returns_df.shape[0] < 2 or lookback_returns_df.shape[1] != 5:
            current_date += timedelta(days=60)
            continue

        optimal_weights = optimize_weights(lookback_returns_df)

        # 2. Forward Performance
        forward_returns_list = []
        for key in combination:
            stock_data = all_stock_data[key]['data']
            forward_mask = (stock_data.index >= rebalance_date) & (stock_data.index <= forward_end_date)
            returns_slice = stock_data.loc[forward_mask, 'returns']
            forward_returns_list.append(returns_slice)

        forward_returns_df = pd.concat(forward_returns_list, axis=1).dropna()
        
        if not forward_returns_df.empty:
            # Calculate gross returns
            period_gross_returns = (forward_returns_df * optimal_weights).sum(axis=1)
            
            # Apply Transaction Costs (deduct from the first day of the period)
            # We approximate this by subtracting the cost from the first return
            # Cost = 0.1% of portfolio value. In return terms, it's just -0.001
            if not period_gross_returns.empty:
                period_gross_returns.iloc[0] -= TRANSACTION_COST
                
            portfolio_full_returns.append(period_gross_returns)

        current_date += timedelta(days=60)

    if not portfolio_full_returns:
        return None

    portfolio_returns = pd.concat(portfolio_full_returns)
    
    # Metrics
    portfolio_metrics = get_metrics(portfolio_returns, None, RISK_FREE_RATE)
    total_profit = (1 + portfolio_returns).prod() - 1
    
    last_year_start = datetime.strptime(END_DATE, '%Y-%m-%d') - timedelta(days=365)
    returns_last_year = portfolio_returns[portfolio_returns.index >= last_year_start]
    profit_last_year = (1 + returns_last_year).prod() - 1 if not returns_last_year.empty else 0

    last_60_days_start = datetime.strptime(END_DATE, '%Y-%m-%d') - timedelta(days=60)
    returns_last_60_days = portfolio_returns[portfolio_returns.index >= last_60_days_start]
    profit_last_60_days = (1 + returns_last_60_days).prod() - 1 if not returns_last_60_days.empty else 0

    # Filtering Logic
    sharpe_ok = portfolio_metrics['sharpe_ratio'] > benchmark_metrics_overall['sharpe_ratio']
    sortino_ok = portfolio_metrics['sortino_ratio'] > benchmark_metrics_overall['sortino_ratio']
    profit_ok = total_profit > 0 and profit_last_year > 0 and profit_last_60_days > 0

    if sharpe_ok and sortino_ok and profit_ok:
        yearly_returns = portfolio_returns.resample('YE').apply(lambda x: (1 + x).prod() - 1)
        yearly_profit = {year.year: profit for year, profit in yearly_returns.to_dict().items()}
        
        return {
            'combination_symbols': ', '.join(tradingsymbols),
            'total_profit_pct': total_profit * 100,
            'profit_last_year_pct': profit_last_year * 100,
            'profit_last_60_days_pct': profit_last_60_days * 100,
            'sharpe_ratio': portfolio_metrics['sharpe_ratio'],
            'sortino_ratio': portfolio_metrics['sortino_ratio'],
            'max_drawdown': portfolio_metrics['max_drawdown'],
            'yearly_profit': yearly_profit,
        }
    return None


def analyze_portfolio():
    # --- Initial Setup ---
    # Define your stocks here. You can paste 50+ stocks.
    predefined_stocks = {
    "NSE_EQ|INE296A01032":"BAJFINANCE",
    "NSE_EQ|INE044A01036":"SUNPHARMA",
    "NSE_EQ|INE674K01013":"ABCAPITAL",
    "NSE_EQ|INE028A01039":"BANKBARODA",
    "NSE_EQ|INE692A01016":"UNIONBANK",
    "NSE_EQ|INE476A01022":"CANBK",
    "NSE_EQ|INE372C01037":"PRECWIRE",
    "NSE_EQ|INE560A01023":"INDIAGLYCO",
    "NSE_EQ|INE926K01017":"AHLEAST",
    "NSE_EQ|INE376G01013":"BIOCON",
    "NSE_EQ|INE849A01020":"TRENT",
    "NSE_EQ|INE364U01010":"IRFC",
    "NSE_EQ|INE133E01013":"TI",
    "NSE_EQ|INE397D01024":"BHARTIARTL",
    "NSE_EQ|INE931S01010":"ADANIENSOL",
    "NSE_EQ|INE199P01028":"GCSL",
    "NSE_EQ|INE474Q01031":"MEDANTA",
    "NSE_EQ|INE075I01017":"HCG",
    "NSE_EQ|INE0L2Y01011":"ARHAM",
    "NSE_EQ|INE591G01025":"COFORGE",
    "NSE_EQ|INE262H01021":"PERSISTENT"
    # Add more stocks here...
    }

    if not predefined_stocks:
        print("Error: No stocks defined.")
        return

    print(f"Starting analysis with {len(predefined_stocks)} manually defined stocks.")
    print(f"Parameters: Risk-Free Rate={RISK_FREE_RATE*100}%, Transaction Cost={TRANSACTION_COST*100}% per rebalance")

    # --- Fetch Benchmark Data ---
    print(f"\nFetching benchmark data for {BENCHMARK_TRADINGSYMBOL}...")
    benchmark_data = fetch_historical_data(BENCHMARK_INSTRUMENT_KEY, START_DATE, END_DATE)
    if benchmark_data.empty:
        print("Fatal Error: Could not fetch benchmark data.")
        return
    benchmark_metrics_overall = get_metrics(benchmark_data['returns'], None, RISK_FREE_RATE)
    print(f"Benchmark Sharpe: {benchmark_metrics_overall['sharpe_ratio']:.2f}")

    # --- Validate & Fetch Stock Data ---
    print(f"\nValidating stocks...")
    valid_stocks = {}
    all_stock_data = {} 
    start_date_dt = datetime.strptime(START_DATE, '%Y-%m-%d')
    end_date_dt = datetime.strptime(END_DATE, '%Y-%m-%d')

    total_stocks = len(predefined_stocks)
    for i, (instrument_key, tradingsymbol) in enumerate(predefined_stocks.items(), 1):
        print(f"Validating {i}/{total_stocks}: {tradingsymbol}...", end='\r')
        hist_data = fetch_historical_data(instrument_key, START_DATE, END_DATE)
        if not hist_data.empty:
            first_date = hist_data.index.min().to_pydatetime()
            last_date = hist_data.index.max().to_pydatetime()
            last_close_price = hist_data['close'].iloc[-1]

            if abs((first_date.date() - start_date_dt.date()).days) < 30 and (end_date_dt.date() - last_date.date()).days < 30 and last_close_price > 100:
                valid_stocks[instrument_key] = tradingsymbol
                all_stock_data[instrument_key] = {'symbol': tradingsymbol, 'data': hist_data}
    
    print(f"\nValidation complete. {len(valid_stocks)} stocks are valid.     ")

    if len(valid_stocks) < 5:
        print("Not enough valid stocks (minimum 5).")
        return

    # --- Smart Screening (Institutional Step) ---
    # If > 20 stocks, select top 20 by Sharpe Ratio to avoid combinatorial explosion
    selected_keys = list(valid_stocks.keys())
    if len(valid_stocks) > 20:
        print(f"\n--- Smart Screening: Reducing universe from {len(valid_stocks)} to 20 ---")
        stock_metrics = []
        for key in selected_keys:
            returns = all_stock_data[key]['data']['returns']
            sharpe = calculate_sharpe_ratio(returns, RISK_FREE_RATE)
            stock_metrics.append((key, sharpe))
        
        # Sort by Sharpe descending
        stock_metrics.sort(key=lambda x: x[1], reverse=True)
        selected_keys = [x[0] for x in stock_metrics[:20]]
        print(f"Selected top 20 stocks based on Sharpe Ratio.")
        for key in selected_keys:
            print(f"  - {valid_stocks[key]}")

    # --- Generate Combinations ---
    stock_combinations = list(itertools.combinations(selected_keys, 5))
    print(f"\nGenerated {len(stock_combinations)} unique 5-stock combinations.")
    print("Starting parallel evaluation...")

    # --- Parallel Execution ---
    # We use partial to pass constant arguments
    worker_func = partial(evaluate_combination, all_stock_data=all_stock_data, benchmark_metrics_overall=benchmark_metrics_overall)
    
    # Use 75% of available cores
    num_processes = max(1, int(multiprocessing.cpu_count() * 0.75))
    print(f"Using {num_processes} worker processes.")

    evaluated_portfolios = []
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Use imap_unordered to track progress
        total_combinations = len(stock_combinations)
        results_iterator = pool.imap_unordered(worker_func, stock_combinations)
        
        for i, result in enumerate(results_iterator, 1):
            if i % 10 == 0 or i == total_combinations:
                print(f"Evaluated {i}/{total_combinations} portfolios...", end='\r')
            if result is not None:
                evaluated_portfolios.append(result)
        
    print(f"\nEvaluation complete.                                ")

    # --- Reporting ---
    print(f"\nAnalysis complete. Found {len(evaluated_portfolios)} outperforming portfolios.")

    if not evaluated_portfolios:
        print("No portfolios met the criteria.")
        return

    sorted_portfolios = sorted(evaluated_portfolios, key=lambda x: x['total_profit_pct'], reverse=True)

    main_report_data = []
    for p in sorted_portfolios:
        main_report_data.append({
            'Portfolio': p['combination_symbols'],
            'Overall Profit (%)': p['total_profit_pct'],
            'Last 1 Year Profit (%)': p['profit_last_year_pct'],
            'Last 60 Days Profit (%)': p['profit_last_60_days_pct'],
            'Sharpe Ratio': p['sharpe_ratio'],
            'Sortino Ratio': p['sortino_ratio'],
            'Max Drawdown': p['max_drawdown']
        })
    
    main_report_df = pd.DataFrame(main_report_data)
    main_report_df.to_csv("report_rebalanced_portfolios.csv", index=False)
    print("Generated 'report_rebalanced_portfolios.csv'")

    yearly_report_data = []
    for p in sorted_portfolios:
        row = {'Portfolio': p['combination_symbols']}
        row.update({f'Profit_{year}_(%)': profit * 100 for year, profit in p['yearly_profit'].items()})
        yearly_report_data.append(row)
        
    yearly_report_df = pd.DataFrame(yearly_report_data).fillna(0)
    if 'Portfolio' in yearly_report_df.columns:
        cols = yearly_report_df.columns.tolist()
        year_cols = sorted([c for c in cols if c != 'Portfolio'])
        yearly_report_df = yearly_report_df[['Portfolio'] + year_cols]
    
    yearly_report_df.to_csv("report_yearly_profits.csv", index=False)
    print("Generated 'report_yearly_profits.csv'")

    print("\n--- Top 5 Portfolios ---")
    print(main_report_df.head(5).to_string())

if __name__ == "__main__":
    # Windows support for multiprocessing
    multiprocessing.freeze_support()
    analyze_portfolio()