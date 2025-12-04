import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, timezone
from scipy import stats
from urllib.parse import quote


# --- Configuration ---
ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI3M0FIOUMiLCJqdGkiOiI2OTFiMWFkZGIyNjM3MzRiMDhmNTRiMzgiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6dHJ1ZSwiaWF0IjoxNzYzMzg0MDI5LCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3NjM0MTY4MDB9.pRI8azceXp7Zj9cxuu1Dth8HWre7zzPrrQJI5SIUdTs"
BASE_URL = "https://api.upstox.com/v3/historical-candle"
START_DATE = "2019-01-01"
END_DATE = "2025-11-17"
RISK_FREE_RATE = 0.0
BENCHMARK_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"
BENCHMARK_TRADINGSYMBOL = "NIFTY 50"


# --- Data Fetching ---
def fetch_historical_data(instrument_key, from_date, to_date):
    """Fetch historical candle data for a given instrument and date range."""
    # URL encode the entire instrument_key
    encoded_instrument_key = quote(instrument_key, safe='')
    # Correct URL format: /unit/interval/ not /unit/:interval/
    url = f"{BASE_URL}/{encoded_instrument_key}/days/1/{to_date}/{from_date}"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Accept": "application/json"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json().get("data", {}).get("candles", [])
        if data:
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
            df["returns"] = df["close"].pct_change()
            return df.dropna()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {instrument_key}: {e}")
        if hasattr(response, 'text'):
            print(f"Response: {response.text}")
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


def calculate_information_ratio(returns, benchmark_returns):
    if returns.empty or benchmark_returns.empty: return 0
    return_difference = returns - benchmark_returns
    tracking_error = np.std(return_difference)
    if tracking_error == 0: return 0
    return np.mean(return_difference) / tracking_error * np.sqrt(252)


def calculate_r_squared_and_jensens_alpha(returns, benchmark_returns, risk_free_rate):
    if returns.empty or benchmark_returns.empty: return 0, 0
    df = pd.DataFrame({'returns': returns, 'benchmark': benchmark_returns}).dropna()
    if len(df) < 2: return 0, 0
    
    beta, intercept, r_value, _, _ = stats.linregress(df['benchmark'], df['returns'])
    r_squared = r_value**2
    
    daily_rf = risk_free_rate / 252
    jensens_alpha = np.mean(df['returns']) - (daily_rf + beta * (np.mean(df['benchmark']) - daily_rf))
    return r_squared, jensens_alpha * 252


def get_metrics(returns, benchmark_returns, risk_free_rate):
    metrics = {}
    metrics['sharpe_ratio'] = calculate_sharpe_ratio(returns, risk_free_rate)
    metrics['sortino_ratio'] = calculate_sortino_ratio(returns, risk_free_rate)
    metrics['max_drawdown'] = calculate_max_drawdown(returns)
    metrics['std_dev'] = calculate_std_dev(returns)
    if benchmark_returns is not None:
        metrics['information_ratio'] = calculate_information_ratio(returns, benchmark_returns)
        metrics['r_squared'], metrics['jensens_alpha'] = calculate_r_squared_and_jensens_alpha(returns, benchmark_returns, risk_free_rate)
    else:
        metrics['information_ratio'] = np.nan
        metrics['r_squared'] = np.nan
        metrics['jensens_alpha'] = np.nan
    return metrics


# --- Main Analysis Logic ---
def analyze_instruments():
    try:
        nse_df = pd.read_csv("NSE.csv")
        # --- Filter for EQUITY instruments with alphabetic trading symbols ---
        nse_df = nse_df[nse_df['tradingsymbol'].str.isalpha()]
        equity_df = nse_df[nse_df["instrument_type"] == "EQUITY"]
    except FileNotFoundError:
        print("Error: NSE.csv not found.")
        return

    print(f"Fetching benchmark data for {BENCHMARK_TRADINGSYMBOL}...")
    benchmark_data = fetch_historical_data(BENCHMARK_INSTRUMENT_KEY, START_DATE, END_DATE)
    if benchmark_data.empty:
        print("Could not fetch benchmark data. Aborting.")
        return

    # Calculate benchmark metrics for all periods
    benchmark_metrics_overall = get_metrics(benchmark_data['returns'], None, RISK_FREE_RATE)
    
    ist_tz = timezone(timedelta(hours=5, minutes=30))
    current_time = datetime.now(ist_tz)
    last_year_benchmark_data = benchmark_data[benchmark_data.index >= (current_time - timedelta(days=365))]
    benchmark_metrics_last_year = get_metrics(last_year_benchmark_data['returns'], None, RISK_FREE_RATE)

    quarterly_benchmark_returns = benchmark_data["returns"].resample('QE')
    avg_benchmark_metrics_quarterly = {
        'sharpe_ratio': quarterly_benchmark_returns.apply(lambda x: calculate_sharpe_ratio(x, risk_free_rate=RISK_FREE_RATE)).mean(),
        'sortino_ratio': quarterly_benchmark_returns.apply(lambda x: calculate_sortino_ratio(x, risk_free_rate=RISK_FREE_RATE)).mean(),
        'max_drawdown': quarterly_benchmark_returns.apply(lambda x: calculate_max_drawdown(x)).mean(),
        'std_dev': quarterly_benchmark_returns.apply(lambda x: calculate_std_dev(x)).mean()
    }

    report1_data, report2_data, report3_data = [], [], []

    for _, row in equity_df.iterrows():
        instrument_key = row["instrument_key"]
        tradingsymbol = row["tradingsymbol"]
        print(f"Processing: {tradingsymbol} ({instrument_key})")

        hist_data = fetch_historical_data(instrument_key, START_DATE, END_DATE)
        if hist_data.empty: continue

        # --- SELECTION CRITERIA: Volume spike between 2025-11-01 and 2025-11-17 ---
        selection_start_date = '2025-01-01'
        selection_end_date = '2025-11-17'
        volume_check_df = hist_data[(hist_data.index >= selection_start_date) & (hist_data.index <= selection_end_date)].copy()

        if len(volume_check_df) > 1:
            volume_check_df['prev_volume'] = volume_check_df['volume'].shift(1)
            # Check if any day's volume is 100x the previous day's volume
            if (volume_check_df['volume'] >= 50 * volume_check_df['prev_volume']).any():
                print(f"  -> Selected: {tradingsymbol} due to volume spike.")
                
                # --- Proceed with metric calculations and reporting ---
                # 1. Overall Report
                aligned_data = pd.DataFrame({'instrument': hist_data['returns'], 'benchmark': benchmark_data['returns']}).dropna()
                instr_metrics_overall = get_metrics(aligned_data['instrument'], aligned_data['benchmark'], RISK_FREE_RATE)
                report1_data.append([instrument_key, tradingsymbol] + list(instr_metrics_overall.values()))

                # 2. Quarterly Average Report
                quarterly_returns = hist_data["returns"].resample('QE')
                avg_instr_metrics_quarterly = {
                    'sharpe_ratio': quarterly_returns.apply(lambda x: calculate_sharpe_ratio(x, risk_free_rate=RISK_FREE_RATE)).mean(),
                    'sortino_ratio': quarterly_returns.apply(lambda x: calculate_sortino_ratio(x, risk_free_rate=RISK_FREE_RATE)).mean(),
                    'max_drawdown': quarterly_returns.apply(lambda x: calculate_max_drawdown(x)).mean(),
                    'std_dev': quarterly_returns.apply(lambda x: calculate_std_dev(x)).mean()
                }
                report2_data.append([instrument_key, tradingsymbol] + list(avg_instr_metrics_quarterly.values()))

                # 3. Last One Year Report
                last_year_data = hist_data[hist_data.index >= (current_time - timedelta(days=365))]
                if not last_year_data.empty:
                    aligned_last_year = pd.DataFrame({'instrument': last_year_data['returns'], 'benchmark': last_year_benchmark_data['returns']}).dropna()
                    if not aligned_last_year.empty:
                        instr_metrics_last_year = get_metrics(aligned_last_year['instrument'], aligned_last_year['benchmark'], RISK_FREE_RATE)
                        report3_data.append([instrument_key, tradingsymbol] + list(instr_metrics_last_year.values()))

    # --- Save Reports ---
    report_cols = ["instrument_key", "tradingsymbol", "sharpe_ratio", "sortino_ratio", "max_drawdown", "std_dev", "information_ratio", "r_squared", "jensens_alpha"]
    
    # Report 1: Overall
    if report1_data:
        benchmark_row_1 = [BENCHMARK_INSTRUMENT_KEY, BENCHMARK_TRADINGSYMBOL] + list(benchmark_metrics_overall.values())
        df1 = pd.DataFrame([benchmark_row_1] + report1_data, columns=report_cols)
        df1 = pd.concat([df1.iloc[[0]], df1.iloc[1:].sort_values(by='sharpe_ratio', ascending=False)], ignore_index=True)
        df1.to_csv("report_overall.csv", index=False)
        print("Successfully generated report_overall.csv")
    else:
        print("No instruments met the criteria for the Overall report.")

    # Report 2: Quarterly Average
    if report2_data:
        report2_cols = ["instrument_key", "tradingsymbol", "avg_quarterly_sharpe", "avg_quarterly_sortino", "avg_quarterly_max_drawdown", "avg_quarterly_std_dev"]
        benchmark_row_2 = [BENCHMARK_INSTRUMENT_KEY, BENCHMARK_TRADINGSYMBOL] + list(avg_benchmark_metrics_quarterly.values())
        df2 = pd.DataFrame([benchmark_row_2] + report2_data, columns=report2_cols)
        df2 = pd.concat([df2.iloc[[0]], df2.iloc[1:].sort_values(by='avg_quarterly_sharpe', ascending=False)], ignore_index=True)
        df2.to_csv("report_quarterly_avg.csv", index=False)
        print("Successfully generated report_quarterly_avg.csv")
    else:
        print("No instruments met the criteria for the Quarterly Average report.")
    
    # Report 3: Last Year
    if report3_data:
        benchmark_row_3 = [BENCHMARK_INSTRUMENT_KEY, BENCHMARK_TRADINGSYMBOL] + list(benchmark_metrics_last_year.values())
        df3 = pd.DataFrame([benchmark_row_3] + report3_data, columns=report_cols)
        df3 = pd.concat([df3.iloc[[0]], df3.iloc[1:].sort_values(by='sharpe_ratio', ascending=False)], ignore_index=True)
        df3.to_csv("report_last_year.csv", index=False)
        print("Successfully generated report_last_year.csv")
    else:
        print("No instruments met the criteria for the Last Year report.")


if __name__ == "__main__":
    analyze_instruments()