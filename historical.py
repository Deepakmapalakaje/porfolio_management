import requests
from datetime import datetime, timedelta
ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI3M0FIOUMiLCJqdGkiOiI2OTFkNWFiODFkZTI2NzdiZDkxNjkxMGMiLCJpc015dHRpQ2xpZW50IjpmYWxzZSwiaXNQbHJpUGxhbiI6dHJ1ZSwiaWF0IjoxNzYzNTMxNDQ4LCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3NjM1ODk2MDB9.7dPxOIudeYCSEo5HNVzWXtn2EhzDFWKwVRwICz7Aui4"

INSTRUMENT_KEY = "NSE_EQ|INE848E01016"  # Example: replace with your symbol

BASE_URL = "https://api.upstox.com/v3/historical-candle"

def fetch_chunk(from_date, to_date):
    """Fetch candles for a specific date range."""
    url = f"{BASE_URL}/{INSTRUMENT_KEY}/day/{to_date}/{from_date}"
    headers = {  # Corrected header name from 'Authorization' to 'authorization'
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Accept": "application/json"
    }

    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        return resp.json().get("data", {}).get("candles", [])
    else:
        print(f"Error {resp.status_code}: {resp.text}")
        return []

def generate_date_ranges(start_year=2000):
    """Generate safe 10-year chunks until today."""
    today = datetime.now().date()
    
    start = datetime(start_year, 1, 1).date()
    ranges = []

    while start <= today:
        end = start.replace(year=start.year + 9)
        if end > today:
            end = today
        ranges.append((start, end))
        start = end + timedelta(days=1)

    return ranges


def fetch_full_history():
    date_chunks = generate_date_ranges()
    full_data = []

    print("Fetching full-day historical data...")
    for (start, end) in date_chunks:
        print(f"Fetching: {start} to {end}")
        candles = fetch_chunk(str(start), str(end))
        full_data.extend(candles)
        print(f"   Retrieved {len(candles)} candles.")

    print(f"\nCompleted! Total candles: {len(full_data)}")
    return full_data


if __name__ == "__main__":
    candles = fetch_full_history()

    # Optionally save as CSV
    import csv
    with open("historical_daily_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume", "oi"])
        writer.writerows(candles)

    print("Saved to historical_daily_data.csv")
