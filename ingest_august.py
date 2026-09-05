import os
import io
import duckdb
import pandas as pd
import requests
from datetime import datetime, timedelta

DB_PATH = "data/candles.duckdb"
START_DATE = datetime(2026, 8, 1)
END_DATE = datetime(2026, 9, 4)

def fetch_and_ingest():
    os.makedirs("data", exist_ok=True)
    conn = duckdb.connect(DB_PATH)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv_candles (
            symbol VARCHAR,
            timestamp TIMESTAMP,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume BIGINT,
            series VARCHAR,
            delivery_qty BIGINT,
            delivery_pct DOUBLE
        )
    """)

    curr_date = START_DATE
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    while curr_date <= END_DATE:
        if curr_date.weekday() >= 5:  # Skip Saturday/Sunday
            curr_date += timedelta(days=1)
            continue

        d_str = curr_date.strftime("%d%m%Y")
        date_iso = curr_date.strftime("%Y-%m-%d")

        exists = conn.execute(f"SELECT COUNT(*) FROM ohlcv_candles WHERE CAST(timestamp AS DATE) = '{date_iso}'").fetchone()[0]
        if exists > 0:
            print(f"⏩ {date_iso} already exists in DB. Skipping.")
            curr_date += timedelta(days=1)
            continue

        url = f"https://archives.nseindia.com/products/content/sec_bhavdata_full_{d_str}.csv"
        print(f"📥 Downloading Bhavcopy for {date_iso}...")

        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                raw_df = pd.read_csv(io.StringIO(resp.text))
                raw_df.columns = [c.strip().upper() for c in raw_df.columns]

                eq_df = raw_df[raw_df['SERIES'].str.strip() == 'EQ'].copy()
                eq_df['timestamp'] = pd.to_datetime(eq_df['DATE1'].str.strip(), format="%d-%b-%Y")
                eq_df['symbol'] = eq_df['SYMBOL'].str.strip()
                eq_df['open'] = pd.to_numeric(eq_df['OPEN_PRICE'], errors='coerce')
                eq_df['high'] = pd.to_numeric(eq_df['HIGH_PRICE'], errors='coerce')
                eq_df['low'] = pd.to_numeric(eq_df['LOW_PRICE'], errors='coerce')
                eq_df['close'] = pd.to_numeric(eq_df['CLOSE_PRICE'], errors='coerce')
                eq_df['volume'] = pd.to_numeric(eq_df['TTL_TRD_QNTY'], errors='coerce').fillna(0).astype('int64')
                eq_df['series'] = 'EQ'
                eq_df['delivery_qty'] = pd.to_numeric(eq_df['DELIV_QTY'], errors='coerce').fillna(0).astype('int64')
                eq_df['delivery_pct'] = pd.to_numeric(eq_df['DELIV_PER'], errors='coerce').fillna(0.0)

                insert_df = eq_df[[
                    'symbol', 'timestamp', 'open', 'high', 'low', 'close', 
                    'volume', 'series', 'delivery_qty', 'delivery_pct'
                ]].dropna(subset=['symbol', 'close'])

                conn.register("tmp_insert", insert_df)
                conn.execute("INSERT INTO ohlcv_candles SELECT * FROM tmp_insert")
                print(f"✅ Ingested {len(insert_df)} stocks for {date_iso}")
            else:
                print(f"⚠️ Market holiday or data missing for {date_iso} (Status: {resp.status_code})")
        except Exception as e:
            print(f"❌ Failed for {date_iso}: {e}")

        curr_date += timedelta(days=1)

    total_count = conn.execute("SELECT COUNT(DISTINCT CAST(timestamp AS DATE)) FROM ohlcv_candles").fetchone()[0]
    print(f"\n🎉 Total Trading Days now in DB: {total_count}")
    conn.close()

if __name__ == "__main__":
    fetch_and_ingest()
