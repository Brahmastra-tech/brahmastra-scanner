import os
import io
import zipfile
import requests
import duckdb
import pandas as pd
from datetime import datetime, timedelta

# ==========================================
# CONFIGURATION
# ==========================================
DB_PATH = "data/candles.duckdb"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*"
}

def init_db(conn):
    """Initializes DuckDB schema with delivery and open interest support."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv_candles (
            symbol VARCHAR,
            timestamp DATE,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            delivery_qty DOUBLE,
            delivery_pct DOUBLE,
            open_interest DOUBLE,
            PRIMARY KEY (symbol, timestamp)
        );
    """)

def fetch_nse_bhavcopy(target_date: datetime):
    """Fetches full NSE daily sec_bhavdata_full CSV including Delivery and OI metrics."""
    date_str = target_date.strftime("%d%m%Y")
    url = f"https://archives.nseindia.com/products/content/sec_bhavdata_full_{date_str}.csv"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            df.columns = df.columns.str.strip()
            return df
        else:
            print(f"⚠️ No Bhavcopy available for {target_date.strftime('%Y-%m-%d')} (HTTP {response.status_code})")
            return None
    except Exception as e:
        print(f"❌ Error downloading Bhavcopy for {target_date.strftime('%Y-%m-%d')}: {e}")
        return None

def process_and_store():
    os.makedirs("data", exist_ok=True)
    conn = duckdb.connect(DB_PATH)
    init_db(conn)

    # Ingest last 5 trading days if missing
    today = datetime.now()
    for i in range(5, -1, -1):
        target_date = today - timedelta(days=i)
        
        # Skip weekends
        if target_date.weekday() >= 5:
            continue

        date_db_format = target_date.strftime("%Y-%m-%d")
        
        # Check if date already ingested
        existing = conn.execute("SELECT COUNT(*) FROM ohlcv_candles WHERE timestamp = ?", [date_db_format]).fetchone()[0]
        if existing > 0:
            print(f"ℹ️ Date {date_db_format} already in DuckDB. Skipping.")
            continue

        print(f"📥 Fetching NSE Bhavcopy for {date_db_format}...")
        df_raw = fetch_nse_bhavcopy(target_date)

        if df_raw is None or df_raw.empty:
            continue

        # Filter strictly for EQ Series (Equity/F&O stocks)
        df_eq = df_raw[df_raw['SERIES'].astype(str).str.strip() == 'EQ'].copy()

        if df_eq.empty:
            continue

        # Map NSE columns to DuckDB schema
        df_eq['symbol'] = df_eq['SYMBOL'].astype(str).str.strip()
        df_eq['timestamp'] = pd.to_datetime(df_eq['DATE1'].astype(str).str.strip(), format="%d-%b-%Y").dt.strftime("%Y-%m-%d")
        df_eq['open'] = pd.to_numeric(df_eq['OPEN_PRICE'], errors='coerce')
        df_eq['high'] = pd.to_numeric(df_eq['HIGH_PRICE'], errors='coerce')
        df_eq['low'] = pd.to_numeric(df_eq['LOW_PRICE'], errors='coerce')
        df_eq['close'] = pd.to_numeric(df_eq['CLOSE_PRICE'], errors='coerce')
        df_eq['volume'] = pd.to_numeric(df_eq['TTL_TRD_QNTY'], errors='coerce')
        df_eq['delivery_qty'] = pd.to_numeric(df_eq['DELIV_QTY'], errors='coerce').fillna(0)
        df_eq['delivery_pct'] = pd.to_numeric(df_eq['DELIV_PER'], errors='coerce').fillna(0)
        df_eq['open_interest'] = 0.0  # Default 0.0 for EQ series if derivative file separate

        final_df = df_eq[['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'delivery_qty', 'delivery_pct', 'open_interest']].dropna(subset=['symbol', 'close'])

        # Upsert into DuckDB
        conn.execute("""
            INSERT OR REPLACE INTO ohlcv_candles 
            SELECT * FROM final_df
        """)
        print(f"✅ Successfully ingested {len(final_df)} stocks for {date_db_format}.")

    conn.close()

if __name__ == "__main__":
    process_and_store()