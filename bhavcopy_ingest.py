import os
import io
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

def init_and_migrate_db(conn):
    """Initializes DuckDB schema and ensures all required columns exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv_candles (
            symbol VARCHAR,
            timestamp DATE,
            timeframe VARCHAR DEFAULT 'D',
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            PRIMARY KEY (symbol, timestamp)
        );
    """)

    conn.execute("ALTER TABLE ohlcv_candles ADD COLUMN IF NOT EXISTS timeframe VARCHAR DEFAULT 'D';")
    conn.execute("ALTER TABLE ohlcv_candles ADD COLUMN IF NOT EXISTS delivery_qty DOUBLE;")
    conn.execute("ALTER TABLE ohlcv_candles ADD COLUMN IF NOT EXISTS delivery_pct DOUBLE;")
    conn.execute("ALTER TABLE ohlcv_candles ADD COLUMN IF NOT EXISTS open_interest DOUBLE;")


def fetch_nse_bhavcopy(target_date: datetime):
    """Fetches full NSE daily sec_bhavdata_full CSV including Delivery metrics."""
    date_str = target_date.strftime("%d%m%Y")
    url = f"https://archives.nseindia.com/products/content/sec_bhavdata_full_{date_str}.csv"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            df.columns = df.columns.str.strip()
            return df
        else:
            return None
    except Exception as e:
        return None


def process_and_store():
    os.makedirs("data", exist_ok=True)
    conn = duckdb.connect(DB_PATH)
    init_and_migrate_db(conn)

    db_cols_info = conn.execute("DESCRIBE ohlcv_candles").fetchall()
    target_table_cols = [col[0].lower() for col in db_cols_info]

    today = datetime.now()
    
    # FETCH LAST 60 CALENDAR DAYS TO SEED ENTIRE 40+ TRADING DAY HISTORY
    print("📥 Ingesting trailing 60 days of NSE Bhavcopy data into DuckDB...")
    ingested_count = 0

    for i in range(60, -1, -1):
        target_date = today - timedelta(days=i)
        
        if target_date.weekday() >= 5:  # Skip weekends
            continue

        date_db_format = target_date.strftime("%Y-%m-%d")
        
        # Check if already present
        existing = conn.execute("SELECT COUNT(*) FROM ohlcv_candles WHERE timestamp = ?", [date_db_format]).fetchone()[0]
        if existing > 0:
            continue

        df_raw = fetch_nse_bhavcopy(target_date)

        if df_raw is None or df_raw.empty:
            continue

        # Filter strictly for EQ Series
        df_eq = df_raw[df_raw['SERIES'].astype(str).str.strip() == 'EQ'].copy()

        if df_eq.empty:
            continue

        df_eq['symbol'] = df_eq['SYMBOL'].astype(str).str.strip()
        df_eq['timestamp'] = pd.to_datetime(df_eq['DATE1'].astype(str).str.strip(), format="%d-%b-%Y").dt.strftime("%Y-%m-%d")
        df_eq['timeframe'] = 'D'
        df_eq['open'] = pd.to_numeric(df_eq['OPEN_PRICE'], errors='coerce')
        df_eq['high'] = pd.to_numeric(df_eq['HIGH_PRICE'], errors='coerce')
        df_eq['low'] = pd.to_numeric(df_eq['LOW_PRICE'], errors='coerce')
        df_eq['close'] = pd.to_numeric(df_eq['CLOSE_PRICE'], errors='coerce')
        df_eq['volume'] = pd.to_numeric(df_eq['TTL_TRD_QNTY'], errors='coerce')
        df_eq['delivery_qty'] = pd.to_numeric(df_eq['DELIV_QTY'], errors='coerce').fillna(0.0)
        df_eq['delivery_pct'] = pd.to_numeric(df_eq['DELIV_PER'], errors='coerce').fillna(0.0)
        
        if 'OPEN_INT' in df_eq.columns:
            df_eq['open_interest'] = pd.to_numeric(df_eq['OPEN_INT'], errors='coerce').fillna(0.0)
        else:
            df_eq['open_interest'] = 0.0

        insert_cols = [col for col in target_table_cols if col in df_eq.columns]
        final_df = df_eq[insert_cols].dropna(subset=['symbol', 'close'])

        col_names_str = ", ".join(insert_cols)
        
        query = f"""
            INSERT OR REPLACE INTO ohlcv_candles ({col_names_str})
            SELECT {col_names_str} FROM final_df
        """
        conn.execute(query)
        ingested_count += 1
        print(f"  └─ Ingested {date_db_format} ({len(final_df)} symbols)")

    total_rows = conn.execute("SELECT COUNT(*) FROM ohlcv_candles").fetchone()[0]
    print(f"✅ Ingestion Complete! Total DB Candle Rows: {total_rows}")
    conn.close()


if __name__ == "__main__":
    process_and_store()