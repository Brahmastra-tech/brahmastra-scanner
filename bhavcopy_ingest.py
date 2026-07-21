import os
import io
import zipfile
import requests
import duckdb
import polars as pl
from datetime import datetime

class BhavcopyIngestor:
    def __init__(self, db_path="data/candles.duckdb"):
        self.db_path = db_path
        os.makedirs("data", exist_ok=True)
        self._init_db()

    def _init_db(self):
        with duckdb.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv_candles (
                    symbol VARCHAR,
                    timeframe VARCHAR,
                    timestamp TIMESTAMP,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume BIGINT,
                    open_interest BIGINT DEFAULT 0,
                    PRIMARY KEY (symbol, timeframe, timestamp)
                );
            """)

    def get_fno_symbols_list(self) -> set:
        url = "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "Accept": "*/*"
        }
        try:
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code == 200:
                df_fno = pl.read_csv(io.BytesIO(res.content))
                fno_symbols = [s.strip() for s in df_fno.select(df_fno.columns[1]).to_series().to_list() if s]
                return set(fno_symbols)
        except Exception as e:
            print(f"⚠️ Warning: Could not fetch F&O list ({e}). Falling back to full EQ list.")
        return set()

    def fetch_and_store_daily(self, date_str: str):
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        date_formatted = dt.strftime("%Y%m%d")
        
        fno_allowed_symbols = self.get_fno_symbols_list()
        url = f"https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{date_formatted}_F_0000.csv.zip"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/"
        }

        print(f"Downloading Bhavcopy for {date_str}...")
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        res = session.get(url, headers=headers, timeout=15)
        
        if res.status_code != 200:
            print(f"❌ Failed to fetch data (Status {res.status_code}). Market closed or file unavailable.")
            return

        with zipfile.ZipFile(io.BytesIO(res.content)) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                df = pl.read_csv(f.read())

        df = df.rename({col: col.strip() for col in df.columns})

        # Filter Equity series (SctySrs == "EQ")
        df_filtered = df.filter(pl.col("SctySrs") == "EQ")
        if fno_allowed_symbols:
            df_filtered = df_filtered.filter(pl.col("TckrSymb").is_in(list(fno_allowed_symbols)))

        # Exact NSE UDiFF Column Schema
        df_clean = df_filtered.select([
            pl.col("TckrSymb").alias("symbol"),
            pl.lit("1D").alias("timeframe"),
            pl.lit(dt).alias("timestamp"),
            pl.col("OpnPric").cast(pl.Float64).alias("open"),
            pl.col("HghPric").cast(pl.Float64).alias("high"),
            pl.col("LwPric").cast(pl.Float64).alias("low"),
            pl.col("ClsPric").cast(pl.Float64).alias("close"),
            pl.col("TtlTradgVol").cast(pl.Int64).alias("volume"),
            pl.col("OpnIntrst").cast(pl.Int64).alias("open_interest")
        ])

        with duckdb.connect(self.db_path) as conn:
            conn.register("temp_bhavcopy", df_clean.to_arrow())
            conn.execute("""
                INSERT OR REPLACE INTO ohlcv_candles 
                SELECT * FROM temp_bhavcopy
            """)
            
        print(f"✓ Stored {len(df_clean)} F&O EQ stocks for {date_str}.")

if __name__ == "__main__":
    ingestor = BhavcopyIngestor()
    today_str = datetime.today().strftime("%Y-%m-%d")
    ingestor.fetch_and_store_daily(today_str)