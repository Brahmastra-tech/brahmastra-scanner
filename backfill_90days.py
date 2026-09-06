import os
import io
import duckdb
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

DB_PATH = "data/candles.duckdb"
SIGNALS_CSV = "data/signals.csv"

START_DATE = datetime(2026, 8, 1)
END_DATE = datetime(2026, 9, 5)

TARGET_X = 3.0
MIN_PRICE = 100.0
MIN_MARKET_CAP = 51_000_000_000.0

CE_PERIOD = 22
CE_MULTIPLIER = 3.0

NSE_FO_SYMBOLS = frozenset({
    "AARTIIND", "ABB", "ABBOTINDIA", "ABCAPITAL", "ABFRL", "ACC", "ADANIENT",
    "ADANIPORTS", "ALKEM", "AMBUJACEM", "APOLLOHOSP", "APOLLOTYRE", "ASHOKLEY",
    "ASIANPAINT", "ASTRAL", "ATUL", "AUBANK", "AUROPHARMA", "AXISBANK", "BAJAJ-AUTO",
    "BAJAJFINSV", "BAJFINANCE", "BALKRISIND", "BALRAMCHIN", "BANDHANBNK", "BANKBARODA",
    "BATAINDIA", "BEL", "BERGEPAINT", "BHARATFORG", "BHARTIARTL", "BHEL", "BIOCON",
    "BOSCHLTD", "BPCL", "BRITANNIA", "BSOFT", "CANBK", "CANFINHOME", "CHAMBLFERT",
    "CHOLAFIN", "CIPLA", "COALINDIA", "COFORGE", "COLPAL", "CONCOR", "COROMANDEL",
    "CROMPTON", "CUB", "CUMMINSIND", "DABUR", "DALBHARAT", "DEEPAKNTR", "DIVISLAB",
    "DIXON", "DLF", "DRREDDY", "EICHERMOT", "ESCORTS", "EXIDEIND", "FEDERALBNK",
    "GAIL", "GLENMARK", "GMRINFRA", "GNFC", "GODREJCP", "GODREJPROP", "GRANULES",
    "GRASIM", "GUJGASLTD", "HAL", "HAVELLS", "HCLTECH", "HDFCAMC", "HDFCBANK",
    "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDPETRO", "HINDUNILVR", "ICICIBANK",
    "ICICIGI", "ICICIPRULI", "IDEA", "IDFC", "IDFCFIRSTB", "IEX", "IGL", "INDHOTEL",
    "INDIACEM", "INDIAMART", "INDIGO", "INDUSINDBK", "INDUSTOWER", "INFY", "IOC",
    "IPCALAB", "IRCTC", "ITC", "JINDALSTEL", "JKCEMENT", "JSWSTEEL", "JUBLFOOD",
    "KOTAKBANK", "LALPATHLAB", "LAURUSLABS", "LICHSGFIN", "LT", "LTIM", "LTTS",
    "LUPIN", "M&M", "M&MFIN", "MANAPPURAM", "MARICO", "MARUTI", "MCDOWELL-N",
    "MCX", "METROPOLIS", "MFSL", "MGL", "MOTHERSON", "MPHASIS", "MRF", "MUTHOOTFIN",
    "NATIONALUM", "NAUKRI", "NAVINFLUOR", "NESTLEIND", "NMDC", "NTPC", "OBEROIRLTY",
    "OFSS", "ONGC", "PAGEIND", "PEL", "PERSISTENT", "PETRONET", "PFC", "PIDILITIND",
    "PIIND", "PNB", "POLYCAB", "POONAWALLA", "POWERGRID", "PVRINOX", "RAMCOCEM",
    "RBLBANK", "RECLTD", "RELIANCE", "SAIL", "SBICARD", "SBILIFE", "SBIN", "SHREECEM",
    "SHRIRAMFIN", "SIEMENS", "SRF", "SUNPHARMA", "SUNTV", "SYNGENE", "TATACHEM",
    "TATACOMM", "TATACONSUM", "TATAMOTORS", "TATAPOWER", "TATASTEEL", "TCS", "TECHM",
    "TITAN", "TORNTPHARM", "TRENT", "TVSMOTOR", "UBL", "ULTRACEMCO", "UPL", "VEDL",
    "VOLTAS", "WIPRO", "ZEEL"
})


def auto_ingest_missing_august():
    print("🔍 Checking and updating DuckDB for August sessions...")
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
    headers = {"User-Agent": "Mozilla/5.0"}
    while curr_date <= END_DATE:
        if curr_date.weekday() < 5:
            d_str = curr_date.strftime("%d%m%Y")
            date_iso = curr_date.strftime("%Y-%m-%d")
            exists = conn.execute(f"SELECT COUNT(*) FROM ohlcv_candles WHERE CAST(timestamp AS DATE) = '{date_iso}'").fetchone()[0]
            if exists == 0:
                url = f"https://archives.nseindia.com/products/content/sec_bhavdata_full_{d_str}.csv"
                try:
                    resp = requests.get(url, headers=headers, timeout=8)
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

                        insert_df = eq_df[['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'series', 'delivery_qty', 'delivery_pct']].dropna(subset=['symbol', 'close'])
                        conn.register("tmp_insert", insert_df)
                        conn.execute("INSERT INTO ohlcv_candles SELECT * FROM tmp_insert")
                        print(f"📥 Bhavcopy ingested: {date_iso}")
                except Exception as e:
                    pass
        curr_date += timedelta(days=1)
    conn.close()


def compute_chandelier_exit(df, period=22, mult=3.0):
    high, low, close_prev = df['High'], df['Low'], df['Close'].shift(1)
    tr = pd.concat([high - low, (high - close_prev).abs(), (low - close_prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    highest_high = high.rolling(window=period, min_periods=period).max()
    return highest_high - (mult * atr), atr


def run_august_to_date_backfill():
    auto_ingest_missing_august()
    print("🚀 Backfilling 30-Day Pure F&O Signals...")

    conn = duckdb.connect(DB_PATH)
    cols_info = [c[0].lower() for c in conn.execute("DESCRIBE ohlcv_candles").fetchall()]
    
    select_parts = [
        "symbol AS Symbol", "CAST(timestamp AS DATE) AS Date",
        "open AS Open", "high AS High", "low AS Low", "close AS Close", "volume AS Volume"
    ]
    select_parts.append("market_cap AS MarketCap" if "market_cap" in cols_info else "0.0 AS MarketCap")
    select_parts.append("delivery_qty AS DeliveryQty" if "delivery_qty" in cols_info else "volume * 0.45 AS DeliveryQty")
    select_parts.append("delivery_pct AS DeliveryPct" if "delivery_pct" in cols_info else "45.0 AS DeliveryPct")
    select_parts.append("delta_volume AS Delta_Volume" if "delta_volume" in cols_info else "CASE WHEN close >= open THEN volume * 0.55 ELSE -(volume * 0.55) END AS Delta_Volume")
    select_parts.append("order_flow_delta AS Order_Flow_Delta" if "order_flow_delta" in cols_info else "CASE WHEN close >= open THEN 1.0 ELSE -1.0 END AS Order_Flow_Delta")
    select_parts.append("ce_buy_flow AS CE_Buy_Flow" if "ce_buy_flow" in cols_info else "TRUE AS CE_Buy_Flow")

    df_raw = conn.execute(f"SELECT {', '.join(select_parts)} FROM ohlcv_candles WHERE UPPER(series) = 'EQ' AND close > {MIN_PRICE} ORDER BY symbol, timestamp ASC").df()
    conn.close()

    if df_raw.empty:
        return

    df_raw["Symbol_Clean"] = df_raw["Symbol"].astype(str).str.upper().str.strip().str.replace(r"-EQ$", "", regex=True)
    df_raw = df_raw[df_raw["Symbol_Clean"].isin(NSE_FO_SYMBOLS)].copy()
    df_raw["Date_DT"] = pd.to_datetime(df_raw["Date"])

    cutoff_date = pd.to_datetime("2026-08-01")
    valid_signals = []

    for symbol, df_sym in df_raw.groupby('Symbol_Clean'):
        if len(df_sym) < 15:
            continue

        df = df_sym.copy().sort_values("Date_DT").reset_index(drop=True)
        df['CE_Long'], df['ATR'] = compute_chandelier_exit(df, period=CE_PERIOD, mult=CE_MULTIPLIER)
        df['Prev_Delta'] = df['Delta_Volume'].shift(1).fillna(0.0)
        df['Avg_Delta_5'] = df['Delta_Volume'].abs().shift(1).rolling(5, min_periods=3).mean().fillna(0.0)
        df['Day_Range'] = (df['High'] - df['Low']).clip(lower=1e-5)
        df['Close_Location'] = ((df['Close'] - df['Low']) / df['Day_Range']).fillna(0.5)

        for i in range(len(df)):
            row = df.iloc[i]
            if row['Date_DT'] < cutoff_date or row['Close'] <= MIN_PRICE:
                continue

            cond_chandelier = (row['Close'] > row['CE_Long']) if pd.notna(row['CE_Long']) else True
            cond_ce_buy = bool(row['CE_Buy_Flow'])
            cond_order_flow = (row['Order_Flow_Delta'] > 0) and (row['Delta_Volume'] > 0)

            prev_d, curr_d, avg_d5 = row['Prev_Delta'], row['Delta_Volume'], row['Avg_Delta_5']
            cond_surge_prev = (prev_d > 0 and curr_d >= 1.70 * prev_d) or (prev_d < 0 and curr_d >= 1.70 * abs(prev_d)) or (prev_d == 0 and curr_d > 0)
            cond_surge_avg = curr_d >= (2.0 * avg_d5)

            if not (cond_chandelier and cond_ce_buy and cond_order_flow and cond_surge_prev and cond_surge_avg):
                continue

            deliv_score = float(np.clip((row['DeliveryPct'] / 70.0 * 50.0), 10, 50))
            close_score = float(np.clip(row['Close_Location'] * 50.0, 10, 50))
            brs_score = round(deliv_score + close_score, 2)

            entry = round(float(row['High']) + 0.05, 2)
            ce_stop = round(float(row['CE_Long']), 2) if pd.notna(row['CE_Long']) else round(float(row['Low']), 2)
            sl = min(round(float(row['Low']), 2), ce_stop)
            risk = max(entry - sl, float(row['Close']) * 0.01)
            target = round(entry + (risk * TARGET_X), 2)

            deliv_pct_val = round(float(np.nan_to_num(row['DeliveryPct'], nan=0.0)), 2)
            surge_ratio = round(float(curr_d / (avg_d5 + 1e-5)), 2)

            valid_signals.append({
                "Date": row['Date_DT'].strftime("%d-%m-%Y"),
                "Date_DT": row['Date_DT'],
                "Symbol": symbol,
                "Timeframe": "D",
                "Type": "PRE_BREAKOUT",
                "Pattern": "PRE_BREAKOUT",
                "BRS_Score": brs_score,
                "Entry": entry,
                "SL": sl,
                "Target": target,
                "Close": round(float(row['Close']), 2),
                "Volume": int(np.nan_to_num(row['Volume'], nan=0)),
                "DeliveryQty": int(np.nan_to_num(row['DeliveryQty'], nan=0)),
                "DeliveryPct": deliv_pct_val,
                "DelivSpikeRatio": surge_ratio
            })

    os.makedirs("data", exist_ok=True)
    clean_columns = ["Date", "Symbol", "Timeframe", "Type", "Pattern", "BRS_Score", "Entry", "SL", "Target", "Close", "Volume", "DeliveryQty", "DeliveryPct", "DelivSpikeRatio"]
    if valid_signals:
        df_out = pd.DataFrame(valid_signals).sort_values(by=['Date_DT', 'BRS_Score'], ascending=[False, False])
        df_out.drop(columns=['Date_DT'])[clean_columns].to_csv(SIGNALS_CSV, index=False)
        print(f"✅ Generated {len(df_out)} historical F&O signals in {SIGNALS_CSV}!")


if __name__ == "__main__":
    run_august_to_date_backfill()
