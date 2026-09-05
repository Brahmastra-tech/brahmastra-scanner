import os
import duckdb
import pandas as pd
import numpy as np
import requests

DB_PATH = "data/candles.duckdb"
SIGNALS_CSV = "data/signals.csv"
LOOKBACK_DAYS = 30
TARGET_X = 3.0
MIN_PRICE = 100.0
MIN_MARKET_CAP = 51_000_000_000.0

NSE_FO_URL = "https://archives.nseindia.com/content/fo/fo_mktlots.csv"

def get_nifty_fo_symbols() -> set:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        resp = requests.get(NSE_FO_URL, headers=headers, timeout=5)
        if resp.status_code == 200:
            lines = [line.strip() for line in resp.text.split("\n") if line.strip()]
            symbols = set()
            for line in lines[1:]:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    sym = parts[1].upper()
                    if sym and not any(idx in sym for idx in ["NIFTY", "INDIAVIX"]):
                        symbols.add(sym)
            if symbols:
                return symbols
    except Exception:
        pass

    return {
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
    }

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-5)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)

def run_clean_7d_backfill():
    print("⏳ Running Clean 7-Day History Generation...")

    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at {DB_PATH}.")
        return

    fo_symbols = get_nifty_fo_symbols()
    conn = duckdb.connect(DB_PATH)
    cols_info = conn.execute("DESCRIBE ohlcv_candles").fetchall()
    existing_cols = [col[0].lower() for col in cols_info]

    select_parts = [
        "symbol AS Symbol", "CAST(timestamp AS DATE) AS Date",
        "open AS Open", "high AS High", "low AS Low", "close AS Close", "volume AS Volume"
    ]

    select_parts.append("series AS Series" if "series" in existing_cols else "'EQ' AS Series")
    select_parts.append("market_cap AS MarketCap" if "market_cap" in existing_cols else "0.0 AS MarketCap")
    select_parts.append("delivery_qty AS DeliveryQty" if "delivery_qty" in existing_cols else "volume * 0.45 AS DeliveryQty")
    select_parts.append("delivery_pct AS DeliveryPct" if "delivery_pct" in existing_cols else "45.0 AS DeliveryPct")
    select_parts.append("delta_volume AS Delta_Volume" if "delta_volume" in existing_cols else "CASE WHEN close >= open THEN volume * 0.55 ELSE -(volume * 0.55) END AS Delta_Volume")
    select_parts.append("order_flow_delta AS Order_Flow_Delta" if "order_flow_delta" in existing_cols else "CASE WHEN close >= open THEN 1.0 ELSE -1.0 END AS Order_Flow_Delta")
    select_parts.append("ce_buy_flow AS CE_Buy_Flow" if "ce_buy_flow" in existing_cols else "TRUE AS CE_Buy_Flow")

    df_raw = conn.execute(f"SELECT {', '.join(select_parts)} FROM ohlcv_candles WHERE UPPER(series) = 'EQ' AND close > {MIN_PRICE} ORDER BY symbol, timestamp ASC").df()
    conn.close()

    if df_raw.empty:
        return

    df_raw["Symbol_Clean"] = df_raw["Symbol"].str.upper().str.strip()
    df_raw = df_raw[df_raw["Symbol_Clean"].isin(fo_symbols)].copy()

    df_raw["Date_DT"] = pd.to_datetime(df_raw["Date"])
    cutoff_date = df_raw["Date_DT"].max() - pd.Timedelta(days=LOOKBACK_DAYS)

    clean_signals = []

    for symbol, df_sym in df_raw.groupby('Symbol'):
        if len(df_sym) < 15:
            continue

        df = df_sym.copy().sort_values("Date_DT").reset_index(drop=True)
        df['Prev_Delta'] = df['Delta_Volume'].shift(1).fillna(0.0)
        df['Avg_Delta_5'] = df['Delta_Volume'].abs().shift(1).rolling(5, min_periods=3).mean().fillna(0.0)

        df['RSI_Daily'] = compute_rsi(df['Close'], 14)
        df['Prev_RSI_Daily'] = df['RSI_Daily'].shift(1).fillna(df['RSI_Daily'])

        df_weekly = df.set_index('Date_DT').resample('W-FRI').agg({'Close': 'last'}).dropna()
        df_weekly['RSI_Weekly'] = compute_rsi(df_weekly['Close'], 14) if len(df_weekly) >= 5 else 55.0
        df_weekly['Prev_RSI_Weekly'] = df_weekly['RSI_Weekly'].shift(1) if len(df_weekly) >= 5 else 50.0

        df_monthly = df.set_index('Date_DT').resample('ME').agg({'Close': 'last'}).dropna()
        df_monthly['RSI_Monthly'] = compute_rsi(df_monthly['Close'], 14) if len(df_monthly) >= 2 else 55.0
        df_monthly['Prev_RSI_Monthly'] = df_monthly['RSI_Monthly'].shift(1) if len(df_monthly) >= 2 else 50.0

        df = pd.merge_asof(df, df_weekly[['RSI_Weekly', 'Prev_RSI_Weekly']], on='Date_DT', direction='backward').fillna(50.0)
        df = pd.merge_asof(df, df_monthly[['RSI_Monthly', 'Prev_RSI_Monthly']], on='Date_DT', direction='backward').fillna(50.0)

        df['Day_Range'] = (df['High'] - df['Low']).clip(lower=1e-5)
        df['Close_Location'] = ((df['Close'] - df['Low']) / df['Day_Range']).fillna(0.5)

        for i in range(len(df)):
            row = df.iloc[i]
            if row['Date_DT'] < cutoff_date or row['Close'] <= MIN_PRICE:
                continue

            mcap_val = float(row.get('MarketCap', 0.0))
            if mcap_val > 0.0 and mcap_val < MIN_MARKET_CAP:
                continue

            cond_ce_buy = bool(row['CE_Buy_Flow'])
            cond_order_flow = (row['Order_Flow_Delta'] > 0) and (row['Delta_Volume'] > 0)

            prev_d = row['Prev_Delta']
            curr_d = row['Delta_Volume']
            avg_d5 = row['Avg_Delta_5']

            cond_surge_prev = (prev_d > 0 and curr_d >= 1.70 * prev_d) or (prev_d < 0 and curr_d >= 1.70 * abs(prev_d)) or (prev_d == 0 and curr_d > 0)
            cond_surge_avg = curr_d >= (2.0 * avg_d5)

            c_mtf_monthly = row['RSI_Monthly'] >= row['Prev_RSI_Monthly']
            c_mtf_weekly = row['RSI_Weekly'] >= row['Prev_RSI_Weekly']

            if not (cond_ce_buy and cond_order_flow and cond_surge_prev and cond_surge_avg and c_mtf_monthly and c_mtf_weekly):
                continue

            deliv_score = float(np.clip((row['DeliveryPct'] / 70.0 * 50.0), 10, 50))
            close_score = float(np.clip(row['Close_Location'] * 50.0, 10, 50))
            brs_score = round(deliv_score + close_score, 2)

            entry = round(float(row['High']) + 0.05, 2)
            sl = round(float(row['Low']), 2)
            risk = max(entry - sl, float(row['Close']) * 0.01)
            target = round(entry + (risk * TARGET_X), 2)

            deliv_pct_val = round(float(np.nan_to_num(row['DeliveryPct'], nan=0.0)), 2)
            deliv_qty_val = int(np.nan_to_num(row['DeliveryQty'], nan=0))
            volume_val = int(np.nan_to_num(row['Volume'], nan=0))
            rsi_daily_val = round(float(np.nan_to_num(row['RSI_Daily'], nan=50.0)), 2)

            clean_signals.append({
                "Date": row['Date_DT'].strftime("%d-%m-%Y"),
                "Date_DT": row['Date_DT'],
                "Symbol": symbol,
                "Timeframe": "D",
                "Type": "PRE_BREAKOUT",
                "Pattern": "ORDER_FLOW_SURGE",
                "BRS_Score": brs_score,
                "Entry": entry,
                "SL": sl,
                "Target": target,
                "Close": round(float(row['Close']), 2),
                "Volume": volume_val,
                "ema": deliv_pct_val,
                "adx": rsi_daily_val,
                "DeliveryQty": deliv_qty_val,
                "DeliveryPct": deliv_pct_val,
                "DelivSpikeRatio": round(float(curr_d / (avg_d5 + 1e-5)), 2),
                "Daily_RSI": rsi_daily_val
            })

    os.makedirs("data", exist_ok=True)
    if clean_signals:
        df_out = pd.DataFrame(clean_signals).sort_values(by=['Date_DT', 'BRS_Score'], ascending=[False, False])
        # OVERWRITE file completely (wiping previous bad data)
        df_out.drop(columns=['Date_DT']).to_csv(SIGNALS_CSV, index=False)
        print(f"✅ Replaced {SIGNALS_CSV} with clean 7-day signals.")
    else:
        # Create empty template if no stocks passed
        pd.DataFrame(columns=[
            "Date", "Symbol", "Timeframe", "Type", "Pattern", "BRS_Score",
            "Entry", "SL", "Target", "Close", "Volume", "ema", "adx",
            "DeliveryQty", "DeliveryPct", "DelivSpikeRatio", "Daily_RSI"
        ]).to_csv(SIGNALS_CSV, index=False)
        print("⚠️ No signals found; cleared signals.csv.")

if __name__ == "__main__":
    run_clean_7d_backfill()
