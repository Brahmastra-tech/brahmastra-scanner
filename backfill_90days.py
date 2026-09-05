import os
import duckdb
import pandas as pd
import numpy as np
import requests

# ==========================================
# CONFIGURATION & UNIVERSE
# ==========================================
DB_PATH = "data/candles.duckdb"
SIGNALS_CSV = "data/signals.csv"
LOOKBACK_DAYS = 30
TARGET_X = 3.0
MIN_PRICE = 100.0
MIN_MARKET_CAP = 51_000_000_000.0  # ₹51 Billion

# Chandelier Exit Parameters
CE_PERIOD = 22
CE_MULTIPLIER = 3.0

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


def compute_chandelier_exit(df: pd.DataFrame, period: int = 22, mult: float = 3.0) -> pd.DataFrame:
    """Computes pure ATR and Chandelier Exit Long line."""
    high = df['High']
    low = df['Low']
    close_prev = df['Close'].shift(1)

    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder's smoothed ATR
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    highest_high = high.rolling(window=period, min_periods=period).max()

    # Chandelier Exit Long Line
    ce_long = highest_high - (mult * atr)
    return ce_long, atr


def run_orderflow_ce_scanner():
    print(f"🚀 Running Pure Price-Action Engine: Chandelier Exit + Order Flow + Delta Force...")

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

    df_raw = conn.execute(
        f"SELECT {', '.join(select_parts)} FROM ohlcv_candles WHERE UPPER(series) = 'EQ' AND close > {MIN_PRICE} ORDER BY symbol, timestamp ASC"
    ).df()
    conn.close()

    if df_raw.empty:
        print("⚠️ No EQ candle data available.")
        return

    df_raw["Symbol_Clean"] = df_raw["Symbol"].str.upper().str.strip()
    df_raw = df_raw[df_raw["Symbol_Clean"].isin(fo_symbols)].copy()

    df_raw["Date_DT"] = pd.to_datetime(df_raw["Date"])
    cutoff_date = df_raw["Date_DT"].max() - pd.Timedelta(days=LOOKBACK_DAYS)

    valid_signals = []

    for symbol, df_sym in df_raw.groupby('Symbol'):
        if len(df_sym) < CE_PERIOD + 5:
            continue

        df = df_sym.copy().sort_values("Date_DT").reset_index(drop=True)

        # 1. Chandelier Exit Calculation
        df['CE_Long'], df['ATR'] = compute_chandelier_exit(df, period=CE_PERIOD, mult=CE_MULTIPLIER)

        # 2. Delta Volume Force Baselines
        df['Prev_Delta'] = df['Delta_Volume'].shift(1).fillna(0.0)
        df['Avg_Delta_5'] = df['Delta_Volume'].abs().shift(1).rolling(5, min_periods=3).mean().fillna(0.0)

        # 3. Bar Dynamics
        df['Day_Range'] = (df['High'] - df['Low']).clip(lower=1e-5)
        df['Close_Location'] = ((df['Close'] - df['Low']) / df['Day_Range']).fillna(0.5)

        for i in range(len(df)):
            row = df.iloc[i]
            if row['Date_DT'] < cutoff_date or row['Close'] <= MIN_PRICE:
                continue

            # MCap Validation
            mcap_val = float(row.get('MarketCap', 0.0))
            if mcap_val > 0.0 and mcap_val < MIN_MARKET_CAP:
                continue

            # --- CONDITION 1: CHANDELIER EXIT BREAKOUT / RETENTION ---
            # Price closes above the Chandelier line, confirming bullish institutional floor
            cond_chandelier = (row['Close'] > row['CE_Long']) if pd.notna(row['CE_Long']) else True
            cond_ce_flag = bool(row['CE_Buy_Flow'])

            # --- CONDITION 2: POSITIVE ORDER FLOW DELTA ---
            cond_order_flow = (row['Order_Flow_Delta'] > 0) and (row['Delta_Volume'] > 0)

            # --- CONDITION 3: DELTA VOLUME FORCE (SURGE) ---
            prev_d = row['Prev_Delta']
            curr_d = row['Delta_Volume']
            avg_d5 = row['Avg_Delta_5']

            cond_surge_prev = (prev_d > 0 and curr_d >= 1.70 * prev_d) or \
                              (prev_d < 0 and curr_d >= 1.70 * abs(prev_d)) or \
                              (prev_d == 0 and curr_d > 0)
            cond_surge_avg = curr_d >= (2.0 * avg_d5)
            cond_delta_force = cond_surge_prev and cond_surge_avg

            # Strictly Price Action & Order Flow Verification
            if not (cond_chandelier and cond_ce_flag and cond_order_flow and cond_delta_force):
                continue

            deliv_score = float(np.clip((row['DeliveryPct'] / 70.0 * 50.0), 10, 50))
            close_score = float(np.clip(row['Close_Location'] * 50.0, 10, 50))
            brs_score = round(deliv_score + close_score, 2)

            entry = round(float(row['High']) + 0.05, 2)
            # Use Chandelier Exit or Low for Stop Loss
            ce_stop = round(float(row['CE_Long']), 2) if pd.notna(row['CE_Long']) else round(float(row['Low']), 2)
            sl = min(round(float(row['Low']), 2), ce_stop)
            risk = max(entry - sl, float(row['Close']) * 0.01)
            target = round(entry + (risk * TARGET_X), 2)

            deliv_pct_val = round(float(np.nan_to_num(row['DeliveryPct'], nan=0.0)), 2)
            deliv_qty_val = int(np.nan_to_num(row['DeliveryQty'], nan=0))
            volume_val = int(np.nan_to_num(row['Volume'], nan=0))
            surge_ratio = round(float(curr_d / (avg_d5 + 1e-5)), 2)

            # Columns compatible with index.html
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
                "Volume": volume_val,
                "EMA20": deliv_pct_val,             # Feeds table column cleanly
                "ADX14": surge_ratio,              # Replaced ADX with Delta Force Surge Ratio
                "ema": deliv_pct_val,
                "adx": surge_ratio,
                "DeliveryQty": deliv_qty_val,
                "DeliveryPct": deliv_pct_val,
                "DelivSpikeRatio": surge_ratio,
                "Daily_RSI": surge_ratio
            })

    os.makedirs("data", exist_ok=True)
    if valid_signals:
        df_out = pd.DataFrame(valid_signals).sort_values(by=['Date_DT', 'BRS_Score'], ascending=[False, False])
        df_out.drop(columns=['Date_DT']).to_csv(SIGNALS_CSV, index=False)
        print(f"✅ Exported {len(df_out)} pure Order Flow + CE signals to {SIGNALS_CSV}.")
    else:
        pd.DataFrame(columns=[
            "Date", "Symbol", "Timeframe", "Type", "Pattern", "BRS_Score",
            "Entry", "SL", "Target", "Close", "Volume", "EMA20", "ADX14", "ema", "adx",
            "DeliveryQty", "DeliveryPct", "DelivSpikeRatio", "Daily_RSI"
        ]).to_csv(SIGNALS_CSV, index=False)
        print(f"⚠️ 0 signals found under pure Order Flow & CE rules. Cleared {SIGNALS_CSV}.")


if __name__ == "__main__":
    run_orderflow_ce_scanner()
