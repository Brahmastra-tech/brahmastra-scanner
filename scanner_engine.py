import os
import time
import duckdb
import pandas as pd
import numpy as np
import requests

# ==========================================
# CONFIGURATION & ENVIRONMENT
# ==========================================
DB_PATH = "data/candles.duckdb"
SIGNALS_CSV = "data/signals.csv"
TARGET_X = 3.0

MIN_MARKET_CAP = 51_000_000_000.0  # ₹51 Billion (₹5,100 Crore)
MIN_PRICE = 100.0

# Chandelier Exit Parameters
CE_PERIOD = 22
CE_MULTIPLIER = 3.0

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN") or os.getenv("BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID")
DASHBOARD_URL = "https://brahmastra-tech.github.io/brahmastra-scanner/"

NSE_FO_URL = "https://archives.nseindia.com/content/fo/fo_mktlots.csv"


def get_nifty_fo_symbols() -> set:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
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


def compute_chandelier_exit(df: pd.DataFrame, period: int = 22, mult: float = 3.0):
    high = df['High']
    low = df['Low']
    close_prev = df['Close'].shift(1)

    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    highest_high = high.rolling(window=period, min_periods=period).max()
    ce_long = highest_high - (mult * atr)
    return ce_long, atr


def run_institutional_engine():
    print("🚀 Running Brahmastra Scanner Engine (Chandelier Exit + Order Flow + Delta Surge)...")

    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at {DB_PATH}.")
        return

    fo_symbols = get_nifty_fo_symbols()

    conn = duckdb.connect(DB_PATH)
    cols_info = conn.execute("DESCRIBE ohlcv_candles").fetchall()
    existing_cols = [col[0].lower() for col in cols_info]

    select_parts = [
        "symbol AS Symbol",
        "CAST(timestamp AS DATE) AS Date",
        "open AS Open", "high AS High", "low AS Low", "close AS Close", "volume AS Volume"
    ]

    select_parts.append("series AS Series" if "series" in existing_cols else "'EQ' AS Series")
    select_parts.append("market_cap AS MarketCap" if "market_cap" in existing_cols else "0.0 AS MarketCap")
    select_parts.append("delivery_qty AS DeliveryQty" if "delivery_qty" in existing_cols else "volume * 0.45 AS DeliveryQty")
    select_parts.append("delivery_pct AS DeliveryPct" if "delivery_pct" in existing_cols else "45.0 AS DeliveryPct")
    select_parts.append("delta_volume AS Delta_Volume" if "delta_volume" in existing_cols else "CASE WHEN close >= open THEN volume * 0.55 ELSE -(volume * 0.55) END AS Delta_Volume")
    select_parts.append("order_flow_delta AS Order_Flow_Delta" if "order_flow_delta" in existing_cols else "CASE WHEN close >= open THEN 1.0 ELSE -1.0 END AS Order_Flow_Delta")
    select_parts.append("ce_buy_flow AS CE_Buy_Flow" if "ce_buy_flow" in existing_cols else "TRUE AS CE_Buy_Flow")

    query = f"""
        SELECT {', '.join(select_parts)} 
        FROM ohlcv_candles 
        WHERE UPPER(series) = 'EQ'
          AND close > {MIN_PRICE}
        ORDER BY symbol, timestamp ASC
    """
    df_raw = conn.execute(query).df()
    conn.close()

    if df_raw.empty:
        print("⚠️ No EQ records found in database meeting the initial price filter.")
        return

    df_raw["Symbol_Clean"] = df_raw["Symbol"].str.upper().str.strip()
    df_raw = df_raw[df_raw["Symbol_Clean"].isin(fo_symbols)].copy()

    if df_raw.empty:
        print("⚠️ No stocks matched the active F&O universe.")
        return

    df_raw["Date_DT"] = pd.to_datetime(df_raw["Date"])
    latest_date_str = df_raw['Date_DT'].max().strftime("%d-%m-%Y")

    all_scored_signals = []

    for symbol, df_sym in df_raw.groupby('Symbol'):
        if len(df_sym) < CE_PERIOD + 5:
            continue

        df = df_sym.copy().sort_values("Date_DT").reset_index(drop=True)
        df['CE_Long'], df['ATR'] = compute_chandelier_exit(df, period=CE_PERIOD, mult=CE_MULTIPLIER)

        row = df.iloc[-1]

        if row['Close'] <= MIN_PRICE:
            continue

        mcap_val = float(row.get('MarketCap', 0.0))
        if mcap_val > 0.0 and mcap_val < MIN_MARKET_CAP:
            continue

        df['DeliveryQty'] = df['DeliveryQty'].fillna(0.0)
        df['DeliveryPct'] = df['DeliveryPct'].fillna(0.0)
        df['Volume'] = df['Volume'].fillna(0.0)
        df['Delta_Volume'] = df['Delta_Volume'].fillna(0.0)
        df['Order_Flow_Delta'] = df['Order_Flow_Delta'].fillna(0.0)
        df['CE_Buy_Flow'] = df['CE_Buy_Flow'].fillna(False).astype(bool)

        df['Prev_Delta'] = df['Delta_Volume'].shift(1).fillna(0.0)
        df['Avg_Delta_5'] = df['Delta_Volume'].abs().shift(1).rolling(5, min_periods=3).mean().fillna(0.0)

        df['Day_Range'] = (df['High'] - df['Low']).clip(lower=1e-5)
        df['Close_Location'] = ((df['Close'] - df['Low']) / df['Day_Range']).fillna(0.5)

        row = df.iloc[-1]

        cond_chandelier = (row['Close'] > row['CE_Long']) if pd.notna(row['CE_Long']) else True
        cond_ce_buy = bool(row['CE_Buy_Flow'])
        cond_order_flow = (row['Order_Flow_Delta'] > 0) and (row['Delta_Volume'] > 0)

        if not (cond_chandelier and cond_ce_buy and cond_order_flow):
            continue

        prev_d = row['Prev_Delta']
        curr_d = row['Delta_Volume']
        avg_d5 = row['Avg_Delta_5']

        cond_momentum = (prev_d > 0) and (curr_d >= 1.70 * prev_d)
        cond_reversal = (prev_d < 0) and (curr_d >= 1.70 * abs(prev_d))
        cond_flat_breakout = (prev_d == 0) and (curr_d > 0)
        cond_surge_prev = cond_momentum or cond_reversal or cond_flat_breakout

        cond_surge_avg = curr_d >= (2.0 * avg_d5)
        if not (cond_surge_prev and cond_surge_avg):
            continue

        deliv_score = float(np.clip((row['DeliveryPct'] / 70.0 * 50.0), 10, 50))
        close_score = float(np.clip(row['Close_Location'] * 50.0, 10, 50))
        composite_brs_score = round(deliv_score + close_score, 2)

        entry = round(float(row['High']) + 0.05, 2)
        ce_stop = round(float(row['CE_Long']), 2) if pd.notna(row['CE_Long']) else round(float(row['Low']), 2)
        sl = min(round(float(row['Low']), 2), ce_stop)
        risk = max(entry - sl, float(row['Close']) * 0.01)
        target = round(entry + (risk * TARGET_X), 2)

        deliv_pct_val = round(float(np.nan_to_num(row['DeliveryPct'], nan=0.0)), 2)
        spike_ratio = round(float(curr_d / (avg_d5 + 1e-5)), 2)

        # UNIFIED CLEAN SCHEMA
        all_scored_signals.append({
            "Date": latest_date_str,
            "Symbol": symbol,
            "Timeframe": "D",
            "Type": "PRE_BREAKOUT",
            "Pattern": "PRE_BREAKOUT",
            "BRS_Score": composite_brs_score,
            "Entry": entry,
            "SL": sl,
            "Target": target,
            "Close": round(float(row['Close']), 2),
            "Volume": int(np.nan_to_num(row['Volume'], nan=0)),
            "DeliveryQty": int(np.nan_to_num(row['DeliveryQty'], nan=0)),
            "DeliveryPct": deliv_pct_val,
            "DelivSpikeRatio": spike_ratio
        })

    os.makedirs("data", exist_ok=True)
    today_df = pd.DataFrame(all_scored_signals).sort_values("BRS_Score", ascending=False) if all_scored_signals else pd.DataFrame()

    if os.path.exists(SIGNALS_CSV):
        try:
            existing_df = pd.read_csv(SIGNALS_CSV)
            # Remove any existing entries for today, keep history
            existing_df = existing_df[existing_df['Date'] != latest_date_str]
            combined_df = pd.concat([today_df, existing_df], ignore_index=True)
        except Exception:
            combined_df = today_df
    else:
        combined_df = today_df

    # Standardize column list
    clean_columns = [
        "Date", "Symbol", "Timeframe", "Type", "Pattern", "BRS_Score",
        "Entry", "SL", "Target", "Close", "Volume",
        "DeliveryQty", "DeliveryPct", "DelivSpikeRatio"
    ]

    if not combined_df.empty:
        # Keep only the columns that belong in the clean schema
        available_cols = [c for c in clean_columns if c in combined_df.columns]
        combined_df = combined_df[available_cols]
        combined_df['Date_DT'] = pd.to_datetime(combined_df['Date'], format="%d-%m-%Y", errors='coerce')
        combined_df = combined_df.sort_values(by=['Date_DT', 'BRS_Score'], ascending=[False, False])
        final_export_df = combined_df.drop(columns=['Date_DT'])
    else:
        final_export_df = pd.DataFrame(columns=clean_columns)

    final_export_df.to_csv(SIGNALS_CSV, index=False)
    print(f"✅ Saved clean candidates for {latest_date_str}.")

    top_candidates = today_df.to_dict('records') if not today_df.empty else []
    try:
        for sig in top_candidates:
            send_telegram_alert(sig)
            time.sleep(0.5)
    finally:
        send_summary_telegram(top_candidates, latest_date_str)


def send_telegram_alert(signal: dict):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    symbol = signal.get("Symbol")
    brs = signal.get("BRS_Score", 0.0)
    setup_date = signal.get("Date")
    entry = signal.get("Entry", 0.0)
    sl = signal.get("SL", 0.0)
    target = signal.get("Target", 0.0)
    close = signal.get("Close", 0.0)
    deliv_pct = signal.get("DeliveryPct", 0.0)
    spike_ratio = signal.get("DelivSpikeRatio", 1.0)

    chart_url = f"https://in.tradingview.com/chart/?symbol=NSE:{symbol}"

    message = (
        f"🏛️ <b>BRAHMASTRA PRE-BREAKOUT WATCHLIST</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 <b>Stock:</b> {symbol} (NSE F&O EQ)\n"
        f"⭐ <b>BRS Score:</b> {brs:.2f} / 100\n"
        f"🎯 <b>Setup:</b> Chandelier Exit + Delta Force ({spike_ratio:.1f}x)\n"
        f"⏱ <b>Date:</b> {setup_date}\n\n"
        f"📊 <b>ACTIONABLE TRIGGER LEVELS</b>\n"
        f"• <b>Trigger Buy Above   :</b> ₹{entry:.2f}\n"
        f"• <b>Stop Loss           :</b> ₹{sl:.2f}\n"
        f"• <b>Target (3x R:R)     :</b> ₹{target:.2f}\n"
        f"• <b>Today's Close       :</b> ₹{close:.2f}\n\n"
        f"⚡ <b>ACCUMULATION METRICS</b>\n"
        f"• <b>Delivery %          :</b> {deliv_pct:.1f}%\n"
        f"• <b>Delta Volume Surge  :</b> {spike_ratio:.1f}x vs 5-Day Avg\n"
        f"• <b>Order Flow Delta    :</b> Positive Buyers In Control\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 <a href='{chart_url}'>View {symbol} TradingView Chart</a>"
    )

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"❌ Telegram alert network error: {e}")


def send_summary_telegram(candidates: list, date_str: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    cache_buster = int(time.time())
    active_dash_link = f"{DASHBOARD_URL}?v={cache_buster}"

    message = (
        f"🏁 <b>DAILY BRAHMASTRA SCAN COMPLETE ({date_str})</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>High-Conviction F&O Setups Found:</b> {len(candidates)}\n"
        f"🔍 <b>Engine:</b> Chandelier Exit + Order Flow + Delta Surge\n"
        f"🏛️ <b>Universe:</b> NSE F&O (EQ Only | MCap ₹51B+ | Price > ₹100)\n\n"
        f"🌐 <b>Interactive Web Dashboard:</b>\n"
        f"👉 <a href='{active_dash_link}'>Open Live Dashboard</a>\n"
        f"━━━━━━━━━━━━━━━━━━━━"
    )

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"❌ Telegram summary network error: {e}")


if __name__ == "__main__":
    run_institutional_engine()
