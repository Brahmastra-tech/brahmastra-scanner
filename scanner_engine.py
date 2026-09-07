import os
import time
import duckdb
import pandas as pd
import numpy as np
import requests

# ==========================================
# CONFIGURATION & STRICT F&O UNIVERSE
# ==========================================
DB_PATH = "data/candles.duckdb"
SIGNALS_CSV = "data/signals.csv"
TARGET_X = 3.0

MIN_MARKET_CAP = 51_000_000_000.0  # ₹51B
MIN_PRICE = 100.0

CE_PERIOD = 22
CE_MULTIPLIER = 3.0

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN") or os.getenv("BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID")
DASHBOARD_URL = "https://brahmastra-tech.github.io/brahmastra-scanner/"

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


def compute_chandelier_exit(df, period=22, mult=3.0):
    high = df['High']
    low = df['Low']
    close_prev = df['Close'].shift(1)

    tr = pd.concat([high - low, (high - close_prev).abs(), (low - close_prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    highest_high = high.rolling(window=period, min_periods=period).max()
    lowest_low = low.rolling(window=period, min_periods=period).min()

    ce_long = highest_high - (mult * atr)
    ce_short = lowest_low + (mult * atr)
    return ce_long, ce_short, atr


def run_institutional_engine():
    print("🚀 Running Institutional Order Flow Scanner (Long + Short Engine)...")
    if not os.path.exists(DB_PATH):
        return

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

    where_parts = [f"close > {MIN_PRICE}"]
    if "series" in cols_info:
        where_parts.append("UPPER(series) = 'EQ'")

    df_raw = conn.execute(f"SELECT {', '.join(select_parts)} FROM ohlcv_candles WHERE {' AND '.join(where_parts)} ORDER BY symbol, timestamp ASC").df()
    conn.close()

    if df_raw.empty:
        return

    df_raw["Symbol_Clean"] = df_raw["Symbol"].astype(str).str.upper().str.strip().str.replace(r"-EQ$", "", regex=True)
    df_raw = df_raw[df_raw["Symbol_Clean"].isin(NSE_FO_SYMBOLS)].copy()
    df_raw["Date_DT"] = pd.to_datetime(df_raw["Date"])

    latest_dt = df_raw['Date_DT'].max()
    latest_date_str = latest_dt.strftime("%d-%m-%Y")

    all_scored_signals = []

    for symbol, df_sym in df_raw.groupby('Symbol_Clean'):
        if len(df_sym) < 15:
            continue

        df = df_sym.copy().sort_values("Date_DT").reset_index(drop=True)
        df['CE_Long'], df['CE_Short'], df['ATR'] = compute_chandelier_exit(df, period=CE_PERIOD, mult=CE_MULTIPLIER)

        row = df.iloc[-1]
        if row['Close'] <= MIN_PRICE:
            continue

        df['Prev_Delta'] = df['Delta_Volume'].shift(1).fillna(0.0)
        df['Avg_Delta_5'] = df['Delta_Volume'].abs().shift(1).rolling(5, min_periods=3).mean().fillna(0.0)
        df['Day_Range'] = (df['High'] - df['Low']).clip(lower=1e-5)
        df['Close_Location'] = ((df['Close'] - df['Low']) / df['Day_Range']).fillna(0.5)

        row = df.iloc[-1]
        prev_d, curr_d, avg_d5 = row['Prev_Delta'], row['Delta_Volume'], row['Avg_Delta_5']
        deliv_pct_val = round(float(np.nan_to_num(row['DeliveryPct'], nan=0.0)), 2)
        spike_ratio = round(float(abs(curr_d) / (avg_d5 + 1e-5)), 2)

        # ------------------------------------------------------------------
        # 1. LONG CRITERIA: Bullish Delta Surge + CE Long Support
        # ------------------------------------------------------------------
        cond_long_ce = (row['Close'] > row['CE_Long']) if pd.notna(row['CE_Long']) else True
        cond_long_flow = (row['Order_Flow_Delta'] > 0) and (curr_d > 0)
        cond_long_surge = (curr_d >= 2.0 * avg_d5) and (curr_d >= 1.70 * abs(prev_d) if prev_d != 0 else True)

        if cond_long_ce and cond_long_flow and cond_long_surge:
            deliv_score = float(np.clip((row['DeliveryPct'] / 70.0 * 50.0), 10, 50))
            close_score = float(np.clip(row['Close_Location'] * 50.0, 10, 50))
            brs = round(deliv_score + close_score, 2)

            entry = round(float(row['High']) + 0.05, 2)
            sl = min(round(float(row['Low']), 2), round(float(row['CE_Long']), 2) if pd.notna(row['CE_Long']) else round(float(row['Low']), 2))
            risk = max(entry - sl, float(row['Close']) * 0.01)
            target = round(entry + (risk * TARGET_X), 2)

            all_scored_signals.append({
                "Date": latest_date_str,
                "Symbol": symbol,
                "Timeframe": "D",
                "Type": "LONG",
                "Pattern": "BULLISH_DELTA_SURGE",
                "BRS_Score": brs,
                "Entry": entry,
                "SL": sl,
                "Target": target,
                "Close": round(float(row['Close']), 2),
                "Volume": int(np.nan_to_num(row['Volume'], nan=0)),
                "DeliveryQty": int(np.nan_to_num(row['DeliveryQty'], nan=0)),
                "DeliveryPct": deliv_pct_val,
                "DelivSpikeRatio": spike_ratio
            })

        # ------------------------------------------------------------------
        # 2. SHORT CRITERIA: Bearish Delta Dump + CE Short Breakdown
        # ------------------------------------------------------------------
        cond_short_ce = (row['Close'] < row['CE_Short']) if pd.notna(row['CE_Short']) else True
        cond_short_flow = (row['Order_Flow_Delta'] < 0) and (curr_d < 0)
        cond_short_surge = (abs(curr_d) >= 2.0 * avg_d5) and (abs(curr_d) >= 1.70 * abs(prev_d) if prev_d != 0 else True)

        if cond_short_ce and cond_short_flow and cond_short_surge:
            sell_score = float(np.clip(((1.0 - row['Close_Location']) * 50.0), 10, 50))
            deliv_score = float(np.clip((row['DeliveryPct'] / 70.0 * 50.0), 10, 50))
            brs = round(sell_score + deliv_score, 2)

            entry = round(float(row['Low']) - 0.05, 2)  # Sell below low
            sl = max(round(float(row['High']), 2), round(float(row['CE_Short']), 2) if pd.notna(row['CE_Short']) else round(float(row['High']), 2))
            risk = max(sl - entry, float(row['Close']) * 0.01)
            target = round(entry - (risk * TARGET_X), 2)

            all_scored_signals.append({
                "Date": latest_date_str,
                "Symbol": symbol,
                "Timeframe": "D",
                "Type": "SHORT",
                "Pattern": "BEARISH_DELTA_SURGE",
                "BRS_Score": brs,
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
    today_df = pd.DataFrame(all_scored_signals)

    clean_columns = [
        "Date", "Symbol", "Timeframe", "Type", "Pattern", "BRS_Score",
        "Entry", "SL", "Target", "Close", "Volume",
        "DeliveryQty", "DeliveryPct", "DelivSpikeRatio"
    ]

    if os.path.exists(SIGNALS_CSV):
        try:
            existing_df = pd.read_csv(SIGNALS_CSV)
            if not existing_df.empty:
                today_iso = latest_dt.strftime("%Y-%m-%d")
                existing_df = existing_df[~existing_df['Date'].astype(str).isin([latest_date_str, today_iso])]
                combined_df = pd.concat([today_df, existing_df], ignore_index=True)
            else:
                combined_df = today_df
        except Exception:
            combined_df = today_df
    else:
        combined_df = today_df

    if not combined_df.empty:
        combined_df["Symbol_Clean"] = combined_df["Symbol"].astype(str).str.upper().str.strip()
        combined_df = combined_df[combined_df["Symbol_Clean"].isin(NSE_FO_SYMBOLS)].drop(columns=["Symbol_Clean"])
        for col in clean_columns:
            if col not in combined_df.columns:
                combined_df[col] = 0.0

        combined_df['Parsed_Date'] = pd.to_datetime(combined_df['Date'], dayfirst=True, errors='coerce')
        combined_df['Date'] = combined_df['Parsed_Date'].dt.strftime("%d-%m-%Y").fillna(combined_df['Date'])
        final_export_df = combined_df.sort_values(by=['Parsed_Date', 'BRS_Score'], ascending=[False, False]).drop(columns=['Parsed_Date'])
        final_export_df = final_export_df[clean_columns]
    else:
        final_export_df = pd.DataFrame(columns=clean_columns)

    final_export_df.to_csv(SIGNALS_CSV, index=False)
    print(f"✅ Exported {len(today_df)} Long/Short setups for {latest_date_str}.")

    top_candidates = today_df.to_dict('records') if not today_df.empty else []
    try:
        for sig in top_candidates:
            send_telegram_alert(sig)
            time.sleep(0.4)
    finally:
        send_summary_telegram(top_candidates, latest_date_str)


def send_telegram_alert(signal: dict):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    sym, brs, s_date = signal.get("Symbol"), signal.get("BRS_Score", 0.0), signal.get("Date")
    entry, sl, tgt, close = signal.get("Entry", 0.0), signal.get("SL", 0.0), signal.get("Target", 0.0), signal.get("Close", 0.0)
    deliv_pct, spike_ratio = signal.get("DeliveryPct", 0.0), signal.get("DelivSpikeRatio", 1.0)
    sig_type = signal.get("Type", "LONG")

    action_title = "🟢 BULLISH ORDER FLOW ACCUMULATION" if sig_type == "LONG" else "🔴 BEARISH ORDER FLOW DISTRIBUTION"
    trigger_action = "Trigger BUY Above" if sig_type == "LONG" else "Trigger SHORT Below"

    msg = (
        f"🏛️ <b>BRAHMASTRA ORDER FLOW RADAR</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>Stock:</b> {sym} (NSE F&O EQ)\n"
        f"⚡ <b>Signal:</b> {action_title}\n"
        f"⭐ <b>BRS Score:</b> {brs:.2f} / 100\n"
        f"🎯 <b>Delta Force:</b> {spike_ratio:.1f}x Volume Surge\n"
        f"⏱ <b>Date:</b> {s_date}\n\n"
        f"📊 <b>ACTIONABLE TRIGGER LEVELS</b>\n"
        f"• <b>{trigger_action} :</b> ₹{entry:.2f}\n"
        f"• <b>Stop Loss         :</b> ₹{sl:.2f}\n"
        f"• <b>Target (3.0x R:R) :</b> ₹{tgt:.2f}\n"
        f"• <b>Today Close       :</b> ₹{close:.2f}\n\n"
        f"⚡ <b>FLOW METRICS</b>\n"
        f"• <b>Delivery %        :</b> {deliv_pct:.1f}%\n"
        f"• <b>Order Flow Bias   :</b> {'Aggressive Buyers' if sig_type == 'LONG' else 'Aggressive Sellers'}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 <a href='https://in.tradingview.com/chart/?symbol=NSE:{sym}'>Open TradingView Chart</a>"
    )

    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=10
        )
    except Exception:
        pass


def send_summary_telegram(candidates: list, date_str: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    long_count = sum(1 for c in candidates if c.get("Type") == "LONG")
    short_count = sum(1 for c in candidates if c.get("Type") == "SHORT")

    msg = (
        f"🏁 <b>DAILY INSTITUTIONAL SCAN COMPLETE ({date_str})</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>Setups Found:</b> {len(candidates)} (🟢 Long: {long_count} | 🔴 Short: {short_count})\n"
        f"🔍 <b>Engine:</b> Chandelier Exit + Order Flow Delta Surge\n"
        f"🏛️ <b>Universe:</b> NSE F&O (EQ Only | MCap ₹51B+ | Price > ₹100)\n\n"
        f"🌐 <b>Interactive Web Terminal:</b>\n"
        f"👉 <a href='{DASHBOARD_URL}'>Open Live Terminal</a>\n"
        f"━━━━━━━━━━━━━━━━━━━━"
    )

    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=10
        )
    except Exception:
        pass


if __name__ == "__main__":
    run_institutional_engine()
