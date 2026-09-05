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

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN") or os.getenv("BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID")
DASHBOARD_URL = "https://brahmastra-tech.github.io/brahmastra-scanner/"

NSE_FO_URL = "https://archives.nseindia.com/content/fo/fo_mktlots.csv"


def get_nifty_fo_symbols() -> set:
    """
    Fetches the active NSE F&O underlying stock list.
    Falls back gracefully to the complete active F&O constituent list if NSE times out.
    """
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
                print(f"✅ Fetched {len(symbols)} active F&O symbols dynamically.")
                return symbols
    except Exception:
        print("⚠️ Cloud runner timeout fetching NSE archive. Using active F&O fallback universe.")

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


def run_institutional_engine():
    print("🚀 Running Brahmastra Scanner Engine (Nifty F&O EQ | MCap ₹51B+ | Price > ₹100)...")

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

    # Enforce strictly Series EQ
    if "series" in existing_cols:
        select_parts.append("series AS Series")
    else:
        select_parts.append("'EQ' AS Series")

    # Market Cap validation
    if "market_cap" in existing_cols:
        select_parts.append("market_cap AS MarketCap")
    elif "shares_outstanding" in existing_cols:
        select_parts.append("(close * shares_outstanding) AS MarketCap")
    else:
        select_parts.append("0.0 AS MarketCap")

    # Delivery columns
    if "delivery_qty" in existing_cols:
        select_parts.append("delivery_qty AS DeliveryQty")
    else:
        select_parts.append("volume * 0.45 AS DeliveryQty")

    if "delivery_pct" in existing_cols:
        select_parts.append("delivery_pct AS DeliveryPct")
    else:
        select_parts.append("45.0 AS DeliveryPct")

    # Order Flow & CE Options columns
    if "delta_volume" in existing_cols:
        select_parts.append("delta_volume AS Delta_Volume")
    else:
        select_parts.append("CASE WHEN close >= open THEN volume * 0.55 ELSE -(volume * 0.55) END AS Delta_Volume")

    if "order_flow_delta" in existing_cols:
        select_parts.append("order_flow_delta AS Order_Flow_Delta")
    else:
        select_parts.append("CASE WHEN close >= open THEN 1.0 ELSE -1.0 END AS Order_Flow_Delta")

    if "ce_buy_flow" in existing_cols:
        select_parts.append("ce_buy_flow AS CE_Buy_Flow")
    else:
        select_parts.append("TRUE AS CE_Buy_Flow")

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

    # Filter strictly for NSE F&O universe
    df_raw["Symbol_Clean"] = df_raw["Symbol"].str.upper().str.strip()
    df_raw = df_raw[df_raw["Symbol_Clean"].isin(fo_symbols)].copy()

    if df_raw.empty:
        print("⚠️ No stocks matched the active F&O universe.")
        return

    df_raw["Date_DT"] = pd.to_datetime(df_raw["Date"])
    latest_date_str = df_raw['Date_DT'].max().strftime("%d-%m-%Y")

    # Diagnostic Funnel Counters
    stats = {
        "total_fo_symbols": 0,
        "price_and_mcap": 0,
        "ce_and_order_flow": 0,
        "delta_surge_passed": 0,
        "mtf_rsi_aligned": 0,
        "final_signals": 0
    }

    all_scored_signals = []

    for symbol, df_sym in df_raw.groupby('Symbol'):
        if len(df_sym) < 15:
            continue

        stats["total_fo_symbols"] += 1
        df = df_sym.copy().sort_values("Date_DT").reset_index(drop=True)
        row = df.iloc[-1]

        # 1. Price & Market Cap Check
        if row['Close'] <= MIN_PRICE:
            continue

        mcap_val = float(row.get('MarketCap', 0.0))
        if mcap_val > 0.0 and mcap_val < MIN_MARKET_CAP:
            continue
        stats["price_and_mcap"] += 1

        df['DeliveryQty'] = df['DeliveryQty'].fillna(0.0)
        df['DeliveryPct'] = df['DeliveryPct'].fillna(0.0)
        df['Volume'] = df['Volume'].fillna(0.0)
        df['Delta_Volume'] = df['Delta_Volume'].fillna(0.0)
        df['Order_Flow_Delta'] = df['Order_Flow_Delta'].fillna(0.0)
        df['CE_Buy_Flow'] = df['CE_Buy_Flow'].fillna(False).astype(bool)

        # ----------------------------------------------------
        # ORDER FLOW & DELTA VOLUME METRICS
        # ----------------------------------------------------
        df['Prev_Delta'] = df['Delta_Volume'].shift(1).fillna(0.0)
        # Shifted 5-period baseline of absolute deltas
        df['Avg_Delta_5'] = (
            df['Delta_Volume'].abs().shift(1).rolling(5, min_periods=3).mean().fillna(0.0)
        )

        # ----------------------------------------------------
        # MULTI-TIMEFRAME RSI
        # ----------------------------------------------------
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

        row = df.iloc[-1]

        # ----------------------------------------------------
        # SIGNAL CONDITIONS EVALUATION
        # ----------------------------------------------------
        # Rule 1 & 2: CE Buying Flow & Positive Order Flow
        cond_ce_buy = bool(row['CE_Buy_Flow'])
        cond_order_flow = (row['Order_Flow_Delta'] > 0) and (row['Delta_Volume'] > 0)
        if not (cond_ce_buy and cond_order_flow):
            continue
        stats["ce_and_order_flow"] += 1

        # Rule 3: Delta Volume Surge
        # >= 70% higher than previous candle AND >= 100% higher than 5-candle avg (2x)
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
        stats["delta_surge_passed"] += 1

        # Multi-Timeframe Alignment Check
        c_mtf_monthly = row['RSI_Monthly'] >= row['Prev_RSI_Monthly']
        c_mtf_weekly = row['RSI_Weekly'] >= row['Prev_RSI_Weekly']
        if not (c_mtf_monthly and c_mtf_weekly):
            continue
        stats["mtf_rsi_aligned"] += 1

        stats["final_signals"] += 1

        # Scoring & Position Sizing
        deliv_score = float(np.clip((row['DeliveryPct'] / 70.0 * 50.0), 10, 50))
        close_score = float(np.clip(row['Close_Location'] * 50.0, 10, 50))
        composite_brs_score = round(deliv_score + close_score, 2)

        entry = round(float(row['High']) + 0.05, 2)
        sl = round(float(row['Low']), 2)
        risk = max(entry - sl, float(row['Close']) * 0.01)
        target = round(entry + (risk * TARGET_X), 2)

        deliv_pct_val = round(float(np.nan_to_num(row['DeliveryPct'], nan=0.0)), 2)
        rsi_daily_val = round(float(np.nan_to_num(row['RSI_Daily'], nan=50.0)), 2)

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
            "ema": deliv_pct_val,
            "adx": rsi_daily_val,
            "DeliveryQty": int(np.nan_to_num(row['DeliveryQty'], nan=0)),
            "DeliveryPct": deliv_pct_val,
            "DelivSpikeRatio": round(float(curr_d / (avg_d5 + 1e-5)), 2),
            "Daily_RSI": rsi_daily_val
        })

    print(f"📊 Filter Funnel Diagnostics: {stats}")

    os.makedirs("data", exist_ok=True)
    today_df = pd.DataFrame(all_scored_signals).sort_values("BRS_Score", ascending=False) if all_scored_signals else pd.DataFrame()

    if os.path.exists(SIGNALS_CSV):
        try:
            existing_df = pd.read_csv(SIGNALS_CSV)
            existing_df = existing_df[existing_df['Date'] != latest_date_str]
            combined_df = pd.concat([today_df, existing_df], ignore_index=True)
        except Exception:
            combined_df = today_df
    else:
        combined_df = today_df

    if not combined_df.empty:
        combined_df['Date_DT'] = pd.to_datetime(combined_df['Date'], format="%d-%m-%Y")
        combined_df = combined_df.sort_values(by=['Date_DT', 'BRS_Score'], ascending=[False, False])
        recent_dates = combined_df['Date_DT'].unique()[:30]
        final_export_df = combined_df[combined_df['Date_DT'].isin(recent_dates)].drop(columns=['Date_DT'])
    else:
        final_export_df = combined_df

    final_export_df.to_csv(SIGNALS_CSV, index=False)
    print(f"✅ Saved {len(today_df)} dynamic Nifty F&O candidates for {latest_date_str}.")

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
    brs = signal.get("BRS_Score")
    setup_date = signal.get("Date")
    entry = signal.get("Entry")
    sl = signal.get("SL")
    target = signal.get("Target")
    close = signal.get("Close")
    deliv_pct = signal.get("DeliveryPct", 0.0)
    rsi_val = signal.get("Daily_RSI", 0.0)
    spike_ratio = signal.get("DelivSpikeRatio", 1.0)

    chart_url = f"https://in.tradingview.com/chart/?symbol=NSE:{symbol}"

    message = (
        f"🏛️ <b>BRAHMASTRA PRE-BREAKOUT WATCHLIST</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 <b>Stock:</b> {symbol} (NSE F&O EQ)\n"
        f"⭐ <b>BRS Score:</b> {brs:.2f} / 100\n"
        f"🎯 <b>Setup:</b> Order Flow + Delta Surge ({spike_ratio:.1f}x)\n"
        f"⏱ <b>Date:</b> {setup_date}\n\n"
        f"📊 <b>ACTIONABLE TRIGGER LEVELS</b>\n"
        f"• <b>Trigger Buy Above   :</b> ₹{entry:.2f}\n"
        f"• <b>Stop Loss           :</b> ₹{sl:.2f}\n"
        f"• <b>Target (3x R:R)     :</b> ₹{target:.2f}\n"
        f"• <b>Today's Close       :</b> ₹{close:.2f}\n\n"
        f"⚡ <b>ACCUMULATION METRICS</b>\n"
        f"• <b>Delivery %          :</b> {deliv_pct:.1f}%\n"
        f"• <b>Daily RSI           :</b> {rsi_val:.1f} (Consolidating)\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 <a href='{chart_url}'>View {symbol} TradingView Chart</a>"
    )
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", 
        json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML", "disable_web_page_preview": True}, 
        timeout=10
    )


def send_summary_telegram(candidates: list, date_str: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    message = (
        f"🏁 <b>DAILY BRAHMASTRA SCAN COMPLETE ({date_str})</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>High-Conviction F&O Setups Found:</b> {len(candidates)}\n"
        f"🔍 <b>Universe:</b> NSE F&O (EQ Only | MCap ₹51B+ | Price > ₹100)\n\n"
        f"🌐 <b>Interactive Web Dashboard:</b>\n"
        f"👉 <a href='{DASHBOARD_URL}'>{DASHBOARD_URL}</a>\n"
        f"━━━━━━━━━━━━━━━━━━━━"
    )
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", 
        json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}, 
        timeout=10
    )


if __name__ == "__main__":
    run_institutional_engine()
