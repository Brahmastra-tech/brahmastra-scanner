import os
import time
import duckdb
import pandas as pd
import numpy as np
import requests

DB_PATH = "data/candles.duckdb"
SIGNALS_CSV = "data/signals.csv"
ACTIVE_WATCHLIST_CSV = "data/active_watchlist.csv"
MAX_HOLD_DAYS = 3

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


def send_telegram(html_text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Missing Telegram credentials.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": html_text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        res = requests.post(url, json=payload, timeout=10)
        if res.status_code != 200:
            print(f"❌ Telegram API Error ({res.status_code}): {res.text}")
        else:
            print("🚀 Successfully sent Breakout Radar tables to Telegram!")
    except Exception as err:
        print(f"⚠️ Telegram Network Error: {err}")


def update_lifecycle_and_track():
    print("🎯 Running Breakout Radar Tracker Engine...")
    if not os.path.exists(SIGNALS_CSV):
        print("⚠️ signals.csv missing.")
        return

    signals_df = pd.read_csv(SIGNALS_CSV)
    if signals_df.empty:
        print("ℹ️ signals.csv is empty.")
        return

    signals_df['Symbol_Clean'] = signals_df['Symbol'].astype(str).str.upper().str.strip()
    signals_df = signals_df[signals_df['Symbol_Clean'].isin(NSE_FO_SYMBOLS)].copy()

    signals_df['Date_Parsed'] = pd.to_datetime(signals_df['Date'], dayfirst=True, errors='coerce')
    signals_df['Date_Norm'] = signals_df['Date_Parsed'].dt.strftime("%Y-%m-%d")

    # Connect to DuckDB & pull the latest available session candles
    conn = duckdb.connect(DB_PATH)
    candles_df = conn.execute("""
        SELECT symbol AS Symbol, CAST(timestamp AS DATE) AS Date, high AS High, low AS Low, close AS Close
        FROM ohlcv_candles
        WHERE CAST(timestamp AS DATE) = (SELECT MAX(CAST(timestamp AS DATE)) FROM ohlcv_candles)
    """).df()
    conn.close()

    if candles_df.empty:
        # Fallback to signals date if DuckDB empty
        latest_session_norm = signals_df['Date_Norm'].max()
        latest_session_display = signals_df['Date_Parsed'].dt.strftime("%d-%m-%Y").max()
        candle_dict = {}
    else:
        latest_session_norm = pd.to_datetime(candles_df['Date'].max()).strftime("%Y-%m-%d")
        latest_session_display = pd.to_datetime(candles_df['Date'].max()).strftime("%d-%m-%Y")
        candle_dict = candles_df.set_index("Symbol").to_dict("index")

    # Load active watchlist
    wl_df = pd.DataFrame()
    if os.path.exists(ACTIVE_WATCHLIST_CSV):
        try:
            wl_df = pd.read_csv(ACTIVE_WATCHLIST_CSV)
            if not wl_df.empty and 'Symbol' in wl_df.columns:
                wl_df['Symbol_Clean'] = wl_df['Symbol'].astype(str).str.upper().str.strip()
                wl_df = wl_df[wl_df['Symbol_Clean'].isin(NSE_FO_SYMBOLS)].copy()
        except Exception:
            wl_df = pd.DataFrame()

    existing_keys = set()
    if not wl_df.empty and 'Date_Norm' in wl_df.columns:
        existing_keys = set(zip(wl_df['Date_Norm'].astype(str), wl_df['Symbol_Clean']))

    # Intake new signals
    new_items = []
    for _, row in signals_df.iterrows():
        key = (str(row['Date_Norm']), row['Symbol_Clean'])
        if key not in existing_keys:
            is_today = (str(row['Date_Norm']) == latest_session_norm)
            direction = "SHORT" if "BEARISH" in str(row.get('Pattern', '')).upper() or str(row.get('Type', '')).upper() == "SHORT" else "LONG"
            new_items.append({
                "Date": row['Date'],
                "Date_Norm": str(row['Date_Norm']),
                "Symbol": row['Symbol_Clean'],
                "Type": direction,
                "Entry": float(row['Entry']),
                "SL": float(row['SL']),
                "Target": float(row['Target']),
                "Close": float(row['Close']),
                "Status": "PENDING",
                "Days_Active": 1 if is_today else 2,
                "Trigger_Date": ""
            })

    if new_items:
        wl_df = pd.concat([wl_df, pd.DataFrame(new_items)], ignore_index=True)

    # Evaluate lifecycle states
    updated = []
    for _, row in wl_df.iterrows():
        sym = row['Symbol']
        status = row['Status']
        sig_date_norm = str(row.get('Date_Norm', ''))
        sig_type = str(row.get('Type', 'LONG')).upper()
        entry = float(row['Entry'])
        sl = float(row['SL'])
        target = float(row['Target'])
        days = int(row.get('Days_Active', 1))

        if status in ["EXPIRED", "STOPPED_OUT", "TARGET_HIT"]:
            updated.append(row.to_dict())
            continue

        # Day 0 setups stay PENDING
        if sig_date_norm == latest_session_norm:
            row['Status'] = "PENDING"
            row['Days_Active'] = 1
            updated.append(row.to_dict())
            continue

        # Day 1+ setups: evaluate against today's actual candle
        if sym in candle_dict:
            c = candle_dict[sym]
            hi, lo = float(c['High']), float(c['Low'])

            if sig_type == "LONG":
                if status == "PENDING":
                    if lo <= sl:
                        row['Status'] = "STOPPED_OUT"
                    elif hi >= entry:
                        row['Status'] = "TRIGGERED"
                        row['Trigger_Date'] = latest_session_norm
                    else:
                        days += 1
                        row['Days_Active'] = days
                        if days > MAX_HOLD_DAYS:
                            row['Status'] = "EXPIRED"
                elif status == "TRIGGERED":
                    if lo <= sl:
                        row['Status'] = "STOPPED_OUT"
                    elif hi >= target:
                        row['Status'] = "TARGET_HIT"

            elif sig_type == "SHORT":
                if status == "PENDING":
                    if hi >= sl:
                        row['Status'] = "STOPPED_OUT"
                    elif lo <= entry:
                        row['Status'] = "TRIGGERED"
                        row['Trigger_Date'] = latest_session_norm
                    else:
                        days += 1
                        row['Days_Active'] = days
                        if days > MAX_HOLD_DAYS:
                            row['Status'] = "EXPIRED"
                elif status == "TRIGGERED":
                    if hi >= sl:
                        row['Status'] = "STOPPED_OUT"
                    elif lo <= target:
                        row['Status'] = "TARGET_HIT"

        updated.append(row.to_dict())

    final_wl = pd.DataFrame(updated)
    os.makedirs("data", exist_ok=True)
    final_wl.to_csv(ACTIVE_WATCHLIST_CSV, index=False)
    print(f"💾 Updated active_watchlist.csv ({len(final_wl)} setups tracked)")

    send_telegram_radar_table(final_wl, latest_session_display)


def send_telegram_radar_table(df: pd.DataFrame, date_str: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    triggered = df[df["Status"] == "TRIGGERED"]
    pending = df[df["Status"] == "PENDING"]

    msg_lines = [
        "🏛️ <b>BRAHMASTRA ORDER FLOW RADAR</b>",
        f"📅 <i>Session Date: {date_str}</i>",
        "━━━━━━━━━━━━━━━━━━━━\n"
    ]

    # SECTION 1: Active Triggered Positions
    msg_lines.append(f"🚀 <b>TRIGGERED POSITIONS ({len(triggered)})</b>")
    if not triggered.empty:
        t_header = f"{'Symbol':<9} {'Bias':<5} {'Entry':<8} {'SL':<8} {'Tgt':<8}"
        t_sep = "-" * len(t_header)
        t_rows = [t_header, t_sep]
        for _, r in triggered.iterrows():
            sym = str(r['Symbol'])[:8]
            bias = "BUY" if str(r.get('Type', 'LONG')).upper() == "LONG" else "SELL"
            t_rows.append(f"{sym:<9} {bias:<5} {float(r['Entry']):<8.1f} {float(r['SL']):<8.1f} {float(r['Target']):<8.1f}")
        t_block = "\n".join(t_rows)
        msg_lines.append(f"<pre>{t_block}</pre>")
        links = " | ".join([f"<a href='https://in.tradingview.com/chart/?symbol=NSE:{r['Symbol']}'>{r['Symbol']}</a>" for _, r in triggered.iterrows()])
        msg_lines.append(f"📈 <b>Charts:</b> {links}\n")
    else:
        msg_lines.append("<i>No active positions triggered today.</i>\n")

    # SECTION 2: Pending Breakout Watchlist
    msg_lines.append(f"⏳ <b>WAITING FOR TRIGGER ({len(pending)})</b>")
    if not pending.empty:
        p_header = f"{'Symbol':<9} {'Bias':<5} {'Age':<4} {'Level':<8} {'SL':<8}"
        p_sep = "-" * len(p_header)
        p_rows = [p_header, p_sep]
        for _, r in pending.iterrows():
            sym = str(r['Symbol'])[:8]
            bias = "LONG" if str(r.get('Type', 'LONG')).upper() == "LONG" else "SHRT"
            age = f"T+{int(r.get('Days_Active', 1))}"
            p_rows.append(f"{sym:<9} {bias:<5} {age:<4} {float(r['Entry']):<8.1f} {float(r['SL']):<8.1f}")
        p_block = "\n".join(p_rows)
        msg_lines.append(f"<pre>{p_block}</pre>")
        p_links = " | ".join([f"<a href='https://in.tradingview.com/chart/?symbol=NSE:{r['Symbol']}'>{r['Symbol']}</a>" for _, r in pending.iterrows()])
        msg_lines.append(f"📈 <b>Charts:</b> {p_links}\n")
    else:
        msg_lines.append("<i>No setups currently pending.</i>\n")

    msg_lines.append(f"🌐 <a href='{DASHBOARD_URL}'>Open Live Terminal</a>")
    send_telegram("\n".join(msg_lines))


if __name__ == "__main__":
    update_lifecycle_and_track()
