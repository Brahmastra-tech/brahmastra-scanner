import os
import time
import duckdb
import pandas as pd
import numpy as np
import requests
from datetime import datetime

# ==========================================
# CONFIGURATION & CANONICAL F&O UNIVERSE
# ==========================================
DB_PATH = "data/candles.duckdb"
SIGNALS_CSV = "data/signals.csv"
ACTIVE_WATCHLIST_CSV = "data/active_watchlist.csv"
MAX_HOLD_DAYS = 3  # T+3 Expiry Rule

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN") or os.getenv("BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID")
DASHBOARD_URL = "[https://brahmastra-tech.github.io/brahmastra-scanner/](https://brahmastra-tech.github.io/brahmastra-scanner/)"

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


def send_telegram(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"[https://api.telegram.org/bot](https://api.telegram.org/bot){TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True
    }
    try:
        res = requests.post(url, json=payload, timeout=10)
        if res.status_code != 200:
            print(f"❌ Telegram Error ({res.status_code}): {res.text}")
        else:
            print("🚀 Radar table successfully delivered to Telegram!")
    except Exception as err:
        print(f"⚠️ Telegram Network Error: {err}")


def update_lifecycle_and_track():
    print("🎯 Running Breakout Radar & Lifecycle Engine...")

    if not os.path.exists(SIGNALS_CSV):
        print("⚠️ signals.csv not found.")
        return

    signals_df = pd.read_csv(SIGNALS_CSV)
    if signals_df.empty or 'Symbol' not in signals_df.columns:
        print("ℹ️ signals.csv is empty.")
        return

    # Strict F&O filtering
    signals_df['Symbol_Clean'] = signals_df['Symbol'].astype(str).str.upper().str.strip()
    signals_df = signals_df[signals_df['Symbol_Clean'].isin(NSE_FO_SYMBOLS)].copy()

    # Parse all dates into uniform YYYY-MM-DD strings
    signals_df['Date_Obj'] = pd.to_datetime(signals_df['Date'], dayfirst=True, errors='coerce')
    signals_df['Date_Norm'] = signals_df['Date_Obj'].dt.strftime("%Y-%m-%d")

    # Fetch latest session data from DuckDB
    conn = duckdb.connect(DB_PATH)
    candles_df = conn.execute("""
        SELECT 
            symbol AS Symbol, 
            CAST(timestamp AS DATE) AS Date,
            high AS High,
            low AS Low,
            close AS Close
        FROM ohlcv_candles
        WHERE timestamp = (SELECT MAX(timestamp) FROM ohlcv_candles)
    """).df()
    conn.close()

    if candles_df.empty:
        print("⚠️ DuckDB candles empty.")
        return

    latest_session_norm = pd.to_datetime(candles_df['Date'].max()).strftime("%Y-%m-%d")
    latest_session_display = pd.to_datetime(candles_df['Date'].max()).strftime("%d-%m-%Y")
    candle_dict = candles_df.set_index("Symbol").to_dict("index")

    # Load existing active watchlist
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

    # Ingest fresh signals
    new_items = []
    for _, row in signals_df.iterrows():
        key = (str(row['Date_Norm']), row['Symbol_Clean'])
        if key not in existing_keys:
            is_today = (str(row['Date_Norm']) == latest_session_norm)
            new_items.append({
                "Date": row['Date'],
                "Date_Norm": str(row['Date_Norm']),
                "Symbol": row['Symbol_Clean'],
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

    # Process state machine
    updated = []
    for _, row in wl_df.iterrows():
        sym = row['Symbol']
        status = row['Status']
        sig_date_norm = str(row.get('Date_Norm', ''))
        entry = float(row['Entry'])
        sl = float(row['SL'])
        target = float(row['Target'])
        days = int(row.get('Days_Active', 1))

        # Locked terminal states
        if status in ["EXPIRED", "STOPPED_OUT", "TARGET_HIT"]:
            updated.append(row.to_dict())
            continue

        # Rule: Today's fresh alert cannot be evaluated against today's candle
        if sig_date_norm == latest_session_norm:
            row['Status'] = "PENDING"
            row['Days_Active'] = 1
            updated.append(row.to_dict())
            continue

        # Evaluate older signals (T-1, T-2) against today's price action
        if sym in candle_dict:
            c = candle_dict[sym]
            hi = float(c['High'])
            lo = float(c['Low'])

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

        updated.append(row.to_dict())

    final_wl = pd.DataFrame(updated)
    os.makedirs("data", exist_ok=True)
    final_wl.to_csv(ACTIVE_WATCHLIST_CSV, index=False)

    # Dispatch formatted Markdown table
    send_telegram_radar_table(final_wl, latest_session_display)


def send_telegram_radar_table(df: pd.DataFrame, date_str: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    triggered = df[df["Status"] == "TRIGGERED"]
    pending = df[df["Status"] == "PENDING"]

    lines = [
        "🏛️ *BRAHMASTRA BREAKOUT RADAR*",
        f"📅 _Session Date: {date_str}_\n"
    ]

    # Table 1: Triggered Breakouts
    lines.append(f"🚀 *BREAKOUT TRIGGERED NOW ({len(triggered)})*")
    if not triggered.empty:
        t_header = f"{'Symbol':<9} {'Entry':<8} {'SL':<8} {'Target':<8}"
        t_sep = "-" * len(t_header)
        t_rows = [t_header, t_sep]
        for _, r in triggered.iterrows():
            sym = str(r['Symbol'])[:8]
            t_rows.append(f"{sym:<9} {float(r['Entry']):<8.1f} {float(r['SL']):<8.1f} {float(r['Target']):<8.1f}")
        t_block = "\n".join(t_rows)
        lines.append(f"```\n{t_block}\n```")
        links = " | ".join([f"[{r['Symbol']}](https://in.tradingview.com/chart/?symbol=NSE:{r['Symbol']})" for _, r in triggered.iterrows()])
        lines.append(f"📈 *Charts:* {links}\n")
    else:
        lines.append("_No open triggered breakouts._\n")

    # Table 2: Waiting for Breakout
    lines.append(f"⏳ *WAITING FOR BREAKOUT ({len(pending)})*")
    if not pending.empty:
        p_header = f"{'Symbol':<9} {'Age':<4} {'Trigger':<8} {'SL':<8}"
        p_sep = "-" * len(p_header)
        p_rows = [p_header, p_sep]
        for _, r in pending.iterrows():
            sym = str(r['Symbol'])[:8]
            age = f"T+{int(r['Days_Active'])}"
            p_rows.append(f"{sym:<9} {age:<4} {float(r['Entry']):<8.1f} {float(r['SL']):<8.1f}")
        p_block = "\n".join(p_rows)
        lines.append(f"```\n{p_block}\n```")
        p_links = " | ".join([f"[{r['Symbol']}](https://in.tradingview.com/chart/?symbol=NSE:{r['Symbol']})" for _, r in pending.iterrows()])
        lines.append(f"📈 *Charts:* {p_links}\n")
    else:
        lines.append("_No pending setups in radar._\n")

    lines.append(f"🌐 [Open Web Dashboard]({DASHBOARD_URL})")

    send_telegram("\n".join(lines))


if __name__ == "__main__":
    update_lifecycle_and_track()
