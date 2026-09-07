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


def send_telegram_msg(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram token or chat ID missing.")
        return
    url = f"[https://api.telegram.org/bot](https://api.telegram.org/bot){TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code != 200:
            print(f"❌ Telegram API Error ({resp.status_code}): {resp.text}")
        else:
            print("🚀 Successfully delivered radar table to Telegram!")
    except Exception as e:
        print(f"⚠️ Telegram network error: {e}")


def update_lifecycle_and_track():
    print("🎯 Running Breakout Lifecycle Tracker Engine...")

    if not os.path.exists(SIGNALS_CSV):
        print("⚠️ signals.csv not found.")
        return

    signals_df = pd.read_csv(SIGNALS_CSV)
    if signals_df.empty or 'Symbol' not in signals_df.columns:
        print("ℹ️ No signals found in signals.csv.")
        return

    # Strictly filter for F&O
    signals_df['Symbol_Clean'] = signals_df['Symbol'].astype(str).str.upper().str.strip()
    signals_df = signals_df[signals_df['Symbol_Clean'].isin(NSE_FO_SYMBOLS)].copy()

    signals_df['Date_Parsed'] = pd.to_datetime(signals_df['Date'], dayfirst=True, errors='coerce')
    signals_df['Date_Norm'] = signals_df['Date_Parsed'].dt.strftime("%Y-%m-%d")

    # Connect to DuckDB to retrieve the latest candle date
    conn = duckdb.connect(DB_PATH)
    latest_candles_df = conn.execute("""
        SELECT 
            symbol AS Symbol, 
            CAST(timestamp AS DATE) as Date,
            open AS Open,
            high AS High,
            low AS Low,
            close AS Close
        FROM ohlcv_candles
        WHERE timestamp = (SELECT MAX(timestamp) FROM ohlcv_candles)
    """).df()
    conn.close()

    if latest_candles_df.empty:
        print("⚠️ No candle data available in DuckDB.")
        return

    # Normalize latest session date string (e.g. '2026-09-04')
    latest_session_dt = pd.to_datetime(latest_candles_df['Date'].max()).strftime("%Y-%m-%d")
    candle_map = latest_candles_df.set_index("Symbol").to_dict("index")

    # Load existing active watchlist if available
    watchlist_df = pd.DataFrame()
    if os.path.exists(ACTIVE_WATCHLIST_CSV):
        try:
            watchlist_df = pd.read_csv(ACTIVE_WATCHLIST_CSV)
            if not watchlist_df.empty:
                watchlist_df['Symbol_Clean'] = watchlist_df['Symbol'].astype(str).str.upper().str.strip()
                watchlist_df = watchlist_df[watchlist_df['Symbol_Clean'].isin(NSE_FO_SYMBOLS)].copy()
        except Exception:
            watchlist_df = pd.DataFrame()

    existing_set = set()
    if not watchlist_df.empty and 'Date_Norm' in watchlist_df.columns:
        existing_set = set(zip(watchlist_df['Date_Norm'], watchlist_df['Symbol_Clean']))

    # Intake fresh signals
    new_rows = []
    for _, s_row in signals_df.iterrows():
        key = (s_row['Date_Norm'], s_row['Symbol_Clean'])
        if key not in existing_set:
            new_rows.append({
                "Date": s_row['Date'],
                "Date_Norm": s_row['Date_Norm'],
                "Symbol": s_row['Symbol_Clean'],
                "Entry": float(s_row['Entry']),
                "SL": float(s_row['SL']),
                "Target": float(s_row['Target']),
                "Close": float(s_row['Close']),
                "Status": "PENDING",
                "Days_Active": 1 if s_row['Date_Norm'] == latest_session_dt else 2,
                "Trigger_Date": ""
            })

    if new_rows:
        watchlist_df = pd.concat([watchlist_df, pd.DataFrame(new_rows)], ignore_index=True)

    updated_records = []
    for _, row in watchlist_df.iterrows():
        sym = row["Symbol"]
        status = row["Status"]
        sig_date = str(row.get("Date_Norm", ""))
        entry = float(row["Entry"])
        sl = float(row["SL"])
        target = float(row["Target"])
        days = int(row.get("Days_Active", 1))

        if status in ["EXPIRED", "STOPPED_OUT", "TARGET_HIT"]:
            updated_records.append(row.to_dict())
            continue

        # RULE: A signal generated on today's session must stay PENDING for tomorrow
        if sig_date == latest_session_dt:
            row["Status"] = "PENDING"
            row["Days_Active"] = 1
            updated_records.append(row.to_dict())
            continue

        # For older signals (T-1, T-2), check today's actual price action
        if sym in candle_map:
            today_candle = candle_map[sym]
            today_high = float(today_candle["High"])
            today_low = float(today_candle["Low"])

            if status == "PENDING":
                if today_low <= sl:
                    row["Status"] = "STOPPED_OUT"
                elif today_high >= entry:
                    row["Status"] = "TRIGGERED"
                    row["Trigger_Date"] = latest_session_dt
                else:
                    days += 1
                    row["Days_Active"] = days
                    if days > MAX_HOLD_DAYS:
                        row["Status"] = "EXPIRED"

            elif status == "TRIGGERED":
                if today_low <= sl:
                    row["Status"] = "STOPPED_OUT"
                elif today_high >= target:
                    row["Status"] = "TARGET_HIT"

        updated_records.append(row.to_dict())

    final_df = pd.DataFrame(updated_records)
    os.makedirs("data", exist_ok=True)
    final_df.to_csv(ACTIVE_WATCHLIST_CSV, index=False)

    send_clean_table_telegram(final_df, latest_session_dt)


def send_clean_table_telegram(df: pd.DataFrame, date_str: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    triggered_df = df[df["Status"] == "TRIGGERED"]
    pending_df = df[df["Status"] == "PENDING"]

    msg = f"🏛️ <b>BRAHMASTRA BREAKOUT RADAR</b>\n"
    msg += f"📅 <i>Session: {date_str}</i>\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━\n\n"

    # SECTION 1: Triggered Table
    msg += f"🚀 <b>BREAKOUT TRIGGERED NOW ({len(triggered_df)})</b>\n"
    if not triggered_df.empty:
        header = f"{'Symbol':<9} {'Entry':<8} {'SL':<8} {'Target':<8}"
        sep = "-" * 35
        table_lines = [header, sep]
        for _, r in triggered_df.iterrows():
            table_lines.append(f"{str(r['Symbol'])[:8]:<9} {float(r['Entry']):<8.1f} {float(r['SL']):<8.1f} {float(r['Target']):<8.1f}")
        table_body = "\n".join(table_lines)
        msg += f"<pre>{table_body}</pre>\n"
        links = " | ".join([f"<a href='[https://in.tradingview.com/chart/?symbol=NSE](https://in.tradingview.com/chart/?symbol=NSE):{r['Symbol']}'>{r['Symbol']}</a>" for _, r in triggered_df.iterrows()])
        msg += f"📈 <b>Charts:</b> {links}\n\n"
    else:
        msg += "<i>No triggered trades active.</i>\n\n"

    # SECTION 2: Waiting for Breakout Table
    msg += f"⏳ <b>WAITING FOR BREAKOUT ({len(pending_df)})</b>\n"
    if not pending_df.empty:
        header_p = f"{'Symbol':<9} {'Age':<4} {'Trigger':<8} {'SL':<8}"
        sep_p = "-" * 31
        table_lines_p = [header_p, sep_p]
        for _, r in pending_df.iterrows():
            table_lines_p.append(f"{str(r['Symbol'])[:8]:<9} T+{int(r['Days_Active']):<2} {float(r['Entry']):<8.1f} {float(r['SL']):<8.1f}")
        table_body_p = "\n".join(table_lines_p)
        msg += f"<pre>{table_body_p}</pre>\n"
        links_p = " | ".join([f"<a href='[https://in.tradingview.com/chart/?symbol=NSE](https://in.tradingview.com/chart/?symbol=NSE):{r['Symbol']}'>{r['Symbol']}</a>" for _, r in pending_df.iterrows()])
        msg += f"📈 <b>Charts:</b> {links_p}\n\n"
    else:
        msg += "<i>No pending setups in radar.</i>\n\n"

    msg += f"━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"🌐 <a href='{DASHBOARD_URL}'>Open Web Dashboard</a>"

    send_telegram_msg(msg)


if __name__ == "__main__":
    update_lifecycle_and_track()
