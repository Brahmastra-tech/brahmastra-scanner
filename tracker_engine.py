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


def send_telegram_msg(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
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
        print(f"⚠️ Telegram network error: {e}")


def sync_signals_to_watchlist():
    if not os.path.exists(SIGNALS_CSV):
        return pd.DataFrame()

    try:
        signals_df = pd.read_csv(SIGNALS_CSV)
    except Exception:
        return pd.DataFrame()

    if signals_df.empty or 'Symbol' not in signals_df.columns:
        return pd.DataFrame()

    signals_df['Symbol_Clean'] = signals_df['Symbol'].astype(str).str.upper().str.strip()
    signals_df = signals_df[signals_df['Symbol_Clean'].isin(NSE_FO_SYMBOLS)].copy()

    watchlist_df = pd.DataFrame()
    if os.path.exists(ACTIVE_WATCHLIST_CSV):
        try:
            watchlist_df = pd.read_csv(ACTIVE_WATCHLIST_CSV)
            if not watchlist_df.empty:
                watchlist_df['Symbol_Clean'] = watchlist_df['Symbol'].astype(str).str.upper().str.strip()
                watchlist_df = watchlist_df[watchlist_df['Symbol_Clean'].isin(NSE_FO_SYMBOLS)].copy()
        except Exception:
            watchlist_df = pd.DataFrame()

    existing_keys = set()
    if not watchlist_df.empty and 'Date' in watchlist_df.columns and 'Symbol' in watchlist_df.columns:
        existing_keys = set(zip(watchlist_df['Date'], watchlist_df['Symbol']))

    new_records = []
    for _, row in signals_df.iterrows():
        key = (str(row['Date']), str(row['Symbol']))
        if key not in existing_keys:
            new_records.append({
                "Date": row['Date'],
                "Symbol": row['Symbol'],
                "Entry": float(row['Entry']),
                "SL": float(row['SL']),
                "Target": float(row['Target']),
                "Close": float(row['Close']),
                "Status": "PENDING",
                "Days_Active": 0,
                "Trigger_Date": ""
            })

    if new_records:
        new_df = pd.DataFrame(new_records)
        watchlist_df = pd.concat([watchlist_df, new_df], ignore_index=True) if not watchlist_df.empty else new_df

    return watchlist_df


def update_lifecycle_and_track():
    print("🎯 Running Unified F&O Lifecycle Table Engine...")

    watchlist = sync_signals_to_watchlist()
    if watchlist.empty:
        print("ℹ️ No active F&O setups found to evaluate.")
        return

    if not os.path.exists(DB_PATH):
        return

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
        return

    latest_date_str = str(latest_candles_df['Date'].max())
    candle_map = latest_candles_df.set_index("Symbol").to_dict("index")
    updated_records = []

    for _, row in watchlist.iterrows():
        status = row["Status"]
        sym = str(row["Symbol"]).upper().strip()
        entry = float(row["Entry"])
        sl = float(row["SL"])
        days = int(row.get("Days_Active", 0))

        if status in ["EXPIRED", "STOPPED_OUT", "TARGET_HIT"]:
            updated_records.append(row.to_dict())
            continue

        if sym not in candle_map:
            updated_records.append(row.to_dict())
            continue

        today = candle_map[sym]
        today_high = float(today["High"])
        today_low = float(today["Low"])

        if status == "PENDING":
            days += 1
            if today_low <= sl:
                row["Status"] = "STOPPED_OUT"
                row["Days_Active"] = days
            elif today_high >= entry:
                row["Status"] = "TRIGGERED"
                row["Trigger_Date"] = latest_date_str
                row["Days_Active"] = days
            elif days >= MAX_HOLD_DAYS:
                row["Status"] = "EXPIRED"
                row["Days_Active"] = days
            else:
                row["Days_Active"] = days

        elif status == "TRIGGERED":
            if today_low <= sl:
                row["Status"] = "STOPPED_OUT"
            elif today_high >= float(row["Target"]):
                row["Status"] = "TARGET_HIT"

        updated_records.append(row.to_dict())

    updated_df = pd.DataFrame(updated_records)
    os.makedirs("data", exist_ok=True)
    updated_df.to_csv(ACTIVE_WATCHLIST_CSV, index=False)

    # Broadcast single consolidated table dashboard to Telegram
    send_consolidated_table_telegram(updated_df, latest_date_str)


def send_consolidated_table_telegram(df: pd.DataFrame, date_str: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    triggered_df = df[df["Status"] == "TRIGGERED"]
    pending_df = df[df["Status"] == "PENDING"]

    msg_parts = [
        f"🏛️ <b>BRAHMASTRA BREAKOUT RADAR</b>",
        f"📅 <i>Session: {date_str}</i>\n"
    ]

    # 1. Triggered Table (Breakout Active)
    msg_parts.append(f"🚀 <b>BREAKOUT TRIGGERED NOW ({len(triggered_df)})</b>")
    if not triggered_df.empty:
        header = f"{'Symbol':<10} {'Entry':<8} {'SL':<8} {'Target':<8}"
        sep = "-" * len(header)
        rows = [header, sep]
        for _, r in triggered_df.iterrows():
            sym = str(r['Symbol'])[:9]
            entry_s = f"{float(r['Entry']):.1f}"
            sl_s = f"{float(r['SL']):.1f}"
            tgt_s = f"{float(r['Target']):.1f}"
            rows.append(f"{sym:<10} {entry_s:<8} {sl_s:<8} {tgt_s:<8}")
        
        table_str = "\n".join(rows)
        msg_parts.append(f"<pre>\n{table_str}\n</pre>")

        # TradingView direct Links
        tv_links = " | ".join([f"<a href='https://in.tradingview.com/chart/?symbol=NSE:{r['Symbol']}'>{r['Symbol']}</a>" for _, r in triggered_df.iterrows()])
        msg_parts.append(f"📈 <b>Charts:</b> {tv_links}\n")
    else:
        msg_parts.append("<i>No active triggered breakouts today.</i>\n")

    # 2. Waiting to Breakout Table (Pending T+1 to T+3)
    msg_parts.append(f"⏳ <b>WAITING FOR BREAKOUT ({len(pending_df)})</b>")
    if not pending_df.empty:
        header_p = f"{'Symbol':<9} {'Age':<4} {'Buy>':<8} {'SL':<8}"
        sep_p = "-" * len(header_p)
        rows_p = [header_p, sep_p]
        for _, r in pending_df.iterrows():
            sym = str(r['Symbol'])[:8]
            age = f"T+{r['Days_Active']}"
            entry_s = f"{float(r['Entry']):.1f}"
            sl_s = f"{float(r['SL']):.1f}"
            rows_p.append(f"{sym:<9} {age:<4} {entry_s:<8} {sl_s:<8}")

        table_str_p = "\n".join(rows_p)
        msg_parts.append(f"<pre>\n{table_str_p}\n</pre>")

        tv_links_p = " | ".join([f"<a href='https://in.tradingview.com/chart/?symbol=NSE:{r['Symbol']}'>{r['Symbol']}</a>" for _, r in pending_df.iterrows()])
        msg_parts.append(f"📈 <b>Charts:</b> {tv_links_p}\n")
    else:
        msg_parts.append("<i>No setups currently pending.</i>\n")

    msg_parts.append(f"🌐 <a href='{DASHBOARD_URL}'>Open Full Terminal Dashboard</a>")

    full_message = "\n".join(msg_parts)
    send_telegram_msg(full_message)
    print("✅ Dispatched table-formatted dashboard to Telegram.")


if __name__ == "__main__":
    update_lifecycle_and_track()
