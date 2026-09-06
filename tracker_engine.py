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
MAX_HOLD_DAYS = 3  # T+3 Expiry rule

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

    # Enforce strict F&O filter
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
    print("🎯 Running F&O Lifecycle Tracker Engine...")

    watchlist = sync_signals_to_watchlist()
    if watchlist.empty:
        print("ℹ️ No active F&O setups in watchlist to evaluate.")
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
        target = float(row["Target"])
        days = int(row.get("Days_Active", 0))

        # Preserve terminal states silently (no spam)
        if status in ["EXPIRED", "STOPPED_OUT", "TARGET_HIT"]:
            updated_records.append(row.to_dict())
            continue

        if sym not in candle_map:
            updated_records.append(row.to_dict())
            continue

        today = candle_map[sym]
        today_high = float(today["High"])
        today_low = float(today["Low"])
        today_close = float(today["Close"])

        # ---------------------------------------------
        # PENDING STATE
        # ---------------------------------------------
        if status == "PENDING":
            days += 1
            if today_low <= sl:
                row["Status"] = "STOPPED_OUT"
                row["Days_Active"] = days
            elif today_high >= entry:
                row["Status"] = "TRIGGERED"
                row["Trigger_Date"] = latest_date_str
                row["Days_Active"] = days

                # ONLY SEND INDIVIDUAL TELEGRAM ON ACTUAL TRIGGER
                send_telegram_msg(
                    f"🚀 <b>BREAKOUT TRIGGERED NOW!</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"📈 <b>Stock:</b> {sym} (NSE F&O EQ)\n"
                    f"🎯 <b>Crossed Entry Level :</b> ₹{entry:.2f} (High ₹{today_high:.2f})\n"
                    f"🛑 <b>Active Stop Loss     :</b> ₹{sl:.2f}\n"
                    f"🎯 <b>Target (3.0x R:R)    :</b> ₹{target:.2f}\n"
                    f"⏱ <b>Session Date         :</b> {latest_date_str}\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"📈 <a href='https://in.tradingview.com/chart/?symbol=NSE:{sym}'>Open TradingView Chart</a>"
                )
            elif days >= MAX_HOLD_DAYS:
                row["Status"] = "EXPIRED"
                row["Days_Active"] = days
            else:
                row["Days_Active"] = days

        # ---------------------------------------------
        # TRIGGERED STATE
        # ---------------------------------------------
        elif status == "TRIGGERED":
            if today_low <= sl:
                row["Status"] = "STOPPED_OUT"
                send_telegram_msg(
                    f"🛑 <b>STOP LOSS HIT</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"📉 <b>Stock:</b> {sym} (Low hit ₹{today_low:.2f} <= SL ₹{sl:.2f})\n"
                    f"🚫 Close position immediately."
                )
            elif today_high >= target:
                row["Status"] = "TARGET_HIT"
                send_telegram_msg(
                    f"🏆 <b>TARGET ACHIEVED (3x R:R)!</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"💰 <b>Stock:</b> {sym} (High reached ₹{today_high:.2f} >= Target ₹{target:.2f})\n"
                    f"🎉 Book profits or trail SL."
                )

        updated_records.append(row.to_dict())

    updated_df = pd.DataFrame(updated_records)
    os.makedirs("data", exist_ok=True)
    updated_df.to_csv(ACTIVE_WATCHLIST_CSV, index=False)

    # SEND CONSOLIDATED TELEGRAM DASHBOARD
    send_active_dashboard_telegram(updated_df, latest_date_str)


def send_active_dashboard_telegram(df: pd.DataFrame, date_str: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    # Filter strictly for active/pending F&O setups
    pending_df = df[df["Status"] == "PENDING"]
    triggered_df = df[df["Status"] == "TRIGGERED"]

    msg_lines = [
        f"🏛️ <b>BRAHMASTRA F&O RADAR DASHBOARD</b>",
        f"📅 <i>Session Date: {date_str}</i>",
        f"━━━━━━━━━━━━━━━━━━━━"
    ]

    # Section 1: Active Triggered Trades
    msg_lines.append(f"🟢 <b>ACTIVE POSITIONS ({len(triggered_df)})</b>")
    if not triggered_df.empty:
        for _, r in triggered_df.iterrows():
            msg_lines.append(
                f"• <b>{r['Symbol']}</b> | Entry: ₹{r['Entry']:.2f} | SL: ₹{r['SL']:.2f} | Target: ₹{r['Target']:.2f}"
            )
    else:
        msg_lines.append("<i>No open active positions.</i>")

    msg_lines.append("\n━━━━━━━━━━━━━━━━━━━━")

    # Section 2: Pending Breakout Watchlist (T+1 to T+3)
    msg_lines.append(f"⏳ <b>PENDING BREAKOUT WATCHLIST ({len(pending_df)})</b>")
    if not pending_df.empty:
        for _, r in pending_df.iterrows():
            hold_str = f"T+{r['Days_Active']}"
            msg_lines.append(
                f"• <b>{r['Symbol']}</b> ({hold_str}) ➔ <b>Buy Above:</b> ₹{r['Entry']:.2f} | SL: ₹{r['SL']:.2f}"
            )
    else:
        msg_lines.append("<i>No pending breakout setups.</i>")

    msg_lines.append("━━━━━━━━━━━━━━━━━━━━")
    msg_lines.append(f"🌐 <a href='{DASHBOARD_URL}'>Open Interactive Web Dashboard</a>")

    full_message = "\n".join(msg_lines)
    send_telegram_msg(full_message)
    print("✅ Dispatched unified F&O radar dashboard to Telegram.")


if __name__ == "__main__":
    update_lifecycle_and_track()
