import os
import time
import duckdb
import pandas as pd
import numpy as np
import requests
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
DB_PATH = "data/candles.duckdb"
SIGNALS_CSV = "data/signals.csv"
ACTIVE_WATCHLIST_CSV = "data/active_watchlist.csv"
MAX_HOLD_DAYS = 3  # T+3 sessions before expiring an untriggered setup

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN") or os.getenv("BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID")


def get_db_connection():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}")
    return duckdb.connect(DB_PATH)


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


def sync_new_signals_to_watchlist():
    """Ingests fresh signals into the active watchlist state machine."""
    if not os.path.exists(SIGNALS_CSV):
        return pd.DataFrame()

    signals_df = pd.read_csv(SIGNALS_CSV)
    if signals_df.empty:
        return pd.DataFrame()

    watchlist_df = pd.DataFrame()
    if os.path.exists(ACTIVE_WATCHLIST_CSV):
        try:
            watchlist_df = pd.read_csv(ACTIVE_WATCHLIST_CSV)
        except Exception:
            watchlist_df = pd.DataFrame()

    existing_keys = set()
    if not watchlist_df.empty and 'Symbol' in watchlist_df.columns and 'Date' in watchlist_df.columns:
        existing_keys = set(zip(watchlist_df['Date'], watchlist_df['Symbol']))

    new_records = []
    for _, row in signals_df.iterrows():
        key = (row['Date'], row['Symbol'])
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
    print("🎯 Running Breakout Lifecycle Tracker Engine...")

    watchlist = sync_new_signals_to_watchlist()
    if watchlist.empty:
        print("⚠️ No watchlist records found to track.")
        return

    conn = get_db_connection()
    # Get latest session candle data for all stocks
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
        print("⚠️ No recent market candle data found in DuckDB.")
        return

    candle_map = latest_candles_df.set_index("Symbol").to_dict("index")
    updated_records = []

    for _, row in watchlist.iterrows():
        status = row["Status"]
        sym = row["Symbol"]
        entry = float(row["Entry"])
        sl = float(row["SL"])
        target = float(row["Target"])
        days = int(row.get("Days_Active", 0))

        if status in ["EXPIRED", "STOPPED_OUT", "TARGET_HIT"]:
            # Retain terminal states
            updated_records.append(row.to_dict())
            continue

        if sym not in candle_map:
            updated_records.append(row.to_dict())
            continue

        today_candle = candle_map[sym]
        today_high = float(today_candle["High"])
        today_low = float(today_candle["Low"])
        today_close = float(today_candle["Close"])
        today_date_str = str(today_candle["Date"])

        # -------------------------------------------------------------
        # 1. State: PENDING (Waiting for breakout trigger)
        # -------------------------------------------------------------
        if status == "PENDING":
            days += 1
            # Check for invalidation first (Low violates Stop Loss before breaking out)
            if today_low <= sl:
                row["Status"] = "STOPPED_OUT"
                row["Days_Active"] = days
                send_telegram_msg(
                    f"❌ <b>SETUP CANCELLED / INVALIDATED</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"📈 <b>Stock:</b> {sym}\n"
                    f"⚠️ <b>Reason:</b> Base breached before breakout (Low ₹{today_low:.2f} <= SL ₹{sl:.2f})\n"
                    f"🚫 <b>Action:</b> Remove any resting GTT/limit orders."
                )
            # Check for trigger
            elif today_high >= entry:
                row["Status"] = "TRIGGERED"
                row["Trigger_Date"] = today_date_str
                row["Days_Active"] = days
                send_telegram_msg(
                    f"🚀 <b>BREAKOUT TRIGGERED NOW!</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"📈 <b>Stock:</b> {sym}\n"
                    f"🎯 <b>Trigger Crossed:</b> ₹{entry:.2f} (High hit ₹{today_high:.2f})\n"
                    f"🛑 <b>Active SL:</b> ₹{sl:.2f}\n"
                    f"🎯 <b>Target (3x):</b> ₹{target:.2f}\n"
                    f"⏱ <b>Trigger Session:</b> {today_date_str}"
                )
            # Check for time expiry (Max hold window exceeded)
            elif days >= MAX_HOLD_DAYS:
                row["Status"] = "EXPIRED"
                row["Days_Active"] = days
                send_telegram_msg(
                    f"⏳ <b>SETUP EXPIRED (T+{days})</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"📈 <b>Stock:</b> {sym}\n"
                    f"⚠️ <b>Reason:</b> No trigger within {MAX_HOLD_DAYS} sessions. Volume delta momentum absorbed.\n"
                    f"🚫 <b>Action:</b> Setup discarded."
                )
            else:
                row["Days_Active"] = days

        # -------------------------------------------------------------
        # 2. State: TRIGGERED (Active trade in progress)
        # -------------------------------------------------------------
        elif status == "TRIGGERED":
            if today_low <= sl:
                row["Status"] = "STOPPED_OUT"
                send_telegram_msg(
                    f"🛑 <b>STOP LOSS HIT</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"📉 <b>Stock:</b> {sym}\n"
                    f"Exit trade at SL: ₹{sl:.2f} (Low hit ₹{today_low:.2f})"
                )
            elif today_high >= target:
                row["Status"] = "TARGET_HIT"
                send_telegram_msg(
                    f"🏆 <b>TARGET (3x R:R) ACHIEVED!</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"💰 <b>Stock:</b> {sym}\n"
                    f"Target hit: ₹{target:.2f} (High reached ₹{today_high:.2f})"
                )

        updated_records.append(row.to_dict())

    updated_df = pd.DataFrame(updated_records)
    os.makedirs("data", exist_ok=True)
    updated_df.to_csv(ACTIVE_WATCHLIST_CSV, index=False)
    print(f"✅ Active watchlist updated ({len(updated_df)} tracked setups).")


if __name__ == "__main__":
    update_lifecycle_and_track()
