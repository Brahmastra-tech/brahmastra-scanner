import os
import time
import duckdb
import pandas as pd
import numpy as np
import requests

# ==========================================
# CONFIGURATION
# ==========================================
DB_PATH = "data/candles.duckdb"
SIGNALS_CSV = "data/signals.csv"

TARGET_X = 3.0

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN") or os.getenv("BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID")
DASHBOARD_URL = "https://brahmastra-tech.github.io/brahmastra-scanner/"


def scan_symbol_exact(symbol, df_sym):
    alerts = []

    if df_sym.empty or len(df_sym) < 3:
        return alerts

    df = df_sym.copy().sort_values("Date").reset_index(drop=True)

    # -------------------------------------------------------------
    # 1. STRICT DELIVERY DATA VERIFICATION
    # -------------------------------------------------------------
    deliv_qty_col = None
    for col in ['delivery_qty', 'delivery_volume', 'DeliveryQty', 'DeliveryVolume']:
        if col in df.columns:
            deliv_qty_col = col
            break

    deliv_pct_col = None
    for col in ['delivery_pct', 'DeliveryPct', 'deliv_pct']:
        if col in df.columns:
            deliv_pct_col = col
            break

    # Strictly reject scanning if real delivery columns are missing
    if deliv_qty_col is None or deliv_pct_col is None:
        return alerts

    df['DeliveryQty'] = pd.to_numeric(df[deliv_qty_col], errors='coerce')
    df['DeliveryPct'] = pd.to_numeric(df[deliv_pct_col], errors='coerce')

    df['Prev_DelivQty'] = df['DeliveryQty'].shift(1)

    seen = set()

    for i in range(1, len(df)):
        row = df.iloc[i]

        open_p = float(row["Open"])
        high_p = float(row["High"])
        low_p = float(row["Low"])
        close_p = float(row["Close"])
        volume = float(row["Volume"])

        # Skip rows with missing delivery data
        if pd.isna(row["DeliveryQty"]) or pd.isna(row["DeliveryPct"]) or pd.isna(row["Prev_DelivQty"]):
            continue

        deliv_qty = float(row["DeliveryQty"])
        prev_deliv_qty = float(row["Prev_DelivQty"])
        deliv_pct = float(row["DeliveryPct"])

        # Reject if previous delivery quantity was zero or invalid
        if prev_deliv_qty <= 0:
            continue

        # -------------------------------------------------------------
        # EXACT 6 DELIVERY ACCUMULATION RULES
        # -------------------------------------------------------------
        c1_vol = volume > 500000                                  # [0] Volume > 500000
        c2_deliv_spike = deliv_qty > (prev_deliv_qty * 3.0)      # [0] Delivery Vol > Prev Delivery Vol * 3
        c3_deliv_pct = deliv_pct > 55.0                           # [0] Delivery % > 55
        c4_close_min = close_p >= (open_p * 0.99)                # [0] Close >= Open * 0.99
        c5_close_max = close_p <= (open_p * 1.02)                # [0] Close <= Open * 1.02
        c6_range_squeeze = (high_p - low_p) <= (close_p * 0.03)   # [0] High - Low <= Close * 0.03

        if not (c1_vol and c2_deliv_spike and c3_deliv_pct and c4_close_min and c5_close_max and c6_range_squeeze):
            continue

        # FIX 1: Include row["Date"] in the key so new dates aren't skipped!
        date_str = pd.to_datetime(row["Date"]).strftime("%d-%m-%Y")
        key = (symbol, date_str, "DELIVERY_ACCUMULATION")

        if key not in seen:
            seen.add(key)

            entry = round(high_p, 2)
            sl = round(low_p, 2)
            target = round(entry + (entry - sl) * TARGET_X, 2)
            spike_ratio = round(deliv_qty / prev_deliv_qty, 2)

            alerts.append({
                "Date": date_str,
                "Symbol": symbol,
                "Timeframe": "D",
                "Type": "PRE_BREAKOUT",
                "Pattern": "DELIVERY_ACCUMULATION",
                "Entry": entry,
                "SL": sl,
                "Target": target,
                "Close": round(close_p, 2),
                "Volume": int(volume),
                "DeliveryQty": int(deliv_qty),
                "DeliveryPct": round(deliv_pct, 2),
                "DelivSpikeRatio": spike_ratio
            })

    return alerts


def send_telegram_alert(signal: dict):
    """Sends individual stock alert to Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram credentials missing. Skipping individual alert.")
        return

    symbol = signal.get("Symbol")
    setup_date = signal.get("Date")
    entry = signal.get("Entry")
    sl = signal.get("SL")
    target = signal.get("Target")
    close = signal.get("Close")
    deliv_pct = signal.get("DeliveryPct", 0.0)
    deliv_spike = signal.get("DelivSpikeRatio", 0.0)

    emoji = "🚀"
    chart_url = f"https://in.tradingview.com/chart/?symbol=NSE:{symbol}"

    message = (
        f"{emoji} <b>BRAHMASTRA ACCUMULATION WATCHLIST</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 <b>Stock:</b> {symbol} (NSE F&O)\n"
        f"🎯 <b>Pattern:</b> 3x Delivery Accumulation Squeeze\n"
        f"⏱ <b>Setup Date:</b> {setup_date}\n\n"
        f"📊 <b>ACTIONABLE TRIGGER LEVELS</b>\n"
        f"• <b>Trigger Buy Above   :</b> ₹{entry:.2f}\n"
        f"• <b>Stop Loss           :</b> ₹{sl:.2f}\n"
        f"• <b>Target (3x)         :</b> ₹{target:.2f}\n"
        f"• <b>Today's Close       :</b> ₹{close:.2f}\n\n"
        f"⚡ <b>DELIVERY CONVICTION METRICS</b>\n"
        f"• <b>Delivery %          :</b> {deliv_pct:.1f}%\n"
        f"• <b>Delivery Spike      :</b> {deliv_spike:.2f}x vs Yesterday\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 <a href='{chart_url}'>View {symbol} TradingView Chart</a>"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        res = requests.post(url, json=payload, timeout=10)
        if res.status_code == 200:
            print(f"✅ Telegram alert sent for {symbol}")
        else:
            print(f"❌ Telegram API Error ({res.status_code}): {res.text}")
    except Exception as e:
        print(f"❌ Exception sending Telegram alert: {e}")


def send_summary_telegram(signals_today: list, date_str: str):
    """Sends scan execution summary to Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram credentials missing. Skipping summary dispatch.")
        return

    total_count = len(signals_today)

    message = (
        f"🏁 <b>DAILY DELIVERY ACCUMULATION SCAN COMPLETE ({date_str})</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>High-Delivery Squeeze Candidates Found Today:</b> {total_count}\n\n"
        f"🌐 <b>Interactive Web Dashboard & Full History:</b>\n"
        f"👉 <a href='{DASHBOARD_URL}'>{DASHBOARD_URL}</a>\n"
        f"━━━━━━━━━━━━━━━━━━━━"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": False
    }
    try:
        res = requests.post(url, json=payload, timeout=10)
        if res.status_code == 200:
            print(f"✅ Telegram summary sent successfully for {date_str}")
        else:
            print(f"❌ Telegram Summary API Error ({res.status_code}): {res.text}")
    except Exception as e:
        print(f"❌ Exception sending Telegram summary: {e}")


def run_scanner():
    print("🚀 Running Fixed Delivery Accumulation Scanner Engine...")

    if not os.path.exists(DB_PATH):
        print(f"❌ Database file not found at {DB_PATH}!")
        return

    conn = duckdb.connect(DB_PATH)

    df_raw = conn.execute("""
        SELECT * FROM ohlcv_candles ORDER BY symbol, timestamp ASC
    """).df()

    if df_raw.empty:
        print("⚠️ No candles available in DuckDB.")
        return

    # Standardize Column Names
    df_raw.rename(columns={
        'symbol': 'Symbol', 'timestamp': 'Date', 'open': 'Open',
        'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
    }, inplace=True)

    latest_date_str = pd.to_datetime(df_raw['Date'].max()).strftime("%d-%m-%Y")
    print(f"🔍 Database Market Date: {latest_date_str} | Total Symbols: {df_raw['Symbol'].nunique()}")

    symbols = df_raw['Symbol'].unique()
    all_signals = []

    for sym in symbols:
        df_sym = df_raw[df_raw['Symbol'] == sym]
        alerts = scan_symbol_exact(sym, df_sym)
        if alerts:
            all_signals.extend(alerts)

    os.makedirs("data", exist_ok=True)

    if not all_signals:
        print("ℹ️ No signals matched strict conditions overall.")
        pd.DataFrame(columns=['Date', 'Symbol', 'Timeframe', 'Type', 'Pattern', 'Entry', 'SL', 'Target', 'Close', 'Volume', 'DeliveryQty', 'DeliveryPct', 'DelivSpikeRatio']).to_csv(SIGNALS_CSV, index=False)
        send_summary_telegram([], latest_date_str)
        return

    # Export clean history to CSV
    export_df = pd.DataFrame(all_signals)
    date_col = "Date" if "Date" in export_df.columns else "date"
    export_df["Date_DT"] = pd.to_datetime(export_df[date_col], format="%d-%m-%Y")

    export_df = export_df.sort_values("Date_DT", ascending=False).drop(columns=["Date_DT"])
    export_df.to_csv(SIGNALS_CSV, index=False)
    print(f"✅ Saved {len(export_df)} total historical signals to {SIGNALS_CSV}.")

    # Extract today's signals accurately using matching formatted strings
    today_signals = export_df[export_df[date_col] == latest_date_str].to_dict('records')
    print(f"📊 Candidates for Today ({latest_date_str}): {len(today_signals)}")

    try:
        for sig in today_signals:
            send_telegram_alert(sig)
            time.sleep(0.5)
    finally:
        send_summary_telegram(today_signals, latest_date_str)


if __name__ == "__main__":
    run_scanner()