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

    if df_sym.empty or len(df_sym) < 5:
        return alerts

    df = df_sym.copy().sort_values("Date").reset_index(drop=True)

    # Ensure delivery columns exist; fallback to estimated delivery volume if schema varies
    if 'DeliveryQty' not in df.columns or 'DeliveryPct' not in df.columns:
        df['DeliveryQty'] = df['Volume'] * 0.5
        df['DeliveryPct'] = 50.0

    df['Prev_DelivQty'] = df['DeliveryQty'].shift(1)

    seen = set()

    for i in range(1, len(df)):
        row = df.iloc[i]

        open_p = float(row["Open"])
        high_p = float(row["High"])
        low_p = float(row["Low"])
        close_p = float(row["Close"])
        volume = float(row["Volume"])
        deliv_qty = float(row["DeliveryQty"]) if pd.notna(row["DeliveryQty"]) else 0.0
        prev_deliv_qty = float(row["Prev_DelivQty"]) if pd.notna(row["Prev_DelivQty"]) else 0.0
        deliv_pct = float(row["DeliveryPct"]) if pd.notna(row["DeliveryPct"]) else 0.0

        # -------------------------------------------------------------
        # ACCUMULATION SCANNER CONDITIONS (Optimized for High Conviction)
        # -------------------------------------------------------------
        c1_vol = volume >= 400000                                # Liquidity Floor
        c2_deliv_spike = deliv_qty >= (prev_deliv_qty * 1.8)     # Delivery Spike vs Prev Day
        c3_deliv_pct = deliv_pct >= 50.0                          # Delivery Share %
        c4_close_min = close_p >= (open_p * 0.985)               # Open-Close Range Bounds
        c5_close_max = close_p <= (open_p * 1.025)               
        c6_range_squeeze = (high_p - low_p) <= (close_p * 0.035)  # Range Squeeze (High - Low)

        if not (c1_vol and c2_deliv_spike and c3_deliv_pct and c4_close_min and c5_close_max and c6_range_squeeze):
            continue

        key = (symbol, "DELIVERY_ACCUMULATION")

        if key not in seen:
            seen.add(key)

            entry = round(high_p, 2)
            sl = round(low_p, 2)
            target = round(entry + (entry - sl) * TARGET_X, 2)

            alerts.append({
                "Date": pd.to_datetime(row["Date"]).strftime("%d-%m-%Y"),
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
                "DelivSpikeRatio": round(deliv_qty / prev_deliv_qty, 2) if prev_deliv_qty > 0 else 1.8
            })

    return alerts


def send_telegram_alert(signal: dict):
    """Sends individual stock alert to Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram credentials missing. Skipping individual alert.")
        return

    symbol = signal.get("Symbol")
    sig_type = signal.get("Type", "PRE_BREAKOUT")
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
        f"🎯 <b>Pattern:</b> Delivery Accumulation Squeeze\n"
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
    print("🚀 Initializing Delivery Accumulation Scanner Engine...")

    if not os.path.exists(DB_PATH):
        print(f"❌ Database file not found at {DB_PATH}!")
        return

    conn = duckdb.connect(DB_PATH)

    try:
        df_raw = conn.execute("""
            SELECT symbol AS Symbol, CAST(timestamp AS DATE) AS Date, open AS Open, high AS High, low AS Low, close AS Close, volume AS Volume,
                   delivery_qty AS DeliveryQty, delivery_pct AS DeliveryPct
            FROM ohlcv_candles
            ORDER BY symbol, timestamp ASC
        """).df()
    except Exception:
        df_raw = conn.execute("""
            SELECT symbol AS Symbol, CAST(timestamp AS DATE) AS Date, open AS Open, high AS High, low AS Low, close AS Close, volume AS Volume
            FROM ohlcv_candles
            ORDER BY symbol, timestamp ASC
        """).df()

    if df_raw.empty:
        print("⚠️ No candles available in DuckDB.")
        return

    latest_date = pd.to_datetime(df_raw['Date'].max()).strftime("%d-%m-%Y")
    print(f"🔍 Database Market Date: {latest_date} | Total Symbols: {df_raw['Symbol'].nunique()}")

    symbols = df_raw['Symbol'].unique()
    new_signals = []

    # 1. Evaluate scanner logic across all historical DuckDB candles
    for sym in symbols:
        df_sym = df_raw[df_raw['Symbol'] == sym]
        alerts = scan_symbol_exact(sym, df_sym)
        if alerts:
            new_signals.extend(alerts)

    os.makedirs("data", exist_ok=True)

    # Convert new scan results to DataFrame
    if new_signals:
        new_df = pd.DataFrame(new_signals)
    else:
        new_df = pd.DataFrame(columns=['Date', 'Symbol', 'Timeframe', 'Type', 'Pattern', 'Entry', 'SL', 'Target', 'Close', 'Volume', 'DeliveryQty', 'DeliveryPct', 'DelivSpikeRatio'])

    # 2. READ & MERGE WITH EXISTING SIGNALS.CSV TO PRESERVE HISTORICAL DASHBOARD DATA
    if os.path.exists(SIGNALS_CSV):
        try:
            old_df = pd.read_csv(SIGNALS_CSV)
            if not old_df.empty:
                combined_df = pd.concat([new_df, old_df], ignore_index=True)
                # Deduplicate by Date and Symbol so old historical signals are kept intact
                final_export_df = combined_df.drop_duplicates(subset=["Date", "Symbol"], keep="first")
            else:
                final_export_df = new_df
        except Exception:
            final_export_df = new_df
    else:
        final_export_df = new_df

    # 3. Sort by Date and Export
    if not final_export_df.empty:
        date_col = "Date" if "Date" in final_export_df.columns else "date"
        final_export_df["Date_DT"] = pd.to_datetime(final_export_df[date_col], format="%d-%m-%Y")
        export_df = final_export_df.sort_values("Date_DT", ascending=False).drop(columns=["Date_DT"])
    else:
        export_df = final_export_df

    export_df.to_csv(SIGNALS_CSV, index=False)
    print(f"✅ Saved/Preserved {len(export_df)} total historical signals in {SIGNALS_CSV}.")

    # 4. Extract today's signals for Telegram broadcast
    if not export_df.empty:
        date_col = "Date" if "Date" in export_df.columns else "date"
        today_signals = export_df[export_df[date_col] == latest_date].to_dict('records')
    else:
        today_signals = []

    print(f"📊 Candidates for Today ({latest_date}): {len(today_signals)}")

    # 5. Dispatch Telegram notifications
    try:
        for sig in today_signals:
            send_telegram_alert(sig)
            time.sleep(0.5)
    finally:
        send_summary_telegram(today_signals, latest_date)


if __name__ == "__main__":
    run_scanner()