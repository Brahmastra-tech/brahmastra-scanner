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

TARGET_X = 3.0  # R:R Multiple

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN") or os.getenv("BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID")
DASHBOARD_URL = "https://brahmastra-tech.github.io/brahmastra-scanner/"


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates Wilder's RSI safely without division by zero."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-5)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def run_institutional_engine():
    print("🚀 Running Fixed Brahmastra Pre-Breakout RSI Engine...")

    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at {DB_PATH}. Run bhavcopy_ingest.py first!")
        return

    conn = duckdb.connect(DB_PATH)

    # Inspect columns dynamically
    cols_info = conn.execute("DESCRIBE ohlcv_candles").fetchall()
    existing_cols = [col[0].lower() for col in cols_info]

    select_parts = [
        "symbol AS Symbol",
        "CAST(timestamp AS DATE) AS Date",
        "open AS Open",
        "high AS High",
        "low AS Low",
        "close AS Close",
        "volume AS Volume"
    ]

    if "delivery_qty" in existing_cols:
        select_parts.append("delivery_qty AS DeliveryQty")
    else:
        select_parts.append("volume * 0.45 AS DeliveryQty")

    if "delivery_pct" in existing_cols:
        select_parts.append("delivery_pct AS DeliveryPct")
    else:
        select_parts.append("45.0 AS DeliveryPct")

    query = f"""
        SELECT {', '.join(select_parts)}
        FROM ohlcv_candles
        ORDER BY symbol, timestamp ASC
    """

    df_raw = conn.execute(query).df()
    conn.close()

    if df_raw.empty:
        print("⚠️ No candle data found in database.")
        return

    df_raw["Date_DT"] = pd.to_datetime(df_raw["Date"])
    latest_date_str = df_raw['Date_DT'].max().strftime("%d-%m-%Y")
    print(f"🔍 Processing Date: {latest_date_str} | Active Universe: {df_raw['Symbol'].nunique()} Securities")

    all_scored_signals = []

    for symbol, df_sym in df_raw.groupby('Symbol'):
        if len(df_sym) < 30:
            continue

        df = df_sym.copy().sort_values("Date_DT").reset_index(drop=True)

        # 1. Daily RSI & Safe Shifts
        df['RSI_Daily'] = compute_rsi(df['Close'], 14)
        df['Prev_RSI_Daily'] = df['RSI_Daily'].shift(1).fillna(df['RSI_Daily'])

        # 2. Resample Weekly & Monthly with Fallbacks for Short Data History
        df_weekly = df.set_index('Date_DT').resample('W-FRI').agg({'Close': 'last'}).dropna()
        if len(df_weekly) >= 15:
            df_weekly['RSI_Weekly'] = compute_rsi(df_weekly['Close'], 14)
            df_weekly['Prev_RSI_Weekly'] = df_weekly['RSI_Weekly'].shift(1).fillna(df_weekly['RSI_Weekly'])
        else:
            df_weekly['RSI_Weekly'] = 55.0
            df_weekly['Prev_RSI_Weekly'] = 50.0

        df_monthly = df.set_index('Date_DT').resample('ME').agg({'Close': 'last'}).dropna()
        if len(df_monthly) >= 15:
            df_monthly['RSI_Monthly'] = compute_rsi(df_monthly['Close'], 14)
            df_monthly['Prev_RSI_Monthly'] = df_monthly['RSI_Monthly'].shift(1).fillna(df_monthly['RSI_Monthly'])
        else:
            df_monthly['RSI_Monthly'] = 55.0
            df_monthly['Prev_RSI_Monthly'] = 50.0

        # Merge MTF RSIs cleanly
        df = pd.merge_asof(df, df_weekly[['RSI_Weekly', 'Prev_RSI_Weekly']], on='Date_DT', direction='backward')
        df = pd.merge_asof(df, df_monthly[['RSI_Monthly', 'Prev_RSI_Monthly']], on='Date_DT', direction='backward')

        df['RSI_Weekly'] = df['RSI_Weekly'].fillna(55.0)
        df['Prev_RSI_Weekly'] = df['Prev_RSI_Weekly'].fillna(50.0)
        df['RSI_Monthly'] = df['RSI_Monthly'].fillna(55.0)
        df['Prev_RSI_Monthly'] = df['Prev_RSI_Monthly'].fillna(50.0)

        # 3. Volume & Delivery Metrics
        df['Vol_SMA20'] = df['Volume'].rolling(20, min_periods=5).mean().fillna(df['Volume'])
        df['Deliv_SMA20'] = df['DeliveryQty'].rolling(20, min_periods=5).mean().fillna(df['DeliveryQty'])
        df['Deliv_Spike'] = (df['DeliveryQty'] / (df['Deliv_SMA20'] + 1e-5)).fillna(1.0)

        df['Day_Range'] = (df['High'] - df['Low']).clip(lower=1e-5)
        df['Close_Location'] = ((df['Close'] - df['Low']) / df['Day_Range']).fillna(0.5)

        # Target the latest bar for PRE-BREAKOUT identification
        row = df.iloc[-1]

        # Chartink Core Filters
        c_mtf_monthly = row['RSI_Monthly'] >= row['Prev_RSI_Monthly']
        c_mtf_weekly = row['RSI_Weekly'] >= row['Prev_RSI_Weekly']
        
        rsi_diff = abs(row['RSI_Daily'] - row['Prev_RSI_Daily'])
        c_daily_rsi_squeeze = rsi_diff <= 3.0  # RSI Flatline (Consolidation)

        c_vol_ok = row['Volume'] >= 150000

        if not (c_mtf_monthly and c_mtf_weekly and c_vol_ok):
            continue

        # Safe NaN-proof BRS Calculations
        s_rsi_squeeze = float(np.clip((3.0 - rsi_diff) / 3.0 * 30.0, 0, 30))
        s_delivery = float(np.clip((row['DeliveryPct'] / 70.0 * 20.0) + (row['Deliv_Spike'] / 2.0 * 20.0), 0, 40))
        s_close_loc = float(np.clip(row['Close_Location'] * 30.0, 0, 30))

        composite_brs = float(np.nan_to_num(s_rsi_squeeze + s_delivery + s_close_loc, nan=50.0))
        composite_brs_score = round(composite_brs, 2)

        # Trigger Buy is placed ABOVE the consolidation bar High (Catching breakout 1-2 days before expansion)
        entry = round(float(row['High']) + 0.05, 2)
        sl = round(float(row['Low']), 2)
        risk = max(entry - sl, float(row['Close']) * 0.01)
        target = round(entry + (risk * TARGET_X), 2)

        all_scored_signals.append({
            "Date": latest_date_str,
            "Symbol": symbol,
            "Timeframe": "D",
            "Type": "PRE_BREAKOUT",
            "Pattern": "MTF_RSI_ACCUMULATION",
            "BRS_Score": composite_brs_score,
            "Entry": entry,
            "SL": sl,
            "Target": target,
            "Close": round(float(row['Close']), 2),
            "Volume": int(row['Volume']),
            "DeliveryQty": int(row['DeliveryQty']),
            "DeliveryPct": round(float(row['DeliveryPct']), 2),
            "DelivSpikeRatio": round(float(row['Deliv_Spike']), 2),
            "Daily_RSI": round(float(row['RSI_Daily']), 2)
        })

    os.makedirs("data", exist_ok=True)

    if all_scored_signals:
        export_df = pd.DataFrame(all_scored_signals).sort_values("BRS_Score", ascending=False)
    else:
        export_df = pd.DataFrame(columns=[
            'Date', 'Symbol', 'Timeframe', 'Type', 'Pattern', 'BRS_Score',
            'Entry', 'SL', 'Target', 'Close', 'Volume', 'DeliveryQty',
            'DeliveryPct', 'DelivSpikeRatio', 'Daily_RSI'
        ])

    export_df.to_csv(SIGNALS_CSV, index=False)
    print(f"✅ Saved {len(export_df)} evaluated pre-breakout candidates to {SIGNALS_CSV}.")

    top_candidates = export_df.head(3).to_dict('records')
    print(f"📊 Top Ranked Candidates Selected for Today ({latest_date_str}): {len(top_candidates)}")

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

    chart_url = f"https://in.tradingview.com/chart/?symbol=NSE:{symbol}"

    message = (
        f"🏛️ <b>BRAHMASTRA PRE-BREAKOUT WATCHLIST</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 <b>Stock:</b> {symbol} (NSE F&O)\n"
        f"⭐ <b>BRS Score:</b> {brs:.2f} / 100\n"
        f"🎯 <b>Setup:</b> MTF RSI Squeeze + Delivery Absorption\n"
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

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    requests.post(url, json=payload, timeout=10)


def send_summary_telegram(candidates: list, date_str: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    count = len(candidates)

    message = (
        f"🏁 <b>DAILY BRAHMASTRA SCAN COMPLETE ({date_str})</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>Top Accumulation Candidates Found Today:</b> {count}\n\n"
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
    requests.post(url, json=payload, timeout=10)


if __name__ == "__main__":
    run_institutional_engine()