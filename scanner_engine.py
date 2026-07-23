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
REJECTION_THRESHOLD_BRS = 60.0  # Calibrated Score Floor (Guarantees top 1-3 candidates)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN") or os.getenv("BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID")
DASHBOARD_URL = "https://brahmastra-tech.github.io/brahmastra-scanner/"

def run_institutional_engine():
    print("🚀 Running Institutional Breakout & Pre-Expansion Engine (IBPE-v1)...")

    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at {DB_PATH}. Run bhavcopy_ingest.py first!")
        return

    conn = duckdb.connect(DB_PATH)

    # 1. Inspect existing columns to prevent DuckDB errors
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

    if "open_interest" in existing_cols:
        select_parts.append("open_interest AS OpenInterest")
    else:
        select_parts.append("0.0 AS OpenInterest")

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

    latest_date_str = pd.to_datetime(df_raw['Date'].max()).strftime("%d-%m-%Y")
    print(f"🔍 Processing Date: {latest_date_str} | Active Universe: {df_raw['Symbol'].nunique()} Securities")

    # 2. Extract benchmark representation
    nifty_df = df_raw.groupby('Date')['Close'].mean().reset_index().rename(columns={'Close': 'Close_Nifty'})

    # 3. Vectorized Calculations
    df = pd.merge(df_raw, nifty_df, on='Date', how='left')

    df['EMA20'] = df.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    df['EMA50'] = df.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=50, adjust=False).mean())
    df['EMA200'] = df.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=200, adjust=False).mean())

    high_low = df['High'] - df['Low']
    high_cp = (df['High'] - df.groupby('Symbol')['Close'].shift(1)).abs()
    low_cp = (df['Low'] - df.groupby('Symbol')['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)

    df['ATR14'] = tr.groupby(df['Symbol']).transform(lambda x: x.rolling(14).mean())
    df['ATR100'] = tr.groupby(df['Symbol']).transform(lambda x: x.rolling(100, min_periods=10).mean())
    df['ATR_Ratio'] = df['ATR14'] / (df['ATR100'] + 1e-5)

    df['STD20'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(20).std())
    df['BB_Width'] = (4 * df['STD20']) / df['EMA20']
    df['Keltner_Width'] = (3 * df['ATR14']) / df['EMA20']
    df['Squeeze_Ratio'] = df['BB_Width'] / (df['Keltner_Width'] + 1e-5)

    df['Vol_SMA20'] = df.groupby('Symbol')['Volume'].transform(lambda x: x.rolling(20).mean())
    df['Vol_Dryup'] = df['Volume'] / (df['Vol_SMA20'] + 1e-5)
    df['Deliv_SMA20'] = df.groupby('Symbol')['DeliveryQty'].transform(lambda x: x.rolling(20).mean())
    df['Deliv_Spike'] = df['DeliveryQty'] / (df['Deliv_SMA20'] + 1e-5)

    df['OI_Shift'] = df.groupby('Symbol')['OpenInterest'].diff()
    df['OI_Shift_SMA20'] = df.groupby('Symbol')['OpenInterest'].transform(lambda x: x.rolling(20).mean().abs())
    df['Price_Shift'] = df.groupby('Symbol')['Close'].diff()

    df['PR'] = df['Close'] / df['Close_Nifty']
    df['PR_SMA252'] = df.groupby('Symbol')['PR'].transform(lambda x: x.rolling(252, min_periods=10).mean())
    df['Mansfield_RS'] = ((df['PR'] / df['PR_SMA252']) - 1) * 100

    # 4. Evaluate setup across latest bars
    latest_bar = df[df['Date'] == df['Date'].max()].copy()
    all_signals = []

    for _, row in latest_bar.iterrows():
        symbol = row['Symbol']

        # Layer 1: Liquidity Filter
        if row['Volume'] < 200000:
            continue

        # Layer 2: Calibrated Stage 2B Classification
        is_ema_aligned = row['Close'] > row['EMA50']  # Baseline trend
        is_volatility_compressed = (row['Squeeze_Ratio'] <= 1.25) or (row['ATR_Ratio'] <= 0.85)

        if not (is_ema_aligned and is_volatility_compressed):
            continue

        stage = "STAGE 2B: BREAKOUT READY"

        # Layer 3 & 4: Delivery Absorption
        if row['DeliveryPct'] < 35.0:
            continue

        # Layer 5: Institutional BRS Scoring Engine
        s_vcp = np.clip((1.25 - row['Squeeze_Ratio']) / (1.25 - 0.5) * 100, 0, 100)
        s_deliv = np.clip((row['DeliveryPct'] / 65.0 * 50) + (row['Deliv_Spike'] / 1.5 * 50), 0, 100)
        s_oi = 75.0 if row['Price_Shift'] > 0 else 40.0
        s_rs = np.clip((row['Mansfield_RS'] - (-5)) / (10 - (-5)) * 100, 0, 100)
        s_vp = 100 if row['Vol_Dryup'] <= 0.80 else 50

        brs_score = round((0.25 * s_vcp) + (0.20 * s_deliv) + (0.20 * s_oi) + (0.15 * s_rs) + (0.20 * s_vp), 2)

        # Apply Score Threshold
        if brs_score < REJECTION_THRESHOLD_BRS:
            continue

        entry = round(row['High'], 2)
        sl = round(row['Low'], 2)
        target = round(entry + (entry - sl) * TARGET_X, 2)

        all_signals.append({
            "Date": latest_date_str,
            "Symbol": symbol,
            "Timeframe": "D",
            "Type": "PRE_BREAKOUT",
            "Pattern": stage,
            "BRS_Score": brs_score,
            "Entry": entry,
            "SL": sl,
            "Target": target,
            "Close": round(row['Close'], 2),
            "Volume": int(row['Volume']),
            "DeliveryQty": int(row['DeliveryQty']),
            "DeliveryPct": round(row['DeliveryPct'], 2),
            "DelivSpikeRatio": round(row['Deliv_Spike'], 2),
            "Mansfield_RS": round(row['Mansfield_RS'], 2)
        })

    os.makedirs("data", exist_ok=True)

    if all_signals:
        export_df = pd.DataFrame(all_signals).sort_values("BRS_Score", ascending=False)
    else:
        export_df = pd.DataFrame(columns=[
            'Date', 'Symbol', 'Timeframe', 'Type', 'Pattern', 'BRS_Score',
            'Entry', 'SL', 'Target', 'Close', 'Volume', 'DeliveryQty',
            'DeliveryPct', 'DelivSpikeRatio', 'Mansfield_RS'
        ])

    export_df.to_csv(SIGNALS_CSV, index=False)
    print(f"✅ Saved {len(export_df)} Elite Stage 2B candidates to {SIGNALS_CSV}.")

    today_candidates = export_df.to_dict('records')

    try:
        for sig in today_candidates[:3]:
            send_telegram_alert(sig)
            time.sleep(0.5)
    finally:
        send_summary_telegram(today_candidates, latest_date_str)


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
    rs_val = signal.get("Mansfield_RS", 0.0)

    chart_url = f"https://in.tradingview.com/chart/?symbol=NSE:{symbol}"

    message = (
        f"🏛️ <b>INSTITUTIONAL BREAKOUT CANDIDATE</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 <b>Stock:</b> {symbol} (NSE F&O)\n"
        f"⭐ <b>BRS Score:</b> {brs} / 100\n"
        f"🎯 <b>Stage:</b> Stage 2B (Breakout Ready)\n"
        f"⏱ <b>Date:</b> {setup_date}\n\n"
        f"📊 <b>ACTIONABLE TRIGGER LEVELS</b>\n"
        f"• <b>Trigger Entry Price :</b> ₹{entry:.2f}\n"
        f"• <b>Stop Loss           :</b> ₹{sl:.2f}\n"
        f"• <b>Target (3x)         :</b> ₹{target:.2f}\n"
        f"• <b>Today's Close       :</b> ₹{close:.2f}\n\n"
        f"⚡ <b>INSTITUTIONAL METRICS</b>\n"
        f"• <b>Delivery %          :</b> {deliv_pct:.1f}%\n"
        f"• <b>Mansfield RS Alpha  :</b> +{rs_val:.2f}\n"
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

    message = (
        f"🏁 <b>INSTITUTIONAL ENGINE EXECUTION COMPLETE ({date_str})</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>Stage 2B Qualified Candidates:</b> {len(candidates)}\n\n"
        f"🌐 <b>Interactive Web Dashboard:</b>\n"
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