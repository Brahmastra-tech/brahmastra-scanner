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
TARGET_X = 3.0

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN") or os.getenv("BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID")
DASHBOARD_URL = "https://brahmastra-tech.github.io/brahmastra-scanner/"


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-5)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def run_institutional_engine():
    print("🚀 Running Brahmastra Scanner Engine...")

    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at {DB_PATH}.")
        return

    conn = duckdb.connect(DB_PATH)
    cols_info = conn.execute("DESCRIBE ohlcv_candles").fetchall()
    existing_cols = [col[0].lower() for col in cols_info]

    select_parts = [
        "symbol AS Symbol",
        "CAST(timestamp AS DATE) AS Date",
        "open AS Open", "high AS High", "low AS Low", "close AS Close", "volume AS Volume"
    ]
    if "delivery_qty" in existing_cols: select_parts.append("delivery_qty AS DeliveryQty")
    else: select_parts.append("volume * 0.45 AS DeliveryQty")

    if "delivery_pct" in existing_cols: select_parts.append("delivery_pct AS DeliveryPct")
    else: select_parts.append("45.0 AS DeliveryPct")

    df_raw = conn.execute(f"SELECT {', '.join(select_parts)} FROM ohlcv_candles ORDER BY symbol, timestamp ASC").df()
    conn.close()

    if df_raw.empty:
        print("⚠️ Database empty.")
        return

    df_raw["Date_DT"] = pd.to_datetime(df_raw["Date"])
    latest_date_str = df_raw['Date_DT'].max().strftime("%d-%m-%Y")

    all_scored_signals = []

    for symbol, df_sym in df_raw.groupby('Symbol'):
        if len(df_sym) < 15:
            continue

        df = df_sym.copy().sort_values("Date_DT").reset_index(drop=True)
        df['DeliveryQty'] = df['DeliveryQty'].fillna(0.0)
        df['DeliveryPct'] = df['DeliveryPct'].fillna(0.0)
        df['Volume'] = df['Volume'].fillna(0.0)

        df['RSI_Daily'] = compute_rsi(df['Close'], 14)
        df['Prev_RSI_Daily'] = df['RSI_Daily'].shift(1).fillna(df['RSI_Daily'])

        df_weekly = df.set_index('Date_DT').resample('W-FRI').agg({'Close': 'last'}).dropna()
        df_weekly['RSI_Weekly'] = compute_rsi(df_weekly['Close'], 14) if len(df_weekly) >= 5 else 55.0
        df_weekly['Prev_RSI_Weekly'] = df_weekly['RSI_Weekly'].shift(1) if len(df_weekly) >= 5 else 50.0

        df_monthly = df.set_index('Date_DT').resample('ME').agg({'Close': 'last'}).dropna()
        df_monthly['RSI_Monthly'] = compute_rsi(df_monthly['Close'], 14) if len(df_monthly) >= 2 else 55.0
        df_monthly['Prev_RSI_Monthly'] = df_monthly['RSI_Monthly'].shift(1) if len(df_monthly) >= 2 else 50.0

        df = pd.merge_asof(df, df_weekly[['RSI_Weekly', 'Prev_RSI_Weekly']], on='Date_DT', direction='backward').fillna(50.0)
        df = pd.merge_asof(df, df_monthly[['RSI_Monthly', 'Prev_RSI_Monthly']], on='Date_DT', direction='backward').fillna(50.0)

        df['Day_Range'] = (df['High'] - df['Low']).clip(lower=1e-5)
        df['Close_Location'] = ((df['Close'] - df['Low']) / df['Day_Range']).fillna(0.5)

        row = df.iloc[-1]

        c_mtf_monthly = row['RSI_Monthly'] >= row['Prev_RSI_Monthly']
        c_mtf_weekly = row['RSI_Weekly'] >= row['Prev_RSI_Weekly']
        rsi_diff = abs(row['RSI_Daily'] - row['Prev_RSI_Daily'])
        c_daily_rsi_squeeze = rsi_diff <= 3.5

        if not (c_mtf_monthly and c_mtf_weekly and c_daily_rsi_squeeze):
            continue

        deliv_score = float(np.clip((row['DeliveryPct'] / 70.0 * 50.0), 10, 50))
        close_score = float(np.clip(row['Close_Location'] * 50.0, 10, 50))
        composite_brs_score = round(deliv_score + close_score, 2)

        entry = round(float(row['High']) + 0.05, 2)
        sl = round(float(row['Low']), 2)
        risk = max(entry - sl, float(row['Close']) * 0.01)
        target = round(entry + (risk * TARGET_X), 2)

        deliv_pct_val = round(float(np.nan_to_num(row['DeliveryPct'], nan=0.0)), 2)
        rsi_daily_val = round(float(np.nan_to_num(row['RSI_Daily'], nan=50.0)), 2)

        all_scored_signals.append({
            "Date": latest_date_str,
            "Symbol": symbol,
            "Timeframe": "D",
            "Type": "PRE_BREAKOUT",
            "Pattern": "PRE_BREAKOUT",
            "BRS_Score": composite_brs_score,
            "Entry": entry,
            "SL": sl,
            "Target": target,
            "Close": round(float(row['Close']), 2),
            "Volume": int(np.nan_to_num(row['Volume'], nan=0)),
            "ema": deliv_pct_val,  # Map Delivery % directly to UI column
            "adx": rsi_daily_val,  # Map Daily RSI directly to UI column
            "DeliveryQty": int(np.nan_to_num(row['DeliveryQty'], nan=0)),
            "DeliveryPct": deliv_pct_val,
            "DelivSpikeRatio": 1.0,
            "Daily_RSI": rsi_daily_val
        })

    os.makedirs("data", exist_ok=True)
    today_df = pd.DataFrame(all_scored_signals).sort_values("BRS_Score", ascending=False).head(3) if all_scored_signals else pd.DataFrame()

    if os.path.exists(SIGNALS_CSV):
        try:
            existing_df = pd.read_csv(SIGNALS_CSV)
            existing_df = existing_df[existing_df['Date'] != latest_date_str]
            combined_df = pd.concat([today_df, existing_df], ignore_index=True)
        except Exception:
            combined_df = today_df
    else:
        combined_df = today_df

    if not combined_df.empty:
        combined_df['Date_DT'] = pd.to_datetime(combined_df['Date'], format="%d-%m-%Y")
        combined_df = combined_df.sort_values(by=['Date_DT', 'BRS_Score'], ascending=[False, False])
        recent_dates = combined_df['Date_DT'].unique()[:30]
        final_export_df = combined_df[combined_df['Date_DT'].isin(recent_dates)].drop(columns=['Date_DT'])
    else:
        final_export_df = combined_df

    final_export_df.to_csv(SIGNALS_CSV, index=False)
    print(f"✅ Saved Top candidates for {latest_date_str}.")

    top_candidates = today_df.to_dict('records') if not today_df.empty else []
    try:
        for sig in top_candidates:
            send_telegram_alert(sig)
            time.sleep(0.5)
    finally:
        send_summary_telegram(top_candidates, latest_date_str)


def send_telegram_alert(signal: dict):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
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
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML", "disable_web_page_preview": True}, timeout=10)


def send_summary_telegram(candidates: list, date_str: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    message = (
        f"🏁 <b>DAILY BRAHMASTRA SCAN COMPLETE ({date_str})</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>Top Accumulation Candidates Found Today:</b> {len(candidates)}\n\n"
        f"🌐 <b>Interactive Web Dashboard:</b>\n"
        f"👉 <a href='{DASHBOARD_URL}'>{DASHBOARD_URL}</a>\n"
        f"━━━━━━━━━━━━━━━━━━━━"
    )
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}, timeout=10)


if __name__ == "__main__":
    run_institutional_engine()