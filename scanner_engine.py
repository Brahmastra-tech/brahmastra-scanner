import os
import duckdb
import polars as pl
import requests
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
DB_PATH = "data/candles.duckdb"
SIGNALS_CSV = "data/signals.csv"
TARGET_X = 3.0  # Target = SL * 3.0

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def send_telegram_alert(signal: dict):
    """Sends a formatted, interactive signal alert to Telegram for TODAY'S signals only."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram credentials not found. Skipping live alert dispatch.")
        return

    symbol = signal["symbol"]
    sig_type = signal["type"]
    date_str = str(signal["date"])
    entry = signal["entry"]
    sl = signal["sl"]
    target = signal["target"]
    close = signal["close"]
    vol_ratio = signal["vol_ratio"]

    emoji = "🚀" if sig_type == "PRE_BREAKOUT" else "📉"
    
    message = (
        f"{emoji} <b>BRAHMASTRA SIGNAL DETECTED</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 <b>Stock:</b> {symbol} (NSE F&O)\n"
        f"🎯 <b>Pattern:</b> {sig_type}\n"
        f"⏱ <b>Timeframe:</b> Daily (1D) | <b>Date:</b> {date_str}\n\n"
        f"📊 <b>TRADE LEVELS</b>\n"
        f"• <b>Entry Price :</b> ₹{entry:.2f}\n"
        f"• <b>Stop Loss   :</b> ₹{sl:.2f}\n"
        f"• <b>Target (3x) :</b> ₹{target:.2f}\n"
        f"• <b>Close Price :</b> ₹{close:.2f}\n\n"
        f"⚡ <b>VOLATILITY METRICS</b>\n"
        f"• <b>Volume Ratio :</b> {vol_ratio:.2f}x (vs 10 MA)\n"
        f"• <b>Compression  :</b> {signal['compression']:.4f}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🔗 <b>Interactive Links:</b>\n"
        f"📈 <a href='https://in.tradingview.com/chart/?symbol=NSE:{symbol}'>Open TradingView Chart</a>"
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
            print(f"✓ Telegram alert sent for {symbol} ({sig_type})")
        else:
            print(f"❌ Failed to send Telegram alert ({res.status_code}): {res.text}")
    except Exception as e:
        print(f"❌ Telegram exception: {e}")


def run_scanner():
    """Executes pre-breakout/breakdown strategy logic across DuckDB candles."""
    if not os.path.exists(DB_PATH):
        print("❌ Database file not found!")
        return

    conn = duckdb.connect(DB_PATH)
    
    # 1. Fetch candles from DuckDB
    df = conn.execute("""
        SELECT symbol, timeframe, CAST(timestamp AS DATE) AS date, open, high, low, close, volume
        FROM ohlcv_candles
        ORDER BY symbol, timestamp ASC
    """).pl()

    if df.is_empty():
        print("⚠️ No candles available in DuckDB.")
        return

    latest_date = df.select(pl.max("date")).item()
    print(f"🔍 Running Scanner Engine (Latest Date in DB: {latest_date})...")

    # 2. Compute 5-day rolling range & 10-day volume average
    df_metrics = df.with_columns([
        pl.col("high").rolling_max(window_size=5).over("symbol").alias("High5"),
        pl.col("low").rolling_min(window_size=5).over("symbol").alias("Low5"),
        pl.col("volume").rolling_mean(window_size=10).over("symbol").alias("AvgVol10"),
        pl.col("close").shift(1).over("symbol").alias("PrevClose")
    ]).with_columns([
        ((pl.col("High5") - pl.col("Low5")) / pl.col("PrevClose")).alias("Compression")
    ])

    # 3. Shift metrics by 1 day to evaluate setup
    df_shifted = df_metrics.with_columns([
        pl.col("Compression").shift(1).over("symbol").alias("PrevCompression"),
        pl.col("volume").shift(1).over("symbol").alias("PrevVolume"),
        pl.col("AvgVol10").shift(1).over("symbol").alias("PrevAvgVol10"),
        pl.col("High5").shift(1).over("symbol").alias("PrevHigh5"),
        pl.col("Low5").shift(1).over("symbol").alias("PrevLow5")
    ])

    # 4. Filter for setup conditions (Compression <= 0.05 and dry volume)
    valid_setup = (pl.col("PrevCompression") <= 0.05) & (pl.col("PrevVolume") < pl.col("PrevAvgVol10"))

    breakouts = df_shifted.filter(valid_setup & (pl.col("close") > pl.col("PrevHigh5"))).with_columns([
        pl.lit("PRE_BREAKOUT").alias("type"),
        pl.col("PrevHigh5").alias("entry"),
        pl.col("PrevLow5").alias("sl"),
        (pl.col("PrevHigh5") + (pl.col("PrevHigh5") - pl.col("PrevLow5")) * TARGET_X).alias("target")
    ])

    breakdowns = df_shifted.filter(valid_setup & (pl.col("close") < pl.col("PrevLow5"))).with_columns([
        pl.lit("PRE_BREAKDOWN").alias("type"),
        pl.col("PrevLow5").alias("entry"),
        pl.col("PrevHigh5").alias("sl"),
        (pl.col("PrevLow5") - (pl.col("PrevHigh5") - pl.col("PrevLow5")) * TARGET_X).alias("target")
    ])

    all_signals = pl.concat([breakouts, breakdowns]).sort("date", descending=True)

    if all_signals.is_empty():
        print("ℹ️ No signals detected.")
        return

    # 5. Save COMPLETE historical signals to signals.csv for Dashboard archive
    signals_to_save = all_signals.select([
        pl.col("date").cast(pl.Utf8),
        pl.col("symbol"),
        pl.col("type"),
        pl.col("entry").round(2),
        pl.col("sl").round(2),
        pl.col("target").round(2),
        pl.col("close").round(2),
        pl.col("PrevCompression").round(4).alias("compression"),
        (pl.col("volume") / pl.col("PrevAvgVol10")).round(2).alias("vol_ratio")
    ])

    os.makedirs("data", exist_ok=True)
    signals_to_save.write_csv(SIGNALS_CSV)
    print(f"✅ Saved {len(signals_to_save)} total historical signals to {SIGNALS_CSV} for Dashboard archive.")

    # 6. Filter and dispatch TODAY'S alerts ONLY to Telegram
    today_signals = signals_to_save.filter(pl.col("date") == str(latest_date))
    
    if today_signals.is_empty():
        print(f"ℹ️ No new signals triggered for today ({latest_date}). Telegram quiet.")
    else:
        print(f"📢 Sending {len(today_signals)} Telegram alerts for today ({latest_date})...")
        for sig in today_signals.to_dicts():
            send_telegram_alert(sig)


if __name__ == "__main__":
    run_scanner()