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
TARGET_X = 3.0       # Target multiplier (3x Risk)
MAX_SL_PCT = 0.015   # 1.5% Max Stop Loss Cap (Adjustable)
EMA_PERIOD = 20
ADX_PERIOD = 14
MIN_ADX = 20.0

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DASHBOARD_URL = "https://brahmastra-tech.github.io/brahmastra-scanner/"


def send_telegram_alert(signal: dict):
    """Sends individual formatted alert to Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    symbol = signal["symbol"]
    sig_type = signal["type"]
    date_str = signal["date"]
    entry = signal["entry"]
    sl = signal["sl"]
    target = signal["target"]
    close = signal["close"]
    vol_ratio = signal["vol_ratio"]
    ema = signal["ema"]
    adx = signal["adx"]

    emoji = "🚀" if sig_type == "PRE_BREAKOUT" else "📉"
    
    message = (
        f"{emoji} <b>BRAHMASTRA SIGNAL DETECTED</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 <b>Stock:</b> {symbol} (NSE F&O)\n"
        f"🎯 <b>Pattern:</b> {sig_type}\n"
        f"⏱ <b>Timeframe:</b> Daily (1D) | <b>Date:</b> {date_str}\n\n"
        f"📊 <b>TRADE LEVELS (Tighter SL)</b>\n"
        f"• <b>Entry Price :</b> ₹{entry:.2f}\n"
        f"• <b>Stop Loss   :</b> ₹{sl:.2f} (Risk Cap)\n"
        f"• <b>Target (3x) :</b> ₹{target:.2f}\n"
        f"• <b>Close Price :</b> ₹{close:.2f}\n\n"
        f"⚡ <b>CONFIRMATION INDICATORS</b>\n"
        f"• <b>Volume Ratio :</b> {vol_ratio:.2f}x (vs 10 MA)\n"
        f"• <b>20 EMA       :</b> ₹{ema:.2f}\n"
        f"• <b>14 ADX       :</b> {adx:.2f}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 <a href='https://in.tradingview.com/chart/?symbol=NSE:{symbol}'>View TradingView Chart</a>"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    requests.post(url, json=payload, timeout=10)


def send_summary_telegram(signal_count: int, date_str: str):
    """Sends a summary message containing the Backtest / Web Dashboard link after all alerts."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    message = (
        f"🏁 <b>DAILY SCAN COMPLETE ({date_str})</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>Total High-Probability Signals:</b> {signal_count}\n\n"
        f"🌐 <b>Historical Backtest Terminal & Interactive Dashboard:</b>\n"
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


def calculate_adx_polars(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    df = df.with_columns([
        (pl.col("high") - pl.col("high").shift(1)).alias("up_move"),
        (pl.col("low").shift(1) - pl.col("low")).alias("down_move"),
        (pl.col("high") - pl.col("low")).alias("tr1"),
        (pl.col("high") - pl.col("close").shift(1)).abs().alias("tr2"),
        (pl.col("low") - pl.col("close").shift(1)).abs().alias("tr3")
    ])

    df = df.with_columns([
        pl.max_horizontal(["tr1", "tr2", "tr3"]).alias("tr"),
        pl.when((pl.col("up_move") > pl.col("down_move")) & (pl.col("up_move") > 0))
          .then(pl.col("up_move"))
          .otherwise(0.0).alias("plus_dm"),
        pl.when((pl.col("down_move") > pl.col("up_move")) & (pl.col("down_move") > 0))
          .then(pl.col("down_move"))
          .otherwise(0.0).alias("minus_dm")
    ])

    df = df.with_columns([
        pl.col("tr").rolling_mean(window_size=period).over("symbol").alias("atr"),
        pl.col("plus_dm").rolling_mean(window_size=period).over("symbol").alias("smooth_plus_dm"),
        pl.col("minus_dm").rolling_mean(window_size=period).over("symbol").alias("smooth_minus_dm")
    ])

    df = df.with_columns([
        (100 * pl.col("smooth_plus_dm") / pl.col("atr")).alias("plus_di"),
        (100 * pl.col("smooth_minus_dm") / pl.col("atr")).alias("minus_di")
    ])

    df = df.with_columns([
        (100 * (pl.col("plus_di") - pl.col("minus_di")).abs() / (pl.col("plus_di") + pl.col("minus_di"))).alias("dx")
    ])

    return df.with_columns([
        pl.col("dx").rolling_mean(window_size=period).over("symbol").alias("adx")
    ])


def run_scanner():
    if not os.path.exists(DB_PATH):
        print("❌ Database file not found!")
        return

    conn = duckdb.connect(DB_PATH)
    
    df = conn.execute("""
        SELECT symbol, timeframe, timestamp AS datetime_val, open, high, low, close, volume
        FROM ohlcv_candles
        ORDER BY symbol, timestamp ASC
    """).pl()

    if df.is_empty():
        return

    latest_date_dt = df.select(pl.max("datetime_val")).item()
    latest_date_str = latest_date_dt.strftime("%d-%m-%Y")

    df = df.with_columns([
        pl.col("close").ewm_mean(span=EMA_PERIOD, adjust=False).over("symbol").alias("ema20")
    ])
    df = calculate_adx_polars(df, period=ADX_PERIOD)

    df_metrics = df.with_columns([
        pl.col("high").rolling_max(window_size=5).over("symbol").alias("High5"),
        pl.col("low").rolling_min(window_size=5).over("symbol").alias("Low5"),
        pl.col("volume").rolling_mean(window_size=10).over("symbol").alias("AvgVol10"),
        pl.col("close").shift(1).over("symbol").alias("PrevClose")
    ]).with_columns([
        ((pl.col("High5") - pl.col("Low5")) / pl.col("PrevClose")).alias("Compression")
    ])

    df_shifted = df_metrics.with_columns([
        pl.col("Compression").shift(1).over("symbol").alias("PrevCompression"),
        pl.col("volume").shift(1).over("symbol").alias("PrevVolume"),
        pl.col("AvgVol10").shift(1).over("symbol").alias("PrevAvgVol10"),
        pl.col("High5").shift(1).over("symbol").alias("PrevHigh5"),
        pl.col("Low5").shift(1).over("symbol").alias("PrevLow5")
    ])

    valid_setup = (pl.col("PrevCompression") <= 0.05) & (pl.col("PrevVolume") < pl.col("PrevAvgVol10"))

    # Tighter SL Logic Applied (capped at MAX_SL_PCT or PrevLow5, whichever is tighter)
    breakouts = df_shifted.filter(
        valid_setup & 
        (pl.col("close") > pl.col("PrevHigh5")) & 
        (pl.col("close") > pl.col("ema20")) & 
        (pl.col("adx") >= MIN_ADX)
    ).with_columns([
        pl.lit("PRE_BREAKOUT").alias("type"),
        pl.col("PrevHigh5").alias("entry"),
        pl.max_horizontal([pl.col("PrevLow5"), pl.col("PrevHigh5") * (1 - MAX_SL_PCT)]).alias("sl")
    ]).with_columns([
        (pl.col("entry") + (pl.col("entry") - pl.col("sl")) * TARGET_X).alias("target")
    ])

    breakdowns = df_shifted.filter(
        valid_setup & 
        (pl.col("close") < pl.col("PrevLow5")) & 
        (pl.col("close") < pl.col("ema20")) & 
        (pl.col("adx") >= MIN_ADX)
    ).with_columns([
        pl.lit("PRE_BREAKDOWN").alias("type"),
        pl.col("PrevLow5").alias("entry"),
        pl.min_horizontal([pl.col("PrevHigh5"), pl.col("PrevLow5") * (1 + MAX_SL_PCT)]).alias("sl")
    ]).with_columns([
        (pl.col("entry") - (pl.col("sl") - pl.col("entry")) * TARGET_X).alias("target")
    ])

    all_signals = pl.concat([breakouts, breakdowns]).sort(["symbol", "datetime_val"])

    if all_signals.is_empty():
        return

    # First instance filter
    all_signals = all_signals.with_columns([
        pl.col("datetime_val").diff().over(["symbol", "type"]).dt.total_days().alias("days_since_last_signal")
    ])
    
    first_instance_signals = all_signals.filter(
        (pl.col("days_since_last_signal").is_null()) | (pl.col("days_since_last_signal") > 1)
    ).sort("datetime_val", descending=True)

    formatted_signals = first_instance_signals.select([
        pl.col("datetime_val").dt.strftime("%d-%m-%Y").alias("date"),
        pl.col("symbol"),
        pl.col("type"),
        pl.col("entry").round(2),
        pl.col("sl").round(2),
        pl.col("target").round(2),
        pl.col("close").round(2),
        pl.col("ema20").round(2).alias("ema"),
        pl.col("adx").round(2).alias("adx"),
        pl.col("PrevCompression").round(4).alias("compression"),
        (pl.col("volume") / pl.col("PrevAvgVol10")).round(2).alias("vol_ratio")
    ])

    os.makedirs("data", exist_ok=True)
    formatted_signals.write_csv(SIGNALS_CSV)

    # Dispatch alerts
    today_signals = formatted_signals.filter(pl.col("date") == latest_date_str)
    
    if not today_signals.is_empty():
        for sig in today_signals.to_dicts():
            send_telegram_alert(sig)
        
        # SEND SUMMARY WEBLINK AT THE VERY END
        send_summary_telegram(len(today_signals), latest_date_str)


if __name__ == "__main__":
    run_scanner()