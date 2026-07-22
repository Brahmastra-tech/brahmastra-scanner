import os
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

TARGET_X = 3.0

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN") or os.getenv("BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID")
DASHBOARD_URL = "https://brahmastra-tech.github.io/brahmastra-scanner/"


def calc_adx(df, period=14):
    """Calculates ADX using Wilder's directional movement principles."""
    if df is None or len(df) < period + 2:
        return pd.Series([np.nan] * len(df), index=df.index)

    high = df['High']
    low = df['Low']
    close = df['Close']

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()

    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(period).mean()
    return adx


def calc_rsi(series, period=14):
    """Standard Wilder's RSI calculation."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_weekly_rsi(df):
    """Resamples daily data to weekly, calculates 14 RSI, and merges back."""
    df_temp = df.copy()
    df_temp['Date_DT'] = pd.to_datetime(df_temp['Date'])
    df_temp.set_index('Date_DT', inplace=True)

    # Resample to weekly OHLCV
    weekly = df_temp.resample('W-FRI').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    weekly['Weekly_RSI'] = calc_rsi(weekly['Close'], period=14)

    # Merge weekly RSI back into daily candles using backward fill
    df_temp = pd.merge_asof(
        df_temp.sort_values('Date_DT'),
        weekly[['Weekly_RSI']].sort_index(),
        left_index=True,
        right_index=True,
        direction='backward'
    )

    return df_temp.reset_index(drop=True)


def scan_symbol_exact(symbol, df_sym):
    alerts = []

    if df_sym.empty or len(df_sym) < 30:
        return alerts

    df = df_sym.copy().sort_values("Date").reset_index(drop=True)

    # Compute Weekly RSI
    try:
        df = add_weekly_rsi(df)
    except Exception:
        df['Weekly_RSI'] = np.nan

    # -------------------------
    # BOBD BASE CALCULATIONS
    # -------------------------

    df["High5"] = df["High"].shift(1).rolling(5).max()
    df["Low5"] = df["Low"].shift(1).rolling(5).min()

    df["PrevClose"] = df["Close"].shift(1)

    df["AvgVol10"] = df["Volume"].shift(1).rolling(10).mean()

    df["Compression"] = (
        (df["High5"] - df["Low5"])
        / df["PrevClose"]
    )

    df["EMA20"] = (
        df["Close"]
        .ewm(span=20, adjust=False)
        .mean()
    )

    df["ADX"] = calc_adx(df, 14)

    seen = set()

    for i in range(20, len(df)):

        row = df.iloc[i]

        if pd.isna(row["Compression"]):
            continue

        compression_ok = (
            row["Compression"] <= 0.05
        )

        volume_ok = (
            row["Volume"] < row["AvgVol10"]
        )

        adx_ok = (
            pd.notna(row["ADX"])
            and row["ADX"] >= 20
        )

        if not (
            compression_ok
            and volume_ok
            and adx_ok
        ):
            continue

        weekly_rsi = float(row["Weekly_RSI"]) if pd.notna(row.get("Weekly_RSI")) else 50.0

        # -----------------------------
        # BULLISH PRE-BREAKOUT
        # -----------------------------

        if row["Close"] > row["EMA20"]:

            # Filter: Weekly RSI MUST be > 50
            if weekly_rsi <= 50.0:
                continue

            key = (symbol, "PRE_BREAKOUT")

            if key not in seen:

                seen.add(key)

                entry = round(row["High5"], 2)
                sl = round(row["Low5"], 2)
                target = round(
                    entry + (entry - sl) * TARGET_X,
                    2
                )

                ema_dist = round(
                    abs(row["Close"] - row["EMA20"])
                    / row["EMA20"] * 100,
                    2
                )

                alerts.append({

                    "Date":
                        pd.to_datetime(row["Date"]).strftime("%d-%m-%Y"),

                    "Symbol":
                        symbol,

                    "Timeframe":
                        "D",

                    "Type":
                        "PRE_BREAKOUT",

                    "Entry":
                        entry,

                    "SL":
                        sl,

                    "Target":
                        target,

                    "Close":
                        round(row["Close"], 2),

                    "Compression":
                        round(row["Compression"], 4),

                    "AvgVol10":
                        int(row["AvgVol10"]),

                    "Volume":
                        int(row["Volume"]),

                    "EMA":
                        round(row["EMA20"], 2),

                    "EMA_Dist%":
                        ema_dist,

                    "ADX":
                        round(row["ADX"], 2),

                    "Weekly_RSI":
                        round(weekly_rsi, 2)

                })

        # -----------------------------
        # BEARISH PRE-BREAKDOWN
        # -----------------------------

        elif row["Close"] < row["EMA20"]:

            # Filter: Weekly RSI MUST be < 45
            if weekly_rsi >= 45.0:
                continue

            key = (symbol, "PRE_BREAKDOWN")

            if key not in seen:

                seen.add(key)

                entry = round(row["Low5"], 2)
                sl = round(row["High5"], 2)

                target = round(
                    entry - (sl - entry) * TARGET_X,
                    2
                )

                ema_dist = round(
                    abs(row["Close"] - row["EMA20"])
                    / row["EMA20"] * 100,
                    2
                )

                alerts.append({

                    "Date":
                        pd.to_datetime(row["Date"]).strftime("%d-%m-%Y"),

                    "Symbol":
                        symbol,

                    "Timeframe":
                        "D",

                    "Type":
                        "PRE_BREAKDOWN",

                    "Entry":
                        entry,

                    "SL":
                        sl,

                    "Target":
                        target,

                    "Close":
                        round(row["Close"], 2),

                    "Compression":
                        round(row["Compression"], 4),

                    "AvgVol10":
                        int(row["AvgVol10"]),

                    "Volume":
                        int(row["Volume"]),

                    "EMA":
                        round(row["EMA20"], 2),

                    "EMA_Dist%":
                        ema_dist,

                    "ADX":
                        round(row["ADX"], 2),

                    "Weekly_RSI":
                        round(weekly_rsi, 2)

                })

    return alerts


def send_telegram_alert(signal: dict):
    """Sends individual alert payload to Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram credentials missing. Skipping individual alert.")
        return

    symbol = signal.get("Symbol") or signal.get("symbol")
    sig_type = signal.get("Type") or signal.get("type")
    setup_date = signal.get("Date") or signal.get("date")
    entry = signal.get("Entry") or signal.get("entry")
    sl = signal.get("SL") or signal.get("sl")
    target = signal.get("Target") or signal.get("target")
    close = signal.get("Close") or signal.get("close")
    avgvol = signal.get("AvgVol10") or signal.get("avgvol10", 1)
    volume = signal.get("Volume") or signal.get("volume", 0)
    vol_ratio = round(volume / avgvol, 2) if avgvol > 0 else 1.0
    ema = signal.get("EMA") or signal.get("ema")
    adx = signal.get("ADX") or signal.get("adx")
    weekly_rsi = signal.get("Weekly_RSI") or signal.get("weekly_rsi", "N/A")

    emoji = "🚀" if sig_type == "PRE_BREAKOUT" else "📉"
    chart_url = f"https://in.tradingview.com/chart/?symbol=NSE:{symbol}"

    message = (
        f"{emoji} <b>BRAHMASTRA PRE-BREAKOUT WATCHLIST</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 <b>Stock:</b> {symbol} (NSE F&O)\n"
        f"🎯 <b>Pattern:</b> {sig_type}\n"
        f"⏱ <b>Setup Date:</b> {setup_date}\n\n"
        f"📊 <b>ACTIONABLE TRIGGER LEVELS</b>\n"
        f"• <b>Trigger Entry Price :</b> ₹{entry:.2f}\n"
        f"• <b>Stop Loss           :</b> ₹{sl:.2f}\n"
        f"• <b>Target (3x)         :</b> ₹{target:.2f}\n"
        f"• <b>Today's Close       :</b> ₹{close:.2f}\n\n"
        f"⚡ <b>MACRO CONVICTION METRICS</b>\n"
        f"• <b>Weekly RSI          :</b> {weekly_rsi}\n"
        f"• <b>20 EMA              :</b> ₹{ema:.2f}\n"
        f"• <b>14 ADX              :</b> {adx:.2f}\n"
        f"• <b>Volume Ratio        :</b> {vol_ratio:.2f}x\n"
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
    """Sends scan execution summary to Telegram breaking down Long & Short counts."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram credentials missing. Skipping summary dispatch.")
        return

    breakouts = [s for s in signals_today if s.get("Type") == "PRE_BREAKOUT" or s.get("type") == "PRE_BREAKOUT"]
    breakdowns = [s for s in signals_today if s.get("Type") == "PRE_BREAKDOWN" or s.get("type") == "PRE_BREAKDOWN"]
    total_count = len(signals_today)

    message = (
        f"🏁 <b>DAILY SCAN COMPLETE ({date_str})</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>Total Candidates Found Today:</b> {total_count}\n"
        f"🚀 <b>Pre-Breakout (Long | W-RSI > 50)  :</b> {len(breakouts)}\n"
        f"📉 <b>Pre-Breakdown (Short | W-RSI < 45) :</b> {len(breakdowns)}\n\n"
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
            print(f"✅ Telegram summary sent for {date_str}")
        else:
            print(f"❌ Telegram Summary API Error ({res.status_code}): {res.text}")
    except Exception as e:
        print(f"❌ Exception sending Telegram summary: {e}")


def run_scanner():
    print("🚀 Initializing Scanner Engine with Weekly RSI Filter...")
    print(f"🔑 Telegram Secret Status: Bot Token={'FOUND' if TELEGRAM_BOT_TOKEN else 'MISSING'}, Chat ID={'FOUND' if TELEGRAM_CHAT_ID else 'MISSING'}")

    if not os.path.exists(DB_PATH):
        print(f"❌ Database file not found at {DB_PATH}!")
        return

    conn = duckdb.connect(DB_PATH)

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
    all_signals = []

    for sym in symbols:
        df_sym = df_raw[df_raw['Symbol'] == sym]
        alerts = scan_symbol_exact(sym, df_sym)
        if alerts:
            all_signals.extend(alerts)

    os.makedirs("data", exist_ok=True)

    if not all_signals:
        print("ℹ️ No signals matched conditions overall.")
        pd.DataFrame(columns=['Date', 'Symbol', 'Timeframe', 'Type', 'Entry', 'SL', 'Target', 'Close', 'Compression', 'AvgVol10', 'Volume', 'EMA', 'EMA_Dist%', 'ADX', 'Weekly_RSI']).to_csv(SIGNALS_CSV, index=False)
        send_summary_telegram([], latest_date)
        return

    all_df = pd.DataFrame(all_signals)
    
    date_col = "Date" if "Date" in all_df.columns else "date"
    all_df["Date_DT"] = pd.to_datetime(all_df[date_col], format="%d-%m-%Y")

    export_df = all_df.sort_values("Date_DT", ascending=False).drop(columns=["Date_DT"])
    export_df.to_csv(SIGNALS_CSV, index=False)
    print(f"✅ Saved {len(export_df)} total unique signals to {SIGNALS_CSV}.")

    today_signals = export_df[export_df[date_col] == latest_date].to_dict('records')
    print(f"📊 Candidates for Today ({latest_date}): {len(today_signals)}")

    if today_signals:
        for sig in today_signals:
            send_telegram_alert(sig)

    send_summary_telegram(today_signals, latest_date)


if __name__ == "__main__":
    run_scanner()