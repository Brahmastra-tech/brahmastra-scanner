import os
import duckdb
import pandas as pd
import numpy as np
import requests
from datetime import datetime

# ==========================================
# CONFIGURATION (HIGH-QUALITY PRE-BREAKOUT)
# ==========================================
DB_PATH = "data/candles.duckdb"
SIGNALS_CSV = "data/signals.csv"

COMPRESSION_MAX = 0.05       # 5% Max Compression Range
PROXIMITY_MAX_PCT = 0.018    # Close must be within 1.8% of High5/Low5 Trigger
TARGET_X = 3.0              # Target = Entry + (Entry - SL) * 3.0

EMA_PERIOD = 20
ADX_PERIOD = 14
MIN_ADX = 20.0              # Aligned with BOBD logic
APPLY_EMA_FILTER = True
APPLY_ADX_FILTER = True

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN") or os.getenv("BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID")
DASHBOARD_URL = "https://brahmastra-tech.github.io/brahmastra-scanner/"


def calc_adx(df, period=14):
    """Exact ADX calculation aligned with BOBD_Fixedv3.py."""
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


def scan_symbol_exact(symbol, df_sym):
    """
    FRESH PRE-BREAKOUT WATCHLIST:
    Scans for tight compression near trigger levels with EMA and ADX confirmation.
    """
    alerts = []
    if df_sym.empty or len(df_sym) < 20:
        return alerts

    df = df_sym.copy().sort_values("Date").reset_index(drop=True)
    
    # Rolling Setup Metrics
    df['High5'] = df['High'].rolling(5).max()
    df['Low5'] = df['Low'].rolling(5).min()
    df['AvgVol10'] = df['Volume'].rolling(10).mean()
    df['PrevClose'] = df['Close'].shift(1)
    df['Compression'] = (df['High5'] - df['Low5']) / df['PrevClose']
    
    # Indicators
    df['EMA20'] = df['Close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    df['ADX'] = calc_adx(df, period=ADX_PERIOD)

    for i in range(len(df)):
        candle = df.iloc[i]

        try:
            compression = float(candle['Compression'])
            volume = float(candle['Volume'])
            avgvol = float(candle['AvgVol10'])
            high5 = float(candle['High5'])
            low5 = float(candle['Low5'])
            close = float(candle['Close'])
            ema = float(candle['EMA20'])
            adx = float(candle['ADX']) if pd.notnull(candle['ADX']) else 0.0
            setup_date = pd.to_datetime(candle['Date']).strftime("%d-%m-%Y")
        except Exception:
            continue

        # Core Compression & Volume Filter
        if compression <= COMPRESSION_MAX and volume < avgvol:

            # ADX Filter
            if APPLY_ADX_FILTER and adx < MIN_ADX:
                continue

            # PRE_BREAKOUT WATCHLIST
            if close >= ema:
                if APPLY_EMA_FILTER and (pd.isna(ema) or close < ema):
                    continue

                # Ensure price is close to breakout level (within 1.8%)
                dist_to_trigger = (high5 - close) / close
                if dist_to_trigger > PROXIMITY_MAX_PCT:
                    continue

                entry = round(high5, 2)
                sl = round(low5, 2)
                tgt = round(entry + (entry - sl) * TARGET_X, 2)
                ema_dist = round(abs(close - ema) / ema * 100, 2) if ema > 0 else 0.0

                alerts.append({
                    "date": setup_date,
                    "symbol": symbol,
                    "timeframe": "D",
                    "type": "PRE_BREAKOUT",
                    "pattern": "PRE_BREAKOUT",
                    "entry": entry,
                    "sl": sl,
                    "target": tgt,
                    "close": round(close, 2),
                    "compression": round(compression, 4),
                    "avgvol10": int(avgvol),
                    "volume": int(volume),
                    "ema": round(ema, 2),
                    "ema_dist_pct": ema_dist,
                    "adx": round(adx, 2)
                })

            # PRE_BREAKDOWN WATCHLIST
            elif close < ema:
                if APPLY_EMA_FILTER and (pd.isna(ema) or close > ema):
                    continue

                # Ensure price is close to breakdown level (within 1.8%)
                dist_to_trigger = (close - low5) / close
                if dist_to_trigger > PROXIMITY_MAX_PCT:
                    continue

                entry = round(low5, 2)
                sl = round(high5, 2)
                tgt = round(entry - (sl - entry) * TARGET_X, 2)
                ema_dist = round(abs(close - ema) / ema * 100, 2) if ema > 0 else 0.0

                alerts.append({
                    "date": setup_date,
                    "symbol": symbol,
                    "timeframe": "D",
                    "type": "PRE_BREAKDOWN",
                    "pattern": "PRE_BREAKDOWN",
                    "entry": entry,
                    "sl": sl,
                    "target": tgt,
                    "close": round(close, 2),
                    "compression": round(compression, 4),
                    "avgvol10": int(avgvol),
                    "volume": int(volume),
                    "ema": round(ema, 2),
                    "ema_dist_pct": ema_dist,
                    "adx": round(adx, 2)
                })

    return alerts


def send_telegram_alert(signal: dict):
    """Sends individual alert to Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram credentials missing. Skipping individual alert.")
        return

    symbol = signal["symbol"]
    sig_type = signal["type"]
    setup_date = signal["date"]
    entry = signal["entry"]
    sl = signal["sl"]
    target = signal["target"]
    close = signal["close"]
    avgvol = signal["avgvol10"]
    volume = signal["volume"]
    vol_ratio = round(volume / avgvol, 2) if avgvol > 0 else 1.0
    ema = signal["ema"]
    adx = signal["adx"]

    emoji = "🚀" if sig_type == "PRE_BREAKOUT" else "📉"
    chart_url = f"https://in.tradingview.com/chart/?symbol=NSE:{symbol}"

    message = (
        f"{emoji} <b>BRAHMASTRA PRE-BREAKOUT WATCHLIST</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 <b>Stock:</b> {symbol} (NSE F&O)\n"
        f"🎯 <b>Pattern:</b> {sig_type}\n"
        f"⏱ <b>Timeframe:</b> Daily (1D) | <b>Setup Date:</b> {setup_date}\n\n"
        f"📊 <b>ACTIONABLE TRIGGER LEVELS FOR TOMORROW</b>\n"
        f"• <b>Trigger Entry Price :</b> ₹{entry:.2f}\n"
        f"• <b>Stop Loss           :</b> ₹{sl:.2f}\n"
        f"• <b>Target (3x)         :</b> ₹{target:.2f}\n"
        f"• <b>Today's Close       :</b> ₹{close:.2f}\n\n"
        f"⚡ <b>CONDITIONS PASSED</b>\n"
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


def send_summary_telegram(signal_count: int, date_str: str):
    """Sends summary message to Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram credentials missing. Skipping summary dispatch.")
        return

    if signal_count > 0:
        status_text = f"📊 <b>High-Quality Watchlist Candidates Found Today:</b> {signal_count}"
    else:
        status_text = f"ℹ️ <b>No Qualified Pre-Breakout Stocks Found Today (0 Stocks)</b>"

    message = (
        f"🏁 <b>DAILY SCAN COMPLETE ({date_str})</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{status_text}\n\n"
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
    print("🚀 Initializing Pre-Breakout Scanner Engine...")
    print(f"🔑 Telegram Secret Status: Bot Token={'FOUND' if TELEGRAM_BOT_TOKEN else 'MISSING'}, Chat ID={'FOUND' if TELEGRAM_CHAT_ID else 'MISSING'}")

    if not os.path.exists(DB_PATH):
        print(f"❌ Database file not found at {DB_PATH}!")
        return

    conn = duckdb.connect(DB_PATH)
    
    # 1. Fetch candles from DuckDB
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

    # 2. Iterate per symbol
    for sym in symbols:
        df_sym = df_raw[df_raw['Symbol'] == sym]
        alerts = scan_symbol_exact(sym, df_sym)
        if alerts:
            all_signals.extend(alerts)

    os.makedirs("data", exist_ok=True)

    if not all_signals:
        print("ℹ️ No pre-breakout signals matched conditions overall.")
        pd.DataFrame(columns=['date', 'symbol', 'timeframe', 'type', 'pattern', 'entry', 'sl', 'target', 'close', 'compression', 'avgvol10', 'volume', 'ema', 'ema_dist_pct', 'adx']).to_csv(SIGNALS_CSV, index=False)
        send_summary_telegram(0, latest_date)
        return

    # 3. Save History: Deduplicate per Date + Symbol + Pattern (keeps fresh setup per date)
    all_df = pd.DataFrame(all_signals)
    all_df["Date_DT"] = pd.to_datetime(all_df["date"], format="%d-%m-%Y")

    all_df = (
        all_df.sort_values("Date_DT")
              .drop_duplicates(subset=["date", "symbol", "pattern"], keep="last")
              .sort_values("Date_DT", ascending=False)
    )

    export_df = all_df.drop(columns=["Date_DT"])
    export_df.to_csv(SIGNALS_CSV, index=False)
    print(f"✅ Saved {len(export_df)} quality signals to {SIGNALS_CSV}.")

    # 4. Identify today's candidates
    today_signals = export_df[export_df['date'] == latest_date].to_dict('records')
    print(f"📊 Fresh Quality Candidates for Today ({latest_date}): {len(today_signals)}")

    # 5. Dispatch Telegram Alerts
    if today_signals:
        for sig in today_signals:
            send_telegram_alert(sig)
    else:
        print(f"ℹ️ 0 candidates for today ({latest_date}).")

    # 6. Always send completion summary
    send_summary_telegram(len(today_signals), latest_date)


if __name__ == "__main__":
    run_scanner()