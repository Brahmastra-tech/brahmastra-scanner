import os
import duckdb
import pandas as pd
import numpy as np
import requests
from datetime import datetime

# ==========================================
# CONFIGURATION (100% BOBD GUI PARITY)
# ==========================================
DB_PATH = "data/candles.duckdb"
SIGNALS_CSV = "data/signals.csv"

COMPRESSION_MAX = 0.05   # 5% Compression
TARGET_X = 3.0          # Target = Entry + (Entry - SL) * 3.0

EMA_PERIOD = 20
ADX_PERIOD = 14
MIN_ADX = 20.0
APPLY_EMA_FILTER = True
APPLY_ADX_FILTER = True

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
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

    plus_di = 100 * (plus_dm.rolling(period).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(period).sum() / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(period).mean()
    return adx


def scan_symbol_exact(symbol, df_sym):
    """
    Exact scan logic mirroring BOBD_Fixedv3.py (abv.csv format):
    - Uses Setup Candle i Date
    - Uses Raw Low5/High5 for SL
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

    for i in range(len(df) - 1):
        prev = df.iloc[i]      # Setup Candle i (Date assigned from here)
        day1 = df.iloc[i + 1]  # Trigger Candle i+1

        try:
            compression = float(prev['Compression'])
            volume = float(prev['Volume'])
            avgvol = float(prev['AvgVol10'])
            high5 = float(prev['High5'])
            low5 = float(prev['Low5'])
            
            day1_close = float(day1['Close'])
            day1_ema = float(day1['EMA20'])
            day1_adx = float(day1['ADX']) if pd.notnull(day1['ADX']) else 0.0
            
            # Attributed to SETUP CANDLE DATE (matches abv.csv)
            setup_date = pd.to_datetime(prev['Date']).strftime("%d-%m-%Y")
        except Exception:
            continue

        # 1. Base Setup Condition (Compression + Low Volume)
        if compression <= COMPRESSION_MAX and volume < avgvol:

            # ADX Filter
            if APPLY_ADX_FILTER and day1_adx < MIN_ADX:
                continue

            # PRE_BREAKOUT
            if day1_close > high5:
                if APPLY_EMA_FILTER and (pd.isna(day1_ema) or day1_close < day1_ema):
                    continue

                entry = round(high5, 2)
                sl = round(low5, 2)  # Raw Low5 (matches abv.csv)
                tgt = round(entry + (entry - sl) * TARGET_X, 2)
                ema_dist = round(abs(day1_close - day1_ema) / day1_ema * 100, 2) if pd.notnull(day1_ema) and day1_ema > 0 else 0.0

                alerts.append({
                    "Date": setup_date,
                    "Symbol": symbol,
                    "Timeframe": "D",
                    "Type": "PRE_BREAKOUT",
                    "Entry": entry,
                    "SL": sl,
                    "Target": tgt,
                    "Close": round(day1_close, 2),
                    "Compression": round(compression, 4),
                    "AvgVol10": int(avgvol),
                    "Volume": int(day1['Volume']),
                    "EMA": round(day1_ema, 2) if pd.notnull(day1_ema) else 0.0,
                    "EMA_Dist%": ema_dist,
                    "ADX": round(day1_adx, 2)
                })

            # PRE_BREAKDOWN
            elif day1_close < low5:
                if APPLY_EMA_FILTER and (pd.isna(day1_ema) or day1_close > day1_ema):
                    continue

                entry = round(low5, 2)
                sl = round(high5, 2)  # Raw High5 (matches abv.csv)
                tgt = round(entry - (sl - entry) * TARGET_X, 2)
                ema_dist = round(abs(day1_close - day1_ema) / day1_ema * 100, 2) if pd.notnull(day1_ema) and day1_ema > 0 else 0.0

                alerts.append({
                    "Date": setup_date,
                    "Symbol": symbol,
                    "Timeframe": "D",
                    "Type": "PRE_BREAKDOWN",
                    "Entry": entry,
                    "SL": sl,
                    "Target": tgt,
                    "Close": round(day1_close, 2),
                    "Compression": round(compression, 4),
                    "AvgVol10": int(avgvol),
                    "Volume": int(day1['Volume']),
                    "EMA": round(day1_ema, 2) if pd.notnull(day1_ema) else 0.0,
                    "EMA_Dist%": ema_dist,
                    "ADX": round(day1_adx, 2)
                })

    return alerts


def send_telegram_alert(signal: dict):
    """Sends individual alert to Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    symbol = signal["Symbol"]
    sig_type = signal["Type"]
    date_str = signal["Date"]
    entry = signal["Entry"]
    sl = signal["SL"]
    target = signal["Target"]
    close = signal["Close"]
    avgvol = signal["AvgVol10"]
    volume = signal["Volume"]
    vol_ratio = round(volume / avgvol, 2) if avgvol > 0 else 1.0
    ema = signal["EMA"]
    adx = signal["ADX"]

    emoji = "🚀" if sig_type == "PRE_BREAKOUT" else "📉"
    
    message = (
        f"{emoji} <b>BRAHMASTRA SIGNAL DETECTED</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 <b>Stock:</b> {symbol} (NSE F&O)\n"
        f"🎯 <b>Pattern:</b> {sig_type}\n"
        f"⏱ <b>Timeframe:</b> Daily (1D) | <b>Setup Date:</b> {date_str}\n\n"
        f"📊 <b>TRADE LEVELS</b>\n"
        f"• <b>Entry Price :</b> ₹{entry:.2f}\n"
        f"• <b>Stop Loss   :</b> ₹{sl:.2f}\n"
        f"• <b>Target (3x) :</b> ₹{target:.2f}\n"
        f"• <b>Close Price :</b> ₹{close:.2f}\n\n"
        f"⚡ <b>INDICATORS</b>\n"
        f"• <b>20 EMA       :</b> ₹{ema:.2f}\n"
        f"• <b>14 ADX       :</b> {adx:.2f}\n"
        f"• <b>Volume Ratio :</b> {vol_ratio:.2f}x\n"
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
    """Sends summary web dashboard link after alerts."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    message = (
        f"🏁 <b>DAILY SCAN COMPLETE ({date_str})</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>Signals Found Today:</b> {signal_count}\n\n"
        f"🌐 <b>Interactive Web Dashboard & History:</b>\n"
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


def run_scanner():
    if not os.path.exists(DB_PATH):
        print("❌ Database file not found!")
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
    print(f"🔍 Running Scanner Engine (Market Date: {latest_date})...")

    symbols = df_raw['Symbol'].unique()
    all_signals = []

    # 2. Iterate per symbol
    for sym in symbols:
        df_sym = df_raw[df_raw['Symbol'] == sym]
        alerts = scan_symbol_exact(sym, df_sym)
        if alerts:
            all_signals.extend(alerts)

    if not all_signals:
        print("ℹ️ No signals matched conditions.")
        return

    # 3. Export to CSV matching exact abv.csv schema
    export_df = pd.DataFrame(all_signals)
    export_df['Date_DT'] = pd.to_datetime(export_df['Date'], format="%d-%m-%Y")
    export_df = export_df.sort_values('Date_DT', ascending=False).drop(columns=['Date_DT'])

    os.makedirs("data", exist_ok=True)
    export_df.to_csv(SIGNALS_CSV, index=False)
    print(f"✅ Saved {len(export_df)} signals matching abv.csv format to {SIGNALS_CSV}.")

    # 4. Dispatch Telegram Alerts for Latest Market Date
    today_signals = export_df[export_df['Date'] == latest_date].to_dict('records')
    
    if today_signals:
        print(f"📢 Sending {len(today_signals)} alerts for today ({latest_date})...")
        for sig in today_signals:
            send_telegram_alert(sig)
        send_summary_telegram(len(today_signals), latest_date)
    else:
        print(f"ℹ️ No new signals triggered today ({latest_date}).")


if __name__ == "__main__":
    run_scanner()