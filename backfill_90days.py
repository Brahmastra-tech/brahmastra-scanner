import os
import duckdb
import pandas as pd
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
DB_PATH = "data/candles.duckdb"
SIGNALS_CSV = "data/signals.csv"
BACKTEST_CSV = "data/backtest_results.csv"
TARGET_X = 3.0  # 3:1 Reward-to-Risk Ratio


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates Wilder's RSI safely without NaN issues."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-5)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def run_90day_backfill():
    print("⏳ Running Historical 90-Day Backfill & Signals Generator...")

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
    if "delivery_qty" in existing_cols:
        select_parts.append("delivery_qty AS DeliveryQty")
    else:
        select_parts.append("volume * 0.45 AS DeliveryQty")

    if "delivery_pct" in existing_cols:
        select_parts.append("delivery_pct AS DeliveryPct")
    else:
        select_parts.append("45.0 AS DeliveryPct")

    query = f"SELECT {', '.join(select_parts)} FROM ohlcv_candles ORDER BY symbol, timestamp ASC"
    df_raw = conn.execute(query).df()
    conn.close()

    if df_raw.empty:
        print("⚠️ No data in DuckDB.")
        return

    df_raw["Date_DT"] = pd.to_datetime(df_raw["Date"])
    
    # Store candidates grouped by Date
    daily_candidates_map = {}

    for symbol, df_sym in df_raw.groupby('Symbol'):
        if len(df_sym) < 15:
            continue

        df = df_sym.copy().sort_values("Date_DT").reset_index(drop=True)

        # 1. Indicator Calculations
        df['RSI_Daily'] = compute_rsi(df['Close'], 14)
        df['Prev_RSI_Daily'] = df['RSI_Daily'].shift(1).fillna(df['RSI_Daily'])

        df_weekly = df.set_index('Date_DT').resample('W-FRI').agg({'Close': 'last'}).dropna()
        if len(df_weekly) >= 5:
            df_weekly['RSI_Weekly'] = compute_rsi(df_weekly['Close'], 14)
            df_weekly['Prev_RSI_Weekly'] = df_weekly['RSI_Weekly'].shift(1).fillna(df_weekly['RSI_Weekly'])
        else:
            df_weekly['RSI_Weekly'] = 55.0
            df_weekly['Prev_RSI_Weekly'] = 50.0

        df_monthly = df.set_index('Date_DT').resample('ME').agg({'Close': 'last'}).dropna()
        if len(df_monthly) >= 2:
            df_monthly['RSI_Monthly'] = compute_rsi(df_monthly['Close'], 14)
            df_monthly['Prev_RSI_Monthly'] = df_monthly['RSI_Monthly'].shift(1).fillna(df_monthly['RSI_Monthly'])
        else:
            df_monthly['RSI_Monthly'] = 55.0
            df_monthly['Prev_RSI_Monthly'] = 50.0

        df = pd.merge_asof(df, df_weekly[['RSI_Weekly', 'Prev_RSI_Weekly']], on='Date_DT', direction='backward')
        df = pd.merge_asof(df, df_monthly[['RSI_Monthly', 'Prev_RSI_Monthly']], on='Date_DT', direction='backward')

        df['RSI_Weekly'] = df['RSI_Weekly'].fillna(55.0)
        df['Prev_RSI_Weekly'] = df['Prev_RSI_Weekly'].fillna(50.0)
        df['RSI_Monthly'] = df['RSI_Monthly'].fillna(55.0)
        df['Prev_RSI_Monthly'] = df['Prev_RSI_Monthly'].fillna(50.0)

        df['Deliv_SMA20'] = df['DeliveryQty'].rolling(20, min_periods=1).mean().fillna(df['DeliveryQty'])
        df['Deliv_Spike'] = (df['DeliveryQty'] / (df['Deliv_SMA20'] + 1e-5)).fillna(1.0)
        df['Day_Range'] = (df['High'] - df['Low']).clip(lower=1e-5)
        df['Close_Location'] = ((df['Close'] - df['Low']) / df['Day_Range']).fillna(0.5)

        # 2. Loop through all historical bars
        for i in range(len(df)):
            row = df.iloc[i]
            date_str = row['Date_DT'].strftime("%d-%m-%Y")

            c_mtf_monthly = row['RSI_Monthly'] >= row['Prev_RSI_Monthly']
            c_mtf_weekly = row['RSI_Weekly'] >= row['Prev_RSI_Weekly']
            rsi_diff = abs(row['RSI_Daily'] - row['Prev_RSI_Daily'])
            c_daily_rsi_squeeze = rsi_diff <= 3.5

            deliv_score = float(np.clip((row['DeliveryPct'] / 70.0 * 50.0), 10, 50))
            close_score = float(np.clip(row['Close_Location'] * 50.0, 10, 50))
            brs_score = round(deliv_score + close_score, 2)

            entry = round(float(row['High']) + 0.05, 2)
            sl = round(float(row['Low']), 2)
            risk = max(entry - sl, float(row['Close']) * 0.01)
            target = round(entry + (risk * TARGET_X), 2)

            deliv_pct_val = round(float(row['DeliveryPct']), 2)
            rsi_daily_val = round(float(row['RSI_Daily']), 2)

            sig_data = {
                "Date": date_str,
                "Symbol": symbol,
                "Timeframe": "D",
                "Type": "PRE_BREAKOUT",
                "Pattern": "PRE_BREAKOUT",
                "BRS_Score": brs_score,
                "Entry": entry,
                "SL": sl,
                "Target": target,
                "Close": round(float(row['Close']), 2),
                "Volume": int(row['Volume']),
                "EMA20": deliv_pct_val,
                "ADX14": rsi_daily_val,
                "DeliveryQty": int(row['DeliveryQty']),
                "DeliveryPct": deliv_pct_val,
                "DelivSpikeRatio": round(float(row['Deliv_Spike']), 2),
                "Daily_RSI": rsi_daily_val
            }

            if date_str not in daily_candidates_map:
                daily_candidates_map[date_str] = []

            daily_candidates_map[date_str].append(sig_data)

    # 3. Extract Top 3 Candidates for EVERY historical date
    all_final_signals = []
    for d_str, candidates in daily_candidates_map.items():
        top_3_day = pd.DataFrame(candidates).sort_values("BRS_Score", ascending=False).head(3)
        all_final_signals.extend(top_3_day.to_dict('records'))

    os.makedirs("data", exist_ok=True)

    if all_final_signals:
        signals_df = pd.DataFrame(all_final_signals)
        # Save complete multi-day history to signals.csv
        signals_df.to_csv(SIGNALS_CSV, index=False)
        print(f"✅ Successfully backfilled signals.csv with {len(signals_df)} historical records across {len(daily_candidates_map)} dates!")
    else:
        print("⚠️ No historical signals generated.")


if __name__ == "__main__":
    run_90day_backfill()