import duckdb
import polars as pl
import pandas as pd

def run_pre_breakout_scanner(db_path="data/candles.duckdb", target_x=3.0):
    conn = duckdb.connect(db_path)
    
    # 1. Load OHLCV data directly from DuckDB into Polars
    df = conn.execute("""
        SELECT symbol, timeframe, timestamp AS Date, open, high, low, close, volume
        FROM ohlcv_candles
        ORDER BY symbol, timestamp ASC
    """).pl()
    
    if df.is_empty():
        print("⚠️ No candle data available in DuckDB.")
        return pl.DataFrame()

    # 2. Vectorized Rolling Window Computations with Polars
    df_metrics = df.with_columns([
        pl.col("high").rolling_max(window_size=5).over("symbol").alias("High5"),
        pl.col("low").rolling_min(window_size=5).over("symbol").alias("Low5"),
        pl.col("volume").rolling_mean(window_size=10).over("symbol").alias("AvgVol10"),
        pl.col("close").shift(1).over("symbol").alias("PrevClose")
    ]).with_columns([
        ((pl.col("High5") - pl.col("Low5")) / pl.col("PrevClose")).alias("Compression")
    ])

    # 3. Lagged Shift to Check Breakout / Breakdown Conditions
    df_shifted = df_metrics.with_columns([
        pl.col("Compression").shift(1).over("symbol").alias("PrevCompression"),
        pl.col("volume").shift(1).over("symbol").alias("PrevVolume"),
        pl.col("AvgVol10").shift(1).over("symbol").alias("PrevAvgVol10"),
        pl.col("High5").shift(1).over("symbol").alias("PrevHigh5"),
        pl.col("Low5").shift(1).over("symbol").alias("PrevLow5")
    ])

    # Filter for signals matching compression <= 0.05 & dry volume
    valid_setup = (pl.col("PrevCompression") <= 0.05) & (pl.col("PrevVolume") < pl.col("PrevAvgVol10"))
    
    breakouts = df_shifted.filter(valid_setup & (pl.col("close") > pl.col("PrevHigh5"))).with_columns([
        pl.lit("PRE_BREAKOUT").alias("Type"),
        pl.col("PrevHigh5").alias("Entry"),
        pl.col("PrevLow5").alias("SL"),
        (pl.col("PrevHigh5") + (pl.col("PrevHigh5") - pl.col("PrevLow5")) * target_x).alias("Target")
    ])

    breakdowns = df_shifted.filter(valid_setup & (pl.col("close") < pl.col("PrevLow5"))).with_columns([
        pl.lit("PRE_BREAKDOWN").alias("Type"),
        pl.col("PrevLow5").alias("Entry"),
        pl.col("PrevHigh5").alias("SL"),
        (pl.col("PrevLow5") - (pl.col("PrevHigh5") - pl.col("PrevLow5")) * target_x).alias("Target")
    ])

    signals = pl.concat([breakouts, breakdowns]).sort("Date", descending=True)
    
    print(f"✅ Scanner Engine Execution Complete! Found {len(signals)} signals.")
    return signals

if __name__ == "__main__":
    signals_df = run_pre_breakout_scanner()
    if not signals_df.is_empty():
        print(signals_df.select(["Date", "symbol", "Type", "Entry", "SL", "Target", "close"]))
