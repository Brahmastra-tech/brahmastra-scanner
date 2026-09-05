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
self.latest = sector_df.copy()

# -------------------------------------------------
# Sector Confirmation
# -------------------------------------------------

market["confirm_sector"] = (
    (market["sector_strength_score"] >= 60)
    &
    (market["sector_rank"] <= 5)
)

Then keep:

return market, sector_df
Optional Debug

Before return market, sector_df, temporarily add:

print(
    "Sector Confirmations:",
    market["confirm_sector"].sum()
)
