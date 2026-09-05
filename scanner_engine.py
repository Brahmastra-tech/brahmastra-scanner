Change

Immediately after this block:

self.latest = sector_df.copy()

add:

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
