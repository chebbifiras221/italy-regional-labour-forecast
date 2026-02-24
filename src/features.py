from __future__ import annotations

import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["geo", "year"]).copy()

    # Lag features per region
    df["unemp_rate_lag1"] = df.groupby("geo")["unemp_rate"].shift(1)
    df["gdp_lag1"] = df.groupby("geo")["gdp"].shift(1)

    # YoY GDP growth (%)
    df["gdp_yoy_pct"] = (
        (df["gdp"] - df.groupby("geo")["gdp"].shift(1))
        / df.groupby("geo")["gdp"].shift(1)
        * 100.0
    )

    # Target: next-year unemployment
    df["target_unemp_next_year"] = df.groupby("geo")["unemp_rate"].shift(-1)

    return df