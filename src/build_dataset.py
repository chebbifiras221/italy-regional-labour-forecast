from __future__ import annotations

from pathlib import Path
import pandas as pd

from .eurostat_api import (
    EurostatDataset,
    fetch_jsonstat,
    jsonstat_to_df,
    filter_italy_nuts2,
    pick_first_available,
)
from .utils import ensure_dir


def build_raw_tables(out_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = ensure_dir(out_dir)

    # Unemployment rate by NUTS2 region (tgs00010)
    unemp_ds = EurostatDataset(
        code="tgs00010",
        params={
            "lang": "EN",
            "format": "JSON",
        },
    )
    unemp_js = fetch_jsonstat(unemp_ds.code, unemp_ds.params)
    unemp_df = jsonstat_to_df(unemp_js)
    unemp_df = filter_italy_nuts2(unemp_df, geo_col="geo")

    # Some datasets contain multiple units/frequencies; select sensible defaults
    if "freq" in unemp_df.columns:
        freq = pick_first_available(unemp_df, "freq", ["A"])  # Annual preferred
        if freq is not None:
            unemp_df = unemp_df[unemp_df["freq"] == freq]
    if "unit" in unemp_df.columns:
        unit = pick_first_available(unemp_df, "unit", ["PC"])  # Percent preferred (if present)
        if unit is not None:
            unemp_df = unemp_df[unemp_df["unit"] == unit]

    # GDP at current market prices by NUTS2 (nama_10r_2gdp)
    gdp_ds = EurostatDataset(
        code="nama_10r_2gdp",
        params={
            "lang": "EN",
            "format": "JSON",
        },
    )
    gdp_js = fetch_jsonstat(gdp_ds.code, gdp_ds.params)
    gdp_df = jsonstat_to_df(gdp_js)
    gdp_df = filter_italy_nuts2(gdp_df, geo_col="geo")

    # Prefer common GDP measure: na_item=B1GQ (GDP), unit=MIO_EUR if present
    if "na_item" in gdp_df.columns:
        na_item = pick_first_available(gdp_df, "na_item", ["B1GQ"])
        if na_item is not None:
            gdp_df = gdp_df[gdp_df["na_item"] == na_item]
    if "unit" in gdp_df.columns:
        unit = pick_first_available(gdp_df, "unit", ["MIO_EUR", "EUR_HAB"])
        if unit is not None:
            gdp_df = gdp_df[gdp_df["unit"] == unit]
    if "freq" in gdp_df.columns:
        freq = pick_first_available(gdp_df, "freq", ["A"])
        if freq is not None:
            gdp_df = gdp_df[gdp_df["freq"] == freq]

    # Save raw
    unemp_df.to_csv(Path(out_dir) / "unemployment_raw.csv", index=False)
    gdp_df.to_csv(Path(out_dir) / "gdp_raw.csv", index=False)

    return unemp_df, gdp_df


def build_processed_dataset(raw_dir: str | Path, processed_dir: str | Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    processed_dir = ensure_dir(processed_dir)

    unemp = pd.read_csv(raw_dir / "unemployment_raw.csv")
    gdp = pd.read_csv(raw_dir / "gdp_raw.csv")

    # Normalize columns
    unemp = unemp.rename(
        columns={
            "time": "year",
            "value": "unemp_rate",
            "geo_name": "region",
        }
    )
    gdp = gdp.rename(
        columns={
            "time": "year",
            "value": "gdp",
            "geo_name": "region",
        }
    )

    # Year to int where possible
    unemp["year"] = pd.to_numeric(unemp["year"], errors="coerce").astype("Int64")
    gdp["year"] = pd.to_numeric(gdp["year"], errors="coerce").astype("Int64")

    # Keep essentials
    unemp = unemp[["geo", "region", "year", "unemp_rate"]].dropna(subset=["geo", "year"])
    gdp = gdp[["geo", "region", "year", "gdp"]].dropna(subset=["geo", "year"])

    # Merge (left join keeps unemployment rows)
    df = unemp.merge(
        gdp[["geo", "year", "gdp"]],
        on=["geo", "year"],
        how="left",
    )

    # Basic cleaning
    df = df.sort_values(["geo", "year"]).reset_index(drop=True)

    out_path = Path(processed_dir) / "regional_panel.csv"
    df.to_csv(out_path, index=False)
    return df