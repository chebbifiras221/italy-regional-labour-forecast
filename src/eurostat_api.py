from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests


EUROSTAT_BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"


@dataclass(frozen=True)
class EurostatDataset:
    code: str
    params: Dict[str, str]


def _safe_get(d: Dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def fetch_jsonstat(dataset_code: str, params: Dict[str, str], timeout: int = 60) -> Dict[str, Any]:
    """
    Fetch a JSON-stat 2.0 dataset from Eurostat Statistics API.
    API structure documented by Eurostat: {host}/dissemination/statistics/1.0/data/{DATASET_CODE}?...
    """
    params = dict(params)
    params.setdefault("format", "JSON")
    params.setdefault("lang", "EN")

    url = f"{EUROSTAT_BASE}/{dataset_code}"
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def jsonstat_to_df(js: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert Eurostat JSON-stat 2.0 to a tidy DataFrame with one row per observation.
    This function is intentionally generic and does not assume specific dimensions.
    """
    dim_ids: List[str] = js["id"]  # e.g. ["freq","unit","geo","time"]
    dim_sizes: List[int] = js["size"]
    dim_obj: Dict[str, Any] = js["dimension"]

    # Build ordered categories for each dimension (index -> code, code -> label)
    dim_codes: List[List[str]] = []
    dim_labels: List[Dict[str, str]] = []

    for dim in dim_ids:
        cat = dim_obj[dim]["category"]
        # category.index is mapping code -> position
        index_map: Dict[str, int] = cat["index"]
        # invert by position
        codes_by_pos = [None] * len(index_map)
        for code, pos in index_map.items():
            codes_by_pos[pos] = code
        codes = [c for c in codes_by_pos if c is not None]
        labels = cat.get("label", {})
        dim_codes.append(codes)
        dim_labels.append(labels)

    # Values can be dict (sparse) or list
    values = js.get("value", {})
    if isinstance(values, list):
        # dense array
        value_map = {i: v for i, v in enumerate(values) if v is not None}
    elif isinstance(values, dict):
        value_map = {int(k): v for k, v in values.items() if v is not None}
    else:
        value_map = {}

    # Compute multipliers to decode flat index -> multidim coordinates
    multipliers: List[int] = []
    m = 1
    for size in reversed(dim_sizes):
        multipliers.append(m)
        m *= size
    multipliers = list(reversed(multipliers))

    rows: List[Dict[str, Any]] = []
    for flat_i, v in value_map.items():
        coords: List[int] = []
        remainder = flat_i
        for size, mult in zip(dim_sizes, multipliers):
            idx = remainder // mult
            remainder = remainder % mult
            # safety clamp
            if idx >= size:
                idx = size - 1
            coords.append(idx)

        row: Dict[str, Any] = {}
        for dim, idx, codes, labels in zip(dim_ids, coords, dim_codes, dim_labels):
            code = codes[idx]
            row[dim] = code
            # human label (if available)
            row[f"{dim}_name"] = labels.get(code, code)
        row["value"] = v
        rows.append(row)

    return pd.DataFrame(rows)


def filter_italy_nuts2(df: pd.DataFrame, geo_col: str = "geo") -> pd.DataFrame:
    """
    Keep Italy NUTS2 regions.
    Empirically, NUTS2 codes are length 4, start with 'IT' (e.g., ITC1, ITF3).
    """
    if geo_col not in df.columns:
        return df
    mask = df[geo_col].astype(str).str.match(r"^IT.{2}$")
    return df.loc[mask].copy()


def pick_first_available(df: pd.DataFrame, dim: str, preferred: List[str]) -> str | None:
    if dim not in df.columns:
        return None
    available = list(pd.unique(df[dim].dropna().astype(str)))
    for p in preferred:
        if p in available:
            return p
    return available[0] if available else None