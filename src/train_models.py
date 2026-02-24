from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from .utils import ensure_dir, write_json


def _rmse(y_true, y_pred) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))


def train_time_aware(df_feat: pd.DataFrame, out_dir: str | Path) -> Tuple[pd.DataFrame, Dict]:

    out_dir = ensure_dir(out_dir)

    df = df_feat.dropna(subset=["target_unemp_next_year"]).copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])

    max_year = int(df["year"].max())
    test_years = [max_year - 1, max_year]

    train = df[~df["year"].isin(test_years)].copy()
    test = df[df["year"].isin(test_years)].copy()

    if len(train) < 50 or len(test) < 20:
        train, test = train_test_split(df, test_size=0.2, random_state=42)

    X_train = train[["geo", "year", "unemp_rate", "gdp", "unemp_rate_lag1", "gdp_lag1", "gdp_yoy_pct"]]
    y_train = train["target_unemp_next_year"]

    X_test = test[["geo", "year", "unemp_rate", "gdp", "unemp_rate_lag1", "gdp_lag1", "gdp_yoy_pct"]]
    y_test = test["target_unemp_next_year"]

    cat_features = ["geo"]
    num_features = ["year", "unemp_rate", "gdp", "unemp_rate_lag1", "gdp_lag1", "gdp_yoy_pct"]

    preproc = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_features),
        ]
    )

    models = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=400,
            random_state=42,
            n_jobs=-1,
        ),
    }

    metrics: Dict[str, Dict] = {}
    preds_all = []

    for name, model in models.items():

        pipe = Pipeline([("pre", preproc), ("model", model)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        metrics[name] = {
            "MAE": float(mean_absolute_error(y_test, y_pred)),
            "RMSE": float(_rmse(y_test, y_pred)),
            "R2": float(r2_score(y_test, y_pred)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        }

        tmp = test[["geo", "year", "region", "unemp_rate"]].copy()
        tmp["model"] = name
        tmp["y_true_next_year"] = y_test.to_numpy()
        tmp["y_pred_next_year"] = y_pred
        tmp["residual"] = tmp["y_true_next_year"] - tmp["y_pred_next_year"]

        preds_all.append(tmp)

    pred_df = pd.concat(preds_all, ignore_index=True)
    pred_df.to_csv(Path(out_dir) / "predictions.csv", index=False)
    write_json(Path(out_dir) / "metrics.json", metrics)

    return pred_df, metrics