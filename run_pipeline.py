from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.build_dataset import build_raw_tables, build_processed_dataset
from src.features import add_features
from src.train_models import train_time_aware
from src.utils import ensure_dir


ROOT = Path(__file__).resolve().parent

RAW_DIR = ensure_dir(ROOT / "data" / "raw")
PROCESSED_DIR = ensure_dir(ROOT / "data" / "processed")
MODELS_DIR = ensure_dir(ROOT / "models")


def main() -> None:
    print("1) Downloading raw data from Eurostat...")
    build_raw_tables(RAW_DIR)

    print("2) Building processed panel dataset...")
    panel = build_processed_dataset(RAW_DIR, PROCESSED_DIR)

    print("3) Creating features...")
    feat = add_features(panel)
    feat_path = PROCESSED_DIR / "regional_panel_features.csv"
    feat.to_csv(feat_path, index=False)

    print("4) Training models...")
    _, metrics = train_time_aware(feat, MODELS_DIR)
    print("Done. Metrics:")
    for m, vals in metrics.items():
        print(m, vals)

    print("\nRun the dashboard:")
    print("  streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()