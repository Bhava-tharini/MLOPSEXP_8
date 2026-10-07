"""
Feature management / feature-store concept.

This lightweight local feature store keeps a versioned feature table.
For a production implementation, this module can be replaced by Feast
without changing the model-training interface.
"""
import pandas as pd
from pathlib import Path
from src.config import PROCESSED_DATA

FEATURE_STORE_DIR = Path(__file__).resolve().parents[1] / "data" / "feature_store"

def materialize():
    FEATURE_STORE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROCESSED_DATA)
    path = FEATURE_STORE_DIR / "crop_features.csv"
    df.to_csv(path, index=False)
    return path

if __name__ == "__main__":
    print("Feature store:", materialize())
