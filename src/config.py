from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "crop_recommendation.csv"
PROCESSED_DATA = ROOT / "data" / "processed" / "features.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "crop_model.joblib"
FEATURES_PATH = MODEL_DIR / "selected_features.json"
REFERENCE_PATH = ROOT / "data" / "processed" / "reference_features.csv"
CURRENT_PATH = ROOT / "data" / "processed" / "current_features.csv"
METRICS_PATH = ROOT / "reports" / "metrics.json"
