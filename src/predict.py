import json
import joblib
import pandas as pd
from src.config import MODEL_PATH, FEATURES_PATH

def predict(values: dict):
    model = joblib.load(MODEL_PATH)
    features = json.loads(FEATURES_PATH.read_text())
    row = pd.DataFrame([values])
    row["NPK_total"] = row["N"] + row["P"] + row["K"]
    row["temp_humidity_ratio"] = row["temperature"] / (row["humidity"] + 1e-6)
    row["rainfall_log"] = __import__("numpy").log1p(row["rainfall"])
    return model.predict(row[features])[0]
