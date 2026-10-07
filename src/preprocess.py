import pandas as pd
from src.config import RAW_DATA, PROCESSED_DATA

FEATURES = ["N","P","K","temperature","humidity","ph","rainfall"]

def load_data():
    df = pd.read_csv(RAW_DATA)
    df = df.drop_duplicates().dropna()
    return df

def engineer_features(df):
    out = df.copy()
    out["NPK_total"] = out["N"] + out["P"] + out["K"]
    out["temp_humidity_ratio"] = out["temperature"] / (out["humidity"] + 1e-6)
    out["rainfall_log"] = __import__("numpy").log1p(out["rainfall"])
    return out

def save_processed(df):
    PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA, index=False)

if __name__ == "__main__":
    df = engineer_features(load_data())
    save_processed(df)
    print(f"Processed rows: {len(df)}")
