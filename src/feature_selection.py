import json
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from src.config import PROCESSED_DATA, FEATURES_PATH

BASE_FEATURES = [
    "N","P","K","temperature","humidity","ph","rainfall",
    "NPK_total","temp_humidity_ratio","rainfall_log"
]

def select_features(df, k=7):
    X = df[BASE_FEATURES]
    y = df["label"]
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    selected = list(X.columns[selector.get_support()])
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURES_PATH.write_text(json.dumps(selected, indent=2))
    return selected

if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_DATA)
    print("Selected features:", select_features(df))
