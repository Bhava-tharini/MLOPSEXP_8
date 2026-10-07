import pandas as pd
from pathlib import Path
from src.preprocess import load_data, engineer_features, save_processed
from features.feature_store import materialize
from src.feature_selection import select_features

df = engineer_features(load_data())
save_processed(df)
select_features(df)
materialize()

processed = pd.read_csv(Path("data/processed/features.csv"))
processed.sample(frac=0.7, random_state=42).to_csv("data/processed/reference_features.csv", index=False)
processed.sample(frac=0.7, random_state=99).to_csv("data/processed/current_features.csv", index=False)

print("Data setup complete.")
