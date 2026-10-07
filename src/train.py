import json
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.config import PROCESSED_DATA, FEATURES_PATH, MODEL_PATH, METRICS_PATH

def train():
    df = pd.read_csv(PROCESSED_DATA)
    features = json.loads(FEATURES_PATH.read_text())
    X, y = df[features], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("smart-agriculture-crop-prediction")

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42)
    }

    best = None
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            metrics = {
                "accuracy": accuracy_score(y_test, pred),
                "precision_weighted": precision_score(y_test, pred, average="weighted", zero_division=0),
                "recall_weighted": recall_score(y_test, pred, average="weighted", zero_division=0),
                "f1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0),
            }
            mlflow.log_param("model", name)
            mlflow.log_param("n_features", len(features))
            mlflow.log_params({"features": ",".join(features)})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
    model,
    "model",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)

            if best is None or metrics["f1_weighted"] > best["metrics"]["f1_weighted"]:
                best = {"model": model, "name": name, "metrics": metrics}

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["model"], MODEL_PATH)
    METRICS_PATH.write_text(json.dumps({
        "best_model": best["name"],
        **best["metrics"]
    }, indent=2))
    print("Best model:", best["name"])
    print(json.dumps(best["metrics"], indent=2))

if __name__ == "__main__":
    train()
