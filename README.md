# 🌾 Smart Agriculture Prediction — MLOps Mini Project

## Project
Build an Automated Crop/Weather Prediction MLOps Pipeline.

### Prediction problem
Crop recommendation using soil and weather features.

### Input features
- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- pH
- Rainfall

### Output
Recommended crop.

## MLOps requirements covered

| Requirement | Implementation |
|---|---|
| Feature engineering | `src/preprocess.py` |
| Feature selection | `src/feature_selection.py` |
| Experiment tracking | MLflow in `src/train.py` |
| Feature management/store concept | `features/feature_store.py` |
| DAG | `airflow/crop_prediction_dag.py` |
| Automated retraining | `scripts/retrain_if_drift.py` |
| Drift detection | `monitoring/drift_detection.py` |
| Serving | FastAPI |
| Containerization | Docker |
| Version control | Git/GitHub |

## ⚠️ Dataset note
The included CSV is a **synthetic demo dataset** created so the project runs immediately without external downloads. For final college submission, replace it with your instructor-approved/public crop recommendation dataset and rerun the pipeline. Do not claim the included synthetic data is an external benchmark dataset.

## Quick start — Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python scripts/setup_data.py
python -m src.train
```

## Run MLflow

```powershell
mlflow ui
```

Open the displayed local MLflow address in your browser.

## Run API

```powershell
uvicorn api.app:app --reload
```

Open:
`http://127.0.0.1:8000/docs`

Example request:

```json
{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 20.8,
  "humidity": 82,
  "ph": 6.5,
  "rainfall": 202
}
```

## Run drift detection

```powershell
python scripts/retrain_if_drift.py
```

If drift is detected, the script automatically runs preprocessing, feature selection and model training again.

## Airflow DAG

The DAG is provided in:

`airflow/crop_prediction_dag.py`

For Windows, use Docker/WSL/Linux for a full Airflow environment. The main ML pipeline itself does not require Airflow.

## Docker

```powershell
docker compose up --build
```

Then open:
`http://127.0.0.1:8000/docs`

## Project flow

Dataset
→ Data validation
→ Feature engineering
→ Feature selection
→ Feature management
→ Model training
→ MLflow tracking
→ Model selection
→ Deployment
→ Drift monitoring
→ Automated retraining

## Suggested viva explanation

"This project implements an MLOps pipeline for crop recommendation. Soil and weather features are engineered and selected before model training. MLflow tracks experiments and metrics. A feature-store concept manages reusable features. An Airflow DAG automates the workflow. Drift detection compares reference and current feature distributions, and retraining is triggered when significant drift is detected. The selected model is served through a FastAPI endpoint."
