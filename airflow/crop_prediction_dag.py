from datetime import datetime
from pathlib import Path
import subprocess
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator


# Project root folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_command(command):
    """Run a project command safely on Windows."""
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True
    )


def run_preprocess():
    run_command([sys.executable, "-m", "src.preprocess"])


def run_feature_selection():
    run_command([sys.executable, "-m", "src.feature_selection"])


def run_training():
    run_command([sys.executable, "-m", "src.train"])


def run_drift_check():
    run_command([sys.executable, "scripts/retrain_if_drift.py"])


with DAG(
    dag_id="smart_agriculture_mlops_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
    tags=["mlops", "agriculture"],
) as dag:

    preprocess = PythonOperator(
        task_id="preprocess",
        python_callable=run_preprocess,
    )

    feature_selection = PythonOperator(
        task_id="feature_selection",
        python_callable=run_feature_selection,
    )

    train = PythonOperator(
        task_id="train_and_track",
        python_callable=run_training,
    )

    monitor = PythonOperator(
        task_id="drift_check_and_retrain",
        python_callable=run_drift_check,
    )

    preprocess >> feature_selection >> train >> monitor