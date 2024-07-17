import os
from datetime import datetime

import kaggle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

import mlflow
import mlflow.sklearn
from airflow import DAG
from airflow.decorators import task
from airflow.utils.log.logging_mixin import LoggingMixin

logger = LoggingMixin().log

NAIVE_BAYES_FEATURES = [
    "cap-diameter",
    "cap-shape",
    "gill-attachment",
    "gill-color",
]

LR_FEATURES = [
    "stem-height",
    "stem-width",
    "stem-color",
    "season",
]


def prepare_data(dataset_path, features):
    data = pd.read_csv(dataset_path)
    y = data["class"]
    X = data[features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def train_and_log_model(X_train, X_test, y_train, y_test, model, model_name):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    logger.info(f"{model_name} accuracy: {accuracy:.2f}")

    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=f"{model_name}",
            registered_model_name=f"{model_name}",
        )
        mlflow.log_metric(f"{model_name}_accuracy", accuracy)
        mlflow.log_params(
            {f"{model_name}_{k}": v for k, v in model.get_params().items()}
        )

    logger.info(f"{model_name} model logged with MLFlow run ID: {run.info.run_id}")


with DAG(
    "Mushroom-Classifier-Pipeline",
    start_date=datetime(2024, 7, 17),
    description="A DAG to train and log Mushroom Classifier models",
    schedule_interval="0 0 * * *",
    max_active_runs=1,
    catchup=False,
) as dag:

    @task()
    def download_dataset():
        try:
            download_path = os.environ.get("AIRFLOW_DATA_PATH", "/opt/airflow/data")
            kaggle.api.dataset_download_files(
                "prishasawhney/mushroom-dataset", path=download_path, unzip=True
            )

            logger.info(f"Dataset downloaded successfully to {download_path}")
            return os.path.join(download_path, "mushroom_cleaned.csv")
        except Exception as e:
            logger.error(f"Error downloading dataset: {str(e)}")
            raise

    @task()
    def train_logistic_regression(dataset_path: str):
        X_train, X_test, y_train, y_test = prepare_data(dataset_path, LR_FEATURES)
        train_and_log_model(
            X_train,
            X_test,
            y_train,
            y_test,
            LogisticRegression(),
            "logistic_regression_mushroom_classifier",
        )

    @task()
    def train_naive_bayes(dataset_path: str):
        X_train, X_test, y_train, y_test = prepare_data(
            dataset_path, NAIVE_BAYES_FEATURES
        )
        train_and_log_model(
            X_train,
            X_test,
            y_train,
            y_test,
            GaussianNB(),
            "naive_bayes_mushroom_classifier",
        )

    dataset_path = download_dataset()
    train_logistic_regression(dataset_path)
    train_naive_bayes(dataset_path)
