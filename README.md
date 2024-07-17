
## Overview

This project implements a machine learning pipeline for classifying mushrooms using Apache Airflow, MLflow, and FastAPI.

## Project Structure
```sh
.
├── README.md
├── docker-compose.yaml
├── airflow
│   ├── Dockerfile
│   ├── dags
│   │   └── mushroom_classifier_pipeline.py
│   └── requirements.txt
├── inference
│   ├── Dockerfile
│   ├── app
│   │   ├── main.py
│   │   ├── model.py
│   │   └── schema.py
│   ├── requirements.txt
│   └── server.py
│   └── tests
│       └── test_main.py
├── mlflow
    ├── Dockerfile
    └── requirements.txt
```

This project consists of three main components:
1. **Airflow**: Orchestrates the training pipeline for two machine learning models (Naive Bayes and Logistic Regression).
2. **MLflow**: Tracks experiments, stores trained models, and manages model versions.
3. **FastAPI**: Provides an API for model inference.


## Tech Stack

* **Python** as the primary programming language.
* **Apache Airflow** for orchestrating the machine learning pipeline and workflow management.
* **Scikit-learn** for implementing machine learning models (Naive Bayes and Logistic Regression).
* **MLflow** for experiment tracking, model versioning, and model registry.
* **FastAPI** for creating a high-performance API for model inference.
* **Docker & Docker Compose** for containerization and easy deployment of all components.
* **Pandas** for data manipulation and preprocessing.
* **PostgreSQL** as the backend database for Airflow and MLFlow.


## Quick Start
1. Ensure Docker and Docker Compose are installed on your system.
2. Run the following commands:
```shell
git clone git@github.com:HamdyTawfeek/mlops-pipeline.git
cd mlops-pipeline
docker-compose up --build
```

## Usage

1. Access the Airflow web interface at `http://localhost:8080` to *trigger* the training pipeline.
   * username: `admin`
   * password: `admin`
2. View experiment results and model versions in the MLflow UI at `http://localhost:5000`.
3. Make predictions using the FastAPI service at `http://localhost:8000/predict`.
   *  Check API documentation at `http://localhost:8000/docs`.
4. [Optional] After the models are registered in MLFlow, you can run the tests `docker-compose run --rm test`


## API Usage Example

The `/predict` endpoint routes requests to two different models based on the input features:

1. `naive_bayes_mushroom_classifier`: Used when features are `{"cap-diameter", "cap-shape", "gill-attachment", "gill-color"}`
2. `logistic_regression_mushroom_classifier`: Used when features are `{"stem-height", "stem-width", "stem-color", "season"}`

Here's an example using `curl` for the logistic regression model:

```bash
curl --location 'http://localhost:8000/predict' \
--header 'Content-Type: application/json' \
--data '{
  "stem_height": 3.8074667544799388,
  "stem_width": 1545,
  "stem_color": 11,
  "season": 1.8042727086281731
}'
```

Example Response
```json
{
  "prediction":  "edible",
  "probability": 0.8211
}
```


For the Naive Bayes model, you would use different features:

```bash
curl --location 'http://localhost:8000/predict' \
--header 'Content-Type: application/json' \
--data '{
  "cap-diameter": 1372,
  "cap-shape": 2,
  "gill-attachment": 2,
  "gill-color": 10
}'
```

Example Response
```json
{
  "prediction":  "poisonous",
  "probability":  0.4051
}
```


## MLOps Components Overview and Design Decisions

MLOps is a set of practices that aims to deploy and maintain machine learning models in production reliably and efficiently. Here are the key components of a typical MLOps system and how they're implemented in this project:

1. **Data Management**
   - Involves collecting, storing, and versioning data used for training and evaluation.
   - In this project: We used the Kaggle API to collect our data. For simplicity, we didn't use feature store or data versioning.

2. **Feature Engineering**
   - Processes raw data into features suitable for machine learning models.
   - May include feature selection, extraction, and transformation.
   - In this project: Minimal feature engineering was performed as the dataset is simple and the project instructions didn't require extensive preprocessing.

3. **Model Development**
   - Encompasses the process of designing, training, and evaluating machine learning models.
   - Includes algorithm selection, hyperparameter tuning, and model validation.
   - In this project: We used simple scikit-learn to train the models without much feature engineering.

4. **Experiment Tracking**
   - Records all experiments, including model parameters, performance metrics, and artifacts.
   - In this project: We used MLflow experiments to track the training experiments.

5. **Model Versioning**
   - Manages different versions of models as they evolve over time.
   - Enables rollback to previous versions if needed.
   - In this project: We used MLflow to version the models we trained.

6. **Pipeline Orchestration**
   - Automates and manages the workflow of data processing, model training, and deployment.
   - Ensures reproducibility and reduces manual intervention.
   - In this project: We used Airflow to orchestrate our data pipeline.

7. **Application and Model Deployment**
   - Involves putting trained models into production environments.
   - In this project: We are using docker to containerize our applications and MLflow to consume our models.

8. **Model Serving**
   - Provides an interface (often an API) for making predictions using the deployed model.
   - Handles scaling and performance optimization for inference requests.
   - In this project: We used FastAPI to make predictions with our models.

9. **Monitoring and Logging**
   - Tracks the performance and health of deployed models in real-time.
   - Detects issues such as model drift or performance degradation.
   - In this project: 
     - We implemented basic monitoring and logging using simple logging mechanisms.
     - Model performance metrics are submitted to MLflow, allowing for basic tracking of model behavior over time.
     - For more advanced projects, tools like Sentry (for real-time error tracking), Grafana (for metrics visualization), and Evidently (for ML-specific monitoring) could be integrated to provide more comprehensive monitoring and logging capabilities.

10. **Continuous Integration/Continuous Deployment/Continuous Training (CI/CD/CT)**
    - Automates testing and deployment of machine learning pipelines.
    - Ensures that code changes and model updates are reliably deployed to production.
    - In this project:
      - Continuous Training: Airflow runs our DAG on a schedule, automatically retraining our models with the latest data.
      - Continuous Deployment: After training, the models are automatically pushed to MLflow, our model registry.
      - Continuous Integration: The FastAPI application automatically consumes the latest model from MLflow, ensuring that the most recent model version is always in use for predictions.
    
    This setup creates a seamless pipeline where model training, versioning, and deployment are automated, reducing manual intervention and ensuring that the latest model is always available for inference.

11. **Infrastructure Management**
    - Handles the underlying computational resources and environments.
    - Using tools like Terraform for managing cloud resources and kubernetes for environment managment.
    - In this project: We used a local docker environment.

These components work together to create a robust system for developing, deploying, and maintaining machine learning models in production. In this project, we've implemented several key aspects of the MLOps pipeline, focusing on data collection, model development, experiment tracking, and pipeline orchestration. Some components, such as advanced model deployment strategies, comprehensive monitoring, and sophisticated infrastructure management, may not be fully implemented in this project due to its scope and simplicity.