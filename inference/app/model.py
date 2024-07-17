import functools
import logging

import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=None)
def load_model(model_name):
    try:
        model = mlflow.sklearn.load_model(f"models:/{model_name}/latest")
        logger.info(f"{model_name} Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


def select_model(feature_keys):
    naive_bayes_keys = {"cap-diameter", "cap-shape", "gill-attachment", "gill-color"}
    logistic_regression_keys = {"stem-height", "stem-width", "stem-color", "season"}

    if naive_bayes_keys.issubset(feature_keys):
        return "naive_bayes_mushroom_classifier"
    elif logistic_regression_keys.issubset(feature_keys):
        return "logistic_regression_mushroom_classifier"
    else:
        raise ValueError("Unsupported feature set")
