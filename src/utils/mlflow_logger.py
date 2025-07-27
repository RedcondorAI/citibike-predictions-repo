from dotenv import load_dotenv
load_dotenv()
import logging
import os

import mlflow
from mlflow.models import infer_signature

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_mlflow_tracking():
    """
    Set up MLflow tracking URI using environment variables.
    """
    uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(uri)
    logger.info("MLflow tracking URI set.")
    return mlflow


def log_model_to_mlflow(
    model,
    input_data,
    experiment_name,
    metric_name="mae",
    model_name=None,
    params=None,
    score=None,
):
    """
    Log model, metrics, and optionally register the model in MLflow.
    """
    try:
        mlflow.set_experiment(experiment_name)
        logger.info(f"Experiment set to: {experiment_name}")

        with mlflow.start_run():
            if params:
                mlflow.log_params(params)
                logger.info(f"Logged parameters: {params}")

            if score is not None:
                mlflow.log_metric(metric_name, score)
                logger.info(f"Logged {metric_name}: {score}")

            signature = infer_signature(input_data, model.predict(input_data))
            logger.info("Model signature inferred.")

            if not model_name:
                model_name = model.__class__.__name__

            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model_artifact",
                signature=signature,
                input_example=input_data,
                # registered_model_name=model_name,
            )
            logger.info(f"Model logged with name: {model_name}")
            return model_info

    except Exception as e:
        logger.error(f"Error logging to MLflow: {e}")
        raise
