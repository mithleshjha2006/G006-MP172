import os
import mlflow

EXPERIMENT_NAME = "Rental-Price-Prediction"
REGISTERED_MODEL_NAME = "RentalPricePredictor"

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def configure_mlflow(tracking_uri=None):

    uri = tracking_uri or os.getenv(
        "MLFLOW_TRACKING_URI",
        DEFAULT_TRACKING_URI
    )

    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)

    experiment = mlflow.set_experiment(EXPERIMENT_NAME)

    return experiment