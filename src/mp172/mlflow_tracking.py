
import os
import mlflow


EXPERIMENT_NAME = "Rental-Price-Prediction"

REGISTERED_MODEL_NAME = "RentalPricePredictor"

DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"


def configure_mlflow(tracking_uri=None):

    uri = tracking_uri or os.getenv(
        "MLFLOW_TRACKING_URI",
        DEFAULT_TRACKING_URI
    )

    # Configure experiment tracking
    mlflow.set_tracking_uri(uri)

    # Configure model registry
    mlflow.set_registry_uri(uri)

    # Create or select experiment
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)

    return experiment
