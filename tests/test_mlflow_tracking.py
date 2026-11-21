
import mlflow
import mlflow.sklearn
import numpy as np

from mlflow.tracking import MlflowClient

from sklearn.linear_model import Ridge

from mp172.mlflow_tracking import (
    configure_mlflow,
    EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME
)


def test_experiment_creation(tmp_path):

    uri = f"sqlite:///{tmp_path / 'tracking.db'}"

    experiment = configure_mlflow(
        tracking_uri=uri
    )

    assert experiment.name == EXPERIMENT_NAME
    assert experiment.experiment_id is not None

    assert mlflow.get_tracking_uri() == uri


def test_parameter_and_metric_logging(tmp_path):

    uri = f"sqlite:///{tmp_path / 'tracking.db'}"

    configure_mlflow(tracking_uri=uri)

    with mlflow.start_run() as run:

        mlflow.log_param("alpha", 1.0)
        mlflow.log_param("model_type", "Ridge")

        mlflow.log_metric("validation_mae", 5170.28)
        mlflow.log_metric("validation_r2", 0.345)

        run_id = run.info.run_id

    client = MlflowClient(tracking_uri=uri)

    saved_run = client.get_run(run_id)

    assert saved_run.data.params["alpha"] == "1.0"

    assert (
        saved_run.data.params["model_type"]
        == "Ridge"
    )

    assert (
        saved_run.data.metrics["validation_mae"]
        == 5170.28
    )

    assert (
        saved_run.data.metrics["validation_r2"]
        == 0.345
    )


def test_model_registration_and_versioning(tmp_path):

    uri = f"sqlite:///{tmp_path / 'registry.db'}"

    configure_mlflow(tracking_uri=uri)

    X = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0]
    ])

    y = np.array([10.0, 20.0, 30.0, 40.0])

    model = Ridge(alpha=1.0)
    model.fit(X, y)

    model_name = REGISTERED_MODEL_NAME + "Test"

    with mlflow.start_run():

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format="skops",
            skops_trusted_types=["numpy.dtype"]
        )

        registered = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=model_name
        )

    client = MlflowClient(tracking_uri=uri)

    versions = client.search_model_versions(
        f"name='{model_name}'"
    )

    assert len(versions) >= 1

    assert registered.version is not None

    client.set_registered_model_alias(
        name=model_name,
        alias="candidate",
        version=registered.version
    )

    candidate = client.get_model_version_by_alias(
        name=model_name,
        alias="candidate"
    )

    assert candidate.version == registered.version
