"""Generate the final holdout predictions for MP172.

Run from the repository root:

    # On the machine where the champion MLflow artifact is available:
    PYTHONPATH=src python -m mp172.holdout_prediction --export-champion

    # On Rajvansh's machine, after pulling models/model.joblib:
    PYTHONPATH=src python -m mp172.holdout_prediction

The output CSV contains exactly: record_id,prediction
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from mlflow.tracking import MlflowClient

from mp172.data_cleaning import clean_furnishing, clean_sharing
from mp172.model_training import build_features
from mp172.mlflow_tracking import REGISTERED_MODEL_NAME


# ---------------------------------------------------------------------------
# Paths and model registry configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_PATH = ROOT / "data" / "raw" / "holdout_features.csv"
MODEL_DIR = ROOT / "models"
PORTABLE_MODEL_PATH = MODEL_DIR / "model.joblib"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"
PREDICTIONS_DIR = ROOT / "predictions"
PREDICTIONS_PATH = PREDICTIONS_DIR / "holdout_predictions.csv"

MLFLOW_DB_PATH = ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.resolve()}"
MODEL_ALIAS = "champion"
EXPECTED_CHAMPION_VERSION = 3

EXPECTED_OUTPUT_COLUMNS = ["record_id", "prediction"]


# ---------------------------------------------------------------------------
# MLflow: export the champion model to a portable repository file
# ---------------------------------------------------------------------------
def configure_mlflow() -> None:
    """Configure this checkout's local MLflow tracking and model registry."""
    if not MLFLOW_DB_PATH.exists():
        raise FileNotFoundError(
            f"MLflow database not found: {MLFLOW_DB_PATH}. "
            "On this machine, use the committed models/model.joblib instead; "
            "do not create a new empty MLflow database."
        )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_registry_uri(MLFLOW_TRACKING_URI)


def export_champion_model() -> tuple[Any, dict[str, Any]]:
    """Load @champion from MLflow and export it as models/model.joblib.

    Run this on the machine where the Stage 4 local artifact directory exists.
    The exported joblib file can then be committed so another teammate can
    predict without access to the original machine's absolute artifact path.
    """
    configure_mlflow()
    client = MlflowClient()

    try:
        champion_version = client.get_model_version_by_alias(
            name=REGISTERED_MODEL_NAME,
            alias=MODEL_ALIAS,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not resolve MLflow alias @{MODEL_ALIAS} for "
            f"{REGISTERED_MODEL_NAME}. Verify that the champion alias exists."
        ) from exc

    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
    print(f"Loading champion from MLflow: {model_uri}")
    print(f"Champion registry version: {champion_version.version}")
    if int(champion_version.version) != EXPECTED_CHAMPION_VERSION:
        raise RuntimeError(
            f"Expected promoted champion registry version "
            f"{EXPECTED_CHAMPION_VERSION}, but @{MODEL_ALIAS} points to "
            f"version {champion_version.version}. Refusing to export the wrong model."
        )

    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as exc:
        raise RuntimeError(
            "The MLflow champion alias was found, but its model artifact "
            "could not be loaded. This commonly means the local artifact "
            "directory is missing or its absolute path belongs to another "
            "computer. Run this export step on the machine where "
            "mlartifacts_stage4/ is available, then commit models/model.joblib."
        ) from exc

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    temporary_model_path = MODEL_DIR / "model.joblib.tmp"
    joblib.dump(model, temporary_model_path)
    temporary_model_path.replace(PORTABLE_MODEL_PATH)

    expected_columns = get_expected_model_columns(model)
    metadata = {
        "registered_model_name": REGISTERED_MODEL_NAME,
        "alias": MODEL_ALIAS,
        "registry_version": int(champion_version.version),
        "run_id": champion_version.run_id,
        "model_uri": model_uri,
        "model_file": str(PORTABLE_MODEL_PATH.relative_to(ROOT)),
        "expected_input_columns": expected_columns,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
    }

    MODEL_METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Portable champion saved: {PORTABLE_MODEL_PATH}")
    print(f"Model metadata saved: {MODEL_METADATA_PATH}")
    print(f"Exported registry version: {champion_version.version}")

    return model, metadata


# ---------------------------------------------------------------------------
# Model input schema and date feature compatibility
# ---------------------------------------------------------------------------
def get_expected_model_columns(model: Any) -> list[str]:
    """Read the exact dataframe columns expected by the fitted model."""
    expected = getattr(model, "feature_names_in_", None)

    if expected is None:
        named_steps = getattr(model, "named_steps", {})
        preprocessor = named_steps.get("preprocessor")
        expected = getattr(preprocessor, "feature_names_in_", None)

    if expected is None:
        raise ValueError(
            "Could not determine the model's expected input columns. "
            "The champion should be the fitted sklearn pipeline registered "
            "by the MP172 project."
        )

    return [str(column) for column in expected]


def build_features_for_model(holdout: pd.DataFrame, model: Any) -> pd.DataFrame:
    """Build the same input schema used when the champion was trained.

    The saved v1/v2 pipeline expects month, day_of_week, day_of_month and
    is_weekend. The project's shared build_features() function also creates
    date-derived cyclical features; we add the expected date columns and then
    select the exact fitted model column names and order.
    """
    try:
        features = build_features(holdout)
    except Exception as exc:
        raise ValueError(
            "Could not build model features from holdout_features.csv. "
            "Check that the file contains the model's required raw fields "
            "and a valid YYYY-MM-DD date column."
        ) from exc

    dates = pd.to_datetime(holdout["date"], errors="raise")
    date_features = {
        "month": dates.dt.month.astype("int64"),
        "day_of_month": dates.dt.day.astype("int64"),
        "day_of_week": dates.dt.dayofweek.astype("int64"),
        "is_weekend": (dates.dt.dayofweek >= 5).astype("int64"),
    }

    for column, values in date_features.items():
        if column not in features.columns:
            features[column] = values.to_numpy()

    expected_columns = get_expected_model_columns(model)
    missing_columns = [
        column for column in expected_columns
        if column not in features.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Holdout feature engineering is missing model inputs: "
            f"{missing_columns}. Available engineered columns: "
            f"{list(features.columns)}"
        )

    # Preserve row order so predictions map to the original record IDs.
    return features.loc[:, expected_columns].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Load portable model on Rajvansh's machine
# ---------------------------------------------------------------------------
def load_portable_champion() -> tuple[Any, dict[str, Any]]:
    """Load the portable model file; export from MLflow if it is not present."""
    if not PORTABLE_MODEL_PATH.exists():
        print(
            "Portable model is not present. Attempting to export @champion "
            "from this machine's MLflow registry..."
        )
        return export_champion_model()

    print(f"Loading portable champion: {PORTABLE_MODEL_PATH}")
    model = joblib.load(PORTABLE_MODEL_PATH)

    if MODEL_METADATA_PATH.exists():
        metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    else:
        metadata = {
            "registered_model_name": REGISTERED_MODEL_NAME,
            "alias": MODEL_ALIAS,
            "registry_version": None,
            "note": "Model metadata sidecar was not found.",
        }

    print("Champion metadata:", metadata)
    return model, metadata


# ---------------------------------------------------------------------------
# Holdout prediction and output validation
# ---------------------------------------------------------------------------
def generate_holdout_predictions(model: Any) -> pd.DataFrame:
    """Predict every holdout record and return exactly two output columns."""
    if not HOLDOUT_PATH.exists():
        raise FileNotFoundError(f"Holdout file not found: {HOLDOUT_PATH}")

    holdout = pd.read_csv(HOLDOUT_PATH)
    if holdout.empty:
        raise ValueError("holdout_features.csv contains no records.")

    if "record_id" not in holdout.columns:
        raise ValueError("holdout_features.csv must contain record_id.")

    if holdout["record_id"].isna().any():
        raise ValueError("holdout_features.csv contains missing record_id values.")

    if "rent_rs" in holdout.columns:
        raise ValueError(
            "The holdout feature file unexpectedly contains rent_rs. "
            "Do not use target values when generating holdout predictions."
        )

    # Apply row-preserving feature cleaning used by the training pipeline.
    # Do not call clean_dataset() here because it can drop records; the assignment
    # requires one prediction for every holdout record.
    holdout_for_features = clean_furnishing(holdout.copy())
    holdout_for_features = clean_sharing(holdout_for_features)

    X_holdout = build_features_for_model(holdout_for_features, model)
    predictions = np.asarray(model.predict(X_holdout), dtype=float).reshape(-1)

    if len(predictions) != len(holdout):
        raise RuntimeError(
            f"Model returned {len(predictions)} predictions for "
            f"{len(holdout)} holdout records."
        )

    if not np.isfinite(predictions).all():
        raise ValueError("Model produced NaN or infinite predictions.")

    output = pd.DataFrame({
        "record_id": holdout["record_id"].reset_index(drop=True),
        "prediction": predictions,
    })

    # Enforce the assignment's exact output schema and row count.
    output = output[EXPECTED_OUTPUT_COLUMNS]
    if list(output.columns) != EXPECTED_OUTPUT_COLUMNS:
        raise AssertionError("Output columns do not match the required schema.")
    if len(output) != len(holdout):
        raise AssertionError("Output does not contain one row per holdout record.")

    return output


def save_and_validate_predictions(output: pd.DataFrame) -> None:
    """Write and re-read the CSV to validate its exact schema and contents."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(PREDICTIONS_PATH, index=False)

    saved = pd.read_csv(PREDICTIONS_PATH)
    if list(saved.columns) != EXPECTED_OUTPUT_COLUMNS:
        raise AssertionError(
            f"Saved CSV columns are {list(saved.columns)}, expected "
            f"{EXPECTED_OUTPUT_COLUMNS}."
        )
    if len(saved) != len(output):
        raise AssertionError("Saved prediction count does not match holdout row count.")
    if saved["record_id"].isna().any() or saved["prediction"].isna().any():
        raise AssertionError("Saved prediction CSV contains missing values.")
    if not np.isfinite(saved["prediction"].to_numpy(dtype=float)).all():
        raise AssertionError("Saved prediction CSV contains non-finite predictions.")

    print(f"Predictions saved: {PREDICTIONS_PATH}")
    print("Output columns:", list(saved.columns))
    print("Prediction rows:", len(saved))
    print("Prediction preview:")
    print(saved.head(10).to_string(index=False))


# ---------------------------------------------------------------------------
# Main / CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the MLflow champion and generate MP172 holdout predictions."
    )
    parser.add_argument(
        "--export-champion",
        action="store_true",
        help="Export MLflow @champion to models/model.joblib and exit.",
    )
    args = parser.parse_args()

    if args.export_champion:
        export_champion_model()
        return

    model, metadata = load_portable_champion()
    registry_version = metadata.get("registry_version")
    if registry_version is not None:
        print(f"Using champion registry version: {registry_version}")

    output = generate_holdout_predictions(model)
    save_and_validate_predictions(output)

    print("\nHOLDOUT PREDICTIONS COMPLETED")
    print("Input records:", len(output))
    print("Output file:", PREDICTIONS_PATH.relative_to(ROOT))
    print("Required schema: record_id,prediction")


if __name__ == "__main__":
    main()
