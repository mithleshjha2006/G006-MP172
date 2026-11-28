"""Stage 4: drift monitoring, v2 retraining, and model promotion.

Run from the project root:
    PYTHONPATH=src python -m mp172.stage4_monitoring

The v1 model is loaded from the checked-in handoff directory because the
existing registered v1 artifact uses an mlflow-artifacts URI that cannot be
resolved directly through the project's SQLite-only tracking configuration.
"""

import hashlib
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error

from mp172.data_cleaning import clean_dataset
from mp172.model_training import (
    TARGET,
    build_features,
    remove_suspicious_targets,
)
from mp172.mlflow_tracking import REGISTERED_MODEL_NAME


# ============================================================================
# Configuration
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_RAW_PATH = ROOT / "data" / "raw" / "train.csv"
TRAIN_CLEAN_PATH = ROOT / "data" / "processed" / "train_cleaned.csv"
LATER_PATH = ROOT / "data" / "raw" / "later.csv"
V1_HANDOFF_PATH = ROOT / "handoff" / "RentalPricePredictor_champion_v1"

FINDINGS_PATH = ROOT / "reports" / "findings.yaml"
REPORT_PATH = ROOT / "reports" / "stage4_report.json"

# Keep the existing SQLite database and registry. Stage 4 gets its own
# experiment whose artifacts are stored locally rather than under
# mlflow-artifacts:/, which requires an HTTP artifact server.
TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
STAGE4_EXPERIMENT_NAME = "Rental-Price-Prediction-Stage4"
STAGE4_ARTIFACT_DIR = ROOT / "mlartifacts_stage4"

PSI_BINS = 10
EPSILON = 0.0001
LATER_TRAIN_RATIO = 0.70
PROMOTION_THRESHOLD = 0.08


# ============================================================================
# MLflow configuration
# ============================================================================

def configure_stage4_mlflow():
    """Use the existing SQLite registry and a local artifact directory."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)

    STAGE4_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(STAGE4_EXPERIMENT_NAME)

    if experiment is None:
        client.create_experiment(
            name=STAGE4_EXPERIMENT_NAME,
            artifact_location=STAGE4_ARTIFACT_DIR.resolve().as_uri(),
        )
    elif not experiment.artifact_location.startswith("file:"):
        raise RuntimeError(
            f"Experiment {STAGE4_EXPERIMENT_NAME!r} already has a non-local "
            f"artifact location: {experiment.artifact_location!r}. Change "
            "STAGE4_EXPERIMENT_NAME to a fresh experiment name."
        )

    mlflow.set_experiment(STAGE4_EXPERIMENT_NAME)
    print("MLflow tracking URI:", mlflow.get_tracking_uri())
    print("MLflow experiment:", STAGE4_EXPERIMENT_NAME)
    print("Stage 4 artifact directory:", STAGE4_ARTIFACT_DIR)


# ============================================================================
# PSI calculation and drift detection
# ============================================================================

def calculate_psi(reference, current, bins=PSI_BINS):
    """Calculate PSI using bin boundaries derived only from reference data.

    Ten quantile bins are requested. Duplicate boundaries are removed when a
    feature has too few distinct values (for example, integer count columns).
    Both datasets are evaluated against exactly the same boundaries.
    """
    reference = (
        pd.to_numeric(pd.Series(reference), errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )
    current = (
        pd.to_numeric(pd.Series(current), errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )

    if reference.size == 0 or current.size == 0:
        raise ValueError("PSI cannot be calculated for an empty feature.")

    unique_values = np.unique(reference)

    if unique_values.size == 1:
        # Keep a valid three-interval partition for a constant reference.
        value = float(unique_values[0])
        delta = max(abs(value) * 1e-6, 1e-6)
        edges = np.array([-np.inf, value - delta, value + delta, np.inf])
    elif unique_values.size <= bins:
        # For low-cardinality numeric features, split between observed values.
        midpoints = (unique_values[:-1] + unique_values[1:]) / 2.0
        edges = np.concatenate(([-np.inf], midpoints, [np.inf]))
    else:
        quantiles = np.quantile(reference, np.linspace(0.0, 1.0, bins + 1))
        internal_edges = np.unique(quantiles[1:-1])
        edges = np.concatenate(([-np.inf], internal_edges, [np.inf]))

    reference_counts = (
        pd.Series(pd.cut(reference, bins=edges, include_lowest=True))
        .value_counts(sort=False)
        .to_numpy(dtype=float)
    )
    current_counts = (
        pd.Series(pd.cut(current, bins=edges, include_lowest=True))
        .value_counts(sort=False)
        .to_numpy(dtype=float)
    )

    actual_bin_count = len(edges) - 1
    reference_pct = (reference_counts + EPSILON) / (
        len(reference) + EPSILON * actual_bin_count
    )
    current_pct = (current_counts + EPSILON) / (
        len(current) + EPSILON * actual_bin_count
    )

    psi = np.sum(
        (current_pct - reference_pct)
        * np.log(current_pct / reference_pct)
    )
    return float(psi)


def detect_drift(train, later):
    """Calculate PSI for every numeric column shared by train and later.

    This intentionally includes numeric target/leakage columns because the
    assignment requests PSI for every numeric column. Those columns are not
    supplied to the prediction model as features.
    """
    numeric_columns = [
        column
        for column in train.select_dtypes(include=[np.number]).columns
        if column in later.columns
        and pd.api.types.is_numeric_dtype(later[column])
    ]
    if not numeric_columns:
        raise ValueError("No shared numeric columns found for PSI monitoring.")

    print(f"\nSTEP 2: PSI drift detection ({PSI_BINS} requested bins)")
    psi_results = {}
    for column in numeric_columns:
        value = calculate_psi(train[column], later[column], bins=PSI_BINS)
        psi_results[column] = value
        print(f"{column}: {value:.6f}")

    most_drifted = max(psi_results, key=psi_results.get)
    print("Numeric columns monitored:", len(numeric_columns))
    print("Most drifted column:", most_drifted)
    print("Highest PSI:", psi_results[most_drifted])
    return psi_results, most_drifted


# ============================================================================
# Dataset loading and model-compatible feature engineering
# ============================================================================

def load_data():
    print("\nSTEP 1: Loading datasets")
    if not TRAIN_CLEAN_PATH.exists():
        raise FileNotFoundError(f"Cleaned training data not found: {TRAIN_CLEAN_PATH}")
    if not LATER_PATH.exists():
        raise FileNotFoundError(f"Later dataset not found: {LATER_PATH}")

    train = pd.read_csv(TRAIN_CLEAN_PATH)
    later_raw = pd.read_csv(LATER_PATH)
    later_clean, cleaning_errors = clean_dataset(later_raw)

    print("Cleaned training rows:", len(train))
    print("Later rows before cleaning:", len(later_raw))
    print("Later rows after cleaning:", len(later_clean))
    print("Later data validation:", cleaning_errors)
    return train, later_clean, later_raw


def get_expected_model_columns(model):
    """Read the input feature names recorded by the fitted v1 pipeline."""
    expected = getattr(model, "feature_names_in_", None)

    if expected is None:
        named_steps = getattr(model, "named_steps", {})
        preprocessor = named_steps.get("preprocessor")
        expected = getattr(preprocessor, "feature_names_in_", None)

    if expected is None:
        raise ValueError(
            "Cannot determine v1's expected input columns. Inspect the saved "
            "pipeline's preprocessor and provide its original feature schema."
        )
    return list(expected)


def build_features_for_model(df, model):
    """Build features and adapt them to the exact saved v1 input schema.

    The current build_features() creates date_ordinal and cyclical date
    columns. The saved handoff v1 additionally expects month, day_of_month,
    day_of_week, and is_weekend. Add these fields, then select the exact
    expected columns in the exact order recorded by the fitted pipeline.
    """
    features = build_features(df)
    dates = pd.to_datetime(df["date"], errors="raise")

    date_features = {
        "month": dates.dt.month,
        "day_of_month": dates.dt.day,
        "day_of_week": dates.dt.dayofweek,
        "is_weekend": (dates.dt.dayofweek >= 5).astype(int),
    }
    for column, values in date_features.items():
        if column not in features.columns:
            features[column] = values.to_numpy()

    expected_columns = get_expected_model_columns(model)
    missing = [column for column in expected_columns if column not in features.columns]
    if missing:
        raise ValueError(
            f"Feature engineering still does not create v1's expected columns: {missing}. "
            f"Available columns: {list(features.columns)}"
        )

    # Match v1's exact schema and ordering. This is also used to train v2.
    return features.loc[:, expected_columns].copy()


def load_v1_model():
    print("\nSTEP 3: Loading Stage 3 model (v1)")
    if not V1_HANDOFF_PATH.exists():
        raise FileNotFoundError(f"Local v1 handoff model not found: {V1_HANDOFF_PATH}")

    client = MlflowClient()
    registered_v1 = client.get_model_version(
        name=REGISTERED_MODEL_NAME,
        version="1",
    )

    # Load the locally verified exported model. The registered v1 artifact
    # points to mlflow-artifacts:/ and needs a separate HTTP artifact server.
    model = mlflow.sklearn.load_model(str(V1_HANDOFF_PATH))
    print("Registered model version:", registered_v1.version)
    print("Loaded local v1 model:", V1_HANDOFF_PATH)
    print("Expected v1 input columns:", get_expected_model_columns(model))
    return model


# ============================================================================
# Supervised data preparation and v1 locality monitoring
# ============================================================================

def prepare_supervised_data(df, model):
    usable, suspicious_removed = remove_suspicious_targets(df)
    if usable.empty:
        raise ValueError("No usable labelled records remain after target checks.")

    features = build_features_for_model(usable, model)
    target = usable[TARGET].astype(float)
    return features, target, suspicious_removed


def evaluate_v1_by_locality(v1_model, later):
    """Evaluate v1 separately for every locality using all labelled later data."""
    print("\nSTEP 4: V1 MAE by locality on later.csv")
    usable, suspicious_removed = remove_suspicious_targets(later)
    if usable.empty:
        raise ValueError("No later records remain for locality evaluation.")

    features = build_features_for_model(usable, v1_model)
    target = usable[TARGET].astype(float)
    predictions = v1_model.predict(features)

    results = {}
    localities = sorted(usable["locality"].dropna().unique())
    for locality in localities:
        mask = usable["locality"].eq(locality).to_numpy()
        group_mae = mean_absolute_error(
            target.to_numpy()[mask],
            predictions[mask],
        )
        results[str(locality)] = float(group_mae)
        print(f"{locality}: MAE = {group_mae:.2f}")

    if not results:
        raise ValueError("No locality groups were available for evaluation.")

    worst_group = max(results, key=results.get)
    overall_mae = float(mean_absolute_error(target, predictions))
    print("Overall v1 MAE on later.csv:", overall_mae)
    print("Worst-served locality:", worst_group)
    print("Suspicious target rows excluded:", suspicious_removed)
    return results, worst_group, overall_mae


# ============================================================================
# Chronological split and v2 retraining
# ============================================================================

def chronological_split(later_raw):
    """Sort later data by date, split 70/30, then clean each partition."""
    print("\nSTEP 5: Chronological 70/30 split")
    ordered = later_raw.copy()
    ordered["_sort_date"] = pd.to_datetime(ordered["date"], errors="raise")
    ordered = ordered.sort_values(
        ["_sort_date", "record_id"],
        kind="mergesort",
    ).drop(columns=["_sort_date"]).reset_index(drop=True)

    split_index = int(len(ordered) * LATER_TRAIN_RATIO)
    if split_index <= 0 or split_index >= len(ordered):
        raise ValueError("Not enough later records for a chronological 70/30 split.")

    first_70_raw = ordered.iloc[:split_index].copy()
    last_30_raw = ordered.iloc[split_index:].copy()

    # Split before cleaning so rows cannot migrate between time partitions.
    first_70, train_errors = clean_dataset(first_70_raw)
    last_30, check_errors = clean_dataset(last_30_raw)
    if first_70.empty or last_30.empty:
        raise ValueError("Cleaning produced an empty 70% or 30% partition.")

    print("First 70% before cleaning:", len(first_70_raw))
    print("Last 30% before cleaning:", len(last_30_raw))
    print("First 70% after cleaning:", len(first_70))
    print("Last 30% after cleaning:", len(last_30))
    print("Retraining period:", first_70_raw["date"].min(), "to", first_70_raw["date"].max())
    print("Evaluation period:", last_30_raw["date"].min(), "to", last_30_raw["date"].max())
    print("Training split cleaning checks:", train_errors)
    print("Evaluation split cleaning checks:", check_errors)
    return first_70, last_30


def retrain_v2(v1_model, train, first_70):
    """Fit a clone of v1 on cleaned train plus the first 70% of later data."""
    print("\nSTEP 6: Retraining v2")
    combined = pd.concat([train, first_70], ignore_index=True)
    X_train, y_train, suspicious_removed = prepare_supervised_data(combined, v1_model)

    # Keep v1's model family and hyperparameters; refit on the expanded data.
    v2_model = clone(v1_model)
    v2_model.fit(X_train, y_train)

    print("Original cleaned training rows:", len(train))
    print("First-70% later rows after cleaning:", len(first_70))
    print("Combined rows used for training:", len(X_train))
    print("Suspicious target rows excluded:", suspicious_removed)
    print("V2 training completed.")
    return v2_model, X_train


# ============================================================================
# V1/v2 comparison on the final 30%
# ============================================================================

def compare_models(v1_model, v2_model, last_30):
    print("\nSTEP 7: Comparing v1 and v2 on final 30%")
    X_check, y_check, suspicious_removed = prepare_supervised_data(last_30, v1_model)
    if X_check.empty:
        raise ValueError("No labelled rows remain in the final 30% evaluation partition.")

    v1_predictions = v1_model.predict(X_check)
    v2_predictions = v2_model.predict(X_check)

    mae_v1 = float(mean_absolute_error(y_check, v1_predictions))
    mae_v2 = float(mean_absolute_error(y_check, v2_predictions))
    improvement = (mae_v1 - mae_v2) / mae_v1 if mae_v1 > 0 else 0.0

    # Client rule: v2's MAE must be at least 8% lower than v1's MAE.
    promote = mae_v1 > 0 and mae_v2 <= mae_v1 * (1.0 - PROMOTION_THRESHOLD)

    print(f"V1 MAE: {mae_v1:.4f}")
    print(f"V2 MAE: {mae_v2:.4f}")
    print(f"MAE improvement: {improvement * 100:.2f}%")
    print(f"Required improvement: {PROMOTION_THRESHOLD * 100:.2f}%")
    print("Decision:", "PROMOTE V2" if promote else "KEEP V1")
    print("Suspicious target rows excluded:", suspicious_removed)
    return mae_v1, mae_v2, float(improvement), promote


# ============================================================================
# Model registration and champion promotion
# ============================================================================

def register_v2(v2_model, X_train, mae_v1, mae_v2, improvement, promote):
    """Register the Stage 4 retrained model.

    The repository's registry already has MLflow version 2 from the earlier
    model-benchmark run. MLflow version numbers are monotonic and cannot be
    reused, so this logical retrained model v2 is registered as the next
    available registry version (expected to be 3 in this project).
    """
    print("\nSTEP 8: Logging and registering logical model v2")
    client = MlflowClient()

    existing_versions = client.search_model_versions(
        f"name='{REGISTERED_MODEL_NAME}'"
    )

    # Make the step safe to resume if registration succeeded but a later
    # operation (for example, updating findings.yaml) failed.
    stage4_versions = []
    for version in existing_versions:
        if not version.run_id:
            continue
        run = client.get_run(version.run_id)
        tags = run.data.tags
        if (
            tags.get("model_stage") == "stage4_retraining"
            and tags.get("logical_model_version") == "v2"
        ):
            stage4_versions.append(version)

    if stage4_versions:
        registered = max(stage4_versions, key=lambda item: int(item.version))
        registry_version = int(registered.version)
        print(
            "Found existing Stage 4 logical v2 at registry version:",
            registry_version,
        )
    else:
        original_data_hash = hashlib.sha256(
            TRAIN_RAW_PATH.read_bytes()
        ).hexdigest()
        signature = infer_signature(
            X_train,
            v2_model.predict(X_train),
        )

        with mlflow.start_run(run_name="stage4-retraining-v2"):
            mlflow.log_params({
                "data_sha256": original_data_hash,
                "logical_model_version": "v2",
                "training_strategy": "cleaned_train_plus_first_70_percent_later",
                "later_training_ratio": LATER_TRAIN_RATIO,
                "promotion_threshold": PROMOTION_THRESHOLD,
                "base_model_version": 1,
                "psi_bins": PSI_BINS,
                "mae_v1_last30": mae_v1,
                "mae_v2_last30": mae_v2,
                "mae_improvement_percent": improvement * 100.0,
            })
            mlflow.log_metrics({
                "mae_v1_last30": mae_v1,
                "mae_v2_last30": mae_v2,
                "mae_improvement_percent": improvement * 100.0,
            })
            mlflow.set_tag("model_stage", "stage4_retraining")
            mlflow.set_tag("logical_model_version", "v2")
            mlflow.set_tag(
                "promotion_decision",
                "promote_v2" if promote else "keep_v1",
            )

            model_info = mlflow.sklearn.log_model(
                sk_model=v2_model,
                name="stage4_v2",
                signature=signature,
                input_example=X_train.head(5),
                serialization_format="cloudpickle",
            )
            registered = mlflow.register_model(
                model_uri=model_info.model_uri,
                name=REGISTERED_MODEL_NAME,
            )

        registry_version = int(registered.version)
        client.set_model_version_tag(
            name=REGISTERED_MODEL_NAME,
            version=str(registry_version),
            key="model_stage",
            value="stage4_retraining",
        )
        client.set_model_version_tag(
            name=REGISTERED_MODEL_NAME,
            version=str(registry_version),
            key="logical_model_version",
            value="v2",
        )

    # The alias follows the client's promotion rule. If the rule fails, keep v1.
    champion_registry_version = registry_version if promote else 1
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=str(champion_registry_version),
    )

    print("Logical model version: v2")
    print("Actual MLflow registry version:", registry_version)
    print("Champion alias points to registry version:", champion_registry_version)
    return registry_version, champion_registry_version


# ============================================================================
# findings.yaml and report output
# ============================================================================

def update_findings(
    psi_results,
    most_drifted,
    locality_mae,
    worst_group,
    mae_v1,
    mae_v2,
    improvement,
    promote,
    v2_version,
    champion_version,
):
    print("\nSTEP 9: Updating findings.yaml")
    if FINDINGS_PATH.exists():
        with FINDINGS_PATH.open("r", encoding="utf-8") as file:
            findings = yaml.safe_load(file) or {}
    else:
        findings = {}

    monitoring = findings.setdefault("monitoring", {})
    psi_section = monitoring.setdefault("psi", {})
    for feature, value in psi_results.items():
        psi_section[feature] = float(value)

    monitoring["most_drifted_feature"] = most_drifted
    mae_section = monitoring.setdefault("mae_by_group", {})
    for locality, value in locality_mae.items():
        mae_section[locality] = float(value)

    monitoring["worst_served_group"] = worst_group
    monitoring["mae_v1_on_check"] = float(mae_v1)
    monitoring["mae_v2_on_check"] = float(mae_v2)
    monitoring["decision"] = "promote_v2" if promote else "keep_v1"
    # v2 is the logical retrained model. Its actual registry version may be
    # higher because the registry already contained an earlier version 2.
    monitoring["v2_version"] = 2
    monitoring["v2_registry_version"] = int(v2_version)
    monitoring["champion_version"] = int(champion_version)
    monitoring["mae_improvement_percent"] = float(improvement * 100.0)

    FINDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FINDINGS_PATH.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            findings,
            file,
            sort_keys=False,
            allow_unicode=True,
        )
    print("Updated:", FINDINGS_PATH)


def save_report(
    psi_results,
    most_drifted,
    locality_mae,
    worst_group,
    mae_v1,
    mae_v2,
    improvement,
    promote,
    v2_version,
    champion_version,
):
    report = {
        "psi": psi_results,
        "most_drifted_feature": most_drifted,
        "mae_by_locality": locality_mae,
        "worst_served_locality": worst_group,
        "mae_v1_last30": mae_v1,
        "mae_v2_last30": mae_v2,
        "improvement_percent": improvement * 100.0,
        "promotion_decision": "promote_v2" if promote else "keep_v1",
        "v2_version": 2,
        "v2_registry_version": v2_version,
        "champion_version": champion_version,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)
    print("Saved report:", REPORT_PATH)


# ============================================================================
# Main Stage 4 pipeline
# ============================================================================

def main():
    print("=" * 60)
    print("STAGE 4: DRIFT DETECTION AND MODEL RETRAINING")
    print("=" * 60)

    configure_stage4_mlflow()

    train, later, later_raw = load_data()
    psi_results, most_drifted = detect_drift(train, later)

    v1_model = load_v1_model()
    locality_mae, worst_group, _overall_mae = evaluate_v1_by_locality(
        v1_model,
        later,
    )

    first_70, last_30 = chronological_split(later_raw)
    v2_model, X_train = retrain_v2(v1_model, train, first_70)
    mae_v1, mae_v2, improvement, promote = compare_models(
        v1_model,
        v2_model,
        last_30,
    )

    v2_version, champion_version = register_v2(
        v2_model,
        X_train,
        mae_v1,
        mae_v2,
        improvement,
        promote,
    )

    update_findings(
        psi_results,
        most_drifted,
        locality_mae,
        worst_group,
        mae_v1,
        mae_v2,
        improvement,
        promote,
        v2_version,
        champion_version,
    )
    save_report(
        psi_results,
        most_drifted,
        locality_mae,
        worst_group,
        mae_v1,
        mae_v2,
        improvement,
        promote,
        v2_version,
        champion_version,
    )

    print("\n" + "=" * 60)
    print("STAGE 4 COMPLETED")
    print("=" * 60)
    print("Most drifted column:", most_drifted)
    print("Worst-served locality:", worst_group)
    print(f"V1 MAE on final 30%: {mae_v1:.4f}")
    print(f"V2 MAE on final 30%: {mae_v2:.4f}")
    print(f"V2 improvement: {improvement * 100:.2f}%")
    print("Promotion:", "PROMOTE V2" if promote else "KEEP V1")
    print("Champion version:", champion_version)


if __name__ == "__main__":
    main()
