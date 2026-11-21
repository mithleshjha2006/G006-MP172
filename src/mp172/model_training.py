
import os

# Limit native library threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from threadpoolctl import threadpool_limits

from mp172.data_cleaning import clean_dataset
from mp172.mlflow_tracking import (
    configure_mlflow,
    REGISTERED_MODEL_NAME
)


# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data/processed/train_cleaned.csv"
LATER_PATH = ROOT / "data/raw/later.csv"

TARGET = "rent_rs"

NUMERIC_FEATURES = [
    "distance_to_campus_km",
    "room_area_sqft",
    "sharing",
    "meals_per_day",
    "building_age_yr",
    "wifi_mbps",
    "date_ordinal",
    "month_sin",
    "month_cos",
    "weekday_sin",
    "weekday_cos"
]

CATEGORICAL_FEATURES = [
    "furnishing",
    "locality"
]


# ==================================================
# DATA QUALITY
# ==================================================

def remove_suspicious_targets(df):

    df = df.copy()

    target = pd.to_numeric(df[TARGET], errors="coerce")

    agreement = pd.to_numeric(
        df["rent_in_signed_agreement"],
        errors="coerce"
    ).replace(0, np.nan)

    suspicious = (target / agreement) > 5
    suspicious = suspicious.fillna(False)

    return df.loc[~suspicious].copy(), int(suspicious.sum())


# ==================================================
# FEATURE ENGINEERING
# ==================================================

def build_features(df):

    dates = pd.to_datetime(
        df["date"].astype(str),
        format="%Y-%m-%d",
        errors="coerce"
    )

    if dates.isna().any():
        raise ValueError("Invalid dates found.")

    features = df[
        [
            "distance_to_campus_km",
            "room_area_sqft",
            "sharing",
            "meals_per_day",
            "building_age_yr",
            "wifi_mbps",
            "furnishing",
            "locality"
        ]
    ].copy()

    # Date trend
    features["date_ordinal"] = dates.map(
        lambda date: date.toordinal()
    )

    # Cyclical month features
    features["month_sin"] = np.sin(
        2 * np.pi * dates.dt.month / 12
    )

    features["month_cos"] = np.cos(
        2 * np.pi * dates.dt.month / 12
    )

    # Cyclical weekday features
    features["weekday_sin"] = np.sin(
        2 * np.pi * dates.dt.dayofweek / 7
    )

    features["weekday_cos"] = np.cos(
        2 * np.pi * dates.dt.dayofweek / 7
    )

    features[NUMERIC_FEATURES] = (
        features[NUMERIC_FEATURES].astype("float64")
    )

    return features


# ==================================================
# PREPROCESSING
# ==================================================

def build_preprocessor(scale_numeric=False):

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale_numeric:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    return ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            NUMERIC_FEATURES
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        )
    ])


def build_pipeline(model, scale_numeric=False):

    return Pipeline([
        (
            "preprocessor",
            build_preprocessor(scale_numeric)
        ),
        ("model", model)
    ])


# ==================================================
# MODEL CANDIDATES
# ==================================================

def get_models():

    return [
        (
            "ridge_alpha_1",
            build_pipeline(
                Ridge(alpha=1.0),
                scale_numeric=True
            )
        ),
        (
            "ridge_alpha_10",
            build_pipeline(
                Ridge(alpha=10.0),
                scale_numeric=True
            )
        ),
        (
            "ridge_alpha_100",
            build_pipeline(
                Ridge(alpha=100.0),
                scale_numeric=True
            )
        ),
        (
            "random_forest",
            build_pipeline(
                RandomForestRegressor(
                    n_estimators=400,
                    max_depth=None,
                    min_samples_leaf=3,
                    max_features=0.8,
                    random_state=42,
                    n_jobs=1
                )
            )
        ),
        (
            "extra_trees",
            build_pipeline(
                ExtraTreesRegressor(
                    n_estimators=500,
                    min_samples_leaf=3,
                    max_features=0.8,
                    random_state=42,
                    n_jobs=1
                )
            )
        ),
        (
            "gradient_boosting",
            build_pipeline(
                GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.03,
                    max_depth=2,
                    min_samples_leaf=5,
                    loss="huber",
                    random_state=42
                )
            )
        )
    ]


# ==================================================
# EVALUATION
# ==================================================

def evaluate_model(y_true, predictions):

    return {
        "mae": float(
            mean_absolute_error(y_true, predictions)
        ),
        "rmse": float(
            np.sqrt(mean_squared_error(y_true, predictions))
        ),
        "r2": float(
            r2_score(y_true, predictions)
        )
    }


# ==================================================
# TRAINING AND BENCHMARKING
# ==================================================

def train():

    print("\nSTEP 1: Loading data...", flush=True)

    train_df = pd.read_csv(TRAIN_PATH)
    valid_df = pd.read_csv(LATER_PATH)

    print("Training records:", len(train_df))
    print("Validation records:", len(valid_df))

    print("\nSTEP 2: Cleaning validation data...", flush=True)

    valid_df, validation_errors = clean_dataset(valid_df)

    print("Validation errors:", validation_errors)

    print("\nSTEP 3: Removing suspicious targets...", flush=True)

    train_df, train_removed = remove_suspicious_targets(train_df)
    valid_df, valid_removed = remove_suspicious_targets(valid_df)

    print("Training records removed:", train_removed)
    print("Validation records removed:", valid_removed)

    print("\nSTEP 4: Target distribution...", flush=True)

    print("\nTraining target:")
    print(train_df[TARGET].describe())

    print("\nValidation target:")
    print(valid_df[TARGET].describe())

    X_train = build_features(train_df)
    y_train = train_df[TARGET]

    X_valid = build_features(valid_df)
    y_valid = valid_df[TARGET]

    print("\nTraining features:", X_train.shape)
    print("Validation features:", X_valid.shape)

    print("\nSTEP 5: Configure MLflow...", flush=True)

    configure_mlflow()

    results = []

    # ------------------------------------------------
    # BASELINE MODEL
    # ------------------------------------------------

    print("\nSTEP 6: Training baseline...", flush=True)

    baseline = DummyRegressor(strategy="mean")

    with threadpool_limits(limits=1):

        baseline.fit(X_train, y_train)
        baseline_predictions = baseline.predict(X_valid)

    baseline_metrics = evaluate_model(
        y_valid,
        baseline_predictions
    )

    print("\nBASELINE RESULTS")
    print(baseline_metrics)

    # ------------------------------------------------
    # MODEL BENCHMARKING
    # ------------------------------------------------

    with mlflow.start_run(
        run_name="model-benchmark"
    ) as parent_run:

        mlflow.log_params({
            "validation_strategy": "chronological",
            "target": TARGET,
            "training_rows": len(X_train),
            "validation_rows": len(X_valid),
            "training_suspicious_removed": train_removed,
            "validation_suspicious_removed": valid_removed
        })

        mlflow.log_metrics({
            "baseline_mae": baseline_metrics["mae"],
            "baseline_rmse": baseline_metrics["rmse"],
            "baseline_r2": baseline_metrics["r2"]
        })

        models = get_models()

        for model_name, model in models:

            print(
                f"\nTraining {model_name}...",
                flush=True
            )

            with mlflow.start_run(
                run_name=model_name,
                nested=True
            ) as child_run:

                with threadpool_limits(limits=1):

                    model.fit(X_train, y_train)

                    predictions = model.predict(X_valid)

                metrics = evaluate_model(
                    y_valid,
                    predictions
                )

                mlflow.log_param("model_name", model_name)

                mlflow.log_metrics({
                    "validation_mae": metrics["mae"],
                    "validation_rmse": metrics["rmse"],
                    "validation_r2": metrics["r2"]
                })

                result = {
                    "name": model_name,
                    "model": model,
                    "run_id": child_run.info.run_id,
                    **metrics
                }

                results.append(result)

                print(model_name, metrics)

        # ------------------------------------------------
        # COMPARE RESULTS
        # ------------------------------------------------

        print("\nSTEP 7: MODEL COMPARISON")

        comparison = pd.DataFrame([
            {
                "Model": "Dummy baseline",
                "MAE": baseline_metrics["mae"],
                "RMSE": baseline_metrics["rmse"],
                "R2": baseline_metrics["r2"]
            }
        ] + [
            {
                "Model": result["name"],
                "MAE": result["mae"],
                "RMSE": result["rmse"],
                "R2": result["r2"]
            }
            for result in results
        ])

        comparison = comparison.sort_values(
            "MAE"
        ).reset_index(drop=True)

        print(comparison.to_string(index=False))

        mlflow.log_dict(
            comparison.to_dict(orient="records"),
            "benchmark_results.json"
        )

        # Select by validation MAE
        best = min(
            results,
            key=lambda result: result["mae"]
        )

        print("\nBEST MODEL:", best["name"])

        print("MAE:", best["mae"])
        print("RMSE:", best["rmse"])
        print("R2:", best["r2"])

        # ------------------------------------------------
        # COMPARE AGAINST BASELINE
        # ------------------------------------------------

        beats_baseline = (
            best["mae"] < baseline_metrics["mae"]
            and best["r2"] > baseline_metrics["r2"]
        )

        mlflow.log_metrics({
            "best_model_mae": best["mae"],
            "best_model_rmse": best["rmse"],
            "best_model_r2": best["r2"]
        })

        mlflow.log_param(
            "selected_model",
            best["name"]
        )

        # Do not register a model that fails
        # to outperform the simple baseline.
        if not beats_baseline:

            print(
                "\nNo candidate beat the baseline on both MAE and R2."
            )

            print(
                "Model registration skipped. "
                "Investigate possible temporal distribution shift."
            )

            mlflow.set_tag(
                "model_selection_status",
                "no_candidate_better_than_baseline"
            )

            return

        # ------------------------------------------------
        # REGISTER BEST MODEL
        # ------------------------------------------------

        print("\nSTEP 8: Registering best model...", flush=True)

        best_predictions = best["model"].predict(X_valid)

        signature = infer_signature(
            X_train,
            best["model"].predict(X_train)
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=best["model"],
            name="best_model",
            signature=signature,
            input_example=X_train.head(5),
            serialization_format="skops",
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree"
            ]
        )

        model_version = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=REGISTERED_MODEL_NAME
        )

        client = MlflowClient()

        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias="candidate",
            version=model_version.version
        )

        print("\nMODEL REGISTERED SUCCESSFULLY")
        print("Model:", REGISTERED_MODEL_NAME)
        print("Version:", model_version.version)
        print("Alias: candidate")

        mlflow.set_tag(
            "model_selection_status",
            "registered"
        )

    print("\nBENCHMARKING COMPLETED!")


if __name__ == "__main__":
    train()
