
import pandas as pd


def clean_furnishing(df):
    """Standardize furnishing categories without modifying the input."""
    cleaned = df.copy()

    mapping = {
        "sem.": "semi",
        "ful.": "full",
        "unf.": "unfurnished",
    }

    cleaned["furnishing"] = (
        cleaned["furnishing"]
        .astype("string")
        .str.strip()
        .str.lower()
        .replace(mapping)
    )

    return cleaned

def clean_sharing(df):
    """Replace the invalid sharing sentinel with a missing value."""
    cleaned = df.copy()

    cleaned["sharing"] = pd.to_numeric(
        cleaned["sharing"], errors="coerce"
    )

    cleaned["sharing"] = cleaned["sharing"].replace(
        9999, pd.NA
    )

    return cleaned

def remove_duplicates(df):
    """Remove duplicate records without modifying the original dataset."""
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates()
    cleaned = cleaned.reset_index(drop=True)
    return cleaned

def remove_missing_essential(df):
    """Remove records missing essential fields."""
    cleaned = df.copy()

    essential_columns = ["record_id", "date", "rent_rs"]

    # Remove records containing missing essential values.
    cleaned = cleaned.dropna(subset=essential_columns)

    # Remove records with blank record IDs or dates.
    for col in ["record_id", "date"]:
        cleaned = cleaned[
            cleaned[col].astype("string").str.strip().ne("").fillna(False)
        ]

    return cleaned.reset_index(drop=True)

def validate_numeric_data(df):
    """Identify invalid numerical values without changing the input."""
    errors = {}

    if "distance_to_campus_km" in df.columns:
        distance = pd.to_numeric(
            df["distance_to_campus_km"], errors="coerce"
        )
        errors["negative_distance"] = int((distance < 0).sum())

    if "wifi_mbps" in df.columns:
        wifi = pd.to_numeric(
            df["wifi_mbps"], errors="coerce"
        )
        errors["wifi_above_limit"] = int((wifi > 550).sum())

    return errors

def clean_dataset(df):
    """Apply the basic cleaning steps to a dataset."""
    cleaned = remove_duplicates(df)
    cleaned = remove_missing_essential(cleaned)
    cleaned = clean_furnishing(cleaned)
    cleaned = clean_sharing(cleaned)

    validation_errors = validate_numeric_data(cleaned)

    return cleaned, validation_errors

def main():
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / "data" / "raw" / "train.csv"

    print("Loading training dataset...")
    train = pd.read_csv(input_path)

    print("Original rows:", len(train))

    cleaned, errors = clean_dataset(train)

    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "train_cleaned.csv"
    cleaned.to_csv(output_path, index=False)

    print("Cleaned dataset saved:", output_path)


    print("Cleaned rows:", len(cleaned))
    print("Removed rows:", len(train) - len(cleaned))

    print("\nNumerical validation:")
    print(errors)

    print("\nMissing values:")
    print(cleaned.isna().sum())

    print("\nFurnishing categories:")
    print(cleaned["furnishing"].value_counts(dropna=False))

    print("\nRemaining sharing sentinel count:")
    print((cleaned["sharing"] == 9999).sum())


if __name__ == "__main__":
    main()

   