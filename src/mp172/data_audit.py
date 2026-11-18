from pathlib import Path
from datetime import date
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

train = pd.read_csv(ROOT / "data/raw/train.csv")
later = pd.read_csv(ROOT / "data/raw/later.csv")
holdout = pd.read_csv(ROOT / "data/raw/holdout_features.csv")

numeric_columns = [
    "distance_to_campus_km",
    "room_area_sqft",
    "sharing",
    "meals_per_day",
    "building_age_yr",
    "wifi_mbps"
]

print("Train shape:", train.shape)
print("Later shape:", later.shape)
print("Holdout shape:", holdout.shape)


print("\n--- 1. LEAKING COLUMN ---")

missing_from_holdout = sorted(
    set(train.columns) - set(holdout.columns) - {"rent_rs"}
)

ratio = train["rent_rs"] / train["rent_in_signed_agreement"]
valid_rows = ratio <= 5

for col in missing_from_holdout:
    correlation = train.loc[
        valid_rows, [col, "rent_rs"]
    ].corr().iloc[0, 1]

    print("Column:", col)
    print("Correlation:", correlation)


print("\n--- 2. UNIT MIXUP ---")

for col in numeric_columns:
    data = train

    if col == "sharing":
        data = train[train[col] != 9999]

    print("\nColumn:", col)

    stats = data.groupby("locality")[col].agg(
        ["min", "mean", "max"]
    )

    print(stats)


print("\n--- 3. SENTINEL VALUE ---")

print(train["sharing"].value_counts())

print(
    "9999 count:",
    (train["sharing"] == 9999).sum()
)


print("\n--- 4. DUPLICATE ROWS ---")

print("Duplicate rows:", train.duplicated().sum())

print(
    "Duplicate record IDs:",
    train["record_id"].duplicated().sum()
)


print("\n--- 5. STUCK SENSOR ---")

records = train[
    ["date", "record_id", "locality"] + numeric_columns
].to_dict(orient="records")

best_runs = {}

for locality in train["locality"].unique():

    local_records = sorted(
        [r for r in records if r["locality"] == locality],
        key=lambda r: (r["date"], r["record_id"])
    )

    for col in numeric_columns:

        run_start = 0

        for i in range(1, len(local_records) + 1):

            if (
                i == len(local_records)
                or local_records[i][col] != local_records[run_start][col]
            ):

                count = i - run_start

                start = date.fromisoformat(
                    local_records[run_start]["date"]
                )

                end = date.fromisoformat(
                    local_records[i - 1]["date"]
                )

                days = (end - start).days + 1

                if (
                    col not in best_runs
                    or days > best_runs[col]["days"]
                ):
                    best_runs[col] = {
                        "locality": locality,
                        "value": local_records[run_start][col],
                        "rows": count,
                        "days": days,
                        "start": start,
                        "end": end
                    }

                run_start = i

for col, result in best_runs.items():
    print(
        col,
        "Locality:", result["locality"],
        "Value:", result["value"],
        "Rows:", result["rows"],
        "Days:", result["days"],
        "Start:", result["start"],
        "End:", result["end"]
    )

longest_run = max(
    best_runs.values(),
    key=lambda x: x["days"]
)

print("Longest run:", longest_run["days"], "days")
print("Potential stuck sensor:", longest_run["days"] >= 14)


print("\n--- 6. TARGET TYPOS ---")

target_typos = ratio > 5

print("Target typo count:", target_typos.sum())

print(
    train.loc[
        target_typos,
        [
            "record_id",
            "rent_in_signed_agreement",
            "rent_rs"
        ]
    ].to_string(index=False)
)


print("\n--- 7. SPELLING VARIANTS ---")

print("Original furnishing values:")
print(train["furnishing"].value_counts())

mapping = {
    "sem.": "semi",
    "ful.": "full",
    "unf.": "unfurnished"
}

clean_furnishing = (
    train["furnishing"]
    .astype(str)
    .str.strip()
    .str.lower()
    .replace(mapping)
)

print("\nNormalized furnishing values:")
print(clean_furnishing.value_counts())

print("True levels:", clean_furnishing.nunique())

print("\nLater dataset furnishing values:")
print(later["furnishing"].value_counts())

print("\n--- AUDIT COMPLETED ---")