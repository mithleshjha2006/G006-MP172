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

def find_longest_same_value_run(df, column, *, by_duration=False):
    """Find the longest run of repeated valid readings in chronological order."""
    ordered = df.sort_values(["date", "record_id"])
    readings = ordered[column].tolist()
    dates = ordered["date"].tolist()
    best_run = None
    best_rank = (0, 0)
    run_start = 0

    for i, value in enumerate(readings):
        if pd.isna(value) or (column == "sharing" and value == 9999):
            run_start = i + 1
            continue

        if i > run_start and value != readings[i - 1]:
            run_start = i

        rows = i - run_start + 1
        if rows < 2:
            continue

        start = date.fromisoformat(dates[run_start])
        end = date.fromisoformat(dates[i])
        days = (end - start).days + 1
        rank = (days, rows) if by_duration else (rows, days)
        if rank > best_rank:
            best_rank = rank
            best_run = {
                "value": value,
                "rows": rows,
                "days": days,
                "start": start,
                "end": end,
            }

    return best_run


print("\n--- 5. STUCK SENSOR ---")

best_runs = {}
for col in numeric_columns:
    result = find_longest_same_value_run(train, col)
    if result is not None:
        best_runs[col] = result

for col, result in best_runs.items():
    print(
        col,
        "Value:", result["value"],
        "Rows:", result["rows"],
        "Days:", result["days"],
        "Start:", result["start"],
        "End:", result["end"],
    )

longest_run = max(
    best_runs.values(), key=lambda run: (run["rows"], run["days"]), default=None
)
if longest_run is None:
    print("Longest run: no repeated readings")
else:
    print("Longest run:", longest_run["days"], "days")
    print("Longest run rows:", longest_run["rows"])

duration_runs = (
    find_longest_same_value_run(train, col, by_duration=True) for col in numeric_columns
)

max_days = max((run["days"] for run in duration_runs if run is not None), default=0)
print("Maximum run duration:", max_days, "days")
print("Potential stuck sensor:", max_days >= 14)


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