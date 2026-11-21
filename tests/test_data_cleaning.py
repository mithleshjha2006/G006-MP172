
import pandas as pd

from src.mp172.data_cleaning import (
    clean_furnishing,
    clean_sharing,
)


def test_clean_furnishing():
    df = pd.DataFrame({
        "furnishing": [" Semi ", "FULL", "sem.", "unf."]
    })

    result = clean_furnishing(df)

    assert result["furnishing"].tolist() == [
        "semi", "full", "semi", "unfurnished"
    ]


def test_clean_sharing():
    df = pd.DataFrame({
        "sharing": [1, 2, 9999, 4]
    })

    result = clean_sharing(df)

    assert pd.isna(result.loc[2, "sharing"])
    assert result.loc[0, "sharing"] == 1
    assert result.loc[1, "sharing"] == 2
    assert result.loc[3, "sharing"] == 4

from src.mp172.data_cleaning import (
    remove_duplicates,
    remove_missing_essential,
    validate_numeric_data,
    clean_dataset,
)


def test_remove_duplicates():
    df = pd.DataFrame({
        "record_id": ["R001", "R001", "R002"],
        "rent_rs": [7000, 7000, 9000],
    })

    result = remove_duplicates(df)

    assert len(result) == 2
    assert result["record_id"].tolist() == ["R001", "R002"]


def test_remove_missing_essential():
    df = pd.DataFrame({
        "record_id": ["R001", None, "R003", "R004"],
        "date": [
            "2026-01-01",
            "2026-01-02",
            "",
            "2026-01-04",
        ],
        "rent_rs": [7000, 8000, 9000, None],
    })

    result = remove_missing_essential(df)

    assert len(result) == 1
    assert result.iloc[0]["record_id"] == "R001"


def test_validate_numeric_data():
    df = pd.DataFrame({
        "distance_to_campus_km": [2, -3, 5],
        "wifi_mbps": [100, 600, 550],
    })

    result = validate_numeric_data(df)

    assert result["negative_distance"] == 1
    assert result["wifi_above_limit"] == 1


def test_clean_dataset():
    df = pd.DataFrame({
        "record_id": ["R001", "R001", "R002"],
        "date": [
            "2026-01-01",
            "2026-01-01",
            "2026-01-02",
        ],
        "rent_rs": [7000, 7000, 8000],
        "furnishing": [" Semi ", " Semi ", "FULL"],
        "sharing": [2, 2, 9999],
        "distance_to_campus_km": [2, 2, -1],
        "wifi_mbps": [100, 100, 600],
    })

    cleaned, errors = clean_dataset(df)

    assert len(cleaned) == 2
    assert cleaned["furnishing"].tolist() == ["semi", "full"]
    assert cleaned["sharing"].isna().sum() == 1
    assert errors["negative_distance"] == 1
    assert errors["wifi_above_limit"] == 1

    # The original dataset must remain unchanged.
    assert len(df) == 3
    assert df.loc[2, "sharing"] == 9999
