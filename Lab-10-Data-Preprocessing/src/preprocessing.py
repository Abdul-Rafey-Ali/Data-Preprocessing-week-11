"""Lab 10 — Data Preprocessing (Pandas)

This script performs common preprocessing steps needed before training ML models:
- Load CSV
- Print shape (rows/columns)
- Check missing values
- Example: show unique values for a selected column
- Fill missing values (mode)
- Drop unnecessary columns (id if present)
- Convert datatypes to integers where possible
- Split into X (features) and y (target)
- Convert object/categorical columns into integer codes
- Save cleaned dataset

Project expects:
- data/dataset.csv

Output:
- outputs/cleaned_dataset.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COL = "loan_status"  # last column in the provided dataset


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}. "
            "Place your dataset at data/dataset.csv"
        )
    # some datasets may contain rows with extra commas when fields are missing.
    # we handle this safely by letting pandas re-try with the python engine.
    # Use a flexible parser to handle occasional malformed rows.
    # If rows contain extra commas, we skip them instead of crashing.
    return pd.read_csv(
        csv_path,
        engine="python",
        on_bad_lines="skip",
    )




def export_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def print_basic_info(df: pd.DataFrame) -> None:
    print(f"We have {df.shape[0]} rows.")
    print(f"We have {df.shape[1]} columns")


def check_missing_values(df: pd.DataFrame) -> None:
    # Lab requirement: np.sum(pd.isnull(data))
    total_missing = int(np.sum(pd.isnull(df)))
    print("Total missing values:", total_missing)

    missing_by_col = df.isnull().sum()
    missing_cols = missing_by_col[missing_by_col > 0]
    print("Missing by column (only non-zero):")
    print(missing_cols if len(missing_cols) else "No missing values detected.")


def show_unique_values(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        print(f"Column '{column}' not found. Skipping unique values check.")
        return

    # Lab requirement: data['columnname'].unique()
    print(f"Unique values in '{column}':")
    print(df[column].unique())


def fill_missing_values_mode(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using mode per column.

    - For numeric columns, mode often works (it yields the most common value).
    - For categorical columns, mode is usually the correct choice.
    """

    df = df.copy()

    for col in df.columns:
        if df[col].isnull().any():
            mode_series = df[col].mode(dropna=True)
            if len(mode_series) == 0:
                # Edge case: column is entirely null
                continue
            df[col] = df[col].fillna(mode_series.iloc[0])

    return df


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Lab requirement: drop id column if present
    if "id" in df.columns:
        df.drop("id", axis=1, inplace=True)
    return df


def convert_numeric_columns_to_int(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to int where safe.

    We convert float-like integer columns (e.g., 12.0 -> 12) to int.
    """

    df = df.copy()

    for col in df.columns:
        # Skip target column conversion here; we handle numeric conversion after factorization if needed
        if df[col].dtype == "float64":
            # If all non-null values are integer-valued, convert
            as_int = df[col].dropna().astype(np.int64)
            if np.allclose(as_int.astype(float), df[col].dropna().values):
                df[col] = df[col].astype(np.int64)

    return df


def split_x_y(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset")

    # Lab requirement:
    # x = data.iloc[:, 0:-1]
    # y = data.iloc[:,-1]
    # But we use target_col explicitly for safety.

    y = df[target_col]
    x = df.drop(columns=[target_col])

    print("X shape:", x.shape)
    print("y shape:", y.shape)

    return x, y


def convert_object_columns_to_int_codes(x: pd.DataFrame) -> pd.DataFrame:
    """Convert object columns into integer codes using factorize()."""

    x = x.copy()

    # Lab requirement:
    # cat_columns = x.select_dtypes(['object']).columns
    cat_columns = x.select_dtypes(["object"]).columns

    if len(cat_columns):
        x[cat_columns] = x[cat_columns].apply(lambda s: pd.factorize(s)[0])

    return x


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    # Prefer GitHub structure: data/dataset.csv
    csv_path = repo_root / "data" / "dataset.csv"

    data = load_data(csv_path)

    # 1) Reading csv file done above

    # 2) Print rows and columns / shape
    print_basic_info(data)

    # 3) Checking null values
    check_missing_values(data)

    # 4) Checking unique values of a specific column (education as per example)
    # If your dataset doesn't have 'education', we fall back to any categorical column.
    if "education" in data.columns:
        show_unique_values(data, "education")
    else:
        # pick first object column
        obj_cols = data.select_dtypes(["object"]).columns
        if len(obj_cols):
            show_unique_values(data, str(obj_cols[0]))

    # 5) Filling null values (mode)
    data = fill_missing_values_mode(data)

    # 6-8) Type conversion to int + dropping unnecessary columns + check dtypes
    data = drop_unnecessary_columns(data)

    print("\nData types before numeric conversion:")
    print(data.dtypes)

    # Convert float->int when safe
    data = convert_numeric_columns_to_int(data)

    # 8) Check datatypes of all columns
    print("\nData types after numeric conversion:")
    print(data.dtypes)

    # 9) Splitting into X and y
    x, y = split_x_y(data, TARGET_COL)

    # 10) Converting Object columns into Int codes
    x = convert_object_columns_to_int_codes(x)

    # Rebuild a cleaned dataset with encoded features + original y
    cleaned = x.copy()
    cleaned[TARGET_COL] = y.values

    # Save outputs
    out_path = repo_root / "outputs" / "cleaned_dataset.csv"
    export_csv(out_path, cleaned)

    print(f"\nCleaned dataset saved to: {out_path}")


if __name__ == "__main__":
    main()

