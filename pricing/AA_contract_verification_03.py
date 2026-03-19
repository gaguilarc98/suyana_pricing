import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import yaml


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


REQUIRED_COLUMNS_ANOMALIES: Dict[str, str] = {
    "crop": "object",
    "year": "int64",
    "Zona": "object",
    "pixel_id": "object",
    "region": "object",
    "anomaly_sum": "float64",
    "Has": "float64",
}


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _human_readable_dtype(dtype) -> str:
    # Normalize pandas / numpy dtypes to a compact string for comparison
    if pd.api.types.is_integer_dtype(dtype):
        return "int64"
    if pd.api.types.is_float_dtype(dtype):
        return "float64"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime64"
    return "object"


def check_required_columns(df: pd.DataFrame, required: Dict[str, str]) -> List[str]:
    errors: List[str] = []

    missing = [c for c in required.keys() if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")

    unexpected = [c for c in df.columns if c not in required.keys()]
    if unexpected:
        errors.append(f"Unexpected columns present (check data_contract.md or upstream scripts): {unexpected}")

    for col, expected_dtype in required.items():
        if col not in df.columns:
            continue
        actual = _human_readable_dtype(df[col].dtype)
        if expected_dtype != actual:
            errors.append(f"Column '{col}' has dtype '{actual}', expected '{expected_dtype}'")

    return errors


def check_value_ranges(df: pd.DataFrame) -> List[str]:
    errors: List[str] = []

    if "year" in df.columns:
        if df["year"].isna().any():
            errors.append("Column 'year' contains NaNs")

    if "anomaly_sum" in df.columns:
        if df["anomaly_sum"].isna().any():
            errors.append("Column 'anomaly_sum' contains NaNs")

    if "Has" in df.columns:
        if (df["Has"] <= 0).any():
            errors.append("Column 'Has' must be strictly positive for all rows")

    if "region" in df.columns:
        unique_regions = sorted(df["region"].dropna().unique())
        if not unique_regions:
            errors.append("Column 'region' has no non-null values")

    return errors


def verify_contract_for_step_03(config_path: str = "config.yaml") -> None:
    """
    Verify that the cumulative anomalies file feeding script 03
    (`cumulative_anomalies.parquet`) follows the contract defined in data_contract.md
    for section '3. Cumulative Anomalies'.
    """
    config = load_config(config_path)
    output_dir = Path(config["paths"]["output_dir"])
    anomalies_file = output_dir / "cumulative_anomalies.parquet"

    if not anomalies_file.exists():
        raise FileNotFoundError(f"Expected input file for step 03 not found: {anomalies_file}")

    logging.info(f"Loading cumulative anomalies from {anomalies_file}")
    df = pd.read_parquet(anomalies_file)

    logging.info("Checking required columns and dtypes against data_contract.md (section 3)...")
    errors = check_required_columns(df, REQUIRED_COLUMNS_ANOMALIES)

    logging.info("Checking basic value ranges and null constraints...")
    errors.extend(check_value_ranges(df))

    if errors:
        logging.error("Contract verification for step 03 FAILED with the following issues:")
        for e in errors:
            logging.error(f" - {e}")
        raise ValueError("AA_contract_verification_03: data contract violations detected. See log for details.")

    logging.info("Contract verification for step 03 PASSED. Input to 03_triggers_and_losses.py is consistent with data_contract.md.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AA: Contract verification for data feeding into script 03 (triggers and losses)"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    verify_contract_for_step_03(args.config)

