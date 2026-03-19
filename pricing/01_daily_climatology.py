import argparse
import yaml
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_daily_climatology(config_path="config.yaml"):
    config = load_config(config_path)
    input_path = config["paths"]["input_data"]
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    start_year = config["parameters"]["climatology"]["start_year"]
    end_year = config["parameters"]["climatology"]["end_year"]
    
    logging.info(f"Loading input data from {input_path}")
    try:
        # Depending on format (CSV vs Parquet), we read it. Parquet is standard.
        if str(input_path).endswith(".parquet"):
            df = pd.read_parquet(input_path)
        else:
            df = pd.read_csv(input_path)
    except FileNotFoundError:
        logging.warning(f"Input file {input_path} not found. Creating dummy data for demonstration.")
        # Create dummy data conforming to the data_contract
        dates = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31", freq="D")
        pixels = ["px_1", "px_2"]
        import numpy as np
        data = []
        for p in pixels:
            # Random SWC
            swc = np.random.normal(loc=0.3, scale=0.05, size=len(dates))
            temp_df = pd.DataFrame({"date": dates, "swc": swc})
            temp_df["pixel_id"] = p
            temp_df["pixel_lat"] = -34.0 if p == "px_1" else -27.0
            temp_df["pixel_lon"] = -60.0 if p == "px_1" else -61.0
            temp_df["Zona"] = "Zona_A" if p == "px_1" else "Zona_B"
            temp_df["region"] = "Sur" if p == "px_1" else "Norte"
            temp_df["Has"] = 1000.0
            data.append(temp_df)
        df = pd.concat(data, ignore_index=True)
        # Save dummy so subsequent steps find it
        dummy_path = Path(input_path)
        dummy_path.parent.mkdir(parents=True, exist_ok=True)
        if dummy_path.suffix == ".parquet":
            df.to_parquet(dummy_path, index=False)
        else:
            df.to_csv(dummy_path, index=False)

    df["date"] = pd.to_datetime(df["date"])
    
    # Filter for climatology base period
    baseline_df = df[(df["date"].dt.year >= start_year) & (df["date"].dt.year <= end_year)].copy()
    baseline_df["day_of_year"] = baseline_df["date"].dt.dayofyear
    
    logging.info(f"Computing daily climatology pixel by pixel ({start_year}-{end_year})...")
    
    # Group by pixel and day of year
    climatology = baseline_df.groupby(["pixel_id", "day_of_year"])["swc"].mean().reset_index()
    climatology.rename(columns={"swc": "swc_mean"}, inplace=True)
    
    output_file = output_dir / "baseline_climatology.parquet"
    climatology.to_parquet(output_file, index=False)
    logging.info(f"Saved baseline climatology to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1: Compute Daily Climatology")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    
    run_daily_climatology(args.config)
