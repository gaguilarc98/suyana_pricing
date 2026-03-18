import argparse
import yaml
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_cumulative_anomalies(config_path="config.yaml"):
    config = load_config(config_path)
    input_path = config["paths"]["input_data"]
    output_dir = Path(config["paths"]["output_dir"])
    
    logging.info("Loading input data and baseline climatology...")
    if str(input_path).endswith(".parquet"):
        df_obs = pd.read_parquet(input_path)
    else:
        df_obs = pd.read_csv(input_path)
        
    df_obs["date"] = pd.to_datetime(df_obs["date"])
    df_obs["day_of_year"] = df_obs["date"].dt.dayofyear
    
    climatology_path = output_dir / "baseline_climatology.parquet"
    df_clim = pd.read_parquet(climatology_path)
    
    # Merge observation with climatology
    df = pd.merge(df_obs, df_clim, on=["pixel_id", "day_of_year"], how="left")
    
    # Calculate anomaly
    df["anomaly"] = df["swc"] - df["swc_mean"]
    
    crops = config["parameters"]["crops"]
    target_crop = config["parameters"].get("target_crop", "All")
    crop_results = []
    
    logging.info("Accumulating anomalies per crop window...")
    for crop_name, crop_params in crops.items():
        if target_crop != "All" and crop_name != target_crop:
            continue
            
        logging.info(f"Processing window for {crop_name}")
        
        start_m = crop_params["start_month"]
        start_d = crop_params["start_day"]
        end_m = crop_params["end_month"]
        end_d = crop_params["end_day"]
        crosses_year = crop_params["crosses_year"]
        
        # We assign a campaign_year to each date.
        # If the window crosses the year (e.g., Dec to Mar), 
        # Dec 1999 and Jan-Mar 2000 belong to campaign_year 2000.
        
        # Determine campaign year
        def get_campaign_year(row):
            y = row["date"].year
            m = row["date"].month
            d = row["date"].day
            
            if crosses_year:
                # E.g., start = Dec, end = Mar
                # If month is at or after start_month, it belongs to the NEXT year's campaign
                if m > start_m or (m == start_m and d >= start_d):
                    return y + 1
                else:
                    return y
            else:
                return y
        
        # Filter dates within window
        def in_window(row):
            m = row["date"].month
            d = row["date"].day
            
            # Simple date comparison using an integer proxy, e.g. MMDD
            current_md = m * 100 + d
            start_md = start_m * 100 + start_d
            end_md = end_m * 100 + end_d
            
            if crosses_year:
                # Window is like 1201 to 0331
                # The date is in the window if >= start_md OR <= end_md
                return (current_md >= start_md) or (current_md <= end_md)
            else:
                # Window is like 0115 to 0415
                return (current_md >= start_md) and (current_md <= end_md)
        
        df["campaign_year"] = df.apply(get_campaign_year, axis=1)
        df["in_window"] = df.apply(in_window, axis=1)
        
        # Filter strictly to in-window records
        df_window = df[df["in_window"] == True].copy()
        
        # Aggregate sums per pixel and year
        agg_df = df_window.groupby(["campaign_year", "pixel_id", "Zona", "region", "Has"])["anomaly"].sum().reset_index()
        agg_df.rename(columns={"anomaly": "anomaly_sum", "campaign_year": "year"}, inplace=True)
        agg_df["crop"] = crop_name
        
        crop_results.append(agg_df)
        
    final_df = pd.concat(crop_results, ignore_index=True)
    
    # Save output
    output_file = output_dir / "cumulative_anomalies.parquet"
    final_df.to_parquet(output_file, index=False)
    logging.info(f"Saved cumulative anomalies to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2: Compute Cumulative Anomalies per Crop Window")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    
    run_cumulative_anomalies(args.config)
