import argparse
import yaml
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_pricing_quote(config_path="config.yaml"):
    config = load_config(config_path)
    output_dir = Path(config["paths"]["output_dir"])
    
    loading_factor = config["parameters"]["pricing"].get("loading_factor", 1.5)
    margin = config["parameters"]["pricing"].get("margin", 0.20)
    
    annual_losses_file = output_dir / "aggregate_annual_losses.parquet"
    logging.info(f"Loading aggregate annual losses from {annual_losses_file}")
    df = pd.read_parquet(annual_losses_file)
    
    logging.info("Calculating Average Annual Loss (AAL) and Pricing...")
    
    # Calculate AAL and Standard Deviation of historical losses
    pricing = df.groupby("crop")["loss_usd"].agg(["mean", "std"]).reset_index()
    pricing.rename(columns={"mean": "AAL_usd", "std": "StdDev_usd"}, inplace=True)
    
    # Handle NaNs for standard deviation if only 1 year of data
    pricing["StdDev_usd"] = pricing["StdDev_usd"].fillna(0)
    
    # Simplified Technical Premium = AAL + (Loading Factor * StdDev)
    pricing["technical_premium_usd"] = pricing["AAL_usd"] + (loading_factor * pricing["StdDev_usd"])
    
    # Commercial Premium = Technical Premium / (1 - Margin)
    pricing["commercial_premium_usd"] = pricing["technical_premium_usd"] / (1 - margin)
    
    # Sort with Combined at the top if present
    pricing["is_combined"] = pricing["crop"] == "Combined"
    pricing = pricing.sort_values(["is_combined", "crop"], ascending=[False, True]).drop("is_combined", axis=1)
    
    csv_out = output_dir / "pricing_quote.csv"
    pricing.to_csv(csv_out, index=False)
    # Also save as Parquet for data contract
    pricing.to_parquet(output_dir / "pricing_quote.parquet", index=False)
    
    logging.info(f"Saved pricing quote to {csv_out}")
    print("\n--- Pricing Quote Overview ---")
    print(pricing.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 5: Compute Pricing Quotes")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    
    run_pricing_quote(args.config)
