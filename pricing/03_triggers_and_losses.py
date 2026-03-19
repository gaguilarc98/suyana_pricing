import argparse
import yaml
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_triggers_and_losses(config_path="config.yaml"):
    config = load_config(config_path)
    output_dir = Path(config["paths"]["output_dir"])
    
    start_year = config["parameters"]["climatology"]["start_year"]
    end_year = config["parameters"]["climatology"]["end_year"]
    
    anomalies_file = output_dir / "cumulative_anomalies.parquet"
    logging.info(f"Loading cumulative anomalies from {anomalies_file}")
    df = pd.read_parquet(anomalies_file)
    
    # Filter to baseline period to compute percentiles
    baseline_df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]
    
    logging.info("Computing historical percentiles (P1, P5, P10) per pixel and crop...")
    # Since it's a drought index, worse droughts are more negative. 
    # We take the 1st, 5th, and 10th percentiles of the anomaly sum.
    def compute_percentiles(g):
        return pd.Series({
            "P1": np.percentile(g["anomaly_sum"], 1),
            "P5": np.percentile(g["anomaly_sum"], 5),
            "P10": np.percentile(g["anomaly_sum"], 10)
        })
        
    percentiles = baseline_df.groupby(["crop", "pixel_id"]).apply(compute_percentiles).reset_index()
    
    # Save percentiles reference table
    pct_file = output_dir / "historical_percentiles.parquet"
    percentiles.to_parquet(pct_file, index=False)
    logging.info(f"Saved historical percentiles to {pct_file}")
    
    # Merge percentiles to the main dataset
    df = pd.merge(df, percentiles, on=["crop", "pixel_id"], how="left")
    
    payout_struct = config["parameters"]["payout_structure"]
    region_costs = config["parameters"]["regions"]
    
    def assign_trigger_and_loss(row):
        anomaly = row["anomaly_sum"]
        # Determine tier
        if anomaly <= row["P1"]:
            tier = "P1"
        elif anomaly <= row["P5"]:
            tier = "P5"
        elif anomaly <= row["P10"]:
            tier = "P10"
        else:
            tier = "None"
            
        if tier == "None":
            return pd.Series({"trigger_tier": tier, "loss_usd": 0.0})
            
        # Determine mapped cost key per tier
        if tier == "P1":
            tier_cost_key = "cost_max"
        elif tier == "P5":
            tier_cost_key = "cost_midpoint"
        elif tier == "P10":
            tier_cost_key = "cost_min"
            
        region = row["region"]
        # Default to 0 if region or specific cost tier not found
        cost_per_ha = region_costs.get(region, {}).get(tier_cost_key, 0)
        
        payout_ratio = payout_struct.get(tier, 0.0)
        
        # Calculate final loss (using 'Has' or area)
        loss = row["Has"] * cost_per_ha * payout_ratio
        
        return pd.Series({"trigger_tier": tier, "loss_usd": loss})
        
    logging.info("Evaluating triggers and computing historical losses...")
    loss_metrics = df.apply(assign_trigger_and_loss, axis=1)
    df = pd.concat([df, loss_metrics], axis=1)
    
    # Filter the exact columns expected by the contract to avoid superset ambiguity
    final_cols = ['crop', 'year', 'Zona', 'pixel_id', 'trigger_tier', 'loss_usd']
    
    # Contract also allows `region`, `Has`, `anomaly_sum`, `P1`, `P5`, `P10`
    # Let's include the ones specifically mentioned in the contract updates explicitly
    extended_cols = final_cols + ['region', 'Has', 'anomaly_sum', 'P1', 'P5', 'P10']
    final_df = df[extended_cols].copy()
    
    loss_file = output_dir / "historical_losses.parquet"
    final_df.to_parquet(loss_file, index=False)
    logging.info(f"Saved historical losses to {loss_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3: Compute Triggers and Losses")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    
    run_triggers_and_losses(args.config)
