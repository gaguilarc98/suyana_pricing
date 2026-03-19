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

def run_bootstrap_aep(config_path="config.yaml"):
    config = load_config(config_path)
    output_dir = Path(config["paths"]["output_dir"])
    n_iterations = config["parameters"]["bootstrap"].get("n_iterations", 10000)
    seed = config["parameters"]["bootstrap"].get("seed", 42)
    
    loss_file = output_dir / "historical_losses.parquet"
    logging.info(f"Loading historical losses from {loss_file}")
    df = pd.read_parquet(loss_file)
    
    # 1. Aggregate annual losses per crop & combined
    logging.info("Aggregating annual losses...")
    annual_crop = df.groupby(["crop", "year"])["loss_usd"].sum().reset_index()
    annual_combined = df.groupby(["year"])["loss_usd"].sum().reset_index()
    annual_combined["crop"] = "Combined"
    
    annual_agg = pd.concat([annual_crop, annual_combined], ignore_index=True)
    agg_file = output_dir / "aggregate_annual_losses.parquet"
    annual_agg.to_parquet(agg_file, index=False)
    logging.info(f"Saved aggregate annual losses to {agg_file}")
    
    # 2. Bootstrap AEP
    logging.info(f"Running bootstrap with {n_iterations} iterations (seed={seed})...")
    np.random.seed(seed)
    
    aep_results = []
    
    for crop in annual_agg["crop"].unique():
        crop_df = annual_agg[annual_agg["crop"] == crop]
        historical_years = crop_df["year"].unique()
        
        # If no years, skip
        if len(historical_years) == 0:
            continue
            
        losses_array = crop_df.set_index("year")["loss_usd"].to_dict()
        years_list = list(losses_array.keys())
        
        # Generate bootstrap indices
        bootstrapped_years = np.random.choice(years_list, size=(n_iterations, len(years_list)), replace=True)
        
        # Sum of losses for each bootstrapped 'portfolio' year.
        # Wait, usually for agricultural products, independent crop windows are sampled.
        # But AEP is about loss distribution. We can simply map the year index to the loss.
        # Assuming we sample portfolio-years (each iteration is 1 year's worth drawn from history)
        # Actually, simpler: AEP is constructed by simply sampling the annual losses with replacement.
        # So we draw n_iterations * 1 year. This gives us n_iterations possible yearly outcomes.
        
        yearly_draws = np.random.choice([losses_array[y] for y in years_list], size=n_iterations, replace=True)
        
        # Sort descending to compute exceedance
        sorted_losses = np.sort(yearly_draws)[::-1]
        
        # Compute probabilities
        ranks = np.arange(1, n_iterations + 1)
        exceedance_prob = ranks / n_iterations
        return_periods = 1.0 / exceedance_prob
        
        aep_df = pd.DataFrame({
            "crop": crop,
            "rank": ranks,
            "return_period_years": return_periods,
            "exceedance_probability": exceedance_prob,
            "loss_usd": sorted_losses
        })
        
        aep_results.append(aep_df)
        
    final_aep = pd.concat(aep_results, ignore_index=True)
    aep_file = output_dir / "aep_curves.parquet"
    final_aep.to_parquet(aep_file, index=False)
    logging.info(f"Saved bootstrapped AEP curves to {aep_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4: Construct Bootstrapped AEP")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    
    run_bootstrap_aep(args.config)
