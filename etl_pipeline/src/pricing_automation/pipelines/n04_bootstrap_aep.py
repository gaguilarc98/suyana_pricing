from .utils import *

import argparse
import logging


def run_bootstrap_aep(df_orig, gdf_aoi, params_bootstrap):
    n_iterations = params_bootstrap.get("n_iterations", 10000)
    seed = params_bootstrap.get("seed", 42)
    loss_usd_ha = params_bootstrap.get('loss_usd_ha', 1)
    
    LOCATION_NAME = 'location_id'
    gdf_aoi = add_area_column(gdf_aoi)
    gdf_aoi['area_ha'] = gdf_aoi['area_km2']*100
    df = df_orig.copy()
    df = df.merge(
        gdf_aoi[[LOCATION_NAME, 'area_ha']],
        how = 'left',
        on = [LOCATION_NAME]
    )
    df['loss_usd'] = df['area_ha'] * df['total_payout'] #* loss_usd_ha

    # 1. Aggregate annual losses per crop & combined
    logging.info("Aggregating annual losses...")
    annual_crop = df.groupby(["crop", "window_year"])["loss_usd"].sum().reset_index()
    annual_combined = df.groupby(["window_year"])["loss_usd"].sum().reset_index()
    annual_combined["crop"] = "Combined"
    
    annual_agg = pd.concat([annual_crop, annual_combined], ignore_index=True)
    logging.info(f"Computes aggregate annual losses")
    
    # 2. Bootstrap AEP
    logging.info(f"Running bootstrap with {n_iterations} iterations (seed={seed})...")
    np.random.seed(seed)
    
    aep_results = []
    
    for crop in annual_agg["crop"].unique():
        crop_df = annual_agg[annual_agg["crop"] == crop]
        historical_years = crop_df["window_year"].unique()
        
        # If no years, skip
        if len(historical_years) == 0:
            continue
            
        losses_array = crop_df.set_index("window_year")["loss_usd"].to_dict()
        years_list = list(losses_array.keys())
        
        # Generate bootstrap indices
        #bootstrapped_years = np.random.choice(years_list, size=(n_iterations, len(years_list)), replace=True)
        
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
        
    df_aep = pd.concat(aep_results, ignore_index=True)
    
    return annual_agg, df_aep


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4: Construct Bootstrapped AEP")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    
    run_bootstrap_aep(args.config)
