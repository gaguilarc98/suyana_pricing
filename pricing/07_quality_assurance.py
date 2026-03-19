import argparse
import yaml
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def qa_pipeline(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    output_dir = Path(config["paths"]["output_dir"])
    
    logging.info("Starting Quality Assurance Sanity Checks...")
    errors_found = 0
    
    md_lines = [
        "# Quality Assurance Report\n",
        "## Pipeline Configuration & Parameters\n",
    ]
    
    # Extract Loss per Hectare & Payout Structure from config
    payout_struct = config.get("parameters", {}).get("payout_structure", {})
    regions = config.get("parameters", {}).get("regions", {})
    
    md_lines.append("### Loss Ranges per Hectare by Region\n")
    for region, data in regions.items():
        cmin = data.get("cost_min", 0)
        cmid = data.get("cost_midpoint", 0)
        cmax = data.get("cost_max", 0)
        md_lines.append(f"*   **{region}**: `${cmin:,.2f}` (Min) | `${cmid:,.2f}` (Mid) | `${cmax:,.2f}` (Max)\n")
        
    md_lines.append("\n### Adjustments per Covered Hectare (Payout Structure)\n")
    for tier, payout in payout_struct.items():
        md_lines.append(f"*   **{tier}**: `{payout * 100:.0f}%`\n")
        
    md_lines.append("\n## Data Contract Compliance\n")
    
    # 0. Data Contract Definition
    data_contract = {
        "baseline_climatology.parquet": ["pixel_id", "day_of_year", "swc_mean"],
        "cumulative_anomalies.parquet": ["crop", "year", "Zona", "pixel_id", "region", "anomaly_sum", "Has"],
        "historical_percentiles.parquet": ["crop", "pixel_id", "P1", "P5", "P10"],
        "historical_losses.parquet": ["crop", "year", "Zona", "pixel_id", "region", "Has", "anomaly_sum", "P1", "P5", "P10", "trigger_tier", "loss_usd"],
        "aggregate_annual_losses.parquet": ["crop", "year", "loss_usd"],
        "aep_curves.parquet": ["crop", "rank", "return_period_years", "exceedance_probability", "loss_usd"],
        "pricing_quote.parquet": ["crop", "AAL_usd", "StdDev_usd", "technical_premium_usd", "commercial_premium_usd"]
    }
    
    # 1. Check if files exist and verify Data Contract (for parquets)
    expected_files = list(data_contract.keys()) + ["contract_slip.md"]
    
    for ef in expected_files:
        filepath = output_dir / ef
        if not filepath.exists():
            logging.error(f"QA Failed: Missing expected output: {ef}")
            md_lines.append(f"* ❌ **Missing expected output file**: `{ef}`\n")
            errors_found += 1
            continue
            
        # Verify Data Contract for parquet files
        if ef.endswith(".parquet"):
            try:
                # Read just the schema/columns without loading the full DataFrame into memory if possible, 
                # but pandas read_parquet is fine for these summarized files
                df = pd.read_parquet(filepath)
                expected_cols = set(data_contract[ef])
                actual_cols = set(df.columns)
                
                missing_cols = expected_cols - actual_cols
                col_list_str = ", ".join(df.columns)
                if missing_cols:
                    logging.error(f"QA Failed: {ef} is missing required data contract columns: {missing_cols}")
                    md_lines.append(f"### ❌ `{filepath.name}`\n")
                    md_lines.append(f"*   **Path**: `{filepath.absolute()}`\n")
                    md_lines.append(f"*   **Status**: Missing columns `{missing_cols}`\n")
                    md_lines.append(f"*   **Found Columns**: `{col_list_str}`\n\n")
                    errors_found += 1
                else:
                    md_lines.append(f"### ✅ `{filepath.name}`\n")
                    md_lines.append(f"*   **Path**: `{filepath.absolute()}`\n")
                    md_lines.append(f"*   **Status**: All data contract columns present.\n")
                    md_lines.append(f"*   **Found Columns**: `{col_list_str}`\n\n")
            except Exception as e:
                logging.error(f"QA Failed: Could not read or verify contract for {ef}. Error: {e}")
                md_lines.append(f"### ❌ `{filepath.name}`\n")
                md_lines.append(f"*   **Path**: `{filepath.absolute()}`\n")
                md_lines.append(f"*   **Status**: Could not read or verify contract. Error: {e}\n\n")
                errors_found += 1
    
    if errors_found == 0:
        logging.info("QA Passed: All generated parquets comply with the Data Contract.")
            
    if errors_found > 0:
        logging.error("QA Failed: Data contract breached or missing files.")
        
    md_lines.append("\n## Sanity Checks\n")
        
    # 2. Assert on Percentiles
    # For a drought index, worse droughts are more negative
    # So P1 (1st percentile) should be <= P5 <= P10
    pct_df = pd.read_parquet(output_dir / "historical_percentiles.parquet")
    violators = pct_df[(pct_df["P1"] > pct_df["P5"]) | (pct_df["P5"] > pct_df["P10"])]
    
    md_lines.append("### Historical Percentiles\n")
    if len(violators) > 0:
        logging.error(f"QA Failed: Found {len(violators)} rows where percentiles are mathematically incorrect (P1 should be <= P5 <= P10).")
        md_lines.append(f"❌ **Failed**: Found {len(violators)} rows where percentiles are mathematically incorrect.\n\n")
        errors_found += 1
    else:
        logging.info("QA Passed: Percentile ordering is mathematically sound.")
        md_lines.append("✅ **Passed**: Percentile ordering is mathematically sound.\n\n")
        md_lines.append(pct_df[['P1', 'P5', 'P10']].describe().to_markdown() + "\n\n")
        
    # 3. Assert on Losses
    losses_df = pd.read_parquet(output_dir / "historical_losses.parquet")
    md_lines.append("### Historical Losses\n")
    if (losses_df["loss_usd"] < 0).any():
        logging.error("QA Failed: Negative losses detected.")
        md_lines.append("❌ **Failed**: Negative losses detected.\n\n")
        errors_found += 1
    else:
        logging.info("QA Passed: All generated historical losses are non-negative.")
        md_lines.append("✅ **Passed**: All generated historical losses are non-negative.\n\n")
        md_lines.append(losses_df[['loss_usd']].describe().to_markdown() + "\n\n")
        
    # 3.1 Assert on Empirical Percentile Activations
    md_lines.append("### Empirical Trigger Frequencies\n")
    # A P10 trigger implies P10, P5, or P1 was hit.
    # A P5 trigger implies P5 or P1 was hit.
    # Total possible triggers per pixel is the total number of years (e.g. 30).
    total_pixel_years = len(losses_df)
    
    if total_pixel_years > 0:
        p1_activations = len(losses_df[losses_df["trigger_tier"] == "P1"])
        p5_activations = len(losses_df[losses_df["trigger_tier"] == "P5"]) + p1_activations
        p10_activations = len(losses_df[losses_df["trigger_tier"] == "P10"]) + p5_activations
        
        freq_p1 = p1_activations / total_pixel_years
        freq_p5 = p5_activations / total_pixel_years
        freq_p10 = p10_activations / total_pixel_years
        
        # Build frequency verification table
        freq_data = {
            "Tier": ["P10 (<= P10)", "P5 (<= P5)", "P1 (<= P1)"],
            "Expected Frequency": ["10.00%", "6.67%", "3.33%"],
            "Observed Frequency": [f"{freq_p10*100:.2f}%", f"{freq_p5*100:.2f}%", f"{freq_p1*100:.2f}%"],
            "Total Activations": [p10_activations, p5_activations, p1_activations]
        }
        freq_df = pd.DataFrame(freq_data)
        
        # Assert tolerances (allowing +/- 2% given data lumpiness over 30 years)
        if abs(freq_p10 - 0.10) > 0.02 or abs(freq_p5 - 0.0667) > 0.02 or abs(freq_p1 - 0.0333) > 0.02:
            logging.warning("QA Warning: Empirical trigger frequencies violate the +/- 2% expected tolerance. Check for discrete leaps in SWC.")
            md_lines.append("⚠️ **Warning**: Empirical trigger frequencies violate the +/- 2% expected tolerance.\n\n")
            # We treat this as a warning, not failure, as small samples naturally deviate.
        else:
            logging.info("QA Passed: Empirical trigger frequencies align with expected bounds.")
            md_lines.append("✅ **Passed**: Empirical trigger frequencies align with expected bounds.\n\n")
            
        md_lines.append(freq_df.to_markdown(index=False) + "\n\n")
    else:
        logging.error("QA Failed: Cannot compute empirical frequencies on empty losses dataframe.")
        md_lines.append("❌ **Failed**: Cannot compute empirical frequencies on empty losses dataframe.\n\n")
        errors_found += 1
        
    # 4. Assert Pricing Logic
    pricing_df = pd.read_parquet(output_dir / "pricing_quote.parquet")
    md_lines.append("### Pricing Logic\n")
    pricing_passed = True
    if (pricing_df["AAL_usd"] < 0).any():
        logging.error("QA Failed: Negative AAL detected.")
        md_lines.append("❌ **Failed**: Negative AAL detected.\n")
        errors_found += 1
        pricing_passed = False
        
    if (pricing_df["commercial_premium_usd"] < pricing_df["AAL_usd"]).any():
        logging.error("QA Failed: Commercial Premium is lower than AAL (Check loading variables).")
        md_lines.append("❌ **Failed**: Commercial Premium is lower than AAL.\n")
        errors_found += 1
        pricing_passed = False
        
    if pricing_passed:
        logging.info("QA Passed: Pricing logic sanity checks cleared.")
        md_lines.append("✅ **Passed**: Pricing logic sanity checks cleared.\n\n")
        md_lines.append(pricing_df.to_markdown(index=False) + "\n\n")

    anom_df = pd.read_parquet(output_dir / "cumulative_anomalies.parquet")
    md_lines.append("### Cumulative Anomalies\n")
    if anom_df.empty:
        logging.error("QA Failed: Cumulative anomalies dataframe is empty.")
        md_lines.append("❌ **Failed**: Cumulative anomalies dataframe is empty.\n")
        errors_found += 1
    elif anom_df["anomaly_sum"].isna().any():
        logging.error("QA Failed: Null anomaly sums detected.")
        md_lines.append("❌ **Failed**: Null anomaly sums detected.\n")
        errors_found += 1
    else:
        logging.info("QA Passed: Anomalies generated successfully without nulls.")
        md_lines.append("✅ **Passed**: Anomalies generated successfully without nulls.\n\n")
        md_lines.append(anom_df[['anomaly_sum']].describe().to_markdown() + "\n\n")
        
    md_lines.append("\n## Original Pipeline Checks\n")
    # Pixel Counts
    pixel_count = anom_df["pixel_id"].nunique()
    md_lines.append("### Spatial Coverage\n")
    md_lines.append(f"*   **Total Unique Pixels Validated**: `{pixel_count:,}` pixels\n\n")
    
    # Trigger Stats
    md_lines.append("### Actuarial Triggers\n")
    total_records = len(losses_df)
    n_triggers = len(losses_df[losses_df["trigger_tier"] != "None"])
    if total_records > 0:
        trigger_pct = (n_triggers / total_records) * 100
        md_lines.append(f"*   **Total Trigger Events**: `{n_triggers:,}` triggers (`{trigger_pct:.1f}%` of records)\n\n")
        
    # Financial Extremes
    agg_losses_df = pd.read_parquet(output_dir / "aggregate_annual_losses.parquet")
    md_lines.append("### Financial Extremes\n")
    if not agg_losses_df.empty:
        # Looking at original pipeline logic, average annual loss and max annual sum
        # usually filter for the Combined portfolio.
        combined_agg = agg_losses_df[agg_losses_df["crop"] == "Combined"]
        if not combined_agg.empty:
            mean_loss = combined_agg["loss_usd"].mean()
            max_loss = combined_agg["loss_usd"].max()
            max_year = combined_agg.loc[combined_agg["loss_usd"].idxmax(), "year"]
            
            md_lines.append(f"*   **Mean Annual Loss (Combined)**: `${mean_loss / 1e6:,.2f}M`\n")
            md_lines.append(f"*   **Max Annual Loss (Combined)**: `${max_loss / 1e6:,.2f}M` (Year {int(max_year)})\n\n")
        else:
            md_lines.append("*   ❌ `Combined` crop portfolio string not found in aggregate losses.\n\n")

    # Add Visualizations
    plots_dir = output_dir / "plots"
    if plots_dir.exists() and any(plots_dir.iterdir()):
        md_lines.append("\n## Visualizations & Plots\n")
        md_lines.append("To facilitate human review and quickly assess the pipeline outcomes, the structural plots are included below:\n\n")
        
        md_lines.append("### 1. Annual Exceedance Probability (AEP)\n")
        if (plots_dir / "aep_portfolio.png").exists():
            md_lines.append("![Portfolio AEP](plots/aep_portfolio.png)\n\n")
        if (plots_dir / "aep_per_crop.png").exists():
            md_lines.append("![Per-Crop AEP](plots/aep_per_crop.png)\n\n")
        
        md_lines.append("### 2. Historical Loss Trends\n")
        if (plots_dir / "loss_trends.png").exists():
            md_lines.append("![Annual Losses and Trend](plots/loss_trends.png)\n\n")
        if (plots_dir / "crop_annual_loss_timeseries.png").exists():
            md_lines.append("![Loss Timeseries by Crop](plots/crop_annual_loss_timeseries.png)\n\n")
        
        if (plots_dir / "crop_correlation_heatmaps.png").exists() or (plots_dir / "crop_scatter_matrix.png").exists():
            md_lines.append("### 3. Spatial Correlations & Multi-Crop Activations\n")
            if (plots_dir / "crop_correlation_heatmaps.png").exists():
                md_lines.append("![Crop Correlation Heatmap](plots/crop_correlation_heatmaps.png)\n\n")
            if (plots_dir / "crop_scatter_matrix.png").exists():
                md_lines.append("![Crop Scatter Matrix](plots/crop_scatter_matrix.png)\n\n")


    md_lines.append("\n## Final QA Status\n")
    if errors_found == 0:
        logging.info("✅ SUCCESS: All Quality Assurance Sanity Checks Passed!")
        md_lines.append("✅ **SUCCESS**: All Quality Assurance Sanity Checks Passed!\n")
    else:
        logging.warning(f"❌ FAILED: {errors_found} errors detected during QA.")
        md_lines.append(f"❌ **FAILED**: {errors_found} errors detected during QA.\n")
        
    with open(output_dir / "qa_report.md", "w") as f:
        f.writelines(md_lines)
    logging.info(f"Report generated: {output_dir / 'qa_report.md'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 7: Quality Assurance Checks")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    
    qa_pipeline(args.config)
