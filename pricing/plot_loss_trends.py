import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yaml
from scipy import stats

def main():
    print("======================================================================")
    print("VISUALIZATION: LOSS TRENDS")
    print("======================================================================")
    
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    start_year = config["parameters"]["climatology"]["start_year"]
    end_year = config["parameters"]["climatology"]["end_year"]
    
    output_dir = os.path.join(config["paths"]["output_dir"], "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    agg_path = os.path.join(config["paths"]["output_dir"], "aggregate_annual_losses.parquet")
    
    if not os.path.exists(agg_path):
        print(f"ERROR: {agg_path} not found. Run pipeline first.")
        return

    # Load data
    df_hist = pd.read_parquet(agg_path)
    
    # -------------------------------------------------------------------------
    # Plot 1: Annual Portfolio Loss Bar Chart + Trendline
    # -------------------------------------------------------------------------
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Filter for total portfolio
    df_combined = df_hist[df_hist["crop"] == "Combined"].sort_values("year")
    if df_combined.empty:
        print("No 'Combined' portfolio data available for trends.")
        return
        
    years = df_combined["year"].values
    losses_m = df_combined["loss_usd"].values / 1e6
    mean_loss = losses_m.mean()
    
    OUTLIER_YEAR = 2022
    
    # Left subplot: Bar chart of annual losses
    colors = ['#d62728' if y == OUTLIER_YEAR else '#1f77b4' for y in years]
    ax1.bar(years, losses_m, color=colors, alpha=0.85, width=0.8)
    ax1.axhline(mean_loss, color='black', linestyle='--', linewidth=1.2,
               label=f'Mean ${mean_loss:.2f}M')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Annual Loss ($M)')
    ax1.set_title(f'Annual Drought Losses — {start_year}-{end_year}')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Right subplot: Trend analysis
    ax2.plot(years, losses_m, 'o-', color='#1f77b4', linewidth=1.5,
            markersize=5, label='Annual loss')

    # Trend line derivation
    slope_all, int_all, r_val, p_all, std_err = stats.linregress(years, losses_m)
    ax2.plot(years, slope_all * years + int_all,
            '--', color='black', linewidth=1.5,
            label=f'Trend (all): {slope_all:+.2f} M/yr  ($r^2$={r_val**2:.2f}, $p$={p_all:.3f})')
            
    mask = years != OUTLIER_YEAR
    slope_ex, int_ex, r_ex, p_ex, _ = stats.linregress(years[mask], losses_m[mask])
    ax2.plot(years, slope_ex * years + int_ex,
            ':', color='#d62728', linewidth=1.5,
            label=f'Trend (excl. {OUTLIER_YEAR}): {slope_ex:+.2f} M/yr ($r^2$={r_ex**2:.2f}, $p$={p_ex:.3f})')
            
    ax2.set_title('Annual Losses Linear Trend')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Loss ($M)')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    out_trends = os.path.join(output_dir, "loss_trends.png")
    plt.tight_layout()
    plt.savefig(out_trends, dpi=150)
    print(f"✓ Saved plot → {out_trends}")
    plt.close(fig1)

    # -------------------------------------------------------------------------
    # Plot 2: Per-Crop Annual Stacked Bar Chart
    # -------------------------------------------------------------------------
    crops = [c for c in df_hist["crop"].unique() if c != "Combined"]
    if not crops:
        return
        
    fig2, ax3 = plt.subplots(figsize=(14, 6))
    
    # Pivot table to organize years on index and crops as columns
    df_crops = df_hist[df_hist["crop"] != "Combined"].pivot(index="year", columns="crop", values="loss_usd").fillna(0)
    df_crops = df_crops / 1e6 # Convert to millions
    
    # Ensure all years in climatology are explicitly plotted
    all_years = list(range(start_year, end_year + 1))
    df_crops = df_crops.reindex(all_years, fill_value=0)
    
    bottom = np.zeros(len(all_years))
    palette = plt.cm.tab10.colors
    
    for i, crop in enumerate(crops):
        if crop in df_crops.columns:
            vals = df_crops[crop].values
            ax3.bar(all_years, vals, bottom=bottom, color=palette[i % 10], alpha=0.85, label=crop, width=0.8)
            bottom += vals
            
    ax3.set_title(f'Annual Drought Losses by Crop — {start_year}-{end_year}', fontsize=12, fontweight="bold")
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Loss ($M)')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    out_crop_ts = os.path.join(output_dir, "crop_annual_loss_timeseries.png")
    plt.tight_layout()
    plt.savefig(out_crop_ts, dpi=150)
    print(f"✓ Saved plot → {out_crop_ts}")
    plt.close(fig2)

if __name__ == "__main__":
    main()
