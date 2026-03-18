import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yaml

def main():
    print("======================================================================")
    print("VISUALIZATION: CORRELATIONS")
    print("======================================================================")
    
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    start_year = config["parameters"]["climatology"]["start_year"]
    end_year = config["parameters"]["climatology"]["end_year"]
    
    output_dir = os.path.join(config["paths"]["output_dir"], "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    hist_path = os.path.join(config["paths"]["output_dir"], "historical_losses.parquet")
    
    if not os.path.exists(hist_path):
        print(f"ERROR: {hist_path} not found. Run pipeline first.")
        return

    # Load data
    df_hist = pd.read_parquet(hist_path)
    
    # -------------------------------------------------------------------------
    # Plot 1: Crop Correlation Heatmaps 
    # -------------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    
    # Build a pivot table of annual losses by crop
    df_annual_crop = df_hist.groupby(["year", "crop"])["loss_usd"].sum().unstack(fill_value=0)
    
    # Drop rows/columns that make no sense
    if "None" in df_annual_crop.columns:
        df_annual_crop = df_annual_crop.drop(columns=["None"])
        
    crops = df_annual_crop.columns.tolist()
    
    if len(crops) > 1:
        # Calculate Pearson correlations between crops
        pearson_mat = df_annual_crop.corr(method="pearson").fillna(0)
        
        cax = ax1.matshow(pearson_mat, cmap="RdYlGn", vmin=-1, vmax=1)
        fig1.colorbar(cax, fraction=0.046, pad=0.04)
        
        ax1.set_xticks(range(len(crops)))
        ax1.set_xticklabels(crops, rotation=45, ha='left')
        ax1.set_yticks(range(len(crops)))
        ax1.set_yticklabels(crops)
        ax1.set_title("Pearson r (annual losses)", fontweight="bold", pad=20)
        
        for i in range(len(crops)):
            for j in range(len(crops)):
                val = pearson_mat.iloc[i, j]
                ax1.text(j, i, f"{val:.2f}", ha="center", va="center", color="black" if abs(val) < 0.5 else "white")

        out_heat = os.path.join(output_dir, "crop_correlation_heatmaps.png")
        plt.tight_layout()
        plt.savefig(out_heat, dpi=150)
        print(f"✓ Saved plot → {out_heat}")
    else:
        print("Not enough crops to compute correlation matrix.")
    plt.close(fig1)

    # -------------------------------------------------------------------------
    # Plot 2: Scatter-Plot Matrix
    # -------------------------------------------------------------------------
    if len(crops) > 1:
        n_c = len(crops)
        fig3, axes3 = plt.subplots(n_c, n_c, figsize=(3 * n_c, 3 * n_c))
        palette = plt.cm.tab10.colors
        
        for i, ci in enumerate(crops):
            for j, cj in enumerate(crops):
                ax = axes3[i, j]
                if i == j:
                    # Diagonal histogram
                    ax.hist(df_annual_crop[ci].values / 1e6, bins=12, color=palette[i % 10], alpha=0.7, edgecolor="white")
                    ax.set_title(ci, fontsize=10, fontweight="bold")
                else:
                    # Off-diagonal scatter
                    xi = df_annual_crop[cj].values / 1e6
                    yj = df_annual_crop[ci].values / 1e6
                    
                    ax.scatter(xi, yj, s=18, alpha=0.6, color=palette[i % 10])
                    if xi.std() > 0 and yj.std() > 0:
                        coef = np.polyfit(xi, yj, 1)
                        xx = np.linspace(xi.min(), xi.max(), 50)
                        ax.plot(xx, np.polyval(coef, xx), "--", lw=0.8, color="gray")
                        
                        r_val = pearson_mat.loc[ci, cj]
                        ax.text(0.05, 0.88, f"r={r_val:.2f}", transform=ax.transAxes,
                                fontsize=9, color="darkred" if abs(r_val) > 0.5 else "black")
                                
                ax.tick_params(labelsize=8)
                if j == 0: ax.set_ylabel(f"{ci} ($M)", fontsize=9)
                if i == n_c - 1: ax.set_xlabel(f"{cj} ($M)", fontsize=9)
                
        fig3.suptitle("Scatter-Plot Matrix — Annual Losses ($M)", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_spm = os.path.join(output_dir, "crop_scatter_matrix.png")
        plt.savefig(out_spm, dpi=150)
        print(f"✓ Saved plot → {out_spm}")
        plt.close(fig3)

if __name__ == "__main__":
    main()
