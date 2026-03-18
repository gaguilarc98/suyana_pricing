import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yaml

def main():
    print("======================================================================")
    print("VISUALIZATION: AEP CURVES")
    print("======================================================================")
    
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    start_year = config["parameters"]["climatology"]["start_year"]
    end_year = config["parameters"]["climatology"]["end_year"]
    
    output_dir = os.path.join(config["paths"]["output_dir"], "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    aep_path = os.path.join(config["paths"]["output_dir"], "aep_curves.parquet")
    agg_path = os.path.join(config["paths"]["output_dir"], "aggregate_annual_losses.parquet")
    
    if not os.path.exists(aep_path) or not os.path.exists(agg_path):
        print(f"ERROR: Required parquet files not found in {config['paths']['output_dir']}. Run pipeline first.")
        return

    # Load data
    df_boot = pd.read_parquet(aep_path)
    df_hist = pd.read_parquet(agg_path)
    
    # -------------------------------------------------------------------------
    # Plot 1: Portfolio AEP
    # -------------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    
    # Isolate combined portfolio
    boot_combined = df_boot[df_boot["crop"] == "Combined"]
    hist_combined = df_hist[df_hist["crop"] == "Combined"]
    
    if not boot_combined.empty and not hist_combined.empty:
        # Sort values
        hist_losses = np.sort(hist_combined["loss_usd"].values)[::-1]
        boot_losses = np.sort(boot_combined["loss_usd"].values)[::-1]
        
        hist_probs = np.arange(1, len(hist_losses) + 1) / (len(hist_losses) + 1)
        boot_probs = np.arange(1, len(boot_losses) + 1) / (len(boot_losses) + 1)
        
        ax1.plot(hist_losses / 1e6, hist_probs, color="black", linewidth=2.5, label="Historical")
        ax1.plot(boot_losses / 1e6, boot_probs, color="#D32F2F", linewidth=2.5, linestyle="--", 
                 label=f"Bootstrap (n={len(boot_losses):,})")
                 
        # Mean verticals
        h_mean = hist_losses.mean()
        b_mean = boot_losses.mean()
        ax1.axvline(h_mean / 1e6, color="black", linestyle=":", alpha=0.5)
        ax1.axvline(b_mean / 1e6, color="#D32F2F", linestyle=":", alpha=0.5)
        ax1.text(h_mean / 1e6, 0.92, f' Hist mean\n ${h_mean/1e6:.2f}M', fontsize=8, color='black', va='center')
        ax1.text(b_mean / 1e6, 0.75, f' Boot mean\n ${b_mean/1e6:.2f}M', fontsize=8, color='#D32F2F', va='center')

        ax1.set_xlabel("Annual Drought Loss ($M)", fontsize=12)
        ax1.set_ylabel("Exceedance Probability", fontsize=12)
        ax1.set_title(f"AEP Curve — Drought Portfolio\n{start_year}–{end_year}", fontsize=12, fontweight="bold")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(left=0)
        ax1.set_ylim(0, 1)

        out_port = os.path.join(output_dir, "aep_portfolio.png")
        plt.tight_layout()
        plt.savefig(out_port, dpi=150, bbox_inches="tight")
        print(f"✓ Saved plot → {out_port}")
    plt.close(fig1)

    # -------------------------------------------------------------------------
    # Plot 2: Per-Crop Historical AEP
    # -------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    crops = [c for c in df_hist["crop"].unique() if c != "Combined"]
    palette = plt.cm.tab10.colors
    
    for i, crop in enumerate(crops):
        c_hist = df_hist[df_hist["crop"] == crop]
        if not c_hist.empty:
            c_losses = np.sort(c_hist["loss_usd"].values)[::-1]
            c_probs = np.arange(1, len(c_losses) + 1) / (len(c_losses) + 1)
            ax2.plot(c_losses / 1e6, c_probs, lw=2.0, color=palette[i % 10], label=crop)

    ax2.set_xlabel("Annual Loss per Crop ($M)", fontsize=12)
    ax2.set_ylabel("Exceedance Probability", fontsize=12)
    ax2.set_title(f"Per-crop Historical AEP · {start_year}–{end_year}", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    ax2.set_ylim(0, 1)

    out_crop = os.path.join(output_dir, "aep_per_crop.png")
    plt.tight_layout()
    plt.savefig(out_crop, dpi=150, bbox_inches="tight")
    print(f"✓ Saved plot → {out_crop}")
    plt.close(fig2)

if __name__ == "__main__":
    main()
