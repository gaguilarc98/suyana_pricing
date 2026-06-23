from .utils import *

import argparse
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def run_pricing_quote(df, params_quote):
    
    loading_factor = params_quote.get("loading_factor", 1.5)
    margin = params_quote.get("margin", 0.20)

    # Read annual_agg DataFrame    
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
    
    logging.info(f"Generated pricing quote")
    print("\n--- Pricing Quote Overview ---")
    print(pricing.to_string(index=False))

    return pricing

def plot_aep(df_hist, df_boot, params_process):
    tw = params_process["time_window"]
    if isinstance(tw, dict):
        t0, t1 = tw["full"][0], tw["full"][1]
    else:
        t0, t1 = tw[0], tw[1]
    start_year = pd.to_datetime(t0).year
    end_year = pd.to_datetime(t1).year
   
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

        # Max / mean rate annotation
        h_max = hist_losses.max()
        rate_pct = h_max / h_mean * 100
        ax1.text(
            0.98, 0.98, f"Max / Mean: {rate_pct:.0f}%",
            transform=ax1.transAxes, fontsize=9,
            ha="right", va="top", color="black",
        )

        ax1.set_xlabel("Annual Drought Loss ($M)", fontsize=12)
        ax1.set_ylabel("Exceedance Probability", fontsize=12)
        ax1.set_title(f"AEP Curve — Drought Portfolio · {start_year}–{end_year}", fontsize=12, loc="left")
        ax1.legend(fontsize=10, frameon=False)
        ax1.grid(True, alpha=0.3)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.set_xlim(left=0)
        ax1.set_ylim(0, 1)

        plt.tight_layout()
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
    ax2.set_title(f"Per-crop Historical AEP · {start_year}–{end_year}", fontsize=12, loc="left")
    ax2.legend(fontsize=10, frameon=False)
    ax2.grid(True, alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_xlim(left=0)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.close(fig2)
    
    return fig1, fig2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 5: Compute Pricing Quotes")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    
    run_pricing_quote(args.config)
