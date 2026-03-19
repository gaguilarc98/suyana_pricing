# Pricing Pipeline

Parametric drought insurance pricing pipeline. Transforms pixel-level soil water content (SWC) time series into actuarial loss curves and commercial premiums.

## Position in the End-to-End Architecture

```
data-request/   ← Downloads raw ERA5/CHIRPS climate data (.nc files)
      |
      v
etl/            ← [PENDING] Transforms .nc files → tabular Parquet, loads into PostgreSQL
      |
      v
pricing/        ← THIS COMPONENT
      |
      v
Outputs (pricing quotes, AEP curves, contract slip)
```

> **Note for ETL team (Gabriel):** The pricing pipeline expects a pixel-level
> tabular file (Parquet recommended) as its starting point. The exact schema is
> defined in [`data_contract.md`](data_contract.md) under **Section 1 – Input Data**.
> Key columns: `Zona`, `date`, `pixel_id`, `pixel_lat`, `pixel_lon`, `swc`,
> `region`, `Has`. Once the ETL step produces this file, update `paths.input_data`
> in [`config.yaml`](config.yaml) to point to it.

## Pipeline Steps

| Script | Description |
|--------|-------------|
| `AA_contract_verification_03.py` | Pre-step gate: validates input data against the data contract before the pipeline runs |
| `01_daily_climatology.py` | Computes pixel-level daily climatology (baseline mean SWC per day-of-year) |
| `02_cumulative_anomalies.py` | Accumulates daily anomalies over each crop's seasonal window, per pixel and year |
| `03_triggers_and_losses.py` | Derives historical percentile thresholds (P1/P5/P10) and maps trigger events to monetary losses |
| `04_bootstrap_aep.py` | Bootstraps 10,000 resampled years to build Annual Exceedance Probability (AEP) curves |
| `05_pricing_quote.py` | Computes AAL, StdDev, technical premium, and commercial premium per crop |
| `06_presentation_and_slip.py` | Renders a Markdown contract term slip from the pricing quote |
| `07_quality_assurance.py` | Validates all intermediate outputs against the data contract and actuarial sanity checks |
| `plot_aep_curves.py` | Plots historical and bootstrapped AEP curves |
| `plot_correlations.py` | Heatmap and scatter matrix of cross-crop loss correlations |
| `plot_loss_trends.py` | Annual loss bar charts and trendlines |

## Setup

```bash
pip install -r requirements.txt
```

## Configuration

All pipeline parameters (crop windows, regional costs, payout tiers, pricing factors) are controlled via [`config.yaml`](config.yaml). The only field that needs updating per deployment is `paths.input_data`.

## Running the Pipeline

```bash
# 0. Validate input data contract
python AA_contract_verification_03.py

# 1–7. Run steps in order
python 01_daily_climatology.py
python 02_cumulative_anomalies.py
python 03_triggers_and_losses.py
python 04_bootstrap_aep.py
python 05_pricing_quote.py
python 06_presentation_and_slip.py
python 07_quality_assurance.py

# Optional: generate visualisations
python plot_aep_curves.py
python plot_correlations.py
python plot_loss_trends.py
```

All scripts accept `--config <path>` to point to an alternative config file.

## Outputs

All outputs land in `outputs/` (configured via `paths.output_dir` in `config.yaml`):

| File | Description |
|------|-------------|
| `baseline_climatology.parquet` | Daily mean SWC per pixel |
| `cumulative_anomalies.parquet` | Seasonal anomaly sums per pixel/crop/year |
| `historical_percentiles.parquet` | P1/P5/P10 trigger thresholds per pixel/crop |
| `historical_losses.parquet` | Trigger events and monetary losses |
| `aggregate_annual_losses.parquet` | Portfolio-level annual losses |
| `aep_curves.parquet` | Bootstrapped AEP curves with return periods |
| `pricing_quote.csv / .parquet` | Final pricing table (AAL, StdDev, premiums) |
| `contract_slip.md` | Human-readable insurance term slip |
| `qa_report.md` | Full QA report with pass/fail status for all checks |
| `plots/` | AEP curves, correlation heatmaps, loss trend charts |

The `outputs/` folder committed here contains the results of a verified run on the sample dataset, including a passing QA report (`outputs/qa_report.md`).

## Data Contract

See [`data_contract.md`](data_contract.md) for the full schema of every intermediate file produced and consumed by the pipeline.
