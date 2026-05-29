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

## Automated Reports (Claude)

[`generate_report.py`](generate_report.py) consumes everything in `outputs/` (parquets, plots, `qa_report.md`, `contract_slip.md`, `pricing_quote.csv`) plus the project configs, and uses the Anthropic API with vision to produce three LaTeX deliverables in Suyana brand style:

- `report.tex` / `report.pdf` — technical report (corporate tone, full depth, formulas live in a technical appendix)
- `brief.tex` / `brief.pdf` — single-page client one-pager for a non-technical reader (e.g. a producer); zero jargon, no formulas
- `presentation.tex` / `presentation.pdf` — Beamer deck for a client or reinsurer following the suyana-presentation skill's client-facing rules (20–25 slides, design-before-results, four-lens framework for trigger parameters)
- `presentation.pptx` — optional editable PowerPoint deck based on the Beamer narrative and Suyana visual style, rebuilt with native text, open table layouts, shapes, connectors, and preview-friendly editable chart drawings

Claude actually reads the plot images (multimodal input) and writes a coherent narrative across data, methodology, and results in the chosen language.

### One-time setup

1. Get a personal key at https://console.anthropic.com/settings/keys.
2. Copy the example file and paste your key:
   ```bash
   cp .env.example .env
   # edit .env and set ANTHROPIC_API_KEY=sk-ant-...
   ```
   `pricing/.env` is gitignored — never commit it. The script also reads `ANTHROPIC_API_KEY` from your shell environment if you prefer to export it there.
3. Install the new dependency:
   ```bash
   pip install -r requirements.txt
   ```
4. (Optional) Install a LaTeX distribution (e.g. MacTeX, TeX Live) so the script can compile to PDF. Without it, run with `--skip-compile` and compile the `.tex` files yourself.

### Run

```bash
python generate_report.py \
    --out outputs/report_argentina_$(date +%Y%m%d) \
    --region "Argentina" \
    --product-name "Producto Paramétrico de Sequía" \
    --month-year "Abril 2026" \
    --emit-pptx
```

Useful flags:

| Flag | Default | Purpose |
|------|---------|---------|
| `--only` | `all` | `report`, `brief`, `deck`, or `all` |
| `--lang` | `es` | Output language: `es`, `en`, or `pt`. Switches the babel package, footer labels (e.g. *Confidencial* / *Confidential*), and the language Claude writes in. |
| `--model` | `claude-sonnet-4-6` | Override the Claude model. Default is Sonnet 4.6 (best coding model, cost-efficient for structured LaTeX). Pass `claude-opus-4-7` for deeper reasoning when narrative quality matters more than cost. |
| `--skip-compile` | off | Emit `.tex` only; skip `pdflatex` |
| `--emit-pptx` | off | Create an editable native PowerPoint deck using `beamer_to_pptx.py`. Text, open table layouts, shapes, connectors, and chart drawings are PowerPoint objects; no full-slide images are used. |
| `--region` | `Argentina` | Region label used in titles and prose |

The PPTX converter is intended as the final pipeline step: it normalizes text
to preview-safe ASCII, uses integer OOXML geometry, clears removable macOS
preview metadata on the generated file, and adds a narrative subtitle to every
slide so the subtitle track alone communicates the story.

To create an editable PPTX for an existing Beamer output folder after the fact:

```bash
python beamer_to_pptx.py outputs/report_argentina_20260429
```

### Output folder

Each run produces a self-contained, shareable folder:

```
outputs/report_<region>_<YYYYMMDD>/
├── brand/                  Suyana logos (if present in repo)
├── figures/                Copies of every plot from outputs/plots/
├── report.tex / .pdf       Technical report
├── brief.tex / .pdf        One-page client brief
├── presentation.tex / .pdf Beamer deck
├── presentation.pptx       Optional PowerPoint deck when --emit-pptx is used
└── prompt_inputs.json      Snapshot of what was sent to Claude (reproducibility)
```

## Plot Style (Suyana Palette)

[`suyana_style.py`](suyana_style.py) holds the Suyana plot conventions — palette constants (`SGREEN`, `SBLACK`, `SGRAY`, accent colors), matplotlib `rcParams`, and axis formatters (`fmt_percent_unit`, `fmt_dollars`, `fmt_temp_c`). The three `plot_*.py` scripts call `suyana_style.apply()` at startup so every figure produced by the pipeline ships in Suyana brand. Mirrors the suyana-color-palette skill — see that skill for the underlying conventions.
