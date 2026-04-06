# Pipeline Guide

This guide explains how to run the `pricing-automation` pipeline, what each step does, and how the configuration files control its behavior.

For an interactive walkthrough, open [`notebooks/test_pipeline.ipynb`](notebooks/test_pipeline.ipynb).

---

## How the pipeline works

The project uses [Kedro](https://docs.kedro.org) to organize the data pipeline as a sequence of nodes — each node is a plain Python function with declared inputs and outputs. Kedro resolves the dependency order automatically and handles loading/saving data via the **catalog**.

The pipeline runs in one of three **namespaces**, each targeting a different climate variable:

| Namespace | Variable | Description |
|-----------|----------|-------------|
| `swc` | Soil Water Content | Moisture-based drought index |
| `prcp` | Precipitation | Rainfall-based drought index |
| `temp` | Temperature | Temperature-based drought index |

All three share the same structure. The only difference is in the processing step: `swc` and `prcp` use `process_data_request`, while `temp` uses `process_data_temp`. The active namespace is chosen at runtime.

---

## Configuration files

### `conf/base/globals.yml`

Defines the top-level paths and identifiers used throughout the project. These rarely need to change.

```yaml
data_path: /home/jupyter-gabriel/suyana   # Root folder for all data
dir_name: colombia_davivienda             # Scenario folder name
provider: ERA5                            # Climate data provider
field: swc                                # Climate variable shortname
```

### `conf/base/parameters.yml`

Controls the behavior of each pipeline step. Most tuning happens here.

| Section | What it controls |
|---------|-----------------|
| `params_s` | Spatial filter — which regions to include/exclude and bounding box override |
| `params_t` | Temporal range — `start_year` and `end_year` of historical data |
| `params_request` | Identifies the lead and provider for the data request |
| `params_process` | Climate variable name, time window, smoothing settings, aggregation mode |
| `create_trigger` | Drought windows (date ranges), percentile thresholds, and grouping columns |
| `params_bootstrap` | Number of bootstrap iterations, random seed, and loss unit |
| `params_quote` | Loading factor and profit margin for the final premium calculation |

Key parameters to know:

```yaml
params_s:
  include: {'DEPTO': ['CASANARE', 'META']}  # Filter geometry by column values
  override_gdf: true                         # If true, uses the lat/lon box below instead of geometry
  minx: -60  # Bounding box when override_gdf is true
  maxx: -55
  miny: -30
  maxy: -25

params_process:
  variable: 'swvl1'           # ERA5 variable name
  time_window: ['2024-01-01', '2025-12-31']  # Coverage window for anomaly computation
  summarize_mode: 'pixel'     # How to aggregate spatial data ('pixel', 'nearest', 'within')

create_trigger:
  windows:
    1: {dates: ['03-15', '04-24'], crop: 'all'}  # Growing season windows
    2: {dates: ['04-25', '06-04'], crop: 'all'}
    3: {dates: ['06-05', '07-15'], crop: 'all'}
  percentile_dict:
    side: 'upper'
    layers: {90: 0.4, 95: 0.5, 99: 1.0}    # Percentile threshold → payout fraction

params_bootstrap:
  n_iterations: 1000
  seed: 42
  loss_usd: 1                # Unit loss per trigger event

params_quote:
  loading_factor: 1.5        # Multiplier on top of AAL
  margin: 0.20               # Commercial margin
```

Any parameter can be overridden at runtime without editing the YAML file (see [Running the pipeline](#running-the-pipeline) below).

### `conf/base/catalog.yml`

Defines every dataset the pipeline reads or writes. Each entry has a `type` (how to read/write it) and a `filepath` (where it lives). The `{namespace}` placeholder is automatically replaced with the active pipeline name (`swc`, `prcp`, or `temp`).

Path variables like `${globals:dir_name}` and `${_arg.provider}` are resolved at runtime from `globals.yml` and the `_arg` block at the top of the catalog.

**Dataset types used:**

| Type | Format | Used for |
|------|--------|----------|
| `geopandas.GenericDataset` | GeoPackage (`.gpkg`) | Input geometry and AOI |
| `NetCDFPartitionedDataset` | NetCDF (`.nc`) | Raw ERA5 climate data (one file per year) |
| `XarrayZarrDataset` | Zarr | Processed anomalies and climatology (chunked arrays) |
| `pandas.ParquetDataset` | Parquet | Tabular outputs (triggers, AEP, pricing) |
| `matplotlib.MatplotlibDataset` | PNG | AEP curve plots |

---

## Running the pipeline

### Setup

Open [`notebooks/test_pipeline.ipynb`](notebooks/test_pipeline.ipynb) and follow the cells. The notebook uses the Kedro IPython extension to load the project:

```python
%load_ext kedro.ipython
%cd <project_path>
%reload_kedro .
```

This gives you three objects: `session`, `context`, and `catalog`.

### Step 0 — Set runtime parameters

Override any parameter from `parameters.yml` or `globals.yml` without touching the files:

```python
runtime_params = {
    'lead_id': 'Special-AR-BA',
    'provider': 'ERA5',
    'field': 'swc',
    'dir_name': 'argentina',
    'start_year': 2024,
    'end_year': 2025,
}
session_trial = session.create(runtime_params=runtime_params)
catalog_trial = session_trial.load_context().catalog
```

You can then inspect any catalog entry:

```python
catalog_trial.load('swc.df_cluster')
```

### Running nodes and tags

Run a single node by name:

```python
session_trial.run(pipeline_name='swc', node_names=['swc.get_aoi'])
```

Run a group of nodes by tag:

```python
session_trial.run(pipeline_name='swc', tags=['extract'])
```

Run the full pipeline end-to-end:

```python
session.create(runtime_params=runtime_params).run(pipeline_name='swc')
```

Kedro resolves all dependencies automatically — outputs from earlier nodes become inputs for later ones, so you don't need to manage the order.

---

## Pipeline steps

The pipeline is defined in [`src/pricing_automation/pipelines/pipeline.py`](src/pricing_automation/pipelines/pipeline.py) and registered in [`src/pricing_automation/pipeline_registry.py`](src/pricing_automation/pipeline_registry.py).

---

### Step 1 — `get_aoi` · tag: `extract`

Defines the Area of Interest (AOI) from the input geometry file.

**How it works:** reads the geometry file (`gdf_request`), applies the spatial filters from `params_s` (`include`/`exclude` column filters), and either derives a bounding box from the filtered geometry or uses the explicit lat/lon override if `override_gdf: true`.

| | Dataset | Catalog entry | Format |
|--|---------|--------------|--------|
| **Input** | Portfolio geometry | `gdf_request` | GeoPackage |
| **Input** | Spatial parameters | `params:params_s` | — |
| **Output** | Bounding box dictionary | `{namespace}.dict_bounds` | Memory |
| **Output** | Filtered AOI geometry | `{namespace}.gdf_aoi` | GeoPackage |

---

### Step 2 — `extract_data` · tag: `extract`

Downloads ERA5 climate data for the bounding box and time range.

**How it works:** uses `dict_bounds` to define the spatial window and `params_t` for the year range. Data is fetched from the CDS API and saved as partitioned NetCDF files (one per year).

| | Dataset | Catalog entry | Format |
|--|---------|--------------|--------|
| **Input** | Bounding box | `{namespace}.dict_bounds` | Memory |
| **Input** | Request parameters | `params:params_request` | — |
| **Input** | Time parameters | `params:params_t` | — |
| **Output** | Raw climate data | `{namespace}.ds_request` | NetCDF (partitioned) |

> This is the slowest step — roughly one hour per 30 years of data over a 4×4 degree area.

---

### Step 3 — `process_data_request` / `process_data_temp` · tag: `process`

Computes the climate anomaly index from the raw data.

**How it works:** clips the raw data to the AOI, computes a historical climatology (long-term average), and subtracts it to produce anomalies. For `swc`/`prcp`, this uses `process_data_request`; for `temp`, it uses `process_data_temp` which applies temperature-specific transformations. Results are saved as chunked Zarr arrays for efficient access.

| | Dataset | Catalog entry | Format |
|--|---------|--------------|--------|
| **Input** | Raw climate data | `{namespace}.ds_request` | NetCDF |
| **Input** | AOI geometry | `{namespace}.gdf_aoi` | GeoPackage |
| **Input** | Process parameters | `params:params_process`, `params:params_s` | — |
| **Output** | Anomaly time series | `{namespace}.ds_processed` | Zarr |
| **Output** | Climatology | `{namespace}.ds_climatology` | Zarr |

---

### Step 4 — `summarize_processed_data` · tag: `process`

Assigns climate pixels to insurance locations and aggregates daily values.

**How it works:** matches each grid pixel (or the nearest pixel) to a location in the AOI geometry, producing a flat table of daily index values per location. The `summarize_mode` parameter controls the spatial matching strategy (`pixel`, `nearest`, or `within`).

| | Dataset | Catalog entry | Format |
|--|---------|--------------|--------|
| **Input** | Anomaly time series | `{namespace}.ds_processed` | Zarr |
| **Input** | AOI geometry | `{namespace}.gdf_aoi` | GeoPackage |
| **Input** | Process parameters | `params:params_process` | — |
| **Output** | Daily index per location | `{namespace}.df_cluster` | Parquet |
| **Output** | Pixel-to-location mapping | `{namespace}.df_pixels` | Parquet |

---

### Step 5 — `generate_triggers` · tag: `triggers`

Identifies drought trigger events within each growing season window.

**How it works:** for each window defined in `create_trigger.windows`, it accumulates the anomaly index over the window dates, computes historical percentile thresholds, and flags years where the index exceeded a threshold. Payout fractions are assigned based on which percentile layer was breached (e.g., 90th = 40%, 95th = 50%, 99th = 100%).

| | Dataset | Catalog entry | Format |
|--|---------|--------------|--------|
| **Input** | Daily index per location | `{namespace}.df_cluster` | Parquet |
| **Input** | Trigger parameters | `params:create_trigger`, `params:params_request` | — |
| **Output** | Trigger events per year | `{namespace}.df_triggers` | Parquet |
| **Output** | Percentile thresholds | `{namespace}.df_percentiles` | Parquet |

---

### Step 6 — `run_bootstrap_aep` · tag: `aep` / `pricing`

Estimates the loss distribution using bootstrap resampling.

**How it works:** resamples the historical trigger years with replacement (`n_iterations` times) to build an empirical distribution of annual portfolio losses. This produces an AEP (Annual Exceedance Probability) curve — the probability that losses exceed a given amount in any given year.

| | Dataset | Catalog entry | Format |
|--|---------|--------------|--------|
| **Input** | Trigger events | `{namespace}.df_triggers` | Parquet |
| **Input** | AOI geometry | `{namespace}.gdf_aoi` | GeoPackage |
| **Input** | Bootstrap parameters | `params:params_bootstrap` | — |
| **Output** | Annual aggregated losses | `{namespace}.df_annual_agg` | Parquet |
| **Output** | AEP curve table | `{namespace}.df_aep` | Parquet |

---

### Step 7 — `run_pricing_quote` · tag: `pricing`

Computes the final insurance premium.

**How it works:** takes the Average Annual Loss (AAL) from the bootstrap distribution, applies the `loading_factor` to add a risk buffer, and adds the commercial `margin` to produce the quoted premium rate.

| | Dataset | Catalog entry | Format |
|--|---------|--------------|--------|
| **Input** | Annual aggregated losses | `{namespace}.df_annual_agg` | Parquet |
| **Input** | Pricing parameters | `params:params_quote` | — |
| **Output** | Pricing quote table | `{namespace}.df_pricing` | Parquet |

---

### Step 8 — `plot_aep` · tag: `pricing`

Generates AEP curve visualizations.

**How it works:** plots the historical AEP curve alongside the bootstrap distribution — one chart for the full portfolio and one per crop type. Saved as PNG files.

| | Dataset | Catalog entry | Format |
|--|---------|--------------|--------|
| **Input** | Annual aggregated losses | `{namespace}.df_annual_agg` | Parquet |
| **Input** | AEP curve table | `{namespace}.df_aep` | Parquet |
| **Input** | Process parameters | `params:params_process` | — |
| **Output** | Portfolio AEP chart | `{namespace}.plt_portfolio` | PNG |
| **Output** | Per-crop AEP chart | `{namespace}.plt_aep` | PNG |

---

## Output folder structure

All outputs land under `data/{dir_name}/` (resolved from `globals.yml`):

```
data/colombia_davivienda/
├── geometries/{provider}/
│   └── location_geometries.gpkg       # Filtered AOI
├── sources/{provider}/
│   └── {field}_<year>.nc              # Raw ERA5 downloads
├── features/{provider}/
│   ├── {field}_anomalies.zarr         # Processed anomalies
│   ├── {field}_climatology.zarr       # Historical climatology
│   └── {field}_pixel_assignment.parquet
├── outputs/{provider}/
│   ├── {field}_daily_indices.parquet
│   ├── {field}_window_triggers.parquet
│   ├── {field}_percentiles.parquet
│   ├── aggregate_annual_losses.parquet
│   ├── aep_curves.parquet
│   └── pricing_quote.parquet
└── displays/
    ├── aep_portfolio.png
    └── aep_per_crop.png
```

---

## Quick reference — node names and tags

| Node name | Tag(s) | Function |
|-----------|--------|----------|
| `{ns}.get_aoi` | `extract` | `get_aoi` |
| `{ns}.extract_data` | `extract` | `extract_data` |
| `{ns}.process_data_request` | `process` | `process_data_request` / `process_data_temp` |
| `{ns}.summarize_processed_data` | `process` | `summarize_processed_data` |
| `{ns}.generate_triggers` | `triggers` | `generate_triggers` |
| `{ns}.run_bootstrap_aep` | `aep`, `pricing` | `run_bootstrap_aep` |
| `{ns}.run_pricing_quote` | `pricing` | `run_pricing_quote` |
| `{ns}.plot_aep` | `pricing` | `plot_aep` |

Replace `{ns}` with the pipeline name: `swc`, `prcp`, or `temp`.
