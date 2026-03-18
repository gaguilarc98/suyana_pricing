# Pricing Pipeline Data Contract

This document defines the standard data interfaces (DataFrames and files) passed between the scripts in the Pricing Pipeline.

## 1. Input Data (`swc_long` or equivalent)
The pipeline assumes pixel-level data is already downloaded and extracted into a long-format tabular structure (e.g., Parquet or CSV).

**Columns expected:**
* `Zona` (string): Name of the geographic zone/polygon.
* `date` (datetime): The date of the observation.
* `pixel_id` (string): Unique identifier for the pixel (e.g., `lat_lon`).
* `pixel_lat` (float): Latitude of the pixel.
* `pixel_lon` (float): Longitude of the pixel.
* `swc` (float): Daily soil-water content (or other index like NDVI).
* `region` (string): Categorical grouping for pricing tiers (e.g., `Sur`, `Norte`).
* `Has` (float): Total hectares mapped to this zone (or pixel).

## 2. Climatology Baseline (`01_daily_climatology.py` -> Out)
The computed daily climatology per pixel.
* `pixel_id` (string)
* `day_of_year` (int) 
* `swc_mean` (float): Historical mean for this day.

## 3. Cumulative Anomalies (`02_cumulative_anomalies.py` -> Out)
For each pixel and crop window, the accumulated anomalies over the season per year.
* `crop` (string): The crop analyzed.
* `year` (int): Campaign year (usually defined by the harvest year).
* `Zona` (string)
* `pixel_id` (string)
* `region` (string)
* `anomaly_sum` (float): Sum of daily (observed - climatology) over the active window.
* `Has` (float): Area of the pixel in hectares (computed or carried over).

## 4. Triggers and Historical Losses (`03_triggers_and_losses.py` -> Out)
### 4.1. Historical Percentiles (Reference)
* `crop` (string)
* `pixel_id` (string)
* `P1`, `P5`, `P10` (float): Threshold values for the anomaly sums.

### 4.2. Historical Losses
* `crop` (string)
* `year` (int)
* `Zona` (string)
* `pixel_id` (string)
* `region` (string)
* `Has` (float)
* `anomaly_sum` (float)
* `P1`, `P5`, `P10` (float)
* `trigger_tier` (string): E.g., `P1`, `P5`, `P10`, or `None`.
* `loss_usd` (float): The monetary loss computed for this pixel in this year.

## 5. Portfolios & AEP (`04_bootstrap_aep.py` -> Out)
### 5.1. Aggregate Annual Losses
* `crop` (string) or "Combined"
* `year` (int)
* `loss_usd` (float)

### 5.2. AEP Data
* `crop` (string) or "Combined"
* `rank` (int)
* `return_period_years` (float)
* `exceedance_probability` (float)
* `loss_usd` (float)

## 6. Pricing Quotes (`05_pricing_quote.py` -> Out)
A summary table for pricing.
* `crop` (string)
* `AAL_usd` (float): Average Annual Loss.
* `StdDev_usd` (float): Standard Deviation of losses.
* `technical_premium_usd` (float): Technical Premium (AAL + factor * StdDev).
* `commercial_premium_usd` (float): Fully loaded price.
