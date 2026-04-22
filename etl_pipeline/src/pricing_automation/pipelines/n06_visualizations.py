"""
n06_visualizations.py
─────────────────────────────────────────────────────────────────────────────
Visualization nodes for the pricing-automation pipeline.

Four plots, each saved as a PNG via the Kedro catalog:

  1. plot_anomaly_map          — mean anomaly field over the full time window
  2. plot_trend_map            — linear trend (slope) per pixel over the period
  3. plot_trigger_frequency_map— choropleth: % of years each location triggered
  4. plot_anomaly_timeseries   — annual mean anomaly + trend line (portfolio)
─────────────────────────────────────────────────────────────────────────────
"""

from .utils import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import geopandas as gpd
import xarray as xr
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _period_label(params_process: dict) -> str:
    t0 = pd.to_datetime(params_process["time_window"][0]).year
    t1 = pd.to_datetime(params_process["time_window"][1]).year
    return f"{t0}–{t1}"


def _anom_var(params_process: dict) -> str:
    return f"anom_{params_process['variable']}"


def _add_colombia_border(ax, gdf_aoi):
    """Overlay AOI geometry in light grey."""
    gdf_aoi.boundary.plot(ax=ax, linewidth=0.4, color="#888888", zorder=3)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Mean anomaly map
# ─────────────────────────────────────────────────────────────────────────────

def plot_anomaly_map(
    ds_processed: xr.Dataset,
    gdf_aoi: gpd.GeoDataFrame,
    params_process: dict,
) -> plt.Figure:
    """
    Spatial map of the time-mean anomaly for the full analysis window.

    Inputs
    ------
    ds_processed  : xr.Dataset with 'anom_{variable}' (time, lat, lon)
    gdf_aoi       : GeoDataFrame with AOI boundaries
    params_process: pipeline params (variable, time_window)

    Output
    ------
    matplotlib Figure
    """
    var = _anom_var(params_process)
    period = _period_label(params_process)

    t0, t1 = params_process["time_window"]
    da = ds_processed[var].sel(time=slice(t0, t1)).mean(dim="time")

    vmax = float(np.nanpercentile(np.abs(da.values), 95))
    vmax = vmax if vmax > 0 else 1.0

    fig, ax = plt.subplots(figsize=(9, 10))
    im = da.plot(
        ax=ax,
        cmap="RdBu",
        vmin=-vmax,
        vmax=vmax,
        add_colorbar=False,
        zorder=1,
    )
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(f"Mean anomaly  [{params_process['variable']}]", fontsize=10)

    _add_colombia_border(ax, gdf_aoi)

    ax.set_title(
        f"Mean Anomaly Field · {period}",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. Trend map
# ─────────────────────────────────────────────────────────────────────────────

def _pixel_trend(values: np.ndarray) -> float:
    """Return OLS slope (units/year) ignoring NaN."""
    mask = ~np.isnan(values)
    if mask.sum() < 5:
        return np.nan
    x = np.arange(len(values), dtype=float)[mask]
    y = values[mask]
    slope, *_ = stats.linregress(x, y)
    return float(slope)


def plot_trend_map(
    ds_processed: xr.Dataset,
    params_process: dict,
) -> plt.Figure:
    """
    Linear trend of the anomaly field at each grid pixel.
    Positive = drying / warming trend; negative = wetting / cooling.

    Inputs
    ------
    ds_processed  : xr.Dataset with 'anom_{variable}' (time, lat, lon)
    params_process: pipeline params (variable, time_window)

    Output
    ------
    matplotlib Figure
    """
    var = _anom_var(params_process)
    period = _period_label(params_process)

    t0, t1 = params_process["time_window"]
    da = ds_processed[var].sel(time=slice(t0, t1))

    # Resample to annual mean, then rechunk so time is a single chunk
    da_annual = da.resample(time="1YE").mean(skipna=True).chunk({"time": -1})

    # Apply trend pixel-by-pixel using xr.apply_ufunc
    trend = xr.apply_ufunc(
        _pixel_trend,
        da_annual,
        input_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    vmax = float(np.nanpercentile(np.abs(trend.values), 95))
    vmax = vmax if vmax > 0 else 0.01

    fig, ax = plt.subplots(figsize=(9, 10))
    im = trend.plot(
        ax=ax,
        cmap="RdBu",
        vmin=-vmax,
        vmax=vmax,
        add_colorbar=False,
        zorder=1,
    )
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(f"Trend  [{params_process['variable']} / year]", fontsize=10)

    ax.set_title(
        f"Anomaly Trend Field · {period}",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Trigger frequency choropleth
# ─────────────────────────────────────────────────────────────────────────────

def plot_trigger_frequency_map(
    df_triggers: pd.DataFrame,
    gdf_aoi: gpd.GeoDataFrame,
    params_trigger: dict,
) -> plt.Figure:
    """
    Choropleth map showing the fraction of years each location activated
    at least one trigger (any window, any percentile layer).

    Inputs
    ------
    df_triggers   : DataFrame from generate_triggers
    gdf_aoi       : GeoDataFrame with location_id and geometry
    params_trigger: create_trigger params (used for window labels)

    Output
    ------
    matplotlib Figure
    """
    # Detect activation columns (P90_activated, P95_activated, etc.)
    act_cols = [c for c in df_triggers.columns if c.endswith("_activated")]
    if not act_cols:
        raise ValueError("df_triggers has no '*_activated' columns.")

    # A location-year is "activated" if ANY layer triggered
    df = df_triggers.copy()
    df["any_activated"] = df[act_cols].max(axis=1)

    # Fraction of years with at least one activation per location
    freq = (
        df.groupby("location_id")["any_activated"]
        .mean()
        .reset_index()
        .rename(columns={"any_activated": "trigger_freq"})
    )

    gdf_plot = gdf_aoi.merge(freq, on="location_id", how="left")
    gdf_plot["trigger_freq"] = gdf_plot["trigger_freq"].fillna(0)

    fig, ax = plt.subplots(figsize=(9, 10))
    gdf_plot.plot(
        column="trigger_freq",
        ax=ax,
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        linewidth=0.2,
        edgecolor="#555555",
        legend=True,
        legend_kwds={
            "label": "Trigger activation frequency (fraction of years)",
            "orientation": "vertical",
            "fraction": 0.03,
            "pad": 0.02,
            "format": mticker.PercentFormatter(xmax=1, decimals=0),
        },
    )

    ax.set_title(
        "Trigger Activation Frequency by Location",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. Anomaly time series
# ─────────────────────────────────────────────────────────────────────────────

def plot_anomaly_timeseries(
    df_cluster: pd.DataFrame,
    params_process: dict,
) -> plt.Figure:
    """
    Annual mean anomaly for the full portfolio with OLS trend line.
    Bars are colored red (negative = drought stress) / blue (positive).

    Inputs
    ------
    df_cluster    : DataFrame from summarize_processed_data
                    (columns: time, anom_{variable}, ...)
    params_process: pipeline params (variable, time_window)

    Output
    ------
    matplotlib Figure
    """
    var = _anom_var(params_process)
    period = _period_label(params_process)

    if var not in df_cluster.columns:
        raise ValueError(f"Column '{var}' not found in df_cluster.")

    df = df_cluster.copy()
    df["time"] = pd.to_datetime(df["time"])

    t0, t1 = params_process["time_window"]
    df = df[(df["time"] >= t0) & (df["time"] <= t1)]

    # Annual portfolio mean
    annual = (
        df.groupby(df["time"].dt.year)[var]
        .mean()
        .reset_index()
        .rename(columns={"time": "year", var: "mean_anom"})
    )

    x = annual["year"].values.astype(float)
    y = annual["mean_anom"].values

    # OLS trend
    slope, intercept, r, p, _ = stats.linregress(x, y)
    trend_y = slope * x + intercept

    colors = ["#D32F2F" if v < 0 else "#1565C0" for v in y]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x, y, color=colors, alpha=0.75, width=0.8, zorder=2)
    ax.plot(
        x,
        trend_y,
        color="#333333",
        linewidth=2,
        linestyle="--",
        label=f"Trend: {slope:+.4f}/yr  (p={p:.3f})",
        zorder=3,
    )
    ax.axhline(0, color="#555555", linewidth=0.8, zorder=1)

    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel(f"Mean anomaly  [{params_process['variable']}]", fontsize=11)
    ax.set_title(
        f"Annual Mean Anomaly — Portfolio · {period}",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    plt.tight_layout()
    plt.close(fig)
    return fig
