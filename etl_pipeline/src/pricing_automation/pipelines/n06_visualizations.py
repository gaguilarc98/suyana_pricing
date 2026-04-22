"""
n06_visualizations.py
─────────────────────────────────────────────────────────────────────────────
Visualization nodes for the pricing-automation pipeline.

Four plots, each saved as a PNG via the Kedro catalog:

  1. plot_variability_map       — std dev of the climate variable per pixel
  2. plot_trend_map             — linear trend by decade with contextily basemap
  3. plot_trigger_frequency_map — choropleth: % of years each location triggered
  4. plot_anomaly_timeseries    — annual mean anomaly + trend line (portfolio)
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
import contextily as ctx
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


def _da_to_gdf(da: xr.DataArray) -> gpd.GeoDataFrame:
    """Convert a 2-D (lat, lon) DataArray to a GeoDataFrame of points in Web Mercator."""
    df = da.to_dataframe(name="value").reset_index().dropna(subset=["value"])
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326"
    )
    return gdf.to_crs(epsg=3857)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Variability map (std dev per pixel)
# ─────────────────────────────────────────────────────────────────────────────

def plot_variability_map(
    ds_processed: xr.Dataset,
    gdf_aoi: gpd.GeoDataFrame,
    params_process: dict,
) -> plt.Figure:
    """
    Spatial map of the inter-annual standard deviation of the climate variable.
    Shows where year-to-year variability (and thus risk) is highest.

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

    # Annual mean per pixel, then std across years
    da_annual = ds_processed[var].sel(time=slice(t0, t1)).resample(time="1YE").mean()
    da_std = da_annual.std(dim="time")

    fig, ax = plt.subplots(figsize=(9, 10))
    im = da_std.plot(
        ax=ax,
        cmap="YlOrRd",
        add_colorbar=False,
        zorder=1,
    )
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(f"Inter-annual std dev  [{params_process['variable']}]", fontsize=10)

    gdf_aoi.boundary.plot(ax=ax, linewidth=0.4, color="#555555", zorder=3)

    ax.set_title(
        f"SWC Inter-annual Variability · {period}",
        fontsize=13, fontweight="bold", pad=10,
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
    """Return OLS slope in units/decade ignoring NaN."""
    mask = ~np.isnan(values)
    if mask.sum() < 3:
        return np.nan
    x = np.arange(len(values), dtype=float)[mask]
    y = values[mask]
    slope, *_ = stats.linregress(x, y)
    return float(slope) * 10  # convert yr⁻¹ → dec⁻¹


def _compute_trend_da(da: xr.DataArray) -> xr.DataArray:
    """Annual mean → rechunk → pixel-wise OLS slope."""
    da_annual = da.resample(time="1YE").mean(skipna=True).chunk({"time": -1})
    return xr.apply_ufunc(
        _pixel_trend,
        da_annual,
        input_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )


def _add_basemap(ax, crs_epsg: int = 3857):
    """Add contextily basemap (CartoDB Positron — light, no labels)."""
    try:
        ctx.add_basemap(
            ax,
            crs=f"EPSG:{crs_epsg}",
            source=ctx.providers.CartoDB.PositronNoLabels,
            zoom="auto",
            alpha=0.6,
        )
    except Exception:
        pass  # skip silently if offline


def plot_trend_map(
    ds_processed: xr.Dataset,
    gdf_aoi: gpd.GeoDataFrame,
    params_process: dict,
) -> plt.Figure:
    """
    Decadal linear trend maps with contextily basemap.

    The full time window is split into decades; each panel shows the
    per-pixel OLS slope (variable units / year) for that decade.
    Positive = drying trend; negative = wetting trend.

    Inputs
    ------
    ds_processed  : xr.Dataset with 'anom_{variable}' (time, lat, lon)
    gdf_aoi       : GeoDataFrame with AOI boundaries
    params_process: pipeline params (variable, time_window)

    Output
    ------
    matplotlib Figure (one column per decade)
    """
    var = _anom_var(params_process)
    t0 = pd.to_datetime(params_process["time_window"][0])
    t1 = pd.to_datetime(params_process["time_window"][1])

    # Build decade boundaries
    decade_starts = range(
        (t0.year // 10) * 10,
        t1.year + 1,
        10,
    )
    decades = []
    for d0 in decade_starts:
        d1 = d0 + 9
        start = max(t0.year, d0)
        end = min(t1.year, d1)
        if end >= start + 2:  # need at least 3 years
            decades.append((start, end))

    n = len(decades)
    if n == 0:
        raise ValueError("Not enough data to split into decades.")

    # Reproject AOI to Web Mercator for contextily
    gdf_merc = gdf_aoi.to_crs(epsg=3857)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 9), constrained_layout=True)
    if n == 1:
        axes = [axes]

    # Compute global vmax across all decades for a shared colorbar
    all_trends = []
    trend_das = []
    for (d0, d1) in decades:
        da = ds_processed[var].sel(time=slice(f"{d0}-01-01", f"{d1}-12-31"))
        t = _compute_trend_da(da)
        trend_das.append(t)
        vals = t.values
        all_trends.append(np.nanpercentile(np.abs(vals), 95))
    vmax = max(all_trends) if all_trends else 0.01
    vmax = vmax if vmax > 0 else 0.01

    cmap = plt.cm.RdBu
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = None
    for ax, (d0, d1), trend in zip(axes, decades, trend_das):
        # Project trend grid to Web Mercator via GeoDataFrame of points
        gdf_t = _da_to_gdf(trend)
        sc = ax.scatter(
            gdf_t.geometry.x,
            gdf_t.geometry.y,
            c=gdf_t["value"],
            cmap=cmap,
            norm=norm,
            s=18,
            linewidths=0,
            zorder=4,
        )
        im = sc

        # Basemap + AOI border
        _add_basemap(ax, crs_epsg=3857)
        gdf_merc.boundary.plot(ax=ax, linewidth=0.4, color="#333333", zorder=5)

        ax.set_title(f"{d0}s  ({d0}–{d1})", fontsize=12, fontweight="bold")
        ax.set_axis_off()

    # Shared colorbar
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, shrink=0.7)
    cb.set_label(f"Trend  [{params_process['variable']} / decade]", fontsize=11)
    cb.ax.axhline(0, color="white", linewidth=1.5)

    fig.suptitle(
        f"SWC Decadal Trend · {t0.year}–{t1.year}",
        fontsize=14, fontweight="bold", y=1.01,
    )

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
