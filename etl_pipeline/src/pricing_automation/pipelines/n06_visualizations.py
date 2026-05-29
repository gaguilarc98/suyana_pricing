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
    tw = params_process["time_window"]
    if isinstance(tw, dict):
        t0 = pd.to_datetime(tw["full"][0]).year
        t1 = pd.to_datetime(tw["full"][1]).year
    else:
        t0 = pd.to_datetime(tw[0]).year
        t1 = pd.to_datetime(tw[1]).year
    return f"{t0}–{t1}"


def _anom_var(params_process: dict, ds=None) -> str:
    """Return the anomaly variable name, auto-detecting from dataset if needed."""
    variable = params_process.get("variable")
    # Planet-style: variable is a dict → use the explicit anom_variable key
    if isinstance(variable, dict):
        anom_v = params_process.get("anom_variable", "swc_adjusted")
        candidate = f"anom_{anom_v}"
    else:
        candidate = f"anom_{variable}"
    # If the candidate doesn't exist in the dataset, fall back to first anom_* var
    if ds is not None and candidate not in ds.data_vars:
        anom_vars = [v for v in ds.data_vars if str(v).startswith("anom_")]
        if anom_vars:
            candidate = str(anom_vars[0])
    return candidate


def _time_window(params_process: dict):
    """Return (t0, t1) regardless of ERA5-list or Planet-dict layout."""
    tw = params_process["time_window"]
    if isinstance(tw, dict):
        return tw["full"][0], tw["full"][1]
    return tw[0], tw[1]


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
    var = _anom_var(params_process, ds_processed)  # type: ignore[name-defined]  # ds_processed is in scope here
    period = _period_label(params_process)
    t0, t1 = _time_window(params_process)

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

    ax.set_title(f"SWC Inter-annual Variability · {period}", fontsize=13, loc="left", pad=10)
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
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
    Full-period linear trend per pixel with contextily basemap.
    Units: variable / decade. Positive = drying; negative = wetting.
    """
    var = _anom_var(params_process, ds_processed)
    _t0, _t1 = _time_window(params_process)
    t0 = pd.to_datetime(_t0)
    t1 = pd.to_datetime(_t1)

    da = ds_processed[var].sel(time=slice(str(t0.date()), str(t1.date())))
    trend = _compute_trend_da(da)

    vmax = float(np.nanpercentile(np.abs(trend.values), 95))
    vmax = vmax if vmax > 0 else 0.01
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    gdf_merc = gdf_aoi.to_crs(epsg=3857)
    gdf_t = _da_to_gdf(trend)

    fig, ax = plt.subplots(figsize=(9, 10))
    sc = ax.scatter(
        gdf_t.geometry.x, gdf_t.geometry.y,
        c=gdf_t["value"], cmap="RdBu", norm=norm,
        s=18, linewidths=0, zorder=4,
    )
    _add_basemap(ax, crs_epsg=3857)
    gdf_merc.boundary.plot(ax=ax, linewidth=0.4, color="#333333", zorder=5)

    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, shrink=0.85)
    cb.set_label(f"Trend  [{params_process['variable']} / decade]", fontsize=10)
    cb.ax.axhline(0, color="white", linewidth=1.5)

    ax.set_title(f"SWC Trend · {t0.year}–{t1.year}", fontsize=13, loc="left")
    ax.set_axis_off()
    plt.tight_layout()
    plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Trigger frequency choropleth
# ─────────────────────────────────────────────────────────────────────────────

def plot_trigger_frequency_map(
    df_payouts: pd.DataFrame,
    gdf_aoi: gpd.GeoDataFrame,
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
    # Detect activation by computing payouts above zero
    df = df_payouts.copy()
    df['any_activated'] = (df['perc_payout'] > 0).astype(int)
 
    # Max per (location, year): 1 if any row activated that year
    df = (
        df.groupby(['location_id', 'window_year'])['any_activated']
        .max()
        .reset_index()
    )
 
    # Fraction of years with at least one activation per location
    freq = (
        df.groupby('location_id')['any_activated']
        .mean()
        .reset_index()
        .rename(columns={'any_activated': 'trigger_freq'})
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

    ax.set_title("Trigger Activation Frequency by Location", fontsize=13, loc="left", pad=10)
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
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
    # df_cluster is a DataFrame — detect anomaly column from it directly
    anom_cols = [c for c in df_cluster.columns if str(c).startswith("anom_")]
    var = _anom_var(params_process)
    if var not in df_cluster.columns and anom_cols:
        var = anom_cols[0]
    period = _period_label(params_process)

    if var not in df_cluster.columns:
        raise ValueError(f"Column '{var}' not found in df_cluster.")

    df = df_cluster.copy()
    df["time"] = pd.to_datetime(df["time"])

    t0, t1 = _time_window(params_process)
    df = df[(df["time"] >= t0) & (df["time"] <= (t1 or df["time"].max()))]

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
    ax.set_title(f"Annual Mean Anomaly — Portfolio · {period}", fontsize=13, loc="left")
    ax.legend(fontsize=10, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    plt.tight_layout()
    plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. Activation map with payout fraction
# ─────────────────────────────────────────────────────────────────────────────

def create_activation_map(
    df_payouts: pd.DataFrame,
    gdf: gpd.GeoDataFrame,
    params_map: dict,
    params_s: dict = {},
):
    """
    Plot activation map showing perc_payout intensity per location per year.
    Zero payout -> midnightblue. Non-zero -> green-to-red gradient by intensity.
 
    params_map keys: window, crop, tail, season_name,
                     start_year (optional), location_var (optional).
    """
    window      = params_map['window']
    crop        = params_map['crop']
    tail        = params_map['tail']
    season_name = params_map['season_name']
    location_var = params_s.get('location_var', 'location_id')
 
    gdf = subset_geometry(gdf, params_s)
    gdf = gdf.to_crs(epsg='4326')
 
    df = df_payouts[
        (df_payouts['window'] == window) &
        (df_payouts['crop']   == crop)   &
        (df_payouts['tail']   == tail)
    ].copy()
 
    if 'start_year' in params_map:
        df = df[df['window_year'] >= params_map['start_year']].copy()
 
    df['window_year'] = df['window_year'].astype(int)
 
    # Max perc_payout per (location, year) in case of duplicates
    df = (
        df.groupby([location_var, 'window_year'])['perc_payout']
        .max()
        .reset_index()
    )
 
    gdf_season = gdf.merge(df, how='inner', on=location_var)
 
    # Colormap: midnightblue for zero, green->red for non-zero
    cmap_grad = LinearSegmentedColormap.from_list('payout', ['darkgreen', 'gold', 'crimson'])
    vmax      = df['perc_payout'].max()
 
    def plot_year(ax_i, gdf_year):
        """Plot a single year, splitting zero and non-zero payout polygons."""
        gdf_zero    = gdf_year[gdf_year['perc_payout'] == 0]
        gdf_nonzero = gdf_year[gdf_year['perc_payout']  > 0]
        gdf_na      = gdf_year[gdf_year['perc_payout'].isna()]
 
        if not gdf_zero.empty:
            gdf_zero.plot(ax=ax_i, color='midnightblue',
                          edgecolor='black', linewidth=0.2, alpha=0.7)
        if not gdf_nonzero.empty:
            gdf_nonzero.plot(column='perc_payout', ax=ax_i, cmap=cmap_grad,
                             vmin=0, vmax=vmax,
                             edgecolor='black', linewidth=0.2, alpha=0.7)
        if not gdf_na.empty:
            gdf_na.plot(ax=ax_i, color='lightgrey',
                        edgecolor='black', linewidth=0.2, alpha=0.5)
 
    list_years = sorted(gdf_season['window_year'].dropna().unique().astype(int))
    n_years    = len(list_years)
 
    if n_years <= 15:
        fig, ax = plt.subplots(3, 5, figsize=(18, 14.5))
        years = np.arange(min(list_years), min(list_years) + 15, 1)
    elif n_years <= 20:
        fig, ax = plt.subplots(4, 5, figsize=(24, 14.5))
        years = np.arange(min(list_years), min(list_years) + 20, 1)
    elif n_years <= 25:
        fig, ax = plt.subplots(5, 5, figsize=(24, 14.5))
        years = np.arange(min(list_years), min(list_years) + 25, 1)
    else:
        fig, ax = plt.subplots(5, 6, figsize=(24, 14.5))
        years = np.arange(max(list_years) - 29, max(list_years) + 1, 1)
 
    ax = ax.flatten()
 
    for i, year in enumerate(years):
        gdf_year = gdf_season[gdf_season['window_year'] == year]
        if gdf_year.empty:
            ax[i].axis('off')
            continue
 
        plot_year(ax[i], gdf_year)
 
        ax[i].set_aspect('auto')# 'equal', 'auto'
        ax[i].xaxis.set_major_formatter(FuncFormatter(format_longitude))
        ax[i].yaxis.set_major_formatter(FuncFormatter(format_latitude))
        ax[i].tick_params(axis='x', labelcolor='gray', labelsize=7, rotation=0)
        ax[i].tick_params(axis='y', labelcolor='gray', labelsize=7)
        ax[i].set_title(str(year), size=10)
 
    # Shared colorbar for non-zero payout gradient
    sm = cm.ScalarMappable(cmap=cmap_grad, norm=mcolors.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', #fraction=0.02, 
                        pad=0.02, shrink=0.6)
    cbar.set_label('Payout fraction', size=10)
 
    plt.suptitle(
        f"Activation map -- {season_name} | window {window} | {crop} | {tail} tail",
        fontsize=14, y=0.94
    )
    #plt.tight_layout()
    plt.close(fig)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. Plot time series of index
# ─────────────────────────────────────────────────────────────────────────────

def cumulate_over_window(df_orig, variable, window, time_dim='time'):
    """Cumulate variable daily within a seasonal window. Returns df with
    cum_{variable} column, season and season_year for filtering."""
    df           = df_orig.copy()
    df[time_dim] = pd.to_datetime(df[time_dim])
    month_day    = df[time_dim].dt.strftime('%m-%d')

    start_md, end_md = window[0], window[1]
    start_month, start_day = map(int, start_md.split('-'))
    end_month,   end_day   = map(int, end_md.split('-'))
    crosses_year = (end_month < start_month) or (
        end_month == start_month and end_day < start_day
    )

    if crosses_year:
        df['in_window']      = np.where(
            (month_day >= start_md) | (month_day <= end_md), 1, np.nan
        )
        df['window_year'] = np.where(
            month_day <= end_md,
            df[time_dim].dt.year - 1,
            df[time_dim].dt.year
        )
    else:
        df['in_window']      = np.where(
            (month_day >= start_md) & (month_day <= end_md), 1, np.nan
        )
        df['window_year'] = df[time_dim].dt.year

    df[f'cum_{variable}'] = (
        df.groupby(['window_year', 'in_window'])[variable]
        .cumsum()
        .fillna(0)
    )
    return df


def plot_activation_history(
    df_swc: pd.DataFrame,
    df_payouts: pd.DataFrame,
    params_plot: dict,
    params_trigger: dict,
    params_s: dict = {},
):
    """
    Two-panel plot: climatology/anomaly time series (top) and
    daily cumulated index with payout bars per window-year (bottom).

    params_plot keys: id_area, id_name, window, crop, tail,
                      season_name, start_time, variable,
                      clim_variable, anom_variable.
    """
    # ---- unpack params -------------------------------------------------------
    id_location       = params_plot['id_location']
    id_name       = params_plot['id_name']
    window_key    = params_plot['window']
    crop          = params_plot['crop']
    tail          = params_plot['tail']
    season_name   = params_plot['season_name']
    start_time    = pd.to_datetime(params_plot['start_time'])
    variable      = params_plot['variable']
    cum_variable  = params_plot['cum_variable']
    clim_var      = params_plot['clim_variable']
    anom_var      = params_plot['anom_variable']
    location_var  = params_s.get('location_var', 'location_id')

    lapse = params_trigger['windows'][window_key]

    # ---- filter df_swc -------------------------------------------------------
    df_swc = df_swc[df_swc[location_var] == id_location].copy()
    df_swc['time'] = pd.to_datetime(df_swc['time'])
    df_swc = df_swc[df_swc['time'] >= start_time].copy()

    # ---- daily cumulation within window -------------------------------------
    # Keep all rows so zeros outside the window produce the saw-tooth shape.
    # Filtering to in_window only would cause slanted lines between windows.
    df_cum = cumulate_over_window(df_swc, cum_variable, lapse)

    # ---- filter df_payouts ---------------------------------------------------
    df_pay = df_payouts[
        (df_payouts[location_var] == id_location) &
        (df_payouts['window']     == window_key) &
        (df_payouts['crop']       == crop) &
        (df_payouts['tail']       == tail)
    ].copy()
    df_pay['window_year'] = df_pay['window_year'].astype(int)

    # Max perc_payout per window_year (collapse duplicates)
    df_bars = (
        df_pay.groupby('window_year')['perc_payout']
        .max()
        .reset_index()
    )

    # Bar positions and widths from window lapse
    start_month, start_day = map(int, lapse[0].split('-'))
    end_month,   end_day   = map(int, lapse[1].split('-'))
    crosses_year = (end_month < start_month) or (
        end_month == start_month and end_day < start_day
    )
    bar_positions = pd.to_datetime(
        df_bars['window_year'].astype(str) + '-' + lapse[0]
    )
    end_year = df_bars['window_year'] + crosses_year
    bar_ends = pd.to_datetime(end_year.astype(str) + '-' + lapse[1])
    bar_widths = (bar_ends - bar_positions).dt.days

    # ---- colormap for bars ---------------------------------------------------
    cmap_bars = LinearSegmentedColormap.from_list(
        'payout', ['darkgreen', 'gold', 'crimson']
    )
    vmax = df_bars['perc_payout'].max() if df_bars['perc_payout'].max() > 0 else 1
    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    bar_colors = [
        cmap_bars(norm(v)) if v > 0 else mcolors.to_rgba('midnightblue')
        for v in df_bars['perc_payout']
    ]

    # ---- build figure --------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))

    # -- top panel: climatology and anomaly ------------------------------------
    ax1.plot(df_swc['time'], df_swc[clim_var], lw=1,
             color='darkgreen', label=f'Climatology ({clim_var})')

    mask_below = df_swc[clim_var] > df_swc[variable]
    mask_above = df_swc[clim_var] <= df_swc[variable]
    ax1.fill_between(df_swc['time'], df_swc[clim_var], df_swc[variable],
                     where=mask_below, interpolate=True,
                     color='darkred', alpha=0.4, label='Below climatology')
    ax1.fill_between(df_swc['time'], df_swc[variable], df_swc[clim_var],
                     where=mask_above, interpolate=True,
                     color='midnightblue', alpha=0.4, label='Above climatology')

    # Shade window periods
    for yr in df_bars['window_year']:
        start = pd.to_datetime(f'{yr}-{lapse[0]}')
        end   = pd.to_datetime(f'{int(yr) + crosses_year}-{lapse[1]}')
        ax1.axvspan(start, end, color='orange', alpha=0.15)

    ax1.set_title('a) Climatology and Anomalies', loc='left', fontsize=10)
    ax1.set_ylabel(variable)
    ax1.legend(loc='upper right', fontsize=8)

    # -- bottom panel: daily cumulation + payout bars -------------------------
    ax2.plot(df_cum['time'], df_cum[f'cum_{cum_variable}'], lw=1,
             color='dimgray', label=f'Cumulated {cum_variable} (in-window)')
    ax2.set_title(
        f'b) Trigger activation history -- {season_name}', loc='left', fontsize=10
    )
    ax2.set_ylabel(f'Cumulated {cum_variable}')

    ax2y = ax2.twinx()
    for pos, width, color, val in zip(
        bar_positions, bar_widths, bar_colors, df_bars['perc_payout']
    ):
        ax2y.bar(
            pos, height=val, width=width,
            align='edge', alpha=0.45, color=color
        )

    ax2y.set_ylim(0, vmax * 1.1)
    ax2y.set_ylabel('Payout fraction', fontsize=9)

    ax2.legend(loc='upper left', fontsize=8)

    # Align x-axis limits across both panels
    xlim = ax1.get_xlim()
    ax2.set_xlim(xlim)

    plt.suptitle(
        f'{id_name} | {season_name} | window {window_key} | {crop} | {tail} tail',
        fontsize=12, y=0.98
    )
    plt.tight_layout()
    plt.close(fig)
    return fig