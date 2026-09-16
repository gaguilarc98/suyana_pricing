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

from matplotlib.lines import Line2D

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
    if variable in ['tmin', 'tmax']:
        candidate = 'intensity_cold_spell'
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
    params_indices: create_trigger params (used for window labels)

    Output
    ------
    matplotlib Figure
    """
    # Detect activation by computing payouts above zero
    df = df_payouts.copy()
    df['any_activated'] = (df['payout_pct'] > 0).astype(int)
 
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

    gdf_aoi = gdf_aoi.drop_duplicates(subset=['geometry'])

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
# 5. Plot lead location context
# ─────────────────────────────────────────────────────────────────────────────


def plot_context(gdf_context, params_request, gdf_locations=None, location_id_join=False):

    lead_id = params_request.get('lead_id', None)
    figsize = (8,5)
    title = None
    if lead_id is not None:
        title = ' '.join(str(lead_id).split('_')).upper()

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    gdf_aoi_ = gdf_context.drop_duplicates(subset=['geometry'])
    gdf_aoi_ = gdf_aoi_.to_crs(epsg=3857)
    gdf_aoi_.plot(ax=ax, color='lightgray', alpha=0.75, edgecolor='black', linewidth=0.25)

    minx, miny, maxx, maxy = gdf_aoi_.total_bounds

    if gdf_locations is not None:
        gdf_locations_ = gdf_locations.drop_duplicates(subset=['geometry'])
        gdf_locations_ = gdf_locations_.to_crs(epsg=3857)

        if location_id_join:
            gdf_properties_ = gdf_aoi_[gdf_aoi_['location_id'].isin(gdf_locations_['location_id'])].copy()
        else:
            gdf_properties = gdf_locations_.drop(columns=['location_id'])
            gdf_properties = gdf_properties.sjoin_nearest(gdf_aoi_, how='left')
            gdf_properties_ = gdf_aoi_[gdf_aoi_['location_id'].isin(gdf_properties['location_id'])].copy()

        gdf_properties_.plot(ax=ax, color='darkblue', alpha=0.75, edgecolor='darkred')

        gdf_locations_.plot(ax=ax, color='red', edgecolor='darkred', linewidth=0.5)
        bounds = gdf_locations_.total_bounds
        minx = min(minx, bounds[0])
        miny = min(miny, bounds[1])
        maxx = max(maxx, bounds[2])
        maxy = max(maxy, bounds[3])

        # Manual legend handle, decoupled from the actual geometry type plotted
        legend_handle = Line2D(
            [0], [0], marker='*', color='none',
            markerfacecolor='red', markeredgecolor='darkred',
            markersize=10, label=title
        )
        ax.legend(handles=[legend_handle])

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    ctx.add_basemap(ax, source="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", zoom=8, reset_extent=False)

    ax.set_axis_off()
    if title is not None:
        ax.set_title(title, size=10, fontweight='bold')
    plt.close()

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. Average payout map with payout fraction
# ─────────────────────────────────────────────────────────────────────────────


def plot_average_payout(
    df_payouts: pd.DataFrame,
    gdf: gpd.GeoDataFrame,
    params_map: dict,
):
    """
    Plot activation map showing payout_pct intensity per location per year.
    Zero payout -> midnightblue. Non-zero -> green-to-red gradient by intensity.

    params_map keys: window, crop, tail, season_name,
                     start_year (optional), location_var (optional), subset (optional).
    """
    title = params_map.get('title', 'Activation map -- | window | tail')
    params_subset = params_map.get('subset', {})
    location_var = 'location_id'

    gdf = gdf.to_crs(epsg='4326')

    df = subset_geometry(df_payouts, params_subset)

    data_max = df['payout_pct'].max()  # or masked.max() to ignore zeros
    vmax = data_max if data_max > 0 else 1.0

    if 'start_year' in params_map:
        df = df[df['window_year'] >= params_map['start_year']].copy()

    df['window_year'] = df['window_year'].astype(int)

    # Max payout_pct per (location, year) in case of duplicates
    df = (
        df.groupby([location_var, 'window_year'])['payout_pct']
        .mean()
        .reset_index()
    )

    gdf_season = gdf.merge(df, how='inner', on=location_var)

    # Colormap: midnightblue for zero, green->red for non-zero
    cmap_grad = LinearSegmentedColormap.from_list('payout', ['darkgreen', 'gold', 'crimson'])

    def plot_year(ax_i, gdf_year):
        """Plot a single year, splitting zero and non-zero payout polygons."""
        gdf_zero    = gdf_year[gdf_year['payout_pct'] == 0]
        gdf_nonzero = gdf_year[gdf_year['payout_pct']  > 0]
        gdf_na      = gdf_year[gdf_year['payout_pct'].isna()]

        if not gdf_zero.empty:
            gdf_zero.plot(ax=ax_i, color='midnightblue',
                          edgecolor='black', linewidth=0.2, alpha=0.7)
        if not gdf_nonzero.empty:
            gdf_nonzero.plot(column='payout_pct', ax=ax_i, cmap=cmap_grad,
                             vmin=0, vmax=vmax,
                             edgecolor='black', linewidth=0.2, alpha=0.7)
        if not gdf_na.empty:
            gdf_na.plot(ax=ax_i, color='lightgrey',
                        edgecolor='black', linewidth=0.2, alpha=0.5)

    years = sorted(gdf_season['window_year'].dropna().unique().astype(int).tolist())
    n_years = len(years)

    ncols = min(5, n_years) if n_years <= 5 else 5
    nrows = int(np.ceil(n_years / ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows + 1))
    ax = np.atleast_1d(ax).flatten()

    for i, year in enumerate(years):
        axi = ax[i]
        gdf_year = gdf_season[gdf_season['window_year'] == year]

        plot_year(axi, gdf_year)

        axi.set_aspect('equal')
        axi.xaxis.set_major_formatter(FuncFormatter(format_longitude))
        axi.yaxis.set_major_formatter(FuncFormatter(format_latitude))
        axi.tick_params(axis='x', labelcolor='gray', labelsize=7)
        axi.tick_params(axis='y', labelcolor='gray', labelsize=7)
        axi.set_title(str(year), size=10)

    for j in range(n_years, len(ax)):
        ax[j].axis('off')

    # Shared colorbar for non-zero payout gradient
    sm = cm.ScalarMappable(cmap=cmap_grad, norm=mcolors.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax.tolist(), orientation='vertical', pad=0.02, shrink=0.6)
    cbar.set_label('Payout fraction', size=10)

    plt.suptitle(title, fontsize=14, y=0.96, fontweight='bold')
    plt.close(fig)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. Average payout map with payout fraction
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


def plot_time_series(
    df_swc: pd.DataFrame,
    params_series: dict,
    params_indices: dict,
    df_payouts: pd.DataFrame = None,
):
    """
    Two-panel plot: climatology/anomaly time series (top) and
    daily cumulated index with payout bars per window-year (bottom).

    df_payouts is optional. When omitted, the payout bars, their twinx axis,
    and the colormap are skipped; the window shading is derived from the
    time range instead of the payout years.

    params_series keys: window, season_name, start_time, variable,
                      cum_variable, clim_variable, anom_variable,
                      title (optional).
    params_indices keys: windows (dict of window_key -> ("MM-DD", "MM-DD")),
                         params_subset (passed to subset_geometry).
    """
    # ---- unpack params -------------------------------------------------------
    window_key    = params_series['window']
    start_time    = pd.to_datetime(params_series.get('start_time', None))
    end_time      = pd.to_datetime(params_series.get('end_time', None))
    variable      = params_series['variable']
    cum_variable  = params_series['cum_variable']
    clim_var      = params_series['clim_variable']
    title         = params_series.get('title', None)

    lapse         = params_indices['windows'][window_key]
    params_subset = params_series.get('subset', {})
    figsize       = params_series.get('figsize', (12,7))

    # ---- subset swc frame via geometry --------------------------------------
    df_swc = subset_geometry(df_swc, params_subset).copy()
    df_swc['time'] = pd.to_datetime(df_swc['time'])
    if start_time is not None:
        df_swc = df_swc[df_swc['time'] >= start_time].copy()
    if end_time is not None:
        df_swc = df_swc[df_swc['time'] <= end_time].copy()


    # ---- daily cumulation within window -------------------------------------
    # Keep all rows so zeros outside the window produce the saw-tooth shape.
    df_cum = cumulate_over_window(df_swc, cum_variable, lapse)

    # ---- window boundary helpers (needed for shading regardless of payouts) -
    start_month, start_day = map(int, lapse[0].split('-'))
    end_month,   end_day   = map(int, lapse[1].split('-'))
    crosses_year = (end_month < start_month) or (
        end_month == start_month and end_day < start_day
    )

    # ---- payout bars (only when df_payouts is provided) ---------------------
    df_bars = None
    if df_payouts is not None:
        df_pay = subset_geometry(df_payouts, params_subset).copy()
        df_pay['window_year'] = df_pay['window_year'].astype(int)

        # Max payout_pct per window_year (collapse duplicates)
        df_bars = (
            df_pay.groupby('window_year')['payout_pct']
            .max()
            .reset_index()
        )

        bar_positions = pd.to_datetime(
            df_bars['window_year'].astype(str) + '-' + lapse[0]
        )
        end_year = df_bars['window_year'] + crosses_year
        bar_ends = pd.to_datetime(end_year.astype(str) + '-' + lapse[1])
        bar_widths = (bar_ends - bar_positions).dt.days

        cmap_bars = LinearSegmentedColormap.from_list(
            'payout', ['darkgreen', 'gold', 'crimson']
        )
        vmax = df_bars['payout_pct'].max() if df_bars['payout_pct'].max() > 0 else 1
        norm = mcolors.Normalize(vmin=0, vmax=vmax)
        bar_colors = [
            cmap_bars(norm(v)) if v > 0 else mcolors.to_rgba('midnightblue')
            for v in df_bars['payout_pct']
        ]

    # ---- build figure --------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # -- top panel: climatology and anomaly ------------------------------------
    ax1.plot(df_swc['time'], df_swc[clim_var], lw=1.5,
             color='darkgreen', label=f'Climatology ({clim_var})')

    mask_below = df_swc[clim_var] > df_swc[variable]
    mask_above = df_swc[clim_var] <= df_swc[variable]
    ax1.fill_between(df_swc['time'], df_swc[clim_var], df_swc[variable],
                     where=mask_below, interpolate=True,
                     color='darkred', alpha=0.4, label='Below climatology')
    ax1.fill_between(df_swc['time'], df_swc[variable], df_swc[clim_var],
                     where=mask_above, interpolate=True,
                     color='midnightblue', alpha=0.4, label='Above climatology')

    # Shade window periods. Use payout years if available, else derive from range.
    if df_bars is not None:
        shade_years = df_bars['window_year'].tolist()
    else:
        shade_years = range(df_swc['time'].dt.year.min(),
                            df_swc['time'].dt.year.max() + 1)
    for yr in shade_years:
        start = pd.to_datetime(f'{int(yr)}-{lapse[0]}')
        end   = pd.to_datetime(f'{int(yr) + crosses_year}-{lapse[1]}')
        ax1.axvspan(start, end, color='orange', alpha=0.15)

    ax1.set_title('a) Climatology and Anomalies', loc='left', fontsize=10)
    ax1.set_ylabel(variable)
    ax1.legend(loc='upper right', frameon=False, fontsize=8)

    # -- bottom panel: daily cumulation + payout bars -------------------------
    ax2.fill_between(df_cum['time'], df_cum[f'cum_{cum_variable}'], lw=2,
             color='black')
    ax2.fill_between(df_cum['time'], df_cum[f'cum_{cum_variable}'],
             color='lightgray', label=f'Cumulated {cum_variable} (in-window)')
    ax2.set_title(
        'b) Index History for Time Window', loc='left', fontsize=10
    )
    ax2.set_ylabel(f'Cumulated {cum_variable}')

    if df_bars is not None:
        ax2y = ax2.twinx()
        for pos, width, color, val in zip(
            bar_positions, bar_widths, bar_colors, df_bars['payout_pct']
        ):
            ax2y.bar(
                pos, height=val, width=width,
                align='edge', alpha=0.45, color=color
            )
        ax2y.set_ylim(0, vmax * 1.1)
        ax2y.set_ylabel('Payout fraction', fontsize=9)

    ax2.legend(loc='upper right', frameon=False, fontsize=8)

    # ---- pin x-axis to the subset range on both panels ----------------------
    right = df_swc['time'].max()
    ax1.set_xlim(start_time, right)
    ax2.set_xlim(start_time, right)

    # ---- optional overall title ---------------------------------------------
    if title:
        plt.suptitle(title, fontsize=12, y=0.98, fontweight='bold')

    plt.tight_layout()
    plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 8. Map of intensity of events
# ─────────────────────────────────────────────────────────────────────────────


def _assign_window_year(time_coord, period_start, period_end):
    """Map each timestamp to the year its aggregation window starts in.

    Returns an xr.DataArray (same time dim) of integer window-years, with NaN
    for timesteps that fall outside the [period_start, period_end] window.

    A window crosses the year boundary when period_start > period_end
    (e.g. "11-01" -> "03-31"). In that case Nov/Dec belong to year Y and
    Jan..Mar belong to window-year Y-1.
    """
    sm, sd = (int(x) for x in period_start.split('-'))
    em, ed = (int(x) for x in period_end.split('-'))

    months = time_coord.dt.month
    days = time_coord.dt.day
    cal_year = time_coord.dt.year

    # md as a sortable integer MMDD for easy range tests.
    md = months * 100 + days
    start_md = sm * 100 + sd
    end_md = em * 100 + ed

    crosses = start_md > end_md

    if not crosses:
        in_window = (md >= start_md) & (md <= end_md)
        window_year = xr.where(in_window, cal_year, np.nan)
    else:
        # On/after start (e.g. Nov-Dec): belongs to its own calendar year.
        in_tail = md >= start_md
        # On/before end (e.g. Jan-Mar): belongs to the previous calendar year.
        in_head = md <= end_md
        window_year = xr.where(in_tail, cal_year,
                       xr.where(in_head, cal_year - 1, np.nan))

    return window_year.rename('window_year')


def plot_intensity_map(
    ds: xr.Dataset,
    gdf: gpd.GeoDataFrame,
    params: dict,
    gdf_locations=None,
):
    """Yearly sum of a 0/1 event field per pixel, as pcolormesh subplots with
    a shared colorbar and the gdf boundaries overlaid.

    params keys:
        field : str, default 'cold_spell'
        title : str, optional
        bar_label : str, optional
        start_year : optional
        mask_zeroes : bool, default False
        period_start, period_end : "MM-DD", optional
            Restrict each year's aggregation to the day-of-year window
            [period_start, period_end]. If the window crosses the calendar-year
            boundary (e.g. "11-01" to "03-31"), all timesteps in it are attributed
            to the year in which the window *starts*. Pass both or neither.
    """
    params = params or {}
    ds_orig = ds.copy()
    time_dim = get_time_coordinate(ds_orig)

    field = params.get('variable', 'cold_spell')
    title = params.get('title', 'Event frequency by year')
    bar_label = params.get('bar_label', 'Event frequency')
    mask_zeroes = params.get('mask_zeroes', False)
    period_start = params.get('period_start')
    period_end = params.get('period_end')

    lon_var, lat_var, _ = get_coordinates(ds_orig)
    gdf = gdf.to_crs(epsg='4326')

    if mask_zeroes:
        ds_orig[field] = xr.where(ds_orig[field] == 0, np.nan, ds_orig[field])

    # Decide the grouping coordinate.
    if period_start is not None and period_end is not None:
        window_year = _assign_window_year(
            ds_orig[time_dim], period_start, period_end
        )
        # Keep only timesteps inside the window (NaN window_year = outside).
        in_window = window_year.notnull()
        sub = ds_orig[field].where(in_window, drop=False)
        wy = window_year.where(in_window)

        # Group the masked field by the integer window-year label.
        freq = (sub.groupby(wy.rename('window_year'))
                   .sum(dim=time_dim, min_count=1))
        year_coord = 'window_year'
    else:
        freq = (ds_orig[field]
                .groupby(ds_orig[time_dim].dt.year)
                .sum(dim=time_dim, min_count=1))
        year_coord = freq.dims[0]

    if 'start_year' in params:
        freq = freq.sel({year_coord: freq[year_coord] >= params['start_year']})
    if 'end_year' in params:
        freq = freq.sel({year_coord: freq[year_coord] <= params['end_year']})

    years = [int(y) for y in freq[year_coord].values]
    n_years = len(years)
    vmax = float(np.nanpercentile(freq, 99)) or 1.0 #freq.max()
    vmin = 0 or float(np.nanpercentile(freq, 1)) #freq.min()

    ncols = min(5, n_years) if n_years <= 5 else 5
    nrows = int(np.ceil(n_years / ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows + 1))
    ax = np.atleast_1d(ax).flatten()

    cmap = LinearSegmentedColormap.from_list('freq', ['darkgreen', 'gold', 'crimson'])
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    lons = ds_orig[lon_var].values
    lats = ds_orig[lat_var].values

    for i, year in enumerate(years):
        axi = ax[i]
        layer = freq.sel({year_coord: year})

        axi.pcolormesh(lons, lats, layer.values, cmap=cmap, norm=norm, shading='auto')
        gdf.boundary.plot(ax=axi, edgecolor='gray', linewidth=0.25)
        if gdf_locations is not None:
            gdf_locations.plot(ax=axi, color='blue', marker='*', markersize=30)

        axi.set_aspect('equal')
        axi.xaxis.set_major_formatter(FuncFormatter(format_longitude))
        axi.yaxis.set_major_formatter(FuncFormatter(format_latitude))
        axi.tick_params(axis='x', labelcolor='gray', labelsize=7)
        axi.tick_params(axis='y', labelcolor='gray', labelsize=7)
        axi.set_title(str(year), size=10)

    for j in range(n_years, len(ax)):
        ax[j].axis('off')

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax.tolist(), orientation='vertical', pad=0.02, shrink=0.6)
    cbar.set_label(bar_label, size=10)

    plt.suptitle(title, fontsize=14, y=0.96, fontweight='bold')
    plt.close(fig)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 9. Time series of payouts
# ─────────────────────────────────────────────────────────────────────────────

BLUE  = "#2563EB"
AMBER = "#D97706"
GREEN = "#16A34A"
RED   = "#DC2626"
BG    = "#F8FAFC"
GRID  = "#6B819E"

fmt_millions = mticker.FuncFormatter(lambda x, _: f"${x:.0f}")

def _build_color_map(groups, cmap="Turbo"):
    cmap = cm.get_cmap(cmap)
    n = len(groups)
    return {g: (cmap(2*i / (2*n)), cmap((2*i+1) / (2*n)))
            for i, g in enumerate(groups)}

def plot_annual_payouts(df, params_annual):
    """
    Plot average/total annual payouts, optionally split into subplots by a categorical column.

    params_annual keys: payout_col, title (required); split_col, premium_col, year_col
    (default 'window_year'), agg (default 'mean'), payout_label, premium_label, ylabel,
    figsize, palette (default 'Set2'), bar (default False) (optional).
    """
    payout_col = params_annual['payout_col']
    title = params_annual['title']
    split_col = params_annual.get('split_col', None)
    premium_col = params_annual.get('premium_col', None)
    year_col = params_annual.get('year_col', 'window_year')
    agg = params_annual.get('agg', 'mean')
    payout_label = params_annual.get('payout_label', 'Pago total')
    premium_label = params_annual.get('premium_label', 'Prima total')
    ylabel = params_annual.get('ylabel', None)
    figsize = params_annual.get('figsize', None)
    palette = params_annual.get('palette', 'Set2')
    bar = params_annual.get('bar', False)

    plot_cols = [payout_col] + ([premium_col] if premium_col else [])

    def _draw(ax, sub, c_prem, c_pay):
        if premium_col:
            if bar:
                ax.bar(sub[year_col], sub[premium_col], color=c_prem, lw=2.2,
                    label=premium_label)
            else:
                ax.plot(sub[year_col], sub[premium_col], color=c_prem, lw=2.2,
                    marker="o", markersize=5, label=premium_label)
        ax.plot(sub[year_col], sub[payout_col], color=c_pay, lw=2.2,
                marker="s", markersize=5, label=payout_label, linestyle="--")
        ax.set_xticks(sorted(sub[year_col].unique().astype(int)))
        for _, row in sub.iterrows():
            if row[payout_col] > 0:
                ax.annotate(fmt_millions(row[payout_col], None),
                            xy=(row[year_col], row[payout_col]),
                            xytext=(0, 8), textcoords="offset points",
                            ha="center", fontsize=9, color=c_pay,
                            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                      edgecolor=c_pay, linewidth=0.8))
        ax.yaxis.set_major_formatter(fmt_millions)
        ax.grid(True)
        ax.legend(frameon=False, fontsize=10)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=10)

    if premium_col:
        df = df.dropna(subset=[payout_col, premium_col]).copy()

    if split_col is None:
        annual = df.groupby(year_col)[plot_cols].agg(agg).reset_index()
        fig, ax = plt.subplots(figsize=figsize or (12, 5))
        c_prem, c_pay = sns.color_palette(palette, 2)
        _draw(ax, annual, c_prem, c_pay)
        ax.set_xlabel("Año", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        fig.tight_layout()
        plt.close()
        return fig

    groups = df[split_col].unique()
    color_map = _build_color_map(groups, palette)

    fig, axes = plt.subplots(len(groups), 1,
                             figsize=figsize or (14, 2.8 * len(groups)),
                             sharex=True)
    if len(groups) == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    for ax, g in zip(axes, groups):
        sub = (df[df[split_col] == g]
               .groupby(year_col)[plot_cols].agg(agg).reset_index())
        c_prem, c_pay = color_map.get(g, (BLUE, AMBER))
        _draw(ax, sub, c_prem, c_pay)
        ax.set_title(f"{split_col.capitalize()}: {str(g)}",
                     fontsize=10, fontweight="semibold", y=1.05)

    axes[-1].set_xlabel("Año", fontsize=11)
    fig.tight_layout()
    plt.close()
    return fig