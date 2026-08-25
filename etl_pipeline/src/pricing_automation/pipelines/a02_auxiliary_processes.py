from .utils import *

#———————————————————————————————————————————
# AUXILIARY FUNCTIONS
#———————————————————————————————————————————


def add_time_coordinate(ds_orig, level='week', time_dim=None, n_days=7):
    """Add one or more climatology grouping coordinates.
    `level` may be a single level or a list. Each is added via the original
    single-level logic. Returns the dataset with every requested coord present.
    """
    ds = ds_orig.copy()
    if time_dim is None:
        time_dim = get_time_coordinate(ds)

    levels = [level] if isinstance(level, str) else list(level)
    for lvl in levels:
        if lvl == 'dayofyear':
            ds = ds.assign_coords({lvl: ds[time_dim].dt.dayofyear})
        elif lvl == 'week':
            ds = ds.assign_coords({lvl: ds[time_dim].dt.isocalendar().week})
        elif lvl == 'month':
            ds = ds.assign_coords({lvl: ds[time_dim].dt.month})
        elif lvl == 'day':
            ds = ds.assign_coords({lvl: ds[time_dim].dt.day})
        elif lvl == 'year':
            ds = ds.assign_coords({lvl: ds[time_dim].dt.year})
        elif lvl == 'hour':
            ds = ds.assign_coords({lvl: ds[time_dim].dt.hour})
        elif lvl == 'fixed_days':
            step = int(n_days)
            coord_name = f'fixed_days_{step}'
            doy = ds[time_dim].dt.dayofyear
            ds = ds.assign_coords({coord_name: ((doy - 1) // step).astype('int64')})
        else:
            raise ValueError(f"No support for level {lvl}")
    return ds

def get_smooth_series(ds, field, smooth_window):
    """Smooth time series with a time window"""
    time = get_time_coordinate(ds)
    if smooth_window>0:
        da = ds[field].rolling({time: smooth_window}, min_periods=1).mean()
    return da


#———————————————————————————————————————————
# RENAME VARIABLES
#———————————————————————————————————————————

RENAME_DICT = {
    'swc': ['swc', 'swvl1', 'soil_water_content'],
    'prcp': ['prcp', 'tp', 'total_precipitation', 'precipitation', 'precip', 'prc', 'pcp', 'precip_rnl'],
    'tmin': ['tmin', 't2m', '2t', 't', 'mn2t6', 'mn2t', 'mn2t24'],
    'tmax': ['tmax', 't2m', '2t', 't', 'mx2t6', 'mx2t', 'mx2t24'],
    'windspeed': ['windspeed', 'wind_speed', 'wind'],
    'windgust': ['windgust', 'fg10'], 
    'wind': ['fg10'],
}

def rename_vars(ds_orig, variable=None):
    """Rename variables in the original dataset with the registered names"""
    if variable is None:
        raise KeyError(f'{variable} is not in the dataset, a variable name must be provided.')
    ds = ds_orig.copy()

    # Get coordinate names
    lon, lat, time = get_coordinates(ds)

    # Get list of possible variable names
    if variable not in RENAME_DICT:
        raise KeyError(f'{variable} is not available in the registered variables in n02_process_data')

    list_vars = list(set(ds.data_vars).intersection(set(RENAME_DICT[variable])))
    if len(list_vars)==0:
        print(f'Either the variable name is not registered in n02_process_data or the variable is wrong.')
        raise KeyError(f'There is no variable in the dataset matching the requested variable: {variable}')

    var_name = list_vars[0]
    print(f'{var_name} was detected among the variables. It will be used and renamed to {variable}')

    dict_rename = {var_name: variable}
    if lon is not None:
        dict_rename[lon] = 'lon'
        print(f'Renaming {lon} to "lon"')
    if lat is not None:
        dict_rename[lat] = 'lat'
        print(f'Renaming {lat} to "lat"')
    if time is not None:
        dict_rename[time] = 'time'
        print(f'Renaming {time} to "time"')

    ds = ds.rename(dict_rename)

    return ds


#———————————————————————————————————————————
# CLEAN DATASET
#———————————————————————————————————————————


def clean_data(
    ds_orig, 
    date_range = (None, None), 
    var: {list or str} = None,
    na_replace = None,  
    normalize_time = True
):
    """Clean dataset by slicing in time and replacing missing values"""
    # Get coordinate names
    lon, lat, time = get_coordinates(ds_orig)
    # Remove duplicates
    ds = ds_orig.drop_duplicates(..., keep='first')

    # Select variable
    if var is not None:
        if isinstance(var, str):
            var = [var]
        if isinstance(var, list) and set(var).issubset(ds.data_vars):
            ds = ds[var].copy()
        else:
            raise KeyError(f"{var} must be a string or list of variable names present in the dataset.")

    # Convert time to datetime format
    if normalize_time:
        ds[time] = pd.to_datetime(ds[time]).normalize()# + pd.offsets.DateOffset(normalize=True)
    
    # Select data within the time window
    if date_range[0] is None and date_range[1] is None:
        pass
    else:
        ds = ds.sel(time = slice(date_range[0], date_range[1]))
    
    # Replace values with nan (65535 for Planet)
    if isinstance(na_replace, (int, float)):
        ds = ds.where(ds != na_replace, np.nan)

    return ds


#———————————————————————————————————————————
# CDF MATCHING DATASET
#———————————————————————————————————————————


def get_window_days(doy, window=30):
    """Return list of days-of-year within ±window of doy (wrap around at 365)."""
    days = np.arange(doy - window, doy + window + 1)
    return ((days - 1) % 365) + 1
 
 
def get_cdf_fixed_data(
    ds: xr.Dataset, 
    ds_base: xr.Dataset, 
    field_target: str, 
    field_base: str,
    group_cols: list, 
    level: str = None,
    window_size: int = 45,
    min_pool: int = 5,
    n_jobs: int = 6,
) -> xr.DataArray:
    """
    Performs a rolling day-of-year CDF (quantile) matching of the target field onto
    the distribution of the base field.
 
    This is the reproduction of Planet's actual harmonization. For each pixel/group
    and each day-of-year, the empirical distribution is built from all days whose
    day-of-year falls within +/-window_size (wrapping at year end), on BOTH the
    source and base sides. Each daily target value is converted to its empirical
    quantile in the windowed source pool and mapped to the interpolated value at
    the same quantile of the windowed base pool. Interpolation (np.quantile) on the
    base side is what reproduces Planet's smoothness and distribution, replacing the
    discrete rank-bucket lookup used previously.
 
    Args:
        - ds : (xr.Dataset) Dataset whose field_target will be rescaled
        - ds_base : (xr.Dataset) Dataset providing the reference distribution (field_base)
        - field_target : (str) Name of the field to rescale in ds
        - field_base : (str) Name of the reference field in ds_base
        - group_cols : (list) Grouping dimensions (must be a subset of ds.dims), e.g. [location_id]
        - level : (str) Time level for the rolling window. Only 'dayofyear' is supported.
        - window_size : (int) Half-width in days of the day-of-year window (Planet uses 45)
        - min_pool : (int) Minimum number of valid values required on each side to match
        - n_jobs : (int) Parallel workers across day-of-year
    Returns:
        - xr.DataArray with the rescaled field, indexed by the original dims of ds
    """
    # Get dimensions and check that group_cols is a subset of the dimensions
    dims = list(ds.sizes.keys())
    if not set(group_cols).issubset(set(dims)):
        raise ValueError('group_cols must be a subset of dims from first Dataset')
    if level is None:
        level = 'dayofyear'
    if level != 'dayofyear':
        raise ValueError("get_cdf_fixed_data only supports level='dayofyear'")
 
    # Add day-of-year coordinate on both sides and move to dataframes
    ds_o = ds.copy()
    ds_b = ds_base.copy()
    ds_o = add_time_coordinate(ds, level=level)
    ds_b = add_time_coordinate(ds_base, level=level)
 
    df_o = ds_o.to_dataframe().reset_index()
    df_b = ds_b.to_dataframe().reset_index()
 
    # Pre-group the base pools by day-of-year for fast windowed lookup
    base_by_doy = {d: g for d, g in df_b.groupby('dayofyear')}
 
    def cdf_matching_day(day):
        df_orig = df_o[df_o['dayofyear'] == day]
        if df_orig.empty:
            return None
 
        # Build the +/-window day-of-year base pool for this day
        list_days = get_window_days(day, window=window_size)
        base_parts = [base_by_doy[d] for d in list_days if d in base_by_doy]
        if not base_parts:
            return None
        df_base = pd.concat(base_parts, axis=0)
 
        # Source pool uses the SAME windowed days so the empirical quantile of each
        # daily value is estimated over the local seasonal distribution, not a single day
        src_pool_all = df_o[df_o['dayofyear'].isin(list_days)]
 
        out_parts = []
        for key, g_orig in df_orig.groupby(group_cols):
            mask_base = np.ones(len(df_base), dtype=bool)
            mask_src = np.ones(len(src_pool_all), dtype=bool)
            for col, val in zip(group_cols, key if isinstance(key, tuple) else (key,)):
                mask_base &= (df_base[col].values == val)
                mask_src &= (src_pool_all[col].values == val)
 
            base_pool = df_base.loc[mask_base, field_base].dropna().values
            src_pool = src_pool_all.loc[mask_src, field_target].dropna().values
 
            g_out = g_orig[dims].copy()
            if len(base_pool) < min_pool or len(src_pool) < min_pool:
                g_out[field_base] = np.nan
            else:
                src_sorted = np.sort(src_pool)
                vals = g_orig[field_target].values
                # Empirical quantile of each value within the windowed source pool
                q = np.searchsorted(src_sorted, vals, side='right') / len(src_sorted)
                q = np.clip(q, 0.0, 1.0)
                # Interpolated value at that quantile of the windowed base pool
                mapped = np.quantile(base_pool, q)
                mapped[np.isnan(vals)] = np.nan
                g_out[field_base] = mapped
            out_parts.append(g_out)
 
        return pd.concat(out_parts, axis=0, ignore_index=True)
 
    df_list = Parallel(n_jobs=n_jobs)(
        delayed(cdf_matching_day)(day) for day in np.arange(1, 367)
    )
    df_list = [d for d in df_list if d is not None]
 
    df_list = pd.concat(df_list, axis=0, ignore_index=True)
    df_list = df_list.sort_values(by=group_cols + ['time'])
 
    da = df_list.set_index(dims).to_xarray()[field_base]
 
    return da



#———————————————————————————————————————————
# ADD CLIMATOLOGY TO DATASET
#———————————————————————————————————————————


def add_climatology(ds, ds_clim, field, var_name='climatology', level = 'week'):
    '''Adds a pre-computed climatology to the original dataset'''
    lookup = ds[level]
    clim_values = ds_clim[field].sel({level: lookup})
    ds[var_name] = clim_values

    return ds


def get_climatology(ds, field, var_name='climatology', level='dayofyear', smooth_window=0, time_dim=None, func='mean'):
    '''Get climatologies for the specified variable at the level selected'''
    if time_dim is None:
        time_dim = get_time_coordinate(ds)
    
    ds = add_time_coordinate(ds, time_dim=time_dim, level= level)

    # Only keep full years
    year = ds[time_dim].dt.year
    year_counts = year.groupby(year).count()
    full_years = year_counts.where(year_counts >= 360, drop=True).coords[year.name]

    ds_full = ds.sel({time_dim: ds[time_dim].dt.year.isin(full_years)})

    # Lazy climatology
    if func == 'mean':
        ds_clim = ds_full[[field]].groupby(level).mean(dim=time_dim)
    elif func == 'std':
        ds_clim = ds_full[[field]].groupby(level).std(dim=time_dim)
    if smooth_window > 0:
        ds_clim = xr.concat([
            ds_clim.isel({level: slice(-smooth_window, None)}), 
            ds_clim, 
            ds_clim.isel({level: slice(None, smooth_window)})
        ], dim=level)
        
        ds_clim = ds_clim.rolling({level: 2*smooth_window+1}, center=True, min_periods=1).mean()
        ds_clim = ds_clim.isel({level: slice(smooth_window, -smooth_window)})

    dict_chunk = {}
    for dim in ds_clim.dims:
        dict_chunk[dim] = -1
    ds_clim = ds_clim.chunk(dict_chunk)

    # Add interpolated climatology (map week to week)
    ds = add_climatology(ds, ds_clim, field, var_name, level)
    
    ds_clim = ds_clim.rename({field: var_name})

    return ds, ds_clim 


def add_neg_anomaly(ds, field, clim_field, keep_neg_anom):
    """Add anomaly and negative anomaly variables to dataset"""
    # First value will not be filled if it is nan, so drop it
    if ds[field].isel(time=0).isnull().all():
        ds = ds.isel(time=slice(1, None))
    ds[f'anom_{field}'] = ds[field] - ds[clim_field]
    if keep_neg_anom:
        ds[f'neg_anom_{field}'] = xr.where(ds[f'anom_{field}'] < 0, np.abs(ds[f'anom_{field}']), 0)
    
    return ds


def _func_label(var_name, func):
    """Label for a non-quantile reducer: mean -> 'mean', std -> 'std', etc."""
    return f"{var_name}_{str(func)}"


def _quantile_label(var_name, q):
    """0.04 -> 'p04', matching the cold-spell episode convention."""
    perc = f"p{int(round(float(q) * 100)):02d}"
    return f"{var_name}_{perc}" 


def get_time_coordinate(ds):
    """Identify the time dimension name in an xarray Dataset."""
    for name in ds.sizes.keys():
        if ds[name].dtype.kind == 'M':  # numpy datetime64
            return name
    for name in ds.sizes.keys():
        if hasattr(ds[name], 'dt'):
            return name
    raise ValueError(f"No time coordinate found in dataset. Coordinates: {list(ds.sizes)}")


def add_climatology_fields(ds, ds_clim, field, var_name='climatology', level='week'):
    """Add a precomputed climatology back to the original dataset.

    `level` may be a single level name or a list of level names. Selection is
    pointwise on the raw dataset's per-timestep level coordinates, so the
    climatology broadcasts onto every (time, ...) cell. Works for one level
    (e.g. 'week') or several (e.g. ['dayofyear', 'hour']).
    """
    levels = [level] if isinstance(level, str) else list(level)

    # Pointwise selectors: one DataArray per level, all aligned on the raw
    # dataset's time axis. Vectorized .sel matches each timestep to its
    # (dayofyear, hour, ...) climatology value at once.
    selectors = {lvl: ds[lvl] for lvl in levels}
    clim_values = ds_clim[field].sel(selectors)

    ds[var_name] = clim_values
    return ds


def get_window_values(target, targets_sorted, window, cycle_length=None):
    """Values within +/- `window` of `target` along a cyclical axis.

    Two wrap modes:
      - cycle_length given (e.g. 365 for dayofyear, 12 for month, 24 for hour):
        wrap in VALUE space over the full cycle. Correct even when the axis only
        partially covers the cycle (a seasonal subset), since interior windows
        stay in true value space and only the genuine cycle boundary wraps.
        Values produced that aren't present in the data are simply absent from
        the later .isin() match, which is harmless.
      - cycle_length None: wrap over POSITIONS in the sorted unique values
        (use when the unique values themselves form the complete cycle, e.g.
        fixed_days bins 0..52). `window` is then in units of positions/bins.

    Reproduces get_window_days exactly via cycle_length=365 for a 1-based
    dayofyear axis.
    """
    targets_sorted = np.asarray(targets_sorted)
    if cycle_length is not None:
        # Value-space wrap. Assume 1-based for dayofyear/week/month, 0-based for
        # hour; handle both by wrapping on the modulus and restoring the base.
        base = int(targets_sorted.min())
        vals = np.arange(target - window, target + window + 1)
        wrapped = ((vals - base) % cycle_length) + base
        return np.unique(wrapped)
    n = len(targets_sorted)
    pos = int(np.where(targets_sorted == target)[0][0])
    offsets = np.arange(pos - window, pos + window + 1)
    return targets_sorted[offsets % n]


def get_climatology_windowed(
    ds,
    field,
    var_name='climatology',
    levels=('dayofyear', 'hour'),
    window=7,
    window_level=None,
    time_dim=None,
    func='quantile',
    quantiles=None,
):
    """Windowed climatology over one or more levels, added back to the dataset.
    Exactly one level is the WINDOWED axis: for each of its target values, raw
    observations within +/- `window` of it are pooled before reducing. 
    Args:
        levels       : single level name or list (e.g. ['dayofyear', 'hour']).
        window       : +/- half-width for the windowed axis, in that axis' units.
        window_level : which level gets the window. Defaults to 'dayofyear' or first level.
        func         : 'quantile' (needs `quantiles`) or mean/std/min/max/median.
    Returns:
        ds      : original dataset with the climatology variable(s) merged in.
        ds_clim : standalone climatology indexed by the requested levels.
    """
    if time_dim is None:
        time_dim = get_time_coordinate(ds)

    levels = [levels] if isinstance(levels, str) else list(levels)

    if window_level is None:
        window_level = 'dayofyear' if 'dayofyear' in levels else levels[0]
    if window_level not in levels:
        raise ValueError(f"window_level {window_level!r} must be one of levels {levels}.")

    other_levels = [lvl for lvl in levels if lvl != window_level]

    if func == 'quantile':
        if not quantiles:
            raise ValueError("func='quantile' requires a non-empty `quantiles` list.")
        bad = [q for q in quantiles if not (0.0 <= float(q) <= 1.0)]
        if bad:
            raise ValueError(
                f"Quantiles must be fractions in [0, 1]; got out-of-range values: {bad}. "
                "Use 0.04 for the 4th percentile, not 4."
            )

    # Cycle length per windowed level. None -> position/bin wrap (fixed_days),
    # 0 -> no wrap (year).
    cycle_map = {'dayofyear': 365, 'month': 12, 'week': 52, 'hour': 24}
    if window_level in cycle_map:
        cycle_length = cycle_map[window_level]
    elif window_level.startswith('fixed_days_'):
        cycle_length = None          # position/bin wrap over unique bins
    elif window_level == 'year':
        cycle_length = 0             # sentinel: no wrap
    else:
        raise ValueError(f"Unsupported window_level {window_level!r}.")

    # Add all grouping coords.
    ds = add_time_coordinate(ds, time_dim=time_dim, level=levels)

    # Whole group per chunk (quantile requirement) + eager grouper labels.
    ds = ds.chunk({time_dim: -1}).load()

    window_axis_values = ds[window_level]
    targets = np.unique(window_axis_values.values)

    def _support(target):
        if cycle_length == 0:
            # Linear, no wrap (year).
            return np.arange(int(target) - window, int(target) + window + 1)
        return get_window_values(int(target), targets, window, cycle_length=cycle_length)

    def _reduce(obj):
        """Reduce over time, grouping by other_levels exactly if any."""
        if other_levels:
            g = obj.groupby(other_levels[0]) if len(other_levels) == 1 else obj.groupby(other_levels)
            if func == 'quantile':
                return g.quantile(list(quantiles), dim=time_dim)
            return getattr(g, func)(dim=time_dim)
        if func == 'quantile':
            return obj.quantile(list(quantiles), dim=time_dim)
        return getattr(obj, func)(dim=time_dim)

    pieces = []
    for target in targets:
        support = _support(target)
        sub = ds[[field]].where(window_axis_values.isin(support), drop=True)
        if sub[time_dim].size == 0:
            continue
        red = _reduce(sub)
        red = red.expand_dims({window_level: [int(target)]})
        pieces.append(red)

    if not pieces:
        raise ValueError("No data fell within any windowed target value.")

    ds_clim = xr.concat(pieces, dim=window_level)

    def _rechunk(dsc):
        return dsc.chunk({dim: -1 for dim in dsc.dims})

    if func == 'quantile':
        clim_vars = {}
        for q in quantiles:
            name = _quantile_label(var_name, q)
            clim_vars[name] = ds_clim[field].sel(quantile=q).drop_vars('quantile')
        ds_clim = xr.Dataset(clim_vars)
        ds_clim = _rechunk(ds_clim)
        for q in quantiles:
            name = _quantile_label(var_name, q)
            ds = add_climatology_fields(ds, ds_clim, name, var_name=name, level=levels)
    else:
        out_name = _func_label(var_name, func)
        ds_clim = ds_clim.rename({field: out_name})
        ds_clim = _rechunk(ds_clim)
        ds = add_climatology_fields(ds, ds_clim, out_name, var_name=out_name, level=levels)

    return ds, ds_clim


def add_normalized_variable(ds, field, clim_field, std_field, var_name):
    """Create a normalized version of the variable"""
    ds[var_name] = xr.where(
        ds[std_field]>0, (ds[field] - ds[clim_field])/ds[std_field], np.nan
    )
    
    return ds


#———————————————————————————————————————————
# COMPUTE EVENTS FROM DATASET
#———————————————————————————————————————————


def cumulative_flag_counts(flag, dim="time"):
    """Count consecutive runs of 1s along `dim`, resetting to 0 when flag=0.
    Works with Dask-backed arrays.
    """
    cs = flag.cumsum(dim)
    base = cs.where(flag == 0).ffill(dim).fillna(0)
    runs = (cs - base) * flag
    return runs.astype("int32")


def cumulative_flag_values(values, flag, dim="time"):
    """Running sum of `values` within consecutive 1-runs of `flag` along `dim`.

    Same reset logic as cumulative_flag_counts, but accumulates `values` instead
    of counting. Resets to 0 when flag=0, so each episode accumulates from its
    own start. Pass values = intensity to get cumulative cold/heat degree-hours
    within the current episode. Dask-friendly.
    """
    contrib = values * flag
    cs = contrib.cumsum(dim)
    base = cs.where(flag == 0).ffill(dim).fillna(0)
    return (cs - base) * flag


def get_extreme_events(ds, field, var_name, threshold, num_days=3, side = 'lower'):
    """Add flag of events, count days of events to dataset"""
    time = get_time_coordinate(ds)
    flag_name = f'flag_{var_name}'
    if side == 'lower':
        ds[flag_name] = xr.where(ds[field]<=threshold, 1, 0)
    elif side == 'upper':
        ds[flag_name] = xr.where(ds[field]>=threshold, 1, 0)
    
    count_name = f'n_{var_name}'
    #ds[count_name] = ds[flag_name].rolling({time: num_days}, min_periods=num_days).sum()
    ds[count_name] = cumulative_flag_counts(ds[flag_name], dim=time)
    ds[var_name] = xr.where(ds[count_name]==num_days, 1, 0)

    return ds


def flag_and_intensity(ds, field, threshold_field, side='lower', hard_threshold=None):
    """Flag extreme cells and measure their intensity against a per-cell threshold.
    Args:
        ds: Dataset containing `field` and `threshold_field`.
        field: Name of the observation variable.
        threshold_field: Name of the per-cell threshold variable.
        side: 'lower' flags obs <= threshold, 'upper' flags obs >= threshold.
        hard_threshold: Optional absolute cutoff.
    Returns:
        ds with flag_{var_name} and intensity_{var_name} added.
    """
    obs = ds[field]
    thr = ds[threshold_field]

    if side == 'lower':
        cond = obs <= thr
        if hard_threshold is not None:
            cond = cond & (obs <= hard_threshold)
        depth = (thr - obs).clip(min=0)
    elif side == 'upper':
        cond = obs >= thr
        if hard_threshold is not None:
            cond = cond & (obs >= hard_threshold)
        depth = (obs - thr).clip(min=0)
    else:
        raise ValueError(f"side must be 'lower' or 'upper', got {side!r}")

    flag = xr.where(cond, 1, 0)
    intensity = depth.where(cond, 0)
    flag = xr.where(obs.isnull(), np.nan, flag)
    intensity = xr.where(obs.isnull(), np.nan, intensity)
    return flag, intensity


def get_extreme_episodes(
    ds,
    flag_field,
    intensity_field,
    var_name='cold_spell',
    #num_periods=2,
    time_dim=None,
):
    """Accumulate consecutive runs and intensity from precomputed flag/intensity.
    Args:
        ds: Dataset containing `flag_field` and `intensity_field`.
        flag_field: Name of the precomputed 1/0 flag variable.
        intensity_field: Name of the precomputed intensity variable.
        var_name: Base name for the output variables.
        num_periods: Run length at which an episode is marked.
        time_dim: Time dimension name; inferred if None.
    Returns:
        ds with n_, cum_intensity_, and {var_name} added.
    """
    if time_dim is None:
        time_dim = get_time_coordinate(ds)

    flag = ds[flag_field]
    intensity = ds[intensity_field]

    count_name = f'n_{var_name}'
    cdh_name = f'cum_intensity_{var_name}'

    ds[count_name] = cumulative_flag_counts(flag, dim=time_dim)
    ds[cdh_name] = cumulative_flag_values(intensity, flag, dim=time_dim)
    #ds[var_name] = xr.where(ds[count_name] == num_periods, 1, 0)

    return ds


#———————————————————————————————————————————
# SPATIAL VALIDATION
#———————————————————————————————————————————

def spatial_smooth(da: xr.DataArray, pixel_window: int = 5,
                   lat_dim: str = 'lat', lon_dim: str = 'lon') -> xr.DataArray:
    """
    Apply spatial moving-average smoothing over lat/lon only.
    Args:
        da: Input DataArray.
        pixel_window: Size of the moving window along each spatial axis.
        lat_dim, lon_dim: Names of the spatial dimensions to smooth.
    Returns:
        Smoothed DataArray with the same coords and dims as the input.
    """
    lon, lat, time = get_coordinates(da)
    lon_dim =  lon if lon is not None else lon_dim
    lat_dim =  lat if lat is not None else lat_dim
    sizes = tuple(pixel_window if d in (lat_dim, lon_dim) else 1
                  for d in da.dims)

    #smoothed = uniform_filter(da.values, size=sizes, mode='nearest')
    smoothed = da.rolling({lon_dim: pixel_window, lat_dim: pixel_window}, center=True, min_periods=1).sum()

    return xr.DataArray(smoothed, coords=da.coords, dims=da.dims,
                        name=f'{da.name}_smooth_{pixel_window}')

def spatial_rolling_stat(
    da: xr.DataArray, 
    pixel_window: int = 5,
    stat: str = 'mean',
    lat_dim: str = 'lat', 
    lon_dim: str = 'lon',
    min_periods: int = 1
) -> xr.DataArray:
    """
    Apply a spatial rolling-window statistic over lat/lon only.
    Args:
        da: Input DataArray.
        window_size: Size of the moving window along each spatial axis.
        stat: Name of the rolling reducer to apply ('mean', 'sum', 'std',
            'median', 'max', 'min', etc.).
        lat_dim, lon_dim: Names of the spatial dimensions to smooth.
        min_periods: Minimum valid pixels in a window for a non-NaN result.
    Returns:
        DataArray of the windowed statistic, same coords and dims as input.
    """
    lon, lat, time = get_coordinates(da)
    lon_dim =  lon if lon is not None else lon_dim
    lat_dim =  lat if lat is not None else lat_dim
    roll = da.rolling({lat_dim: pixel_window, lon_dim: pixel_window},
                      center=True, min_periods=min_periods)

    try:
        reducer = getattr(roll, stat)
    except AttributeError:
        raise ValueError(f"Unknown rolling stat: {stat!r}")

    smoothed = reducer()

    return smoothed.rename(f'{da.name}_{stat}_{pixel_window}')