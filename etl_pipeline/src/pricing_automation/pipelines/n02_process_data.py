from .utils import *

from .a02_create_clusters import *
from .a02_preprocess_planet_data import *

#———————————————————————————————————————————
# AUXILIARY FUNCTIONS
#———————————————————————————————————————————


def _get_time_coordinate(ds):
    """Identify the time dimension name in an xarray Dataset."""
    for name in ds.sizes.keys():
        if ds[name].dtype.kind == 'M':  # numpy datetime64
            return name
    for name in ds.sizes.keys():
        if hasattr(ds[name], 'dt'):
            return name
    raise ValueError(f"No time coordinate found in dataset. Coordinates: {list(ds.sizes)}")

def add_time_coordinate(ds_orig, level='week', time_dim=None, n_days=7):
    """Add one or more climatology grouping coordinates.
    `level` may be a single level or a list. Each is added via the original
    single-level logic. Returns the dataset with every requested coord present.
    """
    ds = ds_orig.copy()
    if time_dim is None:
        time_dim = _get_time_coordinate(ds)

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
    time = _get_time_coordinate(ds)
    if smooth_window>0:
        da = ds[field].rolling({time: smooth_window}, min_periods=1).mean()
    return da


RENAME_DICT = {
    'swc': ['swc', 'swvl1', 'soil_water_content'],
    'prcp': ['prcp', 'tp', 'total_precipitation', 'precipitation', 'precip', 'prc', 'pcp'],
    'tmin': ['tmin', 't2m', '2t', 't', 'mn2t6', 'mn2t', 'mn2t24'],
    'tmax': ['tmax', 't2m', '2t', 't', 'mx2t6', 'mx2t', 'mx2t24'],
}

#———————————————————————————————————————————
# RENAME VARIABLES
#———————————————————————————————————————————


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
# SUMMARIZE DATASET
#———————————————————————————————————————————


def summarize_data(ds, red_dims=[None], group_coords=[None], func='mean'):
    """Summarize data along the specified coordinates"""    
    if set(red_dims).issubset(set(list(ds.sizes.keys()))):
        # Average over all grouping columns (dimensions)
        if func == 'mean':
            ds = ds.mean(dim = red_dims)
        elif func == 'min':
            ds = ds.min(dim = red_dims)
        elif func == 'max':
            ds = ds.max(dim = red_dims)
        elif func == 'median':
            ds = ds.median(dim = red_dims)
        elif func == 'sum':
            ds = ds.sum(dim = red_dims)
        else: 
            raise ValueError(f'The summarizing function {func} is not implemented.')
    elif set(group_coords).issubset(set(list(ds.coords))):
        for coord in group_coords:
            if func == 'mean':
                ds = ds.groupby(coord).mean()
            elif func == 'min':
                ds = ds.groupby(coord).max()
            elif func == 'max':
                ds = ds.groupby(coord).max()
            elif func == 'median':
                ds = ds.groupby(coord).median()
            elif func == 'sum':
                ds = ds.groupby(coord).sum()
            else: 
                raise ValueError(f'The summarizing function {func} is not implemented.')
    else:
        raise ValueError(f"Input for red_cols must be a subset of dimensions")
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

'''
def get_cdf_fixed_data(
    ds: xr.Dataset, 
    ds_base: xr.Dataset, 
    field_target: str, 
    field_base: str,
    group_cols: list, 
    level: str = None,
    window_size: int = 30
) -> xr.DataArray:
    """
    Performs a CDF Matching of original dataset field based on values from base dataset field
    Args:
        - ds : (xr.Dataset) Original dataset on which CDF matching will be applied
        - ds_base : (xr.Dataset) Base dataset used for CDF matching
        - field : (str) Name of field to use for CDF
        - group_cols : (list) List of names to group on for CDF (must be a subset of ds.dims)
        - level : (str) Name of time coordinate to group on if None the distribution will be computed from all values
    Returns:
        - xr.DataArray with the fixed valued after applying CDF Matching
    """
    # Get dimensions and checking that group_cols is a subset of the dimensions
    dims = list(ds.sizes.keys())
    if not set(group_cols).issubset(set(dims)):
        raise ValueError('group_cols must be a subset of dims from first Dataset')
    #group_cols = ['climate_area_id']

    # Adding time coordinate to compute CDF
    if level is not None:
        ds_o = add_time_coordinate(ds, level=level)
        ds_b = add_time_coordinate(ds_base, level=level)

    # Convert temporarily to dataframes
    df_o = ds_o.to_dataframe().reset_index()
    df_b = ds_b.to_dataframe().reset_index()
    
    def cdf_matching_day(day):
        list_days = get_window_days(day, window=window_size)
        df_orig = df_o[df_o['dayofyear'] == day].copy()
        df_base = df_b[df_b['dayofyear'].isin(list_days)].copy()

        if df_orig.empty or df_base.empty:
            return

        # Build the base distribution from NON-NULL values only
        df_base = df_base.dropna(subset=[field_base])
        if df_base.empty:
            return

        df_base["rank"] = df_base.groupby(group_cols)[field_base].rank(method="first")
        # count() excludes NaN; size() does not. We already dropped NaN, but use count to be safe.
        df_aux = df_base.groupby(group_cols, as_index=False)[field_base].count()
        df_aux = df_aux.rename(columns={field_base: 'size'})

        # prop_rank: rank() already returns NaN for NaN source values, so they propagate correctly
        df_orig["prop_rank"] = df_orig.groupby(group_cols)[field_target].rank(method="max", pct=True)

        df_orig = df_orig.merge(df_aux, how='left', on=group_cols)
        df_orig['rank'] = np.floor(df_orig['prop_rank'] * df_orig['size'])

        # Clamp rank into the valid range [1, size]: prop_rank==1.0 gives floor(1.0*size)=size,
        # which is a valid rank; but floating error can push it to size, and prop_rank near 0
        # can give 0. Ranks from method="first" run 1..size, so clamp to that.
        df_orig['rank'] = df_orig['rank'].clip(lower=1)

        drop_cols = ["rank"] if level is None else [level, "rank"]

        df_base_dedup = (
            df_base[group_cols + ['rank', field_base]]
            .drop_duplicates(subset=group_cols + ['rank'], keep='first')
        )

        df_orig = df_orig[dims + drop_cols].merge(
            df_base_dedup,
            how='left',
            on=group_cols + ['rank']
        )
        df_orig = df_orig.drop(columns=drop_cols)
        return df_orig

    df_list = Parallel(n_jobs=6)(
        delayed(cdf_matching_day)(day) for day in np.arange(1, 367)
    )

    df_list = pd.concat(df_list, axis=0, ignore_index=True)

    df_list = df_list.sort_values(by = group_cols + ['time'])

    da = df_list.set_index(dims).to_xarray()[field_base]

    return da
'''

'''
def get_cdf_fixed_data(
    ds: xr.Dataset, 
    ds_base: xr.Dataset, 
    field_target: str, 
    field_base: str,
    group_cols: list, 
    level: str = None,
    window_size: int = 30
) -> xr.DataArray:
    """
    Performs a CDF Matching of original dataset field based on values from base dataset field
    Args:
        - ds : (xr.Dataset) Original dataset on which CDF matching will be applied
        - ds_base : (xr.Dataset) Base dataset used for CDF matching
        - field : (str) Name of field to use for CDF
        - group_cols : (list) List of names to group on for CDF (must be a subset of ds.dims)
        - level : (str) Name of time coordinate to group on if None the distribution will be computed from all values
    Returns:
        - xr.DataArray with the fixed valued after applying CDF Matching
    """
    # Get dimensions and checking that group_cols is a subset of the dimensions
    dims = list(ds.sizes.keys())
    if not set(group_cols).issubset(set(dims)):
        raise ValueError('group_cols must be a subset of dims from first Dataset')
    #group_cols = ['climate_area_id']

    # Adding time coordinate to compute CDF
    if level is not None:
        ds_o = add_time_coordinate(ds, level=level)
        ds_b = add_time_coordinate(ds_base, level=level)

    # Convert temporarily to dataframes
    df_o = ds_o.to_dataframe().reset_index()
    df_b = ds_b.to_dataframe().reset_index()
    
    def cdf_matching_day(day):
        list_days = get_window_days(day, window=window_size)
        df_orig = df_o[df_o['dayofyear']==day].copy()
        df_base = df_b[df_b['dayofyear'].isin(list_days)].copy()

        if df_orig.empty or df_base.empty:
            return

        df_base["rank"] = df_base.groupby(group_cols)[field_base].rank(method="first")
        df_aux = df_base.groupby(group_cols, as_index=False)[field_base].size()

        df_orig["prop_rank"] = df_orig.groupby(group_cols)[field_target].rank(method="max", pct=True)

        # Compute absolute ranking compatible with base dataset
        df_orig = df_orig.merge(
            df_aux,
            how = 'left',
            on = group_cols
        )
        df_orig['rank'] = np.floor(df_orig['prop_rank'] * df_orig['size'])

        drop_cols = ["rank"] if level is None else [level, "rank"]

        df_base_dedup = (
            df_base[group_cols + ['rank', field_base]]
            .drop_duplicates(subset=group_cols + ['rank'], keep='first')
        )

        # Get the field from base dataset based on joins by group_cols. level and ranking
        df_orig = df_orig[dims + drop_cols].merge(
            df_base_dedup,
            how='left',
            on=group_cols + ['rank']
        )
        df_orig = df_orig.drop(columns=drop_cols)
        return df_orig

    df_list = Parallel(n_jobs=6)(
        delayed(cdf_matching_day)(day) for day in np.arange(1, 367)
    )

    df_list = pd.concat(df_list, axis=0, ignore_index=True)

    df_list = df_list.sort_values(by = group_cols + ['time'])

    da = df_list.set_index(dims).to_xarray()[field_base]

    return da
'''

'''
def get_cdf_fixed_data(
    ds: xr.Dataset,
    ds_base: xr.Dataset,
    field_target: str,
    field_base: str,
    group_cols: list,
    level: str = None,
    window_size: int = 30
) -> xr.DataArray:
    """
    Performs a CDF Matching of original dataset field based on values from base dataset field
    Args:
        - ds : (xr.Dataset) Original dataset on which CDF matching will be applied
        - ds_base : (xr.Dataset) Base dataset used for CDF matching
        - field : (str) Name of field to use for CDF
        - group_cols : (list) List of names to group on for CDF (must be a subset of ds.dims)
        - level : (str) Name of time coordinate to group on if None the distribution will be computed from all values
    Returns:
        - xr.DataArray with the fixed valued after applying CDF Matching
    """
    # Get dimensions and check that group_cols is a subset of the dimensions
    dims = list(ds.sizes.keys())
    if not set(group_cols).issubset(set(dims)):
        raise ValueError('group_cols must be a subset of dims from first Dataset')
 
    # Add time coordinate to compute CDF
    if level is not None:
        ds_o = add_time_coordinate(ds, level=level)
        ds_b = add_time_coordinate(ds_base, level=level)
    else:
        ds_o = ds
        ds_b = ds_base
 
    # Convert temporarily to dataframes
    df_o = ds_o.to_dataframe().reset_index()
    df_b = ds_b.to_dataframe().reset_index()
 
    # ----- OPTIONAL upper-tail winsorization -----
    # Genuine target outliers can still be pushed into the extreme tail of the
    # base distribution. To cap that, clip the target CDF probability before
    # interpolation. Disabled by default; set 0 < pclip < 1 to enable.
    pclip = None  # e.g. pclip = 0.99
 
    def _map_group(df_orig_g, base_vals):
        """Map one group's target CDF probability onto the base empirical quantiles."""
        out = df_orig_g.copy()
        if base_vals.size == 0:
            out[field_base] = np.nan
            return out
        bq = np.sort(base_vals)
        # Plotting-position probabilities for the sorted base sample
        qp = (np.arange(1, bq.size + 1) - 0.5) / bq.size
        # Target empirical CDF probability within this day-of-year group
        pr = out[field_target].rank(method='average', pct=True).values
        if pclip is not None:
            pr = np.clip(pr, None, pclip)
        out[field_base] = np.interp(pr, qp, bq)
        return out
 
    def cdf_matching_day(day):
        list_days = get_window_days(day, window=window_size)
        df_orig = df_o[df_o['dayofyear'] == day].copy()
        df_base = df_b[df_b['dayofyear'].isin(list_days)].copy()
 
        if df_orig.empty or df_base.empty:
            return
 
        # Per-group base samples (pooled over the day-of-year window)
        base_groups = {
            key: sub[field_base].dropna().values
            for key, sub in df_base.groupby(group_cols)
        }
 
        # Map each target group through its base empirical quantiles
        pieces = []
        for key, sub in df_orig.groupby(group_cols):
            base_vals = base_groups.get(key, np.array([]))
            pieces.append(_map_group(sub, base_vals))
 
        df_orig = pd.concat(pieces, axis=0)
 
        return df_orig[dims + [field_base]]
 
    df_list = Parallel(n_jobs=6)(
        delayed(cdf_matching_day)(day) for day in np.arange(1, 367)
    )
 
    df_list = pd.concat([d for d in df_list if d is not None], axis=0, ignore_index=True)
    df_list = df_list.sort_values(by=group_cols + ['time'])
 
    da = df_list.set_index(dims).to_xarray()[field_base]
 
    return da
'''

#———————————————————————————————————————————
# ASSIGN CLUSTER COORDINATE TO DATASET
#———————————————————————————————————————————


def create_cluster_coord(ds_orig, gdf_orig, cluster_var, method='within', check_var=None, k=6):
    ds = ds_orig.copy()
    gdf = gdf_orig.copy()

    # Extract lon and lat from the dataset
    lon_var, lat_var, time_var = get_coordinates(ds)
    points = ds[[lat_var, lon_var]].to_dataframe().reset_index()

    # Create a DataFrame of points
    points = gpd.GeoDataFrame(
        points,
        geometry=gpd.points_from_xy(points[lon_var], points[lat_var]),
        crs=gdf.crs
    )

    # Perform spatial join
    if method == 'within':
        points = gpd.sjoin(points, gdf[[cluster_var, "geometry"]], how="left", predicate="within")
        # Assign as coordinate to the dataset
        ds_points = points.set_index([lat_var, lon_var])[[cluster_var]]
        ds_points = ds_points.to_xarray().set_coords(cluster_var)
        ds = xr.merge([ds, ds_points])
        # Create a pixel assignation dataframe
        df_cluster = ds[[lon_var, lat_var, cluster_var]].to_dataframe()
        df_cluster = df_cluster.reset_index().dropna(subset=[cluster_var])

    elif method == 'nearest':
        # Match nearest coordinates from grid to centroid of polygons
        gdf['lon'] = gdf.centroid.x
        gdf['lat'] = gdf.centroid.y
        # Filter only valid points by checking if check_var is all null
        if check_var:
            valid_count = (~ds[check_var].isnull()).sum(dim=time_var)
            # Convert the 2-D (lat, lon) DataArray to a plain DataFrame.
            # Avoid ds[[lon_var, lat_var, 'valid_count']] because that only
            # selects *data variables*, and lon/lat/valid_count are coordinates.
            points = valid_count.to_dataframe(name='valid_count').reset_index()
            points = points[points['valid_count'] != 0].copy()

        df_cluster = match_grid_points(points, gdf, k=k)
        df_cluster = df_cluster[[lon_var, lat_var, cluster_var]].copy()

        ds = select_coordinates(ds, df_cluster, grid_coords=(lon_var, lat_var))

    df_cluster['method'] = method

    return ds, df_cluster


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
        time_dim = _get_time_coordinate(ds)
    
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


def _func_label(func):
    """Label for a non-quantile reducer: mean -> 'mean', std -> 'std', etc."""
    return str(func)


def _quantile_label(q):
    """0.04 -> 'p04', matching the cold-spell episode convention."""
    return f"p{int(round(float(q) * 100)):02d}"

def _get_time_coordinate(ds):
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
        levels       : single level name or list (e.g. 'fixed_days_7' or
                       ['dayofyear', 'hour'] or ['month', 'hour']).
        window       : +/- half-width for the windowed axis, in that axis' units.
        window_level : which level gets the window. Defaults to 'dayofyear' if
                       present, else the first level.
        func         : 'quantile' (needs `quantiles`) or mean/std/min/max/median.

    Returns:
        ds      : original dataset with the climatology variable(s) merged in.
        ds_clim : standalone climatology indexed by the requested levels.
    """
    if time_dim is None:
        time_dim = _get_time_coordinate(ds)

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
            name = f"{var_name}_{_quantile_label(q)}"
            clim_vars[name] = ds_clim[field].sel(quantile=q).drop_vars('quantile')
        ds_clim = xr.Dataset(clim_vars)
        ds_clim = _rechunk(ds_clim)
        for q in quantiles:
            name = f"{var_name}_{_quantile_label(q)}"
            ds = add_climatology_fields(ds, ds_clim, name, var_name=name, level=levels)
    else:
        out_name = f"{var_name}_{_func_label(func)}"
        ds_clim = ds_clim.rename({field: out_name})
        ds_clim = _rechunk(ds_clim)
        ds = add_climatology_fields(ds, ds_clim, out_name, var_name=out_name, level=levels)

    return ds, ds_clim


#———————————————————————————————————————————
# COMPUTE EVENTS FROM DATASET
#———————————————————————————————————————————


def cumulative_flag_runs(flag, dim="time"):
    """
    Count consecutive runs of 1s along `dim`, resetting to 0 when flag=0. Works with Dask-backed arrays.
    """
    # Cumulative sum of flag
    cs = flag.cumsum(dim)
    # Value of cumsum at last zero
    reset = cs.where(flag == 0).ffill(dim).fillna(0)
    # Subtract to reset counting
    runs = (cs - reset) * flag

    return runs.astype("int32")


def get_extreme_events(ds, field, var_name, threshold, num_days=3, side = 'lower'):
    """Add flag of events, count days of events to dataset"""
    time = _get_time_coordinate(ds)
    flag_name = f'flag_{var_name}'
    if side == 'lower':
        ds[flag_name] = xr.where(ds[field]<=threshold, 1, 0)
    elif side == 'upper':
        ds[flag_name] = xr.where(ds[field]>=threshold, 1, 0)
    
    count_name = f'n_{var_name}'
    #ds[count_name] = ds[flag_name].rolling({time: num_days}, min_periods=num_days).sum()
    ds[count_name] = cumulative_flag_runs(ds[flag_name], dim=time)
    ds[var_name] = xr.where(ds[count_name]==num_days, 1, 0)

    return ds
    

def add_normalized_variable(ds, field, clim_field, std_field, var_name):
    """Create a normalized version of the variable"""
    ds[var_name] = xr.where(
        ds[std_field]>0, (ds[field] - ds[clim_field])/ds[std_field], np.nan
    )
    
    return ds


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

    Same reset logic as cumulative_flag_runs, but accumulates `values` instead
    of counting. Resets to 0 when flag=0, so each episode accumulates from its
    own start. Pass values = intensity to get cumulative cold/heat degree-hours
    within the current episode. Dask-friendly.
    """
    contrib = values * flag
    cs = contrib.cumsum(dim)
    base = cs.where(flag == 0).ffill(dim).fillna(0)
    return (cs - base) * flag


def get_extreme_episodes(
    ds,
    field,
    threshold_field,
    var_name='cold_spell',
    num_periods=2,
    side='lower',
    time_dim=None,
):
    """Flag below/above-threshold cells, count consecutive runs, measure intensity.

    Compares each cell of `field` against the per-cell, per-timestep threshold in
    `threshold_field` (e.g. the '{var}_p04' climatology merged onto ds by
    get_climatology_windowed), so every cell is judged against its own percentile.

    Adds:
        flag_{var_name}      : 1 where below (lower)/above (upper) threshold.
        n_{var_name}         : consecutive run length along time, reset on flag=0.
        intensity_{var_name} : threshold-obs (lower)/obs-threshold (upper),
                               clipped at 0. Per-cell cold/heat depth.
        {var_name}           : 1 where the run length reaches num_periods.

    Returns ds with the columns added.
    """
    if time_dim is None:
        time_dim = _get_time_coordinate(ds)

    obs = ds[field]
    thr = ds[threshold_field]

    flag_name = f'flag_{var_name}'
    count_name = f'n_{var_name}'
    intensity_name = f'intensity_{var_name}'

    if side == 'lower':
        ds[flag_name] = xr.where(obs <= thr, 1, 0)
        ds[intensity_name] = (thr - obs).clip(min=0)
    elif side == 'upper':
        ds[flag_name] = xr.where(obs >= thr, 1, 0)
        ds[intensity_name] = (obs - thr).clip(min=0)
    else:
        raise ValueError(f"side must be 'lower' or 'upper', got {side!r}")

    ds[count_name] = cumulative_flag_counts(ds[flag_name], dim=time_dim)

    cdh_name = f'cum_intensity_{var_name}'
    ds[cdh_name] = cumulative_flag_values(ds[intensity_name], ds[flag_name], dim=time_dim)

    ds[var_name] = xr.where(ds[count_name] == num_periods, 1, 0)

    return ds


#———————————————————————————————————————————————————————————————
# SLICE MULTIPLE GEOMETRIES
#———————————————————————————————————————————————————————————————

_AREAS_DEFAULTS = {
    "variable":     None,   # required
    "location_var": None,   # required
    "cluster_var":  "cluster_id",
    "min_area_km2": 125,
    "min_pixels":   4,
    "min_clusters": 4,
    "n_comp":       3,
    "bandwidth":    0.03,
    "grid_res":     300,
}

def _slice_single_geometry(ds_orig, gdf_orig, params_areas):
    """
    Slice one polygon into climate-homogeneous sub-polygons using PCA + KMeans + Voronoi.
    Returns the original geometry as a single area when:
      - area < min_area_km2
      - no pixels fall within the geometry
      - n_areas or n_comp collapse to <= 1 after all caps
    """
    p = _AREAS_DEFAULTS | params_areas
 
    variable     = p["variable"]
    location_var = p["location_var"]
    cluster_var  = p["cluster_var"]
    min_area_km2 = p["min_area_km2"]
    max_area_km2 = p["max_area_km2"]
    min_clusters = p["min_clusters"]
    n_comp       = p["n_comp"]
 
    if variable is None or location_var is None:
        raise ValueError("params_areas must include 'variable' and 'location_var'.")
 
    gdf = gdf_orig.copy()
 
    # Area guard
    gdf = add_area_column(gdf, "_area_km2")
    area_km2 = gdf["_area_km2"].iloc[0]
    gdf = gdf.drop(columns=["_area_km2"])
 
    def _single_area(gdf, cluster_var):
        gdf = gdf.copy()
        gdf.insert(0, cluster_var, 0)
        return gdf
 
    if area_km2 < min_area_km2:
        return _single_area(gdf, cluster_var)
 
    # Select pixels within this geometry
    ds = ds_orig.copy()
    lon, lat, time = get_coordinates(ds)

    ds = rename_vars(ds, variable)
 
    ds = slice_dataset_w_geometry(ds, gdf)
    ds, _ = create_cluster_coord(ds, gdf, location_var, method="within")
 
    if sum(ds[location_var].isnull().values.ravel() == False) == 0:  # noqa: E712
        return _single_area(gdf, cluster_var)
 
    # Build pixel-level feature matrix (pixels x time)
    df_sum = ds.to_dataframe().reset_index()
    df_sum = df_sum.dropna(subset=[location_var]).copy()
    df_sum[time] = pd.to_datetime(df_sum[time])
 
    df = df_sum.pivot_table(
        index=[lat, lon], columns=[time], values=[variable], aggfunc="mean"
    )
    df.columns = [f"{col[0]}_{col[1]}" for col in df.columns]
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
 
    n_pixels = df.shape[0]
 
    # n_areas: max of min_clusters and area-based term, capped by available pixels
    n_areas = max(min_clusters, max(1, int(area_km2 / max_area_km2)))
    n_areas = min(n_areas, n_pixels)
    n_comp_eff = min(n_comp, n_pixels, n_areas)
 
    if n_areas <= 1 or n_comp_eff < 1:
        return _single_area(gdf, cluster_var)
 
    # PCA + KMeans
    df_pca = apply_pca(df, n_comp_eff)
    df_pca[cluster_var] = get_cluster(df_pca, n_areas)
    df_pca = df_pca.reset_index()
 
    # Polygonize clusters via Voronoi and intersect with original geometry
    gdf_cluster = create_voronoi_polygons(gdf, df_pca, cluster_var, lon, lat)
    gdf_cluster = gdf_cluster.to_crs(gdf.crs)
    gdf_cluster = gpd.overlay(gdf_cluster, gdf, how="intersection")
    gdf_cluster = clean_polygons(gdf_cluster)
    if cluster_var in gdf_cluster.columns:
        gdf_cluster = gdf_cluster.drop(columns=[cluster_var])
    gdf_cluster = gdf_cluster.reset_index(names=cluster_var)

    gdf_cluster = gdf_cluster.rename(columns={location_var: f'{location_var}_orig'})
    gdf_cluster[location_var] = (
        gdf_cluster[f'{location_var}_orig'].astype(str) + '-' + 
        gdf_cluster[cluster_var].astype(str).str.rjust(2,"0")
    )
    gdf_cluster = add_area_column(gdf_cluster)
 
    return gdf_cluster


def create_climate_areas(
    gdf: gpd.GeoDataFrame,
    ds: xr.Dataset,
    params_areas: dict,
    params_s: dict = {},
) -> gpd.GeoDataFrame:
    """
    Slice every polygon in a GeoDataFrame into climate-homogeneous sub-polygons.
    Parameters:
        gdf : GeoDataFrame that must contain the column specified by ``location_var``.
        ds : xr.Dataset with climate dataset covering all geometries in ``gdf``.
        params_areas : dict parameters. Required: ``variable``, ``location_var``.
            Optional (see ``_AREAS_DEFAULTS`` for defaults):
            ``cluster_var``, ``min_area_km2``, ``min_pixels``, ``min_clusters``, ``n_comp``.
        params_s : dict (optional) passed to ``subset_geometry`` to filter ``gdf`` before processing.
            Supports ``include`` and ``exclude`` sub-dicts keyed by column name.
    Returns:
        GeoDataFrame. One row per sub-polygon with all original columns retained and a new ``cluster_var``.
    """
    cluster_var = params_areas.get("cluster_var", 'cluster_id')
    id_col = params_areas.get("location_var", "location_id")
 
    gdf_work = subset_geometry(gdf, params_s).copy()
    if len(gdf_work) == 0:
        raise ValueError("subset_geometry returned an empty GeoDataFrame. Check params_s.")
 
    results = []
    list_geoms = np.sort(gdf_work[id_col].unique())
    for geom_id in list_geoms:
        gdf_single = gdf_work[gdf_work[id_col] == geom_id].copy()
        try:
            gdf_sliced = _slice_single_geometry(ds, gdf_single, params_areas)
            print(f'[{geom_id}] done', sep=' ')
        except Exception as exc:
            print('')
            print(f"[{geom_id}] slicing failed ({exc}), keeping original geometry.")
            gdf_single.insert(0, cluster_var, 0)
            gdf_sliced = gdf_single
        results.append(gdf_sliced)
 
    return pd.concat(results, axis=0, ignore_index=True)


#———————————————————————————————————————————
# PROCESS DATASET FROM PLANET
#———————————————————————————————————————————


def preprocess_planet(df, params_transform, n_jobs=6):
    """
    Process all tiff files from Planeet in S3
    Args:
        df : pd.DataFrame of the subscription catalog with keys and details
        params_transform : dict of parameters to look for data

    Returns:
        ds_full : xr.Dataset of processed tiffs
    """
    
    list_files = list_files_from_bucket(df, params_transform)
    print('Third version')

    ds_list = Parallel(n_jobs=n_jobs)(
        delayed(process_tiff)(file) for file in list_files
    )

    valid_ds = [ds for ds in ds_list if ds is not None]
    n_dropped = len(ds_list) - len(valid_ds)
    print(f"Files processed: {len(ds_list)} | Dropped (None): {n_dropped} | Valid: {len(valid_ds)}")

    if not valid_ds:
        raise ValueError("No valid datasets after processing. All files returned None.")

    ds_full = xr.concat(valid_ds, dim='time')
    ds_full = ds_full.drop_duplicates(dim=['time', 'lat', 'lon'], keep='first')

    return ds_full

#———————————————————————————————————————————
# PROCESS DATASET FROM PLANET
#———————————————————————————————————————————

'''
def process_data_planet(
    ds1_orig: xr.Dataset, 
    ds2_orig: xr.Dataset, 
    ds_era_orig: xr.Dataset, 
    gdf: gpd.GeoDataFrame, 
    params: dict
) -> tuple:
    """
    Clean and process dataset to get unique time series and add anomalies and climatologies
    Args:
        - ds1_orig : (xr.Dataset) Dataset from first period of Planet
        - ds_era_orig : (xr.Dataset) Dataset from ERA5 to fill gap
        - ds2_orig : (xr.Dataset) Dataset from second period of Planet 
        - gdf : (gpd.GeoDataFrame) GeoDataFrame with geometries
        - params : (dict) Dictionary with parameters to process data
    
    Returns:
        - df_concat : pd.DataFrame with processed data
        - ds_clim : xr.Dataset with climatologies
    """
    # Read parameters from dictionary
    subset = params['subset']
    #location_var = params['location_var']
    variable = params['variable']
    na_replace_era5 = params['na_replace']['era5']
    na_replace_planet = params['na_replace']['planet']
    period_planet1 = params['time_window']['planet1']
    period_planet2 = params['time_window']['planet2']
    period_era5 = params['time_window']['era5']
    interp_resolution = params.get('interp_resolution', 0.01)
    fill_gap_window = params.get('fill_gap_window', 7)
    cdf_time_window = params.get('cdf_time_window', 45)
    time_smooth_window = params.get('time_smooth_window', 21)
    clim_smooth_window = params.get('climatology_smooth_window', 7)
    add_abs_anom = params.get('add_abs_anom', True)
 
    # Subset climate area geometry
    LOCATION_NAME = 'location_id'
    gdf_ca = rename_subset_geometry(gdf, subset, LOCATION_NAME)
 
    # Slice dataset using the given geometry
    ds_era = slice_dataset_w_geometry(ds_era_orig, gdf_ca)
 
    # Interpolate data from ERA5 to a finer resolution
    ds_era = regrid_dataset(ds_era, method='linear', res=interp_resolution)
 
    # Rename variables to use uniform names
    ds1_orig = rename_vars(ds1_orig, variable)
    ds2_orig = rename_vars(ds2_orig, variable)
    ds_era = rename_vars(ds_era, variable)
 
    # Remove unnecesary coordinates
    keep_coords = list(get_coordinates(ds_era)) # list of coords to keep
    ds_era = drop_single_coords(ds_era, except_coords=keep_coords)[[variable]] 
 
    keep_coords_1 = list(get_coordinates(ds1_orig)) # list of coords to keep
    ds1_orig = drop_single_coords(ds1_orig, except_coords=keep_coords)[[variable]]
 
    keep_coords_1 = list(get_coordinates(ds2_orig)) # list of coords to keep
    ds2_orig = drop_single_coords(ds2_orig, except_coords=keep_coords)[[variable]] 
 
    # Clean data with time slices and na replace values
    ds_period1 = clean_data(ds1_orig, date_range=period_planet1, var=variable, na_replace=na_replace_planet)
    ds_period2 = clean_data(ds2_orig, date_range=period_planet2, var=variable, na_replace=na_replace_planet)
    ds_gap = clean_data(ds_era, date_range = ('2002-01-01', None), var=variable, na_replace=na_replace_era5)
    print(f"Cleaning done")
 
    ds_period1, df1_ = create_cluster_coord(ds_period1, gdf_ca, LOCATION_NAME) #Using v2 is faster since it applies a spatial join
    ds_period2, df2_ = create_cluster_coord(ds_period2, gdf_ca, LOCATION_NAME) #Using v2 is faster since it applies a spatial join
    ds_gap, dfg_ = create_cluster_coord(ds_gap, gdf_ca, LOCATION_NAME) #Using v2 is faster since it applies a spatial join
 
    # Summarize the arrays along a specified dimension
    ds_period1 = summarize_data(ds_period1, group_coords=[LOCATION_NAME], func='mean')
    ds_period2 = summarize_data(ds_period2, group_coords=[LOCATION_NAME], func='mean')
    ds_gap = summarize_data(ds_gap, group_coords=[LOCATION_NAME], func='mean')
    print(f"Clustering and summarizing done")
 
    # Planet harmonization:
    #   - AMSR2 (ds_period2) is the baseline and is kept untouched.
    #   - The entire pre-AMSR2 record (AMSR-E era + gap) is represented by the
    #     continuous ERA5 series and CDF-matched once, directly onto the AMSR2
    #     distribution. The AMSR-E sensor values (ds_period1) do not contribute.
    # The CDF matching works on the original (unsmoothed) time series; the rolling
    # day-of-year window inside get_cdf_fixed_data provides the smoothing.
 
    # Build the continuous pre-AMSR2 ERA5 source: everything strictly before AMSR2 start.
    amsr2_start = pd.to_datetime(period_planet2[0])
    pre_end = str((amsr2_start - pd.Timedelta(days=1)).date())
    ds_pre_era = clean_data(ds_gap, date_range=('2002-01-01', pre_end))
 
    # Single-step CDF (quantile) matching: continuous ERA5 -> AMSR2 baseline
    ds_pre_era[f'{variable}_smooth'] = get_cdf_fixed_data(
        ds = ds_pre_era,
        ds_base = ds_period2,
        field_target = variable,
        field_base = variable,
        group_cols = [LOCATION_NAME],
        level = 'dayofyear',
        window_size = cdf_time_window
    )
    print(f"CDF matching done")
 
    # AMSR2 passthrough: the baseline value is the raw observation
    ds_period2[f'{variable}_smooth'] = ds_period2[variable]
 
    # Join the rescaled pre-AMSR2 ERA5 with the raw AMSR2 baseline into one series
    ds_concat = xr.concat([ds_pre_era, ds_period2], dim='time')
    print(f"Concatenating arrays done")
 
    # The harmonized series is the matched/baseline value. Keep the per-source
    # original observation in {variable}_orig for reference.
    ds_concat = ds_concat.rename({variable: f'{variable}_orig'})
    ds_concat[variable] = ds_concat[f'{variable}_smooth']
 
    # Downstream steps in the requested order: clean_data -> climatology -> anomalies.
    # (The first get_smooth_series on the source is intentionally skipped so the
    #  harmonized series matches the reproduction built on the original time series.)
    ds_concat = clean_data(ds_concat, var=[f'{variable}_orig', f'{variable}_smooth', variable])
 
    ds_concat, ds_clim = get_climatology(
        ds_concat, variable, f'clim_{variable}', level='dayofyear', smooth_window=clim_smooth_window
    )
 
    # Compute anomalies using the added climatology
    ds_concat[f'anom_{variable}'] = ds_concat[variable] - ds_concat[f'clim_{variable}']
    print('Climatology and anomaly successfully added')
 
    df_concat = ds_concat.to_dataframe().reset_index()
 
    if add_abs_anom:
        df_concat[f'neg_anom_{variable}'] = np.where(
            df_concat[f'anom_{variable}'] >=0, 0, df_concat[f'anom_{variable}'] * (-1)
        ) 
        df_concat[f'pos_anom_{variable}'] = np.where(
            df_concat[f'anom_{variable}'] <=0, 0, df_concat[f'anom_{variable}']
        ) 
 
    return df_concat, ds_clim
'''

'''
def process_data_planet(
    ds1_orig: xr.Dataset, 
    ds2_orig: xr.Dataset, 
    ds_era_orig: xr.Dataset, 
    gdf: gpd.GeoDataFrame, 
    params: dict
) -> tuple:
    """
    Clean and process dataset to get unique time series and add anomalies and climatologies
    Args:
        - ds1_orig : (xr.Dataset) Dataset from first period of Planet
        - ds_era_orig : (xr.Dataset) Dataset from ERA5 to fill gap
        - ds2_orig : (xr.Dataset) Dataset from second period of Planet 
        - gdf : (gpd.GeoDataFrame) GeoDataFrame with geometries
        - params : (dict) Dictionary with parameters to process data
    
    Returns:
        - df_concat : pd.DataFrame with processed data
        - ds_clim : xr.Dataset with climatologies
    """
    # Read parameters from dictionary
    subset = params['subset']
    #location_var = params['location_var']
    variable = params['variable']
    na_replace_era5 = params['na_replace']['era5']
    na_replace_planet = params['na_replace']['planet']
    period_planet1 = params['time_window']['planet1']
    period_planet2 = params['time_window']['planet2']
    period_era5 = params['time_window']['era5']
    interp_resolution = params.get('interp_resolution', 0.01)
    fill_gap_window = params.get('fill_gap_window', 7)
    cdf_time_window = params.get('cdf_time_window', 45)
    time_smooth_window = params.get('time_smooth_window', 21)
    clim_smooth_window = params.get('climatology_smooth_window', 7)
    add_neg_anom = params.get('add_neg_anom', True)

    # Subset climate area geometry
    LOCATION_NAME = 'location_id'
    gdf_ca = rename_subset_geometry(gdf, subset, LOCATION_NAME)

    # Slice dataset using the given geometry
    ds_era = slice_dataset_w_geometry(ds_era_orig, gdf_ca)

    # Interpolate data from ERA5 to a finer resolution
    ds_era = regrid_dataset(ds_era, method='linear', res=interp_resolution)

    # Rename variables to use uniform names
    ds1_orig = rename_vars(ds1_orig, variable)
    ds2_orig = rename_vars(ds2_orig, variable)
    ds_era = rename_vars(ds_era, variable)

    # Remove unnecesary coordinates
    keep_coords = list(get_coordinates(ds_era)) # list of coords to keep
    ds_era = drop_single_coords(ds_era, except_coords=keep_coords)[[variable]] 

    keep_coords_1 = list(get_coordinates(ds1_orig)) # list of coords to keep
    ds1_orig = drop_single_coords(ds1_orig, except_coords=keep_coords)[[variable]]

    keep_coords_1 = list(get_coordinates(ds2_orig)) # list of coords to keep
    ds2_orig = drop_single_coords(ds2_orig, except_coords=keep_coords)[[variable]] 

    # Clean data with time slices and na replace values
    ds_period1 = clean_data(ds1_orig, date_range=period_planet1, var=variable, na_replace=na_replace_planet)
    ds_period2 = clean_data(ds2_orig, date_range=period_planet2, var=variable, na_replace=na_replace_planet)
    ds_gap = clean_data(ds_era, date_range = ('2002-01-01', None), var=variable, na_replace=na_replace_era5)
    print(f"Cleaning done")

    ds_period1, df1_ = create_cluster_coord(ds_period1, gdf_ca, LOCATION_NAME) #Using v2 is faster since it applies a spatial join
    ds_period2, df2_ = create_cluster_coord(ds_period2, gdf_ca, LOCATION_NAME) #Using v2 is faster since it applies a spatial join
    ds_gap, dfg_ = create_cluster_coord(ds_gap, gdf_ca, LOCATION_NAME) #Using v2 is faster since it applies a spatial join

    # Summarize the arrays along a specified dimension
    ds_period1 = summarize_data(ds_period1, group_coords=[LOCATION_NAME], func='mean')
    ds_period2 = summarize_data(ds_period2, group_coords=[LOCATION_NAME], func='mean')
    ds_gap = summarize_data(ds_gap, group_coords=[LOCATION_NAME], func='mean')
    print(f"Clustering and summarizing done")

    # Smooth time series given the smooth window
    #ds_period1[f'{variable}_smooth'] = get_smooth_series(ds_period1, variable, fill_gap_window)
    #ds_period2[f'{variable}_smooth'] = get_smooth_series(ds_period2, variable, fill_gap_window)
    ds_period2[f'{variable}_cdf'] = ds_period2[variable]

    # Apply CDF matching to the time series with the second period of planet as basis
    ds_gap[f'{variable}_cdf'] = get_cdf_fixed_data(
        ds = ds_gap, 
        ds_base = ds_period2, 
        field_target = variable, 
        field_base = variable,
        group_cols = [LOCATION_NAME], 
        level = 'dayofyear', 
        window_size = cdf_time_window
    )
    # Use ERA5 as a bridge between Planet AMSRE and Planet AMSR2
    ds_period1[f'{variable}_cdf'] = get_cdf_fixed_data(
        ds = ds_period1, 
        ds_base = ds_gap, 
        field_target = f'{variable}', 
        field_base = f'{variable}_cdf',
        group_cols = [LOCATION_NAME], 
        level = 'dayofyear', 
        window_size = cdf_time_window
    )
    ds_gap = clean_data(ds_gap, date_range=period_era5)
    print(f"CDF matching done")

    # Join and smooth the resulting time series
    ds_concat = xr.concat([ds_period1, ds_gap, ds_period2], dim='time')
    print(f"Concatenating arrays done")
    
    ds_concat = ds_concat.rename({variable: f'{variable}_orig'})
    ds_concat[variable] = get_smooth_series(ds_concat, f'{variable}_cdf', time_smooth_window)

    ds_concat, ds_clim = get_climatology(
        ds_concat, variable, f'clim_{variable}', level='dayofyear', smooth_window=clim_smooth_window
    )
    
    # Compute anomalies using the added climatology
    ds_concat[f'anom_{variable}'] = ds_concat[variable] - ds_concat[f'clim_{variable}']
    print('Climatology and anomaly successfully added')

    df_concat = ds_concat.to_dataframe().reset_index()

    if add_neg_anom:
        df_concat[f'neg_anom_{variable}'] = np.where(
            df_concat[f'anom_{variable}'] >=0, 0, df_concat[f'anom_{variable}'] * (-1)
        ) 

    return df_concat, ds_clim
'''

def process_data_planet(
    ds1_orig: xr.Dataset, 
    ds2_orig: xr.Dataset, 
    ds_era_orig: xr.Dataset, 
    gdf: gpd.GeoDataFrame, 
    params: dict
) -> tuple:
    """
    Clean and process dataset to get unique time series and add anomalies and climatologies
    Args:
        - ds1_orig : (xr.Dataset) Dataset from first period of Planet
        - ds_era_orig : (xr.Dataset) Dataset from ERA5 to fill gap
        - ds2_orig : (xr.Dataset) Dataset from second period of Planet 
        - gdf : (gpd.GeoDataFrame) GeoDataFrame with geometries
        - params : (dict) Dictionary with parameters to process data
    
    Returns:
        - df_concat : pd.DataFrame with processed data
        - ds_clim : xr.Dataset with climatologies
    """
    # Read parameters from dictionary
    subset = params['subset']
    #location_var = params['location_var']
    variable = params['variable']
    na_replace_era5 = params['na_replace']['era5']
    na_replace_planet = params['na_replace']['planet']
    period_planet1 = params['time_window']['planet1']
    period_planet2 = params['time_window']['planet2']
    period_era5 = params['time_window']['era5']
    interp_resolution = params.get('interp_resolution', 0.01)
    fill_gap_window = params.get('fill_gap_window', 7)
    cdf_time_window = params.get('cdf_time_window', 45)
    time_smooth_window = params.get('time_smooth_window', 21)
    clim_smooth_window = params.get('climatology_smooth_window', 7)
    add_abs_anom = params.get('add_abs_anom', True)

    # Subset climate area geometry
    LOCATION_NAME = 'location_id'
    gdf_ca = rename_subset_geometry(gdf, subset, LOCATION_NAME)

    # Slice dataset using the given geometry
    ds_era = slice_dataset_w_geometry(ds_era_orig, gdf_ca)

    # Interpolate data from ERA5 to a finer resolution
    ds_era = regrid_dataset(ds_era, method='linear', res=interp_resolution)

    # Rename variables to use uniform names
    #ds1_orig = rename_vars(ds1_orig, variable)
    ds2_orig = rename_vars(ds2_orig, variable)
    ds_era = rename_vars(ds_era, variable)

    # Remove unnecesary coordinates
    keep_coords = list(get_coordinates(ds_era)) # list of coords to keep
    ds_era = drop_single_coords(ds_era, except_coords=keep_coords)[[variable]] 

    #keep_coords_1 = list(get_coordinates(ds1_orig)) # list of coords to keep
    #ds1_orig = drop_single_coords(ds1_orig, except_coords=keep_coords_1)[[variable]]

    keep_coords_2 = list(get_coordinates(ds2_orig)) # list of coords to keep
    ds2_orig = drop_single_coords(ds2_orig, except_coords=keep_coords_2)[[variable]] 

    # Clean data with time slices and na replace values
    #ds_period1 = clean_data(ds1_orig, date_range=period_planet1, var=variable, na_replace=na_replace_planet)
    ds_period2 = clean_data(ds2_orig, date_range=period_planet2, var=variable, na_replace=na_replace_planet)
    ds_gap = clean_data(ds_era, date_range = ('2002-01-01', None), var=variable, na_replace=na_replace_era5)
    print(f"Cleaning done")

    #ds_period1, df1_ = create_cluster_coord(ds_period1, gdf_ca, LOCATION_NAME) #Using v2 is faster since it applies a spatial join
    ds_period2, df2_ = create_cluster_coord(ds_period2, gdf_ca, LOCATION_NAME) #Using v2 is faster since it applies a spatial join
    ds_gap, dfg_ = create_cluster_coord(ds_gap, gdf_ca, LOCATION_NAME) #Using v2 is faster since it applies a spatial join

    # Summarize the arrays along a specified dimension
    #ds_period1 = summarize_data(ds_period1, group_coords=[LOCATION_NAME], func='mean')
    ds_period2 = summarize_data(ds_period2, group_coords=[LOCATION_NAME], func='mean')
    ds_gap = summarize_data(ds_gap, group_coords=[LOCATION_NAME], func='mean')
    print(f"Clustering and summarizing done")

    # Smooth time series given the smooth window
    #ds_period1[f'{variable}_smooth'] = get_smooth_series(ds_period1, variable, fill_gap_window)
    ds_period2[f'{variable}_smooth'] = get_smooth_series(ds_period2, variable, fill_gap_window)

    # Apply CDF matching to the time series with the second period of planet as basis
    
    ds_gap[f'{variable}_smooth'] = get_cdf_fixed_data(
        ds=ds_gap,
        ds_base=ds_period2,
        field_target=variable,
        field_base=f'{variable}_smooth',
        group_cols=[LOCATION_NAME],
        level='dayofyear',
        window_size=cdf_time_window
    )
    # CHANGED: AMSRE (period1) is now CDF-matched DIRECTLY to AMSR2 (period2),
    # not to the CDF-fixed ERA5 gap. The previous AMSR2 -> ERA5 -> AMSRE chain
    # routed AMSRE through ERA5's wetter, compressed distribution and biased the
    # AMSRE era upward. ERA5 is retained only to fill the gap below.
    #ds_period1[f'{variable}_smooth'] = get_cdf_fixed_data(
    #    ds=ds_period1,
    #    ds_base=ds_period2,
    #    field_target=f'{variable}_smooth',
    #    field_base=f'{variable}_smooth',
    #    group_cols=[LOCATION_NAME],
    #    level='dayofyear',
    #    window_size=cdf_time_window
    #)
    '''
    ds_gap[f'{variable}_smooth'] = get_cdf_fixed_data(
        ds = ds_gap, 
        ds_base = ds_period2, 
        field_target = variable, 
        field_base = f'{variable}_smooth',
        group_cols = [LOCATION_NAME], 
        level = 'dayofyear', 
        window_size = cdf_time_window
    )
    # Use ERA5 as a bridge between Planet AMSRE and Planet AMSR2
    ds_period1[f'{variable}_smooth'] = get_cdf_fixed_data(
        ds = ds_period1, 
        ds_base = ds_gap, 
        field_target = f'{variable}_smooth', 
        field_base = f'{variable}_smooth',
        group_cols = [LOCATION_NAME], 
        level = 'dayofyear', 
        window_size = cdf_time_window
    )
    '''
    ds_period1 = clean_data(ds_gap, date_range=period_planet1)
    ds_gap = clean_data(ds_gap, date_range=period_era5)
    print(f"CDF matching done")

    # Drop any scalar coordinates that ERA5 adds (e.g. 'expver') and are
    # absent from Planet datasets — they cause xr.concat to fail.
    extra_coords = [c for c in ds_gap.coords if c not in ds_period1.coords and c not in ds_period2.coords]
    if extra_coords:
        ds_gap = ds_gap.drop_vars(extra_coords)

    # Join and smooth the resulting time series
    ds_concat = xr.concat([ds_period1, ds_gap, ds_period2], dim='time')
    print(f"Concatenating arrays done")
    
    ds_concat = ds_concat.rename({variable: f'{variable}_orig'})
    ds_concat[variable] = get_smooth_series(ds_concat, f'{variable}_smooth', time_smooth_window)

    ds_concat, ds_clim = get_climatology(
        ds_concat, variable, f'clim_{variable}', level='dayofyear', smooth_window=clim_smooth_window
    )
    ds_concat = add_neg_anomaly(ds_concat, 'swc_adjusted', 'climatology', keep_neg_anom=False)
    print('Climatology and anomaly successfully added')

    df_concat = ds_concat.to_dataframe().reset_index()

    if add_abs_anom:
        df_concat[f'neg_anom_{variable}'] = np.where(
            df_concat[f'anom_{variable}'] >=0, 0, df_concat[f'anom_{variable}'] * (-1)
        ) 
        df_concat[f'pos_anom_{variable}'] = np.where(
            df_concat[f'anom_{variable}'] <=0, 0, df_concat[f'anom_{variable}']
        ) 

    return df_concat, ds_clim

#———————————————————————————————————————————
# PROCESS DATASET WITH ANOMALIES
#———————————————————————————————————————————


def process_data_request(
    ds : xr.Dataset,
    gdf : gpd.GeoDataFrame,
    params : dict,
    params_s: dict,
):
    """
    Clean and process dataset to get unique time series and add anomalies and climatologies
    Args:
        - ds : (xr.Dataset) Dataset with Soil Water Content data
        - gdf : (gpd.GeoDataFrame) GeoDataFrame with geometries
        - params : (dict) Dictionary with parameters to process data
    Returns:
        - pd.DataFrame with processed data
        - xr.Dataset with climatologies
    """
    # Read parameters from dictionary
    variable    = params['variable']
    na_replace  = params['na_replace']
    period      = params['time_window']
    interp_resolution  = params['interp_resolution']
    time_smooth_window = params['time_smooth_window']
    clim_smooth_window = params['climatology_smooth_window']

    # Resolve dict-style params (Planet config) to the value that matches this dataset
    if isinstance(variable, dict):
        # Pick the value whose key matches a data variable in ds
        variable = next(
            (v for k, v in variable.items() if v in ds.data_vars),
            list(variable.values())[0]
        )
    if isinstance(na_replace, dict):
        na_replace = next(
            (v for k, v in na_replace.items() if variable in ds.data_vars),
            None
        )
    if isinstance(period, dict):
        # Use the 'full' range when available, else first list value
        period = period.get('full', list(period.values())[0])
    
    # Subset climate area geometry
    gdf_ca = subset_geometry(gdf, params_s)
    gdf_ca = gdf_ca.to_crs(epsg='4326')

    # Remove unnecesary coordinates
    keep_coords = list(get_coordinates(ds)) # list of coords to keep
    ds = drop_single_coords(ds, except_coords=keep_coords)

    # Slice dataset using the given geometry
    ds_slice = slice_dataset_w_geometry(ds, gdf_ca)

    # Interpolate data from ERA5 to a finer resolution
    ds_slice = regrid_dataset(ds_slice, method='linear', res=interp_resolution)

    # Rename variables to use uniform names
    ds_slice = rename_vars(ds_slice, variable)
    ds_slice = ds_slice[[variable]] # selecting only the variable of interest
    print(f"Renaming variables done")

    # Clean data with time slices and na replace values
    ds_clean = clean_data(ds_slice, date_range = period, var=variable, na_replace=na_replace)
    print(f"Cleaning done")

    # Smooth time series given the smooth window
    if time_smooth_window > 0:
        ds_clean = ds_clean.rename({f'{variable}': f'{variable}_orig'})
        ds_clean[f'{variable}'] = get_smooth_series(ds_clean, f'{variable}_orig', time_smooth_window)
    
    # Compute climatologies and anomalies at the pixel level
    ds_process, ds_clim = get_climatology(
        ds_clean, variable, f'clim_{variable}', level='dayofyear', smooth_window=clim_smooth_window
    )

    # Compute anomalies using the added climatology
    ds_process[f'anom_{variable}'] = ds_process[variable] - ds_process[f'clim_{variable}']

    print('Climatology and anomaly successfully added')
    
    return ds_process, ds_clim


#———————————————————————————————————————————
# PROCESS DATASET WITH EVENTS
#———————————————————————————————————————————


def process_data_temp(
    ds : xr.Dataset,
    gdf : gpd.GeoDataFrame,
    params : dict,
    params_s: dict={},
):
    """
    Clean and process dataset to get unique time series and add anomalies and climatologies
    Args:
        - ds : (xr.Dataset) Dataset with Soil Water Content data
        - gdf : (gpd.GeoDataFrame) GeoDataFrame with geometries
        - params : (dict) Dictionary with parameters to process data
        - params_s : (dict) Dictionary with spatial parameters to subset data
    Returns:
        - pd.DataFrame with processed data
        - xr.Dataset with climatologies
    """
    # Read parameters from dictionary
    variable = params['variable']
    na_replace = params['na_replace']
    period = params['time_window']
    threshold = params['threshold']
    num_days = params['num_days']
    side = params['side']
    interp_resolution = params.get('interp_resolution', 0)
    time_smooth_window = params.get('time_smooth_window', 0)
    clim_smooth_window = params.get('climatology_smooth_window', 0)
    
    # Subset climate area geometry
    gdf_ca = subset_geometry(gdf, params_s)
    gdf_ca = gdf_ca.to_crs(epsg='4326')

    # Remove unnecesary coordinates
    keep_coords = list(get_coordinates(ds)) # list of coords to keep
    ds = drop_single_coords(ds, except_coords=keep_coords)

    # Slice dataset using the given geometry
    ds_slice = slice_dataset_w_geometry(ds, gdf_ca)

    # Interpolate data from ERA5 to a finer resolution
    ds_slice = regrid_dataset(ds_slice, method='linear', res=interp_resolution)

    # Clean data with time slices and na replace values
    ds_clean = clean_data(ds_slice, date_range = period, var=variable, na_replace=na_replace)
    print(f"Cleaning done")

    # Smooth time series given the smooth window
    if time_smooth_window > 0:
        ds_clean = ds_clean.rename({f'{variable}': f'{variable}_orig'})
        ds_clean[f'{variable}'] = get_smooth_series(ds_clean, f'{variable}_orig', time_smooth_window)

    # Compute climatologies and anomalies at the pixel level
    ds_clean, ds_clim = get_climatology(
        ds_clean, variable, f'clim_{variable}', level='dayofyear', smooth_window=clim_smooth_window
    )
    ds_clean, ds_std = get_climatology(
        ds_clean, variable, f'std_{variable}', level='dayofyear', smooth_window=clim_smooth_window, func='std'
    )
    ds_clim = xr.merge([ds_clim, ds_std])

    ds_process = add_normalized_variable(
        ds_clean, variable, 
        f'clim_{variable}', 
        f'std_{variable}',
        f'norm_{variable}'
    )

    # Compute extreme days
    event_name = 'cold_spell' if side=='lower' else 'heat_wave'
    #ds_process = get_extreme_events(ds_clean, variable, event_name, threshold, num_days, side)
    ds_process = get_extreme_events(ds_clean, f'norm_{variable}', event_name, threshold, num_days, side)

    print('Events successfully indetified')

    return ds_process, ds_clim


def process_data_coldspell(
    ds: xr.Dataset,
    gdf: gpd.GeoDataFrame,
    params: dict,
    params_s: dict,
) -> xr.Dataset:
    """Subset, slice, regrid, rename and clean the dataset.

    This is the innocuous front half of process_data_request: geometry subset,
    drop of singleton coords, spatial slice, optional regrid, variable rename,
    and time/NA cleaning. It does not smooth or compute climatologies. The
    returned Dataset carries a single data variable named `variable`, on
    lat/lon/time coordinates.
    """
    variable = params["variable"]
    na_replace = params.get("na_replace", None)
    period = params.get("time_window", (None, None))

    interp_resolution = params.get('interp_resolution', 0)
    time_smooth_window = params.get('time_smooth_window', 0)
    clim_smooth_window = params.get('climatology_smooth_window', 0)

    func = params.get('func', 'mean')
    quantiles = params.get('quantiles', [0.05])
    #threshold = params.get('threshold',)
    num_periods = params.get('num_periods', 4)
    side = params.get('side', 'lower')


    # Subset climate area geometry.
    gdf_ca = subset_geometry(gdf, params_s)
    gdf_ca = gdf_ca.to_crs(epsg="4326")

    # Remove unnecessary singleton coordinates, keeping lon/lat/time.
    keep_coords = list(get_coordinates(ds))
    ds = drop_single_coords(ds, except_coords=keep_coords)

    # Slice to the geometry envelope.
    ds_slice = slice_dataset_w_geometry(ds, gdf_ca)

    # Optional regrid (res=0 is a no-op and returns the dataset unchanged).
    ds_slice = regrid_dataset(ds_slice, method="linear", res=interp_resolution)

    # Rename to uniform lon/lat/time and the canonical variable name.
    ds_slice = rename_vars(ds_slice, variable)
    ds_slice = ds_slice[[variable]]
    print("Renaming variables done")

    # Smooth time series given the smooth window
    if time_smooth_window > 0:
        ds_clean = ds_clean.rename({variable: f'{variable}_orig'})
        ds_clean[variable] = get_smooth_series(ds_clean, f'{variable}_orig', time_smooth_window)

    # Time slice + NA replacement.
    ds_clean = clean_data(ds_slice, date_range=period, var=variable, na_replace=na_replace, normalize_time=False)
    print("Cleaning done")

    # Get climatology statistics    
    ds_process, ds_clim = get_climatology_windowed(
        ds_clean,
        variable,
        var_name='perc',
        levels=('dayofyear','hour'), 
        window_level='dayofyear',
        window=clim_smooth_window, 
        func=func, 
        quantiles=quantiles,
    )
    print(f"Climatology statistics computed")
    # Get extreme episodes
    ds_process = get_extreme_episodes(
        ds_process,
        variable,
        'perc_p05',
        var_name='cold_spell',
        num_periods=num_periods,
        side=side,
    )
    print(f"Extreme events computed")

    return ds_process, ds_clim


def process_data(
    ds : xr.Dataset,
    gdf : gpd.GeoDataFrame,
    params : dict,
    params_s: dict={},       
):
    PERIL_PROCESS = {
        'swc': process_data_request,
        'prcp': process_data_request,
        'tmin': process_data_coldspell,
        'tmax': process_data_temp,
    }

    variable = params['variable']
    ds_process, ds_clim = PERIL_PROCESS[variable](
        ds,
        gdf,
        params,
        params_s
    )
    return ds_process, ds_clim


#———————————————————————————————————————————
# SUMMARIZE GROUPED DATASET
#———————————————————————————————————————————


def summarize_processed_data(
    ds_orig : xr.Dataset,
    gdf : gpd.GeoDataFrame,
    params_process: dict,
):
    """
    Add a layer to group the original pixel values into climate area values
    Args:
        - ds : (xr.Dataset) Dataset with Soil Water Content data
        - gdf : (gpd.GeoDataFrame) GeoDataFrame with geometries
        - params_s : (dict) Dictionary with parameters to process data
    Returns:
        - pd.DataFrame with processed data
        - xr.Dataset with climatologies
    """
    variable = params_process['variable']
    add_neg_anom = params_process.get('add_neg_anom', True)
    mode = params_process.get('summarize_mode', 'within')
    k_neighbors = params_process.get('k_neighbors', 1)

    # Resolve dict-style variable (Planet config) to the value present in dataset
    if isinstance(variable, dict):
        variable = next(
            (v for v in variable.values() if v in ds_orig.data_vars),
            list(variable.values())[0]
        )

    ds = ds_orig.copy()
    # Subset climate area geometry
    LOCATION_NAME = 'location_id'
    gdf_aoi = gdf.to_crs(epsg='4326')

    def nearest_method(ds, gdf_aoi, LOCATION_NAME, check_var, k):
        lon, lat, time =  get_coordinates(ds)
        ds_clean, df_clusters = create_cluster_coord(ds, gdf_aoi, LOCATION_NAME, method='nearest', check_var=variable, k=1)
        ds_sum = summarize_data(ds_clean, group_coords=['points'])
        print(f"Nearest neighbors selection done")

        df_sum = ds_sum.reset_index('points').to_dataframe()
        df_sum = df_sum.reset_index().drop(columns='points', errors='ignore')

        # Round coordinates to avoid float-precision mismatches between
        # df_clusters (numpy source) and df_sum (xarray reset_index source)
        _PREC = 6
        df_key = df_clusters[[lon, lat, LOCATION_NAME]].copy()
        df_key[lon] = df_key[lon].round(_PREC)
        df_key[lat] = df_key[lat].round(_PREC)
        if lon in df_sum.columns:
            df_sum[lon] = df_sum[lon].round(_PREC)
        if lat in df_sum.columns:
            df_sum[lat] = df_sum[lat].round(_PREC)

        # Append location var to dataframe using coordinates
        merge_on = [c for c in [lon, lat] if c in df_sum.columns]
        df_sum = df_key.merge(
            df_sum,
            how='right',
            on=merge_on
        ).drop(columns=[lon, lat], errors='ignore')

        return df_sum, df_clusters

    # Summarize the arrays along a specified dimension
    if mode == 'nearest':
        df_sum, df_clusters = nearest_method(ds, gdf_aoi, LOCATION_NAME, variable, k=k_neighbors)
        print(f"Clustering with nearest neighbor strategy done")

    elif mode == 'within':
        ds_clean, df_clusters = create_cluster_coord(ds, gdf_aoi, LOCATION_NAME) #Using v2 is faster since it applies a spatial join
        ds_sum = summarize_data(ds_clean, group_coords=[LOCATION_NAME])
        print(f"Clustering and summarizing done using within strategy")

        df_sum = ds_sum.to_dataframe().reset_index()
        # Filter locations with all nans
        df_sum = df_sum.groupby(LOCATION_NAME).filter(lambda x: x[variable].count() !=0)

        # Assign pixels to left out geometries due to their size
        list_out = list(set(gdf_aoi[LOCATION_NAME].values) - set(df_sum[LOCATION_NAME].unique()))
        if len(list_out)>0:
            print(f"Applying nearest strategy for {len(list_out)} polygons")
            gdf_out = gdf_aoi[gdf_aoi[LOCATION_NAME].isin(list_out)]

            df_sum_out, df_clusters_out = nearest_method(ds, gdf_out, LOCATION_NAME, variable, k=k_neighbors)

            # Concatenate both datasets
            df_sum = pd.concat([df_sum, df_sum_out], axis=0, ignore_index=True)
            df_clusters = pd.concat([df_clusters, df_clusters_out], axis=0, ignore_index=True)
    
    elif mode == 'pixel':
        ds_clean, df_clusters = create_cluster_coord(ds, gdf_aoi, LOCATION_NAME, method='within')
        lon, lat, time =  get_coordinates(ds_clean)
        df_sum = ds_clean.to_dataframe().reset_index()
        df_sum = df_sum.dropna(subset=[LOCATION_NAME])
        df_sum['pixel_id'] = df_sum[lon].map('{:.3f}'.format) +'_'+ df_sum[lat].map('{:.3f}'.format)
    
    if add_neg_anom:
        df_sum[f'neg_anom_{variable}'] = np.where(
            df_sum[f'anom_{variable}'] >=0, 0, df_sum[f'anom_{variable}'] * (-1)
        )    

    return df_sum, df_clusters 


#———————————————————————————————————————————
# UPDATE PROCESSED DATASET
#———————————————————————————————————————————


def update_cluster_data(
    df_hist, 
    ds_new, 
    ds_clim, 
    gdf, 
    params
)-> pd.DataFrame:
    """
    Update data with newer observations
    Args:
        - df_hist : (pd.DataFrame) DataFrame with original processed data
        - ds_new : (xr.Dataset) Dataset with new observations in time
        - ds_clim: (xr.Dataset) Dataset with already computed climatology
        - gdf : (gpd.GeoDataFrame) GeoDataFrame with geometries
        - params : (dict) Dictionary with parameters to process data
    Returns:
        - pd.DataFrame with updated observations
    """
    # Read parameters from dictionary
    id_area = params['id_area']
    time_smooth_window = params['time_smooth_window']
    
    # Subset climate area geometry
    gdf_ca = gdf.query('climate_area_id == @id_area').copy()

    # Set initial and ending dates
    cut_date = str((df_hist.time.max() - pd.DateOffset(days=time_smooth_window)).date())
    new_date = str((df_hist.time.max() + pd.DateOffset(days=1)).date())

    ds = clean_data(ds_new, date_range=(cut_date, None), na_replace=65535)
    print(f"Cleaning done")

    ds, df_ = create_cluster_coord(ds, gdf_ca, 'climate_area_id') #Using v2 is faster since it applies a spatial join
    ds = summarize_data(ds, group_coords=['climate_area_id'])
    print(f"Clustering and summarizing done")

    # Smooth time series given the smooth window
    ds['swc_smooth'] = get_smooth_series(ds, 'swc', time_smooth_window)
    ds['swc_adjusted'] = ds['swc_smooth']
    print(f"Clustering and summarizing done")

    ds = add_climatology(ds, ds_clim, 'swc_adjusted', 'climatology', level='dayofyear')
    ds = add_neg_anomaly(ds, 'swc_adjusted', 'climatology')
    print('Climatology and anomaly successfully added')

    ds = ds.sel(time = slice(new_date, None))
    df = ds.to_dataframe().reset_index()
    df = df[df_hist.columns]
    
    df_orig = pd.concat([df_hist, df], axis=0).reset_index(drop=True)
    print('Concatenating historic and updated arrays done')
    
    return df_orig