from .utils import *


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

def add_time_coordinate(ds_orig, time_dim=None, level='week'):
    """Add time coordinate to be used for computing climatology."""
    ds = ds_orig.copy()
    
    if time_dim is None:
        time_dim = _get_time_coordinate(ds)
    
    if level == 'dayofyear':
        ds = ds.assign_coords({level: ds[time_dim].dt.dayofyear})
    elif level == 'week':
        ds = ds.assign_coords({level: ds[time_dim].dt.isocalendar().week})
    elif level == 'month':
        ds = ds.assign_coords({level: ds[time_dim].dt.month})
    elif level == 'year':
        ds = ds.assign_coords({level: ds[time_dim].dt.year})
    else:
        raise ValueError(f"No support for level {level}")
    
    return ds

def get_smooth_series(ds, field, smooth_window):
    """Smooth time series with a time window"""
    time = _get_time_coordinate(ds)
    if smooth_window>0:
        da = ds[field].rolling({time: smooth_window}, min_periods=1).mean()
    return da


#———————————————————————————————————————————
# CLEAN DATASET
#———————————————————————————————————————————


def clean_data(
    ds_orig, 
    date_range = (None, None), 
    var: {list or str} = None,
    na_replace = None,  
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

        # Get the field from base dataset based on joins by group_cols. level and ranking
        df_orig = df_orig[dims + drop_cols].merge(
            df_base[group_cols + ['rank', field_base]],
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
            points = (
                ds.assign_coords(valid_count=valid_count)
                [[lon_var, lat_var, 'valid_count']]
                .to_dataframe()
                .reset_index()
            )
            points = points[points['valid_count']!=0].copy()
            
        df_cluster = match_grid_points(points, gdf, k=k)
        df_cluster = df_cluster[[lon_var, lat_var, cluster_var]].copy()

        ds = select_coordinates(ds, df_cluster)
        #ds = ds.reset_index('points')
    
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


#———————————————————————————————————————————
# PROCESS DATASET FROM PLANET
#———————————————————————————————————————————


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
        - df_concat : (pd.DataFrame) with processed data
        - ds_clim : (xr.Dataset) with climatologies
    """
    # Read parameters from dictionary
    subset = params['subset']
    #location_var = params['location_var']
    var_planet = params['variable']['planet']
    var_era5 = params['variable']['era5']
    na_replace_era5 = params['na_replace']['era5']
    na_replace_planet = params['na_replace']['planet']
    period_planet1 = params['time_window']['planet1']
    period_planet2 = params['time_window']['planet2']
    period_era5 = params['time_window']['era5']
    interp_resolution = params['interp_resolution']
    fill_gap_window = params['fill_gap_window']
    cdf_time_window = params['cdf_time_window']
    time_smooth_window = params['time_smooth_window']
    clim_smooth_window = params['climatology_smooth_window']

    # Subset climate area geometry
    LOCATION_NAME = 'location_id'
    gdf_ca = rename_subset_geometry(gdf, subset, LOCATION_NAME)

    # Slice dataset using the given geometry
    ds_era = slice_dataset_w_geometry(ds_era_orig, gdf_ca)

    # Interpolate data from ERA5 to a finer resolution
    ds_era = regrid_dataset(ds_era, method='linear', res=interp_resolution)

    # Clean data with time slices and na replace values
    ds_period1 = clean_data(ds1_orig, date_range=period_planet1, var=var_planet, na_replace=na_replace_planet)
    ds_period2 = clean_data(ds2_orig, date_range=period_planet2, var=var_planet, na_replace=na_replace_planet)
    ds_gap = clean_data(ds_era, date_range = ('2002-01-01', None), var=var_era5, na_replace=na_replace_era5)
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
    ds_period1['swc_smooth'] = get_smooth_series(ds_period1, var_planet, fill_gap_window)
    ds_period2['swc_smooth'] = get_smooth_series(ds_period2, var_planet, fill_gap_window)

    # Apply CDF matching to the time series with the second period of planet as basis
    ds_gap['swc_smooth'] = get_cdf_fixed_data(
        ds = ds_gap, 
        ds_base = ds_period2, 
        field_target = var_era5, 
        field_base = 'swc_smooth',
        group_cols = [LOCATION_NAME], 
        level = 'dayofyear', 
        window_size = cdf_time_window
    )
    # Use ERA5 as a bridge between Planet AMSRE and Planet AMSR2
    ds_period1['swc_smooth'] = get_cdf_fixed_data(
        ds = ds_period1, 
        ds_base = ds_gap, 
        field_target = 'swc_smooth', 
        field_base = 'swc_smooth',
        group_cols = [LOCATION_NAME], 
        level = 'dayofyear', 
        window_size = cdf_time_window
    )
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
    
    ds_concat['swc_adjusted'] = get_smooth_series(ds_concat, 'swc_smooth', time_smooth_window)

    ds_concat, ds_clim = get_climatology(
        ds_concat, 'swc_adjusted', 'climatology', level='dayofyear', smooth_window=clim_smooth_window
    )
    ds_concat = add_neg_anomaly(ds_concat, 'swc_adjusted', 'climatology', keep_neg_anom=False)
    print('Climatology and anomaly successfully added')

    df_concat = ds_concat.to_dataframe().reset_index()

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
    variable = params['variable']
    na_replace = params['na_replace']
    period = params['time_window']
    interp_resolution = params['interp_resolution']
    time_smooth_window = params['time_smooth_window']
    clim_smooth_window = params['climatology_smooth_window']
    
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
        df_sum = df_sum.reset_index().drop(columns='points')
        # Append location var to dataframe using coordinates
        df_sum = df_clusters[[lon, lat, LOCATION_NAME]].merge(
            df_sum,
            how='right',
            on=[lon, lat]
        ).drop(columns=[lon, lat])

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