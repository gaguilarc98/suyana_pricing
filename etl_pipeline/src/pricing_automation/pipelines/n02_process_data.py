from .utils import *

from .a02_create_clusters import *
from .a02_preprocess_planet_data import *
from .a02_auxiliary_processes import *
from .a02_auxiliary_processes import _quantile_label, _func_label


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
    offset = params.get('offset', 0)
    na_replace = params.get("na_replace", None)
    period = params.get("time_window", (None, None))

    interp_resolution = params.get('interp_resolution', 0)
    time_smooth_window = params.get('time_smooth_window', 0)
    clim_smooth_window = params.get('climatology_smooth_window', 0)

    func = params.get('func', 'mean')
    quantiles = params.get('quantiles', [0.05])
    threshold = params.get('threshold', 0)
    #num_periods = params.get('num_periods', 4)
    side = params.get('side', 'lower')
    pixel_window = params.get('pixel_window', 3)

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
    ds_clean[variable] = ds_clean[variable] - offset
    print("Cleaning done")

    # Get climatology statistics    
    ds_process, ds_clim = get_climatology_windowed(
        ds_clean,
        variable,
        var_name=variable,
        levels=('dayofyear','hour'), 
        window_level='dayofyear',
        window=clim_smooth_window, 
        func=func, 
        quantiles=quantiles,
    )
    print(f"Climatology statistics computed")
    
    # Get extreme episodes
    EVENT_NAME = 'cold_spell' if side=='lower' else 'heat_wave'
    THRESHOLD_VAR_NAME = _quantile_label(variable, quantiles[0]) # Default to first value in the list of quantiles
    FLAG_NAME, INTENSITY_NAME = f'flag_{EVENT_NAME}', f'intensity_{EVENT_NAME}'
    ds_process[FLAG_NAME], ds_process[INTENSITY_NAME] = flag_and_intensity(
        ds_process,
        variable,
        THRESHOLD_VAR_NAME,
        side=side,
        hard_threshold=threshold,
    )
    print(f"Extreme events computed")
    
    # Verify that the event happens locally by counting flags around each pixel
    VALID_NAME = f'valid_{EVENT_NAME}'
    flags = ds_process[FLAG_NAME]
    ds_process[VALID_NAME] = spatial_rolling_stat(
        flags, 
        pixel_window=pixel_window, 
        stat='sum'
    ) # Spatial rolling that sums the amount of flags in a box of 'pixel_window' size
    min_size = np.ceil(pixel_window/2)**2 # Minimum amount of pixels to count a flag as valid
    # Keep events only if the number of flags around is bigger than 'min_size'
    valid = ds_process[VALID_NAME]
    intensity = ds_process[INTENSITY_NAME]
    cond = (ds_process[FLAG_NAME] == 1) & (valid >= min_size)  # Validate events with local count of flags
    ds_process[VALID_NAME] = xr.where(flags.isnull(), np.nan, xr.where(cond, 1, 0))
    ds_process[INTENSITY_NAME] = xr.where(flags.isnull(), np.nan, xr.where(cond, intensity, 0))
    print(f'Spatial validation of events complete')

    #ds_process = get_extreme_episodes(
    #    ds_process,
    #    flag_field=VALID_NAME,
    #    intensity_field=INTENSITY_NAME,
    #    var_name=EVENT_NAME,
    #)
    #print(f"Extreme episodes identified")

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
    add_abs_anom = params_process.get('add_abs_anom', True)
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
        ds_clean, df_clusters = create_cluster_coord(ds, gdf_aoi, LOCATION_NAME, method='nearest', check_var=check_var, k=k)
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
        df_sum['pixel_id'] = (
            np.round(df_sum[lon].values, 3).astype(str)
            + '_'
            + np.round(df_sum[lat].values, 3).astype(str)
        )
    
    if add_abs_anom:
        df_sum[f'neg_anom_{variable}'] = np.where(
            df_sum[f'anom_{variable}'] >=0, 0, df_sum[f'anom_{variable}'] * (-1)
        )
        df_sum[f'pos_anom_{variable}'] = np.where(
            df_sum[f'anom_{variable}'] <=0, 0, df_sum[f'anom_{variable}']
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