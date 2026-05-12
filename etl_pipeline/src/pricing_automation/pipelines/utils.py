## SYSTEM LIBRARIES
import os
import requests
import yaml
import time
import tempfile
import json
#import cdsapi

## DATA WRANGLING LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

## GEOGRAPHIC LIBRARIES
import geopandas as gpd
import xarray as xr
import rioxarray

#from rasterio.io import MemoryFile
from shapely import Polygon, LineString, Point
from shapely.geometry import box, Polygon, MultiPolygon, Point, MultiPoint
from scipy.ndimage import uniform_filter, minimum_filter, maximum_filter
from scipy.spatial import cKDTree

## PLOTTING LIBRARIES
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.cm as cm

from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

## STATISTICS LIBRARIES
from scipy import stats
#from scipy.stats import f, norm, t, ks_2samp, genextreme, rankdata, beta
from scipy.signal import butter, filtfilt
#import statsmodels.api as sm
#import statsmodels.formula.api as sfm

## PARALLEL EXECUTION LIBRARIES
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed

## MACHINE LEARNING LIBRARIES

## BINARY WRITING LIBRARIES
import pickle

## SETTINGS

pd.set_option('display.max_columns', 50)  # Set max number of columns to display
pd.set_option('display.max_rows', 20)    # Set max number of rows to display


########____ARRANGING ARRAYS____########


def get_round_value(value, fraq_round):
    return np.round(value*fraq_round)/fraq_round

def lon_360_to_180(lon):
    """Changes the range of longitude values from the interval (0, 360) to (-180, 180)"""
    return (lon+180)%360-180

def lon_180_to_360(lon):
    """Changes the range of longitude values from the interval (-180, 180) to (0, 360)"""
    return lon%360

def convert_lon_to_180(lons):
    """Detect if longitudes are in 180 convention or 360 convention"""
    if min(lons) < 0 or max(lons) > 180:
        lons = lon_360_to_180(lons)
    return lons

def get_coordinates(ds: xr.Dataset):
    """Get lon, lat and time coordinates from dataset"""

    lon_coord = {'lon', 'Lon', 'longitude', 'Longitude', 'x', 'X'}.intersection(set(ds.dims))
    lat_coord = {'lat', 'Lat', 'latitude', 'Latitude', 'y', 'Y'}.intersection(set(ds.dims))
    time_coord = {'time', 'valid_time', 'Time', 'date', 'Date'}.intersection(set(ds.dims))

    if len(lon_coord)>0:
        lon_var = lon_coord.pop()
    else:
        lon_var = None
        #raise NameError('No longitude coordinate found')
    if len(lat_coord)>0:
        lat_var = lat_coord.pop()
    else:
        lat_var = None
        #raise NameError('No latitude coordinate found')
    if len(time_coord)>0:
        time_var = time_coord.pop()
    else:
        time_var = None
    
    return lon_var, lat_var, time_var


def regrid_dataset(
    ds,
    time: list = [],
    lons: list = [],
    lats: list = [],
    method: {'linear', 'nearest'} = 'nearest',
    res: int = 0
):
    """Regrids the original dataset with the specified time, lon and lat arrays"""
    ds_orig = ds.copy()
    if res==0:
        return ds_orig
    lon_orig, lat_orig, time_orig = get_coordinates(ds_orig)

    if len(lons)==0:
        lons = np.arange(ds_orig[lon_orig].min(), ds_orig[lon_orig].max() + res, res)
    if len(lats)==0:
        lats = np.arange(ds_orig[lat_orig].min(), ds_orig[lat_orig].max() + res, res)

    params_orig = {
        lon_orig: lons,
        lat_orig: lats
    }
    if time_orig != None and len(time) > 0:
        params_orig[time_orig]= time

    if method == 'nearest':
        ds_orig = ds_orig.sel(params_orig, method='nearest').assign_coords(params_orig)
    elif method == 'linear':
        ds_orig = ds_orig.interp(params_orig).assign_coords(params_orig)

    return ds_orig


def remove_single_coordinates(ds):
    """Remove coordinates with a single value"""
    for coord in ds.coords:
        if ds[coord].size == 1:
            ds = ds.drop_vars(coord)
    return ds

def drop_single_coords(ds, except_coords:list = []):
    """Remove coordinates that have a length of one"""
    for key in ds.coords:
        cond = (ds[key].size<=1 or np.size(np.unique(ds[key].values))<=1)
        if key not in except_coords and cond:
            ds = ds.drop_vars(key)
        elif key not in except_coords and key not in ds.dims:
            ds = ds.drop_vars(key)
        elif key not in except_coords and not cond:
            raise AssertionError(f'Cannot erase "{key}" as it is a dimension with length greater than one')
    return ds


########____MATCH AND SELECT POINT COORDINATES____########

"""
def match_grid_points(ds_grid, df_orig, orig_coords = ('lon', 'lat'), grid_coords = ('lon', 'lat')):
    '''Compute the nearest neighbour from grid to coordinates in dataframe'''
    # Getting names for actual coordinates
    lon_orig, lat_orig = orig_coords
    lon_grid, lat_grid = grid_coords

    # Target coordinates
    df_coord = df_orig.drop_duplicates(subset=[lat_orig, lon_orig]).copy()
    if grid_coords == orig_coords:
        df_coord = df_coord.rename(columns={
            lat_orig: f'{lat_orig}_orig',
            lon_orig: f'{lon_orig}_orig',
        })
        lat_orig, lon_orig = f'{lat_orig}_orig', f'{lon_orig}_orig'
    
    # Grid coordinates
    lon2d, lat2d = np.meshgrid(ds_grid[lon_grid].values, ds_grid[lat_grid].values)
    df_grid = pd.DataFrame(np.column_stack([lat2d.flatten(), lon2d.flatten()]), columns=['lat', 'lon'])

    # Create a KDTree with grid coordinates as source to match the original coordinates
    tree = cKDTree(df_grid)

    # Compute distance and index from grid nearest to coordinates
    dist, idx = tree.query(df_coord[[lat_orig, lon_orig]].values, k=1)
    df_grid = df_grid.iloc[idx].copy()

    # Copy the coordinates from grid to original dataframe
    df_coord[[lat_grid, lon_grid]] = df_grid[[lat_grid, lon_grid]].values

    return df_coord
"""

def match_grid_points(ds_sample, df_target, sample_coords = ('lon', 'lat'), target_coords = ('lon', 'lat'), k=1):
    '''Compute the nearest neighbour from grid to coordinates in dataframe'''
    # Getting names for actual coordinates
    lon_target, lat_target = target_coords
    lon_sample, lat_sample = sample_coords
   
    # Target coordinates
    df_t = df_target.drop_duplicates(subset=[lon_target, lat_target]).copy()
    df_t = df_t.reset_index(drop=True).reset_index()
    df_target = df_target.merge(
        df_t[['index', lon_target, lat_target]],
        how='left',
        on=[lon_target, lat_target]
    )
    if sample_coords == target_coords:
        df_t = df_t.rename(columns={
            lon_target: f'{lon_target}_target',
            lat_target: f'{lat_target}_target',
        })
        lon_target, lat_target = f'{lon_target}_target', f'{lat_target}_target'

    # Sample coordinates
    if isinstance(ds_sample, pd.DataFrame):
        df_s = ds_sample[[lon_sample, lat_sample]].drop_duplicates().reset_index(drop=True)
    elif isinstance(ds_sample, xr.Dataset):
        df_s = ds_sample[[lon_sample, lat_sample]].to_dataframe().reset_index()

    # Nearest Neighbor Algorithm
    tree = cKDTree(df_s)
    dist, idx = tree.query(df_t[[lon_target, lat_target]], k=k)

    # Join Sample coordinate to target coordinates
    df_join = df_s.iloc[idx.flatten()].copy()
    df_join['index'] = np.repeat(df_t.index, k)
    df_join["distance"] = dist.flatten()
    df_target = df_target.merge(
        df_join,
        how = 'left',
        on = 'index',
        suffixes = ('_target', '')
    )
    df_target = df_target.drop(columns='index')

    return df_target


def select_coordinates(ds, gdf_target, grid_coords = ('lon', 'lat')):
    """Select in-situ coordinate points from grid, ds and gdf must share the same grid coord names"""
    lon_grid, lat_grid = grid_coords
    ds_stacked = ds.stack(points = (lat_grid, lon_grid))

    # Get coordinates from weather stations
    df_coords_station = gdf_target[[lat_grid, lon_grid]].drop_duplicates().reset_index(drop=True)
    target_index = pd.MultiIndex.from_frame(df_coords_station)
    
    #Subset the Master Table with locations of interest
    ds_subset = ds_stacked.sel(points=target_index)
    return ds_subset


########____SUBSET AND SLICE WITH GEOMETRIES____########


def subset_geometry(gdf, params_s={}):
    """Subset a GeoDataFrame with a dictionary of inclusion and exclusion"""
    if 'include' in params_s:
        dict_include = params_s['include']
        for key, list_values in dict_include.items():
            if key not in gdf.columns:
                print(f'{key} not found in columns')
            else:
                gdf = gdf[gdf[key].isin(list_values)].copy()
    if 'exclude' in params_s:
        dict_exclude = params_s['exclude']
        for key, list_values in dict_exclude.items():
            if key not in gdf.columns:
                print(f'{key} not found in columns')
            else:
                gdf = gdf[~(gdf[key].isin(list_values))].copy()
    
    return gdf


def rename_subset_geometry(gdf, params_s, location_name='location_id'):
    """Subset a GeoDataFrame with a dictionary of parameters and rename location variable"""
    gdf_subset = subset_geometry(gdf, params_s)
    if 'location_var' not in params_s:
        raise ValueError(f'User did not provide value for location_var')
    location_var = params_s['location_var']
    if params_s['location_var'] not in gdf.columns:
        raise KeyError(f'The provided variable {location_var} is not present in the columns of GeoDataFrame')
    if location_name in gdf.columns and location_var != location_name:
        gdf_subset = gdf_subset.rename(columns={location_name: f'{location_name}_orig'})
    gdf_subset = gdf_subset.rename(columns={location_var: location_name})
    
    return gdf_subset


def get_enclosing_coords(grid, min_val, max_val):
    """Get enclosing coordinates"""
    if(len(grid[grid <= min_val])) == 0:
        lower = grid.min()
    else:
        lower = grid[grid <= min_val].max().item()
    if (len(grid[grid >= max_val])) == 0:
        upper = grid.max().item()
    else:
        upper = grid[grid >= max_val].min().item()
    return lower, upper


def slice_dataset_w_geometry(ds, gdf):
    """Get enclosing coordinates for Dataset given the geometry"""
    lon, lat, time = get_coordinates(ds)
    lon_min, lat_min, lon_max, lat_max = gdf.total_bounds
    minx, maxx = get_enclosing_coords(ds[lon], lon_min, lon_max)
    miny, maxy = get_enclosing_coords(ds[lat], lat_min, lat_max)

    ds = ds.sel({
        f'{lon}': ds[lon][(ds[lon]>=minx) & (ds[lon]<=maxx)],
        f'{lat}': ds[lat][(ds[lat]>=miny) & (ds[lat]<=maxy)]
    })
    return ds

########____CLEANING POLYGONS____########


def add_area_column(gdf, name_var='area_km2', area_crs="auto"):
    """Adds an 'area_km2' column to the GeoDataFrame with polygon areas in square kilometers"""
    if area_crs == "auto":
        try:
            area_crs = gdf.estimate_utm_crs()  # Local best-fit projection
        except Exception:
            area_crs = "EPSG:6933"  # Global equal-area fallback
    
    gdf_proj = gdf.to_crs(area_crs)
    gdf[name_var] = gdf_proj.area/1e6
    return gdf


def clean_polygons(gdf):
    """Make a quick clean-up of the geometries in a GeoDataFrame to make them valid geometries"""
    gdf = gdf.to_crs('EPSG:4326')
    gdf["geometry"] = gdf.geometry.buffer(0)
    gdf["geometry"] = gdf.geometry.make_valid()
    
    n_invalid = (~ gdf.geometry.is_valid).sum()
    if n_invalid>0:
        print(f'Warning: There are still {n_invalid} invalid geometries after cleaning')
    return gdf


########____DISPLAYING PLOTS____########


# Custom function to format latitude values
def format_latitude(x, pos):
    direction = 'N' if x >= 0 else 'S'  # 'N' for North, 'S' for South
    if x.is_integer():
        return f'{abs(x):.0f}° {direction}'
    else:
        return f'{abs(x):.1f}° {direction}'

# Custom function to format longitude values
def format_longitude(x, pos):
    direction = 'E' if x >= 0 else 'W'  # 'E' for East, 'W' for West
    if x.is_integer():
        return f'{abs(x):.0f}° {direction}'
    else:
        return f'{abs(x):.1f}° {direction}'
