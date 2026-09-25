from .utils import *

import re


def get_planet_api_key(
    df,
    location,
    variable,
    date
):
    
    df['startDate'] = pd.to_datetime(df['startDate'])
    df['endDate'] = pd.to_datetime(df['endDate'])
    date = pd.to_datetime(date)

    df_slice = df[
        (df['idLocation'] == location) &
        (df['variable'] == variable) &
        (date >= df['startDate']) & (date <= df['endDate'])
    ]
    if len(df_slice)==0:
        raise ValueError(f"No {variable} data available for {location} at {date}")
    else:
        return df_slice.iloc[0].to_dict()

'''
def get_file_name(
    date,
    dict_request
):
    key = dict_request['key']
    var = dict_request['variable']
    sen = dict_request['sensor']
    res = dict_request['resolution']
    ver = dict_request['version']

    ymd = pd.to_datetime(date).strftime("%Y%m%d")
    hour = '0130'
    
    year = str(pd.to_datetime(date).year)
    month = str(pd.to_datetime(date).month).rjust(2, '0')
    day = str(pd.to_datetime(date).day).rjust(2, '0')

    url = f'{key}/{year}/{month}/{day}/{var.upper()}-{sen}_{ver}_{res}-{ymd}T{hour}_{var.lower()}.tiff'
    
    return url


def list_files_from_bucket(df, params_transform):
    
    id_location = params_transform.get('id_location')
    prefix = params_transform.get('prefix', None)
    suffix = params_transform.get('suffix', None)
    variable = params_transform.get('variable', 'SWC')
    start_date = params_transform.get('start_date')
    end_date = params_transform.get('end_date')

    id_name = [s for s in [prefix, str(id_location), suffix] if s]
    id_name = "_".join(id_name).replace(' ','')
    bucket_folder = f'{variable}_{id_name}'

    date_range = pd.date_range(start_date, end_date).strftime('%Y-%m-%d')

    list_files = []

    for date in date_range:
        dict_request = get_planet_api_key(df, id_name, variable, date)
        url_file = get_file_name(date, dict_request)
        list_files.append(url_file)
    
    list_files = [f"{bucket_folder}/{f}" for f in list_files]
    
    return list_files
'''

import re
import posixpath
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import s3fs

BUCKET = 'suyana-planet'


def get_file_pattern(date, dict_request):
    """Return the day folder (relative to the bucket folder) and a regex
    for the file name, with the hour slot as a 4-digit wildcard."""
    key = dict_request['key']
    var = dict_request['variable']
    sen = dict_request['sensor']
    res = dict_request['resolution']
    ver = dict_request['version']

    date = pd.to_datetime(date)
    ymd = date.strftime('%Y%m%d')
    day_dir = f"{key}/{date:%Y}/{date:%m}/{date:%d}"

    regex = re.compile(
        rf"^{re.escape(var.upper())}-{re.escape(sen)}_{re.escape(ver)}_{re.escape(res)}"
        rf"-{ymd}T\d{{4}}_{re.escape(var.lower())}\.tiff$"
    )
    return key, day_dir, regex


def list_files_from_bucket(df, params_transform, fs=None, max_workers=16):
    fs = fs or s3fs.S3FileSystem()

    id_location = params_transform.get('id_location')
    prefix = params_transform.get('prefix', None)
    suffix = params_transform.get('suffix', None)
    variable = params_transform.get('variable', 'SWC')
    start_date = params_transform.get('start_date')
    end_date = params_transform.get('end_date')

    id_name = "_".join(s for s in [prefix, str(id_location), suffix] if s).replace(' ', '')
    bucket_folder = f'{variable}_{id_name}'
    base = f"{BUCKET}/{bucket_folder}"

    # Convert catalog dates once instead of on every call
    df = df.copy()
    df['startDate'] = pd.to_datetime(df['startDate'])
    df['endDate'] = pd.to_datetime(df['endDate'])

    date_range = pd.date_range(start_date, end_date).strftime('%Y-%m-%d')

    # Expected patterns per date, and the month folders we need to list
    patterns = []
    month_dirs = set()
    for date in date_range:
        dict_request = get_planet_api_key(df, id_name, variable, date)
        key, day_dir, regex = get_file_pattern(date, dict_request)
        patterns.append((day_dir, regex, date))
        month_dirs.add(posixpath.dirname(day_dir))  # key/YYYY/MM

    # Parallel listing of month folders
    def _list(month_dir):
        try:
            return fs.find(f"{base}/{month_dir}")
        except FileNotFoundError:
            return []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(_list, sorted(month_dirs)))

    files_by_dir = defaultdict(list)
    offset = len(base) + 1
    for paths in results:
        for path in paths:
            folder, name = posixpath.split(path[offset:])
            files_by_dir[folder].append(name)

    # Match each date
    list_files, missing = [], []
    for day_dir, regex, date in patterns:
        matches = sorted(n for n in files_by_dir.get(day_dir, []) if regex.match(n))
        if matches:
            list_files.append(f"{bucket_folder}/{day_dir}/{matches[0]}")
        else:
            missing.append(date)

    if missing:
        print(f"No file found for {len(missing)} dates (e.g. {missing[:5]})")

    return list_files


def process_tiff(file_name):

    file = 's3://suyana-planet/' + file_name
    try:
        ds_array = rioxarray.open_rasterio(file)
        # Search for date within file name
        pattern = r"-(\d{2}\d{2}\d{4})T"
        fecha = re.search(pattern, file_name).group(1)

        scale_factor = ds_array.attrs['scale_factor']

        ds_array = ds_array.sel(band=1).drop_vars(["band", "spatial_ref"])

        ds_array = ds_array.rename({
            'x':'lon', 
            'y':'lat'
        }).to_dataset(name='swc')

        ds_array = ds_array.assign_coords(time=pd.to_datetime(fecha, format='%Y%m%d'))
        ds_array['swc'] = xr.where(
            ds_array['swc']==65535, np.nan, 
            ds_array['swc']*scale_factor #0.01 #/1000
        )

        return ds_array
    except:
        return