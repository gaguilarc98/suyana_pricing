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