from .utils import *

import gc
import io
import gzip
import cdsapi

from threading import Lock
from dataclasses import dataclass, field, fields, asdict
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Literal, Union

from .a01_aoi_period import BoundingBox, get_year_list

MIN_VALID_DAYS = 300
WORKERS_PER_YEAR = 6


#——————————————————————————————————————————————
# Abstract CONFIG AND DOWNLOADER
#——————————————————————————————————————————————


class DataConfig(ABC):
    @abstractmethod
    def to_dict(self) -> dict: ...

    @classmethod
    def from_dict(cls, d: dict):
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

class DataDownloader(ABC):
    def __init__(self, aoi: BoundingBox, params_request: DataConfig):
        self.aoi = aoi
        self.params_request = params_request

    @abstractmethod
    def download_year(self, year: int) -> xr.Dataset: ...

    @abstractmethod
    def download_period(self, start_year: int, end_year: int) -> Dict[str, xr.Dataset]: ...


#——————————————————————————————————————————————
# UCSB DOWNLOADER
#——————————————————————————————————————————————

def get_url_ucsb(
    date: str, 
    origin: Literal['CHIRTS', 'CHIRTS-ERA5', 'CHIRPS-GEFS', 'CHIRPS-GEFS-v12', 'CHIRPS', 'CHIRPS-v2', 'CHIRPS-v3-ERA5', 'CHIRPS-v3-IMERG'] = 'CHIRPS',
    variable: Literal['PRCP', 'TN', 'TX', 'RH'] = 'PRCP',
    date_fct: str = None,
    freq: Literal['daily', 'monthly'] = 'daily',
    **kwargs
) -> str:
    """
    Retrieve URL for downloading data from CHIRPS from UCSB.
    Args:
        date: Requested date (YYYY-MM-DD).
        origin: Data origin ('CHIRTS', 'CHIRTS-ERA5', 'CHIRPS-GEFS', 'CHIRPS-GEFS-v12', 'CHIRPS', 'CHIRPS-v2', 'CHIRPS-v3-ERA5', 'CHIRPS-v3-IMERG').
        variable: Domain ('PRCP', 'TN', 'TX', 'RH').
        date_fct: Date for forecast starting at initial date.
        freq: Domain ('daily', 'monthl').
    Returns:
        URL to access data.
    """
    PATH = f'https://data.chc.ucsb.edu/products'
    if origin not in ['CHIRTS', 'CHIRTS-ERA5', 'CHIRPS-GEFS', 'CHIRPS-GEFS-v12', 'CHIRPS', 'CHIRPS-v2', 'CHIRPS-v3-ERA5', 'CHIRPS-v3-IMERG']:
        raise NotImplementedError(f'{origin} not implemented.')

    if origin == 'CHIRTS':
        if freq == 'daily':
            PRODUCT = f'{PATH}/CHIRTSdaily/v1.0/global_tifs_p05'
        else:
            raise NotImplementedError(f'{freq} not implemented for origin {origin}')
    elif origin == 'CHIRTS-ERA5':
        if freq == 'daily':
            PRODUCT = f'https://data.chc.ucsb.edu/experimental/CHIRTS-ERA5'
        else:
            raise NotImplementedError(f'{freq} not implemented for origin {origin}')
    elif origin == 'CHIRPS-GEFS':
        if freq == 'daily':
            PRODUCT = f'{PATH}/CHIRPS-GEFS/v3/daily/global'
        elif freq == '05-day':
            PRODUCT = f'{PATH}/CHIRPS-GEFS/v3/05_day/global/data'
        else:
            raise NotImplementedError(f'{freq} not implemented for origin {origin}')
    elif origin == 'CHIRPS-GEFS-v12':
        if freq == 'daily':
            PRODUCT = f'{PATH}/CHIRPS-GEFS_precip_v12/daily_16day'
        else:
            raise NotImplementedError(f'{freq} not implemented for origin {origin}')
    elif origin == 'CHIRPS':
        if freq == 'daily':            
            PRODUCT = f'{PATH}/CHIRPS/v3.0/daily/final/IMERGlate-v07/'
        elif freq == 'monthly':
            PRODUCT = f'{PATH}/CHIRPS/v3.0/monthly/global/tifs'
    elif origin == 'CHIRPS-v2':
        if freq == 'daily':
            PRODUCT = f'{PATH}/CHIRPS-2.0/whem_daily/tifs/p05/'
        else:
            raise NotImplementedError(f'{freq} not implemented for origin {origin}')
    elif origin == 'CHIRPS-v3-ERA5':
        if freq == 'daily':
            PRODUCT = f'{PATH}/CHIRPS/v3.0/daily/final/rnl/'
        elif freq == 'monthly':
            PRODUCT = f'{PATH}/CHIRPS/v3.0/monthly/global/tifs/'
    elif origin == 'CHIRPS-v3-IMERG':
        if freq == 'daily':
            PRODUCT = f'{PATH}/CHIRPS/v3.0/daily/final/sat/'
        else:
            raise ValueError(f'{freq} not implemented for origin {origin}')

    date = pd.to_datetime(date)
    year = str(date.year)
    month = str(date.month).rjust(2, '0')
    day = str(date.day).rjust(2, '0')
    ym_dot = date.strftime('%Y.%m')
    ymd_dot = date.strftime('%Y.%m.%d')

    if origin == 'CHIRPS-GEFS' and freq == '05-day': 
        pass
    elif origin in ['CHIRPS-GEFS', 'CHIRPS-GEFS-v12'] and date_fct is None:
        raise ValueError(f'You must provide date_fct for the product {origin}')  
    elif origin in ['CHIRPS-GEFS', 'CHIRPS-GEFS-v12']:
        date_fct = pd.to_datetime(date_fct)
        ymd_dot_fct = date_fct.strftime('%Y.%m.%d')
        ymd_dot_fct_2 = date_fct.strftime('%Y.%m%d')
        
        
    if origin == 'CHIRTS':
        if variable == 'TN':
            URL = f'{PRODUCT}/Tmin/{year}/Tmin.{ymd_dot}.tif'
        elif variable == 'TX':
            URL = f'{PRODUCT}/Tmax/{year}/Tmax.{ymd_dot}.tif'
        elif variable == 'RH':
            URL = f'{PRODUCT}/RHum/{year}/RH.{ymd_dot}.tif'
        else:
            raise ValueError(f'{variable} not found in {origin}')
    elif origin == 'CHIRTS-ERA5':
        if variable == 'TN':
            URL = f'{PRODUCT}/tmin/tifs/daily/{year}/CHIRTS-ERA5.daily_Tmin.{ymd_dot}.tif'
        elif variable == 'TX':
            URL = f'{PRODUCT}/tmax/tifs/daily/{year}/CHIRTS-ERA5.daily_Tmax.{ymd_dot}.tif'
    elif origin in ['CHIRPS-GEFS', 'CHIRPS-GEFS-v12', 'CHIRPS', 'CHIRPS-v2', 'CHIRPS-v3-ERA5', 'CHIRPS-v3-IMERG'] and variable != 'PRCP':
        raise ValueError(f'{variable} not found in {origin}')
    elif origin == 'CHIRPS-GEFS':
        if freq == 'daily':
            URL = f'{PRODUCT}/{year}/{month}/{day}/c3g_{ymd_dot_fct}.tif'
        elif freq == '05-day':
            URL = f'{PRODUCT}/{year}/c3g_{ymd_dot}.tif'
    elif origin == 'CHIRPS-GEFS-v12':
        URL = f'{PRODUCT}/{year}/{month}/{day}/data.{ymd_dot_fct_2}.tif'
    elif origin == 'CHIRPS':
        if freq == 'daily':
            URL = f'{PRODUCT}/{year}/chirps-v3.0.{ymd_dot}.tif'
        if freq == 'monthly':
            URL = f'{PRODUCT}/chirps-v3.0.{ym_dot}.tif'
    elif origin == 'CHIRPS-v2':
        URL = f'{PRODUCT}/{year}/chirps-v2.0.{ymd_dot}.tif.gz'
    elif origin == 'CHIRPS-v3-ERA5':
        if freq == 'daily':
            URL = f'{PRODUCT}/{year}/chirps-v3.0.rnl.{ymd_dot}.tif'
        if freq == 'monthly':
            URL = f'{PRODUCT}/chirps-v3.0.{ym_dot}.tif'
    elif origin == 'CHIRPS-v3-IMERG':
        URL = f'{PRODUCT}/{year}/chirps-v3.0.sat.{ymd_dot}.tif'
    return URL


@dataclass
class UCSBConfig(DataConfig):
    origin: Literal[
        'CHIRTS', 'CHIRTS-ERA5', 'CHIRPS-GEFS', 'CHIRPS-GEFS-v12',
        'CHIRPS', 'CHIRPS-v2', 'CHIRPS-v3-ERA5', 'CHIRPS-v3-IMERG'
    ] = 'CHIRPS'
    variable: Literal['PRCP', 'TN', 'TX', 'RH'] = 'PRCP'
    freq: Literal['daily', 'monthly'] = 'daily'
    date_fct: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict):
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)
    
    def to_dict(self) -> dict:
        return asdict(self)


class UCSBDownloader(DataDownloader):

    def __init__(self, aoi: BoundingBox, params_request: UCSBConfig):
        self.aoi = aoi
        self.max_workers = WORKERS_PER_YEAR
        self.params_request = params_request
        self._lock = Lock()

    @classmethod
    def from_dict(cls, d: dict):
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    def _log(self, msg: str) -> None:
        with self._lock:
            print(msg)


    def _download_day(
        self, date: pd.Timestamp
    ) -> Optional[xr.Dataset]:
        """Download and process a single date's raster data."""
        params_request = self.params_request.to_dict()
        dict_var_name = {'PRCP': 'prcp', 'TN': 'tmin', 'TX': 'tmax', 'RH': 'rh'}
        variable = str(params_request['variable'])
        try:
            url_file = get_url_ucsb(date.strftime('%Y-%m-%d'), **params_request)
            session = requests.Session()

            if url_file.endswith('.gz'):
                r = session.get(url_file, stream=True)
                r.raise_for_status()
                with io.BytesIO(gzip.decompress(r.content)) as tif_stream:
                    with rioxarray.open_rasterio(tif_stream) as ds:
                        ds = ds.rio.clip_box(**self.aoi.to_dict())
                        ds = ds.squeeze().rename({'x': 'lon', 'y': 'lat'})
                        ds = ds.expand_dims({'time': [date]})
                        ds = ds.load()

            elif url_file.endswith('.tif'):
                r = session.get(url_file, stream=True)
                r.raise_for_status()
                with io.BytesIO(r.content) as tif_stream:
                    with rioxarray.open_rasterio(tif_stream) as ds:
                        ds = ds.rio.clip_box(**self.aoi.to_dict())
                        ds = ds.squeeze().rename({'x': 'lon', 'y': 'lat'})
                        ds = ds.expand_dims({'time': [date]})
                        ds = ds.load()  # force load before context manager closes

            print(f'{date.strftime("%Y%m%d")}', end=' ')
            return ds.to_dataset(name=dict_var_name[variable])
        except Exception as e:
            print(f'No data was retrieved for {date}: {e}')
            return None


    def download_year(
        self, year: int
    ) -> xr.Dataset:

        # Build full list of dates for the year
        all_dates = pd.date_range(f'{year}-01-01', f'{year}-12-31')

        print(f'Downloading {len(all_dates)} dates for year {year} with n_jobs={self.max_workers}...')

        results = Parallel(n_jobs=self.max_workers, backend='threading')(
            delayed(self._download_day)(date)
            for date in all_dates
        )

        ds_list = [r for r in results if r is not None]

        if len(ds_list) < MIN_VALID_DAYS:
            self._log(f"{year}: failed ({len(ds_list)}/{len(all_dates)} days valid)")

        try:
            ds_full = xr.concat(ds_list, dim='time').sortby('time')
            gc.collect()
            return ds_full
        except Exception as e:
            print(f'No files returned for {year}: {e}')


    def download_period(
        self, start_year: int, end_year: int
    ) -> Dict[str, xr.Dataset]:
        list_years = get_year_list(start_year, end_year)
        print(f"UCSB: {start_year}-{end_year} ({len(list_years)} years)")
        
        dict_data = {}
        for year in list_years:
            try:
                dict_data[str(year)] = self.download_year(year)
            except Exception as e:
                print(f'Unable to get data for year {year}')
                print(e)
        
        return dict_data



#——————————————————————————————————————————————
# ERA5 DOWNLOADER
#——————————————————————————————————————————————

_MONTHS = [f"{m:02d}" for m in range(1, 13)]
_DAYS = [f"{d:02d}" for d in range(1, 32)]

def _normalise_hours(hours: List[str]) -> List[str]:
    return [f"{h}:00" if ":" not in h else h for h in hours]

@dataclass
class ERA5Config(DataConfig):
    year: List[str] = None
    product_type: Literal[
        'reanalysis-era5-land', 'reanalysis-era5-single-levels', 'reanalysis-era5-pressure-levels',
        'derived-era5-land-daily-statistics', 'derived-era5-single-levels-daily-statistics'
    ] = 'reanalysis-era5-land'
    variable: List[str] = field(
        default_factory=lambda: ['volumetric_soil_water_layer_1']
    )
    month: List[str] = field(default_factory=lambda: _MONTHS)
    day: List[str] = field(default_factory=lambda: _DAYS)
    time: List[str] = field(default_factory=lambda: ['08']) 
    daily_statistic: Literal['daily_mean', 'daily_minimum', 'daily_maximum'] = None
    time_zone: Literal['utc-04:00'] = None
    frequency: Literal['1_hourly', '3_hourly', '6_hourly'] = None
    pressure_level: List[str] = None
    area: BoundingBox = None
    data_format: str = 'netcdf'
    download_format: str = 'unarchived'

    @classmethod
    def from_dict(cls, d: dict):
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    def to_dict(self) -> dict:
        params = {}
        REQUEST = {
            'reanalysis-era5-land': [
                'variable', 'year', 'month', 'day', 'time', 
                'data_format', 'download_format', 'area'
            ],
            'reanalysis-era5-single-levels': [
                'product_type', 'variable', 'year', 'month', 'day', 
                'time', 'data_format', 'download_format', 'area'
            ],
            'reanalysis-era5-pressure-levels': [
                'product_type', 'variable', 'year', 'month', 'day', 
                'time', 'pressure_level', 'data_format', 'download_format', 'area'
            ],
            'derived-era5-land-daily-statistics': [
                'variable', 'year', 'month', 'day', 'daily_statistic', 
                'time_zone', 'frequency', 'area'
            ],
            'derived-era5-single-levels-daily-statistics': [
                'product_type', 'variable', 'year', 'month', 'day', 
                'daily_statistic', 'time_zone', 'frequency', 'area'
            ]
        }
        params = asdict(self)
        params_request = {}
        for param in REQUEST[params['product_type']]:
            params_request[param] = params[param]
        if 'product_type' in params_request.keys():
            params_request['product_type'] = ['reanalysis']
        if 'area' in params_request.keys():
            params_request['area'] = self.area.to_era5_area()
        if 'time' in params_request.keys():
            params_request['time'] = _normalise_hours(self.time)
        return params_request


class ERA5Downloader(DataDownloader):

    def __init__(self, aoi: BoundingBox, params_request: ERA5Config):
        self.aoi = aoi
        self.params_request = params_request

    
    def _build_params(self, year) -> dict:
        self.params_request.area = self.aoi
        self.params_request.year = str(year)
        params = self.params_request.to_dict()
        return params
    

    def download_year(
        self, year:int
    ) -> xr.Dataset:
        
        client = cdsapi.Client()
    
        try:
            # Retrive response from API and read it in a temporal file
            dataset = self.params_request.product_type
            params = self._build_params(year)
            
            response = client.retrieve(dataset, params)
            # Use a temporary file to store the response
            with tempfile.NamedTemporaryFile(suffix=".nc") as tmp_file:
                response.download(tmp_file.name)  # Save to temp file
                # Open the file in memory as an xarray dataset
                ds = xr.open_dataset(tmp_file.name,  engine='netcdf4')

            ds = ds.rename({
                'valid_time': 'time',
                'longitude': 'lon',
                'latitude': 'lat'
            })
            gc.collect()
            return ds
        except Exception as e:
            print(f'Failed to process request for year {year}')
            print(f'Request specification: {params}')
            print(e)
    
        
    def download_period(
        self, start_year: int, end_year: int
    ) -> Dict[str, xr.Dataset]:
        list_years = get_year_list(start_year, end_year)
        print(f"ERA5: {start_year}-{end_year} ({len(list_years)} years)")
        
        dict_data = {}
        for year in list_years:
            try:
                dict_data[str(year)] = self.download_year(year)
            except Exception as e:
                print(f'Unable to get data for year {year}')
                print(e)
        
        return dict_data


