from .utils import *

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from .a01_aoi_period import *
from .a01_data_downloaders import *


#——————————————————————————————————————————————
# REQUEST BUILDERS REGISTRY
#——————————————————————————————————————————————

@dataclass
class ProviderConfig:
    provider: str
    downloader_cls: type[DataDownloader]
    config_cls: type[DataConfig]

    def __post_init__(self):
        if not issubclass(self.downloader_cls, DataDownloader):
            raise TypeError(f"{self.downloader_cls} must be a subclass of DataDownloader")
        if not issubclass(self.config_cls, DataConfig):
            raise TypeError(f"{self.config_cls} must be a subclass of DataConfig")

    def build(self, aoi: BoundingBox, params_request: dict) -> DataDownloader:
        cfg = self.config_cls(**params_request)
        return self.downloader_cls(aoi=aoi, params_request=cfg)


PROVIDER_CONFIG: dict[str, ProviderConfig] = {
    "ERA5": ProviderConfig("ERA5", ERA5Downloader, ERA5Config),
    "UCSB": ProviderConfig("UCSB", UCSBDownloader, UCSBConfig),
}


def get_provider_config(provider: str) -> ProviderConfig:
    key = provider.upper().replace(" ", "_").replace("-", "_")
    if key not in PROVIDER_CONFIG:
        raise ValueError(f"Unknown provider '{provider}'. Valid options: {list(PROVIDER_CONFIG.keys())}")
    return PROVIDER_CONFIG[key]


#——————————————————————————————————————————————
# DEFAULT PERILS REGISTRY
#——————————————————————————————————————————————


PERIL_CONFIGS: dict[tuple, dict] = {
    ("UCSB", "prcp"): dict(
        origin="CHIRPS-v3-ERA5",
        variable="PRCP",
        freq="daily",
    ),
    ("UCSB", "tmin"): dict(
        origin="CHIRTS-ERA5",
        variable="TN",
        freq="daily",
    ),
    ("UCSB", "tmax"): dict(
        origin="CHIRTS-ERA5",
        variable="TX",
        freq="daily",
    ),
    ("ERA5", "swc"): dict(
        product_type="reanalysis-era5-land",
        variable=["volumetric_soil_water_layer_1"],
        time=["08"],
    ),
    ("ERA5", "prcp"): dict(
        product_type="reanalysis-era5-land",
        variable=["total_precipitation"],
        time=["00"],
    ),
    ("ERA5", "tmin"): dict(
        product_type="derived-era5-land-daily-statistics",#"reanalysis-era5-single-levels",
        variable=["2m_temperature"],#["minimum_2m_temperature_since_previous_post_processing"],
        daily_statistic="daily_minimum",
        time_zone="utc-04:00",
        frequency="1_hourly",
        #time=["06"],
    ),
    ("ERA5", "tmax"): dict(
        product_type="derived-era5-land-daily-statistics",#"reanalysis-era5-single-levels",
        variable=["2m_temperature"],#["maximum_2m_temperature_since_previous_post_processing"],
        daily_statistic="daily_maximum",
        time_zone="utc-04:00",
        frequency="1_hourly",
        #time=["18"],
    ),
    ("ERA5", "wind"): dict(
        product_type="reanalysis-era5-land",
        variable=["10m_u_component_of_wind", "10m_v_component_of_wind"],
        time=[str(h).rjust(2, '0') for h in np.arange(0,24,6)],
    ),
    ("PLANET", "swc"): dict(
        product_type="reanalysis-era5-land",
        variable=["volumetric_soil_water_layer_1"],
        time=["08"],
    ),
}


def get_peril_config(provider:str, peril:str) -> dict:
    key = (
        str(provider).upper().replace("-", "").replace("_", ""), 
        str(peril).lower().replace("-", "").replace("_", "")
    )
    if key not in PERIL_CONFIGS:
        print(f"Key (provider, variable): '{key}' not registered as a default configuration.")
        print(f"Trying to override parameters with user input.")
        return {}
    return PERIL_CONFIGS[key]


def list_peril_configs() -> List[str]:
    return list(PERIL_CONFIGS.keys())