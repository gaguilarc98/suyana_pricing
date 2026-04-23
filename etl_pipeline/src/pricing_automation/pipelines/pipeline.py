from kedro.pipeline import Pipeline, node

#from downscale_weather.n00_create_population_grid import *
from .n01_extract_data import *
from .n02_process_data import *
from .n03_create_triggers import *
from .n04_bootstrap_aep import *
from .n05_pricing_quote import *
from .n06_visualizations import *
#from .n04_create_figures import *

def base_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func = get_aoi,
            inputs = ['gdf_request', 'params:params_s'],
            outputs = ['dict_bounds', 'gdf_aoi'],
            name = 'get_aoi',
            tags = ['extract']
        ),
        node(
            func = extract_data,
            inputs = ['dict_bounds', 'params:params_request', 'params:params_t'],
            outputs = 'ds_request',
            name = 'extract_data',
            tags = ['extract']
        ),
    ])

def output_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func = summarize_processed_data,
            inputs = ['ds_processed', 'gdf_aoi', 'params:params_process'],
            outputs = ['df_cluster', 'df_pixels'],
            name = 'summarize_processed_data',
            tags = ['process']
        ),
        node(
            func = generate_triggers,
            inputs = ['df_cluster', 'params:create_trigger', 'params:params_request'],
            outputs = ['df_triggers', 'df_percentiles'],
            name = 'generate_triggers',
            tags = ['triggers']
        ),
        node(
            func = run_bootstrap_aep,
            inputs = ['df_triggers', 'gdf_aoi', 'params:params_bootstrap'],
            outputs = ['df_annual_agg', 'df_aep'],
            name = 'run_bootstrap_aep',
            tags = ['aep', 'pricing']
        ),
        node(
            func = run_pricing_quote,
            inputs = ['df_annual_agg', 'params:params_quote'],
            outputs = 'df_pricing',
            name = 'run_pricing_quote',
            tags = ['pricing']
        ),
        node(
            func = plot_aep,
            inputs = ['df_annual_agg', 'df_aep', 'params:params_process'],
            outputs = ['plt_portfolio', 'plt_aep'],
            name = 'plot_aep',
            tags = ['pricing']
        ),
        node(
            func = plot_variability_map,
            inputs = ['ds_processed', 'gdf_aoi', 'params:params_process'],
            outputs = 'plt_variability_map',
            name = 'plot_variability_map',
            tags = ['viz']
        ),
        node(
            func = plot_trend_map,
            inputs = ['ds_processed', 'gdf_aoi', 'params:params_process'],
            outputs = 'plt_trend_map',
            name = 'plot_trend_map',
            tags = ['viz']
        ),
        node(
            func = plot_trigger_frequency_map,
            inputs = ['df_triggers', 'gdf_aoi', 'params:create_trigger'],
            outputs = 'plt_trigger_freq_map',
            name = 'plot_trigger_frequency_map',
            tags = ['viz']
        ),
        node(
            func = plot_anomaly_timeseries,
            inputs = ['df_cluster', 'params:params_process'],
            outputs = 'plt_anomaly_timeseries',
            name = 'plot_anomaly_timeseries',
            tags = ['viz']
        ),
    ])


def create_pipeline_moisture() -> Pipeline:
    extract_pipeline = base_pipeline()
    process_pipeline = Pipeline([
        node(
            func = process_data_request,
            inputs = ['ds_request', 'gdf_aoi', 'params:params_process', 'params:params_s'],
            outputs = ['ds_processed', 'ds_climatology'],
            name = 'process_data_request',
            tags = ['process']
        )
    ])
    load_pipeline = output_pipeline()
    return extract_pipeline + process_pipeline + load_pipeline


def create_pipeline_temperature() -> Pipeline:
    extract_pipeline = base_pipeline()
    process_pipeline = Pipeline([
        node(
            func = process_data_temp,
            inputs = ['ds_request', 'gdf_aoi', 'params:params_process', 'params:params_s'],
            outputs = ['ds_processed', 'ds_climatology'],
            name = 'process_data_request',
            tags = ['process']
        )
    ])
    load_pipeline = output_pipeline()
    return extract_pipeline + process_pipeline + load_pipeline