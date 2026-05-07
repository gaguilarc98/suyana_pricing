from kedro.pipeline import Pipeline, node

#from downscale_weather.n00_create_population_grid import *
from .n01_extract_data import *
from .n02_process_data import *
from .n03_create_triggers import *
from .n04_bootstrap_aep import *
from .n05_pricing_quote import *
from .n06_visualizations import *

def aoi_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func = get_aoi,
            inputs = ['gdf_request', 'params:params_s'],
            outputs = ['dict_bounds', 'gdf_aoi'],
            name = 'get_aoi',
            tags = ['extract']
        ),
    ])

def extract_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func = extract_data,
            inputs = ['dict_bounds', 'params:params_request', 'params:params_t'],
            outputs = 'ds_request',
            name = 'extract_data',
            tags = ['extract']
        ),
        node(
            func = process_data,
            inputs = ['ds_request', 'gdf_aoi', 'params:params_process', 'params:params_s'],
            outputs = ['ds_processed', 'ds_climatology'],
            name = 'process',
            tags = ['process']
        ),
        node(
            func = summarize_processed_data,
            inputs = ['ds_processed', 'gdf_aoi', 'params:params_process'],
            outputs = ['df_cluster', 'df_pixels'],
            name = 'summarize_processed_data',
            tags = ['process']
        ),
    ])


def trigger_pipeline(**kwargs) -> Pipeline:
    return Pipeline([  
        node(
            func = create_index_values,
            inputs = ['df_cluster', 'params:params_indices'],
            outputs = ['df_indices', 'pkl_fit'],
            name = 'create_index_values',
            tags = ['triggers']
        ),
        node(
            func = generate_payout_policy,
            inputs = ['df_indices', 'pkl_fit', 'params:params_request', 'params:params_indices', 'params:params_contract'],
            outputs = ['df_payouts', 'df_policy'],
            name = 'generate_payout_policy',
            tags = ['triggers']
        ),
    ])


def viz_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func = run_bootstrap_aep,
            inputs = ['df_payouts', 'gdf_aoi', 'params:params_bootstrap'],
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
            inputs = ['df_payouts', 'gdf_aoi'],
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
        node(
            func = create_activation_map,
            inputs = ['df_payouts', 'gdf_aoi', 'params:params_map'],
            outputs = 'plt_activation_map',
            name = 'create_activation_map',
            tags = ['viz']
        ),
        node(
            func = plot_activation_history,
            inputs = ['df_cluster', 'df_payouts', 'params:params_series', 'params:params_indices'],
            outputs = 'plt_time_series',
            name = 'plot_activation_history',
            tags = ['viz']
        ),
    ])


def planet_viz_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func = run_bootstrap_aep,
            inputs = ['df_payouts', 'gdf_climate_areas', 'params:params_bootstrap'],
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
            func = plot_trigger_frequency_map,
            inputs = ['df_payouts', 'gdf_climate_areas'],
            outputs = 'plt_trigger_freq_map',
            name = 'plot_trigger_frequency_map',
            tags = ['viz']
        ), 
        node(
            func = create_activation_map,
            inputs = ['df_payouts', 'gdf_climate_areas', 'params:params_map'],
            outputs = 'plt_activation_map',
            name = 'create_activation_map',
            tags = ['viz']
        ),
        node(
            func = plot_activation_history,
            inputs = ['df_cluster', 'df_payouts', 'params:params_series', 'params:params_indices'],
            outputs = 'plt_time_series',
            name = 'plot_activation_history',
            tags = ['viz']
        ),
    ])


def planet_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func = extract_data,
            inputs = ['dict_bounds', 'params:params_request', 'params:params_t'],
            outputs = 'ds_request',
            name = 'extract_data',
            tags = ['extract']
        ),
        node(
            func = create_climate_areas,
            inputs = ['gdf_aoi', 'ds_request', 'params:params_areas', 'params:params_s'],
            outputs = 'gdf_climate_areas',
            name = 'create_climate_areas',
            tags = ['extract']
        ),
        node(
            func = create_planet_subscriptions,
            inputs = ['params:params_subscription', 'params:params_t', 'params:params_s', 'gdf_aoi', 'params:params_credentials'],
            outputs = 'dt_planet_catalog',
            name = 'extract_data_planet',
            tags = ['extract']
        ), 
        node(
            func = preprocess_planet,
            inputs = ['dt_planet_catalog', 'params:params_transform_1'],
            outputs = 'ds_request_1',
            name = 'preprocess_data_1',
            tags = ['extract']
        ),
        node(
            func = preprocess_planet,
            inputs = ['dt_planet_catalog', 'params:params_transform_2'],
            outputs = 'ds_request_2',
            name = 'preprocess_data_2',
            tags = ['extract']
        ),
        node(
            func = process_data_planet,
            inputs = ['ds_request_1', 'ds_request_2', 'ds_request', 'gdf_climate_areas', 'params:params_process_planet'],
            outputs = ['df_cluster', 'ds_climatology'],
            name = 'process_data',
            tags = ['process']
        ),
    ])


def create_pipeline(**kwargs) -> Pipeline:
    aoi = aoi_pipeline()
    extract = extract_pipeline()
    trigger = trigger_pipeline()
    viz = viz_pipeline()

    return aoi + extract + trigger + viz


def create_planet_pipeline(**kwargs) -> Pipeline:
    aoi = aoi_pipeline()
    planet = planet_pipeline()
    trigger = trigger_pipeline()
    viz = planet_viz_pipeline()

    return aoi + planet + trigger + viz