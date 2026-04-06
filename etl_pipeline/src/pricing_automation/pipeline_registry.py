"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline, pipeline

from pricing_automation.pipelines.pipeline import create_pipeline_moisture, create_pipeline_temperature


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipeline_moisture = create_pipeline_moisture()
    pipeline_temperature = create_pipeline_temperature()

    pipelines = {}
    pipelines["swc"] = pipeline(
        pipeline_moisture, 
        namespace="swc", 
        inputs = ['gdf_request'],
        parameters = ['params_s', 'params_t', 'params_request', 'params_process', 'create_trigger', 'params_bootstrap', 'params_quote']
    )
    pipelines["prcp"] = pipeline(
        pipeline_moisture, 
        namespace="prcp", 
        inputs = ['gdf_request'],
        parameters = ['params_s', 'params_t', 'params_request','params_process', 'create_trigger', 'params_bootstrap', 'params_quote']
    )
    pipelines["temp"] = pipeline(
        pipeline_temperature, 
        namespace="temp", 
        inputs = ['gdf_request'],
        parameters = ['params_s', 'params_t', 'params_request','params_process', 'create_trigger', 'params_bootstrap', 'params_quote']
    )
    pipelines["__default__"] = pipelines['swc']
    return pipelines
