"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline, pipeline

from pricing_automation.pipelines.pipeline import create_pipeline, create_planet_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = {}
    pipelines['pricing'] = create_pipeline()
    pipelines['planet'] = create_planet_pipeline()
    pipelines["__default__"] = pipelines['pricing']
    return pipelines
