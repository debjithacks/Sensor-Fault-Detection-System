# main.py
# Shim to satisfy pickle imports for old pipelines.

# Wafer pipeline expects: main.WaferAggregator
from custom_transformers import WaferAggregator

# Gas pipeline expects: main.FeatureEngineer
from custom_transformers import FeatureEngineer

# Soil pipeline expects soil_pipeline.SoilSensorPipeline,
# but some soil pickles also reference main.SoilSensorPipeline
from custom_transformers import SoilSensorPipeline
