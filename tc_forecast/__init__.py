"""
TC Forecast Package
Modular tropical cyclone track forecasting using GRU-CNN hybrid models


Based on: "Forecasting tropical cyclone tracks in the northwestern Pacific
based on a deep-learning model" (GMD, 2023)
Paper: https://gmd.copernicus.org/articles/16/2167/2023/
"""


__version__ = "2.0.0"
__author__ = "TC Forecast Team"


from .config import load_config


# Data loader with 19-feature computation and environmental data support
from .data_loader import TCDataLoader, get_distance


# Try to import TensorFlow-dependent modules
TCForecastModel = None
TCTrainer = None
calculate_distance_error = None


try:
    from .model import TCForecastModel, calculate_distance_error
    from .trainer import TCTrainer
except ImportError as e:
    # TensorFlow not installed - model and trainer won't be available
    import warnings
    warnings.warn(
        f"Could not import models/trainer: {e}. Install TensorFlow to use these features.")


__all__ = [
    # Config
    'load_config',

    # Data loader
    'TCDataLoader',
    'get_distance',

    # Model
    'TCForecastModel',
    'calculate_distance_error',

    # Training
    'TCTrainer',
]
