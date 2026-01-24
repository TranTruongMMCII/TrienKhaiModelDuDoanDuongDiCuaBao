"""
TC Forecast Package
Modular tropical cyclone track forecasting using GRU-CNN hybrid models
"""


from .data_loader import TCDataLoader
from .config import load_config


# Try to import TensorFlow-dependent modules
TCForecastModel = None
TCTrainer = None


try:
   from .model import TCForecastModel
   from .trainer import TCTrainer
except ImportError as e:
   # TensorFlow not installed - model and trainer won't be available
   import warnings
   warnings.warn(f"Could not import TCForecastModel/TCTrainer: {e}. Install TensorFlow to use these features.")


__all__ = [
   'TCDataLoader',
   'TCForecastModel',
   'TCTrainer',
   'load_config'
]
