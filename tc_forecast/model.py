"""
Model architecture module
GRU-based tropical cyclone track forecasting model
"""


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import List, Tuple
import numpy as np


from .config import Config




class TCForecastModel:
   """
   Tropical Cyclone Forecast Model
  
   Supports GRU-based architecture (extensible to GRU-CNN hybrid)
   All parameters configurable via config file.
  
   Example:
       >>> from tc_forecast import TCForecastModel, load_config
       >>> config = load_config('config.yaml')
       >>> model = TCForecastModel(config)
       >>> keras_model = model.build()
       >>> keras_model.summary()
   """
  
   def __init__(self, config: Config):
       """
       Initialize model with configuration
      
       Args:
           config: Configuration object from load_config()
       """
       self.config = config
       self.model_config = config.model
       self.data_config = config.data
      
       # Set random seed
       if hasattr(config, 'seed'):
           tf.random.set_seed(config.seed)
           np.random.seed(config.seed)
      
       self.model = None
  
   def build(self, input_shape: Tuple[int, int] = None) -> Model:
       """
       Build the forecast model
      
       Args:
           input_shape: (sequence_length, n_features).
                       If None, inferred from config
                      
       Returns:
           Compiled Keras Model
       """
       if input_shape is None:
           seq_len = self.data_config.sequence_length
           n_features = len(self.data_config.feature_columns)
           input_shape = (seq_len, n_features)
      
       model_type = self.model_config.type.upper()
      
       if model_type == "GRU":
           self.model = self._build_gru_model(input_shape)
       elif model_type == "CNN":
           self.model = self._build_cnn_model(input_shape)
       elif model_type == "GRU_CNN":
           self.model = self._build_gru_cnn_model(input_shape)
       else:
           raise ValueError(f"Unknown model type: {model_type}. Use 'GRU', 'CNN', or 'GRU_CNN'.")
      
       # Compile model
       self._compile_model()
      
       return self.model
  
   def _build_gru_model(self, input_shape: Tuple[int, int]) -> Model:
       """
       Build GRU-only model (track data only)
      
       Architecture:
       Input -> GRU layers -> Dense layers -> Output (lat/lon offsets)
      
       Args:
           input_shape: (sequence_length, n_features)
          
       Returns:
           Keras Model
       """
       gru_config = self.model_config.gru
       dense_config = self.model_config.dense
       output_config = self.model_config.output
      
       # Input layer
       inputs = layers.Input(shape=input_shape, name='track_input')
       x = inputs
      
       # GRU layers
       hidden_units = gru_config.hidden_units
       return_sequences = gru_config.return_sequences
       dropout = gru_config.dropout
       recurrent_dropout = gru_config.get('recurrent_dropout', 0.0)
       activation = gru_config.activation
      
       for i, (units, return_seq) in enumerate(zip(hidden_units, return_sequences)):
           x = layers.GRU(
               units=units,
               return_sequences=return_seq,
               activation=activation,
               dropout=dropout,
               recurrent_dropout=recurrent_dropout,
               name=f'gru_{i+1}'
           )(x)
      
       # Dense layers
       dense_units = dense_config.units
       dense_activation = dense_config.activation
       dense_dropout = dense_config.dropout
      
       for i, units in enumerate(dense_units):
           x = layers.Dense(units, activation=dense_activation, name=f'dense_{i+1}')(x)
           if dense_dropout > 0:
               x = layers.Dropout(dense_dropout, name=f'dropout_{i+1}')(x)
      
       # Output layer
       output_units = output_config.units
       output_activation = output_config.activation
      
       outputs = layers.Dense(output_units, activation=output_activation, name='output')(x)
      
       # Create model
       model = Model(inputs=inputs, outputs=outputs, name='TC_Forecast_GRU')
      
       return model
  
   def _build_cnn_model(self, input_shape: Tuple[int, int]) -> Model:
       """
       Build CNN-only model (track data processed with Conv1D)
      
       Architecture:
       Input -> Conv1D layers -> Flatten -> Dense layers -> Output (lat/lon offsets)
      
       Based on paper architecture for spatial feature extraction.
      
       Args:
           input_shape: (sequence_length, n_features)
          
       Returns:
           Keras Model
       """
       cnn_config = self.model_config.get('cnn', {
           'filters': [64, 32],
           'kernel_sizes': [3, 3],
           'pool_sizes': [2, 2],
           'activation': 'relu',
           'dropout': 0.2
       })
       dense_config = self.model_config.dense
       output_config = self.model_config.output
      
       # Input layer
       inputs = layers.Input(shape=input_shape, name='track_input')
       x = inputs
      
       # CNN layers
       filters = cnn_config.get('filters', [64, 32])
       kernel_sizes = cnn_config.get('kernel_sizes', [3, 3])
       pool_sizes = cnn_config.get('pool_sizes', [2, 2])
       activation = cnn_config.get('activation', 'relu')
       dropout = cnn_config.get('dropout', 0.2)
      
       for i, (n_filters, kernel_size, pool_size) in enumerate(zip(filters, kernel_sizes, pool_sizes)):
           x = layers.Conv1D(
               filters=n_filters,
               kernel_size=kernel_size,
               activation=activation,
               padding='same',
               name=f'conv1d_{i+1}'
           )(x)
           x = layers.BatchNormalization(name=f'bn_{i+1}')(x)
           if pool_size > 1:
               x = layers.MaxPooling1D(pool_size=pool_size, name=f'pool_{i+1}')(x)
           if dropout > 0:
               x = layers.Dropout(dropout, name=f'conv_dropout_{i+1}')(x)
      
       # Flatten
       x = layers.Flatten(name='flatten')(x)
      
       # Dense layers
       dense_units = dense_config.units
       dense_activation = dense_config.activation
       dense_dropout = dense_config.dropout
      
       for i, units in enumerate(dense_units):
           x = layers.Dense(units, activation=dense_activation, name=f'dense_{i+1}')(x)
           if dense_dropout > 0:
               x = layers.Dropout(dense_dropout, name=f'dropout_{i+1}')(x)
      
       # Output layer
       output_units = output_config.units
       output_activation = output_config.activation
      
       outputs = layers.Dense(output_units, activation=output_activation, name='output')(x)
      
       # Create model
       model = Model(inputs=inputs, outputs=outputs, name='TC_Forecast_CNN')
      
       return model
  
   def _build_gru_cnn_model(self, input_shape: Tuple[int, int]) -> Model:
       """
       Build GRU-CNN hybrid model
      
       Architecture:
       Input -> Parallel branches:
         1. CNN branch (extract spatial/local patterns)
         2. GRU branch (model temporal dependencies)
       -> Concatenate -> Dense layers -> Output (lat/lon offsets)
      
       Based on paper's GRU-CNN architecture combining CNN feature extraction
       with GRU temporal modeling.
      
       Args:
           input_shape: (sequence_length, n_features)
          
       Returns:
           Keras Model
       """
       gru_config = self.model_config.gru
       cnn_config = self.model_config.get('cnn', {
           'filters': [64, 32],
           'kernel_sizes': [3, 3],
           'pool_sizes': [2, 2],
           'activation': 'relu',
           'dropout': 0.2
       })
       dense_config = self.model_config.dense
       output_config = self.model_config.output
      
       # Input layer
       inputs = layers.Input(shape=input_shape, name='track_input')
      
       # ==================== GRU Branch ====================
       gru_x = inputs
       hidden_units = gru_config.hidden_units
       return_sequences = gru_config.return_sequences
       dropout = gru_config.dropout
       recurrent_dropout = gru_config.get('recurrent_dropout', 0.0)
       activation = gru_config.activation
      
       for i, (units, return_seq) in enumerate(zip(hidden_units, return_sequences)):
           gru_x = layers.GRU(
               units=units,
               return_sequences=return_seq,
               activation=activation,
               dropout=dropout,
               recurrent_dropout=recurrent_dropout,
               name=f'gru_{i+1}'
           )(gru_x)
      
       # ==================== CNN Branch ====================
       cnn_x = inputs
       filters = cnn_config.get('filters', [64, 32])
       kernel_sizes = cnn_config.get('kernel_sizes', [3, 3])
       pool_sizes = cnn_config.get('pool_sizes', [2, 2])
       cnn_activation = cnn_config.get('activation', 'relu')
       cnn_dropout = cnn_config.get('dropout', 0.2)
      
       for i, (n_filters, kernel_size, pool_size) in enumerate(zip(filters, kernel_sizes, pool_sizes)):
           cnn_x = layers.Conv1D(
               filters=n_filters,
               kernel_size=kernel_size,
               activation=cnn_activation,
               padding='same',
               name=f'conv1d_{i+1}'
           )(cnn_x)
           cnn_x = layers.BatchNormalization(name=f'bn_{i+1}')(cnn_x)
           if pool_size > 1:
               cnn_x = layers.MaxPooling1D(pool_size=pool_size, name=f'pool_{i+1}')(cnn_x)
           if cnn_dropout > 0:
               cnn_x = layers.Dropout(cnn_dropout, name=f'conv_dropout_{i+1}')(cnn_x)
      
       # Flatten CNN output
       cnn_x = layers.Flatten(name='cnn_flatten')(cnn_x)
      
       # Add dense layer for CNN features (optional)
       cnn_x = layers.Dense(32, activation=cnn_activation, name='cnn_dense')(cnn_x)
      
       # ==================== Concatenate Branches ====================
       x = layers.Concatenate(name='concat')([gru_x, cnn_x])
      
       # ==================== Dense Layers ====================
       dense_units = dense_config.units
       dense_activation = dense_config.activation
       dense_dropout = dense_config.dropout
      
       for i, units in enumerate(dense_units):
           x = layers.Dense(units, activation=dense_activation, name=f'dense_{i+1}')(x)
           if dense_dropout > 0:
               x = layers.Dropout(dense_dropout, name=f'dropout_{i+1}')(x)
      
       # Output layer
       output_units = output_config.units
       output_activation = output_config.activation
      
       outputs = layers.Dense(output_units, activation=output_activation, name='output')(x)
      
       # Create model
       model = Model(inputs=inputs, outputs=outputs, name='TC_Forecast_GRU_CNN')
      
       return model
  
   def _compile_model(self):
       """Compile model with optimizer, loss, and metrics from config"""
       training_config = self.config.training
      
       # Create learning rate schedule
       lr_config = training_config.optimizer.learning_rate
       schedule_type = lr_config.get('schedule', 'exponential').lower()
      
       if schedule_type == 'exponential':
           lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
               initial_learning_rate=lr_config.initial,
               decay_steps=lr_config.decay_steps,
               decay_rate=lr_config.decay_rate
           )
       elif schedule_type == 'constant':
           lr_schedule = lr_config.initial
       else:
           lr_schedule = lr_config.initial
      
       # Create optimizer
       optimizer_type = training_config.optimizer.type.lower()
       if optimizer_type == 'adam':
           optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
       elif optimizer_type == 'sgd':
           optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
       else:
           optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
      
       # Get loss
       loss = training_config.loss
       if loss == 'mse':
           loss_fn = 'mean_squared_error'
       elif loss == 'mae':
           loss_fn = 'mean_absolute_error'
       else:
           loss_fn = loss
      
       # Get metrics
       metrics_list = training_config.metrics
       keras_metrics = []
       for metric in metrics_list:
           if metric == 'mse':
               keras_metrics.append('mean_squared_error')
           elif metric == 'mae':
               keras_metrics.append('mean_absolute_error')
           elif metric == 'rmse':
               keras_metrics.append(tf.keras.metrics.RootMeanSquaredError())
           else:
               keras_metrics.append(metric)
      
       # Compile
       self.model.compile(
           optimizer=optimizer,
           loss=loss_fn,
           metrics=keras_metrics
       )
      
       print(f"Model compiled with {optimizer_type} optimizer, {loss} loss")
  
   def get_model(self) -> Model:
       """
       Get the built Keras model
      
       Returns:
           Keras Model (builds if not already built)
       """
       if self.model is None:
           self.build()
       return self.model
  
   def summary(self):
       """Print model summary"""
       if self.model is None:
           self.build()
       self.model.summary()
  
   def count_parameters(self) -> int:
       """
       Count total trainable parameters
      
       Returns:
           Number of trainable parameters
       """
       if self.model is None:
           self.build()
       return int(np.sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]))
  
   def save(self, filepath: str):
       """
       Save model weights
      
       Args:
           filepath: Path to save model
       """
       if self.model is None:
           raise ValueError("Model not built yet")
       self.model.save_weights(filepath)
       print(f"Model saved to {filepath}")
  
   def load(self, filepath: str):
       """
       Load model weights
      
       Args:
           filepath: Path to load model from
       """
       if self.model is None:
           self.build()
       self.model.load_weights(filepath)
       print(f"Model loaded from {filepath}")
  
   def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
       """
       Make predictions
      
       Args:
           X: Input sequences (n_samples, seq_len, n_features)
           batch_size: Batch size for prediction
          
       Returns:
           Predictions (n_samples, output_dim)
       """
       if self.model is None:
           raise ValueError("Model not built yet")
      
       return self.model.predict(X, batch_size=batch_size, verbose=0)


