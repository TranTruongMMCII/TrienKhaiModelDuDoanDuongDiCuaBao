"""
Model architecture module
GRU-CNN tropical cyclone track forecasting model


Based on: "Forecasting tropical cyclone tracks in the northwestern Pacific
based on a deep-learning model" (GMD, 2023)


Architecture from paper:
- GRU branch: Track data (8 timesteps × 11 features)
- CNN UV branch: Wind data (8 channels × 21 × 21)
- CNN SST branch: Sea surface temperature (1 channel × 21 × 21)
- CNN Z branch: Geopotential height (3 channels × 46 × 81)
- Concatenate → Dense → Output (12 steps × 2 lat/lon)
"""


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import List, Tuple, Dict, Optional
import numpy as np


from .config import Config


def calculate_distance_error(y_true: np.ndarray, y_pred: np.ndarray,
                             metadata: List[Dict] = None) -> np.ndarray:
    """
    Calculate distance error (km) between predicted and true positions.

    Uses Haversine formula to compute great-circle distance.

    Args:
        y_true: True lat/lon offsets (n_samples, forecast_steps * 2)
        y_pred: Predicted lat/lon offsets
        metadata: Optional metadata with last known positions

    Returns:
        Distance errors in km (n_samples, forecast_steps)
    """
    n_samples = y_true.shape[0]
    n_outputs = y_true.shape[1]
    n_steps = n_outputs // 2

    # Reshape to (n_samples, n_steps, 2)
    y_true_reshaped = y_true.reshape(n_samples, n_steps, 2)
    y_pred_reshaped = y_pred.reshape(n_samples, n_steps, 2)

    # Calculate distance for each step
    R = 6371.0  # Earth radius in km
    errors = np.zeros((n_samples, n_steps))

    for i in range(n_steps):
        lat_true = y_true_reshaped[:, i, 0]
        lon_true = y_true_reshaped[:, i, 1]
        lat_pred = y_pred_reshaped[:, i, 0]
        lon_pred = y_pred_reshaped[:, i, 1]

        # Haversine formula
        dlat = np.radians(lat_pred - lat_true)
        dlon = np.radians(lon_pred - lon_true)
        lat1 = np.radians(lat_true)
        lat2 = np.radians(lat_pred)

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * \
            np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        errors[:, i] = R * c

    return errors


class TCForecastModel:
    """
    Tropical Cyclone Forecast Model

    Implements the GRU-CNN architecture from the original paper:
    - 4 input branches (GRU for track, CNN for UV/SST/Z)
    - Multi-optimizer support
    - Configurable via YAML

    Example:
        >>> from tc_forecast import TCForecastModel, load_config
        >>> config = load_config('config.yaml')
        >>> model = TCForecastModel(config)
        >>> keras_model = model.build()
        >>> keras_model.summary()
    """

    def __init__(self, config: Config):
        """
        Initialize model with configuration.

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

    def build(self, track_shape: Tuple[int, int] = None,
              uv_shape: Tuple[int, int, int] = None,
              sst_shape: Tuple[int, int, int] = None,
              z_shape: Tuple[int, int, int] = None) -> Model:
        """
        Build the forecast model.

        Args:
            track_shape: (sequence_length, n_features) - default (8, 11)
            uv_shape: (channels, height, width) - default (8, 21, 21)
            sst_shape: (channels, height, width) - default (1, 21, 21)
            z_shape: (channels, height, width) - default (3, 46, 81)

        Returns:
            Compiled Keras Model
        """
        model_type = self.model_config.type.upper()

        if model_type == "GRU":
            self.model = self._build_gru_only_model(track_shape)
        elif model_type == "GRU_CNN":
            self.model = self._build_gru_cnn_model(
                track_shape, uv_shape, sst_shape, z_shape)
        else:
            raise ValueError(
                f"Unknown model type: {model_type}. Use 'GRU' or 'GRU_CNN'.")

        # Compile model
        self._compile_model()

        return self.model

    def _build_gru_only_model(self, track_shape: Tuple[int, int] = None) -> Model:
        """
        Build GRU-only model (track data only).

        Architecture: Input → GRU layers → Dense layers → Output

        Args:
            track_shape: (sequence_length, n_features)

        Returns:
            Keras Model
        """
        if track_shape is None:
            seq_len = self.data_config.sequence_length
            # Use top 11 features by default
            n_features = 11 if self.data_config.get('use_top_11', True) else 19
            track_shape = (seq_len, n_features)

        gru_config = self.model_config.gru
        dense_config = self.model_config.dense
        output_config = self.model_config.output

        # Input layer
        inputs = layers.Input(shape=track_shape, name='track_input')
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
            x = layers.Dense(units, activation=dense_activation,
                             name=f'dense_{i+1}')(x)
            if dense_dropout > 0:
                x = layers.Dropout(dense_dropout, name=f'dropout_{i+1}')(x)

        # Output layer: 12 steps × 2 (lat/lon) = 24 values
        output_units = output_config.units
        output_activation = output_config.activation

        outputs = layers.Dense(
            output_units, activation=output_activation, name='output')(x)

        model = Model(inputs=inputs, outputs=outputs, name='TC_GRU_Model')

        return model

    def _build_gru_cnn_model(self, track_shape: Tuple[int, int] = None,
                             uv_shape: Tuple[int, int, int] = None,
                             sst_shape: Tuple[int, int, int] = None,
                             z_shape: Tuple[int, int, int] = None) -> Model:
        """
        Build GRU-CNN hybrid model with 4 branches.

        Architecture from paper (gru_cnn_uv_sst_p_track.py):

        Branch 1 - GRU (Track):
            Input (8, 11) -> GRU(128, relu, return_seq=True) -> GRU(32, relu)

        Branch 2 - CNN UV:
            Input (C, 21, 21) -> Permute -> Conv2D(16, 9x9, s=3, relu) -> BN
            -> MaxPool(3x3, s=2) -> Flatten -> Dense(128) -> Dense(32)

        Branch 3 - CNN SST:
            Input (1, 21, 21) -> Permute -> Conv2D(8, 9x9, s=3, relu) -> BN
            -> MaxPool(3x3, s=2) -> Flatten -> Dense(128) -> Dense(32)

        Branch 4 - CNN Z:
            Input (3, H, W) -> Permute -> Conv2D(16, adapted kernel) -> BN
            -> MaxPool -> Flatten -> Dense(128) -> Dense(32)

        Merge: Concatenate -> Dense(24, relu)
        """
        if track_shape is None:
            seq_len = self.data_config.sequence_length
            n_features = 11 if self.data_config.get('use_top_11', True) else 19
            track_shape = (seq_len, n_features)

        env_config = self.data_config.get('environmental', {})

        if uv_shape is None:
            uv_config = env_config.get(
                'uv', {'channels': 8, 'height': 21, 'width': 21})
            uv_shape = (uv_config['channels'],
                        uv_config['height'], uv_config['width'])

        if sst_shape is None:
            sst_config = env_config.get(
                'sst', {'channels': 1, 'height': 21, 'width': 21})
            sst_shape = (sst_config['channels'],
                         sst_config['height'], sst_config['width'])

        if z_shape is None:
            z_config = env_config.get(
                'z', {'channels': 3, 'height': 46, 'width': 81})
            z_shape = (z_config['channels'],
                       z_config['height'], z_config['width'])

        output_config = self.model_config.output

        # Input layers
        track_input = layers.Input(shape=track_shape, name='track_input')
        uv_input = layers.Input(shape=uv_shape, name='uv_input')
        sst_input = layers.Input(shape=sst_shape, name='sst_input')
        z_input = layers.Input(shape=z_shape, name='z_input')

        # Branch 1: GRU (Track) - paper: GRU(128) -> GRU(32), no dropout
        gru_x = layers.GRU(128, return_sequences=True, activation='relu',
                           name='gru_1')(track_input)
        gru_x = layers.GRU(32, return_sequences=False, activation='relu',
                           name='gru_2')(gru_x)

        # Branch 2: CNN UV - paper: Conv2D(16, 9x9, s=3) -> BN -> MaxPool(3,3, s=2)
        uv_x = layers.Permute((2, 3, 1), name='uv_permute')(uv_input)
        uv_x = layers.Conv2D(16, (9, 9), strides=3, activation='relu',
                             name='uv_conv')(uv_x)
        uv_x = layers.BatchNormalization(name='uv_bn')(uv_x)
        uv_x = layers.MaxPooling2D((3, 3), strides=2,
                                   name='uv_pool')(uv_x)
        uv_x = layers.Flatten(name='uv_flatten')(uv_x)
        uv_x = layers.Dense(128, activation='relu', name='uv_dense1')(uv_x)
        uv_x = layers.Dense(32, activation='relu', name='uv_dense2')(uv_x)

        # Branch 3: CNN SST - paper: Conv2D(8, 9x9, s=3) -> BN -> MaxPool(3,3, s=2)
        sst_x = layers.Permute((2, 3, 1), name='sst_permute')(sst_input)
        sst_x = layers.Conv2D(8, (9, 9), strides=3, activation='relu',
                              name='sst_conv')(sst_x)
        sst_x = layers.BatchNormalization(name='sst_bn')(sst_x)
        sst_x = layers.MaxPooling2D((3, 3), strides=2,
                                    name='sst_pool')(sst_x)
        sst_x = layers.Flatten(name='sst_flatten')(sst_x)
        sst_x = layers.Dense(128, activation='relu', name='sst_dense1')(sst_x)
        sst_x = layers.Dense(32, activation='relu', name='sst_dense2')(sst_x)

        # Branch 4: CNN Z - paper: Conv2D(16, 14x25, s=4) for 46x81 grid
        # Adapted kernel for smaller grids
        z_x = layers.Permute((2, 3, 1), name='z_permute')(z_input)
        z_h, z_w = z_shape[1], z_shape[2]
        if z_h >= 46 and z_w >= 81:
            z_x = layers.Conv2D(16, (14, 25), strides=4,
                                name='z_conv')(z_x)
            z_x = layers.BatchNormalization(name='z_bn')(z_x)
            z_x = layers.MaxPooling2D((5, 11), strides=2,
                                      name='z_pool')(z_x)
        else:
            z_x = layers.Conv2D(16, (5, 5), strides=2,
                                name='z_conv')(z_x)
            z_x = layers.BatchNormalization(name='z_bn')(z_x)
            z_x = layers.MaxPooling2D((2, 2), strides=2,
                                      name='z_pool')(z_x)
        z_x = layers.Flatten(name='z_flatten')(z_x)
        z_x = layers.Dense(128, activation='relu', name='z_dense1')(z_x)
        z_x = layers.Dense(32, activation='relu', name='z_dense2')(z_x)

        # Merge all branches -> output (paper: concat -> Dense(24, relu))
        merged = layers.Concatenate(name='merge_all')(
            [uv_x, sst_x, z_x, gru_x])

        output_units = output_config.units
        output_activation = output_config.activation
        outputs = layers.Dense(output_units, activation=output_activation,
                               name='output')(merged)

        model = Model(
            inputs=[track_input, uv_input, sst_input, z_input],
            outputs=outputs,
            name='TC_GRU_CNN_Model'
        )

        return model

    def _compile_model(self):
        """Compile model with optimizer, loss, and metrics from config."""
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

    def build_with_multi_optimizer(self, track_shape=None, uv_shape=None,
                                   sst_shape=None, z_shape=None):
        """
        Build model with multi-optimizer support (different LR per branch).

        Paper uses different learning rates for each branch:
        - GRU: 0.001
        - CNN branches: 0.0001

        Note: This requires tensorflow_addons for MultiOptimizer.
        For simplicity, we use a single optimizer with the main LR.
        """
        print("Note: Multi-optimizer support requires tensorflow_addons.")
        print("Using single optimizer with learning rate from config.")
        return self.build(track_shape, uv_shape, sst_shape, z_shape)

    def get_model(self) -> Model:
        """Get the built Keras model."""
        if self.model is None:
            self.build()
        return self.model

    def summary(self):
        """Print model summary."""
        if self.model is None:
            self.build()
        self.model.summary()

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        if self.model is None:
            self.build()
        return int(np.sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]))

    def save(self, filepath: str):
        """Save model weights."""
        if self.model is None:
            raise ValueError("Model not built yet")
        self.model.save_weights(filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load model weights."""
        if self.model is None:
            self.build()
        self.model.load_weights(filepath)
        print(f"Model loaded from {filepath}")

    def predict(self, inputs, batch_size: int = 32) -> np.ndarray:
        """
        Make predictions.

        Args:
            inputs: For GRU model: X_track (n_samples, seq_len, n_features)
                    For GRU_CNN model: [X_track, X_uv, X_sst, X_z]
            batch_size: Batch size for prediction

        Returns:
            Predictions (n_samples, output_dim)
        """
        if self.model is None:
            raise ValueError("Model not built yet")

        return self.model.predict(inputs, batch_size=batch_size, verbose=0)
