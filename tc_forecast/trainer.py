"""
Training pipeline module
Handles model training, validation, and evaluation
"""


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import math


from .config import Config
from .model import TCForecastModel




def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
   """
   Calculate great circle distance between two points on Earth
  
   Args:
       lat1, lon1: First point coordinates
       lat2, lon2: Second point coordinates
      
   Returns:
       Distance in kilometers
   """
   ra = 6378.137  # Earth radius in km
  
   # Convert to radians
   lat1_rad = math.radians(lat1)
   lon1_rad = math.radians(lon1)
   lat2_rad = math.radians(lat2)
   lon2_rad = math.radians(lon2)
  
   # Haversine formula
   dlat = lat2_rad - lat1_rad
   dlon = lon2_rad - lon1_rad
  
   a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
   c = 2 * math.asin(math.sqrt(a))
  
   distance = ra * c
   return distance




class TCTrainer:
   """
   Tropical Cyclone Model Trainer
  
   Handles complete training pipeline including:
   - Training with callbacks
   - Validation
   - Evaluation with distance metrics
   - Visualization
  
   Example:
       >>> from tc_forecast import TCTrainer, TCForecastModel, load_config
       >>> config = load_config('config.yaml')
       >>> model = TCForecastModel(config)
       >>> trainer = TCTrainer(config, model)
       >>> history = trainer.train(X_train, y_train, X_val, y_val)
       >>> results = trainer.evaluate(X_test, y_test)
   """
  
   def __init__(self, config: Config, model: TCForecastModel):
       """
       Initialize trainer
      
       Args:
           config: Configuration object
           model: TCForecastModel instance
       """
       self.config = config
       self.model = model
       self.training_config = config.training
       self.eval_config = config.evaluation
       self.history = None
      
       # Create output directory
       output_path = Path(self.eval_config.output_path)
       output_path.mkdir(parents=True, exist_ok=True)
       self.output_path = output_path
  
   def _create_callbacks(self) -> List[keras.callbacks.Callback]:
       """
       Create training callbacks from config
      
       Returns:
           List of Keras callbacks
       """
       callbacks = []
       cb_config = self.training_config.callbacks
      
       # Early stopping
       if cb_config.early_stopping.enabled:
           es_config = cb_config.early_stopping
           callbacks.append(
               keras.callbacks.EarlyStopping(
                   monitor=es_config.monitor,
                   patience=es_config.patience,
                   restore_best_weights=es_config.restore_best_weights,
                   verbose=1
               )
           )
      
       # Model checkpoint
       if cb_config.model_checkpoint.enabled:
           mc_config = cb_config.model_checkpoint
           checkpoint_path = Path(mc_config.filepath)
           checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
          
           callbacks.append(
               keras.callbacks.ModelCheckpoint(
                   filepath=str(checkpoint_path),
                   monitor=mc_config.monitor,
                   save_best_only=mc_config.save_best_only,
                   save_weights_only=True,
                   verbose=1
               )
           )
      
       # Reduce learning rate
       if cb_config.reduce_lr.enabled:
           rl_config = cb_config.reduce_lr
           callbacks.append(
               keras.callbacks.ReduceLROnPlateau(
                   monitor=rl_config.monitor,
                   factor=rl_config.factor,
                   patience=rl_config.patience,
                   min_lr=rl_config.min_lr,
                   verbose=1
               )
           )
      
       return callbacks
  
   def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray = None, y_val: np.ndarray = None
             ) -> keras.callbacks.History:
       """
       Train the model
      
       Args:
           X_train: Training input sequences
           y_train: Training targets
           X_val: Validation input sequences (optional)
           y_val: Validation targets (optional)
          
       Returns:
           Training history
       """
       print("Starting Model Training")
      
       # Get model
       keras_model = self.model.get_model()
      
       # Print model summary
       keras_model.summary()
       print(f"\nTotal parameters: {self.model.count_parameters():,}")
      
       # Create callbacks
       callbacks = self._create_callbacks()
      
       # Prepare validation data
       validation_data = None
       if X_val is not None and y_val is not None:
           validation_data = (X_val, y_val)
      
       # Training parameters
       batch_size = self.training_config.batch_size
       epochs = self.training_config.epochs
      
       print(f"\nTraining config:")
       print(f"  Batch size: {batch_size}")
       print(f"  Epochs: {epochs}")
       print(f"  Train samples: {len(X_train)}")
       if validation_data:
           print(f"  Val samples: {len(X_val)}")
       print()
      
       # Train
       self.history = keras_model.fit(
           X_train, y_train,
           validation_data=validation_data,
           batch_size=batch_size,
           epochs=epochs,
           callbacks=callbacks,
           verbose=1
       )
      
       print("Training Complete!")
      
       # Plot training history
       self._plot_training_history()
      
       return self.history
  
   def _plot_training_history(self):
       """Plot and save training history"""
       if self.history is None:
           return
      
       history = self.history.history
      
       fig, axes = plt.subplots(1, 2, figsize=(14, 5))
      
       # Plot loss
       axes[0].plot(history['loss'], label='Train Loss')
       if 'val_loss' in history:
           axes[0].plot(history['val_loss'], label='Val Loss')
       axes[0].set_xlabel('Epoch')
       axes[0].set_ylabel('Loss')
       axes[0].set_title('Training and Validation Loss')
       axes[0].legend()
       axes[0].grid(True, alpha=0.3)
      
       # Plot MAE if available
       if 'mean_absolute_error' in history:
           axes[1].plot(history['mean_absolute_error'], label='Train MAE')
           if 'val_mean_absolute_error' in history:
               axes[1].plot(history['val_mean_absolute_error'], label='Val MAE')
           axes[1].set_xlabel('Epoch')
           axes[1].set_ylabel('MAE')
           axes[1].set_title('Mean Absolute Error')
           axes[1].legend()
           axes[1].grid(True, alpha=0.3)
      
       plt.tight_layout()
      
       # Save plot
       plot_path = self.output_path / 'training_history.png'
       plt.savefig(plot_path, dpi=150, bbox_inches='tight')
       print(f"Training history plot saved to {plot_path}")
      
       if self.eval_config.plot_results:
           plt.show()
       plt.close()
  
   def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
               metadata: List[Dict] = None, data_loader = None
               ) -> Dict[str, any]:
       """
       Evaluate model on test set
      
       Args:
           X_test: Test input sequences
           y_test: Test targets (normalized)
           metadata: Test metadata with last_lat, last_lon
           data_loader: TCDataLoader for inverse transform
          
       Returns:
           Dictionary with evaluation metrics
       """
       print("Model Evaluation")
      
       # Make predictions
       y_pred = self.model.predict(X_test, batch_size=32)
      
       # Inverse transform if data_loader provided
       if data_loader is not None:
           y_pred_denorm, y_test_denorm = data_loader.inverse_transform_predictions(y_pred, y_test)
       else:
           y_pred_denorm = y_pred
           y_test_denorm = y_test
      
       # Calculate metrics
       results = {}
      
       # MSE and MAE on offsets
       mse = np.mean((y_test_denorm - y_pred_denorm) ** 2)
       mae = np.mean(np.abs(y_test_denorm - y_pred_denorm))
       rmse = np.sqrt(mse)
      
       results['mse'] = float(mse)
       results['mae'] = float(mae)
       results['rmse'] = float(rmse)
      
       print(f"\nOverall Metrics:")
       print(f"  RMSE: {rmse:.4f}")
       print(f"  MAE: {mae:.4f}")
       print(f"  MSE: {mse:.4f}")
      
       # Calculate distance errors if metadata provided
       if metadata is not None:
           distance_errors = self._calculate_distance_errors(
               y_pred_denorm, y_test_denorm, metadata
           )
           results['distance_errors'] = distance_errors
          
           # Per-timestep errors
           if self.eval_config.per_timestep:
               timestep_errors = self._calculate_per_timestep_errors(
                   y_pred_denorm, y_test_denorm
               )
               results['timestep_errors'] = timestep_errors
      
       # Save predictions
       if self.eval_config.save_predictions:
           self._save_predictions(y_pred_denorm, y_test_denorm, results)
      
       print("Evaluation Complete!")
      
       return results
  
   def _calculate_distance_errors(self, y_pred: np.ndarray, y_true: np.ndarray,
                                  metadata: List[Dict]) -> Dict[str, List[float]]:
       """
       Calculate haversine distance errors for each forecast timestep
      
       Args:
           y_pred: Predicted offsets (n_samples, 24)
           y_true: True offsets (n_samples, 24)
           metadata: List with last_lat, last_lon for each sample
          
       Returns:
           Dict with mean distance errors per forecast time
       """
       forecast_steps = self.config.data.forecast_steps
       n_samples = len(y_pred)
      
       # Store distance errors for each forecast step
       distance_errors_per_step = [[] for _ in range(forecast_steps)]
      
       for i in range(n_samples):
           if i >= len(metadata):
               break
          
           last_lat = metadata[i]['last_lat']
           last_lon = metadata[i]['last_lon']
          
           for step in range(forecast_steps):
               # Extract lat/lon offsets
               lat_pred = y_pred[i, step * 2] + last_lat
               lon_pred = y_pred[i, step * 2 + 1] + last_lon
              
               lat_true = y_true[i, step * 2] + last_lat
               lon_true = y_true[i, step * 2 + 1] + last_lon
              
               # Calculate distance
               dist = haversine_distance(lat_pred, lon_pred, lat_true, lon_true)
               distance_errors_per_step[step].append(dist)
      
       # Calculate mean and std for each step
       mean_errors = [np.mean(errors) for errors in distance_errors_per_step]
       std_errors = [np.std(errors) for errors in distance_errors_per_step]
      
       # Print results
       time_interval = self.config.data.time_interval
       print(f"\nDistance Errors (km) by Forecast Time:")
       print(f"{'Time (h)':>10} {'Mean Error':>15} {'Std':>12}")
       for step, (mean_err, std_err) in enumerate(zip(mean_errors, std_errors)):
           forecast_hour = (step + 1) * time_interval
           print(f"{forecast_hour:>10} {mean_err:>15.2f} {std_err:>12.2f}")
      
       # Highlight key forecast hours if configured
       key_hours = getattr(self.config.data, 'key_forecast_hours', None)
       if key_hours:
           print(f"KEY FORECAST HOURS SUMMARY:")
           print(f"{'Time (h)':>10} {'Mean Error':>15} {'Std':>12}")
           for target_hour in key_hours:
               step_idx = (target_hour // time_interval) - 1
               if 0 <= step_idx < len(mean_errors):
                   print(f"{target_hour:>10} {mean_errors[step_idx]:>15.2f} {std_errors[step_idx]:>12.2f}")
      
       return {
           'mean': mean_errors,
           'std': std_errors,
           'all_errors': distance_errors_per_step
       }
  
   def _calculate_per_timestep_errors(self, y_pred: np.ndarray, y_true: np.ndarray
                                      ) -> Dict[str, List[float]]:
       """Calculate RMSE/MAE for each forecast timestep"""
       forecast_steps = self.config.data.forecast_steps
      
       rmse_per_step = []
       mae_per_step = []
      
       for step in range(forecast_steps):
           # Lat error
           lat_pred = y_pred[:, step * 2]
           lat_true = y_true[:, step * 2]
          
           # Lon error
           lon_pred = y_pred[:, step * 2 + 1]
           lon_true = y_true[:, step * 2 + 1]
          
           # Combined RMSE and MAE
           mse_lat = np.mean((lat_pred - lat_true) ** 2)
           mse_lon = np.mean((lon_pred - lon_true) ** 2)
           rmse = np.sqrt((mse_lat + mse_lon) / 2)
          
           mae_lat = np.mean(np.abs(lat_pred - lat_true))
           mae_lon = np.mean(np.abs(lon_pred - lon_true))
           mae = (mae_lat + mae_lon) / 2
          
           rmse_per_step.append(rmse)
           mae_per_step.append(mae)
      
       # Print per-timestep summary
       time_interval = self.config.data.time_interval
       print(f"\nRMSE/MAE by Forecast Time:")
       print(f"{'Time (h)':>10} {'RMSE':>12} {'MAE':>12}")
       for step, (rmse_val, mae_val) in enumerate(zip(rmse_per_step, mae_per_step)):
           forecast_hour = (step + 1) * time_interval
           print(f"{forecast_hour:>10} {rmse_val:>12.4f} {mae_val:>12.4f}")
      
       # Highlight key forecast hours if configured
       key_hours = getattr(self.config.data, 'key_forecast_hours', None)
       if key_hours:
           print(f"KEY FORECAST HOURS:")
           print(f"{'Time (h)':>10} {'RMSE':>12} {'MAE':>12}")
           for target_hour in key_hours:
               step_idx = (target_hour // time_interval) - 1
               if 0 <= step_idx < len(rmse_per_step):
                   print(f"{target_hour:>10} {rmse_per_step[step_idx]:>12.4f} {mae_per_step[step_idx]:>12.4f}")
      
       return {
           'rmse': rmse_per_step,
           'mae': mae_per_step
       }
  
   def _save_predictions(self, y_pred: np.ndarray, y_true: np.ndarray, results: Dict):
       """Save predictions and results to files"""
       timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      
       # Save predictions as numpy arrays
       pred_path = self.output_path / f'predictions_{timestamp}.npz'
       np.savez(pred_path, predictions=y_pred, ground_truth=y_true)
       print(f"\nPredictions saved to {pred_path}")
      
       # Save metrics as JSON
       metrics_path = self.output_path / f'metrics_{timestamp}.json'
      
       # Convert numpy types to Python types for JSON serialization
       results_json = {}
       for key, value in results.items():
           if isinstance(value, dict):
               results_json[key] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                   for k, v in value.items()}
           elif isinstance(value, np.ndarray):
               results_json[key] = value.tolist()
           else:
               results_json[key] = value
      
       with open(metrics_path, 'w') as f:
           json.dump(results_json, f, indent=2)
      
       print(f"Metrics saved to {metrics_path}")
