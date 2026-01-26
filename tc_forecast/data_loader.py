"""
Data loading and preprocessing module
Handles CSV reading, sequence creation, and normalization
Merges IBTrACS track data with atmospheric grid data from output folder
"""


import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.spatial import cKDTree
import warnings
import ast
from datetime import timedelta


from .config import Config, resolve_path




class TCDataLoader:
   """
   Tropical Cyclone Data Loader
  
   Loads track data from multiple CSV files and creates sequences for forecasting.
   All column names are configurable via config file.
  
   Example:
       >>> from tc_forecast import TCDataLoader, load_config
       >>> config = load_config('config.yaml')
       >>> loader = TCDataLoader(config)
       >>> X_train, y_train, X_val, y_val, X_test, y_test = loader.load_and_prepare()
   """
  
   def __init__(self, config: Config):
       """
       Initialize data loader with configuration
      
       Args:
           config: Configuration object from load_config()
       """
       self.config = config
       self.data_config = config.data
       self.scaler_X = None
       self.scaler_y = None
      
       if hasattr(config, 'seed'):
           np.random.seed(config.seed)
  
   def load_multiple_csvs(self, atmospheric_data_dir: str = None) -> pd.DataFrame:
       """
       Load and merge IBTrACS track data with atmospheric grid data
      
       Merges:
       - IBTrACS track data (path from config.data.ibtracs_path)
       - atmospheric_data_dir/*.csv: Atmospheric grids (sst, u*, v*, z*) for each timestep
      
       Args:
           atmospheric_data_dir: Path to directory containing atmospheric CSV files.
                                If None, uses config.data.atmospheric_data_dir
          
       Returns:
           Merged DataFrame with track info + atmospheric variables at storm center
       """
       # Load IBTrACS data from config
       ibtracs_path = resolve_path(self.data_config.ibtracs_path, Path(__file__).parent)
       if not ibtracs_path.exists():
           raise FileNotFoundError(f"IBTrACS file not found: {ibtracs_path}")
      
       print(f"Loading IBTrACS data from: {ibtracs_path}")
       ibtracs_df = pd.read_csv(ibtracs_path)
       ibtracs_df['ISO_TIME'] = pd.to_datetime(ibtracs_df['ISO_TIME'])
       print(f"Loaded {len(ibtracs_df)} IBTrACS records")
      
       # Load atmospheric data from config
       if atmospheric_data_dir is None:
           atmospheric_data_dir = resolve_path(self.data_config.atmospheric_data_dir, Path(__file__).parent)
       else:
           atmospheric_data_dir = Path(atmospheric_data_dir)

       if not atmospheric_data_dir.exists():
           raise FileNotFoundError(f"Atmospheric data directory not found: {atmospheric_data_dir}")
      
       print(f"Loading atmospheric data from: {atmospheric_data_dir}")
      
       # Atmospheric variables
       variable_files = {
           'sst': 'sst.csv',
           'u300': 'u300.csv',
           'u700': 'u700.csv',
           'u850': 'u850.csv',
           'v300': 'v300.csv',
           'v500': 'v500.csv',
           'v700': 'v700.csv',
           'v850': 'v850.csv',
           'z300': 'z300.csv',
           'z500': 'z500.csv',
           'z700': 'z700.csv'
       }
      
       atmos_dfs = {}
      
       for var_name, filename in variable_files.items():
           filepath = atmospheric_data_dir / filename
           if not filepath.exists():
               print(f"Warning: {filename} not found, skipping")
               continue
          
           df = pd.read_csv(filepath)
           df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'])
          
           # Extract center value from grid
           value_col = 'SST' if var_name == 'sst' else 'VALUE'
           df[var_name] = df[value_col].apply(self._extract_center_value)
          
           # Keep only necessary columns for merging
           df = df[['STORM_ID', 'ISO_TIME', 'LAT', 'LONG', var_name]]
           atmos_dfs[var_name] = df
           print(f"Loaded {filename}: {len(df)} rows")
      
       if not atmos_dfs:
           raise ValueError("No atmospheric data files found!")
      
       # Merge all atmospheric data
       print("\nMerging atmospheric data...")
       merged_atmos = None
       for var_name, df in atmos_dfs.items():
           if merged_atmos is None:
               merged_atmos = df
           else:
               # Merge on STORM_ID, ISO_TIME with tolerance for small time differences
               merged_atmos = pd.merge(
                   merged_atmos,
                   df,
                   on=['STORM_ID', 'ISO_TIME', 'LAT', 'LONG'],
                   how='outer'
               )
      
       print(f"Merged atmospheric data: {len(merged_atmos)} rows")
      
       # Merge IBTrACS with atmospheric data
       print("\nMerging IBTrACS with atmospheric data...")
       final_df = pd.merge(
           ibtracs_df,
           merged_atmos,
           on=['STORM_ID', 'ISO_TIME'],
           how='inner',
           suffixes=('_ibtracs', '_atmos')
       )
      
       # Use IBTrACS LAT/LONG as primary, fill missing atmospheric LAT/LONG
       if 'LAT_ibtracs' in final_df.columns:
           final_df['LAT'] = final_df['LAT_ibtracs']
           final_df['LONG'] = final_df['LONG_ibtracs']
           final_df.drop(['LAT_ibtracs', 'LONG_ibtracs', 'LAT_atmos', 'LONG_atmos'], axis=1, inplace=True, errors='ignore')
      
       print(f"Final merged data: {len(final_df)} rows, {len(final_df.columns)} columns")
       print(f"Columns: {final_df.columns.tolist()}")
      
       # Handle missing SST values using nearest neighbor
       if 'sst' in final_df.columns:
           sst_missing_count = final_df['sst'].isna().sum()
           if sst_missing_count > 0:
               print(f"\nFilling {sst_missing_count} missing SST values using nearest neighbor...")
               final_df = self._fill_sst_nearest(final_df)
      
       # Fill other atmospheric variables (forward/backward fill within each storm)
       atmos_vars = [col for col in final_df.columns if col in variable_files.keys()]
       for var in atmos_vars:
           if var == 'sst':
               continue  # Already handled
           missing_count = final_df[var].isna().sum()
           if missing_count > 0:
               print(f"Filling {missing_count} missing {var} values (forward/backward fill)...")
               final_df[var] = final_df.groupby('STORM_ID')[var].transform(lambda x: x.ffill().bfill())
      
       # Sort by STORM_ID and time
       final_df = final_df.sort_values(['STORM_ID', 'ISO_TIME']).reset_index(drop=True)
      
       print(f"\nFinal dataset: {len(final_df)} records, {len(final_df['STORM_ID'].unique())} storms")
      
       return final_df
  
   def _fill_sst_nearest(self, df: pd.DataFrame) -> pd.DataFrame:
       """
       Fill missing SST values using nearest neighbor in time/space
      
       Args:
           df: DataFrame with potentially missing SST values
          
       Returns:
           DataFrame with SST filled
       """
       missing_mask = df['sst'].isna()
       if not missing_mask.any():
           return df
      
       # For each storm, fill missing SST using nearest timestep
       for storm_id in df[missing_mask]['STORM_ID'].unique():
           storm_mask = df['STORM_ID'] == storm_id
           storm_df = df[storm_mask].copy()
          
           # Forward fill then backward fill within storm
           df.loc[storm_mask, 'sst'] = storm_df['sst'].ffill().bfill()
      
       # If still missing, use global mean
       remaining_missing = df['sst'].isna().sum()
       if remaining_missing > 0:
           global_mean = df['sst'].mean()
           df['sst'].fillna(global_mean, inplace=True)
           print(f"  Used global mean SST for {remaining_missing} remaining missing values")
      
       return df
  
   def _parse_array(self, array_str):
       """Parse string representation of array to numpy array"""
       try:
           if pd.isna(array_str):
               return None
           parsed = ast.literal_eval(array_str)
           return np.array(parsed, dtype=np.float32)
       except:
           return None
  
   def _extract_center_value(self, array_str):
       """
       Extract center value from 2D grid array
      
       The VALUE column contains a 2D grid (e.g., 21x21) of atmospheric data
       around the storm center. We extract the center point as the most
       representative value for the storm's location.
      
       Args:
           array_str: String representation of 2D array
          
       Returns:
           Float value at center of grid, or NaN if parsing fails
       """
       try:
           if pd.isna(array_str):
               return np.nan
          
           # Parse string to 2D array
           parsed = ast.literal_eval(array_str)
           arr = np.array(parsed, dtype=np.float32)
          
           # Get center indices
           if arr.ndim == 2:
               center_i = arr.shape[0] // 2
               center_j = arr.shape[1] // 2
               return float(arr[center_i, center_j])
           else:
               # Fallback to mean if unexpected shape
               return float(np.mean(arr))
       except Exception as e:
           print(f"Warning: Failed to extract center value: {e}")
           return np.nan
  
   def load_csv(self, csv_path: str = None) -> pd.DataFrame:
       """
       Load CSV file with dynamic column mapping
      
       Args:
           csv_path: Path to CSV file. If None, uses config.data.csv_path
          
       Returns:
           DataFrame with standardized column names
       """
       if csv_path is None:
           csv_path = self.data_config.csv_path
      
       # Resolve path relative to config file
       csv_path = resolve_path(csv_path, Path(__file__).parent)
      
       if not Path(csv_path).exists():
           raise FileNotFoundError(f"CSV file not found: {csv_path}")
      
       print(f"Loading data from: {csv_path}")
       df = pd.read_csv(csv_path)
      
       print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
      
       # Validate required columns exist
       required_cols = self.data_config.feature_columns
       missing_cols = [col for col in required_cols if col not in df.columns]
       if missing_cols:
           raise ValueError(f"CSV missing required columns: {missing_cols}")
      
       return df
  
   def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
       """
       Preprocess raw data
      
       Args:
           df: Raw DataFrame
          
       Returns:
           Cleaned DataFrame
       """
       df = df.copy()
      
       # Parse timestamp
       time_col = self.data_config.columns.timestamp
       df[time_col] = pd.to_datetime(df[time_col])
      
       # Sort by storm and time
       storm_id_col = self.data_config.columns.storm_id
       df = df.sort_values([storm_id_col, time_col]).reset_index(drop=True)
      
       # Handle missing values in feature columns
       feature_cols = self.data_config.feature_columns
      
       # Fill missing values with forward fill within each storm
       for col in feature_cols:
           if col in df.columns:
               df[col] = df.groupby(storm_id_col)[col].ffill()
               df[col] = df.groupby(storm_id_col)[col].bfill()
      
       # Drop rows with remaining NaNs in feature columns
       initial_len = len(df)
       df = df.dropna(subset=feature_cols)
       dropped = initial_len - len(df)
      
       if dropped > 0:
           print(f"Dropped {dropped} rows with missing feature values")
      
       return df
  
   def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
       """
       Create input sequences and forecast targets
      
       For each valid sequence:
       - Input X: Last 8 timesteps (t-7 to t)
       - Output y: Next 12 timesteps lat/lon offsets (t+1 to t+12)
      
       Args:
           df: Preprocessed DataFrame
          
       Returns:
           X: Input sequences (n_samples, seq_len, n_features)
           y: Target forecasts (n_samples, forecast_steps * 2)  # lat/lon offsets
           metadata: List of dicts with storm_id, start_time for each sample
       """
       sequence_length = self.data_config.sequence_length
       forecast_steps = self.data_config.forecast_steps
       feature_cols = self.data_config.feature_columns
      
       storm_id_col = self.data_config.columns.storm_id
       lat_col = self.data_config.columns.latitude
       lon_col = self.data_config.columns.longitude
       time_col = self.data_config.columns.timestamp
      
       X_list = []
       y_list = []
       metadata_list = []
      
       # Group by storm
       grouped = df.groupby(storm_id_col)
      
       for storm_id, storm_df in grouped:
           storm_df = storm_df.reset_index(drop=True)
           n = len(storm_df)
          
           # Need at least sequence_length + forecast_steps points
           if n < sequence_length + forecast_steps:
               continue
          
           # Create sliding windows
           for i in range(n - sequence_length - forecast_steps + 1):
               # Input sequence
               seq_df = storm_df.iloc[i:i + sequence_length]
               X_seq = seq_df[feature_cols].values
              
               # Forecast targets (lat/lon offsets from last input position)
               last_lat = seq_df[lat_col].iloc[-1]
               last_lon = seq_df[lon_col].iloc[-1]
              
               future_df = storm_df.iloc[i + sequence_length:i + sequence_length + forecast_steps]
              
               # Calculate offsets
               lat_offsets = (future_df[lat_col].values - last_lat).tolist()
               lon_offsets = (future_df[lon_col].values - last_lon).tolist()
              
               # Interleave: [lat0, lon0, lat1, lon1, ...]
               y_seq = []
               for lat_off, lon_off in zip(lat_offsets, lon_offsets):
                   y_seq.extend([lat_off, lon_off])
              
               X_list.append(X_seq)
               y_list.append(y_seq)
              
               metadata_list.append({
                   'storm_id': storm_id,
                   'start_time': seq_df[time_col].iloc[0],
                   'last_time': seq_df[time_col].iloc[-1],
                   'last_lat': last_lat,
                   'last_lon': last_lon
               })
      
       X = np.array(X_list, dtype=np.float32)
       y = np.array(y_list, dtype=np.float32)
      
       print(f"Created {len(X)} sequences from {len(grouped)} storms")
       print(f"X shape: {X.shape}, y shape: {y.shape}")
      
       return X, y, metadata_list
  
   def normalize_data(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray = None, y_val: np.ndarray = None,
                     X_test: np.ndarray = None, y_test: np.ndarray = None
                     ) -> Tuple:
       """
       Normalize data using MinMaxScaler or StandardScaler
      
       Fits scaler on training data only, then transforms all sets.
      
       Args:
           X_train, y_train: Training data
           X_val, y_val: Validation data (optional)
           X_test, y_test: Test data (optional)
          
       Returns:
           Normalized arrays in same order as input
       """
       if not self.data_config.normalize:
           return (X_train, y_train) + ((X_val, y_val) if X_val is not None else ()) + \
                  ((X_test, y_test) if X_test is not None else ())
      
       method = self.data_config.normalization_method.lower()
      
       if method == "minmax":
           self.scaler_X = MinMaxScaler(feature_range=(0, 1))
           self.scaler_y = MinMaxScaler(feature_range=(0, 1))
       elif method == "standard":
           self.scaler_X = StandardScaler()
           self.scaler_y = StandardScaler()
       else:
           raise ValueError(f"Unknown normalization method: {method}")
      
       # Fit on training data
       # X: (n_samples, seq_len, n_features) -> reshape to (n_samples * seq_len, n_features)
       n_train, seq_len, n_features = X_train.shape
       X_train_2d = X_train.reshape(-1, n_features)
      
       self.scaler_X.fit(X_train_2d)
       X_train_norm = self.scaler_X.transform(X_train_2d).reshape(n_train, seq_len, n_features)
      
       # y: (n_samples, n_outputs)
       self.scaler_y.fit(y_train)
       y_train_norm = self.scaler_y.transform(y_train)
      
       results = [X_train_norm, y_train_norm]
      
       # Transform val and test if provided
       if X_val is not None:
           n_val, seq_len, n_features = X_val.shape
           X_val_2d = X_val.reshape(-1, n_features)
           X_val_norm = self.scaler_X.transform(X_val_2d).reshape(n_val, seq_len, n_features)
           y_val_norm = self.scaler_y.transform(y_val)
           results.extend([X_val_norm, y_val_norm])
      
       if X_test is not None:
           n_test, seq_len, n_features = X_test.shape
           X_test_2d = X_test.reshape(-1, n_features)
           X_test_norm = self.scaler_X.transform(X_test_2d).reshape(n_test, seq_len, n_features)
           y_test_norm = self.scaler_y.transform(y_test)
           results.extend([X_test_norm, y_test_norm])
      
       print(f"Data normalized using {method} scaling")
       return tuple(results)
  
   def split_data(self, X: np.ndarray, y: np.ndarray, metadata: List[Dict]
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                            List[Dict], List[Dict], List[Dict]]:
       """
       Split data into train/val/test sets
      
       Args:
           X: Input sequences
           y: Target forecasts
           metadata: Metadata for each sample
          
       Returns:
           X_train, y_train, X_val, y_val, X_test, y_test,
           metadata_train, metadata_val, metadata_test
       """
       train_ratio = self.data_config.train_ratio
       val_ratio = self.data_config.val_ratio
       test_ratio = self.data_config.test_ratio
      
       # Validate ratios
       if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
           raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
      
       n_samples = len(X)
       n_train = int(n_samples * train_ratio)
       n_val = int(n_samples * val_ratio)
      
       # Simple sequential split (can be improved with stratification by storm)
       X_train = X[:n_train]
       y_train = y[:n_train]
       metadata_train = metadata[:n_train]
      
       X_val = X[n_train:n_train + n_val]
       y_val = y[n_train:n_train + n_val]
       metadata_val = metadata[n_train:n_train + n_val]
      
       X_test = X[n_train + n_val:]
       y_test = y[n_train + n_val:]
       metadata_test = metadata[n_train + n_val:]
      
       print(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
      
       # Store metadata for later use
       self.metadata_train = metadata_train
       self.metadata_val = metadata_val
       self.metadata_test = metadata_test
      
       return X_train, y_train, X_val, y_val, X_test, y_test, metadata_train, metadata_val, metadata_test
  
   def load_and_prepare(self, atmospheric_data_dir: str = None
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
       """
       Complete data loading pipeline
      
       Loads IBTrACS track data merged with atmospheric grids, then:
       - Preprocesses data
       - Creates sequences
       - Splits into train/val/test
       - Normalizes
      
       Args:
           atmospheric_data_dir: Directory containing atmospheric CSV files (default: ../output)
          
       Returns:
           X_train, y_train, X_val, y_val, X_test, y_test (all normalized)
           Metadata stored in self.metadata_train, self.metadata_val, self.metadata_test
          
       Example:
           >>> loader = TCDataLoader(config)
           >>> X_train, y_train, X_val, y_val, X_test, y_test = loader.load_and_prepare()
           >>> test_metadata = loader.metadata_test
       """
       print("TC Forecast Data Loading Pipeline")
      
       # Load merged IBTrACS + atmospheric data
       df = self.load_multiple_csvs(atmospheric_data_dir)
      
       df = self.preprocess_data(df)
      
       X, y, metadata = self.create_sequences(df)
      
       result = self.split_data(X, y, metadata)
       X_train, y_train, X_val, y_val, X_test, y_test = result[:6]
      
       X_train, y_train, X_val, y_val, X_test, y_test = self.normalize_data(
           X_train, y_train, X_val, y_val, X_test, y_test
       )
      
       print("Data preparation complete!")
      
       return X_train, y_train, X_val, y_val, X_test, y_test
  
   def inverse_transform_predictions(self, y_pred: np.ndarray, y_true: np.ndarray = None
                                    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
       """
       Inverse transform normalized predictions back to original scale
      
       Args:
           y_pred: Normalized predictions
           y_true: Normalized true values (optional)
          
       Returns:
           Denormalized predictions (and true values if provided)
       """
       if self.scaler_y is None:
           warnings.warn("No scaler found, returning data as-is")
           return (y_pred, y_true) if y_true is not None else (y_pred,)
      
       y_pred_denorm = self.scaler_y.inverse_transform(y_pred)
      
       if y_true is not None:
           y_true_denorm = self.scaler_y.inverse_transform(y_true)
           return y_pred_denorm, y_true_denorm
      
       return y_pred_denorm,



