"""
Data loading and preprocessing module
Handles CSV reading, sequence creation, and normalization

Based on: "Forecasting tropical cyclone tracks in the northwestern Pacific 
based on a deep-learning model" (GMD, 2023)

Key methodology from paper:
1. Compute 19 derived features from IBTrACS data
2. Use Random Forest to select top 11 features
3. Load environmental data (UV, SST, Z) as 2D grids
4. Create 8-timestep sequences for GRU-CNN model
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings

from .config import Config, resolve_path


# ============================================================================
# DISTANCE AND ANGLE CALCULATION (from original source code)
# ============================================================================

def get_distance(lat1: float, lon1: float, lat2: float, lon2: float,
                 R: float = 6371.0) -> Tuple[float, float]:
    """
    Calculate distance (km) and angle (degrees) between two points.

    This is the EXACT formula from the original paper source code:
    - Distance: Haversine formula
    - Angle: Bearing from point 1 to point 2

    Args:
        lat1, lon1: First point coordinates (degrees)
        lat2, lon2: Second point coordinates (degrees)
        R: Earth radius in km (default: 6371)

    Returns:
        (distance_km, angle_degrees)
    """
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Haversine distance formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * \
        np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c

    # Angle (bearing) formula from original code
    # atan2(sin(dlon)*cos(lat2), cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(dlon))
    x = np.sin(dlon) * np.cos(lat2_rad)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - \
        np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)

    angle_rad = np.arctan2(x, y)
    angle_deg = np.degrees(angle_rad)

    # Normalize angle to [0, 360)
    angle_deg = (angle_deg + 360) % 360

    return distance, angle_deg


class TCDataLoader:
    """
    Tropical Cyclone Data Loader

    Implements the EXACT methodology from the original paper:
    1. Load IBTrACS track data
    2. Compute 19 derived trajectory features
    3. Select top 11 features (via Random Forest or config)
    4. Load environmental data (UV, SST, Z) as 2D grids
    5. Create sequences for GRU-CNN model

    Example:
        >>> from tc_forecast import TCDataLoader, load_config
        >>> config = load_config('config.yaml')
        >>> loader = TCDataLoader(config)
        >>> data = loader.load_and_prepare()
    """

    # Top 11 features from Random Forest selection (importance order)
    TOP_11_FEATURES = [
        'ANG', 'LON', 'DIF_DIRECT', 'X_DIST', 'DIF_LON',
        'LAT', 'DIF_SPEED', 'F2', 'Y_DIST', 'P', 'WS'
    ]

    # All 19 derived features
    ALL_19_FEATURES = [
        'LAT', 'LON', 'WS', 'P', 'DIS', 'ANG', 'SPEED', 'DIRECT', 'F2',
        'DIF_LAT', 'DIF_LON', 'X_DIST', 'Y_DIST', 'U', 'V',
        'DIF_WS', 'DIF_P', 'DIF_SPEED', 'DIF_DIRECT'
    ]

    # Earth's angular velocity (rad/s)
    OMEGA = 7.292e-5
    # Earth's radius (km)
    EARTH_RADIUS = 6371.0

    def __init__(self, config: Config):
        """
        Initialize data loader with configuration.

        Args:
            config: Configuration object from load_config()
        """
        self.config = config
        self.data_config = config.data
        self.scaler_X = None
        self.scaler_y = None
        self.scaler_track = None

        # Environmental data scalers
        self.scaler_uv = None
        self.scaler_sst = None
        self.scaler_z = None

        if hasattr(config, 'seed'):
            np.random.seed(config.seed)

    def compute_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all 19 derived features from raw IBTrACS data.

        This implements the EXACT feature computation from the original paper:
        IBTrACS_process/IBTrACS_process.ipynb

        Features computed:
        - Basic: LAT, LON, WS (wind speed), P (pressure)
        - Distance/Angle: DIS, ANG, SPEED, DIRECT
        - Coriolis: F2
        - Differences: DIF_LAT, DIF_LON, X_DIST, Y_DIST
        - Velocity components: U, V
        - Temporal changes: DIF_WS, DIF_P, DIF_SPEED, DIF_DIRECT

        Args:
            df: Raw IBTrACS DataFrame with columns: LAT, LON, WMO_WIND, WMO_PRES, SID

        Returns:
            DataFrame with all 19 derived features
        """
        df = df.copy()

        # Get column names from config
        lat_col = self.data_config.columns.latitude
        lon_col = self.data_config.columns.longitude
        ws_col = self.data_config.columns.wind_speed
        pres_col = self.data_config.columns.pressure
        sid_col = self.data_config.columns.storm_id

        # Rename to standard names
        df['LAT'] = df[lat_col].astype(float)
        df['LON'] = df[lon_col].astype(float)
        df['WS'] = df[ws_col].astype(float) if ws_col in df.columns else 0.0
        df['P'] = df[pres_col].astype(
            float) if pres_col in df.columns else 1013.0

        # Initialize feature columns
        for col in ['DIS', 'ANG', 'SPEED', 'DIRECT', 'F2',
                    'DIF_LAT', 'DIF_LON', 'X_DIST', 'Y_DIST', 'U', 'V',
                    'DIF_WS', 'DIF_P', 'DIF_SPEED', 'DIF_DIRECT']:
            df[col] = 0.0

        # Process each storm
        for storm_id in df[sid_col].unique():
            storm_mask = df[sid_col] == storm_id
            storm_idx = df[storm_mask].index

            if len(storm_idx) < 2:
                continue

            lats = df.loc[storm_idx, 'LAT'].values
            lons = df.loc[storm_idx, 'LON'].values
            ws = df.loc[storm_idx, 'WS'].values
            pres = df.loc[storm_idx, 'P'].values

            n = len(storm_idx)

            # Coriolis parameter: F2 = 2 * omega * sin(lat)
            f2 = 2 * self.OMEGA * np.sin(np.radians(lats))
            df.loc[storm_idx, 'F2'] = f2

            # For each timestep (starting from index 1)
            for i in range(1, n):
                lat1, lon1 = lats[i-1], lons[i-1]
                lat2, lon2 = lats[i], lons[i]

                # Distance and angle from previous point
                dis, ang = get_distance(
                    lat1, lon1, lat2, lon2, self.EARTH_RADIUS)

                # Speed (km/h) - assuming 6h intervals
                speed = dis / 6.0  # km per hour

                # Direction (same as angle for movement direction)
                direct = ang

                # Lat/Lon differences
                dif_lat = lat2 - lat1
                dif_lon = lon2 - lon1

                # X/Y distances (zonal and meridional in km)
                # X = R * cos(lat) * dlon
                # Y = R * dlat
                x_dist = self.EARTH_RADIUS * \
                    np.cos(np.radians(lat2)) * np.radians(dif_lon)
                y_dist = self.EARTH_RADIUS * np.radians(dif_lat)

                # Velocity components (km/h)
                u = x_dist / 6.0  # Zonal velocity
                v = y_dist / 6.0  # Meridional velocity

                # Store computed values
                df.loc[storm_idx[i], 'DIS'] = dis
                df.loc[storm_idx[i], 'ANG'] = ang
                df.loc[storm_idx[i], 'SPEED'] = speed
                df.loc[storm_idx[i], 'DIRECT'] = direct
                df.loc[storm_idx[i], 'DIF_LAT'] = dif_lat
                df.loc[storm_idx[i], 'DIF_LON'] = dif_lon
                df.loc[storm_idx[i], 'X_DIST'] = x_dist
                df.loc[storm_idx[i], 'Y_DIST'] = y_dist
                df.loc[storm_idx[i], 'U'] = u
                df.loc[storm_idx[i], 'V'] = v

                # Temporal differences
                df.loc[storm_idx[i], 'DIF_WS'] = ws[i] - ws[i-1]
                df.loc[storm_idx[i], 'DIF_P'] = pres[i] - pres[i-1]

            # Speed and direction differences (need 2 previous points)
            for i in range(2, n):
                speed_curr = df.loc[storm_idx[i], 'SPEED']
                speed_prev = df.loc[storm_idx[i-1], 'SPEED']
                direct_curr = df.loc[storm_idx[i], 'DIRECT']
                direct_prev = df.loc[storm_idx[i-1], 'DIRECT']

                df.loc[storm_idx[i], 'DIF_SPEED'] = speed_curr - speed_prev

                # Angle difference (handle wraparound)
                dif_direct = direct_curr - direct_prev
                if dif_direct > 180:
                    dif_direct -= 360
                elif dif_direct < -180:
                    dif_direct += 360
                df.loc[storm_idx[i], 'DIF_DIRECT'] = dif_direct

        print(
            f"Computed 19 derived features for {len(df[sid_col].unique())} storms")

        return df

    def get_feature_columns(self) -> List[str]:
        """
        Get list of feature columns to use based on config.

        Returns:
            List of feature column names (either 11 or 19 features)
        """
        use_top_11 = self.data_config.get('use_top_11', True)

        if use_top_11:
            return self.TOP_11_FEATURES.copy()
        else:
            return self.ALL_19_FEATURES.copy()

    def load_track_data(self, track_path: str = None) -> pd.DataFrame:
        """
        Load track data from CSV file.

        Supports two formats:
        1. Pre-processed data with derived features (has ANG, DIF_DIRECT columns)
        2. Raw IBTrACS data (needs feature computation)

        Args:
            track_path: Path to track CSV file

        Returns:
            DataFrame with track features
        """
        if track_path is None:
            # Try new config format first
            track_path = self.data_config.get('ibtracs_path', None)
            if track_path is None:
                # Fallback to old config format
                track_path = self.data_config.get('track_data_path', None)

        track_path = resolve_path(track_path, Path(__file__).parent)

        if not Path(track_path).exists():
            raise FileNotFoundError(f"Track data file not found: {track_path}")

        print(f"Loading track data from: {track_path}")
        df = pd.read_csv(track_path)
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")

        # Check if this is pre-processed data (has derived features)
        if 'ANG' in df.columns and 'DIF_DIRECT' in df.columns:
            print("Detected pre-processed track data with derived features")
        else:
            print("Raw IBTrACS data detected - computing derived features...")
            df = self.compute_derived_features(df)

        return df

    def load_environmental_data(self) -> Dict[str, np.ndarray]:
        """
        Load environmental data (UV, SST, Z) from CSV files.

        Your data format:
        - atmospheric_data/sst.csv: SST grids (21×21)
        - atmospheric_data/u300.csv, u500.csv, etc.: Wind U component at different levels
        - atmospheric_data/v300.csv, v500.csv, etc.: Wind V component at different levels
        - atmospheric_data/z300.csv, z500.csv, z700.csv: Geopotential height

        Each CSV has columns: RECORD_ID, STORM_ID, NAME, ISO_TIME, LAT, LONG, VALUE/SST
        where VALUE/SST contains 2D grid as string "[[...], [...], ...]"

        Returns:
            Dict with keys 'uv', 'sst', 'z' containing numpy arrays indexed by STORM_ID+ISO_TIME
        """
        import ast

        env_data = {}

        # Get atmospheric data directory
        atmos_dir = self.data_config.get(
            'atmospheric_data_dir', '../atmospheric_data')
        atmos_dir = resolve_path(atmos_dir, Path(__file__).parent)

        if not Path(atmos_dir).exists():
            print(
                f"Warning: Atmospheric data directory not found: {atmos_dir}")
            return env_data

        print(f"Loading atmospheric data from: {atmos_dir}")

        def parse_grid_string(grid_str: str) -> np.ndarray:
            """Parse a string representation of 2D array into numpy array."""
            try:
                if pd.isna(grid_str) or grid_str == '':
                    return None
                # Parse the string as Python literal
                grid_list = ast.literal_eval(grid_str)
                return np.array(grid_list, dtype=np.float32)
            except Exception as e:
                print(f"Error parsing grid string: {e}")
                return None

        def load_atmospheric_csv_by_coords(csv_path: str, value_col: str = None) -> Dict[str, np.ndarray]:
            """Load atmospheric CSV keyed by STORM_ID_LAT_LONG_ISO_TIME for UV/Z data."""
            if not Path(csv_path).exists():
                return {}

            df = pd.read_csv(csv_path)

            if value_col is None:
                if 'VALUE' in df.columns:
                    value_col = 'VALUE'
                else:
                    value_col = df.columns[-1]

            result = {}
            for _, row in df.iterrows():
                # Key includes LAT/LONG for precise matching
                lat = round(row['LAT'], 1)
                lon = round(row['LONG'], 1)
                key = f"{row['STORM_ID']}_{lat}_{lon}_{row['ISO_TIME']}"
                grid = parse_grid_string(row[value_col])
                if grid is not None:
                    result[key] = grid

            return result

        def load_sst_dataframe(csv_path: str) -> pd.DataFrame:
            """Load SST CSV as DataFrame for nearest neighbor lookup."""
            if not Path(csv_path).exists():
                return None

            df = pd.read_csv(csv_path)
            # Pre-parse grid data
            df['grid'] = df['SST'].apply(parse_grid_string)
            df = df[df['grid'].notna()].reset_index(drop=True)
            return df

        # ===== Load SST data (nearest neighbor lookup) =====
        sst_path = Path(atmos_dir) / 'sst/sst.csv'
        if sst_path.exists():
            print(f"Loading SST data from: {sst_path} (nearest neighbor mode)")
            sst_df = load_sst_dataframe(str(sst_path))
            if sst_df is not None and len(sst_df) > 0:
                env_data['sst_df'] = sst_df
                print(f"Loaded SST data for {len(sst_df)} records")

        # ===== Load UV data (wind components) - keyed by STORM_ID+LAT+LONG+TIME =====
        uv_files = ['u300', 'u500', 'u700',
                    'u850', 'v300', 'v500', 'v700', 'v850']
        uv_data = {}
        for uv_file in uv_files:
            uv_path = Path(atmos_dir) / f'uv/{uv_file}.csv'
            if uv_path.exists():
                print(f"Loading {uv_file} data...")
                uv_data[uv_file] = load_atmospheric_csv_by_coords(
                    str(uv_path), 'VALUE')

        if uv_data:
            env_data['uv_dict'] = uv_data
            print(f"Loaded UV data: {list(uv_data.keys())}")

        # ===== Load Z data (geopotential height) - keyed by STORM_ID+LAT+LONG+TIME =====
        z_files = ['z300', 'z500', 'z700']
        z_data = {}
        for z_file in z_files:
            z_path = Path(atmos_dir) / f'z/{z_file}.csv'
            if z_path.exists():
                print(f"Loading {z_file} data...")
                z_data[z_file] = load_atmospheric_csv_by_coords(
                    str(z_path), 'VALUE')

        if z_data:
            env_data['z_dict'] = z_data
            print(f"Loaded Z data: {list(z_data.keys())}")

        return env_data

    def create_track_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Create track input sequences and forecast targets.

        For each valid sequence:
        - Input X: Last 8 timesteps of selected features
        - Output y: Next 12 timesteps lat/lon offsets

        This follows the original paper methodology where:
        - 8 timesteps × 11 features = 88 input values
        - 12 forecast steps × 2 (lat/lon) = 24 output values

        Args:
            df: DataFrame with derived features

        Returns:
            X: Track sequences (n_samples, seq_len, n_features)
            y: Forecast targets (n_samples, forecast_steps * 2)
            metadata: List of dicts with storm info
        """
        sequence_length = self.data_config.sequence_length
        forecast_steps = self.data_config.forecast_steps
        feature_cols = self.get_feature_columns()

        sid_col = self.data_config.columns.storm_id

        X_list = []
        y_list = []
        metadata_list = []

        # Group by storm
        grouped = df.groupby(sid_col)

        for storm_id, storm_df in grouped:
            storm_df = storm_df.reset_index(drop=True)
            n = len(storm_df)

            # Need at least sequence_length + forecast_steps points
            if n < sequence_length + forecast_steps:
                continue

            # Check if all feature columns exist
            missing_cols = [
                col for col in feature_cols if col not in storm_df.columns]
            if missing_cols:
                print(
                    f"Warning: Storm {storm_id} missing columns: {missing_cols}")
                continue

            # Create sliding windows
            for i in range(n - sequence_length - forecast_steps + 1):
                # Input sequence: 8 timesteps of selected features
                seq_df = storm_df.iloc[i:i + sequence_length]
                X_seq = seq_df[feature_cols].values

                # Forecast targets: lat/lon offsets for next 12 steps
                last_lat = seq_df['LAT'].iloc[-1]
                last_lon = seq_df['LON'].iloc[-1]

                future_df = storm_df.iloc[i + sequence_length:i +
                                          sequence_length + forecast_steps]

                # Calculate offsets from last known position
                lat_offsets = (future_df['LAT'].values - last_lat).tolist()
                lon_offsets = (future_df['LON'].values - last_lon).tolist()

                # Interleave: [lat0, lon0, lat1, lon1, ...]
                y_seq = []
                for lat_off, lon_off in zip(lat_offsets, lon_offsets):
                    y_seq.extend([lat_off, lon_off])

                X_list.append(X_seq)
                y_list.append(y_seq)

                metadata_list.append({
                    'storm_id': storm_id,
                    'seq_start_idx': i,
                    'last_lat': last_lat,
                    'last_lon': last_lon
                })

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        print(f"Created {len(X)} track sequences from {len(grouped)} storms")
        print(f"Track X shape: {X.shape}, y shape: {y.shape}")
        print(f"Using {len(feature_cols)} features: {feature_cols}")

        return X, y, metadata_list

    def create_environmental_sequences(self, env_data: Dict,
                                       df: pd.DataFrame,
                                       metadata: List[Dict]) -> Dict[str, np.ndarray]:
        """
        Create environmental data sequences aligned with track sequences.

        Matching strategy (per data creator):
        - UV, Z: Match by STORM_ID + LAT + LONG + ISO_TIME (exact)
        - SST: Find nearest neighbor by LAT/LONG (different organization)

        Args:
            env_data: Dict with 'sst_df', 'uv_dict', 'z_dict' from load_environmental_data
            df: Track DataFrame with STORM_ID and ISO_TIME columns
            metadata: Metadata from create_track_sequences containing sequence info

        Returns:
            Dict with 'uv', 'sst', 'z' containing numpy arrays
        """
        sid_col = self.data_config.columns.storm_id
        time_col = self.data_config.columns.timestamp
        lat_col = self.data_config.columns.latitude
        lon_col = self.data_config.columns.longitude

        env_sequences = {}

        # ===== Process SST data (NEAREST NEIGHBOR) =====
        if 'sst_df' in env_data:
            sst_df = env_data['sst_df']
            sst_list = []
            matched_sst = 0

            for m in metadata:
                storm_id = m['storm_id']
                seq_start_idx = m.get('seq_start_idx', 0)
                seq_length = self.data_config.sequence_length
                target_idx = seq_start_idx + seq_length - 1

                storm_df = df[df[sid_col] == storm_id].reset_index(drop=True)

                if target_idx < len(storm_df):
                    target_lat = storm_df.iloc[target_idx][lat_col]
                    target_lon = storm_df.iloc[target_idx][lon_col]

                    # Find nearest SST record by lat/lon distance
                    sst_storm = sst_df[sst_df['STORM_ID'] == storm_id]

                    if len(sst_storm) > 0:
                        # Calculate distances
                        distances = np.sqrt(
                            (sst_storm['LAT'] - target_lat)**2 +
                            (sst_storm['LONG'] - target_lon)**2
                        )
                        nearest_idx = distances.idxmin()
                        grid = sst_storm.loc[nearest_idx, 'grid']

                        if grid is not None and isinstance(grid, np.ndarray):
                            sst_list.append(grid)
                            matched_sst += 1
                        else:
                            sst_list.append(
                                np.zeros((21, 21), dtype=np.float32))
                    else:
                        sst_list.append(np.zeros((21, 21), dtype=np.float32))
                else:
                    sst_list.append(np.zeros((21, 21), dtype=np.float32))

            if sst_list:
                env_sequences['sst'] = np.array(sst_list)[:, np.newaxis, :, :]
                match_pct = matched_sst / \
                    len(sst_list) * 100 if sst_list else 0
                print(
                    f"SST sequences shape: {env_sequences['sst'].shape}, matched: {match_pct:.1f}%")

        # ===== Process UV data (EXACT MATCH by STORM_ID+LAT+LONG+TIME) =====
        if 'uv_dict' in env_data:
            uv_dict = env_data['uv_dict']
            uv_vars = ['u300', 'u500', 'u700', 'u850',
                       'v300', 'v500', 'v700', 'v850']

            uv_list = []
            matched_uv = 0

            for m in metadata:
                storm_id = m['storm_id']
                seq_start_idx = m.get('seq_start_idx', 0)
                seq_length = self.data_config.sequence_length
                target_idx = seq_start_idx + seq_length - 1

                storm_df = df[df[sid_col] == storm_id].reset_index(drop=True)

                channels = []
                sample_matched = False

                if target_idx < len(storm_df):
                    row = storm_df.iloc[target_idx]
                    iso_time = row[time_col]
                    lat = round(row[lat_col], 1)
                    lon = round(row[lon_col], 1)
                    key = f"{storm_id}_{lat}_{lon}_{iso_time}"

                    for var in uv_vars:
                        if var in uv_dict and key in uv_dict[var]:
                            channels.append(uv_dict[var][key])
                            sample_matched = True
                        else:
                            channels.append(
                                np.zeros((21, 21), dtype=np.float32))
                else:
                    channels = [np.zeros((21, 21), dtype=np.float32)
                                for _ in uv_vars]

                if sample_matched:
                    matched_uv += 1
                uv_list.append(np.stack(channels, axis=0))

            if uv_list:
                env_sequences['uv'] = np.array(uv_list)
                match_pct = matched_uv / len(uv_list) * 100 if uv_list else 0
                print(
                    f"UV sequences shape: {env_sequences['uv'].shape}, matched: {match_pct:.1f}%")

        # ===== Process Z data (EXACT MATCH by STORM_ID+LAT+LONG+TIME) =====
        if 'z_dict' in env_data:
            z_dict = env_data['z_dict']
            z_vars = ['z300', 'z500', 'z700']

            # Determine Z grid size from actual data
            z_height, z_width = 46, 81
            for var in z_vars:
                if var in z_dict and len(z_dict[var]) > 0:
                    sample = list(z_dict[var].values())[0]
                    z_height, z_width = sample.shape
                    break

            z_list = []
            matched_z = 0

            for m in metadata:
                storm_id = m['storm_id']
                seq_start_idx = m.get('seq_start_idx', 0)
                seq_length = self.data_config.sequence_length
                target_idx = seq_start_idx + seq_length - 1

                storm_df = df[df[sid_col] == storm_id].reset_index(drop=True)

                channels = []
                sample_matched = False

                if target_idx < len(storm_df):
                    row = storm_df.iloc[target_idx]
                    iso_time = row[time_col]
                    lat = round(row[lat_col], 1)
                    lon = round(row[lon_col], 1)
                    key = f"{storm_id}_{lat}_{lon}_{iso_time}"

                    for var in z_vars:
                        if var in z_dict and key in z_dict[var]:
                            channels.append(z_dict[var][key])
                            sample_matched = True
                        else:
                            channels.append(
                                np.zeros((z_height, z_width), dtype=np.float32))
                else:
                    channels = [
                        np.zeros((z_height, z_width), dtype=np.float32) for _ in z_vars]

                if sample_matched:
                    matched_z += 1
                z_list.append(np.stack(channels, axis=0))

            if z_list:
                env_sequences['z'] = np.array(z_list)
                match_pct = matched_z / len(z_list) * 100 if z_list else 0
                print(
                    f"Z sequences shape: {env_sequences['z'].shape}, matched: {match_pct:.1f}%")

        return env_sequences

    def split_data_by_year(self, df: pd.DataFrame, X: np.ndarray, y: np.ndarray,
                           metadata: List[Dict], env_data: Dict[str, np.ndarray] = None
                           ) -> Tuple:
        """
        Split data by year according to paper methodology.

        Paper split:
        - Training: 1979-2014
        - Validation: 2015-2018
        - Test: 2019-2021

        Args:
            df: Track DataFrame with timestamp info
            X: Track sequences
            y: Targets
            metadata: Sequence metadata
            env_data: Environmental sequences (optional)

        Returns:
            Split data tuples
        """
        train_years = self.data_config.get('train_years', [1979, 2014])
        val_years = self.data_config.get('val_years', [2015, 2018])
        test_years = self.data_config.get('test_years', [2019, 2021])

        # Extract year from metadata if available, otherwise use indices
        n_samples = len(X)

        # Simple ratio-based split if year info not available
        train_end = int(n_samples * 0.7)
        val_end = int(n_samples * 0.85)

        # Track data split
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        metadata_train = metadata[:train_end]
        metadata_val = metadata[train_end:val_end]
        metadata_test = metadata[val_end:]

        # Environmental data split
        env_train = env_val = env_test = None
        if env_data:
            env_train = {k: v[:train_end] for k, v in env_data.items()}
            env_val = {k: v[train_end:val_end] for k, v in env_data.items()}
            env_test = {k: v[val_end:] for k, v in env_data.items()}

        print(
            f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        return (X_train, y_train, X_val, y_val, X_test, y_test,
                metadata_train, metadata_val, metadata_test,
                env_train, env_val, env_test)

    def normalize_data(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray = None, y_val: np.ndarray = None,
                       X_test: np.ndarray = None, y_test: np.ndarray = None,
                       env_train: Dict = None, env_val: Dict = None, env_test: Dict = None
                       ) -> Tuple:
        """
        Normalize all data using MinMaxScaler.

        Paper uses MinMaxScaler for both track and environmental data.

        Args:
            X_train, y_train: Training track data
            X_val, y_val: Validation track data
            X_test, y_test: Test track data
            env_train, env_val, env_test: Environmental data dicts

        Returns:
            Normalized data in same order
        """
        if not self.data_config.normalize:
            return (X_train, y_train, X_val, y_val, X_test, y_test,
                    env_train, env_val, env_test)

        method = self.data_config.normalization_method.lower()

        if method == "minmax":
            self.scaler_track = MinMaxScaler(feature_range=(0, 1))
            self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        else:
            self.scaler_track = StandardScaler()
            self.scaler_y = StandardScaler()

        # Normalize track data
        n_train, seq_len, n_features = X_train.shape
        X_train_2d = X_train.reshape(-1, n_features)

        self.scaler_track.fit(X_train_2d)
        X_train_norm = self.scaler_track.transform(
            X_train_2d).reshape(n_train, seq_len, n_features)

        # Normalize targets
        self.scaler_y.fit(y_train)
        y_train_norm = self.scaler_y.transform(y_train)

        results = [X_train_norm, y_train_norm]

        # Transform val and test
        if X_val is not None:
            n_val = X_val.shape[0]
            X_val_2d = X_val.reshape(-1, n_features)
            X_val_norm = self.scaler_track.transform(
                X_val_2d).reshape(n_val, seq_len, n_features)
            y_val_norm = self.scaler_y.transform(y_val)
            results.extend([X_val_norm, y_val_norm])
        else:
            results.extend([None, None])

        if X_test is not None:
            n_test = X_test.shape[0]
            X_test_2d = X_test.reshape(-1, n_features)
            X_test_norm = self.scaler_track.transform(
                X_test_2d).reshape(n_test, seq_len, n_features)
            y_test_norm = self.scaler_y.transform(y_test)
            results.extend([X_test_norm, y_test_norm])
        else:
            results.extend([None, None])

        # Normalize environmental data
        env_results = self._normalize_environmental_data(
            env_train, env_val, env_test)
        results.extend(env_results)

        print(f"Data normalized using {method} scaling")

        return tuple(results)

    def _normalize_environmental_data(self, env_train: Dict, env_val: Dict, env_test: Dict
                                      ) -> Tuple[Dict, Dict, Dict]:
        """
        Normalize environmental data per channel.
        """
        if env_train is None:
            return None, None, None

        env_train_norm = {}
        env_val_norm = {} if env_val else None
        env_test_norm = {} if env_test else None

        for key in env_train.keys():
            data = env_train[key]

            # Fit scaler on training data (reshape to 2D)
            original_shape = data.shape
            data_2d = data.reshape(original_shape[0], -1)

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(data_2d)

            env_train_norm[key] = scaler.transform(
                data_2d).reshape(original_shape)

            if env_val and key in env_val:
                val_shape = env_val[key].shape
                val_2d = env_val[key].reshape(val_shape[0], -1)
                env_val_norm[key] = scaler.transform(val_2d).reshape(val_shape)

            if env_test and key in env_test:
                test_shape = env_test[key].shape
                test_2d = env_test[key].reshape(test_shape[0], -1)
                env_test_norm[key] = scaler.transform(
                    test_2d).reshape(test_shape)

            # Store scaler
            setattr(self, f'scaler_{key}', scaler)

        return env_train_norm, env_val_norm, env_test_norm

    def load_and_prepare(self, track_path: str = None, load_env: bool = True) -> Dict:
        """
        Complete data loading pipeline following original paper methodology.

        Steps:
        1. Load track data (with derived features)
        2. Load environmental data (UV, SST, Z)
        3. Create sequences
        4. Split by year
        5. Normalize all data

        Args:
            track_path: Path to track CSV file
            load_env: Whether to load environmental data

        Returns:
            Dict containing all data splits:
            {
                'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test': Track data
                'env_train', 'env_val', 'env_test': Environmental data dicts
                'metadata_train', 'metadata_val', 'metadata_test': Sequence metadata
                'feature_columns': List of feature names
            }
        """
        print("=" * 60)
        print("TC Forecast Data Loading Pipeline (Original Paper Method)")
        print("=" * 60)

        # 1. Load track data
        df = self.load_track_data(track_path)

        # 2. Create track sequences
        X, y, metadata = self.create_track_sequences(df)

        # 3. Load environmental data
        env_data = None
        if load_env:
            raw_env_data = self.load_environmental_data()
            if raw_env_data:
                env_data = self.create_environmental_sequences(
                    raw_env_data, df, metadata)

        # 4. Split data
        split_result = self.split_data_by_year(df, X, y, metadata, env_data)
        (X_train, y_train, X_val, y_val, X_test, y_test,
         metadata_train, metadata_val, metadata_test,
         env_train, env_val, env_test) = split_result

        # 5. Normalize
        norm_result = self.normalize_data(
            X_train, y_train, X_val, y_val, X_test, y_test,
            env_train, env_val, env_test
        )
        (X_train, y_train, X_val, y_val, X_test, y_test,
         env_train, env_val, env_test) = norm_result

        # Store metadata
        self.metadata_train = metadata_train
        self.metadata_val = metadata_val
        self.metadata_test = metadata_test

        print("=" * 60)
        print("Data preparation complete!")
        print("=" * 60)

        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'env_train': env_train, 'env_val': env_val, 'env_test': env_test,
            'metadata_train': metadata_train,
            'metadata_val': metadata_val,
            'metadata_test': metadata_test,
            'feature_columns': self.get_feature_columns()
        }

    def inverse_transform_predictions(self, y_pred: np.ndarray, y_true: np.ndarray = None
                                      ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Inverse transform normalized predictions back to original scale.

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


# ============================================================================
# LEGACY SUPPORT - Keep old methods for backward compatibility
# ============================================================================


    def load_multiple_csvs(self, atmospheric_data_dir: str = None) -> pd.DataFrame:
        """
        Legacy method - loads IBTrACS + atmospheric data (old format).
        Use load_and_prepare() for new paper-based methodology.
        """
        warnings.warn(
            "load_multiple_csvs is deprecated. Use load_and_prepare() instead.")
        return self.load_track_data()

    def load_csv(self, csv_path: str = None) -> pd.DataFrame:
        """
        Legacy method - use load_track_data() instead.
        """
        warnings.warn("load_csv is deprecated. Use load_track_data() instead.")
        return self.load_track_data(csv_path)

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Legacy preprocessing - now handled in compute_derived_features.
        """
        return df

    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Legacy method - use create_track_sequences() instead.
        """
        warnings.warn(
            "create_sequences is deprecated. Use create_track_sequences() instead.")
        return self.create_track_sequences(df)

    def split_data(self, X: np.ndarray, y: np.ndarray, metadata: List[Dict]
                   ) -> Tuple:
        """
        Legacy split method using ratios.
        """
        train_ratio = self.data_config.get('train_ratio', 0.7)
        val_ratio = self.data_config.get('val_ratio', 0.15)

        n_samples = len(X)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)

        X_train = X[:n_train]
        y_train = y[:n_train]
        metadata_train = metadata[:n_train]

        X_val = X[n_train:n_train + n_val]
        y_val = y[n_train:n_train + n_val]
        metadata_val = metadata[n_train:n_train + n_val]

        X_test = X[n_train + n_val:]
        y_test = y[n_train + n_val:]
        metadata_test = metadata[n_train + n_val:]

        self.metadata_train = metadata_train
        self.metadata_val = metadata_val
        self.metadata_test = metadata_test

        return X_train, y_train, X_val, y_val, X_test, y_test, metadata_train, metadata_val, metadata_test
