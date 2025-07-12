"""
Data loading utilities for NVML power and regions data.
"""
import pandas as pd
import numpy as np
from tqdm import tqdm


def load_data(power_file, regions_file):
    """
    Loads power and region data with a progress indicator.
    Optimized for potentially large files by specifying dtypes.
    """
    print(f"Loading power data from {power_file}...")
    power_dtypes = {
        'timestamp_epoch_ns': np.int64,
        'power_watts': np.float32,
        'gpu_id': 'category'
    }
    power_df = pd.read_csv(power_file, dtype=power_dtypes)
    
    print(f"Loading regions data from {regions_file}...")
    regions_dtypes = {
        'start_time_epoch_ns': np.int64,
        'end_time_epoch_ns': np.int64,
        'duration_ns': np.int64,
        'name': 'category'
    }
    regions_df = pd.read_csv(regions_file, dtype=regions_dtypes)

    print("Data loaded.")
    return power_df, regions_df


def normalize_timestamps(power_df, regions_df):
    """Normalizes timestamps to have the same reference point."""
    print("Normalizing timestamps...")
    min_timestamp = min(power_df['timestamp_epoch_ns'].min(),
                       regions_df['start_time_epoch_ns'].min())
    
    power_df['time_seconds'] = (power_df['timestamp_epoch_ns'] - min_timestamp) / 1e9
    regions_df['start_time_seconds'] = (regions_df['start_time_epoch_ns'] - min_timestamp) / 1e9
    regions_df['end_time_seconds'] = (regions_df['end_time_epoch_ns'] - min_timestamp) / 1e9
    regions_df['duration_seconds'] = regions_df['duration_ns'] / 1e9
    print("Timestamps normalized.")
    return power_df, regions_df
