"""
Statistics generation module for power and region analysis.
"""
import numpy as np
from tqdm import tqdm


def generate_statistics(power_df, regions_df):
    """Generates statistics on the power/region correlation."""
    print("\n=== GENERATING STATISTICS ===")
    print(f"Total number of power measurements: {len(power_df)}")
    print(f"Total number of regions: {len(regions_df)}")
    print(f"Number of unique region types: {len(regions_df['name'].unique())}")

    print("\n=== STATISTICS BY REGION ===")
    region_summary = regions_df.groupby('name')['duration_ns'].agg(['count', 'sum', 'mean']).reset_index()
    
    for index, row in tqdm(region_summary.iterrows(), total=len(region_summary), desc="Processing region stats"):
        region = row['name']
        count = row['count']
        total_duration_ms = row['sum'] / 1e6
        avg_duration_ms = row['mean'] / 1e6
        
        print(f"\n{str(region).replace('Kokkos::', '')}:")
        print(f"  - Occurrences: {count}")
        print(f"  - Total Duration: {total_duration_ms:.2f} ms")
        print(f"  - Average Duration: {avg_duration_ms:.2f} ms")
    
    print(f"\n=== ANALYZING COMPUTE KERNEL POWER ===")
    compute_regions = regions_df[regions_df['name'] == 'compute_kernel']
    if not compute_regions.empty:
        print(f"Number of compute kernel executions: {len(compute_regions)}")
        
        total_compute_power_integral = 0
        total_compute_duration = 0
        
        power_times = power_df['time_seconds'].values
        power_watts = power_df['power_watts'].values

        for _, region_row in tqdm(compute_regions.iterrows(), total=len(compute_regions), desc="Calculating compute power"):
            start_time = region_row['start_time_seconds']
            end_time = region_row['end_time_seconds']
            duration = region_row['duration_seconds']
            
            idx_start = np.searchsorted(power_times, start_time, side='left')
            idx_end = np.searchsorted(power_times, end_time, side='right')
            
            if idx_start < idx_end:
                avg_power_in_region = power_watts[idx_start:idx_end].mean()
                total_compute_power_integral += avg_power_in_region * duration
                total_compute_duration += duration
        
        if total_compute_duration > 0:
            avg_compute_power = total_compute_power_integral / total_compute_duration
            print(f"Average power during compute kernels: {avg_compute_power:.2f} W")
        else:
            print("No valid power data found within compute kernels to calculate average power.")
    else:
        print("No 'compute_kernel' regions found.")
    print("Statistics generation complete.")
