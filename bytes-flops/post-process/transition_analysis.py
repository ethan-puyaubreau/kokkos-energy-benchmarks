"""
Power transition latency analysis module.
"""
import pandas as pd
import numpy as np
from tqdm import tqdm


def compute_power_transition_latency(power_df, regions_df, threshold_ratio=0.9):
    """
    Measures the latency of the GPU power response at each region transition.
    For each region end -> start of next, finds the delay until power crosses a 
    threshold (threshold_ratio) of the new stable power.
    
    This function calculates the average power for the *specific instance* of the
    current and next region to determine the base and target power levels for the transition.
    This approach aims to provide more accurate transition latency for individual events,
    especially for short regions where a global average might not be representative.
    """
    times = power_df['time_seconds'].values
    watts = power_df['power_watts'].values
    
    # Filter out regions that are containers or too short to represent stable power states.
    # 'global_repetition' is typically a parent region and does not represent a direct
    # power state that transitions from/to. 'inter_repetition_wait' can be very short,
    # making power stabilization unlikely within its duration, and it's also a "filler" region.
    # For very short benchmarks, we'll be more permissive with region filtering
    min_region_duration_ms = 1.0  # Minimum 1ms duration for regions to be considered
    filtered_regions = regions_df[
        (~regions_df['name'].isin(['global_repetition'])) &  # Removed inter_repetition_wait from exclusion
        (regions_df['duration_ns'] >= min_region_duration_ms * 1e6)  # Convert ms to ns
    ].copy()

    # Sort by start time for sequential transitions
    transitions = filtered_regions.sort_values('start_time_seconds').reset_index(drop=True)
    
    # Debug information
    print(f"Total regions before filtering: {len(regions_df)}")
    print(f"Regions after filtering: {len(filtered_regions)}")
    print(f"Available region types: {regions_df['name'].unique()}")
    print(f"Region durations (ms): min={regions_df['duration_ns'].min()/1e6:.3f}, max={regions_df['duration_ns'].max()/1e6:.3f}, mean={regions_df['duration_ns'].mean()/1e6:.3f}")
    
    if len(transitions) < 2:
        print(f"Not enough transitions to analyze (need at least 2, found {len(transitions)})")
        return pd.DataFrame()
    
    transition_results = []
    
    # Helper function to get average power within a segment of a region,
    # intended to capture "stable" power levels at the boundaries.
    def get_stable_power_in_region_segment(region_start_t, region_end_t, is_from_region=True, stable_segment_fraction=0.1, min_stable_duration_s=0.001):
        """
        Calculates average power in a stable segment of a region.
        For 'from_region', it looks at the end of the region.
        For 'to_region', it looks at the beginning of the region.
        Uses a fraction of the region's duration or a minimum duration, whichever is larger,
        to define the stable segment, but never exceeds 50% of the region duration.
        """
        duration = region_end_t - region_start_t
        
        if duration <= 0:  # Handle cases of extremely short or zero duration regions
            return np.nan
        
        # For very short regions, use a larger fraction but cap at 50% of duration
        # For longer regions, use the specified fraction
        max_fraction = 0.5  # Never use more than 50% of the region
        effective_duration = min(
            max(duration * stable_segment_fraction, min_stable_duration_s),
            duration * max_fraction
        )

        if is_from_region: # For base power (from_region), sample from the end of the region
            sample_start_t = max(region_start_t, region_end_t - effective_duration)
            sample_end_t = region_end_t
        else: # For target power (to_region), sample from the beginning of the region
            sample_start_t = region_start_t
            sample_end_t = min(region_end_t, region_start_t + effective_duration)

        idx_start = np.searchsorted(times, sample_start_t, side='left')
        idx_end = np.searchsorted(times, sample_end_t, side='right')
        
        if idx_start < idx_end:
            return watts[idx_start:idx_end].mean()
        return np.nan

    for i in tqdm(range(len(transitions)-1), desc="Calculating power transition latency"):
        curr = transitions.loc[i]
        nxt = transitions.loc[i+1]
        
        t_curr_end = curr['end_time_seconds']
        t_nxt_start = nxt['start_time_seconds']

        # Calculate base power (average power within the stable segment of the current region instance)
        base_power = get_stable_power_in_region_segment(curr['start_time_seconds'], t_curr_end, is_from_region=True)
        
        # Calculate target power (average power within the stable segment of the next region instance)
        target_power = get_stable_power_in_region_segment(t_nxt_start, nxt['end_time_seconds'], is_from_region=False)

        # Skip if base or target power cannot be determined, or if there's no significant change
        if np.isnan(base_power) or np.isnan(target_power):
            print(f"  Skipping transition {curr['name']} -> {nxt['name']}: base_power={base_power:.2f}W, target_power={target_power:.2f}W (NaN values)")
            continue
            
        power_diff = abs(base_power - target_power)
        if power_diff < 0.5:  # Reduced threshold for small power changes
            print(f"  Skipping transition {curr['name']} -> {nxt['name']}: power difference too small ({power_diff:.2f}W)")
            continue
            
        print(f"  Processing transition {curr['name']} -> {nxt['name']}: {base_power:.1f}W -> {target_power:.1f}W (diff: {power_diff:.1f}W)")

        # Calculate the threshold power level for transition detection
        threshold = base_power + threshold_ratio * (target_power - base_power)
        
        # Search for the crossing point starting from the end of the current region.
        # We look at power data points that occur *after* the current region ends.
        search_start_idx = np.searchsorted(times, t_curr_end, 'left')
        
        transition_latency_s = np.nan

        relevant_times = times[search_start_idx:]
        relevant_watts = watts[search_start_idx:]
        
        if relevant_times.size == 0:
            continue # No power data available after the current region ends

        if target_power > base_power:
            # Power is increasing: find the first data point where power is >= threshold
            indices = np.where(relevant_watts >= threshold)[0]
        else:
            # Power is decreasing: find the first data point where power is <= threshold
            indices = np.where(relevant_watts <= threshold)[0]

        if indices.size > 0:
            first_cross_index_in_relevant = indices[0]
            time_of_crossing = relevant_times[first_cross_index_in_relevant]
            # The latency is the time from the end of the current region to the crossing point
            transition_latency_s = time_of_crossing - t_curr_end
        
        transition_results.append({
            'from_region': curr['name'],
            'to_region': nxt['name'],
            'transition_start_time_s': t_curr_end, # Time when the transition period officially begins (end of from_region)
            'base_power_w': base_power,
            'target_power_w': target_power,
            'transition_latency_s': transition_latency_s
        })
        
    return pd.DataFrame(transition_results)
