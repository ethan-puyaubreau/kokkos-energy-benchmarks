"""
Plotting utilities for power analysis visualization.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from tqdm import tqdm


def get_region_colors():
    """Defines consistent colors for each region/kernel type."""
    colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
        '#DDA0DD', '#98D8C8', '#FFB6C1', '#87CEEB', '#F0E68C',
        '#FFA07A', '#20B2AA', '#778899', '#B0C4DE', '#F5DEB3',
    ]
    return colors


def create_correlation_plot(power_df, regions_df):
    """Creates the correlation plot between power and regions."""
    print("Creating correlation plot structure...")
    
    unique_regions = regions_df['name'].unique()
    colors = get_region_colors()
    region_colors = {region: colors[i % len(colors)] for i, region in enumerate(unique_regions)}
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True,
                                       gridspec_kw={'height_ratios': [3, 2, 2]})
    
    print("Plotting power curve...")
    ax1.plot(power_df['time_seconds'], power_df['power_watts'],
             'k-', linewidth=1, alpha=0.7, label='NVML Power')
    ax1.set_ylabel('Power (W)', fontsize=12)
    ax1.set_title('Correlation between NVML Power, Kokkos Regions, and Power Transition Latency', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    print("Plotting regions (this may take a while for many regions)...")
    y_positions = {region: i for i, region in enumerate(unique_regions)}
    
    for _, row in tqdm(regions_df.iterrows(), total=len(regions_df), desc="Rendering regions"):
        region_name = row['name']
        start_time = row['start_time_seconds']
        duration = row['duration_seconds']
        y_pos = y_positions[region_name]
        color = region_colors[region_name]
        
        rect = Rectangle((start_time, y_pos - 0.4), duration, 0.8,
                        facecolor=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.add_patch(rect)
    
    ax2.set_ylabel('Regions/Kernels', fontsize=12)
    ax2.set_ylim(-0.5, len(unique_regions) - 0.5)
    ax2.set_yticks(range(len(unique_regions)))
    ax2.set_yticklabels([region.replace('Kokkos::', '').replace('compute_kernel', 'compute') 
                        for region in unique_regions], fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    legend_patches = [mpatches.Patch(color=region_colors[region], 
                                   label=region.replace('Kokkos::', '').replace('compute_kernel', 'compute'))
                     for region in unique_regions]
    ax2.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.8)
    print("Correlation plot structure created.")
    return fig, ax1, ax2, ax3


def add_power_overlay_to_regions(power_df, regions_df, ax1):
    """Adds a colored overlay to the power curve based on active regions."""
    print("Adding power overlay to regions (this may take a while)...")
    
    unique_regions = regions_df['name'].unique()
    colors = get_region_colors()
    region_colors = {region: colors[i % len(colors)] for i, region in enumerate(unique_regions)}
    
    power_times = power_df['time_seconds'].values
    power_watts = power_df['power_watts'].values

    for _, row in tqdm(regions_df.iterrows(), total=len(regions_df), desc="Applying overlay"):
        region_name = row['name']
        start_time = row['start_time_seconds']
        end_time = row['end_time_seconds']
        color = region_colors[region_name]
        
        idx_start = np.searchsorted(power_times, start_time, side='left')
        idx_end = np.searchsorted(power_times, end_time, side='right')
        
        if idx_start < idx_end:
            power_subset_time = power_times[idx_start:idx_end]
            power_subset_watts = power_watts[idx_start:idx_end]
            
            ax1.fill_between(power_subset_time,
                             power_subset_watts,
                             alpha=0.3, color=color, label=None)
    print("Power overlay added.")


def plot_transition_times(transition_df, ax):
    """Plots the power transition latency results on a given axes using a stem plot."""
    if transition_df.empty or transition_df['transition_latency_s'].isnull().all():
        print("Transition data is empty or contains no valid values, skipping plot.")
        ax.text(0.5, 0.5, 'No valid transition data for latency plot',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        ax.set_ylabel('Transition Latency (s)', fontsize=12)
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.grid(True, alpha=0.3)
        return

    # Use a stem plot for better visibility of individual transition events
    (markerline, stemlines, baseline) = ax.stem(
        transition_df['transition_start_time_s'], # X-axis: The time when the transition starts
        transition_df['transition_latency_s'],    # Y-axis: The calculated latency
        linefmt='grey',
        markerfmt='o',
        bottom=0,
        label='Transition Latency (s)'
    )
    plt.setp(markerline, color='#3498db', markersize=5)
    plt.setp(stemlines, color='#bdc3c7', linewidth=1)
    
    # Add value labels to each point
    for i, (x, y) in enumerate(zip(transition_df['transition_start_time_s'], transition_df['transition_latency_s'])):
        if not np.isnan(y):
            ax.annotate(f'{y:.3f}s', 
                       xy=(x, y), 
                       xytext=(0, 8), 
                       textcoords='offset points',
                       ha='center', 
                       va='bottom',
                       fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    ax.set_ylabel('Transition Latency (s)', fontsize=12)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    # Set a reasonable y-limit to handle potential outliers
    ax.set_ylim(bottom=0, top=transition_df['transition_latency_s'].dropna().quantile(0.99) * 1.2)


def add_config_text_to_plot(fig, global_settings, config_sets):
    """Adds benchmark configuration text to the plot."""
    config_text = "Benchmark Configuration:\n"

    if global_settings:
        config_text += "\nGlobal Settings:\n"
        for key, value in global_settings.items():
            config_text += f"  {key}: {value}\n"
    
    if config_sets:
        config_text += "\nConfiguration Sets:\n"
        for i, cfg in enumerate(config_sets):
            config_text += f"  Set {i+1} ({cfg.get('name', 'N/A')}):\n"
            display_params = {k: v for k, v in cfg.items() if k not in ['name', 'count']}
            sorted_display_params = sorted(display_params.items())
            for j, (param, value) in enumerate(sorted_display_params):
                config_text += f"    {param}: {value}"
                if j < len(sorted_display_params) - 1:
                    config_text += ", \n"
                else:
                    config_text += "\n"
            config_text += f"    Count: {cfg.get('count', 'N/A')}\n"
    
    plt.subplots_adjust(right=0.7)
    fig.text(0.85, 0.95, config_text, 
             transform=fig.transFigure,
             fontsize=8, verticalalignment='top', 
             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="lightgray", lw=0.5, alpha=0.8))
