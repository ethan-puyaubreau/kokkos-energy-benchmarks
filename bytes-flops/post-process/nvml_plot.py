"""
NVML Power Analysis Tool - Main Entry Point

This tool analyzes the correlation between NVML power measurements and Kokkos regions,
providing visualization and statistics for GPU power behavior during code execution.
"""

import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import sys

# Import local modules
from data_loader import load_data, normalize_timestamps
from plotting import (create_correlation_plot, add_power_overlay_to_regions, 
                     plot_transition_times, add_config_text_to_plot)
from statistics import generate_statistics
from transition_analysis import compute_power_transition_latency
from config_parser import parse_benchmark_config
from file_utils import find_data_files, get_config_file_path, generate_output_filename


def main():
    """Main function orchestrating the analysis workflow."""
    parser = argparse.ArgumentParser(description="Analyze the correlation between NVML power and Kokkos regions.")
    parser.add_argument('input_directory', type=str,
                        help='Path to the directory containing NVML power, regions, and benchmark_config.txt files.')
    parser.add_argument('--output', type=str,
                        help='Output file name for the plot.')
    args = parser.parse_args()
    
    input_dir = args.input_directory
    
    # Find required data files
    power_file, regions_file = find_data_files(input_dir)
    config_file = get_config_file_path(input_dir)
    output_file = generate_output_filename(power_file, args.output, input_dir)
    
    print(f"Found power file: {power_file}")
    print(f"Found regions file: {regions_file}")

    # Load and process data
    power_df, regions_df = load_data(power_file, regions_file)
    power_df, regions_df = normalize_timestamps(power_df, regions_df)
    power_df.sort_values(by='time_seconds', inplace=True)
    power_df.reset_index(drop=True, inplace=True)

    # Create visualization
    fig, ax1, ax2, ax3 = create_correlation_plot(power_df, regions_df)
    add_power_overlay_to_regions(power_df, regions_df, ax1)

    # Add benchmark configuration if available
    if config_file:
        global_settings, config_sets = parse_benchmark_config(config_file)
        add_config_text_to_plot(fig, global_settings, config_sets)
    else:
        print("No benchmark_config.txt found. Plot will not include benchmark parameters.", file=sys.stderr)

    # Generate statistics
    generate_statistics(power_df, regions_df)

    # Analyze power transition latency
    print("\n=== MEASURING AND PLOTTING GPU POWER TRANSITION LATENCY ===")
    transition_df = compute_power_transition_latency(power_df, regions_df)
    plot_transition_times(transition_df, ax3)
    
    if not transition_df.empty:
        print(transition_df)
        transition_csv = os.path.join(input_dir, 'power_transition_latency.csv')
        transition_df.to_csv(transition_csv, index=False)
        print(f"Transition latency results saved to {transition_csv}")
    else:
        print("Could not calculate transition latency, no valid transitions found.")

    # Save and display plot
    print(f"\nSaving plot to: {output_file}...")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {output_file}")
    
    print("Displaying plot (close window to continue/exit)...")
    plt.show()

if __name__ == "__main__":
    main()
