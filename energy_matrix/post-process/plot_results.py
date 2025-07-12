#!/usr/bin/env python3

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import os

# Configure matplotlib font sizes for better readability in plots
plt.rcParams.update({
    'font.size': 4,
    'axes.titlesize': 9,
    'axes.labelsize': 5,
    'xtick.labelsize': 5,
    'ytick.labelsize': 5,
    'legend.fontsize': 5
})

# Configure the matplotlib backend to avoid display issues
try:
    # Attempt to use Qt5Agg for interactive display
    matplotlib.use('Qt5Agg')
    print("📱 Qt5Agg backend activated for interactive display")
except ImportError:
    try:
        # Fallback to TkAgg
        matplotlib.use('TkAgg')
        print("💡 TkAgg backend activated for interactive display")
    except ImportError:
        # Fallback to Agg (non-interactive)
        matplotlib.use('Agg')
        print("⚠️ Agg backend (non-interactive) activated - plots will only be saved")

def plot_energy_matrix(csv_file="energy_matrix_results.csv"):
    """
    Generates plots from the energy matrix benchmark results.
    Creates heatmaps for each collected metric.
    """

    if not os.path.exists(csv_file):
        print(f"Error: The file {csv_file} does not exist.")
        print("Please run the benchmark first: ./build/energy_matrix")
        return

    try:
        # Read data from the CSV file, ignoring comment lines
        df = pd.read_csv(csv_file, comment='#')

        # Filter out the 0%/0% case as it doesn't represent meaningful work
        print(f"📊 Original data: {len(df)} entries")
        df_filtered = df[~((df['compute_percent'] == 0) & (df['memory_percent'] == 0))].copy()
        print(f"📊 Filtered data (excluding 0%/0%): {len(df_filtered)} entries")

        # Use the filtered data for subsequent operations
        df = df_filtered

    except Exception as e:
        print(f"❌ Error reading the CSV file: {e}")
        return

    # List of metrics to visualize, along with their plot titles and colormaps
    metrics = [
        ('bandwidth_gb_s', 'Bandwidth (GB/s)', 'viridis'),
        ('execution_time_ms', 'Execution Time (ms)', 'plasma_r'),
        ('avg_gpu_utilization', 'GPU Utilization (%)', 'Reds'),
        ('avg_memory_utilization', 'Memory Utilization (%)', 'Blues'),
        ('avg_power_usage_mW', 'Power Usage (mW)', 'hot'),
        ('avg_sm_clock_MHz', 'SM Clock (MHz)', 'copper'),
        ('avg_mem_clock_MHz', 'Memory Clock (MHz)', 'copper_r'),
        ('avg_temperature_C', 'Temperature (°C)', 'coolwarm')
    ]

    # Calculate grid dimensions for subplots
    n_metrics = len(metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    # Create the figure and subplots for heatmaps
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows))
    # Ensure axes is always a 2D array for consistent indexing
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('GPU Performance Matrix - Compute vs Memory Bound Analysis',
                 fontsize=16, fontweight='bold', y=0.98)

    # Iterate through metrics and create a heatmap for each
    for idx, (metric, title, colormap) in enumerate(metrics):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Check if the metric column exists in the DataFrame
        if metric not in df.columns:
            ax.text(0.5, 0.5, f'Metric {metric}\nnot available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue

        try:
            # Create a pivot table for the heatmap
            matrix = df.pivot(index='compute_percent', columns='memory_percent', values=metric)

            # Handle NaN or infinite values by filling with 0
            matrix = matrix.fillna(0)
            matrix = matrix.replace([np.inf, -np.inf], 0)

            # For visualization, set the 0/0 case to NaN so it appears grayed out
            if 0 in matrix.index and 0 in matrix.columns:
                matrix.loc[0, 0] = np.nan

            # Create the heatmap with a mask for NaN values
            mask = matrix.isnull()
            im = sns.heatmap(matrix, annot=True, fmt='.1f', cmap=colormap, ax=ax,
                           mask=mask, cbar_kws={'label': title.split('(')[1].rstrip(')')})

            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Memory Bound (%)')
            ax.set_ylabel('Compute Bound (%)')

            # Rotate x-axis labels for better readability
            ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='y', rotation=0)

        except Exception as e:
            ax.text(0.5, 0.5, f'Error creating\n{title}:\n{str(e)}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)

    # Remove any unused subplots
    for idx in range(n_metrics, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].remove()

    plt.tight_layout()

    # Save the heatmaps to a file
    output_file = csv_file.replace('.csv', '_heatmaps.png')
    try:
        plt.savefig(output_file, dpi=500, bbox_inches='tight', facecolor='white')
        print(f"Heatmaps saved to: {output_file}")
    except Exception as e:
        print(f"⚠️ Error saving heatmaps: {e}")

    # Create and save separate performance curves
    create_performance_curves(df, csv_file)

    # Display plots if the backend is interactive
    if matplotlib.get_backend() not in ['Agg']:
        try:
            plt.show()
        except Exception as e:
            print(f"⚠️ Could not display plots: {e}")
    else:
        print("💡 Plots saved only (non-interactive backend)")

    # Print detailed statistics
    print_detailed_statistics(df)

def create_performance_curves(df, csv_file):
    """
    Creates line plots to analyze performance trends.
    """
    # Filter out the 0%/0% case
    df = df[~((df['compute_percent'] == 0) & (df['memory_percent'] == 0))].copy()

    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Performance Trends Analysis (excluding 0%/0% case)', fontsize=16, fontweight='bold')

        # 1. Bandwidth vs Compute for different Memory levels
        ax1 = axes[0, 0]
        memory_levels = sorted(df['memory_percent'].unique())
        for memory in memory_levels[::2]:  # Plot every other level for readability
            subset = df[df['memory_percent'] == memory]
            ax1.plot(subset['compute_percent'], subset['bandwidth_gb_s'],
                    marker='o', label=f'Memory {memory}%', linewidth=2)
        ax1.set_xlabel('Compute Bound (%)')
        ax1.set_ylabel('Bandwidth (GB/s)')
        ax1.set_title('Bandwidth vs Compute Level')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Bandwidth vs Memory for different Compute levels
        ax2 = axes[0, 1]
        compute_levels = sorted(df['compute_percent'].unique())
        for compute in compute_levels[::2]:
            subset = df[df['compute_percent'] == compute]
            ax2.plot(subset['memory_percent'], subset['bandwidth_gb_s'],
                    marker='s', label=f'Compute {compute}%', linewidth=2)
        ax2.set_xlabel('Memory Bound (%)')
        ax2.set_ylabel('Bandwidth (GB/s)')
        ax2.set_title('Bandwidth vs Memory Level')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. GPU Utilization vs Power
        ax3 = axes[0, 2]
        scatter = ax3.scatter(df['avg_gpu_utilization'], df['avg_power_usage_mW'],
                             c=df['bandwidth_gb_s'], cmap='viridis', alpha=0.7, s=50)
        ax3.set_xlabel('GPU Utilization (%)')
        ax3.set_ylabel('Power Usage (mW)')
        ax3.set_title('GPU Utilization vs Power')
        plt.colorbar(scatter, ax=ax3, label='Bandwidth (GB/s)')
        ax3.grid(True, alpha=0.3)

        # 4. Performance Efficiency (Bandwidth per Watt) heatmap
        ax4 = axes[1, 0]
        df['efficiency'] = df['bandwidth_gb_s'] / (df['avg_power_usage_mW'] / 1000.0)  # GB/s per Watt
        df['efficiency'] = df['efficiency'].replace([np.inf, -np.inf], 0) # Handle potential inf/NaN from division by zero power
        efficiency_matrix = df.pivot(index='compute_percent', columns='memory_percent', values='efficiency')

        if 0 in efficiency_matrix.index and 0 in efficiency_matrix.columns:
            efficiency_matrix.loc[0, 0] = np.nan

        mask = efficiency_matrix.isnull()
        sns.heatmap(efficiency_matrix, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax4, mask=mask)
        ax4.set_title('Energy Efficiency (GB/s per Watt)')
        ax4.set_xlabel('Memory Bound (%)')
        ax4.set_ylabel('Compute Bound (%)')

        # 5. Temperature vs Performance
        ax5 = axes[1, 1]
        scatter2 = ax5.scatter(df['avg_temperature_C'], df['bandwidth_gb_s'],
                              c=df['avg_power_usage_mW'], cmap='hot', alpha=0.7, s=50)
        ax5.set_xlabel('Temperature (°C)')
        ax5.set_ylabel('Bandwidth (GB/s)')
        ax5.set_title('Temperature vs Performance')
        plt.colorbar(scatter2, ax=ax5, label='Power (mW)')
        ax5.grid(True, alpha=0.3)

        # 6. Clock Speeds Correlation
        ax6 = axes[1, 2]
        ax6.scatter(df['avg_sm_clock_MHz'], df['avg_mem_clock_MHz'],
                   c=df['bandwidth_gb_s'], cmap='plasma', alpha=0.7, s=50)
        ax6.set_xlabel('SM Clock (MHz)')
        ax6.set_ylabel('Memory Clock (MHz)')
        ax6.set_title('Clock Speeds Correlation')
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the performance curves to a file
        curves_file = csv_file.replace('.csv', '_performance_curves.png')
        try:
            plt.savefig(curves_file, dpi=500, bbox_inches='tight', facecolor='white')
            print(f"Performance curves saved to: {curves_file}")
        except Exception as e:
            print(f"⚠️ Error saving performance curves: {e}")

    except Exception as e:
        print(f"❌ Error creating performance curves: {e}")

def print_detailed_statistics(df):
    """
    Displays detailed statistics about the benchmark results.
    """
    try:
        # Filter out the 0%/0% case
        df = df[~((df['compute_percent'] == 0) & (df['memory_percent'] == 0))].copy()

        print("\n" + "="*60)
        print("📊 DETAILED PERFORMANCE ANALYSIS")
        print("="*60)
        print("Note: Statistics exclude the meaningless 0%/0% case")

        # General statistics
        print(f"\n🎯 Performance Summary:")
        print(f"  • Maximum Bandwidth: {df['bandwidth_gb_s'].max():.2f} GB/s")
        print(f"  • Minimum Bandwidth: {df['bandwidth_gb_s'].min():.2f} GB/s")
        print(f"  • Average Bandwidth: {df['bandwidth_gb_s'].mean():.2f} GB/s")
        print(f"  • Standard Deviation: {df['bandwidth_gb_s'].std():.2f} GB/s")

        # Optimal configuration for maximum bandwidth
        max_idx = df['bandwidth_gb_s'].idxmax()
        optimal = df.loc[max_idx]
        print(f"\n🏆 Optimal Configuration (Max Bandwidth):")
        print(f"  • Compute: {optimal['compute_percent']}%")
        print(f"  • Memory: {optimal['memory_percent']}%")
        print(f"  • Bandwidth: {optimal['bandwidth_gb_s']:.2f} GB/s")
        print(f"  • GPU Utilization: {optimal['avg_gpu_utilization']:.1f}%")
        print(f"  • Power: {optimal['avg_power_usage_mW']:.0f} mW")
        print(f"  • Temperature: {optimal['avg_temperature_C']:.1f}°C")

        # Energy analysis
        if 'efficiency' in df.columns:
            max_eff_idx = df['efficiency'].idxmax()
            efficient = df.loc[max_eff_idx]
            print(f"\n⚡ Most Efficient Configuration (Max GB/s per Watt):")
            print(f"  • Compute: {efficient['compute_percent']}%")
            print(f"  • Memory: {efficient['memory_percent']}%")
            print(f"  • Efficiency: {efficient['efficiency']:.3f} GB/s/W")
            print(f"  • Bandwidth: {efficient['bandwidth_gb_s']:.2f} GB/s")
            print(f"  • Power: {efficient['avg_power_usage_mW']:.0f} mW")

        # Analysis by category
        print(f"\n📈 Performance by Category:")

        # Pure compute (memory = 0)
        pure_compute = df[df['memory_percent'] == 0]
        if not pure_compute.empty:
            print(f"  • Pure Compute (Memory=0%):")
            print(f"    - Max: {pure_compute['bandwidth_gb_s'].max():.2f} GB/s")
            print(f"    - Min: {pure_compute['bandwidth_gb_s'].min():.2f} GB/s")
            print(f"    - Avg: {pure_compute['bandwidth_gb_s'].mean():.2f} GB/s")

        # Pure memory (compute = 0)
        pure_memory = df[df['compute_percent'] == 0]
        if not pure_memory.empty:
            print(f"  • Pure Memory (Compute=0%):")
            print(f"    - Max: {pure_memory['bandwidth_gb_s'].max():.2f} GB/s")
            print(f"    - Min: {pure_memory['bandwidth_gb_s'].min():.2f} GB/s")
            print(f"    - Avg: {pure_memory['bandwidth_gb_s'].mean():.2f} GB/s")

        # Hybrid (compute > 0 && memory > 0)
        hybrid = df[(df['compute_percent'] > 0) & (df['memory_percent'] > 0)]
        if not hybrid.empty:
            print(f"  • Hybrid (Compute>0% && Memory>0%):")
            print(f"    - Max: {hybrid['bandwidth_gb_s'].max():.2f} GB/s")
            print(f"    - Min: {hybrid['bandwidth_gb_s'].min():.2f} GB/s")
            print(f"    - Avg: {hybrid['bandwidth_gb_s'].mean():.2f} GB/s")

    except Exception as e:
        print(f"❌ Error calculating statistics: {e}")

if __name__ == "__main__":
    csv_file = "energy_matrix_results.csv"
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]

    plot_energy_matrix(csv_file)