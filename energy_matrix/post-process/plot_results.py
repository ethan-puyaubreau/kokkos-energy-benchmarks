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
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8
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
        # Power usage now in Watts, with dark = high
        ('avg_power_usage_mW', 'Power Usage (W)', 'Greys'),  # <- changed label and colormap
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

    fig.suptitle('GPU Performance Matrix - Compute Load vs Bandwidth Load Analysis',
                 fontsize=16, y=0.98)

    # Iterate through metrics and create a heatmap for each
    for idx, (metric, title, colormap) in enumerate(metrics):
        row = idx // n_cols
        col = idx % n_cols
        # Créer une figure individuelle pour chaque heatmap
        fig_h, ax = plt.subplots(figsize=(8, 6))

        # Check if the metric column exists in the DataFrame
        if metric not in df.columns:
            ax.text(0.5, 0.5, f'Metric {metric}\nnot available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            # Save the figure even if metric is missing
            fig_h.savefig(f"{csv_file.replace('.csv', f'_{metric}_heatmap.png')}", dpi=500, bbox_inches='tight', facecolor='white')
            plt.close(fig_h)
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

            # For power usage, convert mW to W for display
            if metric == 'avg_power_usage_mW':
                matrix = matrix / 1000.0
                colormap = 'YlOrRd'  # Use a colored gradient for power (yellow→red)

            # Create the heatmap with a mask for NaN values
            mask = matrix.isnull()
            im = sns.heatmap(
                matrix, annot=True, fmt='.1f', cmap=colormap, ax=ax,
                mask=mask, cbar_kws={'label': title.split('(')[1].rstrip(')')},
                annot_kws={"size": 10.5}
            )

            ax.set_title(title)
            ax.set_xlabel('Bandwidth Load (%)', fontsize=9.75)  # 5 * 1.75
            ax.set_ylabel('Compute Load (%)', fontsize=9.75)    # 5 * 1.75

            # Rotate x-axis labels for better readability
            ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='y', rotation=0)

        except Exception as e:
            ax.text(0.5, 0.5, f'Error creating\n{title}:\n{str(e)}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)

        # Save each heatmap to a separate file
        out_heatmap = csv_file.replace('.csv', f'_{metric}_heatmap.png')
        try:
            fig_h.savefig(out_heatmap, dpi=500, bbox_inches='tight', facecolor='white')
            print(f"Heatmap saved to: {out_heatmap}")
        except Exception as e:
            print(f"⚠️ Error saving heatmap {metric}: {e}")
        plt.close(fig_h)

    # Remove any unused subplots
    for idx in range(n_metrics, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].remove()

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
        # 1. Bandwidth vs Compute for different Memory levels
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        memory_levels = sorted(df['memory_percent'].unique())
        for memory in memory_levels[::2]:
            subset = df[df['memory_percent'] == memory]
            ax1.plot(subset['compute_percent'], subset['bandwidth_gb_s'],
                    marker='o', label=f'Bandwidth {memory}%', linewidth=2)
        ax1.set_xlabel('Compute Load (%)')
        ax1.set_ylabel('Bandwidth (GB/s)')
        ax1.set_title('Bandwidth vs Compute Load')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()
        out1 = csv_file.replace('.csv', '_bandwidth_vs_compute.png')
        fig1.savefig(out1, dpi=500, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to: {out1}")
        plt.close(fig1)

        # 2. Bandwidth vs Memory for different Compute levels
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        compute_levels = sorted(df['compute_percent'].unique())
        for compute in compute_levels[::2]:
            subset = df[df['compute_percent'] == compute]
            ax2.plot(subset['memory_percent'], subset['bandwidth_gb_s'],
                    marker='s', label=f'Compute {compute}%', linewidth=2)
        ax2.set_xlabel('Bandwidth Load (%)')
        ax2.set_ylabel('Bandwidth (GB/s)')
        ax2.set_title('Bandwidth vs Bandwidth Load')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        out2 = csv_file.replace('.csv', '_bandwidth_vs_bandwidth.png')
        fig2.savefig(out2, dpi=500, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to: {out2}")
        plt.close(fig2)

        # 3. GPU Utilization vs Power (Power in W, dark = high)
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        scatter = ax3.scatter(
            df['avg_gpu_utilization'],
            df['avg_power_usage_mW'] / 1000.0,
            c=df['bandwidth_gb_s'], cmap='viridis', alpha=0.7, s=50
        )
        ax3.set_xlabel('GPU Utilization (%)')
        ax3.set_ylabel('Power Usage (W)')
        ax3.set_title('GPU Utilization vs Power')
        plt.colorbar(scatter, ax=ax3, label='Bandwidth (GB/s)')
        ax3.grid(True, alpha=0.3)
        fig3.tight_layout()
        out3 = csv_file.replace('.csv', '_gpu_util_vs_power.png')
        fig3.savefig(out3, dpi=500, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to: {out3}")
        plt.close(fig3)

        # 4. Performance Efficiency (Bandwidth per Watt) heatmap
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        df['efficiency'] = df['bandwidth_gb_s'] / (df['avg_power_usage_mW'] / 1000.0)
        df['efficiency'] = df['efficiency'].replace([np.inf, -np.inf], 0)
        efficiency_matrix = df.pivot(index='compute_percent', columns='memory_percent', values='efficiency')
        if 0 in efficiency_matrix.index and 0 in efficiency_matrix.columns:
            efficiency_matrix.loc[0, 0] = np.nan
        mask = efficiency_matrix.isnull()
        sns.heatmap(
            efficiency_matrix, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax4, mask=mask,
            annot_kws={"size": 7.5}
        )
        ax4.set_title('Energy Efficiency (GB/s per Watt)')
        ax4.set_xlabel('Bandwidth Load (%)')
        ax4.set_ylabel('Compute Load (%)')
        fig4.tight_layout()
        out4 = csv_file.replace('.csv', '_efficiency_heatmap.png')
        fig4.savefig(out4, dpi=500, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to: {out4}")
        plt.close(fig4)

        # 5. Temperature vs Performance (color by Power in W, dark = high)
        fig5, ax5 = plt.subplots(figsize=(8, 6))
        scatter2 = ax5.scatter(
            df['avg_temperature_C'],
            df['bandwidth_gb_s'],
            c=df['avg_power_usage_mW'] / 1000.0, cmap='Greys', alpha=0.7, s=50
        )
        ax5.set_xlabel('Temperature (°C)')
        ax5.set_ylabel('Bandwidth (GB/s)')
        ax5.set_title('Temperature vs Performance')
        plt.colorbar(scatter2, ax=ax5, label='Power (W)')
        ax5.grid(True, alpha=0.3)
        fig5.tight_layout()
        out5 = csv_file.replace('.csv', '_temperature_vs_performance.png')
        fig5.savefig(out5, dpi=500, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to: {out5}")
        plt.close(fig5)

        # 6. Clock Speeds Correlation
        fig6, ax6 = plt.subplots(figsize=(8, 6))
        ax6.scatter(df['avg_sm_clock_MHz'], df['avg_mem_clock_MHz'],
                   c=df['bandwidth_gb_s'], cmap='plasma', alpha=0.7, s=50)
        ax6.set_xlabel('SM Clock (MHz)')
        ax6.set_ylabel('Memory Clock (MHz)')
        ax6.set_title('Clock Speeds Correlation')
        ax6.grid(True, alpha=0.3)
        fig6.tight_layout()
        out6 = csv_file.replace('.csv', '_clock_speeds_correlation.png')
        fig6.savefig(out6, dpi=500, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to: {out6}")
        plt.close(fig6)

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