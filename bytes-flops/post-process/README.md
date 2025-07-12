# NVML Power Analysis Tool

This tool analyzes the correlation between NVML power measurements and Kokkos regions, providing visualization and statistics for GPU power behavior during code execution.

## Structure

The tool has been modularized for better maintainability and readability:

### Main Entry Point
- `nvml_plot.py` - Main script that orchestrates the analysis workflow

### Core Modules
- `data_loader.py` - Data loading and timestamp normalization utilities
- `plotting.py` - Visualization functions for power curves, regions, and transition latency
- `statistics.py` - Statistical analysis of power and region correlations
- `transition_analysis.py` - Power transition latency measurement and analysis
- `config_parser.py` - Benchmark configuration file parsing
- `file_utils.py` - File I/O utilities for finding and handling data files

## Usage

The main entry point remains the same:

```bash
python nvml_plot.py <input_directory> [--output <output_file>]
```

### Arguments
- `input_directory` - Path to directory containing NVML power, regions, and benchmark_config.txt files
- `--output` (optional) - Output file name for the plot

### Required Files
The input directory should contain:
- `*-nvml-power.csv` - NVML power measurements
- `*-nvml-regions.csv` - Kokkos regions data
- `benchmark_config.txt` (optional) - Benchmark configuration parameters

### Output Files
- PNG plot showing power correlation analysis
- `power_transition_latency.csv` - Power transition latency measurements

## Features

1. **Power Curve Visualization** - Time-series plot of GPU power consumption
2. **Region Overlay** - Visual correlation between power levels and active Kokkos regions
3. **Transition Latency Analysis** - Measurement of GPU power response times during region transitions
4. **Statistical Analysis** - Comprehensive statistics on power consumption by region type
5. **Configuration Display** - Benchmark parameters from configuration file

## Module Dependencies

Each module has minimal dependencies on others:
- `data_loader` - Pure data processing (pandas, numpy)
- `plotting` - Visualization (matplotlib, numpy, tqdm)
- `statistics` - Analysis (numpy, tqdm)  
- `transition_analysis` - Latency calculations (pandas, numpy, tqdm)
- `config_parser` - File parsing (sys)
- `file_utils` - File operations (os, sys)
