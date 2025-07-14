"""
File utilities for handling input/output operations.
"""
import os
import sys


def find_data_files(input_dir):
    """
    Finds power and regions CSV files in the input directory.
    Returns tuple of (power_file, regions_file) or raises SystemExit on error.
    """
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    power_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('-nvml-power.csv')]
    regions_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('-nvml-regions.csv')]

    if not power_files:
        print(f"Error: No file ending with '-nvml-power.csv' found in {input_dir}", file=sys.stderr)
        sys.exit(1)
    if not regions_files:
        print(f"Error: No file ending with '-nvml-regions.csv' found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    return power_files[0], regions_files[0]


def get_config_file_path(input_dir):
    """
    Returns the path to the config file, which can be either:
    1. 'benchmark_config.txt' if it exists in the input directory, or
    2. The only .txt file in the directory if there's exactly one
    """
    # Check for benchmark_config.txt first
    config_file = os.path.join(input_dir, 'benchmark_config.txt')
    if os.path.isfile(config_file):
        return config_file
    
    # Look for any single .txt file
    txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    if len(txt_files) == 1:
        return os.path.join(input_dir, txt_files[0])
    
    return None


def generate_output_filename(power_file, output_arg, input_dir):
    """
    Generates the output filename based on the power file name or user argument.
    """
    if output_arg is None:
        base_name = os.path.basename(power_file).replace('-nvml-power.csv', '')
        return os.path.join(input_dir, f"{base_name}.png")
    else:
        return output_arg
