"""
Benchmark configuration parsing utilities.
"""
import sys


def parse_benchmark_config(config_file_path):
    """
    Parses the benchmark_config.txt file and returns global settings
    and a list of configuration sets.
    """
    global_settings = {}
    config_sets = []
    
    param_descriptions = {
        'P': 'DataType', 'N': 'ProbSize', 'K': 'Stride', 'R': 'RepFactor',
        'D': 'DataReuse', 'U': 'Unroll', 'F': 'Flops/Iter', 'T': 'Threads/Team',
        'S': 'TeamSize', 'B': 'Blocks/SM', 'I': 'Iterations'
    }

    try:
        with open(config_file_path, 'r') as f:
            lines = f.readlines()

        in_config_sets_section = False
        headers = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('#'):
                if 'Configuration sets:' in line and 'name count P N K R D U F T S B I' in line:
                    parts = line.split(':')
                    if len(parts) > 1:
                        headers = parts[1].strip().split()
                        in_config_sets_section = True
                continue

            if '=' in line and not in_config_sets_section:
                key, value = line.split('=', 1)
                global_settings[key.strip()] = value.strip()
            elif in_config_sets_section and headers:
                parts = line.split()
                if len(parts) >= len(headers):
                    config_entry = {}
                    for i, header in enumerate(headers):
                        if i < len(parts):
                            config_entry[header] = parts[i]
                    config_sets.append(config_entry)
                elif parts:
                    print(f"Warning: Malformed config line skipped: '{line}'", file=sys.stderr)

    except FileNotFoundError:
        print(f"Warning: benchmark_config.txt not found at {config_file_path}. Skipping config display.", file=sys.stderr)
        return {}, []
    except Exception as e:
        print(f"Warning: Error parsing benchmark_config.txt: {e}. Skipping config display.", file=sys.stderr)
        return {}, []

    for config_set in config_sets:
        for old_key, new_key in param_descriptions.items():
            if old_key in config_set:
                config_set[new_key] = config_set.pop(old_key)

    return global_settings, config_sets
