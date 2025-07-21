# Energy Benchmark System

This benchmark system runs multiple kernel configurations within Kokkos regions for energy profiling.

## Usage

### Building

From the kokkos-tools-energy root directory:

```bash
cd build
make -C profiling/energy-profiler/energy-benchmark
```

This creates the `energy_benchmark` executable in `build/profiling/energy-profiler/energy-benchmark/`.

### Running

```bash
cd build/profiling/energy-profiler/energy-benchmark
./energy_benchmark [config_file]
```

If no config file is specified, it uses defaults.

## Benchmark Overview

The benchmark executes kernel configurations with the following structure:

1. **Global Repetitions**: Repeats the entire suite multiple times.
2. **Configuration Sets**: Executes predefined kernel types.
3. **Wait Periods**: Configurable delays between kernels and repetitions.

A configuration file template can be found in `templates/benchmark_config.txt`.

## Profiling

The benchmark integrates with Kokkos profiling tools to measure energy consumption and performance metrics.

### Parameters Explanation

- **P**: Precision (1=float, 2=double, 3=int32_t, 4=int64_t)
- **N,K**: Dimensions of the 2D array to allocate
- **R**: How often to loop through the K dimension with each team
- **D**: Distance between loaded elements (stride)
- **U**: How many independent flops to do per load
- **F**: How many times to repeat the U unrolled operations
- **T**: Team size
- **S**: Shared memory per team (controls GPU occupancy)
- **B**: Units for bandwidth reporting (2=GiB, 10=GB)
- **I**: Iterations of the kernel to time over

## Default Configurations

### Bandwidth Bound
Tests memory bandwidth limitations with minimal compute.
- Configuration: P=2, N=100000, K=1024, R=1, D=1, U=1, F=1, T=256, S=6000

### Cache Bound  
Tests cache hierarchy performance with high reuse.
- Configuration: P=2, N=100000, K=1024, R=64, D=1, U=1, F=1, T=512, S=20000

### Compute Bound
Tests computational throughput with high arithmetic intensity.
- Configuration: P=2, N=100000, K=1024, R=1, D=1, U=8, F=64, T=256, S=6000

### Load Slots Used
Tests memory subsystem with specific stride patterns.
- Configuration: P=2, N=20000, K=256, R=32, D=16, U=1, F=1, T=256, S=6000

### Inefficient Load
Tests memory access efficiency with poor stride patterns.
- Configuration: P=2, N=20000, K=256, R=32, D=2, U=1, F=1, T=256, S=20000

## Energy Profiling

This benchmark is designed to work with Kokkos energy profiling tools. The regions allow for precise measurement of:

- Energy consumption per kernel type
- Energy overhead of wait periods
- Energy patterns across multiple repetitions

Use with Kokkos profiling connectors like variorum-connector for energy measurements.

## Output

The benchmark provides detailed performance metrics for each kernel execution:
- Execution time
- Memory bandwidth (GiB/s or GB/s)
- Computational throughput (GFlop/s)

All output is clearly labeled by kernel type and repetition for easy analysis.