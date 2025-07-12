//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <stdexcept>
#include <chrono>
#include <algorithm>
#include <string>
#include <numeric>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <climits>

#if defined(KOKKOS_ENABLE_CUDA)
#include <nvml.h>
#include <thread>
#include <atomic>
#endif

struct BenchmarkParams {
  int P, N, K, R, D, U, F, T, S, B, I;
};

struct ConfigurationSet {
  char name[64];
  int count;
  BenchmarkParams params;
};

struct BenchmarkConfig {
  int global_repetitions;
  ConfigurationSet configurations[5];
  int config_count;
};

struct BenchmarkResult {
    double compute_percent;
    double memory_percent;
    double bandwidth_gb_s;
    double execution_time_ms;
    size_t array_size;
    int compute_intensity;
    int memory_intensity;
    double avg_gpu_utilization;
    double avg_memory_utilization;
    double avg_power_usage_mW;
    double avg_sm_clock_MHz;
    double avg_mem_clock_MHz;
    double avg_temperature_C;
};

#if defined(KOKKOS_ENABLE_CUDA)
void handle_nvml_error(nvmlReturn_t result, const char* func, int line) {
    if (result != NVML_SUCCESS) {
        std::cerr << "NVML Error in " << func << " at line " << line
                  << ": " << nvmlErrorString(result) << std::endl;
    }
}
#define NVML_CHECK(call) handle_nvml_error(call, __FUNCTION__, __LINE__)

struct NvmlRawMetrics {
    std::vector<unsigned int> gpu_utilization;
    std::vector<unsigned int> memory_utilization;
    std::vector<unsigned int> power_usage_mW;
    std::vector<unsigned int> sm_clock_MHz;
    std::vector<unsigned int> mem_clock_MHz;
    std::vector<unsigned int> temperature_C;
};
#endif

bool loadConfig(const char* filename, BenchmarkConfig* outConfig) {
  BenchmarkConfig& config = *outConfig;

  FILE* file = fopen(filename, "r");
  if (!file) {
    fprintf(stderr, "[ERROR] Could not open config file: %s\n", filename);
    return false;
  }

  char line[512];
  bool ok = true;
  config.config_count = 0;

  while (fgets(line, sizeof(line), file)) {
    // Remove newline and carriage return
    line[strcspn(line, "\r\n")] = 0;

    // Skip comments and empty lines
    if (line[0] == '#' || line[0] == '\0' || strlen(line) == 0) {
      continue;
    }

    // Parse key=value pairs
    if (strncmp(line, "global_repetitions=", 19) == 0) {
      config.global_repetitions = atoi(line + 19);
      printf("[CONFIG] global_repetitions = %d\n", config.global_repetitions);
    } else {
      // Parse configuration lines: name count P N K R D U F T S B I
      char name[64];
      int count, P, N, K, R, D, U, F, T, S, B, I;

      int parsed =
          sscanf(line, "%63s %d %d %d %d %d %d %d %d %d %d %d %d", name, &count,
                 &P, &N, &K, &R, &D, &U, &F, &T, &S, &B, &I);

      if (parsed == 13 && config.config_count < 5) {
        strcpy(config.configurations[config.config_count].name, name);
        config.configurations[config.config_count].count = count;
        config.configurations[config.config_count].params = {P, N, K, R, D, U,
                                                             F, T, S, B, I};
        printf(
            "[CONFIG] %s: count=%d P=%d N=%d K=%d R=%d D=%d U=%d F=%d T=%d "
            "S=%d B=%d I=%d\n",
            name, count, P, N, K, R, D, U, F, T, S, B, I);
        config.config_count++;
      } else if (parsed > 0 && parsed != 13) {
        fprintf(stderr,
                "[WARNING] Invalid configuration line (expected 13 values, got "
                "%d): %s\n",
                parsed, line);
        ok = false;
      }
    }
  }

  fclose(file);

  if (config.config_count == 0) {
    fprintf(stderr, "[ERROR] No valid configurations found in config file\n");
    ok = false;
  }

  return ok;
}

// Stride unroll kernels using the same approach as energy_benchmark.cpp
template<typename Scalar>
KOKKOS_INLINE_FUNCTION void run_stride_unroll_kernel(Scalar* data, int N, int K, int R, int D, int U, int F, int S, int B, int I) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N) return;
    
    Scalar value = data[index];
    
    // Apply stride pattern
    for (int r = 0; r < R; r++) {
        for (int u = 0; u < U; u++) {
            for (int f = 0; f < F; f++) {
                // Compute intensive operations
                value = value * 1.1f + 0.1f;
                value = value * value + 1.0f;
            }
            
            // Memory operations with stride
            int stride_idx = (index + u * K) % N;
            if (stride_idx < N) {
                for (int d = 0; d < D; d++) {
                    data[stride_idx] = value;
                    value = data[(stride_idx + d) % N];
                }
            }
        }
    }
    
    data[index] = value;
}

template<typename Scalar>
struct ComputeBoundKernel {
    Kokkos::View<Scalar*> data;
    BenchmarkParams params;

    ComputeBoundKernel(Kokkos::View<Scalar*> data_, BenchmarkParams params_)
        : data(data_), params(params_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i) const {
        if (i >= params.N) return;
        
        Scalar value = data[i];
        
        // Apply the same compute pattern as energy_benchmark.cpp
        for (int r = 0; r < params.R; r++) {
            for (int u = 0; u < params.U; u++) {
                for (int f = 0; f < params.F; f++) {
                    value = value * 1.1 + 0.1;
                    value = value * value + 1.0;
                }
            }
        }
        
        data[i] = value;
    }
};

template<typename Scalar>
struct MemoryBoundKernel {
    Kokkos::View<Scalar*> data;
    BenchmarkParams params;

    MemoryBoundKernel(Kokkos::View<Scalar*> data_, BenchmarkParams params_)
        : data(data_), params(params_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i) const {
        if (i >= params.N) return;
        
        Scalar value = data[i];
        
        // Apply memory-intensive pattern with stride
        for (int r = 0; r < params.R; r++) {
            for (int u = 0; u < params.U; u++) {
                int stride_idx = (i + u * params.K) % params.N;
                for (int d = 0; d < params.D; d++) {
                    data[stride_idx] = value;
                    value = data[(stride_idx + d) % params.N];
                }
            }
        }
        
        data[i] = value;
    }
};

template<typename Scalar>
BenchmarkResult run_energy_benchmark(const BenchmarkParams& compute_params, 
                                     const BenchmarkParams& memory_params,
                                     double compute_percent, double memory_percent) {
    size_t array_size = compute_params.N;
    
    Kokkos::View<Scalar*> data("data", array_size);
    
    // Initialize data
    Kokkos::parallel_for("init", array_size, KOKKOS_LAMBDA(const int i) {
        data(i) = static_cast<Scalar>(i % 1000);
    });
    Kokkos::fence();

#if defined(KOKKOS_ENABLE_CUDA)
    nvmlDevice_t device;
    nvmlReturn_t nvml_init_result = nvmlInit();
    bool nvml_initialized = (nvml_init_result == NVML_SUCCESS);
    if (nvml_initialized) {
        NVML_CHECK(nvmlDeviceGetHandleByIndex(0, &device));
    }

    std::atomic<bool> stop_monitoring(false);
    NvmlRawMetrics metrics;
    std::thread monitor_thread;

    if (nvml_initialized) {
        monitor_thread = std::thread([&]() {
            while (!stop_monitoring.load()) {
                nvmlUtilization_t util;
                unsigned int power_mW, sm_clock, mem_clock, temp_C;
                
                if (nvmlDeviceGetUtilizationRates(device, &util) == NVML_SUCCESS) {
                    metrics.gpu_utilization.push_back(util.gpu);
                    metrics.memory_utilization.push_back(util.memory);
                }
                
                if (nvmlDeviceGetPowerUsage(device, &power_mW) == NVML_SUCCESS) {
                    metrics.power_usage_mW.push_back(power_mW);
                }
                
                if (nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &sm_clock) == NVML_SUCCESS) {
                    metrics.sm_clock_MHz.push_back(sm_clock);
                }
                
                if (nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &mem_clock) == NVML_SUCCESS) {
                    metrics.mem_clock_MHz.push_back(mem_clock);
                }
                
                if (nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp_C) == NVML_SUCCESS) {
                    metrics.temperature_C.push_back(temp_C);
                }
                
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        });
    }
#endif

    Kokkos::Timer timer;

    // Run compute bound kernel if needed
    if (compute_percent > 0) {
        ComputeBoundKernel<Scalar> compute_kernel(data, compute_params);
        Kokkos::parallel_for("compute_bound", array_size, compute_kernel);
        Kokkos::fence();
    }

    // Run memory bound kernel if needed
    if (memory_percent > 0) {
        MemoryBoundKernel<Scalar> memory_kernel(data, memory_params);
        Kokkos::parallel_for("memory_bound", array_size, memory_kernel);
        Kokkos::fence();
    }

    double execution_time_s = timer.seconds();

#if defined(KOKKOS_ENABLE_CUDA)
    if (nvml_initialized) {
        stop_monitoring.store(true);
        if (monitor_thread.joinable()) {
            monitor_thread.join();
        }
        nvmlShutdown();
    }
#endif

    size_t bytes_transferred = array_size * sizeof(Scalar) * 2; // Read + Write
    double bandwidth_gb_s = (bytes_transferred / 1e9) / execution_time_s;

    double avg_gpu_util = 0.0;
    double avg_mem_util = 0.0;
    double avg_power_mW = 0.0;
    double avg_sm_clock = 0.0;
    double avg_mem_clock = 0.0;
    double avg_temp_C = 0.0;

#if defined(KOKKOS_ENABLE_CUDA)
    if (!metrics.gpu_utilization.empty()) {
        avg_gpu_util = std::accumulate(metrics.gpu_utilization.begin(), metrics.gpu_utilization.end(), 0.0) / metrics.gpu_utilization.size();
    }
    if (!metrics.memory_utilization.empty()) {
        avg_mem_util = std::accumulate(metrics.memory_utilization.begin(), metrics.memory_utilization.end(), 0.0) / metrics.memory_utilization.size();
    }
    if (!metrics.power_usage_mW.empty()) {
        avg_power_mW = std::accumulate(metrics.power_usage_mW.begin(), metrics.power_usage_mW.end(), 0.0) / metrics.power_usage_mW.size();
    }
    if (!metrics.sm_clock_MHz.empty()) {
        avg_sm_clock = std::accumulate(metrics.sm_clock_MHz.begin(), metrics.sm_clock_MHz.end(), 0.0) / metrics.sm_clock_MHz.size();
    }
    if (!metrics.mem_clock_MHz.empty()) {
        avg_mem_clock = std::accumulate(metrics.mem_clock_MHz.begin(), metrics.mem_clock_MHz.end(), 0.0) / metrics.mem_clock_MHz.size();
    }
    if (!metrics.temperature_C.empty()) {
        avg_temp_C = std::accumulate(metrics.temperature_C.begin(), metrics.temperature_C.end(), 0.0) / metrics.temperature_C.size();
    }
#endif

    return {compute_percent, memory_percent, bandwidth_gb_s, execution_time_s * 1000.0,
            array_size, compute_params.F, memory_params.D,
            avg_gpu_util, avg_mem_util, avg_power_mW, avg_sm_clock, avg_mem_clock, avg_temp_C};
}

template<typename Scalar>
std::vector<BenchmarkResult> generate_energy_matrix(const BenchmarkConfig& config, int step = 20) {
    std::vector<BenchmarkResult> results;

    // Find compute_bound and bandwidth_bound configurations
    BenchmarkParams compute_params = {};
    BenchmarkParams memory_params = {};
    bool found_compute = false, found_memory = false;

    for (int i = 0; i < config.config_count; i++) {
        if (strcmp(config.configurations[i].name, "compute_bound") == 0) {
            compute_params = config.configurations[i].params;
            found_compute = true;
        } else if (strcmp(config.configurations[i].name, "bandwidth_bound") == 0) {
            memory_params = config.configurations[i].params;
            found_memory = true;
        }
    }

    if (!found_compute || !found_memory) {
        std::cerr << "[ERROR] Required configurations 'compute_bound' and 'bandwidth_bound' not found" << std::endl;
        return results;
    }

    std::cout << "Generating energy matrix (compute vs memory bound)..." << std::endl;
    std::cout << "Array size: " << compute_params.N << " elements" << std::endl;
    std::cout << "Step: " << step << "%" << std::endl;

    int total_tests = ((100 / step) + 1) * ((100 / step) + 1);
    int current_test = 0;

    for(int compute = 0; compute <= 100; compute += step) {
        for(int memory = 0; memory <= 100; memory += step) {
            current_test++;
            
            // Skip the (0,0) case as it has no computational meaning
            if (compute == 0 && memory == 0) {
                results.push_back({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
                continue;
            }

            std::cout << "Progress: " << current_test << "/" << total_tests 
                      << " - Running: Compute=" << compute << "%, Memory=" << memory << "%" << std::endl;

            try {
                BenchmarkResult result = run_energy_benchmark<Scalar>(
                    compute_params, memory_params, compute, memory);
                results.push_back(result);
            } catch(const std::exception& e) {
                std::cerr << "[ERROR] Exception in benchmark (C=" << compute 
                          << "%, M=" << memory << "%): " << e.what() << std::endl;
                results.push_back({static_cast<double>(compute), static_cast<double>(memory), 
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
            }
        }
    }

    return results;
}

void save_results_csv(const std::vector<BenchmarkResult>& results, const std::string& filename,
                     const BenchmarkConfig& config) {
    std::ofstream file(filename);

    file << "# Energy Matrix Benchmark Results\n";
    file << "# Device: " << Kokkos::DefaultExecutionSpace::name() << "\n";
    file << "# Global repetitions: " << config.global_repetitions << "\n";

    file << "compute_percent,memory_percent,bandwidth_gb_s,execution_time_ms,array_size,compute_intensity,memory_intensity,avg_gpu_utilization,avg_memory_utilization,avg_power_usage_mW,avg_sm_clock_MHz,avg_mem_clock_MHz,avg_temperature_C\n";

    for(const auto& result : results) {
        file << result.compute_percent << "," << result.memory_percent << ","
             << result.bandwidth_gb_s << "," << result.execution_time_ms << ","
             << result.array_size << "," << result.compute_intensity << ","
             << result.memory_intensity << "," << result.avg_gpu_utilization << ","
             << result.avg_memory_utilization << "," << result.avg_power_usage_mW << ","
             << result.avg_sm_clock_MHz << "," << result.avg_mem_clock_MHz << ","
             << result.avg_temperature_C << "\n";
    }

    file.close();
    std::cout << "Results saved to: " << filename << std::endl;
}

void print_matrix(const std::vector<BenchmarkResult>& results, int step) {
    std::cout << "\nBandwidth Matrix (GB/s):" << std::endl;
    std::cout << "Rows: Compute %, Columns: Memory %" << std::endl;
    std::cout << "Note: Case (0%,0%) is skipped as it has no computational meaning" << std::endl;

    std::cout << std::setw(8) << "C\\M";
    for(int memory = 0; memory <= 100; memory += step) {
        std::cout << std::setw(8) << memory;
    }
    std::cout << std::endl;

    int cols = (100 / step) + 1;
    for(int compute = 0; compute <= 100; compute += step) {
        std::cout << std::setw(8) << compute;
        for(int memory = 0; memory <= 100; memory += step) {
            int index = (compute / step) * cols + (memory / step);
            if (index < results.size() && !(compute == 0 && memory == 0)) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(1) 
                          << results[index].bandwidth_gb_s;
            } else {
                std::cout << std::setw(8) << "-";
            }
        }
        std::cout << std::endl;
    }
}

void average_results(std::vector<std::vector<BenchmarkResult>>& all_runs) {
    if (all_runs.empty()) return;
    
    size_t num_results = all_runs[0].size();
    int num_runs = all_runs.size();
    
    std::cout << "Averaging results across " << num_runs << " runs..." << std::endl;
    
    for (size_t i = 0; i < num_results; i++) {
        BenchmarkResult& base_result = all_runs[0][i];
        
        if (base_result.compute_percent == 0 && base_result.memory_percent == 0) {
            continue;
        }
        
        double total_bandwidth = 0.0;
        double total_time = 0.0;
        double total_gpu_util = 0.0;
        double total_mem_util = 0.0;
        double total_power = 0.0;
        double total_sm_clock = 0.0;
        double total_mem_clock = 0.0;
        double total_temp = 0.0;
        
        for (int run = 0; run < num_runs; run++) {
            total_bandwidth += all_runs[run][i].bandwidth_gb_s;
            total_time += all_runs[run][i].execution_time_ms;
            total_gpu_util += all_runs[run][i].avg_gpu_utilization;
            total_mem_util += all_runs[run][i].avg_memory_utilization;
            total_power += all_runs[run][i].avg_power_usage_mW;
            total_sm_clock += all_runs[run][i].avg_sm_clock_MHz;
            total_mem_clock += all_runs[run][i].avg_mem_clock_MHz;
            total_temp += all_runs[run][i].avg_temperature_C;
        }
        
        base_result.bandwidth_gb_s = total_bandwidth / num_runs;
        base_result.execution_time_ms = total_time / num_runs;
        base_result.avg_gpu_utilization = total_gpu_util / num_runs;
        base_result.avg_memory_utilization = total_mem_util / num_runs;
        base_result.avg_power_usage_mW = total_power / num_runs;
        base_result.avg_sm_clock_MHz = total_sm_clock / num_runs;
        base_result.avg_mem_clock_MHz = total_mem_clock / num_runs;
        base_result.avg_temperature_C = total_temp / num_runs;
    }
    
    std::cout << "Averaging complete" << std::endl;
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);

    {
        std::cout << "=================================================" << std::endl;
        std::cout << "Energy Matrix Benchmark" << std::endl;
        std::cout << "=================================================" << std::endl;
        std::cout << "Device: " << Kokkos::DefaultExecutionSpace::name() << std::endl << std::endl;

        const char* configFile = "matrix_config.txt";
        int step = 20;

        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--config" || arg == "-c") {
                if (i + 1 < argc) {
                    configFile = argv[++i];
                }
            } else if (arg == "--step" || arg == "-s") {
                if (i + 1 < argc) {
                    step = atoi(argv[++i]);
                }
            }
        }

        BenchmarkConfig config;
        bool configOk = loadConfig(configFile, &config);
        if (!configOk) {
            fprintf(stderr, "[ERROR] Failed to load configuration. Exiting.\n");
            Kokkos::finalize();
            return 1;
        }

        std::cout << "Global repeats: " << config.global_repetitions << std::endl;

        std::vector<std::vector<BenchmarkResult>> all_runs;
        
        for (int repeat = 1; repeat <= config.global_repetitions; repeat++) {
            std::cout << "\nGlobal repetition " << repeat << "/" << config.global_repetitions << std::endl;
            
            auto results = generate_energy_matrix<double>(config, step);
            all_runs.push_back(results);
        }

        average_results(all_runs);
        auto& final_results = all_runs[0];
        
        print_matrix(final_results, step);

        std::string filename = "energy_matrix_results.csv";
        save_results_csv(final_results, filename, config);

        std::cout << "\nBenchmark complete!" << std::endl;
        std::cout << "Results averaged over " << config.global_repetitions << " runs" << std::endl;
    }

    Kokkos::finalize();
    return 0;
}
