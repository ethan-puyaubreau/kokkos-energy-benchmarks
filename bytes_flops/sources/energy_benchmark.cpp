//==============================================================================
// File: energy_benchmark.cpp
//
// Description:
// This file contains the main logic for a Kokkos-based benchmark application.
// It loads benchmark configurations from a file or uses default settings,
// then executes various kernels based on the defined configurations,
// with inter-kernel and inter-repetition waits.
//
// Copyright (2022) National Technology & Engineering Solutions of Sandia, LLC (NTESS).
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//==============================================================================

#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>
#include "bench.hpp" // Contains run_stride_unroll declarations
#include <cstdlib>   // For atoi
#include <cstdio>    // For printf, fprintf, fopen, fclose, fgets, sscanf
#include <cstring>   // For strcpy, strncmp, strcspn
#include <climits>   // For INT_MAX (though not explicitly used in the original code, good for completeness)

// Structure to hold parameters for a single benchmark run.
struct BenchmarkParams {
  int P, N, K, R, D, U, F, T, S, B, I;
};

// Structure to define a set of benchmark configurations.
struct ConfigurationSet {
  char name[64];
  int count;
  BenchmarkParams params;
};

// Structure to hold the overall benchmark configuration.
struct BenchmarkConfig {
  int global_repetitions;
  int inter_kernel_wait_ms;
  int inter_repetition_wait_ms;
  ConfigurationSet configurations[5]; // Fixed size array for configurations
  int config_count;                   // Actual number of configurations loaded
};

// Explicit template instantiations for run_stride_unroll with different data types.
// These declarations indicate that the definitions for these specific template
// instantiations exist elsewhere (e.g., in bench.cpp).
extern template void run_stride_unroll<float>(int, int, int, int, int, int,
                                              int, int, int, int);
extern template void run_stride_unroll<double>(int, int, int, int, int, int,
                                               int, int, int, int);
extern template void run_stride_unroll<int32_t>(int, int, int, int, int, int,
                                                int, int, int, int);
extern template void run_stride_unroll<int64_t>(int, int, int, int, int, int,
                                                int, int, int, int);

/**
 * @brief Sets default benchmark configurations.
 *
 * @param config Reference to the BenchmarkConfig structure to populate with defaults.
 */
void setDefaultConfig(BenchmarkConfig& config) {
  config.global_repetitions       = 3;
  config.inter_kernel_wait_ms     = 100;
  config.inter_repetition_wait_ms = 1000;
  config.config_count             = 0; // Will be set to 5 after populating

  // Define default "bandwidth_bound" configuration
  strcpy(config.configurations[0].name, "bandwidth_bound");
  config.configurations[0].count  = 5;
  config.configurations[0].params = {2, 100000, 1024, 1, 1, 1,
                                     1, 256,    6000, 2, 10};

  // Define default "cache_bound" configuration
  strcpy(config.configurations[1].name, "cache_bound");
  config.configurations[1].count  = 3;
  config.configurations[1].params = {2, 100000, 1024, 64, 1, 1,
                                     1, 512,    20000, 2, 10};

  // Define default "compute_bound" configuration
  strcpy(config.configurations[2].name, "compute_bound");
  config.configurations[2].count  = 4;
  config.configurations[2].params = {2, 100000, 1024, 1, 1, 8,
                                     64, 256,    6000, 2, 10};

  // Define default "load_slots_used" configuration
  strcpy(config.configurations[3].name, "load_slots_used");
  config.configurations[3].count  = 2;
  config.configurations[3].params = {2, 20000, 256, 32, 16, 1,
                                     1, 256,    6000, 2, 10};

  // Define default "inefficient_load" configuration
  strcpy(config.configurations[4].name, "inefficient_load");
  config.configurations[4].count  = 2;
  config.configurations[4].params = {2, 20000, 256, 32, 2, 1,
                                     1, 256,    20000, 2, 10};

  config.config_count = 5; // All 5 default configurations are set
}

/**
 * @brief Loads benchmark configurations from a specified file.
 * If the file cannot be opened or contains invalid entries,
 * default configurations are used or supplemented.
 *
 * @param filename The path to the configuration file.
 * @param outConfig Pointer to the BenchmarkConfig structure to populate.
 * @return true if the file was opened and parsed, false otherwise (defaults used).
 */
bool loadConfig(const char* filename, BenchmarkConfig* outConfig) {
  BenchmarkConfig& config = *outConfig;

  // Set defaults first, they will be overridden by file content if present
  setDefaultConfig(config);

  FILE* file = fopen(filename, "r");
  if (!file) {
    fprintf(stderr, "[ERROR] Could not open config file: %s, using defaults\n",
            filename);
    return false;
  }

  char line[512];
  bool ok           = true;
  config.config_count = 0; // Reset count to parse configurations from file

  // Read the file line by line
  while (fgets(line, sizeof(line), file)) {
    // Remove newline and carriage return characters
    line[strcspn(line, "\r\n")] = 0;

    // Skip comment lines (starting with '#') and empty lines
    if (line[0] == '#' || line[0] == '\0' || strlen(line) == 0) {
      continue;
    }

    // Parse global parameters or individual benchmark configurations
    if (strncmp(line, "global_repetitions=", 19) == 0) {
      config.global_repetitions = atoi(line + 19);
      printf("[CONFIG] global_repetitions = %d\n", config.global_repetitions);
    } else if (strncmp(line, "inter_kernel_wait_ms=", 21) == 0) {
      config.inter_kernel_wait_ms = atoi(line + 21);
      printf("[CONFIG] inter_kernel_wait_ms = %d\n",
             config.inter_kernel_wait_ms);
    } else if (strncmp(line, "inter_repetition_wait_ms=", 25) == 0) {
      config.inter_repetition_wait_ms = atoi(line + 25);
      printf("[CONFIG] inter_repetition_wait_ms = %d\n",
             config.inter_repetition_wait_ms);
    } else {
      // Attempt to parse a configuration line: name count P N K R D U F T S B I
      char name[64];
      int count, P, N, K, R, D, U, F, T, S, B, I;

      int parsed =
          sscanf(line, "%63s %d %d %d %d %d %d %d %d %d %d %d %d", name, &count,
                 &P, &N, &K, &R, &D, &U, &F, &T, &S, &B, &I);

      // If all 13 values are parsed and there's space in the configurations array
      if (parsed == 13 && config.config_count < 5) {
        strcpy(config.configurations[config.config_count].name, name);
        config.configurations[config.config_count].count  = count;
        config.configurations[config.config_count].params = {P, N, K, R, D, U,
                                                             F, T, S, B, I};
        printf(
            "[CONFIG] %s: count=%d P=%d N=%d K=%d R=%d D=%d U=%d F=%d T=%d "
            "S=%d B=%d I=%d\n",
            name, count, P, N, K, R, D, U, F, T, S, B, I);
        config.config_count++;
      } else if (parsed > 0 && parsed != 13) {
        // Warn if a line was partially parsed but not completely
        fprintf(stderr,
                "[WARNING] Invalid configuration line (expected 13 values, got "
                "%d): %s\n",
                parsed, line);
        ok = false;
      }
    }
  }

  fclose(file);

  // If no valid configurations were found in the file, revert to defaults
  if (config.config_count == 0) {
    fprintf(stderr,
            "[WARNING] No valid configurations found in file, using defaults\n");
    setDefaultConfig(config);
    ok = false;
  }

  return ok;
}

/**
 * @brief Runs a single benchmark kernel based on the provided parameters.
 * Uses Kokkos profiling regions for timing.
 *
 * @param params The BenchmarkParams for the kernel.
 * @param kernelName The name of the kernel for profiling.
 */
void runKernel(const BenchmarkParams& params, const char* kernelName) {
  Kokkos::Profiling::pushRegion(kernelName);

  // Dispatch to the correct run_stride_unroll template based on data type (P)
  if (params.P == 1) {
    run_stride_unroll<float>(params.N, params.K, params.R, params.D, params.U,
                             params.F, params.T, params.S, params.B, params.I);
  } else if (params.P == 2) {
    run_stride_unroll<double>(params.N, params.K, params.R, params.D, params.U,
                              params.F, params.T, params.S, params.B, params.I);
  } else if (params.P == 3) {
    run_stride_unroll<int32_t>(params.N, params.K, params.R, params.D, params.U,
                               params.F, params.T, params.S, params.B,
                               params.I);
  } else if (params.P == 4) {
    run_stride_unroll<int64_t>(params.N, params.K, params.R, params.D, params.U,
                               params.F, params.T, params.S, params.B,
                               params.I);
  }

  Kokkos::Profiling::popRegion();
}

/**
 * @brief Creates a busy-wait loop for a specified duration.
 * Used to simulate inter-kernel or inter-repetition delays.
 * Uses Kokkos profiling regions.
 *
 * @param milliseconds The duration to wait in milliseconds.
 * @param regionName The name of the wait region for profiling.
 */
void waitRegion(int milliseconds, const char* regionName) {
  Kokkos::Profiling::pushRegion(regionName);

  // Simple busy-wait loop
  for (volatile int i = 0; i < milliseconds * 10000; i++) {
    for (volatile int j = 0; j < 100; j++) {
      // Empty loop body to consume time
    }
  }

  Kokkos::Profiling::popRegion();
}

int main(int argc, char* argv[]) {
  // Initialize Kokkos library
  Kokkos::initialize();

  // Determine configuration file path
  const char* configFile = "benchmark_config.txt";
  if (argc >= 2) {
    configFile = argv[1]; // Use command-line argument if provided
  }

  // Load benchmark configuration
  BenchmarkConfig config;
  bool configOk = loadConfig(configFile, &config);
  if (!configOk) {
    fprintf(
        stderr,
        "[WARNING] Using default or partial config due to previous errors.\n");
  }

  // Print loaded configuration summary
  printf("Starting benchmark with %d global repetitions\n",
         config.global_repetitions);
  printf("Config: inter_kernel_wait=%dms, inter_repetition_wait=%dms\n",
         config.inter_kernel_wait_ms, config.inter_repetition_wait_ms);

  // Main benchmark loop: Iterate through global repetitions
  for (int rep = 0; rep < config.global_repetitions; rep++) {
    Kokkos::Profiling::pushRegion("global_repetition");

    printf("Global repetition %d/%d\n", rep + 1, config.global_repetitions);

    // Iterate through each defined configuration set
    for (int configIdx = 0; configIdx < config.config_count; configIdx++) {
      const ConfigurationSet& configSet = config.configurations[configIdx];
      printf("  Running %d %s kernels\n", configSet.count, configSet.name);

      // Run each kernel within the current configuration set
      for (int i = 0; i < configSet.count; i++) {
        runKernel(configSet.params, configSet.name);
        // Wait between kernel runs
        waitRegion(config.inter_kernel_wait_ms, "inter_kernel_wait");
      }
    }

    Kokkos::Profiling::popRegion();

    // Wait between global repetitions, except after the last one
    if (rep < config.global_repetitions - 1) {
      waitRegion(config.inter_repetition_wait_ms, "inter_repetition_wait");
    }
  }

  printf("Benchmark completed\n");

  // Finalize Kokkos library
  Kokkos::finalize();

  // Return 0 if configuration was loaded successfully, 1 otherwise
  return configOk ? 0 : 1;
}
