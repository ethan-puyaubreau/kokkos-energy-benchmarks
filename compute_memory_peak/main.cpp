#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <iomanip>

struct BenchmarkConfig {
    double compute_kernel_duration_ms = 100.0;
    double memory_kernel_duration_ms = 50.0;
    int compute_interval_count = 10;
    size_t array_size = 1000000;
    size_t compute_work_size = 100000;

    void print() const {
        std::cout << "Benchmark Configuration:\n";
        std::cout << "  Compute-bound kernel duration: " << compute_kernel_duration_ms << " ms\n";
        std::cout << "  Memory-bound kernel duration: " << memory_kernel_duration_ms << " ms\n";
        std::cout << "  Number of compute intervals: " << compute_interval_count << "\n";
        std::cout << "  Array size: " << array_size << "\n";
        std::cout << "  Compute work size: " << compute_work_size << "\n\n";
    }
};

struct ComputeBoundKernel {
    double target_duration_ms;
    size_t work_size;
    Kokkos::View<double*> data;
    
    ComputeBoundKernel(double duration_ms, size_t size) 
        : target_duration_ms(duration_ms), work_size(size), data("compute_data", size) {
    }
    
    void initialize() {
        auto data_local = data;
        Kokkos::parallel_for("init_compute", work_size, KOKKOS_LAMBDA(int i) {
            data_local(i) = static_cast<double>(i) * 0.001;
        });
        Kokkos::fence();
    }
    
    void run() {
        auto start_time = std::chrono::high_resolution_clock::now();
        double elapsed_ms = 0.0;
        int iterations = 0;
        
        while (elapsed_ms < target_duration_ms) {
            iterations++;
            
            auto data_local = data;
            Kokkos::parallel_for("compute_kernel", work_size, KOKKOS_LAMBDA(int i) {
                double val = data_local(i);
                for (int j = 0; j < 1000; j++) {
                    val = val * 1.0001 + 0.0001;
                    val = Kokkos::sin(val) * Kokkos::cos(val);
                    val = Kokkos::sqrt(Kokkos::abs(val)) + 0.001;
                }
                data_local(i) = val;
            });
            
            Kokkos::fence();
            auto current_time = std::chrono::high_resolution_clock::now();
            elapsed_ms = std::chrono::duration<double, std::milli>(current_time - start_time).count();
        }
        
        std::cout << "Compute-bound kernel executed: " << elapsed_ms << " ms (" 
                  << iterations << " iterations)\n";
    }
};

struct MemoryBoundKernel {
    double target_duration_ms;
    size_t array_size;
    Kokkos::View<double*> src;
    Kokkos::View<double*> dst;
    Kokkos::View<int*> indices;
    
    MemoryBoundKernel(double duration_ms, size_t size) 
        : target_duration_ms(duration_ms), array_size(size),
          src("src_data", size), dst("dst_data", size), indices("indices", size) {
    }
    
    void initialize() {
        auto src_local = src;
        auto dst_local = dst;
        auto indices_local = indices;
        auto size_local = array_size;
        Kokkos::parallel_for("init_memory", array_size, KOKKOS_LAMBDA(int i) {
            src_local(i) = static_cast<double>(i) * 2.0;
            dst_local(i) = 0.0;
            indices_local(i) = (i * 7919) % size_local;
        });
        Kokkos::fence();
    }
    
    void run() {
        auto start_time = std::chrono::high_resolution_clock::now();
        double elapsed_ms = 0.0;
        int iterations = 0;
        
        while (elapsed_ms < target_duration_ms) {
            iterations++;
            
            auto src_local = src;
            auto dst_local = dst;
            auto indices_local = indices;
            auto size_local = array_size;
            Kokkos::parallel_for("memory_kernel", array_size, KOKKOS_LAMBDA(int i) {
                int idx1 = indices_local(i);
                int idx2 = indices_local((i + size_local/2) % size_local);
                int idx3 = indices_local((i + size_local/3) % size_local);
                
                double temp = src_local(idx1) + src_local(idx2) * src_local(idx3);
                dst_local(i) = temp + dst_local((i + 1) % size_local);
            });
            
            Kokkos::fence();
            auto current_time = std::chrono::high_resolution_clock::now();
            elapsed_ms = std::chrono::duration<double, std::milli>(current_time - start_time).count();
            
            if (iterations % 2 == 0) {
                auto temp = src;
                src = dst;
                dst = temp;
            }
        }
        
        std::cout << "Memory-bound kernel executed: " << elapsed_ms << " ms (" 
                  << iterations << " iterations)\n";
    }
};

BenchmarkConfig parse_arguments(int argc, char* argv[]) {
    BenchmarkConfig config;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--compute-duration" && i + 1 < argc) {
            config.compute_kernel_duration_ms = std::atof(argv[++i]);
        } else if (arg == "--memory-duration" && i + 1 < argc) {
            config.memory_kernel_duration_ms = std::atof(argv[++i]);
        } else if (arg == "--intervals" && i + 1 < argc) {
            config.compute_interval_count = std::atoi(argv[++i]);
        } else if (arg == "--array-size" && i + 1 < argc) {
            config.array_size = std::atol(argv[++i]);
        } else if (arg == "--work-size" && i + 1 < argc) {
            config.compute_work_size = std::atol(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --compute-duration <ms>  Compute-bound kernel duration (default: 100)\n";
            std::cout << "  --memory-duration <ms>   Memory-bound kernel duration (default: 50)\n";
            std::cout << "  --intervals <count>      Number of compute intervals (default: 10)\n";
            std::cout << "  --array-size <size>      Memory-bound array size (default: 1000000)\n";
            std::cout << "  --work-size <size>       Compute-bound work size (default: 100000)\n";
            std::cout << "  --help, -h               Show this help message\n";
            exit(0);
        }
    }
    
    return config;
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        std::cout << "=== Kokkos Benchmark: Compute Bound vs Memory Bound ===\n\n";
        
        BenchmarkConfig config = parse_arguments(argc, argv);
        config.print();

        // --- Warmup Phase ---
        std::cout << "Starting 10-second warmup phase...\n";
        Kokkos::Timer warmup_timer;
        size_t warmup_work_size = 1024 * 1024;
        Kokkos::View<double*> warmup_data("warmup_data", warmup_work_size);

        // Initialize warmup data
        Kokkos::parallel_for("warmup_init", warmup_work_size, KOKKOS_LAMBDA(int i) {
            warmup_data(i) = static_cast<double>(i) * 0.1;
        });
        Kokkos::fence(); // Ensure initialization completes

        double elapsed_warmup_ms = 0.0;
        while (elapsed_warmup_ms < 10000.0) {
            Kokkos::parallel_for("warmup_kernel", warmup_work_size, KOKKOS_LAMBDA(int i) {
                warmup_data(i) = warmup_data(i) * 1.00001 + Kokkos::sin(warmup_data(i));
            });
            Kokkos::fence(); // Ensure kernel completes
            elapsed_warmup_ms = warmup_timer.seconds() * 1000.0;
        }
        std::cout << "Warmup finished in " << elapsed_warmup_ms << " ms.\n\n";
        // --- End Warmup Phase ---
        
        ComputeBoundKernel compute_kernel(config.compute_kernel_duration_ms, config.compute_work_size);
        MemoryBoundKernel memory_kernel(config.memory_kernel_duration_ms, config.array_size);
        
        std::cout << "Initializing kernels...\n";
        compute_kernel.initialize();
        memory_kernel.initialize();
        
        std::cout << "Starting benchmark...\n\n";
        
        Kokkos::Timer total_timer;
        
        int half_intervals = config.compute_interval_count / 2;
        int remaining_intervals = config.compute_interval_count - half_intervals;
        
        std::cout << "=== PHASE 1: Compute-Bound Kernels ===\n";
        for (int interval = 0; interval < half_intervals; interval++) {
            std::cout << "Compute-bound kernel " << (interval + 1) << "/" << half_intervals << "...\n";
            compute_kernel.run();
            std::cout << "\n";
        }
        
        std::cout << "=== PHASE 2: Memory-Bound Kernel (middle) ===\n";
        std::cout << "Executing unique memory-bound kernel...\n";
        memory_kernel.run();
        std::cout << "\n";
        
        std::cout << "=== PHASE 3: Remaining Compute-Bound Kernels ===\n";
        for (int interval = 0; interval < remaining_intervals; interval++) {
            std::cout << "Compute-bound kernel " << (interval + 1) << "/" << remaining_intervals << "...\n";
            compute_kernel.run();
            std::cout << "\n";
        }
        
        double total_time = total_timer.seconds();
        
        std::cout << "=== Benchmark Summary ===\n";
        std::cout << "Total execution time: " << std::fixed << std::setprecision(3) 
                  << total_time << " seconds\n";
        std::cout << "Execution pattern: " << half_intervals << " compute + 1 memory + " 
                  << remaining_intervals << " compute\n";
        std::cout << "Total compute kernels: " << config.compute_interval_count << "\n";
        std::cout << "Total memory kernels: 1\n";
        
        std::cout << "\nKokkos Information:\n";
        std::cout << "Execution Space: " << Kokkos::DefaultExecutionSpace::name() << "\n";
        std::cout << "Memory Space: " << Kokkos::DefaultExecutionSpace::memory_space::name() << "\n";
        
    }
    Kokkos::finalize();
    
    return 0;
}