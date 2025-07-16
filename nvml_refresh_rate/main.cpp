#include <nvml.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <thread>
#include <atomic>
#include <random>

#if defined(__i386__) || defined(__x86_64__)
#include <immintrin.h>
#endif

constexpr int N = 512;
#define CHECK_CUDA(call) do { cudaError_t err = (call); if (err != cudaSuccess) { std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " (" << __FILE__ << ":" << __LINE__ << ")\n"; std::exit(EXIT_FAILURE); } } while (0)
#define CHECK_CUBLAS(call) do { cublasStatus_t st = (call); if (st != CUBLAS_STATUS_SUCCESS) { std::cerr << "cuBLAS Error: code " << st << " (" << __FILE__ << ":" << __LINE__ << ")\n"; std::exit(EXIT_FAILURE); } } while (0)
#define NVML_FI_DEV_POWER_INSTANT 186

struct NVMLGuard {
    NVMLGuard() { if (nvmlInit() != NVML_SUCCESS) throw std::runtime_error("NVML init failed"); }
    ~NVMLGuard() { nvmlShutdown(); }
};
struct CublasGuard {
    cublasHandle_t handle;
    CublasGuard() { CHECK_CUBLAS(cublasCreate(&handle)); }
    ~CublasGuard() { cublasDestroy(handle); }
};

std::atomic<bool> stopCompute{false};

void computeLoop(cublasHandle_t handle, float* d_A, float* d_B, float* d_C) {
    const float alpha = 1.0f, beta = 0.0f;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> work_dist(10, 200);
    std::uniform_int_distribution<> sleep_dist(10, 200);
    while (!stopCompute.load(std::memory_order_relaxed)) {
        auto work_duration = std::chrono::milliseconds(work_dist(gen));
        auto work_start = std::chrono::high_resolution_clock::now();
        while (std::chrono::high_resolution_clock::now() - work_start < work_duration) {
            if (stopCompute.load(std::memory_order_relaxed)) break;
            CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N));
            CHECK_CUDA(cudaDeviceSynchronize());
        }
        if (stopCompute.load(std::memory_order_relaxed)) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_dist(gen)));
    }
}

int main() {
    try {
        NVMLGuard nvml;
        CublasGuard blas;
        nvmlDevice_t device;
        if (nvmlDeviceGetHandleByIndex(0, &device) != NVML_SUCCESS) throw std::runtime_error("nvmlDeviceGetHandleByIndex failed");
        size_t bytes = size_t(N) * N * sizeof(float);
        float *d_A, *d_B, *d_C;
        CHECK_CUDA(cudaMalloc(&d_A, bytes));
        CHECK_CUDA(cudaMalloc(&d_B, bytes));
        CHECK_CUDA(cudaMalloc(&d_C, bytes));
        std::vector<float> h_A(N*N, 1.0f), h_B(N*N, 1.0f);
        CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));
        std::thread worker(computeLoop, blas.handle, d_A, d_B, d_C);

        size_t sampleCount = 0;
        std::vector<double> deltas;
        deltas.reserve(2000);

        nvmlFieldValue_t powerField;
        powerField.fieldId = NVML_FI_DEV_POWER_INSTANT;
        if (nvmlDeviceGetFieldValues(device, 1, &powerField) != NVML_SUCCESS) throw std::runtime_error("Initial NVML power read failed");
        unsigned int lastPower = static_cast<unsigned int>(powerField.value.uiVal);
        auto lastTime = std::chrono::high_resolution_clock::now();

        auto tStart = std::chrono::high_resolution_clock::now();
        const auto durationLimit = std::chrono::seconds(5);
        const auto tEnd = tStart + durationLimit;
        auto nextSampleTime = tStart;
        bool firstChange = true;

        while (true) {
            while (std::chrono::high_resolution_clock::now() < nextSampleTime) {
                #if defined(__i386__) || defined(__x86_64__)
                _mm_pause();
                #endif
            }
            if (nextSampleTime >= tEnd) break;
            nextSampleTime += std::chrono::milliseconds(1);

            nvmlFieldValue_t powerFieldNow;
            powerFieldNow.fieldId = NVML_FI_DEV_POWER_INSTANT;
            if (nvmlDeviceGetFieldValues(device, 1, &powerFieldNow) != NVML_SUCCESS) {
                std::cerr << "NVML power read failed — stopping measurement.\n";
                break;
            }
            unsigned int pw = static_cast<unsigned int>(powerFieldNow.value.uiVal);
            ++sampleCount;
            if (pw != lastPower) {
                auto now = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(now - lastTime).count();
                if (firstChange) {
                    firstChange = false;
                } else {
                    deltas.push_back(ms);
                }
                lastPower = pw;
                lastTime  = now;
            }
        }

        stopCompute = true;
        worker.join();

        if (!deltas.empty()) deltas.erase(deltas.begin());

        double totalSec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - tStart).count();
        size_t totalChanges = deltas.size();
        double freq_avg = (totalChanges > 0 ? totalChanges / totalSec : 0.0);
        std::sort(deltas.begin(), deltas.end());
        double min_ms = totalChanges ? deltas.front() : 0.0;
        double max_ms = totalChanges ? deltas.back()  : 0.0;
        double avg_ms = totalChanges ? std::accumulate(deltas.begin(), deltas.end(), 0.0) / totalChanges : 0.0;
        double freq_max = min_ms > 0 ? 1e3 / min_ms : 0.0;
        double freq_min = max_ms > 0 ? 1e3 / max_ms : 0.0;

        std::cout << "\n=== Measurement results ===\n"
                  << "Total duration (s)           : " << std::fixed << std::setprecision(3) << totalSec << "\n"
                  << "Sample count                 : " << sampleCount << "\n"
                  << "Change count                 : " << totalChanges << "\n\n";
        if (totalChanges > 0) {
            std::cout << std::fixed << std::setprecision(3)
                      << "Mean interval (ms)           : " << avg_ms << "\n"
                      << "Min/max interval (ms)        : " << min_ms << " / " << max_ms << "\n"
                      << "Mean frequency (Hz)          : " << freq_avg << "\n"
                      << "Min/max frequency (Hz)       : " << freq_min << " / " << freq_max << "\n";
        } else {
            std::cout << "No power change detected.\n";
        }

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    } catch (const std::exception& ex) {
        std::cerr << "Fatal error: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}