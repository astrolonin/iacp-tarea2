#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error at %s:%d — %s\n",                \
                         __FILE__, __LINE__, cudaGetErrorString(_e));         \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

struct Timer {
    cudaEvent_t start, stop;
    Timer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    ~Timer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void tic(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(start, stream));
    }
    void toc(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stop, stream));
    }
    float elapsed_ms() const {
        float ms;
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};
