#pragma once

#include "kernels.cuh"
#include "utils.cuh"
#include <cstdio>

struct Exp1Result {
    float ms_h2d;
    float ms_kernels;
    float ms_d2h;
    float ms_total;
};

inline Exp1Result run_experiment1(const float* h_dataset, int m, int n) {
    Exp1Result res{};
    Timer timer;

    size_t data_bytes = (size_t)m * n * sizeof(float);
    size_t cov_bytes  = (size_t)n * n * sizeof(float);
    size_t mean_bytes = (size_t)n * sizeof(float);

    // Device allocations
    float *d_data, *d_mean, *d_cov;
    CUDA_CHECK(cudaMalloc(&d_data, data_bytes));
    CUDA_CHECK(cudaMalloc(&d_mean, mean_bytes));
    CUDA_CHECK(cudaMalloc(&d_cov,  cov_bytes));

    // Zero-init covariance matrix
    {
        dim3 bz(TILE_SIZE, TILE_SIZE);
        dim3 gz((n + TILE_SIZE - 1) / TILE_SIZE,
                (n + TILE_SIZE - 1) / TILE_SIZE);
        zero_matrix_kernel<<<gz, bz>>>(d_cov, n);
    }

    // --- H → D (dataset) ---
    timer.tic();
    CUDA_CHECK(cudaMemcpy(d_data, h_dataset, data_bytes, cudaMemcpyHostToDevice));
    timer.toc();
    CUDA_CHECK(cudaDeviceSynchronize());
    res.ms_h2d = timer.elapsed_ms();

    // --- Kernels ---
    timer.tic();

    // Kernel 1: reduce_mean (sum, not average yet)
    {
        int grid = (n + REDUCE_BLOCK - 1) / REDUCE_BLOCK;
        reduce_mean_kernel_v2<<<grid, REDUCE_BLOCK>>>(d_data, d_mean, m, n);
    }

    // Kernel 2: accumulate covariance (all m images at once)
    {
        dim3 bz(TILE_SIZE, TILE_SIZE);
        dim3 gz((n + TILE_SIZE - 1) / TILE_SIZE,
                (n + TILE_SIZE - 1) / TILE_SIZE);
        cov_accumulate_tiled_kernel<<<gz, bz>>>(d_data, d_cov, 0, m, n);
    }

    // Convert mean sum → true average: d_mean *= 1/m
    {
        int grid = (n + REDUCE_BLOCK - 1) / REDUCE_BLOCK;
        scale_vector_kernel<<<grid, REDUCE_BLOCK>>>(d_mean, 1.0f / m, n);
    }

    // Kernel 3: post-process C = C/m - μ·μ^T
    {
        dim3 bz(TILE_SIZE, TILE_SIZE);
        dim3 gz((n + TILE_SIZE - 1) / TILE_SIZE,
                (n + TILE_SIZE - 1) / TILE_SIZE);
        postprocess_cov_kernel<<<gz, bz>>>(d_cov, d_mean, 1.0f / m, n);
    }

    timer.toc();
    CUDA_CHECK(cudaDeviceSynchronize());
    res.ms_kernels = timer.elapsed_ms();

    // Host buffer for receiving C (allocated outside timer)
    float* h_cov;
    CUDA_CHECK(cudaMallocHost(&h_cov, cov_bytes));

    // --- D → H (covariance matrix) ---
    timer.tic();
    CUDA_CHECK(cudaMemcpy(h_cov, d_cov, cov_bytes, cudaMemcpyDeviceToHost));
    timer.toc();
    CUDA_CHECK(cudaDeviceSynchronize());
    res.ms_d2h = timer.elapsed_ms();

    res.ms_total = res.ms_h2d + res.ms_kernels + res.ms_d2h;

    CUDA_CHECK(cudaFreeHost(h_cov));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_mean));
    CUDA_CHECK(cudaFree(d_cov));

    std::printf("[EXP1] H→D: %.3f ms | Kernels: %.3f ms | D→H: %.3f ms | Total: %.3f ms\n",
                res.ms_h2d, res.ms_kernels, res.ms_d2h, res.ms_total);
    return res;
}
