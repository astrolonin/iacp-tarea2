#pragma once

#include "kernels.cuh"
#include "utils.cuh"
#include <cstdio>
#include <algorithm>

struct Exp2Result {
    float ms_total;
    int   num_batches;
    int   batch_size;
};

inline Exp2Result run_experiment2(const float* h_dataset, int m, int n,
                                   int S) {
    Exp2Result res{};
    S = std::max(1, std::min(S, m));
    int batch_size = std::max(1, (m + 2 * S - 1) / (2 * S));
    int B = (m + batch_size - 1) / batch_size;
    res.num_batches = B;
    res.batch_size  = batch_size;

    std::printf("[EXP2] S=%d streams, B=%d batches, batch_size=%d\n",
                S, B, batch_size);

    size_t batch_bytes   = (size_t)batch_size * n * sizeof(float);
    size_t mean_bytes    = (size_t)n * sizeof(float);
    size_t cov_bytes     = (size_t)n * n * sizeof(float);

    cudaStream_t*  streams       = new cudaStream_t[S];
    float**        d_batch       = new float*[S * 2];
    float*         d_partial_sum = nullptr;
    float*         d_cov         = nullptr;
    float*         d_mean        = nullptr;
    cudaEvent_t*   kernel_done   = new cudaEvent_t[B];

    for (int s = 0; s < S; ++s) {
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
        CUDA_CHECK(cudaMalloc(&d_batch[s * 2 + 0], batch_bytes));
        CUDA_CHECK(cudaMalloc(&d_batch[s * 2 + 1], batch_bytes));
    }
    CUDA_CHECK(cudaMalloc(&d_partial_sum, (size_t)B * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cov,   cov_bytes));
    CUDA_CHECK(cudaMalloc(&d_mean,  mean_bytes));
    for (int b = 0; b < B; ++b) CUDA_CHECK(cudaEventCreate(&kernel_done[b]));

    // Pre-allocate pinned host buffer for C (outside timer)
    float* h_cov;
    CUDA_CHECK(cudaMallocHost(&h_cov, cov_bytes));

    {
        dim3 bz(TILE_SIZE, TILE_SIZE);
        dim3 gz((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);
        zero_matrix_kernel<<<gz, bz>>>(d_cov, n);
    }

    Timer total_timer;
    total_timer.tic();

    // --- Streaming pipeline ---
    for (int b = 0; b < B; ++b) {
        int s       = b % S;
        int round   = b / S;
        int buf_idx = round % 2;

        int start_k = b * batch_size;
        int cur_bs  = std::min(batch_size, m - start_k);
        size_t cur_bytes = (size_t)cur_bs * n * sizeof(float);

        CUDA_CHECK(cudaMemcpyAsync(
            d_batch[s * 2 + buf_idx],
            h_dataset + (size_t)start_k * n,
            cur_bytes,
            cudaMemcpyHostToDevice,
            streams[s]));

        if (b > 0)
            CUDA_CHECK(cudaStreamWaitEvent(streams[s], kernel_done[b - 1], 0));

        {
            int grid = (n + REDUCE_BLOCK - 1) / REDUCE_BLOCK;
            partial_mean_kernel<<<grid, REDUCE_BLOCK, 0, streams[s]>>>(
                d_batch[s * 2 + buf_idx],
                d_partial_sum + (size_t)b * n,
                0, cur_bs, n);
        }

        {
            dim3 bz(TILE_SIZE, TILE_SIZE);
            dim3 gz((n + TILE_SIZE - 1) / TILE_SIZE,
                    (n + TILE_SIZE - 1) / TILE_SIZE);
            cov_accumulate_tiled_kernel<<<gz, bz, 0, streams[s]>>>(
                d_batch[s * 2 + buf_idx], d_cov, 0, cur_bs, n);
        }

        CUDA_CHECK(cudaEventRecord(kernel_done[b], streams[s]));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Post-processing ---
    {
        int grid = (n + REDUCE_BLOCK - 1) / REDUCE_BLOCK;
        reduce_partial_means_kernel<<<grid, REDUCE_BLOCK>>>(
            d_partial_sum, d_mean, B, n);
    }
    {
        int grid = (n + REDUCE_BLOCK - 1) / REDUCE_BLOCK;
        scale_vector_kernel<<<grid, REDUCE_BLOCK>>>(d_mean, 1.0f / m, n);
    }
    {
        dim3 bz(TILE_SIZE, TILE_SIZE);
        dim3 gz((n + TILE_SIZE - 1) / TILE_SIZE,
                (n + TILE_SIZE - 1) / TILE_SIZE);
        postprocess_cov_kernel<<<gz, bz>>>(d_cov, d_mean, 1.0f / m, n);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // --- D → H (covariance matrix) ---
    CUDA_CHECK(cudaMemcpy(h_cov, d_cov, cov_bytes, cudaMemcpyDeviceToHost));

    total_timer.toc();
    CUDA_CHECK(cudaDeviceSynchronize());
    res.ms_total = total_timer.elapsed_ms();

    // --- Cleanup ---
    CUDA_CHECK(cudaFreeHost(h_cov));
    for (int b = 0; b < B; ++b) CUDA_CHECK(cudaEventDestroy(kernel_done[b]));
    for (int s = 0; s < S; ++s) {
        CUDA_CHECK(cudaStreamDestroy(streams[s]));
        CUDA_CHECK(cudaFree(d_batch[s * 2 + 0]));
        CUDA_CHECK(cudaFree(d_batch[s * 2 + 1]));
    }
    CUDA_CHECK(cudaFree(d_partial_sum));
    CUDA_CHECK(cudaFree(d_cov));
    CUDA_CHECK(cudaFree(d_mean));
    delete[] streams;
    delete[] d_batch;
    delete[] kernel_done;

    std::printf("[EXP2] Total: %.3f ms (S=%d, B=%d)\n",
                res.ms_total, S, B);
    return res;
}
