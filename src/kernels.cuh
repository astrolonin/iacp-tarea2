#pragma once

#include <cuda_runtime.h>

constexpr int TILE_SIZE    = 32;
constexpr int REDUCE_BLOCK = 256;

// ---------------------------------------------------------------------------
// Kernel 0 — Zero out an n×n matrix
// ---------------------------------------------------------------------------
__global__ void zero_matrix_kernel(float* mat, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < n) mat[(size_t)i * n + j] = 0.0f;
}

// ---------------------------------------------------------------------------
// Kernel 1 — Reduce mean vector across images
//   data    : [m x n] row-major, read-only
//   mean    : [n]       output (sum, not yet divided by m)
//   m, n
// Each block handles one chunk of n. Threads stride over m.
// ---------------------------------------------------------------------------
__global__ void reduce_mean_kernel(const float* __restrict__ data,
                                   float* __restrict__ mean,
                                   int m, int n) {
    extern __shared__ float sh[];
    int tid = threadIdx.x;
    int col = blockIdx.x * blockDim.x + tid;
    if (col >= n) return;

    float acc = 0.0f;
    for (int k = 0; k < m; ++k)
        acc += data[(size_t)k * n + col];

    sh[tid] = acc;           // one value per thread — no intra-block reduction needed
    __syncthreads();

    if (tid == 0) mean[col] = sh[0];
    // Actually each thread already has the full sum for its column.
    // The shared memory sync is just to ensure all threads finish before write.
    // We only need one thread to write, but we write per-thread for simplicity.
    // Overwrite with tid==0 is cleaner:
}

// Simpler version without unnecessary shared memory:
__global__ void reduce_mean_kernel_v2(const float* __restrict__ data,
                                      float* __restrict__ mean,
                                      int m, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;
    float acc = 0.0f;
    for (int k = 0; k < m; ++k)
        acc += data[(size_t)k * n + col];
    mean[col] = acc;
}

// ---------------------------------------------------------------------------
// Kernel 2 — Accumulate sum of outer products Σ v_k · v_k^T  (tiled)
//   data       : [m x n] row-major, read-only
//   cov        : [n x n] row-major, accumulated in-place
//   start_img  : first image index to process
//   num_imgs   : how many images this invocation covers
//   n          : flattened image size
//
// Grid: (ceil(n/TILE), ceil(n/TILE))
// Block: (TILE_SIZE, TILE_SIZE)
// Shared memory: 2 * TILE_SIZE floats
// ---------------------------------------------------------------------------
__global__ void cov_accumulate_tiled_kernel(const float* __restrict__ data,
                                            float* __restrict__ cov,
                                            int start_img,
                                            int num_imgs,
                                            int n) {
    __shared__ float col_A[TILE_SIZE];
    __shared__ float col_B[TILE_SIZE];

    int bx = blockIdx.x * TILE_SIZE;
    int by = blockIdx.y * TILE_SIZE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_c = by + ty;   // row    index into C
    int col_c = bx + tx;   // column index into C
    bool valid_row = (row_c < n);
    bool valid_col = (col_c < n);

    float acc = 0.0f;
    int end_img = start_img + num_imgs;

    for (int k = start_img; k < end_img; ++k) {
        const float* img = data + (size_t)k * n;

        if (valid_col && ty == 0) col_A[tx] = img[bx + tx];
        if (valid_row && tx == 0) col_B[ty] = img[by + ty];
        __syncthreads();

        if (valid_row && valid_col)
            acc += col_B[ty] * col_A[tx];
        __syncthreads();
    }

    if (valid_row && valid_col)
        cov[(size_t)row_c * n + col_c] += acc;
}

// ---------------------------------------------------------------------------
// Kernel 3a — Scale a vector in-place: vec[i] *= scale
// ---------------------------------------------------------------------------
__global__ void scale_vector_kernel(float* vec, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) vec[i] *= scale;
}

// ---------------------------------------------------------------------------
// Kernel 3b — Post-process covariance: C = C / m - μ · μ^T
//   cov  : [n x n] accumulated sum of outer products Σ vv^T
//   mean : [n]     true average (already divided by m)
// ---------------------------------------------------------------------------
__global__ void postprocess_cov_kernel(float* __restrict__ cov,
                                       const float* __restrict__ mean,
                                       float inv_m,
                                       int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || j >= n) return;
    cov[(size_t)i * n + j] = cov[(size_t)i * n + j] * inv_m
                             - mean[i] * mean[j];
}

// ---------------------------------------------------------------------------
// Kernel 4 — Accumulate partial mean for a batch of images
//   Each block: one column chunk, stride over batch images
// ---------------------------------------------------------------------------
__global__ void partial_mean_kernel(const float* __restrict__ data,
                                    float* __restrict__ partial,
                                    int start_img,
                                    int num_imgs,
                                    int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;
    float acc = 0.0f;
    int end_img = start_img + num_imgs;
    for (int k = start_img; k < end_img; ++k)
        acc += data[(size_t)k * n + col];
    partial[col] = acc;
}

// ---------------------------------------------------------------------------
// Kernel 5 — Reduce S partial mean arrays into a single global mean
//   partials : [S x n] row-major
//   global_mean : [n]
// ---------------------------------------------------------------------------
__global__ void reduce_partial_means_kernel(const float* __restrict__ partials,
                                            float* __restrict__ global_mean,
                                            int S, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;
    float acc = 0.0f;
    for (int s = 0; s < S; ++s)
        acc += partials[(size_t)s * n + col];
    global_mean[col] = acc;
}
