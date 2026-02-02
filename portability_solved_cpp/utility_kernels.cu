// Utility Kernels - Copy, embedding lookup, and memory helpers
#include "gemm_gpu_common.cuh"

// ==========================================
// Copy Operations
// ==========================================

__global__ void ker_copy_2d(const float* src, float* dst, int W, int N, int S1, int S2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N * W) {
        int r = i / W;
        int c = i % W;
        dst[r * S2 + c] = src[r * S1 + c];
    }
}

void gpu_copy_2d(const float* src, float* dst, int W, int N, int S1, int S2) {
    ker_copy_2d<<<(N*W+255)/256, 256>>>(src, dst, W, N, S1, S2);
    GPU_DEVICE_SYNC();
}

__global__ void ker_copy_strided(const float* src, float* dst, int src_stride, int width, size_t total_elements) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_elements) {
        size_t b = i / width;
        size_t w = i % width;
        dst[i] = src[b * src_stride + w];
    }
}

void gpu_copy_strided(const float* src, float* dst, int src_stride, int width, size_t batch) {
    size_t total = batch * width;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    ker_copy_strided<<<grid, block>>>(src, dst, src_stride, width, total);
    GPU_DEVICE_SYNC();
}

// ==========================================
// Batched Embedding Lookup
// ==========================================

__global__ void ker_embedding_lookup_batch(const int* tokens, const float* embedding, float* out, int d_model, size_t total_elements) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_elements) {
        size_t b = i / d_model;
        int d = i % d_model;
        int token = tokens[b];
        out[i] = embedding[token * d_model + d];
    }
}

void gpu_embedding_lookup_batch(const int* tokens, const float* embedding, float* out, int d_model, size_t batch) {
    size_t total = batch * d_model;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    ker_embedding_lookup_batch<<<grid, block>>>(tokens, embedding, out, d_model, total);
    GPU_DEVICE_SYNC();
}

// ==========================================
// Memory Helpers
// ==========================================

void malloc_device(float** ptr, size_t bytes) {
    checkCudaErrors(cudaMalloc(ptr, bytes));
}

void free_device(float* ptr) {
    checkCudaErrors(cudaFree(ptr));
}

void copy_to_device(float* d_ptr, const float* h_ptr, size_t bytes) {
    checkCudaErrors(cudaMemcpy(d_ptr, h_ptr, bytes, cudaMemcpyHostToDevice));
}

void copy_to_host(float* h_ptr, const float* d_ptr, size_t bytes) {
    checkCudaErrors(cudaMemcpy(h_ptr, d_ptr, bytes, cudaMemcpyDeviceToHost));
}

// ==========================================
// VRAM Info
// ==========================================

void get_gpu_vram_info(size_t& free_bytes, size_t& total_bytes) {
    cudaMemGetInfo(&free_bytes, &total_bytes);
}

// ==========================================
// Exp LUT Initialization
// ==========================================

void gpu_init_exp_lut() {
    gpu_init_exp_lut_impl();
}
