#pragma once
// Common includes, macros, and device helpers for GPU kernels

#include "gemm_gpu.hpp"
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdint>
#include <cmath>
#include "activations.hpp"
#include <algorithm>
#include <vector>

// ==========================================
// GPU Math Configuration
// ==========================================

#ifndef GPU_REPRO_MATH
#define GPU_REPRO_MATH 1
#endif

#ifndef GPU_FAST_EXP
#define GPU_FAST_EXP 0
#endif

#ifndef GPU_SOFTMAX_LUT
#define GPU_SOFTMAX_LUT 0
#endif

// ==========================================
// Device Helper Functions
// ==========================================

__device__ inline float gpu_exp(float x) {
#if GPU_FAST_EXP
    return __expf(x);
#else
    return repro_exp(x);
#endif
}

// ==========================================
// Softmax LUT (optional)
// ==========================================

#if GPU_SOFTMAX_LUT
static constexpr int EXP_LUT_SIZE = 4097;
static constexpr float EXP_LUT_MIN = -16.0f;
static constexpr float EXP_LUT_MAX = 0.0f;
static constexpr float EXP_LUT_STEP = (EXP_LUT_MAX - EXP_LUT_MIN) / (EXP_LUT_SIZE - 1);
static constexpr float EXP_LUT_INV_STEP = 1.0f / EXP_LUT_STEP;

__device__ __constant__ float g_exp_lut[EXP_LUT_SIZE];

inline void gpu_init_exp_lut_impl() {
    static bool initialized = false;
    if (initialized) return;
    std::vector<float> host(EXP_LUT_SIZE);
    for (int i = 0; i < EXP_LUT_SIZE; ++i) {
        float x = EXP_LUT_MIN + EXP_LUT_STEP * (float)i;
        host[i] = repro_exp(x);
    }
    checkCudaErrors(cudaMemcpyToSymbol(g_exp_lut, host.data(), EXP_LUT_SIZE * sizeof(float)));
    initialized = true;
}

__device__ inline float gpu_exp_lut(float x) {
    if (x <= EXP_LUT_MIN) return 0.0f;
    if (x >= 0.0f) return 1.0f;
    float t = (x - EXP_LUT_MIN) * EXP_LUT_INV_STEP;
    int idx = (int)t;
    if (idx < 0) idx = 0;
    if (idx >= EXP_LUT_SIZE - 1) idx = EXP_LUT_SIZE - 2;
    float frac = t - (float)idx;
    float a = g_exp_lut[idx];
    float b = g_exp_lut[idx + 1];
    return __fmaf_rn(frac, (b - a), a);
}
#else
inline void gpu_init_exp_lut_impl() {}
__device__ inline float gpu_exp_lut(float x) { return gpu_exp(x); }
#endif

// ==========================================
// Warp Reduction Primitives
// ==========================================

__inline__ __device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__inline__ __device__ float warpScanSum(float val) {
    float temp = 0.0f;
    #pragma unroll
    for (int offset = 1; offset <= 16; offset *= 2) {
        temp = __shfl_up_sync(0xFFFFFFFF, val, offset); 
        if (threadIdx.x % 32 >= offset) val += temp;
    }
    return val;
}

__inline__ __device__ double warpReduceSumDouble(double val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__inline__ __device__ double warpScanSumDouble(double val) {
    double temp = 0.0;
    #pragma unroll
    for (int offset = 1; offset <= 16; offset *= 2) {
        temp = __shfl_up_sync(0xFFFFFFFF, val, offset); 
        if (threadIdx.x % 32 >= offset) val += temp;
    }
    return val;
}
