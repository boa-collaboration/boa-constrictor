// Mamba Kernels - Conv1D, SSM, Gate operations for Mamba architecture
#include "gemm_gpu_common.cuh"

// ==========================================
// Single-Step Operations
// ==========================================

// Conv1d Step Kernel
// x_in: [d_inner] (current input)
// state: [d_inner, d_conv] (sliding window)
// weight: [d_inner, d_conv]
// bias: [d_inner]
// out: [d_inner]
__global__ void ker_mamba_conv1d(const float* x_in, float* state, const float* weight, const float* bias, float* out, int d_inner, int d_conv) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Channel index
    if (i < d_inner) {
        float* s = state + i * d_conv;
        const float* w = weight + i * d_conv;
        
        for (int k = 0; k < d_conv - 1; ++k) {
            s[k] = s[k+1];
        }
        s[d_conv-1] = x_in[i];
        
        float sum = bias[i];
        for (int k = 0; k < d_conv; ++k) {
            sum += s[k] * w[k];
        }
        
        #if GPU_REPRO_MATH
        out[i] = repro_silu(sum);
        #else
        float sigmoid = 1.0f / (1.0f + expf(-sum));
        out[i] = sum * sigmoid;
        #endif
    }
}

void gpu_mamba_conv1d(const float* x_in, float* state, const float* weight, const float* bias, float* out, int d_inner, int d_conv) {
    int block = 256;
    int grid = (d_inner + block - 1) / block;
    ker_mamba_conv1d<<<grid, block>>>(x_in, state, weight, bias, out, d_inner, d_conv);
    GPU_DEVICE_SYNC();
}

// SSM Step Kernel
__global__ void ker_mamba_ssm(const float* x, const float* dt, const float* A_log, const float* D, 
                              const float* B, const float* C, float* state, float* out, 
                              int d_inner, int d_state) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Channel index
    if (i < d_inner) {
        float dt_val = dt[i];
        float x_val = x[i];
        float D_val = D[i];
        float y_val = 0.0f;
        
        float* s = state + i * d_state;
        const float* a_log_row = A_log + i * d_state;
        
        for (int n = 0; n < d_state; ++n) {
            #if GPU_REPRO_MATH
            float A = -repro_exp(a_log_row[n]);
            float dA = repro_exp(A * dt_val);
            #else
            float A = -expf(a_log_row[n]);
            float dA = expf(A * dt_val);
            #endif
            float dB = dt_val * B[n];
             
            s[n] = s[n] * dA + dB * x_val;
            y_val += s[n] * C[n];
        }
        
        out[i] = y_val + D_val * x_val;
    }
}

void gpu_mamba_ssm(const float* x, const float* dt, const float* A_log, const float* D, 
                   const float* B, const float* C, float* state, float* out, 
                   int d_inner, int d_state) {
     int block = 256;
     int grid = (d_inner + block - 1) / block;
     ker_mamba_ssm<<<grid, block>>>(x, dt, A_log, D, B, C, state, out, d_inner, d_state);
    GPU_DEVICE_SYNC();
}

// DT Calculation + Softplus
__global__ void ker_add_bias_softplus(float* x, const float* b, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float val = x[i] + b[i];
        #if GPU_REPRO_MATH
        float e = repro_exp(val);
        #else
        float e = expf(val);
        #endif
        x[i] = logf(1.0f + e);
    }
}

void gpu_add_bias_softplus(float* x, const float* b, int N) {
    int block = 256;
    int grid = (N + block - 1) / block;
    ker_add_bias_softplus<<<grid, block>>>(x, b, N);
    GPU_DEVICE_SYNC();
}

// Gate kernel: y = y * silu(z)
__global__ void ker_gate(float* y, const float* z, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        #if GPU_REPRO_MATH
        float s_val = repro_silu(z[i]);
        y[i] = __fmul_rn(y[i], s_val);
        #else
        float sigmoid = 1.0f / (1.0f + expf(-z[i]));
        y[i] = y[i] * (z[i] * sigmoid);
        #endif
    }
}

void gpu_gate(float* y, const float* z, int N) {
    int block = 256;
    int grid = (N + block - 1) / block;
    ker_gate<<<grid, block>>>(y, z, N);
    GPU_DEVICE_SYNC();
}

// Gate kernel with strided z
__global__ void ker_gate_strided(float* y, const float* z, int d_inner, int z_stride, size_t total) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        size_t row = i / d_inner;
        size_t col = i % d_inner;
        float z_val = z[row * z_stride + col];
        #if GPU_REPRO_MATH
        float s_val = repro_silu(z_val);
        y[i] = __fmul_rn(y[i], s_val);
        #else
        float sigmoid = 1.0f / (1.0f + expf(-z_val));
        y[i] = y[i] * (z_val * sigmoid);
        #endif
    }
}

void gpu_gate_strided(float* y, const float* z, int d_inner, int z_stride, size_t total) {
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    ker_gate_strided<<<grid, block>>>(y, z, d_inner, z_stride, total);
    GPU_DEVICE_SYNC();
}

// ==========================================
// Batched Operations
// ==========================================

// Conv1d Batch
__global__ void ker_mamba_conv1d_batch(const float* x_in, float* state, const float* w, const float* b, float* out, int d_inner, int d_conv, int x_stride, size_t total_channels) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_channels) {
        size_t batch_idx = i / d_inner;
        int chan = i % d_inner;
        
        float* s = state + batch_idx * (d_inner * d_conv) + chan * d_conv;
        
        float4 s_vec = ((float4*)s)[0];
        
        // Shift
        s_vec.x = s_vec.y;
        s_vec.y = s_vec.z;
        s_vec.z = s_vec.w;
        s_vec.w = x_in[batch_idx * x_stride + chan];
        
        ((float4*)s)[0] = s_vec;
        
        float sum = b[chan];
        const float* w_chan = w + chan * d_conv;
        float4 w_vec = ((const float4*)w_chan)[0];
        
        sum = __fadd_rn(sum, __fmul_rn(s_vec.x, w_vec.x));
        sum = __fadd_rn(sum, __fmul_rn(s_vec.y, w_vec.y));
        sum = __fadd_rn(sum, __fmul_rn(s_vec.z, w_vec.z));
        sum = __fadd_rn(sum, __fmul_rn(s_vec.w, w_vec.w));
        
        #if GPU_REPRO_MATH
        out[i] = repro_silu(sum);
        #else
        float sigmoid = 1.0f / (1.0f + expf(-sum));
        out[i] = sum * sigmoid;
        #endif
    }
}

void gpu_mamba_conv1d_batch(const float* x_in, float* state, const float* weight, const float* bias, float* out, int d_inner, int d_conv, int x_stride, size_t batch) {
    size_t total = (size_t)batch * d_inner;
    ker_mamba_conv1d_batch<<<(total + 255) / 256, 256>>>(x_in, state, weight, bias, out, d_inner, d_conv, x_stride, total);
}

// SSM Batch
__global__ void ker_mamba_ssm_batch(const float* x_ptr, const float* dt_ptr, const float* A_log, const float* D, 
                                    const float* B_base_ptr, const float* C_base_ptr, float* state, float* out, 
                                    int d_inner, int d_state, int dt_rank, int stride_delta, size_t total_channels) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_channels) {
        size_t batch_idx = i / d_inner;
        int chan = i % d_inner;
        
        const float* batch_delta = B_base_ptr + batch_idx * stride_delta;
        const float* B_vec = batch_delta + dt_rank;
        const float* C_vec = batch_delta + dt_rank + d_state;
        
        float dt = dt_ptr[i];
        float x_val = x_ptr[i];
        
        float* s = state + batch_idx * (size_t)(d_inner * d_state) + (size_t)chan * d_state;
        const float* A_row = A_log + (size_t)chan * d_state;
        
        float y = 0.0f;
        #pragma unroll
        for(int m=0; m<d_state/4; ++m) {
            float4 s_vec = ((float4*)s)[m];
            float4 a_vec = ((const float4*)A_row)[m];
            float4 b_vec = ((const float4*)B_vec)[m];
            float4 c_vec = ((const float4*)C_vec)[m];
            
            float a[4] = {a_vec.x, a_vec.y, a_vec.z, a_vec.w};
            float b[4] = {b_vec.x, b_vec.y, b_vec.z, b_vec.w};
            float c[4] = {c_vec.x, c_vec.y, c_vec.z, c_vec.w};
            float sv[4] = {s_vec.x, s_vec.y, s_vec.z, s_vec.w};
            
            #pragma unroll
            for(int n=0; n<4; ++n) {
                #if GPU_REPRO_MATH
                float a_exp = -repro_exp(a[n]);
                float dt_a = __fmul_rn(a_exp, dt);
                float da = repro_exp(dt_a);
                #else
                float a_exp = -expf(a[n]);
                float da = expf(a_exp * dt);
                #endif
                float db = __fmul_rn(dt, b[n]);
                
                sv[n] = __fadd_rn(__fmul_rn(sv[n], da), __fmul_rn(db, x_val));
                y = __fadd_rn(y, __fmul_rn(sv[n], c[n]));
            }
            ((float4*)s)[m] = make_float4(sv[0], sv[1], sv[2], sv[3]);
        }
        
        float termD = __fmul_rn(x_val, D[chan]);
        y = __fadd_rn(y, termD);
        out[i] = y;
    }
}

void gpu_mamba_ssm_batch(const float* x, const float* dt, const float* A_log, const float* D, 
                         const float* B, const float* C, float* state, float* out, 
                         int d_inner, int d_state, int dt_rank, size_t batch) {
    size_t total = (size_t)batch * d_inner;
    int stride_delta = dt_rank + 2 * d_state;
    ker_mamba_ssm_batch<<<(total + 255) / 256, 256>>>(x, dt, A_log, D, B, C, state, out, d_inner, d_state, dt_rank, stride_delta, total);
}

// Fused Decompression Kernel
__global__ void ker_mamba_ssm_fused_batch(const float* x_conv, const float* x_z, const float* dt_logits, const float* dt_bias, 
                                          const float* A_log, const float* D, 
                                          const float* B_base_ptr, float* state, float* out, 
                                          int d_inner, int d_state, int dt_rank, int stride_delta, size_t total_channels) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_channels) {
        size_t batch_idx = i / d_inner;
        int chan = i % d_inner;
        
        // 1. Inputs
        float x_val = x_conv[i];
        float z_val = x_z[batch_idx * (2 * d_inner) + d_inner + chan];
        
        // 2. Softplus(dt)
        float dt_in = __fadd_rn(dt_logits[i], dt_bias[chan]);
        #if GPU_REPRO_MATH
        float dt_e = repro_exp(dt_in);
        #else
        float dt_e = expf(dt_in);
        #endif
        float dt = logf(__fadd_rn(1.0f, dt_e));
        
        // 3. BC extraction
        const float* batch_delta = B_base_ptr + batch_idx * (size_t)stride_delta;
        const float* B_vec = batch_delta + dt_rank;
        const float* C_vec = batch_delta + dt_rank + d_state;
        
        // 4. SSM
        float* s = state + batch_idx * (size_t)(d_inner * d_state) + (size_t)chan * d_state;
        const float* A_row = A_log + (size_t)chan * d_state;
        
        float y = 0.0f;
        #pragma unroll
        for(int m=0; m<d_state/4; ++m) {
            float4 s_vec = ((float4*)s)[m];
            float4 a_vec = ((const float4*)A_row)[m];
            float4 b_vec = ((const float4*)B_vec)[m];
            float4 c_vec = ((const float4*)C_vec)[m];
            
            float a[4] = {a_vec.x, a_vec.y, a_vec.z, a_vec.w};
            float b[4] = {b_vec.x, b_vec.y, b_vec.z, b_vec.w};
            float c[4] = {c_vec.x, c_vec.y, c_vec.z, c_vec.w};
            float sv[4] = {s_vec.x, s_vec.y, s_vec.z, s_vec.w};
            
            #pragma unroll
            for(int n=0; n<4; ++n) {
                #if GPU_REPRO_MATH
                float a_exp = -repro_exp(a[n]);
                float dt_a = __fmul_rn(a_exp, dt);
                float da = repro_exp(dt_a);
                #else
                float a_exp = -expf(a[n]);
                float da = expf(a_exp * dt);
                #endif
                float db = __fmul_rn(dt, b[n]);
                
                sv[n] = __fadd_rn(__fmul_rn(sv[n], da), __fmul_rn(db, x_val));
                y = __fadd_rn(y, __fmul_rn(sv[n], c[n]));
            }
            ((float4*)s)[m] = make_float4(sv[0], sv[1], sv[2], sv[3]);
        }
        
        float termD = __fmul_rn(x_val, D[chan]);
        y = __fadd_rn(y, termD);
        
        // 5. Gate with SiLU(z)
        #if GPU_REPRO_MATH
        float z_silu = repro_silu(z_val);
        #else
        float z_silu = z_val * (1.0f / (1.0f + expf(-z_val)));
        #endif
        
        out[i] = __fmul_rn(y, z_silu);
    }
}

void gpu_mamba_ssm_fused_batch(const float* x_conv, const float* x_z, const float* dt_logits, const float* dt_bias, 
                               const float* A_log, const float* D, 
                               const float* B_base_ptr, float* state, float* out, 
                               int d_inner, int d_state, int dt_rank, size_t batch) {
    size_t total = (size_t)batch * d_inner;
    int stride_delta = dt_rank + 2 * d_state;
    ker_mamba_ssm_fused_batch<<<(total + 255) / 256, 256>>>(x_conv, x_z, dt_logits, dt_bias, A_log, D, B_base_ptr, state, out, d_inner, d_state, dt_rank, stride_delta, total);
}

// ==========================================
// Chunk Operations
// ==========================================

__global__ void ker_mamba_conv1d_chunk(const float* x_in, float* state, const float* weight, const float* bias, float* out, 
                                       int Batch, int Length, int Dim, int Kernel, int x_stride, size_t total_elements) {
    size_t idx = (size_t)(blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int d = idx % Dim;
    int t = (idx / Dim) % Length;
    int b = (idx / Dim) / Length;

    float val = bias[d];
    
    for (int k = 0; k < Kernel; ++k) {
        int t_eff = t - (Kernel - 1) + k;
        float x_val = 0.0f;
        if (t_eff >= 0) {
            x_val = x_in[(size_t)(b * Length + t_eff) * x_stride + d];
        } else {
            int state_idx = (Kernel - 1) + t_eff;
             x_val = state[(size_t)(b * Dim + d) * Kernel + state_idx];
        }
        float term = __fmul_rn(x_val, weight[d * Kernel + k]);
        val = __fadd_rn(val, term);
    }
    #if GPU_REPRO_MATH
    float term_silu = repro_silu(val);
    out[idx] = term_silu;
    #else
    float sigmoid = 1.0f / (1.0f + expf(-val));
    out[idx] = val * sigmoid;
    #endif
    
    // State update logic
    if (t == Length - 1) {
       for (int k = 0; k < Kernel; ++k) {
           int t_idx = Length - Kernel + k;
           if (t_idx >= 0) {
               float x_val = x_in[(size_t)(b * Length + t_idx) * x_stride + d];
               state[(size_t)(b * Dim + d) * Kernel + k] = x_val;
           }
       }
    }
}

__global__ void ker_mamba_ssm_chunk(const float* u, const float* dt, const float* A_log, const float* D,
                                    const float* B_in, const float* C_in, 
                                    float* state, float* out,
                                    size_t Batch, int Length, int Dim, int State, size_t total_channels) {
    size_t idx = (size_t)(blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_channels) return;
    
    int d = idx % Dim;
    size_t b = idx / Dim;
    
    float h[16]; 
    for(int i=0; i<State; ++i) h[i] = state[(b * Dim + d) * State + i];
    
    float D_val = D[d];
    
    for (int t = 0; t < Length; ++t) {
        size_t flat_idx = (b * Length + t) * Dim + d;
        float u_val = u[flat_idx];
        float dt_val = dt[flat_idx]; 
        
        float y_val = 0.0f;
        for (int n = 0; n < State; ++n) {
            #if GPU_REPRO_MATH
            float a = -repro_exp(A_log[d * State + n]);
            float dt_a = __fmul_rn(a, dt_val);
            float dA = repro_exp(dt_a);
            #else
            float a = -expf(A_log[d * State + n]);
            float dA = expf(a * dt_val);
            #endif
            float B_val = B_in[(b * Length + t) * State + n];
            float dB = __fmul_rn(dt_val, B_val);
            
            float term1 = __fmul_rn(dA, h[n]);
            float term2 = __fmul_rn(dB, u_val);
            h[n] = __fadd_rn(term1, term2);
            
            float C_val = C_in[(b * Length + t) * State + n];
            float term3 = __fmul_rn(h[n], C_val);
            y_val = __fadd_rn(y_val, term3);
        }
        
        float termD = __fmul_rn(D_val, u_val);
        y_val = __fadd_rn(y_val, termD);
        out[flat_idx] = y_val;
    }
    
    for(int i=0; i<State; ++i) state[(b * Dim + d) * State + i] = h[i];
}

void gpu_mamba_conv1d_chunk(const float* x, float* state, const float* weight, const float* bias, float* out, 
                            int Batch, int Length, int Dim, int Kernel, int x_stride) {
    size_t total = (size_t)Batch * Length * Dim;
    int block = 256;
    size_t grid_x_long = (total + block - 1) / block;
    int grid_x = (int)grid_x_long;
    int grid_y = 1;
    const int max_grid_x = 2147483647;
    if (grid_x_long > max_grid_x) {
        grid_y = (int)((grid_x_long + max_grid_x - 1) / max_grid_x);
        grid_x = max_grid_x;
    }
    dim3 grid(grid_x, grid_y, 1);
    ker_mamba_conv1d_chunk<<<grid, block>>>(x, state, weight, bias, out, Batch, Length, Dim, Kernel, x_stride, total);
    checkCudaErrors(cudaGetLastError());
}

void gpu_mamba_ssm_chunk(const float* u, const float* dt, const float* A_log, const float* D,
                         const float* B, const float* C, float* state, float* out,
                         size_t Batch, int Length, int Dim, int State) {
    size_t total = Batch * Dim; 
    int block = 256;
    size_t grid_x_long = (total + block - 1) / block;
    int grid_x = (int)grid_x_long;
    int grid_y = 1;
    const int max_grid_x = 2147483647;
    if (grid_x_long > max_grid_x) {
        grid_y = (int)((grid_x_long + max_grid_x - 1) / max_grid_x);
        grid_x = max_grid_x;
    }
    dim3 grid(grid_x, grid_y, 1);
    ker_mamba_ssm_chunk<<<grid, block>>>(u, dt, A_log, D, B, C, state, out, Batch, Length, Dim, State, total);
    checkCudaErrors(cudaGetLastError());
}
