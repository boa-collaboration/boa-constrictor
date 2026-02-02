#pragma once

#include "gemm_gpu.hpp"
#include <vector>
#include <fstream>
#include <iostream>

// GPU Structures

// Helper for transposing weights (Out, In) -> (In, Out) for RowMajor GEMM
std::vector<float> transpose_weights(const std::vector<float>& w, int rows, int cols) {
    std::vector<float> t(w.size());
    for(int i=0; i<rows; ++i) {
        for(int j=0; j<cols; ++j) {
            t[j*rows + i] = w[i*cols + j];
        }
    }
    return t;
}

struct LayerNormGPU {
    float* weight;
    float* bias;
    
    LayerNormGPU() = default;
    
    void allocate(int size) {
        malloc_device(&weight, size * sizeof(float));
        malloc_device(&bias, size * sizeof(float));
    }
    
    void free() {
        free_device(weight); free_device(bias);
    }
    
    void load_weights(std::ifstream& f) {
        std::vector<float> tmp(256); // Assuming d_model known or read?
        // Actually load_weights needs size.
        // We assume caller knows context or we read generic.
        // For simplicity: hardcoded based on boa_model.hpp usage calling convention?
        // No, caller logic in BoaBlockGPU calls load_weights with implicit size in mind.
        // Let's rely on MambaConfig?
    }
    
    // Better: load_vec helper in parent
};




// Helper to read vector and copy (alloc not needed) to existing
inline void load_vec(std::ifstream& f, float* d_ptr, int size, bool transpose=false, int rows=0, int cols=0) {
    std::vector<float> tmp(size);
    f.read(reinterpret_cast<char*>(tmp.data()), size * sizeof(float));
    if (transpose) {
        auto t = transpose_weights(tmp, rows, cols); // utils.hpp
        copy_to_device(d_ptr, t.data(), size * sizeof(float));
    } else {
        copy_to_device(d_ptr, tmp.data(), size * sizeof(float));
    }
}

struct MambaBlockGPU {
    MambaConfig config;
    int batch_size;
    
    // Weights (Shared)
    float* in_proj_w; 
    float* in_proj_b;
    float* conv1d_w;
    float* conv1d_b;
    float* x_proj_w;
    float* dt_proj_w;
    float* dt_proj_b;
    float* A_log;
    float* D;
    float* out_proj_w;
    float* out_proj_b;
    float* norm_w; // For RMSNorm
    
    // State (Per Batch)
    float* conv_state; // [Batch, d_inner, d_conv]
    float* ssm_state;  // [Batch, d_inner, d_state]
    
    // Buffers (Per Batch)
    float* buf_xz;     // [Batch, 2*d_inner]
    float* buf_x;      // Alias
    float* buf_z;      // Alias
    float* buf_conv;   // [Batch, d_inner]
    float* buf_delta;  // [Batch, dt_rank + 2*d_state] (Row Major)
    float* buf_dt_in;  // [Batch, dt_rank] (Extracted)
    float* buf_dt;     // [Batch, d_inner]
    float* buf_y;      // [Batch, d_inner]
    
    MambaBlockGPU() = default;

    void allocate(MambaConfig c, int batch) {
        config = c;
        batch_size = batch;
        
        // Weights (One copy)
        malloc_device(&in_proj_w, 2 * config.d_inner * config.d_model * sizeof(float));
        malloc_device(&in_proj_b, 2 * config.d_inner * sizeof(float));
        malloc_device(&conv1d_w, config.d_inner * config.d_conv * sizeof(float));
        malloc_device(&conv1d_b, config.d_inner * sizeof(float));
        malloc_device(&x_proj_w, (config.dt_rank + 2 * config.d_state) * config.d_inner * sizeof(float));
        malloc_device(&dt_proj_w, config.d_inner * config.dt_rank * sizeof(float));
        malloc_device(&dt_proj_b, config.d_inner * sizeof(float));
        malloc_device(&A_log, config.d_inner * config.d_state * sizeof(float));
        malloc_device(&D, config.d_inner * sizeof(float));
        malloc_device(&out_proj_w, config.d_model * config.d_inner * sizeof(float));
        malloc_device(&out_proj_b, config.d_model * sizeof(float));
        if(config.use_rmsnorm) malloc_device(&norm_w, config.d_model * sizeof(float));
        else malloc_device(&norm_w, 1);
        
        // State (Batch * Size)
        malloc_device(&conv_state, (size_t)batch_size * config.d_inner * config.d_conv * sizeof(float));
        malloc_device(&ssm_state, (size_t)batch_size * config.d_inner * config.d_state * sizeof(float));
        
        // Buffers
        malloc_device(&buf_xz, (size_t)batch_size * 2 * config.d_inner * sizeof(float));
        buf_x = buf_xz; 
        // buf_z = buf_xz + d_inner (Per batch slice).
        // Wait, buf_xz is [Batch, 2*d_inner]. 
        // Row 0: [x0, z0]. Row 1: [x1, z1].
        // Split pointers? No. 
        // buf_x points to start.
        // buf_z needs to point to... wait.
        // If we split via striding, buf_z is not contiguous block.
        // SOLUTION: Split projection into `in_proj_x` and `in_proj_z`?
        // Or separate buffers `buf_x` and `buf_z`.
        // Let's allocate separate.
        // in_proj normally produces one [2*d] output.
        // If I keep it one [Batch, 2d], accessing x part is strided!
        // `gemm_gpu_batch` produces [Batch, 2d].
        // `conv1d` reads `x`.
        // I need Contiguous `x`.
        // I will copy `buf_xz` (strided) to `buf_x` and `buf_z` (contiguous)?
        // Or implement `split_kernel`?
        // Using `split_kernel` is best.
        
        malloc_device(&buf_x, (size_t)batch_size * config.d_inner * sizeof(float));
        malloc_device(&buf_z, (size_t)batch_size * config.d_inner * sizeof(float));
        
        malloc_device(&buf_conv, (size_t)batch_size * config.d_inner * sizeof(float));
        
        int stride_delta = config.dt_rank + 2 * config.d_state;
        malloc_device(&buf_delta, (size_t)batch_size * stride_delta * sizeof(float));
        malloc_device(&buf_dt_in, (size_t)batch_size * config.dt_rank * sizeof(float));
        
        malloc_device(&buf_dt, (size_t)batch_size * config.d_inner * sizeof(float));
        malloc_device(&buf_y, (size_t)batch_size * config.d_inner * sizeof(float));
        
        reset_cache();
    }
    
    void free() {
        free_device(in_proj_w); free_device(in_proj_b);
        free_device(conv1d_w); free_device(conv1d_b);
        free_device(x_proj_w); free_device(dt_proj_w); free_device(dt_proj_b);
        free_device(A_log); free_device(D);
        free_device(out_proj_w); free_device(out_proj_b);
        free_device(norm_w);
        free_device(conv_state); free_device(ssm_state);
        free_device(buf_xz); free_device(buf_x); free_device(buf_z);
        free_device(buf_conv); free_device(buf_delta); free_device(buf_dt_in); free_device(buf_dt); free_device(buf_y);
    }
    

    
     // Helper to read vector and copy (alloc not needed) to existing

    void load_weights(std::ifstream& f) {
        load_vec(f, in_proj_w, 2*config.d_inner * config.d_model, true, 2*config.d_inner, config.d_model);
        load_vec(f, in_proj_b, 2*config.d_inner); 
        load_vec(f, conv1d_w, config.d_inner * config.d_conv); 
        load_vec(f, conv1d_b, config.d_inner);
        load_vec(f, x_proj_w, (config.dt_rank + 2*config.d_state) * config.d_inner, true, (config.dt_rank + 2*config.d_state), config.d_inner);
        load_vec(f, dt_proj_w, config.d_inner * config.dt_rank, true, config.d_inner, config.dt_rank);
        load_vec(f, dt_proj_b, config.d_inner);
        load_vec(f, A_log, config.d_inner * config.d_state);
        load_vec(f, D, config.d_inner);
        load_vec(f, out_proj_w, config.d_model * config.d_inner, true, config.d_model, config.d_inner);
        load_vec(f, out_proj_b, config.d_model);
        if(config.use_rmsnorm) load_vec(f, norm_w, config.d_model);
    }
    
    void step_batch(float* x_in, float* x_out) {
        #if GPU_DEBUG_LOGITS
        if (batch_size > 0) {
            std::vector<float> h_d(5);
            cudaMemcpy(h_d.data(), x_in, 5*sizeof(float), cudaMemcpyDeviceToHost);
            printf("STEP X_IN[0-4]: %f %f %f %f %f\n", h_d[0], h_d[1], h_d[2], h_d[3], h_d[4]);
        }
        #endif
        
        // 1. In Proj
        gemm_gpu_batch(x_in, in_proj_w, buf_xz, batch_size, 2*config.d_inner, config.d_model);
        gpu_add_bias(buf_xz, in_proj_b, batch_size, 2*config.d_inner);
        
        // 2. Optimized Conv1D (Direct from buf_xz)
        gpu_mamba_conv1d_batch(buf_xz, conv_state, conv1d_w, conv1d_b, buf_conv, config.d_inner, config.d_conv, 2*config.d_inner, batch_size);
        
        // 3. x_proj
        int stride_delta = config.dt_rank + 2*config.d_state;
        gemm_gpu_batch(buf_conv, x_proj_w, buf_delta, batch_size, stride_delta, config.d_inner);
        
        // 4. dt_proj (Strided access from buf_delta, skip copy)
        gemm_gpu_batch_strided(buf_delta, stride_delta, dt_proj_w, config.d_inner, buf_dt, config.d_inner, batch_size, config.d_inner, config.dt_rank);
        
        // 5. Fused SSM Tail: Softplus(logits) + SSM(x) + Gating(z)
        // Passes buf_conv for x, buf_xz directly for z extraction.
        gpu_mamba_ssm_fused_batch(buf_conv, buf_xz, buf_dt, dt_proj_b, A_log, D, buf_delta, ssm_state, buf_y, config.d_inner, config.d_state, config.dt_rank, batch_size);
        
        // 6. Out Proj
        gemm_gpu_batch(buf_y, out_proj_w, x_out, batch_size, config.d_model, config.d_inner);
        gpu_add_bias(x_out, out_proj_b, batch_size, config.d_model);
    }

    void reset_cache() {
        checkCudaErrors(cudaMemset(conv_state, 0, (size_t)batch_size * config.d_inner * config.d_conv * sizeof(float)));
        checkCudaErrors(cudaMemset(ssm_state, 0, (size_t)batch_size * config.d_inner * config.d_state * sizeof(float)));
    }

    // Chunk Buffers
    float* chunk_buf_xz;     
    // float* chunk_buf_x; sent to void
    // float* chunk_buf_z; sent to void
    float* chunk_buf_conv;   
    float* chunk_buf_delta;  
    float* chunk_buf_dt_in;  
    float* chunk_buf_dt;     
    float* chunk_buf_y;
    float* chunk_buf_B;
    float* chunk_buf_C;
    
    int max_chunk_len_alloc = 0;

    void allocate_chunk(int length) {
        if (max_chunk_len_alloc >= length) return;
        if (max_chunk_len_alloc > 0) free_chunk();
        
        max_chunk_len_alloc = length;
        size_t total_tokens = (size_t)batch_size * length;
        
        malloc_device(&chunk_buf_xz, total_tokens * 2 * config.d_inner * sizeof(float));
        // malloc_device(&chunk_buf_x, total_tokens * config.d_inner * sizeof(float));
        // malloc_device(&chunk_buf_z, total_tokens * config.d_inner * sizeof(float));
        malloc_device(&chunk_buf_conv, total_tokens * config.d_inner * sizeof(float));
        
        int stride_delta = config.dt_rank + 2 * config.d_state;
        malloc_device(&chunk_buf_delta, total_tokens * stride_delta * sizeof(float));
        malloc_device(&chunk_buf_dt_in, total_tokens * config.dt_rank * sizeof(float));
        malloc_device(&chunk_buf_dt, total_tokens * config.d_inner * sizeof(float));
        malloc_device(&chunk_buf_y, total_tokens * config.d_inner * sizeof(float));
        malloc_device(&chunk_buf_B, total_tokens * config.d_state * sizeof(float));
        malloc_device(&chunk_buf_C, total_tokens * config.d_state * sizeof(float));
    }
    
    void free_chunk() {
        if (max_chunk_len_alloc == 0) return;
        free_device(chunk_buf_xz); // free_device(chunk_buf_x); free_device(chunk_buf_z);
        free_device(chunk_buf_conv); free_device(chunk_buf_delta); 
        free_device(chunk_buf_dt_in); free_device(chunk_buf_dt); free_device(chunk_buf_y);
        free_device(chunk_buf_B); free_device(chunk_buf_C);
        max_chunk_len_alloc = 0;
    }

    void forward_chunk(float* x_in, float* x_out, int length) {
        int batch_eff = batch_size * length;
        
           #if GPU_DEBUG_LOGITS
           // DEBUG: X_IN
           if (length > 0) {
               std::vector<float> h_d(5);
               cudaMemcpy(h_d.data(), x_in, 5*sizeof(float), cudaMemcpyDeviceToHost);
               printf("CHUNK X_IN[0-4]: %f %f %f %f %f\n", h_d[0], h_d[1], h_d[2], h_d[3], h_d[4]);
           }
           #endif
        
        // 1. In Proj
        gemm_gpu_batch(x_in, in_proj_w, chunk_buf_xz, batch_eff, 2*config.d_inner, config.d_model);
        gpu_add_bias(chunk_buf_xz, in_proj_b, batch_eff, 2*config.d_inner);
        
           #if GPU_DEBUG_LOGITS
           // DEBUG
           if (length > 0) { // Batch 0 T=0 check is tricky here. batch_eff is flattened.
               // Just print always for test
               std::vector<float> h_d(5);
               cudaMemcpy(h_d.data(), chunk_buf_xz, 5*sizeof(float), cudaMemcpyDeviceToHost);
               printf("CHUNK XZ[0-4]: %f %f %f %f %f\n", h_d[0], h_d[1], h_d[2], h_d[3], h_d[4]);
           }
           #endif
        
        // Split no longer needed with strided access
        // gpu_copy_strided(chunk_buf_xz, chunk_buf_x, 2*config.d_inner, config.d_inner, batch_eff);
        // gpu_copy_strided(chunk_buf_xz + config.d_inner, chunk_buf_z, 2*config.d_inner, config.d_inner, batch_eff);
        
        // 2. Conv1d (Strided x input)
        // Input x is at chunk_buf_xz (offset 0), stride 2*d_inner.
        gpu_mamba_conv1d_chunk(chunk_buf_xz, conv_state, conv1d_w, conv1d_b, chunk_buf_conv, batch_size, length, config.d_inner, config.d_conv, 2*config.d_inner);
        
        // 3. Delta Proj
        int stride_delta = config.dt_rank + 2*config.d_state;
        gemm_gpu_batch(chunk_buf_conv, x_proj_w, chunk_buf_delta, batch_eff, stride_delta, config.d_inner);
        
        // Split Delta
        gpu_copy_strided(chunk_buf_delta, chunk_buf_dt_in, stride_delta, config.dt_rank, batch_eff);
        gpu_copy_strided(chunk_buf_delta + config.dt_rank, chunk_buf_B, stride_delta, config.d_state, batch_eff);
        gpu_copy_strided(chunk_buf_delta + config.dt_rank + config.d_state, chunk_buf_C, stride_delta, config.d_state, batch_eff);
        
        // 4. dt Proj
        gemm_gpu_batch(chunk_buf_dt_in, dt_proj_w, chunk_buf_dt, batch_eff, config.d_inner, config.dt_rank);
        gpu_add_bias_softplus_batch(chunk_buf_dt, dt_proj_b, config.d_inner, batch_eff); 
        
        // 5. SSM
        gpu_mamba_ssm_chunk(chunk_buf_conv, chunk_buf_dt, A_log, D, 
                            chunk_buf_B, chunk_buf_C, ssm_state, chunk_buf_y, 
                            batch_size, length, config.d_inner, config.d_state);
                            
        // 6. Gate (Strided z)
        // z is at chunk_buf_xz + d_inner. Stride 2*d_inner.
        gpu_gate_strided(chunk_buf_y, chunk_buf_xz + config.d_inner, config.d_inner, 2*config.d_inner, batch_eff * config.d_inner);
        
        // 7. Out Proj
        gemm_gpu_batch(chunk_buf_y, out_proj_w, x_out, batch_eff, config.d_model, config.d_inner);
        gpu_add_bias(x_out, out_proj_b, batch_eff, config.d_model);
    }
};

struct FeedForwardGPU {
    float* w1; float* b1;
    float* w2; float* b2;
    float* buf; // [Batch, 4*d]
    int d_model; int d_hidden; int batch_size;

    void allocate(int d, int batch) {
        d_model = d; d_hidden = 4*d; batch_size = batch;
        malloc_device(&w1, d_hidden * d_model * sizeof(float));
        malloc_device(&b1, d_hidden * sizeof(float));
        malloc_device(&w2, d_model * d_hidden * sizeof(float));
        malloc_device(&b2, d_model * sizeof(float));
        malloc_device(&buf, (size_t)batch * d_hidden * sizeof(float));
    }
    void free() {
        free_device(w1); free_device(b1); free_device(w2); free_device(b2); free_device(buf);
    }
    void load_weights(std::ifstream& f) {
        load_vec(f, w1, d_hidden * d_model, true, d_hidden, d_model);
        load_vec(f, b1, d_hidden);
        load_vec(f, w2, d_model * d_hidden, true, d_model, d_hidden);
        load_vec(f, b2, d_model);
    }
    
    // Chunk Logic
    float* chunk_buf; 
    int max_alloc = 0;
    
    void allocate_chunk(int length) {
        if (max_alloc >= length) return;
        if (max_alloc > 0) free_device(chunk_buf);
        max_alloc = length;
        size_t total = (size_t)batch_size * length;
        malloc_device(&chunk_buf, total * d_hidden * sizeof(float));
    }
    void free_chunk() {
        if (max_alloc > 0) free_device(chunk_buf);
        max_alloc = 0;
    }
    
    void forward_chunk(float* x, float* out, int length) {
        int batch_eff = batch_size * length;
        gemm_gpu_batch(x, w1, chunk_buf, batch_eff, d_hidden, d_model);
        gpu_add_bias(chunk_buf, b1, batch_eff, d_hidden);
        
        gpu_gelu_batch(chunk_buf, d_hidden, batch_eff); // batch_eff rows, size d_hidden
        
        gemm_gpu_batch(chunk_buf, w2, out, batch_eff, d_model, d_hidden);
        gpu_add_bias(out, b2, batch_eff, d_model);
    }

    void step_batch(float* x, float* out) {
        // x: [batch, d_model], w1: [d_model, d_hidden], buf: [batch, d_hidden]
        gemm_gpu_batch(x, w1, buf, batch_size, d_hidden, d_model);
        gpu_add_bias(buf, b1, batch_size, d_hidden);
        
        gpu_gelu_batch(buf, d_hidden, batch_size);
        
        // buf: [batch, d_hidden], w2: [d_hidden, d_model], out: [batch, d_model]
        gemm_gpu_batch(buf, w2, out, batch_size, d_model, d_hidden);
        gpu_add_bias(out, b2, batch_size, d_model);
    }
};

struct BoaBlockGPU {
    MambaBlockGPU mamba;
    LayerNormGPU ln1;
    LayerNormGPU ln2;
    FeedForwardGPU ff;
    int d_model;
    int batch_size;
    
    BoaBlockGPU() = default;

    void allocate(MambaConfig conf, int batch) {
        d_model = conf.d_model;
        batch_size = batch;
        mamba.allocate(conf, batch);
        ln1.allocate(d_model);
        ln2.allocate(d_model);
        ff.allocate(d_model, batch);
    }
    
    void free() {
        mamba.free(); ln1.free(); ln2.free(); ff.free();
    }
    
    void reset_cache() {
        mamba.reset_cache();
    }
    
    void load_weights(std::ifstream& f) {
        std::vector<float> tmp(d_model);
        f.read((char*)tmp.data(), d_model*4);
        copy_to_device(ln1.weight, tmp.data(), d_model*4);
        f.read((char*)tmp.data(), d_model*4);
        copy_to_device(ln1.bias, tmp.data(), d_model*4);
        
        mamba.load_weights(f);
        
        f.read((char*)tmp.data(), d_model*4);
        copy_to_device(ln2.weight, tmp.data(), d_model*4);
        f.read((char*)tmp.data(), d_model*4);
        copy_to_device(ln2.bias, tmp.data(), d_model*4);
        
        ff.load_weights(f);
    }
    
    void step_batch(float* x_in, float* x_out, float* buf_res) {
        checkCudaErrors(cudaMemcpy(buf_res, x_in, (size_t)batch_size * d_model * sizeof(float), cudaMemcpyDeviceToDevice));
        
        gpu_layernorm_batch(x_in, ln1.weight, ln1.bias, d_model, batch_size);
        mamba.step_batch(x_in, x_in);
        gpu_layernorm_batch(x_in, ln2.weight, ln2.bias, d_model, batch_size);
        ff.step_batch(x_in, x_in);
        gpu_add_batch(x_in, buf_res, d_model, batch_size);
        
        if (x_in != x_out) {
            checkCudaErrors(cudaMemcpy(x_out, x_in, (size_t)batch_size * d_model * sizeof(float), cudaMemcpyDeviceToDevice));
        }
    }

    // Chunk Logic
    void allocate_chunk(int length) {
        mamba.allocate_chunk(length);
        ff.allocate_chunk(length);
    }
    void free_chunk() {
        mamba.free_chunk();
        ff.free_chunk();
    }
    
    // x_in and buf_res are flattened [Batch*Length, d_model]
    void forward_chunk(float* x_in, float* x_out, float* buf_res, int length) {
        int batch_eff = batch_size * length;
        checkCudaErrors(cudaMemcpy(buf_res, x_in, (size_t)batch_eff * d_model * sizeof(float), cudaMemcpyDeviceToDevice));
        
        // LN1
        gpu_layernorm_batch(x_in, ln1.weight, ln1.bias, d_model, batch_eff);
        
        // Mamba
        mamba.forward_chunk(x_in, x_in, length);
        
        // LN2
        gpu_layernorm_batch(x_in, ln2.weight, ln2.bias, d_model, batch_eff);
        
        // FF
        ff.forward_chunk(x_in, x_in, length);
        
        // Residual
        gpu_add_batch(x_in, buf_res, d_model, batch_eff);
        
        if (x_in != x_out) {
            checkCudaErrors(cudaMemcpy(x_out, x_in, (size_t)batch_eff * d_model * sizeof(float), cudaMemcpyDeviceToDevice));
        }
    }
};

struct BoaPredictorGPU {
    BoaPredictorGPU(MambaConfig conf, int v_size, int n_l, int batch) : config(conf), vocab_size(v_size), n_layers(n_l), batch_size(batch) {
        embedding_size = vocab_size * config.d_model;
        malloc_device(&embedding, embedding_size * sizeof(float));
        
        
        blocks = new BoaBlockGPU[n_layers];
        for(int i=0; i<n_layers; ++i) { 
            blocks[i].allocate(config, batch);
        }
        
        malloc_device(&head_w1, config.d_model * config.d_model * sizeof(float));
        malloc_device(&head_b1, config.d_model * sizeof(float));
        malloc_device(&head_w2, vocab_size * config.d_model * sizeof(float));
        malloc_device(&head_b2, vocab_size * sizeof(float));
        
        malloc_device(&buf_x, (size_t)batch * config.d_model * sizeof(float));
        malloc_device(&buf_res, (size_t)batch * config.d_model * sizeof(float));
        malloc_device(&buf_head, (size_t)batch * config.d_model * sizeof(float));
    }
    
    // ... (Members)
    MambaConfig config;
    int vocab_size;
    int n_layers;
    int batch_size;
    long long embedding_size;
    
    float* embedding; 
    BoaBlockGPU* blocks; // Raw pointer array
    float* head_w1; float* head_b1;
    float* head_w2; float* head_b2;
    
    float* buf_x;
    float* buf_res;
    float* buf_head;
    
    void load_weights(std::string path) {
         // Same as before
         // ...
         // COPY FROM PREVIOUS
         std::cout << "DEBUG: Loading weights to GPU..." << std::endl;
         std::ifstream f(path, std::ios::binary);
         std::vector<float> h_emb(embedding_size);
         f.read(reinterpret_cast<char*>(h_emb.data()), embedding_size * sizeof(float));
         copy_to_device(embedding, h_emb.data(), embedding_size * sizeof(float));
         
         for(int i=0; i<n_layers; ++i) blocks[i].load_weights(f);
         
         std::vector<float> h_w1(config.d_model * config.d_model);
         f.read(reinterpret_cast<char*>(h_w1.data()), h_w1.size() * sizeof(float));
         auto t_w1 = transpose_weights(h_w1, config.d_model, config.d_model);
         copy_to_device(head_w1, t_w1.data(), t_w1.size() * sizeof(float));
         
         std::vector<float> h_b1(config.d_model);
         f.read(reinterpret_cast<char*>(h_b1.data()), h_b1.size() * sizeof(float));
         copy_to_device(head_b1, h_b1.data(), h_b1.size() * sizeof(float));
         
         std::vector<float> h_w2(vocab_size * config.d_model);
         f.read(reinterpret_cast<char*>(h_w2.data()), h_w2.size() * sizeof(float));
         auto t_w2 = transpose_weights(h_w2, vocab_size, config.d_model);
         copy_to_device(head_w2, t_w2.data(), t_w2.size() * sizeof(float));
         
         std::vector<float> h_b2(vocab_size);
         f.read(reinterpret_cast<char*>(h_b2.data()), h_b2.size() * sizeof(float));
         copy_to_device(head_b2, h_b2.data(), h_b2.size() * sizeof(float));
    }
    

    // Input: Tokens [Batch]. (Device Pointer)
    // Output: Logits [Batch, Vocab]. (Device Pointer)
    void step_batch(int* d_tokens, float* d_logits_out, bool add_bias = true) {
        // 1. Embedding lookup
        gpu_embedding_lookup_batch(d_tokens, embedding, buf_x, config.d_model, batch_size);
        
        // 2. Blocks
        for(int i=0; i<n_layers; ++i) {
            blocks[i].step_batch(buf_x, buf_x, buf_res);
        }
        
        // 3. Head: Linear -> ReLU -> Linear
        gemm_gpu_batch(buf_x, head_w1, buf_head, batch_size, config.d_model, config.d_model);
        gpu_add_bias(buf_head, head_b1, batch_size, config.d_model);
        gpu_relu(buf_head, batch_size * config.d_model);
        
        gemm_gpu_batch(buf_head, head_w2, d_logits_out, batch_size, vocab_size, config.d_model);
        if (add_bias) {
            gpu_add_bias(d_logits_out, head_b2, batch_size, vocab_size);
        }
    }

    // Chunk Logic
    float* chunk_buf_main;
    float* chunk_buf_res;
    float* chunk_buf_head;
    int max_chunk = 0;
    
    void allocate_chunk(int length) {
         if (max_chunk >= length) return;
         if (max_chunk > 0) free_chunk();
         max_chunk = length;
         // Propagate
         for(int i=0; i<n_layers; ++i) blocks[i].allocate_chunk(length);
         
         size_t total = (size_t)batch_size * length;
         malloc_device(&chunk_buf_main, total * config.d_model * sizeof(float));
         malloc_device(&chunk_buf_res, total * config.d_model * sizeof(float));
         malloc_device(&chunk_buf_head, total * config.d_model * sizeof(float)); 
    }
    
    void free_chunk() {
         if (max_chunk == 0) return;
         for(int i=0; i<n_layers; ++i) blocks[i].free_chunk();
         free_device(chunk_buf_main); free_device(chunk_buf_res); free_device(chunk_buf_head);
         max_chunk = 0;
    }
    
    void forward_chunk(const int* d_tokens, float* d_logits, int length) {
        // d_tokens: [Batch, Length] -> Flattened [Batch*Length]
        int batch_eff = batch_size * length;
        
        gpu_embedding_lookup_batch(d_tokens, embedding, chunk_buf_main, config.d_model, batch_eff);
        
        for(int i=0; i<n_layers; ++i) {
            blocks[i].forward_chunk(chunk_buf_main, chunk_buf_main, chunk_buf_res, length);
        }
        
        // Head
        gemm_gpu_batch(chunk_buf_main, head_w1, chunk_buf_head, batch_eff, config.d_model, config.d_model);
        gpu_add_bias(chunk_buf_head, head_b1, batch_eff, config.d_model);
        gpu_relu(chunk_buf_head, batch_eff * config.d_model);
        
        // Final Logits
        gemm_gpu_batch(chunk_buf_head, head_w2, d_logits, batch_eff, vocab_size, config.d_model);
        gpu_add_bias(d_logits, head_b2, batch_eff, vocab_size);
    }
    static size_t estimate_memory_static(MambaConfig config, int vocab_size, int n_layers, int batch_size, int chunk_len, bool is_compression) {
        size_t din = config.d_inner;
        size_t dmod = config.d_model;
        size_t dstate = config.d_state;
        size_t dconv = config.d_conv;
        size_t dtrank = config.dt_rank;
        size_t stride_delta = dtrank + 2 * dstate;

        // Weights (Fixed)
        size_t w_mamba = (2*din*dmod + 2*din + din*dconv + din + stride_delta*din + din*dtrank + din + din*dstate + din + dmod*din + dmod) * 4;
        size_t w_ff = (4*dmod*dmod + 4*dmod + dmod*4*dmod + dmod) * 4;
        size_t w_head = (dmod*dmod + dmod + vocab_size*dmod + vocab_size) * 4;
        size_t w_emb = (size_t)vocab_size * dmod * 4;
        size_t fixed = (w_mamba + w_ff) * n_layers + w_head + w_emb;

        // Per Batch (Inference states/buffers)
        size_t b_mamba = (din*dconv + din*dstate + 2*din + din + din + stride_delta + dtrank + din + din) * 4;
        size_t b_ff = (4*dmod) * 4;
        size_t b_pred = (dmod + dmod + dmod) * 4;
        size_t per_batch = (size_t)batch_size * ((b_mamba + b_ff) * n_layers + b_pred);

        // Chunk Buffers (Batch * Length)
        // Note: Compression uses forward_chunk (full buffers), Decompression uses step_batch (minimal buffers)
        // but both use d_batch_output [Batch, Length].
        size_t per_chunk = 0;
        if (is_compression) {
            size_t c_mamba = (2*din + din + stride_delta + dtrank + din + din + 2*dstate) * 4;
            size_t c_ff = (4*dmod) * 4;
            size_t c_pred = (dmod + dmod + dmod) * 4;
            size_t c_logits = (size_t)vocab_size * 4;
            per_chunk = (size_t)batch_size * (size_t)chunk_len * ((c_mamba + c_ff) * n_layers + c_pred + c_logits);
        } else {
            // Decompression output buffer
            per_chunk = (size_t)batch_size * (size_t)chunk_len * sizeof(int);
        }

        return fixed + per_batch + per_chunk;
    }

    void reset_cache() {
        for(int i=0; i<n_layers; ++i) blocks[i].reset_cache();
    }
    
    void free() {
        free_device(embedding);
        for(int i=0; i<n_layers; ++i) blocks[i].free();
        delete[] blocks;
        
        free_device(head_w1); free_device(head_b1);
        free_device(head_w2); free_device(head_b2);
        
        free_device(buf_x); free_device(buf_res); free_device(buf_head);
    }
};

