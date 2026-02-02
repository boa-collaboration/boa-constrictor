
// ==========================================
// WARP PARALLEL ENCODING
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

// Inclusive Scan
__inline__ __device__ float warpScanSum(float val) {
    float temp = 0.0f;
    #pragma unroll
    for (int offset = 1; offset <= 16; offset *= 2) {
        temp = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if (threadIdx.x % 32 >= offset) val += temp;
    }
    return val;
}

__global__ void ker_rc_encode_chunk_parallel_warp(const float* logits_base, const int* tokens_base, const int* lengths, 
                                                  RCState* states, unsigned int* out_bufs, 
                                                  int pitch_words, int vocab_size, int batch_size, 
                                                  int chunk_len, int max_len, int start_t, int logit_stride) {
    int b = blockIdx.x; // One block (warp) per batch sequence
    if (b >= batch_size) return;

    int lane = threadIdx.x; // 0..31

    // Load State into registers
    RCState st;
    if (lane == 0) st = states[b];
    // Sync to ensure lane 0 loaded it? No, other lanes don't need it yet.

    int len = lengths[b];
    int valid_len = (len > max_len) ? max_len : len;

    // Pointers for this batch
    const float* my_logits = logits_base + b * logit_stride; // logit_stride is usually chunk_len * vocab
    const int* my_tokens = tokens_base + b * chunk_len; // Tokens are [Batch, Length] usually, or flat? 
    // boa.cpp passes d_batch_data[buf] which is [Batch * Chunk]. So stride is Chunk.
    
    // Output pointer
    unsigned int* my_out = out_bufs + (size_t)b * pitch_words;

    // Shared Memory for CDF
    // We need 257 elements for CDF.
    // Probs can reuse.
    __shared__ volatile float s_cdf[260]; // padded

    // Loop over time steps
    for (int t = start_t; t < valid_len; ++t) {
        int token = my_tokens[t];
        const float* l_ptr = my_logits + t * vocab_size; // 256 floats

        // 1. Load & Max Reduction
        // Each thread loads 8 elements (256 / 32 = 8).
        float my_vals[8];
        float my_max = -1e30f;

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int idx = lane * 8 + i;
            float val = l_ptr[idx];
            my_vals[i] = val; // Store for next step
            my_max = fmaxf(my_max, val);
        }
        float batch_max = warpReduceMax(my_max);
        batch_max = __shfl_sync(0xFFFFFFFF, batch_max, 0); // Broadcast max

        // 2. Exp & Sum
        float my_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float val = gpu_exp_lut(my_vals[i] - batch_max);
            my_vals[i] = val; // Store exp'd val
            my_sum += val;
        }
        float batch_sum = warpReduceSum(my_sum);
        batch_sum = __shfl_sync(0xFFFFFFFF, batch_sum, 0); // Broadcast sum
        
        float inv_sum = 1.0f / batch_sum;

        // 3. Normalize & Local CDF
        float local_cdf = 0.0f;
        
        // This is tricky. We need global Scan.
        // Each thread has 8 elements.
        // Let's first sum our 8 elements to get "thread total".
        float thread_sum = my_sum * inv_sum;
        
        // Prefix scan of thread totals across warp
        float warp_prefix = warpScanSum(thread_sum); 
        float warp_start = warp_prefix - thread_sum; // Exclusive scan result for this thread's block

        // Now compute CDF for 8 elements and store to shared
        float running = warp_start;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int idx = lane * 8 + i;
            // s_cdf[idx] SHOULD be exclusive sum (starts at 0).
            // But we can store Probs or CDF?
            // Standard RC needs: cdf[target] (low) and cdf[target+1] (high).
            // So s_cdf[k] = sum(p_0...p_{k-1}).
            
            // Convert to int scale on fly? Or keep float?
            // "build_cdf_fast" logic converts to int scale.
            // Let's follow "build_cdf_fast" logic:
            // scale = (double)free_weight / norm. float is fine.
            // cdf[i] = (cumulative * scale).
            
            // Let's store Normalized Float CDF in shared.
            // s_cdf[0] = 0.
            // s_cdf[idx] = running.
            // BUT idx=0 must be 0.
            
            s_cdf[idx] = running;
            running += my_vals[i] * inv_sum; // Add prob
        }
        // Last element total
        if (lane == 31) s_cdf[256] = 1.0f; // Close to 1.0

        // 4. Encode (Lane 0 only)
        if (lane == 0) {
            // Replicate rc_encode_symbol_probs logic but with shared memory CDF
            // We need to Convert Float CDF to Int CDF 16-bit-ish scale?
            // ker_rc_encode uses "build_cdf_fast" which maps float to uint.
            /*
            double scale = (double)free_weight / norm;
            unsigned int left = (unsigned int)(cumulative_float * scale) + accumulated_slack;
            */
            // We can do this map here for just the target symbol!
            // We don't need full Int CDF. Just Target Low/High.
            
            float prob_low = s_cdf[token];
            float prob_high = s_cdf[token+1];
            // Fix edge case for token=255 -> s_cdf[256] needed.
            if (token == 255) prob_high = 1.0f;

            // Convert to Int Scale (2^16-ish?)
            // Actually existing logic uses ~2^24 or something.
            // build_cdf_fast: free_weight = TOTAL - vocab(256) - slack...
            // TOTAL = 1<<16? 
            // In exact_math.hpp? #define TOTAL (1<<PRECISION)? No.
            // Let's check build_cdf_fast.
            
            const unsigned int TOTAL = (1<<16); // Check this constant!
            // Assume 16 bit total for now as standard RC.
            
            unsigned int low_int = (unsigned int)(prob_low * (TOTAL - 256)) + token;
            unsigned int high_int = (unsigned int)(prob_high * (TOTAL - 256)) + (token + 1);
            
            // Verify: 
            // For i=0: low=0.
            // For i=256: high = (1.0 * (T-256)) + 256 = T. Correct.
            
            unsigned int left = low_int;
            unsigned int prob = high_int - low_int;
            
            // Standard RC Encode Step
            unsigned long long scale_rc = st.range >> 16; // PRECISION?
            unsigned long long old_lower = st.lower;
            
            st.range = scale_rc * (unsigned long long)prob;
            st.lower = st.lower + scale_rc * (unsigned long long)left;
            
            // Renormalization Loop
            if (st.inverted_num > 0) {
                 unsigned long long sum = st.lower + st.range;
                 if (sum > st.lower) {
                     unsigned int first_word = (st.lower < old_lower) ? st.first_inv_lower_word + 1u : st.first_inv_lower_word;
                     unsigned int subsequent = (st.lower < old_lower) ? 0u : 0xFFFFFFFFu;
                     out_bufs[(size_t)b * pitch_words + st.write_idx_words++] = first_word;
                     for (int k=1; k<st.inverted_num; k++) out_bufs[(size_t)b * pitch_words + st.write_idx_words++] = subsequent;
                     st.inverted_num = 0;
                 }
            }
            while (st.range < (1ull << (64-32))) {
                unsigned int lower_word = (unsigned int)(st.lower >> (64-32));
                st.lower <<= 32; st.range <<= 32;
                if (st.inverted_num > 0) {
                     st.inverted_num++;
                } else {
                     unsigned long long sum = st.lower + st.range;
                     if (sum > st.lower) {
                         out_bufs[(size_t)b * pitch_words + st.write_idx_words++] = lower_word;
                     } else {
                         st.inverted_num = 1;
                         st.first_inv_lower_word = lower_word;
                     }
                }
            }
        }
        // Need sync? Lane 0 is independent now until next iteration Load.
        // But next iteration overwrites s_cdf? Yes.
        // Warp threads are implicitly synced? 
        // Safer to sync before overwriting shared.
        // Or if we use __shfl based logic for next iter, we are fine.
        // But s_cdf is written by all. 
        // Lane 0 reads s_cdf[token].
        // If token corresponds to lane X, lane X might overwrite it next iter.
        // So we need sync.
        // __syncwarp(); 
    }
    
    // Save state
    if (lane == 0) states[b] = st;
}

void gpu_rc_encode_chunk_warp(const float* logits, const int* chunk_data, const int* lengths, 
                             RCState* states, unsigned int* out_bufs,
                             int pitch_words, int vocab_size, int batch_size, 
                             int chunk_len, int max_len, int start_t, int logit_stride) {
    if (vocab_size != 256) {
        printf("Error: Warp Encode requires Vocab=256\n"); return;
    }
    // Launch 1 block (32 threads) per batch
    ker_rc_encode_chunk_parallel_warp<<<batch_size, 32>>>(logits, chunk_data, lengths, states, out_bufs, pitch_words, vocab_size, batch_size, chunk_len, max_len, start_t, logit_stride);
    checkCudaErrors(cudaGetLastError());
}
