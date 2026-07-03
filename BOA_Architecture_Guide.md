# BOA Architecture & Mathematical Guide

## 1. Introduction

Welcome to the comprehensive guide to the BOA architecture, designed to provide a deep understanding of both the mathematical foundations and the practical implementation details (in both PyTorch and C++). BOA represents a state-of-the-art approach to sequence compression and modeling, leveraging the speed of modern recurrent networks and the exact precision of range coding.

This guide bridges the gap from beginner to advanced concepts, exploring how the **Mamba** backbone and **Range Codec** work together seamlessly.

---

## 2. Understanding MAMBA

### 2.1 The Math Behind Mamba

Mamba is based on the State Space Model (SSM) architecture. Instead of relying on full-attention mechanisms (which scale quadratically with sequence length), SSMs operate using a linear, time-invariant system that is discretized for discrete sequences.

The continuous-time system is defined as:
```text
h'(t) = A * h(t) + B * x(t)
y(t) = C * h(t)
```
Where:
- `x(t)` is the input sequence.
- `h(t)` is the hidden state.
- `y(t)` is the output sequence.

To process discrete tokens (like bytes), this system is discretized using a timescale parameter `Δ` (Delta). The discrete equations become:
```text
h_t = A_bar * h_{t-1} + B_bar * x_t
y_t = C * h_t
```
Where `A_bar = exp(Δ * A)` and `B_bar = (Δ * A)^(-1) * (exp(Δ * A) - I) * Δ * B`.

Mamba's crucial innovation is making `B`, `C`, and `Δ` **data-dependent** (functions of the input `x_t`). This allows the model to selectively remember or forget information, breaking the rigidity of standard LTI systems and enabling it to rival Transformers in expressivity while maintaining `O(N)` linear scaling.

### 2.2 PyTorch Implementation

In the PyTorch codebase (`model.py`), Mamba is implemented via block architectures that integrate standard feed-forward networks, SwiGLU activations, and normalization layers alongside the core SSM operations. 
The PyTorch model is designed for high-throughput batch training, utilizing efficient parallel scans where applicable.

### 2.3 C++ / CUDA Implementation

For inference (`boa_gpu.hpp` and `gemm_gpu.hpp`), the Mamba block is highly optimized for GPU execution. The `MambaBlockGPU` structure allocates contiguous memory blocks and utilizes fused kernels to evaluate the SSM equations. During compression, chunk-parallel evaluation evaluates the continuous stream, while during decompression, a step-by-step state recurrence tracks the precise temporal dynamics without relying on costly host-to-device memory transfers.

---

## 3. Understanding the Range Codec

### 3.1 The Math Behind Range Coding

Range coding is a form of entropy coding that achieves near-Shannon-limit compression. It represents an entire message as a fractional number within the interval `[0, 1)`.

As the encoder processes each byte, it divides the current range into sub-ranges proportional to the probability of each possible byte (given by the neural network).
- If the current range is `[L, L + R)`, and the neural network assigns probability `P(x)` to byte `x`, the new range becomes:
  ```text
  New_R = R * P(x)
  New_L = L + R * Cumulative_P(x)
  ```
Because `P(x)` perfectly reflects the true distribution of the data (assuming the neural network is accurate), the final interval size is exactly `Product(P(x_i))`. The number of bits required to specify a number in this interval is `-log2(Product(P(x_i)))`, which is exactly the cross-entropy!

### 3.2 PyTorch & C++ Integration

In Python (`gpu_range_coder.py`), the `gpu.queue.RangeCoderBatch` class bridges the gap to C++. The neural network (in PyTorch) outputs a probability distribution over the 256 possible bytes. Instead of copying these logits back to the CPU, they are kept in GPU memory (`probs_gpu`). The CUDA-based Range Coder reads these probabilities directly, updating the `RCState` entirely on-device, massive parallelizing the range updates across batches.

---

## 4. Understanding the BOA Repo (Beginner to Advanced)

BOA brings Mamba and Range Coding together by using the Mamba network as a conditional probability predictor for the Range Coder.

### 4.1 Compression Flow
1. **Input Data**: The file is read and chunked.
2. **Forward Pass**: The PyTorch/C++ Mamba network processes the history of bytes to output logits for the next byte.
3. **Encoding**: The Range Coder updates its interval using these logits and outputs compressed bits.

### 4.2 Decompression Flow (HydraBOA)
Decompression is inherently sequential (we need byte `t` to predict byte `t+1`). HydraBOA mitigates this by using multiple "heads" that predict several bytes simultaneously based on a shared backbone state, thus reducing the number of sequential evaluations.

### 4.3 Navigating the Code
- **Training**: Handled in `train.py` using `evaluator.py` to check Bits-Per-Byte (BPB) compression ratios.
- **Python Models**: Located in `model.py` and `hydra_model.py`.
- **C++ Kernels**: Located in `portability_solved_cpp/`, primarily `gemm_gpu.hpp`, `boa_gpu.hpp`, and `hydra_boa.cpp`.

This dual-language implementation guarantees maximum flexibility during training (via Python/PyTorch) and absolute maximum throughput during actual compression/decompression tasks (via raw C++/CUDA).
