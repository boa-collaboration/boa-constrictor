from __future__ import annotations

import os
import sys
import sysconfig
import importlib
import importlib.util
import pathlib
import tempfile
from textwrap import dedent
import subprocess
import shutil

def _build_and_import_cuda_extension() -> object:
    try:
        import pybind11  # noqa: F401
    except Exception:
        raise RuntimeError("pybind11 is required to build CUDA extension")

    build_dir = pathlib.Path(tempfile.gettempdir()) / "gpu_range_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    ext_name = "_gpu_range_cuda_ext"
    src_cu = build_dir / (ext_name + ".cu")

    # CUDA source implementing a constriction-compatible word-based Range Coder (u32 words, u64 state)
    cuda_code = dedent(r'''
    #include <pybind11/pybind11.h>
    #include <pybind11/numpy.h>
    #include <pybind11/stl.h>
    #include <cuda_runtime.h>
    #include <cstdint>
    #include <math.h>
    #include <vector>
    #include <stdexcept>
    #include <algorithm>
    #include <cstring>

    namespace py = pybind11;

    static constexpr int PRECISION = 24;

    struct EncState {
        unsigned long long lower;
        unsigned long long range;
        int inverted_num;
        unsigned int first_inv_lower_word;
        int write_idx_words;
    };

    struct DecState {
        unsigned long long lower;
        unsigned long long range;
        unsigned long long point;
        int read_idx_words;
    };

    __device__ void build_cdf_fast(const float* probs_row, int K, uint32_t* cdf) {
        const uint32_t TOTAL = 1u << PRECISION;
        if (K <= 0) { cdf[0] = 0; cdf[1] = TOTAL; return; }
        const uint32_t free_weight = TOTAL - (uint32_t)K;
        double norm = 0.0;
        for (int i = 0; i < K; ++i) norm += (double)probs_row[i];
        if (!(norm > 0.0) || !isfinite(norm)) {
            cdf[0] = 0;
            uint32_t acc = 0;
            for (int i=0;i<K;i++) { cdf[i] = acc; acc += (free_weight / (uint32_t)K) + 1u; }
            cdf[K] = TOTAL; return;
        }
        double scale = (double)free_weight / norm;
        double cumulative_float = 0.0;
        uint32_t accumulated_slack = 0;
        for (int i=0;i<K;i++) {
            uint32_t left = (uint32_t)(cumulative_float * scale) + accumulated_slack;
            cdf[i] = left;
            cumulative_float += (double)probs_row[i];
            accumulated_slack += 1u;
        }
        cdf[K] = TOTAL;
    }

    __device__ __forceinline__ void flush_inverted(uint32_t* out_row, EncState* st) {
        if (st->inverted_num > 0) {
            // We can't know wrap condition without original pre-update lower; use point at seal time.
            // This function is only called in finalize; handled there.
        }
    }

    __global__ void encode_kernel(const int32_t* d_symbols, const float* d_probs, int N, int K,
                                  uint32_t* d_out_words, EncState* states, int pitch_words,
                                  const uint8_t* d_mask) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= N) return;
        if (d_mask && d_mask[idx] == 0) return;
        EncState &st = states[idx];
        const int32_t s = d_symbols[idx];
        const float* probs_row = d_probs + (size_t)idx * (size_t)K;
        if (s < 0 || s >= K) return;
        uint32_t cdf[1025];
        build_cdf_fast(probs_row, K, cdf);
        uint32_t left = cdf[s];
        uint32_t prob = cdf[s+1] - cdf[s];
        unsigned long long scale = st.range >> PRECISION;
        st.range = scale * (unsigned long long)prob;
        unsigned long long old_lower = st.lower;
        st.lower = st.lower + scale * (unsigned long long)left;
        // Handle transition out of inverted
        if (st.inverted_num > 0) {
            unsigned long long sum = st.lower + st.range;
            if (sum > st.lower) {
                uint32_t first_word, subsequent;
                if (st.lower < old_lower) { first_word = st.first_inv_lower_word + 1u; subsequent = 0u; }
                else { first_word = st.first_inv_lower_word; subsequent = 0xFFFFFFFFu; }
                int widx = st.write_idx_words;
                uint32_t* out = d_out_words + (size_t)idx * (size_t)pitch_words;
                out[(size_t)widx++] = first_word;
                for (int i=1;i<st.inverted_num;i++) out[(size_t)widx++] = subsequent;
                st.write_idx_words = widx;
                st.inverted_num = 0;
            }
        }
        // Renormalize if needed (emit possibly multiple words)
        while (st.range < (1ull << (64-32))) {
            uint32_t lower_word = (uint32_t)(st.lower >> (64-32));
            st.lower <<= 32;
            st.range <<= 32;
            if (st.inverted_num > 0) {
                if (st.inverted_num < 0x7FFFFFFF) st.inverted_num += 1;
            } else {
                unsigned long long sum = st.lower + st.range;
                if (sum > st.lower) {
                    int widx = st.write_idx_words;
                    uint32_t* out = d_out_words + (size_t)idx * (size_t)pitch_words;
                    out[(size_t)widx++] = lower_word;
                    st.write_idx_words = widx;
                } else {
                    st.inverted_num = 1;
                    st.first_inv_lower_word = lower_word;
                }
            }
        }
    }

    __global__ void finalize_kernel(int N, uint32_t* d_out_words, EncState* states, int pitch_words, int* sizes_words) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= N) return;
        EncState &st = states[idx];
        uint32_t* out = d_out_words + (size_t)idx * (size_t)pitch_words;
        if (st.range == 0xFFFFFFFFFFFFFFFFull) { sizes_words[idx] = 0; return; }
        unsigned long long point = st.lower + ((1ull << (64-32)) - 1ull);
        if (st.inverted_num > 0) {
            uint32_t first_word, subsequent;
            if (point < st.lower) { first_word = st.first_inv_lower_word + 1u; subsequent = 0u; }
            else { first_word = st.first_inv_lower_word; subsequent = 0xFFFFFFFFu; }
            int widx = st.write_idx_words;
            out[(size_t)widx++] = first_word;
            for (int i=1;i<st.inverted_num;i++) out[(size_t)widx++] = subsequent;
            st.write_idx_words = widx;
            st.inverted_num = 0;
        }
        uint32_t point_word = (uint32_t)(point >> (64-32));
        int widx = st.write_idx_words;
        out[(size_t)widx++] = point_word;
        unsigned long long upper = st.lower + st.range;
        uint32_t upper_word = (uint32_t)(upper >> (64-32));
        if (upper_word == point_word) out[(size_t)widx++] = 0u;
        st.write_idx_words = widx;
        sizes_words[idx] = widx;
    }

    __global__ void init_dec_kernel(int N, const uint32_t* d_in_words, int* sizes_words, DecState* dst, int pitch_words) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= N) return;
        const uint32_t* in = d_in_words + (size_t)idx * (size_t)pitch_words;
        DecState &st = dst[idx];
        st.lower = 0ull; st.range = 0xFFFFFFFFFFFFFFFFull; st.point = 0ull; st.read_idx_words = 0;
        for (int i=0;i<2;i++) {
            uint32_t w = (st.read_idx_words < sizes_words[idx]) ? in[st.read_idx_words++] : 0u;
            st.point = (st.point << 32) | (unsigned long long)w;
        }
    }

    __global__ void decode_step_kernel(int N, int K, const float* d_probs, const uint32_t* d_in_words, int* sizes_words,
                                       DecState* st_arr, int pitch_words, int32_t* out_symbols, const uint8_t* d_mask) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= N) return;
        if (d_mask && d_mask[idx] == 0) return;
        DecState &st = st_arr[idx];
        const float* probs_row = d_probs + (size_t)idx * (size_t)K;
        const uint32_t* in = d_in_words + (size_t)idx * (size_t)pitch_words;
        if (K <= 0) { out_symbols[idx] = 0; return; }
        uint32_t cdf[1025];
        build_cdf_fast(probs_row, K, cdf);
        unsigned long long scale = st.range >> PRECISION;
        unsigned long long q = (st.point - st.lower) / scale;
        if (q >= (1ull<<PRECISION)) q = (1ull<<PRECISION)-1ull;
        uint32_t target = (uint32_t)q;
        int next_symbol = 1;
        while (next_symbol <= K && !(cdf[next_symbol] > target)) ++next_symbol;
        int s = next_symbol - 1; if (s < 0) s = 0; if (s >= K) s = K-1;
        out_symbols[idx] = (int32_t)s;
        uint32_t left = cdf[s]; uint32_t prob = cdf[s+1] - cdf[s];
        st.lower = st.lower + scale * (unsigned long long)left;
        st.range = scale * (unsigned long long)prob;
        while (st.range < (1ull << (64-32))) {
            st.lower <<= 32;
            st.range <<= 32;
            uint32_t w = (st.read_idx_words < sizes_words[idx]) ? in[st.read_idx_words++] : 0u;
            st.point = (st.point << 32) | (unsigned long long)w;
        }
    }

    class RangeCoderBatch {
    public:
        int N, K, pitch;
        EncState* d_enc_states;
        DecState* d_dec_states;
        uint32_t* d_words;
        int* d_sizes; // sizes in words
        RangeCoderBatch(int N_, int K_, int pitch_) : N(N_), K(K_), pitch(pitch_), d_enc_states(nullptr), d_dec_states(nullptr), d_words(nullptr), d_sizes(nullptr) {
            cudaMalloc(&d_enc_states, sizeof(EncState) * N);
            cudaMalloc(&d_dec_states, sizeof(DecState) * N);
            cudaMalloc(&d_words, (size_t)N * (size_t)pitch * sizeof(uint32_t));
            cudaMalloc(&d_sizes, sizeof(int) * N);
            std::vector<EncState> init(N);
            for (int i = 0; i < N; ++i) { init[i].lower=0ull; init[i].range=0xFFFFFFFFFFFFFFFFull; init[i].inverted_num=0; init[i].first_inv_lower_word=0u; init[i].write_idx_words=0; }
            cudaMemcpy(d_enc_states, init.data(), sizeof(EncState) * N, cudaMemcpyHostToDevice);
            std::vector<int> zeros(N, 0);
            cudaMemcpy(d_sizes, zeros.data(), sizeof(int) * N, cudaMemcpyHostToDevice);
            cudaMemset(d_words, 0, (size_t)N * (size_t)pitch * sizeof(uint32_t));
        }
        ~RangeCoderBatch() {
            cudaFree(d_enc_states);
            cudaFree(d_dec_states);
            cudaFree(d_words);
            cudaFree(d_sizes);
        }
        void load_compressed_from_host(py::list compressed_list) {
            if ((int)compressed_list.size() != N) throw std::runtime_error("compressed_list length must equal N");
            std::vector<int> sizes_host(N, 0);
            std::vector<uint32_t> buf((size_t)N * (size_t)pitch, 0u);
            for (int i=0;i<N;i++) {
                py::array arr = py::array::ensure(compressed_list[i]);
                if (!arr || arr.ndim()!=1 || arr.itemsize()!=4 || arr.dtype().kind()!='u') throw std::runtime_error("Each compressed item must be uint32[?]");
                size_t nwords = (size_t)arr.shape(0);
                if (nwords > (size_t)pitch) throw std::runtime_error("Compressed stream exceeds pitch; increase pitch");
                const uint32_t* src = static_cast<const uint32_t*>(arr.data());
                uint32_t* dst = buf.data() + (size_t)i * (size_t)pitch;
                std::memcpy(dst, src, nwords * sizeof(uint32_t));
                sizes_host[i] = (int)nwords;
            }
            cudaMemcpy(d_words, buf.data(), buf.size()*sizeof(uint32_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_sizes, sizes_host.data(), sizeof(int)*N, cudaMemcpyHostToDevice);
        }
        std::vector<int> get_sizes_host() {
            std::vector<int> sizes(N);
            cudaMemcpy(sizes.data(), d_sizes, sizeof(int)*N, cudaMemcpyDeviceToHost);
            return sizes;
        }
        void set_sizes_from_host(py::list sizes_list) {
            if ((int)sizes_list.size()!=N) throw std::runtime_error("sizes_list length must equal N");
            std::vector<int> sizes(N);
            for (int i=0;i<N;i++) sizes[i] = sizes_list[i].cast<int>();
            cudaMemcpy(d_sizes, sizes.data(), sizeof(int)*N, cudaMemcpyHostToDevice);
        }
        void encode_step_from_device(uint64_t symbols_ptr, uint64_t probs_ptr, uint64_t mask_ptr) {
            const int32_t* d_symbols = reinterpret_cast<const int32_t*>(symbols_ptr);
            const float* d_probs = reinterpret_cast<const float*>(probs_ptr);
            const uint8_t* d_mask = reinterpret_cast<const uint8_t*>(mask_ptr);
            int threads = 128; int blocks = (N + threads - 1) / threads;
            encode_kernel<<<blocks, threads>>>(d_symbols, d_probs, N, K, d_words, d_enc_states, pitch, d_mask);
            cudaDeviceSynchronize();
        }
        void finalize() {
            int threads = 128; int blocks = (N + threads - 1) / threads;
            finalize_kernel<<<blocks, threads>>>(N, d_words, d_enc_states, pitch, d_sizes);
            cudaDeviceSynchronize();
        }
        std::vector<py::array_t<uint32_t>> get_compressed_host() {
            std::vector<int> sizes(N);
            cudaMemcpy(sizes.data(), d_sizes, sizeof(int) * N, cudaMemcpyDeviceToHost);
            std::vector<uint32_t> buf((size_t)N * (size_t)pitch);
            cudaMemcpy(buf.data(), d_words, buf.size()*sizeof(uint32_t), cudaMemcpyDeviceToHost);
            std::vector<py::array_t<uint32_t>> out; out.reserve(N);
            for (int i=0;i<N;i++) {
                size_t nwords = (size_t)std::max(0, sizes[i]);
                py::array_t<uint32_t> arr(nwords);
                auto r = arr.mutable_unchecked<1>();
                for (size_t w=0; w<nwords; ++w) r(w) = buf[(size_t)i*(size_t)pitch + w];
                out.push_back(arr);
            }
            return out;
        }
        void init_decoder_from_current_bytes() {
            int threads = 128; int blocks = (N + threads - 1) / threads;
            init_dec_kernel<<<blocks, threads>>>(N, d_words, d_sizes, d_dec_states, pitch);
            cudaDeviceSynchronize();
        }
        void decode_step_to_device(uint64_t probs_ptr, uint64_t out_symbols_ptr, uint64_t mask_ptr) {
            const float* d_probs = reinterpret_cast<const float*>(probs_ptr);
            int32_t* d_out = reinterpret_cast<int32_t*>(out_symbols_ptr);
            const uint8_t* d_mask = reinterpret_cast<const uint8_t*>(mask_ptr);
            int threads = 128; int blocks = (N + threads - 1) / threads;
            decode_step_kernel<<<blocks, threads>>>(N, K, d_probs, d_words, d_sizes, d_dec_states, pitch, d_out, d_mask);
            cudaDeviceSynchronize();
        }
    };

    PYBIND11_MODULE(_gpu_range_cuda_ext, m) {
        m.doc() = "GPU-backed range coder (constriction-compatible, u32 words, u64 state)";
        py::class_<RangeCoderBatch>(m, "RangeCoderBatch")
            .def(py::init<int,int,int>())
            .def("load_compressed_from_host", &RangeCoderBatch::load_compressed_from_host,
                 py::arg("compressed_list"))
            .def("get_sizes_host", &RangeCoderBatch::get_sizes_host)
            .def("set_sizes_from_host", &RangeCoderBatch::set_sizes_from_host,
                 py::arg("sizes_list"))
            .def("encode_step_from_device", &RangeCoderBatch::encode_step_from_device,
                 py::arg("symbols_ptr"), py::arg("probs_ptr"), py::arg("mask_ptr") = 0)
            .def("finalize", &RangeCoderBatch::finalize)
            .def("get_compressed_host", &RangeCoderBatch::get_compressed_host)
            .def("init_decoder_from_current_bytes", &RangeCoderBatch::init_decoder_from_current_bytes)
            .def("decode_step_to_device", &RangeCoderBatch::decode_step_to_device,
                 py::arg("probs_ptr"), py::arg("out_symbols_ptr"), py::arg("mask_ptr") = 0);
    }
    ''')

    src_cu.write_text(cuda_code)

    # Compile with nvcc
    nvcc = shutil.which('nvcc')
    if nvcc is None:
        raise RuntimeError('nvcc not found in PATH')
    from importlib.machinery import EXTENSION_SUFFIXES
    suf = EXTENSION_SUFFIXES[0]
    so_path = build_dir / (ext_name + suf)
    # Build include dirs robustly (handle spaces in paths)
    import pybind11
    inc_dirs = []
    try:
        cfg_paths = sysconfig.get_paths()
        for key in ('include', 'platinclude'):  # e.g., Python.h location
            p = cfg_paths.get(key)
            if p:
                inc_dirs.append(p)
    except Exception:
        pass
    # pybind11 headers
    for k in (pybind11.get_include(False), pybind11.get_include(True)):
        if k and k not in inc_dirs:
            inc_dirs.append(k)

    cmd = [nvcc, str(src_cu), '-shared', '-Xcompiler', '-fPIC', '-o', str(so_path),
           '-std=c++14', '-O3']
    # CUDA runtime library
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH') or '/usr/local/cuda'
    lib64 = pathlib.Path(cuda_home) / 'lib64'
    if lib64.exists():
        cmd += ['-L', str(lib64)]
        # add rpath for runtime loading
        cmd += ['-Xlinker', f'-rpath,{str(lib64)}']
    cmd += ['-lcudart']
    # Add include directories as separate args to preserve spaces
    for inc in inc_dirs:
        cmd += ['-I', inc]
    # Run nvcc
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        raise RuntimeError(f'nvcc build failed: {res.stderr.decode()}')

    spec = importlib.util.spec_from_file_location(ext_name, str(so_path))
    if spec is None or spec.loader is None:
        raise RuntimeError('Failed to load CUDA built extension')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


# Build extension on import: try CUDA if requested, otherwise CPU fallback
_ext = None
_cpu_ext = None  # optional CPU extension placeholder for fallbacks
_ext = _build_and_import_cuda_extension()
    
# Expose a minimal namespace compatible with the used subset: constriction.stream.queue
class _ModelStub:
    def __init__(self, kind: str, **kwargs):
        self.kind = kind
        self.kwargs = kwargs


class stream:
    class model:
        class Categorical(_ModelStub):
            def __init__(self, perfect: bool = False):
                super().__init__("categorical", perfect=perfect)

    class queue:
        class RangeEncoder:
            def __init__(self):
                # If extension provides a RangeEncoder class use it, otherwise prepare Python buffer for GPU function
                if hasattr(_ext, 'RangeEncoder'):
                    self._enc = _ext.RangeEncoder()
                    self._pybuf = None
                else:
                    self._enc = None
                    self._pybuf = {'symbols': [], 'probs': []}

            def clear(self):
                if self._enc is not None:
                    self._enc.clear()
                else:
                    self._pybuf = {'symbols': [], 'probs': []}

            def get_compressed(self):
                if self._enc is not None:
                    return self._enc.get_compressed()
                else:
                    import numpy as np
                    if len(self._pybuf['symbols']) == 0:
                        return np.zeros(0, dtype=np.uint32)
                    symbols = np.asarray(self._pybuf['symbols'], dtype=np.int32)
                    probs = np.asarray(self._pybuf['probs'], dtype=np.float32)
                    # For correctness, delegate to CPU extension's RangeEncoder to produce
                    # the exact same compressed format the CPU decoder expects.
                    if _cpu_ext is not None and hasattr(_cpu_ext, 'RangeEncoder'):
                        cpu_enc = _cpu_ext.RangeEncoder()
                        cpu_enc.encode_categorical(symbols, probs)
                        return cpu_enc.get_compressed()
                    # Fallback: try CUDA encode function (may be incompatible)
                    return _ext.encode_rows_gpu(symbols, probs)

            def encode(self, symbols, model, probs):
                """
                Supports Option 3: encode(symbols, model_family=Categorical, probs).
                - symbols: rank-1 int32 numpy array (len=N)
                - model: constriction.stream.model.Categorical (ignored other than type)
                - probs: rank-2 float32 numpy array with shape (N, K)
                """
                import numpy as np
                # Normalize inputs
                if not hasattr(symbols, "dtype"):
                    symbols = np.array([int(symbols)], dtype=np.int32)
                symbols = np.asarray(symbols, dtype=np.int32)
                probs = np.asarray(probs, dtype=np.float32)
                if symbols.ndim != 1:
                    raise ValueError("symbols must be rank-1")
                if probs.ndim != 2 or probs.shape[0] != symbols.shape[0]:
                    raise ValueError("probs must be rank-2 with probs.shape[0] == len(symbols)")
                if not isinstance(model, _ModelStub) or model.kind != "categorical":
                    raise TypeError("Only Categorical model is supported in this GPU stub")
                if self._enc is not None:
                    # delegate to compiled RangeEncoder
                    self._enc.encode_categorical(symbols, probs)
                else:
                    # buffer for batched GPU encode
                    for s in symbols.tolist():
                        self._pybuf['symbols'].append(int(s))
                    # ensure probs is list of rows
                    for row in probs.astype(np.float32):
                        self._pybuf['probs'].append(row.tolist())

        class RangeDecoder:
            def __init__(self, compressed):
                # Prefer compiled decoder if provided; otherwise fall back to CPU extension
                if hasattr(_ext, 'RangeDecoder'):
                    self._dec = _ext.RangeDecoder(compressed)
                elif _cpu_ext is not None and hasattr(_cpu_ext, 'RangeDecoder'):
                    self._dec = _cpu_ext.RangeDecoder(compressed)
                else:
                    raise RuntimeError('No decoder available in extensions')

            def decode(self, model, probs_or_amt, *rest):
                """
                Supports Option 3: decode(model_family=Categorical, probs) -> symbols array.
                """
                import numpy as np
                if not isinstance(model, _ModelStub) or model.kind != "categorical":
                    raise TypeError("Only Categorical model is supported in this GPU stub")
                probs = np.asarray(probs_or_amt, dtype=np.float32)
                if probs.ndim != 2:
                    raise ValueError("probs must be rank-2")
                return self._dec.decode_categorical(probs)


# For convenience, re-export top-level like constriction
__all__ = ["stream"]

# Optional: GPU batch coder convenience wrapper (requires torch)
class gpu:
    class queue:
        class RangeCoderBatch:
            def __init__(self, N: int, K: int, maxL: int, pitch_bytes: int | None = None):
                if pitch_bytes is None:
                    pitch_bytes = max(256, maxL * 8)
                self.N, self.K, self.maxL = N, K, maxL
                global _ext
                if _ext is None or not hasattr(_ext, 'RangeCoderBatch'):
                    # Try to build CUDA extension lazily if not present
                    try:
                        _ext = _build_and_import_cuda_extension()
                    except Exception as e:
                        raise RuntimeError('CUDA extension with RangeCoderBatch not available') from e
                # CUDA backend expects pitch in 32-bit words
                pitch_words = (int(pitch_bytes) + 3) // 4
                self._batch = _ext.RangeCoderBatch(N, K, pitch_words)

            def load_compressed_list(self, compressed_list):
                # Accept list of np.uint32 arrays (one per stream)
                self._batch.load_compressed_from_host(compressed_list)

            def encode_step(self, symbols_gpu, probs_gpu, mask=None):
                import torch
                assert symbols_gpu.is_cuda and probs_gpu.is_cuda
                assert symbols_gpu.numel() == self.N and probs_gpu.shape == (self.N, self.K)
                if symbols_gpu.dtype != torch.int32:
                    symbols_gpu = symbols_gpu.to(torch.int32)
                if probs_gpu.dtype != torch.float32:
                    probs_gpu = probs_gpu.to(torch.float32)
                mask_ptr = 0
                if mask is not None:
                    assert mask.is_cuda and mask.shape == (self.N,)
                    if mask.dtype != torch.uint8:
                        mask = mask.to(torch.uint8)
                    mask_ptr = int(mask.data_ptr())
                self._batch.encode_step_from_device(int(symbols_gpu.data_ptr()), int(probs_gpu.data_ptr()), mask_ptr)

            def finalize(self):
                self._batch.finalize()

            def get_compressed_list(self):
                return self._batch.get_compressed_host()

            def get_sizes_list(self):
                return self._batch.get_sizes_host()

            def init_decoder(self):
                self._batch.init_decoder_from_current_bytes()

            def decode_step(self, probs_gpu, out_symbols_gpu, mask=None):
                import torch
                assert probs_gpu.is_cuda and out_symbols_gpu.is_cuda
                assert probs_gpu.shape == (self.N, self.K) and out_symbols_gpu.numel() == self.N
                if probs_gpu.dtype != torch.float32:
                    probs_gpu = probs_gpu.to(torch.float32)
                mask_ptr = 0
                if mask is not None:
                    assert mask.is_cuda and mask.shape == (self.N,)
                    if mask.dtype != torch.uint8:
                        mask = mask.to(torch.uint8)
                    mask_ptr = int(mask.data_ptr())
                self._batch.decode_step_to_device(int(probs_gpu.data_ptr()), int(out_symbols_gpu.data_ptr()), mask_ptr)
