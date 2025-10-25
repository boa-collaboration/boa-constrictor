from pathlib import Path
import struct, zlib, math, hashlib
import numpy as np
import torch

def BOA(device, filepath: str, model):
    IS_CUDA = device == "cuda" and torch.cuda.is_available()

    if IS_CUDA:
        from codec import compress_GPU as compress, decompress_GPU as decompress
        device = "cuda"
    else:
        from codec import compress_CPU as compress, decompress_CPU as decompress
        device = "cpu"

    def _uvarint_encode(x: int) -> bytes:
        out = bytearray()
        while True:
            b = x & 0x7F; x >>= 7
            out.append(b | (0x80 if x else 0))
            if not x: break
        return bytes(out)

    def _uvarint_decode(buf: memoryview, pos: int):
        x = 0; s = 0
        while True:
            b = buf[pos]; pos += 1
            x |= (b & 0x7F) << s
            if not (b & 0x80): return x, pos
            s += 7

    def _as_bytes(obj) -> bytes:
        if isinstance(obj, (bytes, bytearray)): return bytes(obj)
        if torch.is_tensor(obj):
            t = obj.detach().contiguous().to("cpu")
            if t.dtype != torch.uint8: t = t.view(torch.uint8)
            return t.numpy().tobytes()
        arr = np.asarray(obj)
        if arr.dtype != np.uint8: arr = arr.view(np.uint8)
        return arr.tobytes()

    def _pad4(b: bytes) -> bytes:
        r = len(b) & 3
        return b if r == 0 else (b + b"\x00" * (4 - r))

    class BoaFile:
        MAGIC = b'BOA2'
        IDX   = b'IDX1'
        VERSION = 1

        def __init__(self, filepath: str, model):
            self.filepath = Path(filepath)
            self.model = model
            self.compressed_data = []
            self.first_bytes = []
            self.lengths = []
            self.metadata = {}

        def _split_to_chunks(self, data_bytes: bytes, seq_size: int = 0, chunks_count: int = 0):
            n = len(data_bytes)
            if seq_size and chunks_count:
                chunk_len = int(seq_size); n_chunks = math.ceil(n / chunk_len)
            elif seq_size:
                chunk_len = int(seq_size); n_chunks = math.ceil(n / chunk_len)
            elif chunks_count:
                n_chunks = max(int(chunks_count), 1); chunk_len = math.ceil(n / n_chunks)
            else:
                raise ValueError("Provide either 'seq_size' or 'chunks_count'.")
            chunks = []
            for i in range(n_chunks):
                s = i * chunk_len; e = min(s + chunk_len, n)
                if s >= e: break
                arr = np.frombuffer(memoryview(data_bytes)[s:e], dtype=np.uint8).astype(np.int64)
                chunks.append(arr)
            last_len = len(chunks[-1]) if chunks else 0
            self.metadata = {
                'chunk_len': int(chunk_len),
                'n_chunks': int(len(chunks)),
                'uncompressed_len': int(n),
                'last_chunk_len': int(last_len if last_len else chunk_len),
            }
            return chunks, int(chunk_len)

        def _model_fingerprint(self) -> bytes:
            name = getattr(self.model, "__class__", type(self.model)).__name__
            return hashlib.blake2s(name.encode(), digest_size=16).digest()

        def _write_file(self, compressed_list, first_bytes, uncompressed_len, chunk_len, last_chunk_len):
            n = len(compressed_list); fp = self._model_fingerprint()
            with open(self.filepath, 'wb') as f:
                f.write(self.MAGIC)
                f.write(struct.pack('<I', self.VERSION))
                f.write(struct.pack('<I', 0))  # flags
                f.write(struct.pack('<Q', uncompressed_len))
                f.write(struct.pack('<I', chunk_len))
                f.write(struct.pack('<I', n))
                f.write(struct.pack('<I', last_chunk_len))
                f.write(struct.pack('<B', len(fp))); f.write(fp)

                offsets = []; off = 0
                for c in compressed_list:
                    offsets.append(off); f.write(c); off += len(c)

                idx = bytearray()
                idx += self.IDX
                idx += bytes(first_bytes)  # n bytes
                prev = 0
                for o in offsets: idx += _uvarint_encode(o - prev); prev = o
                for c in compressed_list: idx += _uvarint_encode(len(c))
                crc = zlib.crc32(idx) & 0xFFFFFFFF
                f.write(idx); f.write(struct.pack('<I', crc))

        def _read_file(self):
            data = Path(self.filepath).read_bytes()
            mm = memoryview(data); p = 0
            if bytes(mm[p:p+4]) != self.MAGIC: raise ValueError("Bad file magic")
            p += 4
            version, = struct.unpack_from('<I', mm, p); p += 4
            if version != self.VERSION: raise ValueError(f"Unsupported version {version}")
            _flags, = struct.unpack_from('<I', mm, p); p += 4
            ulen, = struct.unpack_from('<Q', mm, p); p += 8
            chunk_len, = struct.unpack_from('<I', mm, p); p += 4
            n, = struct.unpack_from('<I', mm, p); p += 4
            last_chunk_len, = struct.unpack_from('<I', mm, p); p += 4
            hlen = mm[p]; p += 1
            _fp = bytes(mm[p:p+hlen]); p += hlen

            crc = struct.unpack_from('<I', mm, len(mm)-4)[0]
            idx_pos = data.rfind(self.IDX)
            if idx_pos < 0: raise ValueError("Index not found")
            if (zlib.crc32(data[idx_pos:len(data)-4]) & 0xFFFFFFFF) != crc:
                raise ValueError("Bad index CRC")

            q = idx_pos + len(self.IDX)
            first_bytes = list(mm[q:q+n]); q += n

            offsets = [0]*n; pos = q; prev = 0
            for i in range(n):
                d, pos = _uvarint_decode(mm, pos); prev += d; offsets[i] = prev
            comp_lens = [0]*n
            for i in range(n):
                L, pos = _uvarint_decode(mm, pos); comp_lens[i] = L

            payload = mm[p:idx_pos]
            compressed_list = [bytes(payload[offsets[i]: offsets[i]+comp_lens[i]]) for i in range(n)]
            full_lens = [int(chunk_len)]*(n-1) + [int(last_chunk_len)]

            self.compressed_data = compressed_list          # bytes
            self.first_bytes = first_bytes                  # list[int]
            self.lengths = full_lens                        # uncompressed lens
            self.metadata = {
                'chunk_len': int(chunk_len),
                'n_chunks': int(n),
                'last_chunk_len': int(last_chunk_len),
                'uncompressed_len': int(ulen),
            }

        def compress(self, npz_path: str, seq_size: int = 0, chunks_count: int = 0, progress: bool = True):
            data = np.load(npz_path)['data']
            if isinstance(data, np.ndarray):
                data_bytes = data.tobytes() if data.dtype == np.uint8 else bytes(data)
            elif isinstance(data, (bytes, bytearray, memoryview)):
                data_bytes = bytes(data)
            else:
                raise TypeError("npz['data'] must be bytes-like or uint8 array")

            chunks_np, chunk_len = self._split_to_chunks(data_bytes, seq_size=seq_size, chunks_count=chunks_count)
            x_list = [torch.from_numpy(c).unsqueeze(0).to(device) for c in chunks_np]
            compressed_list, first_bytes, _ = compress(
                self.model, x_list, device=device, progress=progress
            )
            compressed_bytes = [_as_bytes(c) for c in compressed_list]
            first_bytes = [int(b) & 0xFF for b in first_bytes]

            ulen = len(data_bytes)
            last_chunk_len = self.metadata['last_chunk_len']
            self._write_file(compressed_bytes, first_bytes, ulen, chunk_len, last_chunk_len)

            self.compressed_data = compressed_bytes
            self.first_bytes = first_bytes
            self.lengths = [chunk_len]*(len(compressed_bytes)-1) + [last_chunk_len]
            print(f"Compression complete: {len(self.compressed_data)} chunks, chunk_len={chunk_len}, last={last_chunk_len}")

        def read_from_disk(self):
            self._read_file()
            print("File loaded successfully")

        def decompress(self, progress: bool = True) -> bytes:
            self._read_file()
            print(f"Total compressed size from disk: {sum(len(c) for c in self.compressed_data)} bytes")

            # Convert bytes -> uint32 with 4-byte padding for the GPU decoder
            comp_u32 = [np.frombuffer(_pad4(c), dtype=np.uint32).copy() for c in self.compressed_data]

            decoded_list = decompress(
                self.model, comp_u32, self.lengths, self.first_bytes, device=device, progress=progress
            )
            return b"".join(d.tobytes() if hasattr(d, "tobytes") else bytes(d) for d in decoded_list)

        def get_metadata(self):
            return dict(self.metadata)
        
    return BoaFile(filepath, model)