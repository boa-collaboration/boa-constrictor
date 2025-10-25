import numpy as np
import gpu_range_coder as gr
from concurrent.futures import ThreadPoolExecutor
import torch
from tqdm.auto import tqdm 

@torch.inference_mode()
def compress_GPU(
    model, x_list: list[torch.Tensor], device="cuda", progress=True, num_workers: int = 8
):
    """
    x_list: list of N tensors, each shaped [1, L_i] with uint8 in [0..255]
    Returns:
      compressed_list: list[np.ndarray(uint32)]
      first_bytes: list[int]
      lengths: list[int]
    """
    # Setup
    model.eval().to(device)
    N = len(x_list)
    assert N >= 1, "Need at least one chunk."
    xs = [x.to(device, dtype=torch.long) for x in x_list]
    for i, x in enumerate(xs):
        assert x.ndim == 2 and x.shape[0] == 1, f"Chunk {i} must be [1, L_i]"
        assert x.shape[1] >= 1, f"Chunk {i} must have length >= 1"

    Ls = [int(x.shape[1]) for x in xs]
    maxL = max(Ls)

    # Pack into one [N, maxL] for batched reads (on GPU)
    X = torch.zeros((N, maxL), dtype=torch.long, device=device)
    for i, x in enumerate(xs):
        X[i, :Ls[i]] = x[0]

    first_bytes = X[:, 0].tolist()
    lens_t = torch.tensor(Ls, device=device, dtype=torch.long)

    # GPU batch range encoder (no D2H for probs/symbols)
    # K = 256 for bytes
    batch = gr.gpu.queue.RangeCoderBatch(N, 256, maxL)

    # Streaming state
    inf = model.init_stream(max_len=maxL, batch_size=N, device=device, dtype=torch.float32)
    prev = X[:, 0].clone()  # [N] device

    total_steps = sum(L - 1 for L in Ls)
    pbar = tqdm(total=total_steps, disable=not progress, desc=f"Compress (GPU streams x{N})",
                unit="KB", unit_scale=1/1024, mininterval=0.2)

    # Encode timesteps t = 1..maxL-1
    for t in range(1, maxL):
        # Active lanes this step
        lens_mask = (lens_t > t)  # [N] bool on device
        if not torch.any(lens_mask):
            break

        # Compute probabilities on GPU for current prev
        logits = model.step(prev, inf)
        if logits.ndim == 3:
            logits = logits.squeeze(1)
        probs_gpu = torch.softmax(logits, dim=-1).to(torch.float32)

        # Symbols to encode this step (on GPU)
        syms = X[:, t].to(dtype=torch.int32)

        # Encode on GPU (masked; inactive lanes are skipped)
        # Note: encode_step supports optional mask: torch.bool [N]
        batch.encode_step(syms, probs_gpu, mask=lens_mask)

        # Update prev only for active lanes
        prev = torch.where(lens_mask, X[:, t], prev)

        # Progress: number of lanes still active at step t
        pbar.update(int(lens_mask.sum().item()))

    pbar.close()

    # Finalize on GPU and bring compressed outputs back as np.uint32 lists
    batch.finalize()
    compressed_list = batch.get_compressed_list()
    return compressed_list, first_bytes, Ls


def decompress_GPU(
    model, compressed_list, full_lens: list[int], first_bytes: list[int],
    device="cuda", progress=True, num_workers: int = 8
):
    """
    Returns: list[np.ndarray] each (1, L_i) uint8
    """
    with torch.inference_mode():
        model.eval().to(device)
        N = len(compressed_list)
        assert N >= 1 and len(full_lens) == N and len(first_bytes) == N
        assert all(L >= 1 for L in full_lens)

        maxL = max(full_lens)
        lens_t = torch.tensor(full_lens, device=device, dtype=torch.long)

        # Initialize GPU batch decoder from compressed streams (no D2H for probs)
        dec = gr.gpu.queue.RangeCoderBatch(N, 256, maxL)

        # Output buffer fully on GPU; we copy to host only at the end
        outs_gpu = torch.empty((N, maxL), dtype=torch.uint8, device=device)
        outs_gpu[:, 0] = torch.as_tensor(first_bytes, device=device, dtype=torch.uint8)

        # Streaming state
        inf = model.init_stream(max_len=maxL, batch_size=N, device=device, dtype=torch.float32)
        prev = torch.as_tensor(first_bytes, dtype=torch.long, device=device)

        total_steps = sum(L - 1 for L in full_lens)
        pbar = tqdm(total=total_steps, disable=not progress, desc=f"Decompress (GPU streams x{N})",
                    unit="KB", unit_scale=1/1024, mininterval=0.2)
        
        dec.load_compressed_list(compressed_list)
        dec.init_decoder()
        # Decode timesteps t = 1..maxL-1
        out_syms = torch.empty((N,), dtype=torch.int32, device=device)
        for t in range(1, maxL):
            lens_mask = (lens_t > t)
            if not torch.any(lens_mask):
                break

            # Compute probabilities on GPU for current prev
            logits = model.step(prev, inf)
            if logits.ndim == 3:
                logits = logits.squeeze(1)
            probs_gpu = torch.softmax(logits, dim=-1).to(torch.float32)

            # Decode on GPU into out_syms (masked lanes only)
            dec.decode_step(probs_gpu, out_syms, mask=lens_mask)

            # Write decoded symbols for active lanes and update prev
            outs_gpu[lens_mask, t] = out_syms[lens_mask].to(torch.uint8)
            prev = torch.where(lens_mask, out_syms.to(torch.long), prev)

            pbar.update(int(lens_mask.sum().item()))

        pbar.close()

        # Materialize outputs on host once
        outs = []
        for i in range(N):
            outs.append(outs_gpu[i, :full_lens[i]].detach().to("cpu").numpy().reshape(1, -1))
        return outs

@torch.inference_mode()
def compress_CPU(
    model, x_list: list[torch.Tensor], device="cpu", progress=True, num_workers: int = 8
):
    """
    x_list: list of N tensors, each shaped [1, L_i] with uint8 in [0..255]
    Returns:
      compressed_list: list[np.ndarray(uint32)]
      first_bytes: list[int]
      lengths: list[int]
    """
    # Setup
    model.eval().to(device)
    N = len(x_list)
    assert N >= 1, "Need at least one chunk."
    xs = [x.to(device, dtype=torch.long) for x in x_list]
    for i, x in enumerate(xs):
        assert x.ndim == 2 and x.shape[0] == 1, f"Chunk {i} must be [1, L_i]"
        assert x.shape[1] >= 1, f"Chunk {i} must have length >= 1"

    Ls = [int(x.shape[1]) for x in xs]
    maxL = max(Ls)
    # Pack into one [N, maxL] for batched reads
    X = torch.zeros((N, maxL), dtype=torch.long, device=device)
    for i, x in enumerate(xs):
        X[i, :Ls[i]] = x[0]

    first_bytes = X[:, 0].tolist()

    import constriction
    fam = constriction.stream.model.Categorical(perfect=False)
    encs = [constriction.stream.queue.RangeEncoder() for _ in range(N)]

    # Streaming state
    inf = model.init_stream(max_len=maxL, batch_size=N, device=device, dtype=torch.float32)
    prev = X[:, 0].clone()  # [N] device
    lens_t = torch.tensor(Ls, device=device, dtype=torch.long)

    # CUDA streams and double-buffered pinned host probs
    compute_stream = torch.cuda.Stream(device=device)
    transfer_stream = torch.cuda.Stream(device=device)
    prob_bufs = [
        torch.empty((N, 256), dtype=torch.float32, pin_memory=True),
        torch.empty((N, 256), dtype=torch.float32, pin_memory=True),
    ]
    copy_done = [torch.cuda.Event(enable_timing=False), torch.cuda.Event(enable_timing=False)]

    # Pre-copy symbols to CPU once for host encoders
    X_cpu = X.detach().to("cpu").numpy().astype(np.int32, copy=False)

    # Prime step 1 probabilities
    with torch.cuda.stream(compute_stream):
        logits = model.step(prev, inf)
        if logits.ndim == 3:
            logits = logits.squeeze(1)
        probs_gpu = torch.softmax(logits, dim=-1).to(torch.float32)

    with torch.cuda.stream(transfer_stream):
        transfer_stream.wait_stream(compute_stream)
        prob_bufs[0].copy_(probs_gpu, non_blocking=True)
        copy_done[0].record(transfer_stream)

    total_steps = sum(L - 1 for L in Ls)
    pbar = tqdm(total=total_steps, disable=not progress, desc=f"Compress (streams x{N})",
                unit="KB", unit_scale=1/1024, mininterval=0.2)

    def encode_range(r0, r1, t, probs_np):
        for i in range(r0, r1):
            if t < Ls[i]:
                sym = int(X_cpu[i, t])
                encs[i].encode(np.array([sym], dtype=np.int32), fam, probs_np[i:i+1, :])

    cur, nxt = 0, 1
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for t in range(1, maxL):
            # Wait for D2H completion of probs(t)
            copy_done[cur].synchronize()
            probs_np = prob_bufs[cur].numpy()

            # Parallel lane-wise encoding on CPU
            if num_workers > 1:
                chunk = (N + num_workers - 1) // num_workers
                futs = []
                s = 0
                while s < N:
                    e = min(s + chunk, N)
                    futs.append(pool.submit(encode_range, s, e, t, probs_np))
                    s = e
                for f in futs:
                    f.result()
            else:
                encode_range(0, N, t, probs_np)

            # Update device prev for active lanes (inference_mode still active)
            with torch.cuda.stream(compute_stream):
                lens_mask = (lens_t > t)
                prev = torch.where(lens_mask, X[:, t], prev)

                # Next step probs if any
                if t < maxL - 1:
                    logits = model.step(prev, inf)
                    if logits.ndim == 3:
                        logits = logits.squeeze(1)
                    probs_gpu = torch.softmax(logits, dim=-1).to(torch.float32)

            # D2H copy of next probs
            if t < maxL - 1:
                with torch.cuda.stream(transfer_stream):
                    transfer_stream.wait_stream(compute_stream)
                    prob_bufs[nxt].copy_(probs_gpu, non_blocking=True)
                    copy_done[nxt].record(transfer_stream)

            # Progress: number of lanes still active at step t
            pbar.update(int((np.asarray(Ls) > t).sum()))
            cur, nxt = nxt, cur

    pbar.close()
    compressed_list = [encs[i].get_compressed() for i in range(N)]
    return compressed_list, first_bytes, Ls


def decompress_CPU(
    model, compressed_list, full_lens: list[int], first_bytes: list[int],
    device="cpu", progress=True, num_workers: int = 8
):
    """
    Returns: list[np.ndarray] each (1, L_i) uint8
    """
    with torch.inference_mode():
        model.eval().to(device)
        N = len(compressed_list)
        assert N >= 1 and len(full_lens) == N and len(first_bytes) == N
        assert all(L >= 1 for L in full_lens)

        maxL = max(full_lens)

        def as_u32(comp):
            if isinstance(comp, np.ndarray) and comp.dtype == np.uint32:
                return comp
            elif isinstance(comp, np.ndarray) and comp.dtype == np.uint8:
                return comp.view(np.uint32)
            else:
                return np.frombuffer(np.asarray(comp).tobytes(), dtype=np.uint32)

        import constriction
        fam = constriction.stream.model.Categorical(perfect=False)
        decs = [constriction.stream.queue.RangeDecoder(as_u32(compressed_list[i])) for i in range(N)]
        outs = [np.empty(full_lens[i], dtype=np.uint8) for i in range(N)]
        for i in range(N):
            outs[i][0] = int(first_bytes[i])

        # Streaming state
        inf = model.init_stream(max_len=maxL, batch_size=N, device=device, dtype=torch.float32)
        prev = torch.tensor(first_bytes, dtype=torch.long, device=device)

        # CUDA streams and double-buffered pinned host probs
        compute_stream = torch.cuda.Stream(device=device)
        transfer_stream = torch.cuda.Stream(device=device)
        prob_bufs = [
            torch.empty((N, 256), dtype=torch.float32, pin_memory=True),
            torch.empty((N, 256), dtype=torch.float32, pin_memory=True),
        ]
        copy_done = [torch.cuda.Event(enable_timing=False), torch.cuda.Event(enable_timing=False)]

    # Keep prev on CPU to avoid H2D transfers every step
    prev_cpu = np.array(first_bytes, dtype=np.int32)
    
    # Pre-allocate GPU tensor for batch updates
    prev_gpu_buf = torch.empty(N, dtype=torch.long, device=device, pin_memory=False)

    with torch.inference_mode():
        # Prime step 1 probabilities
        with torch.cuda.stream(compute_stream):
            logits = model.step(prev, inf)
            if logits.ndim == 3:
                logits = logits.squeeze(1)
            probs_gpu = torch.softmax(logits, dim=-1).to(torch.float32)

        with torch.cuda.stream(transfer_stream):
            transfer_stream.wait_stream(compute_stream)
            prob_bufs[0].copy_(probs_gpu, non_blocking=True)
            copy_done[0].record(transfer_stream)

        total_steps = sum(L - 1 for L in full_lens)
        pbar = tqdm(total=total_steps, disable=not progress, desc=f"Decompress (streams x{N})",
                    unit="KB", unit_scale=1/1024, mininterval=0.2)
        lens_arr = np.asarray(full_lens, dtype=np.int64)

        def decode_range(r0, r1, t, probs_np):
            for i in range(r0, r1):
                if t < full_lens[i]:
                    sym = int(decs[i].decode(fam, probs_np[i:i+1, :])[0])
                    outs[i][t] = sym
                    prev_cpu[i] = sym  # Update CPU copy directly

        cur, nxt = 0, 1
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            for t in range(1, maxL):
                # Wait for D2H completion of probs(t)
                copy_done[cur].synchronize()
                probs_np = prob_bufs[cur].numpy()

                # Parallel lane-wise decode on CPU
                if num_workers > 1:
                    chunk = (N + num_workers - 1) // num_workers
                    futs, s = [], 0
                    while s < N:
                        e = min(s + chunk, N)
                        futs.append(pool.submit(decode_range, s, e, t, probs_np))
                        s = e
                    for f in futs:
                        f.result()
                else:
                    decode_range(0, N, t, probs_np)

                # Launch next step probs if any
                if t < maxL - 1:
                    with torch.cuda.stream(compute_stream):
                        # Single efficient H2D copy from CPU numpy to GPU
                        prev_gpu_buf.copy_(torch.from_numpy(prev_cpu).to(device, non_blocking=True))
                        prev = prev_gpu_buf  # Update reference
                        
                        logits = model.step(prev, inf)
                        if logits.ndim == 3:
                            logits = logits.squeeze(1)
                        probs_gpu = torch.softmax(logits, dim=-1).to(torch.float32)

                    with torch.cuda.stream(transfer_stream):
                        transfer_stream.wait_stream(compute_stream)
                        prob_bufs[nxt].copy_(probs_gpu, non_blocking=True)
                        copy_done[nxt].record(transfer_stream)

                pbar.update(int((lens_arr > t).sum()))
                cur, nxt = nxt, cur

        pbar.close()
        return [o.reshape(1, -1) for o in outs]
