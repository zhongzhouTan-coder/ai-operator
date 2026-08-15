"""Fused FlashAttention forward kernel for Ascend NPUs, written in Triton.

The goal of "fused" attention is to compute

    Attention(Q, K, V) = softmax(Q @ K^T * scale) @ V

without ever materializing the full (S x S) attention score matrix.
That matrix is O(S^2) in memory, which is why naive attention runs out of
memory on long sequences. FlashAttention-style tiling avoids it like this:

1. Split the query sequence into tiles of BLOCK_M rows.
2. For each query tile, stream over all key/value tiles of BLOCK_N rows.
3. Maintain a running "online softmax" state (running max `m`, running sum
   `l`, and the accumulated weighted values `acc`), updating it block by
   block with a numerically stable rescale.
4. Write the final output tile and the per-row log-sum-exp back to memory.

Two code paths handle the accumulator differently:

* H < 256  -> the (BLOCK_M, H) fp32 accumulator lives in registers, the fast
              path.
* H >= 256 -> the accumulator is too large for registers, so it is kept in a
              global scratch buffer and updated in 4 slices via CANN
              `extract_slice` / `insert_slice`.

The kernel is launched with a *fixed* grid of 20 programs (one per AI core on
the target NPU). Each program round-robins over all (batch, head, query-tile)
tasks, so the grid size does not depend on the input shape.

Notation: tensors are (B, N, S, H) = (batch, heads, sequence, head dim),
matching torch_npu's `BNSD` layout. `BLOCK_M` / `BLOCK_N` are the query /
key tile sizes (GEMM M/N) and are unrelated to the head count `N`.
"""

import pytest
import time
import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.extra.cann.extension as extension

DEVICE = "npu"


@triton.jit
def _attn_fwd_inner(
    acc_ptr,
    l_i,
    m_i,
    q,
    K_block_ptr,
    V_block_ptr,
    start_m,
    qk_scale,
    BLOCK_M: tl.constexpr,
    H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    S: tl.constexpr,
    fp8_v: tl.constexpr,
):
    # STAGE selects which region of the causal attention this call processes.
    # The caller (in `_attn_fwd`) maps the top-level STAGE value onto one of:
    #
    #   STAGE == 1 -> key blocks strictly *before* the diagonal: full attention,
    #                 no mask is needed (BLOCK_M >= BLOCK_N guaranteed).
    #   STAGE == 2 -> the *diagonal* block: positions where query < key must be
    #                 masked out (BLOCK_M <= BLOCK_N guaranteed).
    #   STAGE == 3 -> non-causal: process the whole sequence, no mask at all.
    #
    # `lo` / `hi` are the [lo, hi) range of key rows (in units of rows) that
    # this call is responsible for.
    if STAGE == 1:
        tl.static_assert(BLOCK_M >= BLOCK_N)
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        tl.static_assert(BLOCK_M <= BLOCK_N)
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)  # compiler hint: lo is block-aligned
    else:
        lo, hi = 0, S

    # Move the K/V block pointers to the first key tile in [lo, hi).
    K_block_ptr = tl.advance(K_block_ptr, (lo, 0))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    # Flattened (row, col) index used to address the global-memory accumulator
    # in the H >= 256 path. row * H + col turns a 2D tile into a 1D offset,
    # matching how `acc` is laid out in the scratch buffer.
    row = tl.arange(0, BLOCK_M)[:, None]
    col_head_dim = tl.arange(0, BLOCK_N)[None, :]
    block2d_acc = row * H + col_head_dim

    # Main inner loop: walk over the key/value tiles in [lo, hi).
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)  # compiler hint: block-aligned

        # ---- Step 1: compute the (BLOCK_M, BLOCK_N) attention scores ----
        k = tl.load(K_block_ptr)          # (BLOCK_N, H)
        trans_k = tl.trans(k)             # (H, BLOCK_N)
        qk = tl.dot(q, trans_k)           # scores Q @ K^T, shape (BLOCK_M, BLOCK_N)

        if STAGE == 2:
            # Causal diagonal tile: keep only query position >= key position.
            # Everything else is set to a large negative number (-1e6) so that
            # exp(...) below collapses it to 0.
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))  # running max over this tile
            qk -= m_ij[:, None]                    # subtract max for stability
        else:
            qk = qk * qk_scale
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]

        # ---- Step 2: exponentiate (unnormalized softmax numerator) ----
        p = tl.math.exp(qk)

        if fp8_v:
            # V is fp8: cast P to fp8 so the P @ V dot product can use the
            # NPU's fp8 tensor-core hardware.
            p_cast = tl.cast(p, tl.float8e5)
        else:
            p_cast = p.to(k.dtype)

        # ---- Step 3: load V and compute this tile's weighted value ----
        v = tl.load(V_block_ptr)          # (BLOCK_N, H)
        pv = tl.dot(p_cast, v)            # (BLOCK_M, H)

        # ---- Step 4: online-softmax running state update ----
        # The softmax denominator for the full row is built incrementally.
        # If the running max changes from m_i to m_ij, all previously
        # accumulated values must be rescaled by alpha = exp(m_i - m_ij).
        l_ij = tl.sum(p, 1)                      # sum of exp for this tile only
        alpha = tl.math.exp(m_i - m_ij)          # rescale factor for old state
        l_i = l_i * alpha + l_ij                 # running denominator

        if H < 256:
            # Register path: acc <- acc * alpha + P @ V.
            # The 3-operand `tl.dot(a, b, acc)` computes acc + a @ b, so we
            # first scale the old accumulator by alpha, then add the new tile.
            acc_ptr = acc_ptr * alpha[:, None]
            acc_ptr = tl.dot(p_cast, v, acc_ptr)
        else:
            # Global-memory path: a (BLOCK_M, H) fp32 tile is too large
            # for registers on this hardware. Load it from the scratch buffer
            # and update it in 4 row-slices to stay within register limits.
            acc = tl.load(acc_ptr + block2d_acc)
            for i in range(4):
                offset = i * (BLOCK_M // 4)
                # Slice out the i-th chunk of rows of the accumulator, the
                # matching chunk of alpha, and the matching chunk of P @ V.
                acc_i = extension.extract_slice(
                    acc, (offset, 0), (BLOCK_M // 4, H), (1, 1)
                )
                alpha_i = extension.extract_slice(alpha, [offset], (BLOCK_M // 4,), [1])
                pv_i = extension.extract_slice(
                    pv, (offset, 0), (BLOCK_M // 4, H), (1, 1)
                )

                # acc_i <- acc_i * alpha_i + pv_i  (same online-softmax update)
                acc_i = acc_i * alpha_i[:, None] + pv_i
                acc = extension.insert_slice(
                    acc, acc_i, (offset, 0), (BLOCK_M // 4, H), (1, 1)
                )
            tl.store(acc_ptr + block2d_acc, acc)

        m_i = m_ij  # running max is now the latest tile max

        # Advance K/V pointers to the next key tile.
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
    return acc_ptr, l_i, m_i


@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    M,
    Out,
    acc,
    sm_scale,
    stride_qb: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qs: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vs: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_on: tl.constexpr,
    stride_os: tl.constexpr,
    stride_oh: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
):
    # Number of query tiles along the sequence dimension, and the total number
    # of (batch, head, query-tile) tasks to process.
    NUM_BLOCKS_M = S // BLOCK_M
    NUM_BLOCKS = NUM_BLOCKS_M * B * N

    pid = tl.program_id(0)

    # The grid is fixed at 20 programs (one per AI core). Each program
    # round-robins over tasks `pid, pid+20, pid+40, ...`, which spreads work
    # evenly across cores without making the grid depend on the input shape.
    for block_idx in range(pid, NUM_BLOCKS, 20):
        # Decode the flat task id back into (b, n, m-block).
        task_bn_idx = block_idx // NUM_BLOCKS_M   # combined (b, n) index
        task_m_idx = block_idx % NUM_BLOCKS_M     # query-tile index along S
        off_b = task_bn_idx // N                  # batch index
        off_n = task_bn_idx % N                   # head index
        # Byte offset to the start of this (b, n) slice of Q/K/V/O.
        qkv_offset = off_b.to(tl.int64) * stride_qb + off_n.to(tl.int64) * stride_qn

        # Q tile: rows [task_m_idx*BLOCK_M, (task_m_idx+1)*BLOCK_M), all heads.
        Q_block_ptr = tl.make_block_ptr(
            base=Q + qkv_offset,
            shape=(S, H),
            strides=(stride_qs, stride_qh),
            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, H),
            order=(1, 0),
        )
        # K/V pointers start at row 0; the inner loop advances them by BLOCK_N.
        V_block_ptr = tl.make_block_ptr(
            base=V + qkv_offset,
            shape=(S, H),
            strides=(stride_vs, stride_vh),
            offsets=(0, 0),
            block_shape=(BLOCK_N, H),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + qkv_offset,
            shape=(S, H),
            strides=(stride_ks, stride_kh),
            offsets=(0, 0),
            block_shape=(BLOCK_N, H),
            order=(1, 0),
        )
        # O tile: same rows as the Q tile.
        O_block_ptr = tl.make_block_ptr(
            base=Out + qkv_offset,
            shape=(S, H),
            strides=(stride_os, stride_oh),
            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, H),
            order=(1, 0),
        )
        # Absolute row ids (for the causal mask) and relative column ids.
        offs_m = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)

        # Online-softmax running state.
        #   m_i = running max of the scores (starts at -inf)
        #   l_i = running sum of exp(scores - m_i) (the softmax denominator)
        # The initial l_i value is arbitrary: for the first tile alpha = 0,
        # so l_i simply becomes l_ij.
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0

        if H < 256:
            # Accumulator lives in registers for the fast path.
            acc_ptr = tl.zeros([BLOCK_M, H], dtype=tl.float32)
        else:
            # Accumulator lives in a global scratch buffer (shape B x N x S x H).
            # Compute the flat offset of this (b, n, m) slice. This relies on
            # the last strides being contiguous (stride_qs = 1).
            acc_offset = (
                off_b.to(tl.int64) * stride_qb // stride_qs * H
                + off_n.to(tl.int64) * stride_qn // stride_qs * H
                + task_m_idx * BLOCK_M * H
            )
            acc_ptr = acc + acc_offset

        q = tl.load(Q_block_ptr)

        # Split the causal mask into at most two passes over K/V:
        #   STAGE = 1 (non-causal)  -> one pass over the whole sequence (STAGE 3)
        #   STAGE = 3 (causal)      -> pass 1: blocks before the diagonal
        #                              pass 2: the diagonal block itself
        # The bit test `STAGE & 1` / `STAGE & 2` lets one value encode both
        # "which passes to run" and "which mask each pass needs".
        if STAGE & 1:
            acc_ptr, l_i, m_i = _attn_fwd_inner(
                acc_ptr,
                l_i,
                m_i,
                q,
                K_block_ptr,
                V_block_ptr,
                task_m_idx,
                sm_scale,
                BLOCK_M=BLOCK_M,
                H=H,
                BLOCK_N=BLOCK_N,
                STAGE=4 - STAGE,
                offs_m=offs_m,
                offs_n=offs_n,
                S=S,
                fp8_v=V.dtype.element_ty == tl.float8e5,
            )

        if STAGE & 2:
            acc_ptr, l_i, m_i = _attn_fwd_inner(
                acc_ptr,
                l_i,
                m_i,
                q,
                K_block_ptr,
                V_block_ptr,
                task_m_idx,
                sm_scale,
                BLOCK_M=BLOCK_M,
                H=H,
                BLOCK_N=BLOCK_N,
                STAGE=2,
                offs_m=offs_m,
                offs_n=offs_n,
                S=S,
                fp8_v=V.dtype.element_ty == tl.float8e5,
            )

        # Finalize the online softmax:
        #   logsumexp = m_i + log(l_i)   (this is exactly what a backward pass
        #                                needs, so it is stored into M)
        #   output    = acc / l_i        (normalize by the denominator)
        m_i += tl.math.log(l_i)
        if H < 256:
            accumulator = acc_ptr / l_i[:, None]
        else:
            # Reload the accumulator tile and normalize it in place.
            row = tl.arange(0, BLOCK_M)[:, None]
            col_head_dim = tl.arange(0, H)[None, :]
            block2d_acc = row * H + col_head_dim
            accumulator = tl.load(acc_ptr + block2d_acc)
            accumulator = accumulator / l_i[:, None]

        # M is laid out as (B * N, S): store one log-sum-exp per row.
        m_ptrs = M + task_bn_idx * S + offs_m

        tl.store(m_ptrs, m_i)
        tl.store(O_block_ptr, accumulator.to(Out.dtype.element_ty))


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, BM, BN):
        # All three heads must share one dimension for this fused kernel.
        H_Q, H_K = q.shape[-1], k.shape[-1]
        H_V = v.shape[-1]
        assert H_Q == H_K, "Head dimensions of Q and K must match"
        assert H_Q == H_V, "Head dimensions of Q and V must match"
        assert H_K in {16, 32, 64, 128, 256}, (
            "Head dimension must be one of [16, 32, 64, 128, 256]"
        )

        # `o` is the output; `M` holds the per-row log-sum-exp that a backward
        # pass would need (only `forward` is implemented in this file, but the
        # values are computed and saved for completeness).
        o = torch.empty_like(q, device=DEVICE)
        # 3 = causal (two passes), 1 = non-causal (single pass). See `_attn_fwd`.
        stage = 3 if causal else 1
        extra_kernel_args = {}

        num_cores = 20  # fixed grid: one program per AI core on the target NPU
        # Global scratch accumulator, only used when H >= 256.
        acc = torch.zeros(
            (q.shape[0], q.shape[1], q.shape[2], H_K),
            dtype=torch.float32,
            device=DEVICE,
        )
        M = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2]),
            dtype=torch.float32,
            device=DEVICE,
        )

        # Launch with a fixed grid of `num_cores` programs. Strides are passed
        # as constexpr so the compiler can specialize the kernel for each layout.
        _attn_fwd[(num_cores,)](
            q,
            k,
            v,
            M,
            o,
            acc,
            sm_scale,
            stride_qb=q.stride(0),
            stride_qn=q.stride(1),
            stride_qs=q.stride(2),
            stride_qh=q.stride(3),
            stride_kb=k.stride(0),
            stride_kn=k.stride(1),
            stride_ks=k.stride(2),
            stride_kh=k.stride(3),
            stride_vb=v.stride(0),
            stride_vn=v.stride(1),
            stride_vs=v.stride(2),
            stride_vh=v.stride(3),
            stride_ob=o.stride(0),
            stride_on=o.stride(1),
            stride_os=o.stride(2),
            stride_oh=o.stride(3),
            B=q.shape[0],
            N=q.shape[1],
            S=q.shape[2],
            H=H_K,
            BLOCK_M=BM,
            BLOCK_N=BN,
            STAGE=stage,
            **extra_kernel_args,
        )

        # Save inputs for a backward pass (not implemented here).
        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.H = H_K
        ctx.causal = causal
        return o


attention = _attention.apply


@pytest.mark.parametrize(
    "B, N, S, H, causal, dtype, BM, BN",
    [
        (1, 1, 128, 128, False, torch.float16, 32, 128),
        (1, 1, 128, 128, False, torch.bfloat16, 64, 128),
        (1, 2, 256, 256, False, torch.bfloat16, 32, 256),
        (2, 2, 128, 256, False, torch.float16, 64, 128),
        (4, 32, 64, 64, False, torch.float16, 32, 64),
        (4, 32, 1024, 64, False, torch.bfloat16, 64, 128),
        (4, 32, 4096, 64, False, torch.float16, 128, 128),
    ],
)
def test_op(B, N, S, H, causal, dtype, BM, BN):
    # The kernel requires S to be divisible by both tile sizes and H to be a
    # multiple of 16 (hardware dot-product alignment).
    if S % BM != 0 or S % BN != 0 or H % 16 != 0:
        pytest.skip("Skipping non-divisible case")

    torch.manual_seed(20)
    q = (
        torch.empty((B, N, S, H), dtype=dtype, device=DEVICE)
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    k = (
        torch.empty((B, N, S, H), dtype=dtype, device=DEVICE)
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    v = (
        torch.empty((B, N, S, H), dtype=dtype, device=DEVICE)
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    sm_scale = 0.5  # fixed softmax scale (in production: 1 / sqrt(H))

    # --- Time and run the Triton kernel ---
    tri_start_time = time.time()
    tri_out = attention(q, k, v, causal, sm_scale, BM, BN)
    tri_end_time = time.time()
    print(
        f"[TRITON] Attention shape:({B}, {N}, {S}, {H}), BM: {BM}, BN: {BN}, dtype: {dtype}, time: {tri_end_time - tri_start_time:.6f}s"
    )

    # --- Reference: torch_npu's fused attention ---
    # input_layout "BNSD" = (Batch, heads, Seq, head_Dim), matching q's shape.
    # pre_tokens / next_tokens = 65535 means "no causal window" (full context).
    # sparse_mode = 0 disables the sparse-attention optimization.
    ref_start_time = time.time()
    ref_out = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        N,
        padding_mask=None,
        atten_mask=None,
        scale=sm_scale,
        keep_prob=1.0,
        input_layout="BNSD",
        pre_tokens=65535,
        next_tokens=65535,
        sparse_mode=0,
    )[0]
    ref_end_time = time.time()
    print(
        f"[REFERENCE] Attention shape:({B}, {N}, {S}, {H}), BM: {BM}, BN: {BN}, dtype: {dtype}, time: {ref_end_time - ref_start_time:.6f}s"
    )

    # Compare with a loose tolerance: fp16/bf16 tile accumulation introduces
    # small numerical differences vs. the reference implementation.
    torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=1e-2, equal_nan=True)
    print(
        f"[PASSED] Attention shape:({B}, {N}, {S}, {H}), BM: {BM}, BN: {BN}, dtype: {dtype}"
    )


if __name__ == "__main__":
    test_op(1, 1, 128, 128, causal=False, dtype=torch.float16, BM=32, BN=128)
    test_op(1, 1, 128, 128, causal=False, dtype=torch.bfloat16, BM=64, BN=128)
    test_op(1, 2, 256, 256, causal=False, dtype=torch.bfloat16, BM=32, BN=256)
    test_op(2, 2, 128, 256, causal=False, dtype=torch.float16, BM=64, BN=128)
    test_op(4, 32, 64, 64, causal=False, dtype=torch.float16, BM=32, BN=64)
    test_op(4, 32, 1024, 64, causal=False, dtype=torch.bfloat16, BM=64, BN=128)
    test_op(4, 32, 4096, 64, causal=False, dtype=torch.float16, BM=128, BN=128)
