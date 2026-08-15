"""
Vector Cmp
1. Case study on performance optimization of Triton operators on NPU, compared with GPU.
2. Ascend do not support Cmp on i32/i64, which leads to vector compute fall back into scalar
"""

import os
import time

import torch
import torch_npu
import triton
import triton.language as tl


def is_npu() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


@triton.jit
def gpu_vector_cmp_kernel(
    X,
    Out,
    Mean,
    Rstd,
    stride_x_row,
    stride_out_row,
    M,
    N,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    an example of layernorm to checkout Vector Cmp
    Out = ((X - E[X]) / sqrt(V[X] + eps)) on dim -1

    just for easy case, we assume that:
    1. BLOCK_N >= X.shape(-1), group_n = 0 only
    2. BLOCK_M = 1, group_m = range(0, row, 1)s
    """
    group_m = tl.program_id(axis=0)
    group_n = tl.program_id(axis=1)
    row = group_m

    Mean = Mean + group_m * M
    Rstd = Rstd + group_m * M
    X = X + row * stride_x_row + group_n * N
    Out = Out + row * stride_out_row + group_n * N

    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    tl.store(Mean + row, mean)
    xbar = tl.where(cols < N, x - mean, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)

    mask = cols < N
    out = (x - mean) * rstd
    tl.store(Out + cols, out, mask=mask)


@triton.jit
def npu_vector_cmp_kernel(
    X,
    Out,
    Mean,
    Rstd,
    stride_x_row,
    stride_out_row,
    M,
    N,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    NPU example shows how to use vec cmp for uppper gpu original triton
    """
    group_m = tl.program_id(axis=0)
    group_n = tl.program_id(axis=1)
    row = group_m

    Mean = Mean + group_m * M
    Rstd = Rstd + group_m * M
    X = X + row * stride_x_row + group_n * N
    Out = Out + row * stride_out_row + group_n * N

    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    tl.store(Mean + row, mean)

    cols_cmp = cols.to(tl.float32)
    xbar = tl.where(cols_cmp < N, x - mean, 0.0)

    var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)

    mask = cols_cmp < N
    out = (x - mean) * rstd
    tl.store(Out + cols, out, mask=mask)


def run(device="cuda"):
    def run_ref(X):
        mean = X.mean(dim=-1, keepdim=True)
        std = X.std(dim=-1, keepdim=True)
        out = (X.to(torch.float32) - mean) / (std + eps)
        return out.to(device_dtype)

    batch_size = 256
    feature_dim = 128
    eps = 1e-6

    if device == "npu":
        vector_cmp_kernel = npu_vector_cmp_kernel
        sync_device = torch.npu
        device_dtype = torch.float32
    else:
        vector_cmp_kernel = gpu_vector_cmp_kernel
        sync_device = torch.cuda
        device_dtype = torch.float32

    X = torch.randn(batch_size, feature_dim, device=device, dtype=device_dtype)
    Out = torch.empty_like(X)
    Mean = torch.empty((batch_size,), device=device, dtype=device_dtype)
    Rstd = torch.empty((batch_size,), device=device, dtype=device_dtype)

    BLOCK_M = 1
    BLOCK_N = triton.next_power_of_2(feature_dim)
    num_warps = min(max(1, BLOCK_N // 256), 8)
    vector_cmp_kernel[(batch_size // BLOCK_M, 1)](
        X,
        Out,
        Mean,
        Rstd,
        feature_dim,
        feature_dim,
        batch_size,
        feature_dim,
        eps,
        BLOCK_M,
        BLOCK_N,
        num_warps=num_warps,
    )
    sync_device.synchronize()

    spend_time = 0
    iterations = 100
    start_time = time.time()
    for i in range(iterations):
        vector_cmp_kernel[(batch_size // BLOCK_M, 1)](
            X,
            Out,
            Mean,
            Rstd,
            feature_dim,
            feature_dim,
            batch_size,
            feature_dim,
            eps,
            BLOCK_M,
            BLOCK_N,
            num_warps=num_warps,
        )
    sync_device.synchronize()
    spend_time += time.time() - start_time

    print(f"==== {device} spend_time: {spend_time / iterations * 1000} ms")

    Out_ref = run_ref(X)
    torch.testing.assert_close(Out, Out_ref, rtol=1e-3, atol=1e-3)
    print(f"==== {device} acc check passed!")


if __name__ == "__main__":
    if is_npu():
        run("npu")
    else:
        run("cuda")
