import time

import torch
import triton
import triton.language as tl
import torch_npu


@triton.jit
def _layer_norm_fwd_fused(
    X, Y, W, B, Mean, Rstd, stride, N, eps, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    Y += pid * stride
    X += pid * stride

    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N

    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    tl.store(Mean + pid, mean)
    tl.store(Rstd + pid, rstd)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)


@torch.inference_mode()
def layer_norm(x, weight, bias, eps=1e-5):
    y = torch.empty_like(x)

    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape

    mean = torch.empty((M,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)

    BLOCK_SIZE = 1024

    kernel = _layer_norm_fwd_fused[(M,)](
        x_arg,
        y,
        weight,
        bias,
        mean,
        rstd,
        x_arg.stride(0),
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return y


def _layer_norm(M, N, dtype, eps=1e-5, device="npu"):
    x_shape = (M, N)
    w_shape = (x_shape[-1],)
    weight = torch.randn(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.randn(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = 0.1 * torch.randn(x_shape, dtype=dtype, device=device)
    x.requires_grad_(True)
    _ = layer_norm(x, weight, bias, eps)  # warmup
    tri_start_time = time.time()
    y_tri = layer_norm(x, weight, bias, eps)
    tri_end_time = time.time()
    print(
        f"Layer Normalization {M},{N} {dtype} time: {tri_end_time - tri_start_time:.6f} seconds"
    )
    ref_start_time = time.time()
    y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps=eps).to(dtype)
    ref_end_time = time.time()
    print(
        f"Reference Layer Normalization {M},{N} {dtype} time: {ref_end_time - ref_start_time:.6f} seconds"
    )

    assert torch.allclose(y_tri, y_ref, atol=1e-5, rtol=0)
    print(f"y_tri: {y_tri}")
    print(f"y_ref: {y_ref}")
    print(f"Layer Normalization {M},{N} {dtype} PASSED!")


if __name__ == "__main__":
    _layer_norm(128, 128, torch.float16)
    _layer_norm(128, 128, torch.bfloat16)
    _layer_norm(128, 128, torch.float32)
