import os
import time

import torch
import triton
import triton.language as tl


@triton.jit
def npu_vector_add_kernel(
    x,
    y,
    z,
    vector_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    len_mask = offset < vector_len
    x1 = tl.load(x + offset, mask=len_mask, other=0)
    y1 = tl.load(y + offset, mask=len_mask, other=0)
    z1 = x1 + y1
    tl.store(z + offset, z1, mask=len_mask)


def run(dtype_name):
    vector_len = 16384
    BLOCK_SIZE = 512
    BLOCK_DIM = 32
    device_name = "npu"

    x = torch.randint(0, 100, (1, vector_len), device=device_name, dtype=dtype_name)
    y = torch.randint(0, 100, (1, vector_len), device=device_name, dtype=dtype_name)
    z = torch.zeros((1, vector_len), device=device_name, dtype=dtype_name)
    npu_vector_add_kernel[(BLOCK_DIM,)](x, y, z, vector_len, BLOCK_SIZE)
    torch.npu.synchronize()

    spend_time = 0
    iterations = 100
    for i in range(iterations):
        start_time = time.time()
        npu_vector_add_kernel[(BLOCK_DIM,)](x, y, z, vector_len, BLOCK_SIZE)
        torch.npu.synchronize()
        spend_time += time.time() - start_time

    print(f"==== {dtype_name} spend_time: {spend_time / iterations * 1000} ms")


if __name__ == "__main__":
    run(torch.int64)
    run(torch.int32)
