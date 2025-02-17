import torch
import inspect

from transformers import set_seed
import os
import time
import torch.nn as nn

from bitsandbytes.nn import Linear4bit
from transformers.activations import ACT2FN

import triton
import triton.language as tl

import pdb
from bitsandbytes.nn import Linear4bit
from transformers.activations import ACT2FN
from unsloth.kernels.utils import fast_dequantize

major_version, minor_version = torch.cuda.get_device_capability()
HAS_BFLOAT16 = (major_version >= 8)
from inspect import currentframe as _C, getframeinfo
_F = lambda c: getframeinfo(c).lineno # Gets line number
WARN = lambda x: print(f"\033[31m{x}\033[0m") # Red colored warnings

# https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string
def NAME(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    names = [var_name for var_name, var_val in callers_local_vars if var_val is var]
    return names[0] if len(names) != 0 else ""

def assert_same(x, y, line, dtype):
    assert(x.dtype == dtype)
    try: torch.testing.assert_close(x, y, check_stride = True)
    except Exception as error:
        raise RuntimeError(
            f"Failed allclose at line [{line}]: {NAME(x)}, {NAME(y)}\n{str(error)}"
        )

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def unsloth_dequantize(weight):
    return fast_dequantize(weight.weight, weight.weight.quant_state)

def bnb_Linear4bit(hd, m, dtype = torch.float16):
    return Linear4bit(
        hd, m, bias = None,
        compute_dtype       = dtype,
        compress_statistics = True,
        quant_type          = "nf4",
    )

class MLP(nn.Module):
    def __init__(self, hd = 4096, m = 14336, dtype = torch.float16):
        super().__init__()
        self.gate_proj = bnb_Linear4bit(hd, m, dtype = dtype)
        self.up_proj   = bnb_Linear4bit(hd, m, dtype = dtype)
        self.down_proj = bnb_Linear4bit(m, hd, dtype = dtype)
        self.act_fn = ACT2FN["silu"]
    def forward(self, x):
        print(self.up_proj.weight)
        print(unsloth_dequantize(self.up_proj))
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

def mlp_forward(X, mlp, fx):
    up   = X @ fx(mlp.  up_proj).t()
    print("ha\n", fx(mlp.  up_proj).t())
    gate = X @ fx(mlp.gate_proj).t()
    h = mlp.act_fn(gate) * up
    down = h @ fx(mlp.down_proj).t()
    return down

def mlp_dequantize(X, mlp, fx):
    a = fx(mlp.  up_proj).t(); torch.cuda.synchronize()
    b = fx(mlp.gate_proj).t(); torch.cuda.synchronize()
    c = fx(mlp.down_proj).t(); torch.cuda.synchronize()
    return a, b, c

def test_dequantize(dequantize_fx):
    elapsed = 0
    options = [
        (5,  777, 1024,  4096, 3409, torch.bfloat16),
        (3, 2048, 4096, 14336, 3408, torch.bfloat16),
        (2, 3333, 2048,  8192, 3407, torch.float16),
    ]
    for (bsz, qlen, hd, m, seed, dt) in options:
        set_seed(seed)
        torch.set_default_dtype(dt)
        mlp = MLP(hd = hd, m = m, dtype = dt).to("cuda")
        X = torch.randn((bsz, qlen, hd), device = "cuda")
        torch.cuda.synchronize()

        # Warmup
        for _ in range(2):
            assert_same( mlp_forward(X, mlp, dequantize_fx), mlp(X), _F(_C()), dt)
            a, b, c = mlp_dequantize(X, mlp, dequantize_fx)
            A, B, C = mlp_dequantize(X, mlp, unsloth_dequantize)
            assert_same(a, A, _F(_C()), dt)
            assert_same(b, B, _F(_C()), dt)
            assert_same(c, C, _F(_C()), dt)

        # Benchmarking
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(1000): mlp_dequantize(X, mlp, dequantize_fx)
        elapsed += time.time() - start
    return elapsed

lookup_table = [0.07958029955, 0.1609302014, 0.2461123019, 0.3379152417, 0.4407098293, 0.5626170039, 0.7229568362, 1.0]
lookup = torch.tensor(lookup_table).cuda()

@triton.jit
def _your_dequantize_nf4_kernel(
    a_ptr: tl.tensor,
    absmax_ptr: tl.tensor,
    out_ptr: tl.tensor,
    blocksize: tl.constexpr,
    n_elements: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    lookup_ptr: tl.tensor,
):
    pdb.set_trace()
    pid_m = tl.program_id(0)
    base_idx = pid_m * TILE_SIZE

    base_offsets = base_idx + tl.arange(0, TILE_SIZE)

    absmax_offsets = base_offsets // blocksize
    local_abs_max = tl.load(absmax_ptr + absmax_offsets)

    qvals_bytes = tl.load(a_ptr + base_offsets, mask=base_offsets < n_elements // 2, other=0.0)

    signs = tl.where((qvals_bytes & (1 << 3)) != 0, -1.0, 1.0)
    signs2 = tl.where((qvals_bytes & (1 << 7)) != 0, -1.0, 1.0)

    first_nibble  = qvals_bytes & 0b0111
    second_nibble = (qvals_bytes >> 4) & 0b0111

    # NOTE: tl.gather is not released yet

    # lookup = tl.load(lookup_ptr + tl.arange(0, 8))
    val0 = signs * tl.load(lookup_ptr + first_nibble) * local_abs_max
    val1 = signs2 * tl.load(lookup_ptr + second_nibble) * local_abs_max

    even_offsets = base_offsets * 2
    odd_offsets = even_offsets + 1

    tl.store(out_ptr + even_offsets, val0.to(tl.bfloat16), mask=even_offsets < n_elements)
    tl.store(out_ptr + odd_offsets, val1.to(tl.bfloat16), mask=odd_offsets < n_elements)

def clz(x):
    n = 32
    for i in range(n):
        if (x >> (n - 1 - i)) & 1:
            return i
    return n

def _your_dequantize_nf4(weight, quant_state):
    n_elements = weight.numel()
    TILE_SIZE = 128

    output = torch.empty(quant_state.shape, dtype=quant_state.dtype, device=weight.device).cuda()

    grid = (triton.cdiv(n_elements, TILE_SIZE),)
    _your_dequantize_nf4_kernel[grid](
        weight,
        quant_state.absmax,
        output,
        quant_state.blocksize,
        n_elements,
        TILE_SIZE,
        lookup,
    )
    return output

def your_dequantize_nf4(weight):
    return _your_dequantize_nf4(weight.weight.data, weight.weight.quant_state)

print(test_dequantize(your_dequantize_nf4) / test_dequantize(unsloth_dequantize))

