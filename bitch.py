"""
Copyright 2025 [ghostway0](github.com/ghostway0)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import bitsandbytes.functional as F
import inspect

from transformers import set_seed
import os
import time
import torch.nn as nn

from bitsandbytes.nn import Linear4bit
from transformers.activations import ACT2FN

import triton
import triton.language as tl

from bitsandbytes.nn import Linear4bit
from transformers.activations import ACT2FN
from unsloth.kernels.utils import fast_dequantize

major_version, minor_version = torch.cuda.get_device_capability()
HAS_BFLOAT16 = (major_version >= 8)
from inspect import currentframe as _C, getframeinfo
_F = lambda c: getframeinfo(c).lineno # Gets line number
WARN = lambda x: print(f"\033[31m{x}\033[0m") # Red colored warnings

bnbdeq = lambda w: F.dequantize_nf4(w.weight, w.weight.quant_state)

# https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string
def NAME(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    names = [var_name for var_name, var_val in callers_local_vars if var_val is var]
    return names[0] if len(names) != 0 else ""

def assert_same(x, y, line=0, dtype=None):
    assert(dtype is None or x.dtype == dtype)
    try: torch.testing.assert_close(x, y, check_stride = True)
    except Exception as error:
        print(f"{x=}")
        print(f"{y=}")
        print(
            f"Failed allclose({line}): {NAME(x)}, {NAME(y)}\n{str(error)}"
        )

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def unsloth_dequantize(weight):
    return fast_dequantize(weight.weight, weight.weight.quant_state)

def bnb_Linear4bit(hd, m, dtype):
    return Linear4bit(
        hd, m, bias = None,
        compute_dtype       = dtype,
        compress_statistics = True,
        quant_type          = "nf4",
    )

class MLP(nn.Module):
    def __init__(self, hd, m, dtype):
        super().__init__()
        self.gate_proj = bnb_Linear4bit(hd, m, dtype = dtype).to("cuda")
        self.up_proj   = bnb_Linear4bit(hd, m, dtype = dtype).to("cuda")
        self.down_proj = bnb_Linear4bit(m, hd, dtype = dtype).to("cuda")
        self.gate_proj.weight.quant_state.dtype = dtype
        self.up_proj  .weight.quant_state.dtype = dtype
        self.down_proj.weight.quant_state.dtype = dtype
        self.up_proj.weight.quant_state.state2.code = self.up_proj.weight.quant_state.state2.code.to(torch.float32)
        self.gate_proj.weight.quant_state.state2.code = self.gate_proj.weight.quant_state.state2.code.to(torch.float32)
        self.down_proj.weight.quant_state.state2.code = self.down_proj.weight.quant_state.state2.code.to(torch.float32)
        self.up_proj.weight.quant_state.code = self.up_proj.weight.quant_state.state2.code.to(torch.float32)
        self.gate_proj.weight.quant_state.code = self.gate_proj.weight.quant_state.state2.code.to(torch.float32)
        self.down_proj.weight.quant_state.code = self.down_proj.weight.quant_state.state2.code.to(torch.float32)
        self.act_fn = ACT2FN["silu"]
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

def mlp_forward(X, mlp, fx):
    up   = X @ fx(mlp.  up_proj).t()
    gate = X @ fx(mlp.gate_proj).t()
    h = mlp.act_fn(gate) * up
    down = h @ fx(mlp.down_proj).t()
    return down

def mlp_dequantize(X, mlp, fx):
    a = fx(mlp.  up_proj).t(); torch.cuda.synchronize()
    b = fx(mlp.gate_proj).t(); torch.cuda.synchronize()
    c = fx(mlp.down_proj).t(); torch.cuda.synchronize()
    return a, b, c

def assert_correct_bnb(weight, dtype):
    assert(weight.weight.dtype == torch.uint8)
    assert(weight.weight.quant_state.dtype == dtype)
    assert(weight.weight.quant_state.absmax.dtype == torch.uint8)
    assert(weight.weight.quant_state.code.dtype == torch.float32)
    assert(weight.weight.quant_state.offset.dtype == torch.float32)
    assert(weight.weight.quant_state.blocksize == 64)
    assert(weight.weight.quant_state.state2.absmax.dtype == torch.float32)
    assert(weight.weight.quant_state.state2.code.dtype == torch.float32)
    assert(weight.weight.quant_state.state2.blocksize == 256)

        # (5,  777, 128,  128, 3409, torch.bfloat16),
        # (5,  777, 128,  128, 3409, torch.float16),
        # (5,  777, 5, 128, 3408, torch.float16),
        # (5,  777, 128,  128, 3408, torch.float32),
def test_dequantize(dequantize_fx):
    print(f"testing {dequantize_fx.__name__}")
    elapsed = 0
    options = [
        # (5,  777, 128,  4, 3409, torch.bfloat16),
        # (5,  777, 128,  4, 3409, torch.float16),
        (3, 2048, 14336, 14336, 3408, torch.bfloat16),
        (3, 2048, 14336, 14336, 3408, torch.float16),
        (2, 3333, 2048,  8192, 3407, torch.float16),
        (2, 3333, 2048,  8192, 3407, torch.bfloat16),
    ]
    for i, (bsz, qlen, hd, m, seed, dt) in enumerate(options):
        print(options[i])
        set_seed(seed)
        torch.set_default_dtype(torch.float32)
        mlp = MLP(hd = hd, m = m, dtype = dt).to("cuda")
        assert_correct_bnb(mlp.up_proj, dt)
        assert_correct_bnb(mlp.gate_proj, dt)
        assert_correct_bnb(mlp.down_proj, dt)
        X = torch.randn((bsz, qlen, hd), device = "cuda", dtype = dt)
        torch.cuda.synchronize()

        # Warmup
        for _ in range(2):
            a, b, c = mlp_dequantize(X, mlp, dequantize_fx)
            # A, B, C = mlp_dequantize(X, mlp, unsloth_dequantize)
            A, B, C = mlp_dequantize(X, mlp, bnbdeq)
            assert_same(a, A, _F(_C()), dt)
            assert_same(b, B, _F(_C()), dt)
            assert_same(c, C, _F(_C()), dt)

        # Benchmarking
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(1000): mlp_dequantize(X, mlp, dequantize_fx)
        elapsed += time.time() - start
    return elapsed

lookup_table = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]
lookup = torch.tensor(lookup_table).cuda()

@triton.jit
def _your_dequantize_nf4_kernel(
    code_ptr: tl.tensor,
    a_ptr: tl.tensor,
    absmax_ptr: tl.tensor,  # compressed absmax
    absmax2_ptr: tl.tensor, # absmax of the absmax
    out_ptr: tl.tensor,
    blocksize: tl.constexpr,
    n_elements: tl.constexpr,
    TILE_SIZE: tl.constexpr, # processes 2 * TILE_SIZE elements
    absmax_blocksize: tl.constexpr,
    absmax_nelems: tl.constexpr,
    absmax_offset: tl.constexpr,
    lookup_ptr: tl.tensor,
):
    pid_m = tl.program_id(0)
    base_idx = pid_m * TILE_SIZE

    base_offsets = base_idx + tl.arange(0, TILE_SIZE)

    absmax = tl.load(absmax2_ptr + base_offsets // (absmax_blocksize * blocksize))
    absmax_bytes = tl.load(absmax_ptr + base_offsets // blocksize)
    local_abs_max = tl.load(code_ptr + absmax_bytes) * absmax + absmax_offset

    qvals_bytes = tl.load(a_ptr + base_offsets, mask=base_offsets < n_elements // 2, other=0)

    first_nibble  = qvals_bytes & 0b1111
    second_nibble = (qvals_bytes >> 4) & 0b1111

    val0 = tl.load(lookup_ptr + first_nibble) * local_abs_max
    val1 = tl.load(lookup_ptr + second_nibble) * local_abs_max

    even_offsets = base_offsets * 2
    odd_offsets = even_offsets + 1

    tl.store(out_ptr + odd_offsets, val0, mask=odd_offsets < n_elements)
    tl.store(out_ptr + even_offsets, val1, mask=even_offsets < n_elements)

# TILE_SIZE 1024
def _your_dequantize_nf4(weight, quant_state, TILE_SIZE = 1024):
    output = torch.empty(quant_state.shape, dtype=quant_state.dtype, device=weight.device, requires_grad = False).cuda()

    grid = (triton.cdiv(output.numel() // 2 + TILE_SIZE - 1, TILE_SIZE),)
    _your_dequantize_nf4_kernel[grid](
        quant_state.state2.code,
        weight,
        quant_state.absmax,
        quant_state.state2.absmax,
        output,
        quant_state.blocksize // 2,
        output.numel(),
        TILE_SIZE,
        quant_state.state2.blocksize,
        quant_state.absmax.numel(),
        quant_state.offset.item(),
        lookup,
    )
    return output.t() if weight.shape[0] == 1 else output

def your_dequantize_nf4(weight):
    return _your_dequantize_nf4(weight.weight.data, weight.weight.quant_state)

if "DEBUG" in os.environ:
    bsz, qlen, hd, m, seed, dt = (2,  3333, 128, 128, 3407, torch.float16)
    # bsz, qlen, hd, m, seed, dt = (3, 2048, 14336, 14336, 3408, torch.float16)
    # bsz, qlen, hd, m, seed, dt =    (2, 3333, 2048,  8192, 3407, torch.float16)
    # bsz, qlen, hd, m, seed, dt = (5,  777, 256, 128, 3410, torch.bfloat16)
    #
    # bsz, qlen, hd, m, seed, dt =   (5,  777, 128, 4, 3409, torch.float16)

    set_seed(seed)
    torch.set_default_dtype(torch.float32)
    mlp = MLP(hd = hd, m = m, dtype = dt).to("cuda")
    X = torch.randn((bsz, qlen, hd), device = "cuda", dtype = dt)
    assert_correct_bnb(mlp.up_proj, dt)
    assert_correct_bnb(mlp.gate_proj, dt)
    assert_correct_bnb(mlp.down_proj, dt)
    torch.cuda.synchronize()

    a, b, c = mlp_dequantize(X, mlp, your_dequantize_nf4)
    A, B, C = mlp_dequantize(X, mlp, bnbdeq)

    fu = mlp_forward(X, mlp, your_dequantize_nf4)
    ft = mlp_forward(X, mlp, bnbdeq)
    assert_same(fu, ft, _F(_C()), dt)

    assert_same(a, A, _F(_C()), dt)
    assert_same(b, B, _F(_C()), dt)
    assert_same(c, C, _F(_C()), dt)
else:
    print(test_dequantize(unsloth_dequantize) / test_dequantize(your_dequantize_nf4))
