#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {
/**
 * Op: two-tap grouped dynamic causal convolution over the full DFlash 2 block columns.
 *
 * Engine layout convention: a contiguous tensor stores element (i0, i1) of shape (i, j) at
 * j*i0size + i (first dimension fastest; see set_contiguous_strides). Under this convention
 * hidden and out are contiguous BF16 [5120,C], dynamic is contiguous BF16 [640,C], and base is
 * contiguous BF16 [2,2,5120] ordered base[use][tap][channel]; the `use` argument (0 or 1)
 * selects which conv instance of the layer this call applies. `dynamic` is one use-slice of the
 * conv kernel_projection GEMM output [1280,C]: rows 0..319 hold the tap-0 coefficient of channel
 * group g = 0..319 and rows 320..639 hold the tap-1 coefficient, with g(c) = c/16 (320 groups of
 * 16 channels).
 *
 * The C block columns hold the whole draft block in batch-major order: column m = b*(k+1) + d
 * holds draft position d = 0..k (d = 0 is the row anchor) of batch row b = 0..B-1, where
 * B = batch_size and k = C/batch_size - 1. For every channel c and column t:
 *
 *   ideal[t,c] = (base[use,0,c] + dynamic[g(c),t])     * hidden[t,c]
 *              + (base[use,1,c] + dynamic[320+g(c),t]) * hidden[t-1,c],
 *
 * with hidden[t-1,c] read as zero at each row's anchor column (t % (k+1) == 0): each batch row is
 * an independent block-diffusion sequence and the conv never wraps across rows.
 *
 * FP32 accumulation with a single BF16 round-to-nearest-even at the end is the registered numeric
 * contract; the reference implementation (/code/dflash model.py `_grouped_dynamic_convolve`)
 * instead upcasts the BF16 base kernel to the hidden dtype and accumulates in BF16, so the FP32
 * arithmetic here is a deliberate hardening deviation.
 *
 * The registered domain is C = (k+1)*B for k in 1..7 draft tokens and B in 1..8 batch rows: the
 * exact 29-value subset of 2..64 (for example 11 and 13 are absent), never the whole interval,
 * and batch_size must equal the B of that factorization (C % batch_size == 0 and
 * C/batch_size in 2..8). Inputs are unchanged, out is the only observable mutation and is
 * completely overwritten, and hidden and out must not overlap. The Op owns no workspace or
 * persistent state; the kernel takes only a stream, performs no host synchronization, and is
 * graph-capturable.
 */
void grouped_dynamic_causal_conv(const Tensor& hidden, const Tensor& dynamic, const Tensor& base,
                                 Tensor& out, std::int32_t use, std::int32_t batch_size,
                                 cudaStream_t stream);

} // namespace ninfer::ops