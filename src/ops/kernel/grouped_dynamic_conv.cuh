#pragma once

// ninfer::ops - two-tap grouped dynamic causal convolution kernel over contiguous BF16 block
// columns (DFlash 2 conv).

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

// Implements: include/ninfer/ops/grouped_dynamic_conv.h
// Match: contiguous BF16 in the engine layout convention (first dimension fastest: element (r, c)
// of [R,C] at c*R + r), C in the registered (k+1)*B domain, sm_120a.
// Layout: 5120 threads (20 CTAs of 256) over channels; thread c loops all C <= 64 columns. Each
// column iteration sweeps 32 consecutive channels per warp, so hidden/dynamic/out traffic
// coalesces. dynamic is the FULL [1280, C] kernel_projection GEMM output holding both uses:
// use u owns rows [u*640, u*640+640) (two taps x 320 groups), so the kernel strides columns by
// 1280 and offsets rows by use*640 (base is offset by use the same way). Each batch row is an
// independent block-diffusion sequence: the anchor column of a row (t % width == 0, width =
// C/batch_size) reads hidden[t-1,c] as zero and the conv never wraps across rows. All arithmetic
// accumulates in FP32; the single BF16 round happens at the store.
__global__ void grouped_dynamic_causal_conv_kernel(const __nv_bfloat16* hidden,
                                                   const __nv_bfloat16* dynamic,
                                                   const __nv_bfloat16* base, __nv_bfloat16* out,
                                                   std::int32_t columns, std::int32_t width,
                                                   std::int32_t use) {
    constexpr std::int32_t kChannels  = 5120;
    constexpr std::int32_t kGroups    = 320;
    constexpr std::int32_t kUseRows   = 2 * kGroups;  // 640: one use = two taps x 320 groups
    constexpr std::int32_t kDynStride = 2 * kUseRows; // 1280: the full dynamic holds both uses
    const std::int32_t c             = static_cast<std::int32_t>(blockIdx.x) * blockDim.x +
                                static_cast<std::int32_t>(threadIdx.x);
    if (c >= kChannels) { return; }

    const std::int32_t group = c / 16;
    const std::int32_t drow  = use * kUseRows;  // this use's row offset within the full buffer
    // base [2,2,5120] = [use][tap][channel]; element (u, tap, ch) at u + tap*2 + ch*4.
    const float base0 = __bfloat162float(base[use + 4 * c]);
    const float base1 = __bfloat162float(base[use + 2 + 4 * c]);

    for (std::int32_t t = 0; t < columns; ++t) {
        const std::int64_t col    = static_cast<std::int64_t>(t);
        const float x_cur         = __bfloat162float(hidden[col * kChannels + c]);
        const float x_prev        = (t % width == 0)
                                        ? 0.0f
                                        : __bfloat162float(hidden[(col - 1) * kChannels + c]);
        const float dyn0          = __bfloat162float(dynamic[col * kDynStride + drow + group]);
        const float dyn1          = __bfloat162float(
                            dynamic[col * kDynStride + drow + (kGroups + group)]);
        out[col * kChannels + c]  =
            __float2bfloat16_rn((base0 + dyn0) * x_cur + (base1 + dyn1) * x_prev);
    }
}

} // namespace ninfer::ops