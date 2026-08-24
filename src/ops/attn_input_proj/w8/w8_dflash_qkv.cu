// DFlash 2 (Qwen3.8-27B) QKV projection: one W8G32_F16S RowSplit parent [6144, 5120]
// (row order [query 4096, key 1024, value 1024]) written to three independent
// contiguous BF16 allocations in a single GEMM. The compute body is the shared
// W8 rowsplit MMA kernel (identical Cfg to the generic r64c128 linear route);
// only the split-output epilogue and the runtime m/k differ. CTA row tiles (BM)
// must not straddle a segment boundary, so both boundaries are asserted below.
#include "ops/linear/w8/w8_rowsplit_gemm_mma.cuh"
#include "ops/linear/w8/w8_rowsplit_gemm_simt.cuh"

#include "core/device.h"
#include "core/tensor.h"
#include "ops/common/math.h"

#include <cuda_runtime.h>

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kQueryRows  = 4096;
constexpr int kKeyRows    = 1024;
constexpr int kValueRows  = 1024;
constexpr int kParentRows = kQueryRows + kKeyRows + kValueRows; // 6144
constexpr int kHidden     = 5120;

using Output = W8SplitOutput3<kQueryRows, kKeyRows, kValueRows>;
using Schedule = W8RowSplitMmaGemmSchedule<64, 128, 64, 16, 2>;

static_assert((kQueryRows % Schedule::BM) == 0);
static_assert(((kQueryRows + kKeyRows) % Schedule::BM) == 0);

template <bool Full>
void launch_variant(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                    cudaStream_t stream) {
    const Output output{static_cast<__nv_bfloat16*>(q.data),
                        static_cast<__nv_bfloat16*>(k.data),
                        static_cast<__nv_bfloat16*>(v.data)};
    const dim3 grid(kParentRows / Schedule::BM,
                    static_cast<unsigned>(div_up(x.ne[1], Schedule::BN)), 1u);
    w8_rowsplit_gemm_mma_kernel<Schedule, Full, W8Epilogue::Store, Output>
        <<<grid, Schedule::THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                                 static_cast<const std::uint8_t*>(weight.qdata),
                                                 static_cast<const std::uint8_t*>(weight.scales),
                                                 output, kParentRows, kHidden, x.ne[1], kHidden);
}

// SIMT variant for small T: fp32 dequant + fp32 FMA accumulation (no bf16 weight
// rounding), matching the accuracy profile the generic W8 linear dispatch uses for
// this [6144, 5120] shape at T <= 16. k = 5120 is 5 whole 1024-K slabs, no tail.
template <int ColsPerTile, bool Full>
void launch_simt_variant(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                         cudaStream_t stream) {
    constexpr int kRowsPerCta = 8;
    constexpr int kStages     = 2;
    static_assert((kQueryRows % kRowsPerCta) == 0 && (kKeyRows % kRowsPerCta) == 0);
    const Output output{static_cast<__nv_bfloat16*>(q.data),
                        static_cast<__nv_bfloat16*>(k.data),
                        static_cast<__nv_bfloat16*>(v.data)};
    const dim3 grid(kParentRows / kRowsPerCta,
                    static_cast<unsigned>(div_up(x.ne[1], ColsPerTile)), 1u);
    w8_rowsplit_gemm_simt_kernel<W8RowSplitSimtSchedule, ColsPerTile, kRowsPerCta, kStages, Full,
                                 W8Epilogue::Store, Output>
        <<<grid, kRowsPerCta * 32, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                                static_cast<const std::uint8_t*>(weight.qdata),
                                                static_cast<const std::uint8_t*>(weight.scales),
                                                output, kParentRows, kHidden, x.ne[1], kHidden,
                                                kHidden / 1024);
}

template <int ColsPerTile>
void launch_simt_route(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                       cudaStream_t stream) {
    if ((x.ne[1] % ColsPerTile) == 0) {
        launch_simt_variant<ColsPerTile, true>(x, weight, q, k, v, stream);
    } else {
        launch_simt_variant<ColsPerTile, false>(x, weight, q, k, v, stream);
    }
}

} // namespace

void w8_dflash_qkv_mma_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                              Tensor& v, cudaStream_t stream) {
    if (x.ne[1] < 1) {
        throw std::invalid_argument("W8 DFlash QKV requires T >= 1");
    }
    if ((x.ne[1] % Schedule::BN) == 0) {
        launch_variant<true>(x, weight, q, k, v, stream);
    } else {
        launch_variant<false>(x, weight, q, k, v, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

void w8_dflash_qkv_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                            Tensor& v, cudaStream_t stream) {
    // Mirror the generic W8 linear route boundaries for k=5120: SIMT (fp32-accurate)
    // for decode-scale T, MMA for throughput T.
    const std::int32_t tokens = x.ne[1];
    if (tokens >= 1 && tokens <= 4) {
        launch_simt_route<4>(x, weight, q, k, v, stream);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (tokens <= 16) {
        launch_simt_route<8>(x, weight, q, k, v, stream);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    w8_dflash_qkv_mma_launch(x, weight, q, k, v, stream);
}

} // namespace ninfer::ops::detail