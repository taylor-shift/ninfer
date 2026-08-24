// DFlash 2 (Qwen3.8-27B) QKV projection: one W8G32_F16S RowSplit parent [6144, 5120]
// (row order [query 4096, key 1024, value 1024]) written to three independent
// contiguous BF16 allocations in a single GEMM. The compute body is the shared
// W8 rowsplit MMA kernel (identical Cfg to the generic r64c128 linear route);
// only the split-output epilogue and the runtime m/k differ. CTA row tiles (BM)
// must not straddle a segment boundary, so both boundaries are asserted below.
#include "ops/linear/w8/w8_rowsplit_gemm_mma.cuh"

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
    w8_dflash_qkv_mma_launch(x, weight, q, k, v, stream);
}

} // namespace ninfer::ops::detail