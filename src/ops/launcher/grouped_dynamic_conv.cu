// ninfer::ops - grouped dynamic causal convolution launcher: 5120 threads over channels, all
// block columns per thread. The wrapper admits only the registered (k+1)*B domain, so the fixed
// launch geometry needs no extent-dependent dispatch.
#include "ops/launcher/grouped_dynamic_conv.h"

#include "core/device.h"
#include "ops/kernel/grouped_dynamic_conv.cuh"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kChannels = 5120;
constexpr std::int32_t kBlock    = 256;

} // namespace

void grouped_dynamic_causal_conv_launch(const Tensor& hidden, const Tensor& dynamic,
                                        const Tensor& base, Tensor& out, std::int32_t use,
                                        std::int32_t batch_size, cudaStream_t stream) {
    const std::int32_t columns = hidden.ne[1];
    const std::int32_t width   = columns / batch_size;
    grouped_dynamic_causal_conv_kernel<<<(kChannels + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(hidden.data),
        static_cast<const __nv_bfloat16*>(dynamic.data), static_cast<const __nv_bfloat16*>(base.data),
        static_cast<__nv_bfloat16*>(out.data), columns, width, use);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail