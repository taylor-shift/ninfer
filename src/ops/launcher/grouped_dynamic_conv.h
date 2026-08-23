#pragma once

// ninfer::ops::detail - private launch prototype for grouped_dynamic_causal_conv.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void grouped_dynamic_causal_conv_launch(const Tensor& hidden, const Tensor& dynamic,
                                        const Tensor& base, Tensor& out, std::int32_t use,
                                        std::int32_t batch_size, cudaStream_t stream);

} // namespace ninfer::ops::detail