#pragma once

// ninfer::ops::detail - private launch prototypes for the DFlash 2 path-selector family.

#include "core/tensor.h"
#include "ninfer/ops/sampling.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void dflash_selector_topk_launch(const Tensor& logits, Tensor& candidates, Tensor& unary,
                                 std::int32_t cols, cudaStream_t stream);

void dflash_selector_scores_launch(const Tensor& candidates, const Tensor& hidden_proj,
                                   const Tensor& anchors, const Tensor& unary,
                                   const Tensor& pred_codebook, const Tensor& succ_codebook,
                                   Tensor& scores, std::int32_t k, std::int32_t b_count,
                                   cudaStream_t stream);

void dflash_selector_walk_launch(const Tensor& scores, const Tensor& candidates,
                                 const SamplingConfig* configs, Tensor& drafts, std::int32_t k,
                                 std::int32_t b_count, cudaStream_t stream);

} // namespace ninfer::ops::detail