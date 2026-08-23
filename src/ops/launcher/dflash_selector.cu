// ninfer::ops - DFlash 2 path-selector launcher: fixed-shape launches over the exact C = k*B
// domains. All launches are graph-capturable: no host syncs and no device-side allocation; the
// walk's host SamplingConfig rows are read once here into kernel parameters.
#include "ops/launcher/dflash_selector.h"

#include "ops/kernel/dflash_selector.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cstdint>

namespace ninfer::ops::detail {

void dflash_selector_topk_launch(const Tensor& logits, Tensor& candidates, Tensor& unary,
                                 std::int32_t cols, cudaStream_t stream) {
    dflash_selector_topk_kernel<<<static_cast<unsigned int>(cols), kSelectorTopkBlock, 0,
                                  stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data),
        static_cast<std::int32_t*>(candidates.data), static_cast<float*>(unary.data),
        logits.ne[0]);
    CUDA_CHECK(cudaGetLastError());
}

void dflash_selector_scores_launch(const Tensor& candidates, const Tensor& hidden_proj,
                                   const Tensor& anchors, const Tensor& unary,
                                   const Tensor& pred_codebook, const Tensor& succ_codebook,
                                   Tensor& scores, std::int32_t k, std::int32_t b_count,
                                   cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned int>(b_count), static_cast<unsigned int>(k));
    dflash_selector_scores_kernel<<<grid, kSelectorScoresBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(candidates.data),
        static_cast<const __nv_bfloat16*>(hidden_proj.data),
        static_cast<const std::int32_t*>(anchors.data), static_cast<const float*>(unary.data),
        static_cast<const __nv_bfloat16*>(pred_codebook.data),
        static_cast<const __nv_bfloat16*>(succ_codebook.data), static_cast<float*>(scores.data),
        k);
    CUDA_CHECK(cudaGetLastError());
}

void dflash_selector_walk_launch(const Tensor& scores, const Tensor& candidates,
                                 const SamplingConfig* configs, Tensor& drafts, std::int32_t k,
                                 std::int32_t b_count, cudaStream_t stream) {
    // Read the host rows once into kernel parameters (graph-capturable; the walk consumes only
    // temperature and seed, spec doc 06 decision D8).
    float temperature[kSelectorMaxB];
    unsigned long long seed[kSelectorMaxB];
    for (int b = 0; b < b_count; ++b) {
        temperature[b] = configs[b].temperature;
        seed[b]        = configs[b].seed;
    }
    for (int b = b_count; b < kSelectorMaxB; ++b) {
        temperature[b] = 0.0f;
        seed[b]        = 0ull;
    }
    dflash_selector_walk_kernel<<<static_cast<unsigned int>(b_count), kSelectorWalkBlock, 0,
                                  stream>>>(
        static_cast<const float*>(scores.data), static_cast<const std::int32_t*>(candidates.data),
        temperature, seed, static_cast<std::int32_t*>(drafts.data), k);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail