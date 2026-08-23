// ninfer::ops - DFlash 2 path-selector launcher: fixed-shape launches over the exact C = k*B
// domains. All launches are graph-capturable: no host syncs and no device-side allocation; the
// walk's host SamplingConfig rows are read once here into kernel parameters.
#include "ops/launcher/dflash_selector.h"

#include "ops/kernel/dflash_selector.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

// NINFER_LOG_OPS=1 (set by the acceptance script for GPU test runs): log every launch with
// its grid and pointer operands so a failure log says exactly what the kernel received.
const bool kLogLaunch = std::getenv("NINFER_LOG_OPS") != nullptr;
void launch_log(const std::string& m) {
    if (kLogLaunch) {
        std::fprintf(stderr, "[dflash_selector.launch] %s\n", m.c_str());
        std::fflush(stderr);
    }
}
std::string hex_ptr(const void* p) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%p", p);
    return std::string(buf);
}

} // namespace

namespace ninfer::ops::detail {

void dflash_selector_topk_launch(const Tensor& logits, Tensor& candidates, Tensor& unary,
                                 std::int32_t cols, cudaStream_t stream) {
    launch_log("topk: launch grid=(" + std::to_string(cols) + ",1,1) block=" +
               std::to_string(kSelectorTopkBlock) + " vocab=" + std::to_string(logits.ne[0]) +
               " logits=" + hex_ptr(logits.data) + " candidates=" + hex_ptr(candidates.data) +
               " unary=" + hex_ptr(unary.data) + " stream=" + hex_ptr(stream));
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
    launch_log("scores: launch grid=(" + std::to_string(b_count) + "," + std::to_string(k) +
               ",1) block=" + std::to_string(kSelectorScoresBlock) + " k=" + std::to_string(k) +
               " B=" + std::to_string(b_count) + " candidates=" + hex_ptr(candidates.data) +
               " hidden=" + hex_ptr(hidden_proj.data) + " anchors=" + hex_ptr(anchors.data) +
               " unary=" + hex_ptr(unary.data) + " pred=" + hex_ptr(pred_codebook.data) +
               " succ=" + hex_ptr(succ_codebook.data) + " scores=" + hex_ptr(scores.data) +
               " stream=" + hex_ptr(stream));
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
    // The per-row sampling config is passed BY VALUE as a kernel parameter (DFlashWalkParams,
    // 96-byte POD): array parameters decay to host pointers, which the device cannot
    // dereference. Graph-capturable: no device-side allocation (spec doc 06 decision D8).
    DFlashWalkParams params{};
    for (int b = 0; b < b_count; ++b) {
        params.temperature[b] = configs[b].temperature;
        params.seed[b]        = configs[b].seed;
    }
    std::string cfg;
    for (int b = 0; b < b_count; ++b) {
        cfg += " t" + std::to_string(b) + "=" + std::to_string(params.temperature[b]) +
               " seed" + std::to_string(b) + "=" + std::to_string(params.seed[b]) + ";";
    }
    launch_log("walk: launch grid=(" + std::to_string(b_count) + ",1,1) block=" +
               std::to_string(kSelectorWalkBlock) + " k=" + std::to_string(k) + " B=" +
               std::to_string(b_count) + " scores=" + hex_ptr(scores.data) +
               " candidates=" + hex_ptr(candidates.data) + " drafts=" + hex_ptr(drafts.data) +
               " stream=" + hex_ptr(stream) + cfg);
    dflash_selector_walk_kernel<<<static_cast<unsigned int>(b_count), kSelectorWalkBlock, 0,
                                  stream>>>(
        static_cast<const float*>(scores.data), static_cast<const std::int32_t*>(candidates.data),
        params, static_cast<std::int32_t*>(drafts.data), k);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail