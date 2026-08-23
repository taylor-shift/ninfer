// ninfer::ops - dflash_selector wrapper: public api validation and launcher dispatch for the
// DFlash 2 path-selector family (dflash_selector_topk / _scores / _walk).
#include "ninfer/ops/dflash_selector.h"

#include "ops/launcher/dflash_selector.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kSelectorVocab = 248320;
constexpr std::int32_t kSelectorTopK  = 16;
constexpr std::int32_t kSelectorRank  = 256;
constexpr std::int32_t kSelectorMaxK  = 7;
constexpr std::int32_t kSelectorMaxB  = 8;

// The exact registered domain of the selector family: draft columns C = k*B with k in [1,7]
// and B in [1,8] (the anchor column is never part of C).
bool is_selector_columns(std::int32_t cols) {
    for (std::int32_t k = 1; k <= kSelectorMaxK; ++k) {
        for (std::int32_t b = 1; b <= kSelectorMaxB; ++b) {
            if (k * b == cols) { return true; }
        }
    }
    return false;
}

void require_matrix(const Tensor& t, std::int32_t rows, std::int32_t cols, DType dtype,
                    const char* op, const char* name) {
    if (t.dtype != dtype || t.ne[0] != rows || t.ne[1] != cols || t.ne[2] != 1 || t.ne[3] != 1 ||
        !t.is_contiguous() || t.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name +
                                    " must be contiguous with the exact registered shape");
    }
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    return lhs_begin < rhs_begin + rhs.bytes() && rhs_begin < lhs_begin + lhs.bytes();
}

} // namespace

void dflash_selector_topk(const Tensor& logits, Tensor& candidates, Tensor& unary,
                          cudaStream_t stream) {
    const char* op = "dflash_selector_topk";
    const std::int32_t cols = logits.ne[1];
    if (logits.ne[0] != kSelectorVocab || !is_selector_columns(cols)) {
        throw std::invalid_argument(std::string(op) + ": logits must be BF16 [248320,C] with "
                                                      "C = k*B, k in [1,7], B in [1,8]");
    }
    require_matrix(logits, kSelectorVocab, cols, DType::BF16, op, "logits");
    require_matrix(candidates, kSelectorTopK, cols, DType::I32, op, "candidates");
    require_matrix(unary, kSelectorTopK, cols, DType::FP32, op, "unary");
    if (overlaps(logits, candidates) || overlaps(logits, unary) || overlaps(candidates, unary)) {
        throw std::invalid_argument(std::string(op) + ": inputs and outputs must not overlap");
    }
    detail::dflash_selector_topk_launch(logits, candidates, unary, cols, stream);
}

void dflash_selector_scores(const Tensor& candidates, const Tensor& hidden_proj,
                            const Tensor& anchors, const Tensor& unary,
                            const Tensor& pred_codebook, const Tensor& succ_codebook,
                            Tensor& scores, cudaStream_t stream) {
    const char* op = "dflash_selector_scores";
    const std::int32_t b_count = anchors.ne[0];
    const std::int32_t k       = scores.ne[0];
    const std::int32_t cols    = candidates.ne[1];
    if (b_count < 1 || b_count > kSelectorMaxB) {
        throw std::invalid_argument(std::string(op) + ": anchors must be I32 [B] with B in [1,8]");
    }
    if (k < 1 || k > kSelectorMaxK || k * b_count != cols || !is_selector_columns(cols)) {
        throw std::invalid_argument(std::string(op) + ": k*B draft columns must match the "
                                                      "registered domain (k in [1,7], B in [1,8])");
    }
    require_matrix(candidates, kSelectorTopK, cols, DType::I32, op, "candidates");
    require_matrix(hidden_proj, kSelectorRank, cols, DType::BF16, op, "hidden_proj");
    require_matrix(anchors, b_count, 1, DType::I32, op, "anchors");
    require_matrix(unary, kSelectorTopK, cols, DType::FP32, op, "unary");
    require_matrix(pred_codebook, kSelectorVocab, kSelectorRank, DType::BF16, op, "pred_codebook");
    require_matrix(succ_codebook, kSelectorVocab, kSelectorRank, DType::BF16, op, "succ_codebook");
    if (scores.dtype != DType::FP32 || scores.ne[0] != k || scores.ne[1] != kSelectorTopK ||
        scores.ne[2] != kSelectorTopK || scores.ne[3] != b_count || !scores.is_contiguous() ||
        scores.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": scores must be contiguous FP32 [k,16,16,B]");
    }
    if (overlaps(candidates, scores) || overlaps(hidden_proj, scores) || overlaps(anchors, scores) ||
        overlaps(unary, scores) || overlaps(pred_codebook, scores) || overlaps(succ_codebook,
                                                                               scores)) {
        throw std::invalid_argument(std::string(op) + ": inputs and outputs must not overlap");
    }
    detail::dflash_selector_scores_launch(candidates, hidden_proj, anchors, unary, pred_codebook,
                                          succ_codebook, scores, k, b_count, stream);
}

void dflash_selector_walk(const Tensor& scores, const Tensor& candidates,
                          const SamplingConfig* configs, Tensor& drafts, cudaStream_t stream) {
    const char* op = "dflash_selector_walk";
    const std::int32_t k       = scores.ne[0];
    const std::int32_t b_count = scores.ne[3];
    const std::int32_t cols    = k * b_count;
    if (configs == nullptr) {
        throw std::invalid_argument(std::string(op) + ": configs must be a non-null host pointer");
    }
    if (k < 1 || k > kSelectorMaxK || b_count < 1 || b_count > kSelectorMaxB ||
        !is_selector_columns(cols)) {
        throw std::invalid_argument(std::string(op) + ": scores must be FP32 [k,16,16,B] with "
                                                      "k in [1,7] and B in [1,8]");
    }
    if (scores.dtype != DType::FP32 || scores.ne[1] != kSelectorTopK ||
        scores.ne[2] != kSelectorTopK || !scores.is_contiguous() || scores.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": scores must be contiguous FP32 [k,16,16,B]");
    }
    require_matrix(candidates, kSelectorTopK, cols, DType::I32, op, "candidates");
    require_matrix(drafts, cols, 1, DType::I32, op, "drafts");
    if (overlaps(scores, drafts) || overlaps(candidates, drafts)) {
        throw std::invalid_argument(std::string(op) + ": inputs and outputs must not overlap");
    }
    detail::dflash_selector_walk_launch(scores, candidates, configs, drafts, k, b_count, stream);
}

} // namespace ninfer::ops