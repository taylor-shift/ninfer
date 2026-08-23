#pragma once

// Implements: include/ninfer/ops/dflash_selector.h
// Match: contiguous BF16/FP32/I32 selector buffers over the exact C = k*B domains (k in [1,7],
// B in [1,8]); fixed 248320-row vocab, 16 candidates, 256-rank codebooks.
//
// Layout law (src/core/tensor.cpp set_contiguous_strides): every contiguous Tensor has
// dimension 0 FASTEST, so element (i0, i1) of [n0, n1] sits at flat element offset i0 + i1*n0
// and each column of a [n0,C] matrix is a contiguous n0-element block at col*n0:
//   logits [248320,C]  element (v, col)    at col*248320 + v
//   candidates [16,C]  element (rank, col) at col*16 + rank
//   unary [16,C]       element (rank, col) at col*16 + rank
//   hidden_proj [256,C] element (d, col)   at col*256 + d
//   scores [k,16,16,B] element (t,p,c,b)   at t + p*k + c*k*16 + b*k*16*16 (t fastest, b slowest)
//   flat_drafts [k*B]  element (t, b)      at b*k + t  (the batch-major view of draft_tokens [k,B])
//   anchors [B]        element (b)         at b
// The engine's packed draft columns (logits/hidden_proj GEMMs and the selector matrices) are
// BATCH-MAJOR: column m = b*k + t with batch row b slowest and 0-based draft position t fastest,
// so every column loop enumerates b outer / t inner. The codebooks are [248320,256] with the
// token dimension fastest: element (v, d) at v + d*248320, so a token's 256-rank row is strided
// and is gathered one d per thread.
//
// Algorithm assumptions: the topk uses one 256-thread CTA per column with a register top-16
// per thread folded by thread 0 (the sampling fast-path merge idiom); the scores use one CTA
// per (t,b) pair with the codebook rows and hidden column upcast to FP32 in shared memory; the
// walk is one CTA per batch row, lane 0 running the <=7 sequential steps while lanes 1..31
// exit early.

#include "ops/kernel/sampling_device.cuh"

#include <cstdint>

namespace ninfer::ops {

inline constexpr std::int32_t kSelectorVocab   = 248320;
inline constexpr std::int32_t kSelectorTopK    = 16;
inline constexpr std::int32_t kSelectorRank    = 256;
inline constexpr std::int32_t kSelectorMaxK    = 7;
inline constexpr std::int32_t kSelectorMaxB    = 8;
inline constexpr int kSelectorTopkBlock        = 256;
inline constexpr int kSelectorScoresBlock      = 256;
inline constexpr int kSelectorWalkBlock        = 32;

// Per-column top-16 with the sampling op's ordering (value descending, lower id on ties, the
// smaller id occupying the higher-rank slot). 248320 = 256*970, so every thread owns a full
// 970-row strided vocab slice; each thread keeps a register top-16, the slice tops are staged
// to shared memory, and thread 0 folds them into the column top-16. The vocab is far larger
// than the 16 slots, so no sentinel slot survives to the output. Column col is the contiguous
// 248320-element block at col*vocab, so the strided read is coalesced.
__launch_bounds__(kSelectorTopkBlock) __global__
    void dflash_selector_topk_kernel(const __nv_bfloat16* logits, std::int32_t* candidates,
                                     float* unary, std::int32_t vocab) {
    const int col = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);

    float local_val[kSelectorTopK];
    std::int32_t local_idx[kSelectorTopK];
#pragma unroll
    for (int j = 0; j < kSelectorTopK; ++j) {
        local_val[j] = -CUDART_INF_F;
        local_idx[j] = INT32_MAX;
    }

    for (int v = tid; v < vocab; v += kSelectorTopkBlock) {
        const float value =
            __bfloat162float(logits[static_cast<std::int64_t>(col) * vocab + v]);
        sampling_insert_candidate(local_val, local_idx, kSelectorTopK, value, v);
    }

    __shared__ float merge_val[kSelectorTopkBlock * kSelectorTopK];
    __shared__ std::int32_t merge_idx[kSelectorTopkBlock * kSelectorTopK];
#pragma unroll
    for (int j = 0; j < kSelectorTopK; ++j) {
        merge_val[tid * kSelectorTopK + j] = local_val[j];
        merge_idx[tid * kSelectorTopK + j] = local_idx[j];
    }
    __syncthreads();

    if (tid != 0) { return; }

    float cand_val[kSelectorTopK];
    std::int32_t cand_idx[kSelectorTopK];
#pragma unroll
    for (int j = 0; j < kSelectorTopK; ++j) {
        cand_val[j] = -CUDART_INF_F;
        cand_idx[j] = INT32_MAX;
    }
    for (int p = 0; p < kSelectorTopkBlock * kSelectorTopK; ++p) {
        const std::int32_t idx = merge_idx[p];
        if (idx == INT32_MAX) { continue; }
        sampling_insert_candidate(cand_val, cand_idx, kSelectorTopK, merge_val[p], idx);
    }
    // Column col's 16 ranks are the contiguous blocks col*16 of the [16,C] outputs.
#pragma unroll
    for (int j = 0; j < kSelectorTopK; ++j) {
        candidates[static_cast<std::int64_t>(col) * kSelectorTopK + j] = cand_idx[j];
        unary[static_cast<std::int64_t>(col) * kSelectorTopK + j]      = cand_val[j];
    }
}

// One CTA per (t,b) draft-position/batch-row pair (grid (b, t)); the draft column is the
// batch-major m = b*k + t. The 16 predecessor rows, 16 successor rows, and hidden column are
// upcast BF16->FP32 into shared memory; the row stride is padded by two floats so the p- and
// c-strided contraction reads stay bank-conflict-free. Each of the 256 threads then contracts
// exactly one (p,c) edge over the 256-rank space.
__launch_bounds__(kSelectorScoresBlock) __global__
    void dflash_selector_scores_kernel(const std::int32_t* candidates,
                                       const __nv_bfloat16* hidden_proj,
                                       const std::int32_t* anchors, const float* unary,
                                       const __nv_bfloat16* pred_codebook,
                                       const __nv_bfloat16* succ_codebook, float* scores,
                                       std::int32_t k) {
    constexpr int kTopK   = kSelectorTopK;
    constexpr int kRank   = kSelectorRank;
    constexpr int kRowPad = kRank + 2;

    const int b   = static_cast<int>(blockIdx.x);
    const int t   = static_cast<int>(blockIdx.y);
    const int col = b * k + t; // batch-major draft column (b slow, t fast)
    const int tid = static_cast<int>(threadIdx.x);

    __shared__ float a_sh[kTopK][kRowPad];
    __shared__ float b_sh[kTopK][kRowPad];
    __shared__ float h_sh[kRank];

    // Hidden column: the contiguous 256-element block at col*256 of [256,C].
    h_sh[tid] = __bfloat162float(hidden_proj[static_cast<std::int64_t>(col) * kRank + tid]);

    // Predecessor row per p-slot: the anchor row of batch row b for every p at t = 0 (all 16
    // slots materialized so the walk can read any of them), otherwise slot p's candidate token
    // of the previous draft column (col - 1 under the batch-major order); the col - 1 read
    // happens only on the t > 0 branch.
    for (int p = 0; p < kTopK; ++p) {
        const std::int32_t pred_id =
            (t == 0) ? anchors[b] : candidates[static_cast<std::int64_t>(col - 1) * kTopK + p];
        // Codebook [vocab,rank] with the token dimension fastest: (pred_id, d) at
        // pred_id + d*vocab. Thread tid supplies d = tid.
        a_sh[p][tid] = __bfloat162float(
            pred_codebook[static_cast<std::int64_t>(pred_id) +
                          static_cast<std::int64_t>(tid) * kSelectorVocab]);
    }
    // Successor row per c-slot: candidate c of the current draft column.
    for (int c = 0; c < kTopK; ++c) {
        const std::int32_t cand_id = candidates[static_cast<std::int64_t>(col) * kTopK + c];
        b_sh[c][tid] = __bfloat162float(
            succ_codebook[static_cast<std::int64_t>(cand_id) +
                          static_cast<std::int64_t>(tid) * kSelectorVocab]);
    }
    __syncthreads();

    const int p = tid & 15; // 0..15
    const int c = tid >> 4; // 0..15
    float acc = 0.0f;
#pragma unroll
    for (int d = 0; d < kRank; ++d) {
        acc += a_sh[p][d] * h_sh[d] * b_sh[c][d];
    }
    const float unary_c = unary[static_cast<std::int64_t>(col) * kTopK + c];
    // scores element (t,p,c,b) at t + p*k + c*k*16 + b*k*16*16 (t fastest, b slowest).
    scores[static_cast<std::int64_t>(t) + static_cast<std::int64_t>(p) * k +
           static_cast<std::int64_t>(c) * k * kTopK +
           static_cast<std::int64_t>(b) * k * kTopK * kTopK] = unary_c + acc;
}

// One CTA per batch row; lane 0 runs the k <= 7 sequential steps (a 16-slot row makes
// per-lane parallel work immaterial), lanes 1..31 exit early. Greedy steps argmax over the 16
// slots with the lower candidate token id breaking ties; sampled steps draw one multinomial
// over softmax(row/temperature) via the counter-based RNG (purpose
// kSamplePurposeDflashSelector, subkeyed per (row, draft position)) and the sampling op's
// inverse-CDF convention.
__launch_bounds__(kSelectorWalkBlock) __global__
    void dflash_selector_walk_kernel(const float* scores, const std::int32_t* candidates,
                                     const float temperature[kSelectorMaxB],
                                     const unsigned long long seed[kSelectorMaxB],
                                     std::int32_t* drafts, std::int32_t k) {
    constexpr int kTopK = kSelectorTopK;
    const int b        = static_cast<int>(blockIdx.x);
    if (static_cast<int>(threadIdx.x) != 0) { return; }

    const float temperature_b      = temperature[b];
    const unsigned long long seed_b = seed[b];
    const std::int64_t row_b_stride = static_cast<std::int64_t>(k) * kTopK * kTopK;
    int pred                       = 0; // the anchor slot
    for (int t = 0; t < k; ++t) {
        const int col = b * k + t; // batch-major draft column
        float row[kTopK];
        std::int32_t ids[kTopK];
        const std::int64_t row_base =
            static_cast<std::int64_t>(t) + static_cast<std::int64_t>(b) * row_b_stride;
#pragma unroll
        for (int c = 0; c < kTopK; ++c) {
            // scores element (t,pred,c,b) at t + pred*k + c*k*16 + b*k*16*16.
            row[c] = scores[row_base + static_cast<std::int64_t>(pred) * k +
                            static_cast<std::int64_t>(c) * k * kTopK];
            // candidates element (rank=c, col) at col*16 + c.
            ids[c] = candidates[static_cast<std::int64_t>(col) * kTopK + c];
        }

        int idx;
        if (temperature_b <= 0.0f) {
            float best_value = -CUDART_INF_F;
            std::int32_t best_id = INT32_MAX;
            idx                  = 0;
#pragma unroll
            for (int c = 0; c < kTopK; ++c) {
                if (sampling_better(row[c], ids[c], best_value, best_id)) {
                    best_value = row[c];
                    best_id    = ids[c];
                    idx        = c;
                }
            }
        } else {
            // softmax(row[c]/temperature): divide first (the reference's exact operation
            // order), then shift by the max and normalize through the inverse-CDF pick.
            float scaled[kTopK];
            float max_v = -CUDART_INF_F;
#pragma unroll
            for (int c = 0; c < kTopK; ++c) {
                scaled[c] = row[c] / temperature_b;
                max_v     = fmaxf(max_v, scaled[c]);
            }
            float weight[kTopK];
            std::int32_t identity[kTopK];
            for (int c = 0; c < kTopK; ++c) {
                weight[c]   = expf(scaled[c] - max_v);
                identity[c] = c;
            }
            // sampling_pick_from_support folds the mass into its goal (goal = u * mass), so
            // the unnormalized weights draw the exact softmax distribution.
            const float u =
                sampling_uniform(seed_b, b, kSamplePurposeDflashSelector,
                                 static_cast<unsigned int>(t));
            idx = sampling_pick_from_support(identity, weight, kTopK, -1, u);
        }

        // flat_drafts is the batch-major view of draft_tokens [k,B]: element (t,b) at b*k + t.
        drafts[col] = ids[idx];
        pred        = idx;
    }
}

} // namespace ninfer::ops