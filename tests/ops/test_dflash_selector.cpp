// Op unit tests for the DFlash 2 path-selector family: dflash_selector_topk,
// dflash_selector_scores, dflash_selector_walk.
//
// Every case is checked against an independent CPU reference built from the exact values the
// kernels read, under the engine layout law (element (r, c) of a contiguous [R, C] tensor at
// c*R + r, first dimension fastest):
//
//   logits       [248320, C]   element (v, col) at col*248320 + v
//   candidates   [16, C]       element (rank, col) at col*16 + rank
//   unary        [16, C]       element (rank, col) at col*16 + rank
//   hidden_proj  [256, C]      element (d, col) at col*256 + d
//   codebooks    [248320, 256] element (v, d) at v + d*248320 (token dimension fastest)
//   scores       [k, 16, 16, B] element (t,p,c,b) at t + p*k + c*k*16 + b*k*16*16
//   flat drafts  [k*B]         element (t,b) at b*k + t (batch-major draft columns m = b*k + t)
//
// The walk's sampled branch is checked against a bit-exact host replica of the counter-based
// draw: sampling_uniform(seed, row, kSamplePurposeDflashSelector, position) + softmax(row/T) +
// inverse-CDF pick (the exact float goal math of sampling_pick_from_support, exclude = -1).
// Each draw additionally asserts a robustness margin so a 1-ulp CPU/GPU expf difference can
// never flip the chosen slot (the fixture is verified deterministic, not merely distributional).
#include "ninfer/ops/dflash_selector.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kVocab = 248320;
constexpr std::int32_t kTopK  = 16;
constexpr std::int32_t kRank  = 256;

// --- deterministic 64-bit LCG (test fixture values only) ----------------------
std::uint64_t lcg_next(std::uint64_t& state) {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return state;
}

// 24-bit LCG bits -> a float in [lo, lo + span): exact f32 scaling (span / 2^24) before a
// single rounding, so the fixture is reproducible bit-for-bit.
float lcg_value(std::uint64_t& state, float lo, float span) {
    return lo + static_cast<float>(lcg_next(state) >> 40) * (span / 16777216.0f);
}

std::vector<std::uint16_t> to_bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

std::vector<double> read_f32_as_double(const void* device, std::size_t n) {
    const std::vector<float> f = from_device<float>(device, n);
    std::vector<double> o(n);
    for (std::size_t i = 0; i < n; ++i) { o[i] = static_cast<double>(f[i]); }
    return o;
}

// scores element (t,p,c,b) at t + p*k + c*k*16 + b*k*16*16 (t fastest, b slowest).
std::size_t scores_offset(std::int32_t t, std::int32_t p, std::int32_t c, std::int32_t b,
                          std::int32_t k) {
    return static_cast<std::size_t>(t) + static_cast<std::size_t>(p) * k +
           static_cast<std::size_t>(c) * k * kTopK + static_cast<std::size_t>(b) * k * kTopK * kTopK;
}

template <class Callable>
int expect_invalid_argument(Callable&& callable, const char* label, const char* expected_fragment) {
    try {
        callable();
    } catch (const std::invalid_argument& error) {
        if (std::string(error.what()).find(expected_fragment) == std::string::npos) {
            std::cerr << label << ": unexpected message: " << error.what() << '\n';
            return 1;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << label << ": expected invalid_argument, got " << error.what() << '\n';
        return 1;
    }
    std::cerr << label << ": expected invalid_argument\n";
    return 1;
}

// --- dflash_selector_topk ------------------------------------------------------
constexpr std::int32_t kTopkColumns = 4;  // C = k*B = 2*2, registered domain

// Column fixtures (all other tokens: LCG fill in [-32, 32), which can never reach a tie or a
// top value):
//   col 0: sixteen distinct top values 100+j (j = 0..15), tokens 12345 + 65537*j (mod vocab);
//          rank 0 is j=15 (value 115) down to rank 15 (value 100).
//   col 1: 512-way tie at 50.0 (tokens 100000..100511): the 16 lowest token ids win.
//   col 2: 17-way tie at 50.0 ({124000} U {124002..124017}, with the 124001 hole): the 16
//          lowest ids win and the 17th (124017) is excluded exactly at the rank-15/16 boundary.
//   col 3: sixteen distinct top values 100.5+j (on the 0.5 BF16 grid, so distinct from col 0's
//          integers), tokens 777777 + 65537*j (mod vocab).
std::vector<std::uint16_t> make_topk_logits() {
    std::vector<float> values(static_cast<std::size_t>(kVocab) * kTopkColumns);
    std::uint64_t state = 0x123456789ABCDEF0ull;
    for (std::size_t i = 0; i < values.size(); ++i) { values[i] = lcg_value(state, -32.0f, 64.0f); }
    const auto set = [&](std::int32_t col, std::int32_t token, float value) {
        values[static_cast<std::size_t>(col) * kVocab + token] = value;
    };
    for (int j = 0; j < kTopK; ++j) {
        set(0, (12345 + j * 65537) % kVocab, 100.0f + static_cast<float>(j));
        set(3, (777777 + j * 65537) % kVocab, 100.5f + static_cast<float>(j));
    }
    for (std::int32_t token = 100000; token < 100512; ++token) { set(1, token, 50.0f); }
    set(2, 124000, 50.0f);
    for (std::int32_t token = 124002; token < 124018; ++token) { set(2, token, 50.0f); }
    round_to_bf16(values);
    return to_bf16_bits(values);
}

struct TopkExpected {
    std::vector<std::int32_t> candidates;  // [16, C], column col at col*16 + rank
    std::vector<double> unary;             // [16, C]
};

// Per-column top-16: value descending, ties broken by the lower token id (the sampling op's
// candidate ordering).
TopkExpected topk_oracle(const std::vector<std::uint16_t>& logits) {
    TopkExpected out;
    out.candidates.resize(static_cast<std::size_t>(kTopK) * kTopkColumns);
    out.unary.resize(static_cast<std::size_t>(kTopK) * kTopkColumns);
    std::vector<std::pair<float, std::int32_t>> entries(kVocab);
    for (std::int32_t col = 0; col < kTopkColumns; ++col) {
        const std::size_t base = static_cast<std::size_t>(col) * kVocab;
        for (std::int32_t token = 0; token < kVocab; ++token) {
            entries[static_cast<std::size_t>(token)] = {bf16_to_f32(logits[base + token]), token};
        }
        std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
            if (a.first != b.first) { return a.first > b.first; }
            return a.second < b.second;
        });
        for (int rank = 0; rank < kTopK; ++rank) {
            out.candidates[static_cast<std::size_t>(col) * kTopK + rank] = entries[rank].second;
            out.unary[static_cast<std::size_t>(col) * kTopK + rank]      = entries[rank].first;
        }
    }
    return out;
}

int run_topk() {
    const auto logits_bits = make_topk_logits();
    const auto expected    = topk_oracle(logits_bits);
    // Fixture sanity: the tie columns resolve to the lowest ids, with the 17th col-2 token
    // excluded at the rank-15/16 boundary.
    if (expected.candidates[kTopK + 0] != 100000 ||
        expected.candidates[kTopK + 15] != 100015) {
        std::cerr << "selector topk fixture: col 1 tie resolution unexpected\n";
        return 1;
    }
    if (expected.candidates[kTopkColumns * kTopK + 2 * kTopK + 0] != 124000 ||
        expected.candidates[kTopkColumns * kTopK + 2 * kTopK + 1] != 124002 ||
        expected.candidates[kTopkColumns * kTopK + 2 * kTopK + 15] != 124016) {
        std::cerr << "selector topk fixture: col 2 boundary tie resolution unexpected\n";
        return 1;
    }

    const std::size_t out_count = static_cast<std::size_t>(kTopK) * kTopkColumns;
    GuardedDeviceBuffer device_logits(logits_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_candidates(out_count * sizeof(std::int32_t));
    GuardedDeviceBuffer device_unary(out_count * sizeof(float));
    device_logits.copy_from_host(logits_bits.data(), device_logits.bytes());
    device_candidates.fill(0xcd);
    device_unary.fill(0xcd);

    Tensor logits_tensor(device_logits.data(), DType::BF16, {kVocab, kTopkColumns});
    Tensor candidates_tensor(device_candidates.data(), DType::I32, {kTopK, kTopkColumns});
    Tensor unary_tensor(device_unary.data(), DType::FP32, {kTopK, kTopkColumns});
    ops::dflash_selector_topk(logits_tensor, candidates_tensor, unary_tensor, nullptr);
    cuda_synchronize();

    int failures = 0;
    failures += verify_exact(
        "selector topk candidates",
        from_device<std::int32_t>(device_candidates.data(), out_count), expected.candidates);
    failures += verify_exact("selector topk unary",
                             read_f32_as_double(device_unary.data(), out_count), expected.unary);
    failures += verify_exact("selector topk logits unchanged",
                             from_device<std::uint16_t>(device_logits.data(), logits_bits.size()),
                             logits_bits);
    failures += device_logits.verify_guards("selector topk logits");
    failures += device_candidates.verify_guards("selector topk candidates");
    failures += device_unary.verify_guards("selector topk unary");
    return failures;
}

// --- dflash_selector_scores ----------------------------------------------------
constexpr std::int32_t kScoresK    = 2;
constexpr std::int32_t kScoresB    = 2;
constexpr std::int32_t kScoresCols = kScoresK * kScoresB;  // 4

// Full [248320, 256] BF16 codebook, token-major (element (v, d) at v + d*248320), values in
// [-0.5, 0.5) from a seeded LCG.
std::vector<std::uint16_t> make_codebook(std::uint64_t seed) {
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(kVocab) * kRank);
    std::uint64_t state = seed;
    for (std::size_t i = 0; i < bits.size(); ++i) { bits[i] = f32_to_bf16(lcg_value(state, -0.5f, 1.0f)); }
    return bits;
}

// Shared [16, 4] candidate table (batch-major draft columns m = b*2 + t): token ids distinct
// within each column. The ids also drive the greedy walk's tie-breaks.
std::vector<std::int32_t> make_walk_candidates() {
    std::vector<std::int32_t> candidates(static_cast<std::size_t>(kTopK) * kScoresCols);
    const std::int32_t bases[kScoresCols] = {1000, 5000, 9000, 13000};
    for (std::int32_t col = 0; col < kScoresCols; ++col) {
        for (int rank = 0; rank < kTopK; ++rank) {
            candidates[static_cast<std::size_t>(col) * kTopK + rank] = bases[col] + rank * 997;
        }
    }
    return candidates;
}

// scores[t,p,c,b] = unary[c, col] + sum_d A(pred(p,t))[d] * h[col][d] * B(cand[c, col])[d],
// with col = b*k + t, pred(0,t)=anchors[b], pred(p,t>0)=candidates[col-1, p]. FP64 reference
// over the exact BF16 codebook/hidden values.
std::vector<double> scores_oracle(const std::vector<std::int32_t>& candidates,
                                  const std::vector<std::uint16_t>& hidden_bits,
                                  const std::vector<std::int32_t>& anchors,
                                  const std::vector<float>& unary,
                                  const std::vector<std::uint16_t>& pred_bits,
                                  const std::vector<std::uint16_t>& succ_bits) {
    const std::int32_t k       = kScoresK;
    const std::int32_t b_count = kScoresB;
    std::vector<double> expected(static_cast<std::size_t>(k) * kTopK * kTopK * b_count, 0.0);
    for (std::int32_t b = 0; b < b_count; ++b) {
        for (std::int32_t t = 0; t < k; ++t) {
            const std::int32_t col = b * k + t;
            for (std::int32_t p = 0; p < kTopK; ++p) {
                const std::int32_t pred =
                    (t == 0) ? anchors[static_cast<std::size_t>(b)]
                             : candidates[static_cast<std::size_t>(col - 1) * kTopK + p];
                for (std::int32_t c = 0; c < kTopK; ++c) {
                    const std::int32_t cand = candidates[static_cast<std::size_t>(col) * kTopK + c];
                    double acc = 0.0;
                    for (std::int32_t d = 0; d < kRank; ++d) {
                        const double a = bf16_to_f32(pred_bits[static_cast<std::size_t>(pred) +
                                                               static_cast<std::size_t>(d) * kVocab]);
                        const double h = bf16_to_f32(hidden_bits[static_cast<std::size_t>(col) * kRank + d]);
                        const double s = bf16_to_f32(succ_bits[static_cast<std::size_t>(cand) +
                                                               static_cast<std::size_t>(d) * kVocab]);
                        acc += a * h * s;
                    }
                    expected[scores_offset(t, p, c, b, k)] =
                        unary[static_cast<std::size_t>(col) * kTopK + c] + acc;
                }
            }
        }
    }
    return expected;
}

int run_scores() {
    const auto pred_bits = make_codebook(0x123456789ABCDEF0ull);
    const auto succ_bits = make_codebook(0xFEDCBA0987654321ull);
    const auto candidates = make_walk_candidates();
    std::vector<float> hidden(static_cast<std::size_t>(kRank) * kScoresCols);
    std::vector<float> unary(static_cast<std::size_t>(kTopK) * kScoresCols);
    fill_uniform(hidden, 501u, -0.5f, 0.5f);
    fill_uniform(unary, 502u, -8.0f, 8.0f);
    round_to_bf16(hidden);
    const auto hidden_bits = to_bf16_bits(hidden);
    const std::vector<std::int32_t> anchors{123457, 200123};

    const auto expected = scores_oracle(candidates, hidden_bits, anchors, unary, pred_bits,
                                        succ_bits);

    const std::size_t codebook_count = static_cast<std::size_t>(kVocab) * kRank;
    const std::size_t scores_count   = static_cast<std::size_t>(kScoresK) * kTopK * kTopK * kScoresB;
    GuardedDeviceBuffer device_candidates(candidates.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_hidden(hidden_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_anchors(anchors.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_unary(unary.size() * sizeof(float));
    GuardedDeviceBuffer device_pred(pred_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_succ(succ_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_scores(scores_count * sizeof(float));
    device_candidates.copy_from_host(candidates.data(), device_candidates.bytes());
    device_hidden.copy_from_host(hidden_bits.data(), device_hidden.bytes());
    device_anchors.copy_from_host(anchors.data(), device_anchors.bytes());
    device_unary.copy_from_host(unary.data(), device_unary.bytes());
    device_pred.copy_from_host(pred_bits.data(), device_pred.bytes());
    device_succ.copy_from_host(succ_bits.data(), device_succ.bytes());
    device_scores.fill(0xcd);

    Tensor candidates_tensor(device_candidates.data(), DType::I32, {kTopK, kScoresCols});
    Tensor hidden_tensor(device_hidden.data(), DType::BF16, {kRank, kScoresCols});
    Tensor anchors_tensor(device_anchors.data(), DType::I32, {kScoresB});
    Tensor unary_tensor(device_unary.data(), DType::FP32, {kTopK, kScoresCols});
    Tensor pred_tensor(device_pred.data(), DType::BF16, {kVocab, kRank});
    Tensor succ_tensor(device_succ.data(), DType::BF16, {kVocab, kRank});
    Tensor scores_tensor(device_scores.data(), DType::FP32,
                         {kScoresK, kTopK, kTopK, kScoresB});
    ops::dflash_selector_scores(candidates_tensor, hidden_tensor, anchors_tensor, unary_tensor,
                                pred_tensor, succ_tensor, scores_tensor, nullptr);
    cuda_synchronize();

    int failures = 0;
    // FP32 accumulation of a 256-term contraction over |term| <= 0.125: the ulp-based
    // tolerance stays ~6x below the worst-case FP32-vs-FP64 accumulation drift.
    constexpr PointwiseCriterion scores_criterion{
        /*absolute*/ 3e-3, /*relative*/ 1e-4};
    failures += verify_pointwise("selector scores",
                                 read_f32_as_double(device_scores.data(), scores_count), expected,
                                 scores_criterion);
    // At t = 0 every p-slot reads the same anchor row: all 16 slots must be bit-identical.
    const std::vector<float> scores_f32 = from_device<float>(device_scores.data(), scores_count);
    for (std::int32_t c = 0; c < kTopK; ++c) {
        for (std::int32_t b = 0; b < kScoresB; ++b) {
            const float reference_slot = scores_f32[scores_offset(0, 0, c, b, kScoresK)];
            for (std::int32_t p = 1; p < kTopK; ++p) {
                if (scores_f32[scores_offset(0, p, c, b, kScoresK)] != reference_slot) {
                    std::cerr << "selector scores: t=0 p-slot p=" << p << " c=" << c
                              << " b=" << b << " differs from the anchor-slot value\n";
                    ++failures;
                }
            }
        }
    }
    failures += verify_exact("selector scores candidates unchanged",
                             from_device<std::int32_t>(device_candidates.data(), candidates.size()),
                             candidates);
    failures += verify_exact("selector scores hidden_proj unchanged",
                             from_device<std::uint16_t>(device_hidden.data(), hidden_bits.size()),
                             hidden_bits);
    failures += verify_exact("selector scores anchors unchanged",
                             from_device<std::int32_t>(device_anchors.data(), anchors.size()),
                             anchors);
    failures += verify_exact("selector scores unary unchanged",
                             from_device<float>(device_unary.data(), unary.size()), unary);
    failures += verify_exact("selector scores pred_codebook unchanged",
                             from_device<std::uint16_t>(device_pred.data(), codebook_count),
                             pred_bits);
    failures += verify_exact("selector scores succ_codebook unchanged",
                             from_device<std::uint16_t>(device_succ.data(), codebook_count),
                             succ_bits);
    failures += device_candidates.verify_guards("selector scores candidates");
    failures += device_hidden.verify_guards("selector scores hidden_proj");
    failures += device_anchors.verify_guards("selector scores anchors");
    failures += device_unary.verify_guards("selector scores unary");
    failures += device_pred.verify_guards("selector scores pred_codebook");
    failures += device_succ.verify_guards("selector scores succ_codebook");
    failures += device_scores.verify_guards("selector scores scores");
    return failures;
}

// --- dflash_selector_walk ------------------------------------------------------
constexpr std::int32_t kWalkK     = 2;
constexpr std::int32_t kWalkB     = 2;
constexpr std::int32_t kWalkCols  = kWalkK * kWalkB;

// Background scores: deterministic values in [-6.0, 6.0) on the exact 0.125 grid.
float walk_background_score(std::int32_t t, std::int32_t p, std::int32_t c, std::int32_t b) {
    const int v = ((((t + 1) * 37 + (p + 1) * 13 + (c + 1) * 7 + (b + 1) * 3) % 97) - 48);
    return static_cast<float>(v) * 0.125f;
}

// Greedy-fixture overrides: exact ties at the slots the walk reads (the background max is
// 6.1875, strictly below every override).
struct ScoreOverride {
    std::int32_t t, p, c, b;
    float value;
};
constexpr std::array<ScoreOverride, 8> kGreedyOverrides = {{
    {0, 0, 3, 0, 10.0f},  // b=0 t=0: tie with slot 7 (ids 3991 < 7979 -> slot 3 wins)
    {0, 0, 7, 0, 10.0f},
    {0, 0, 11, 0, 9.0f},   // second-best distractor tie (never reached)
    {0, 0, 12, 0, 9.0f},
    {1, 3, 9, 0, 11.0f},   // b=0 t=1 reads p=3: tie with slot 2 (ids 13973 > 6994 -> 2 wins)
    {1, 3, 2, 0, 11.0f},
    {0, 0, 5, 1, 12.0f},   // b=1 t=0: unique max
    {1, 5, 13, 1, 13.0f},  // b=1 t=1 reads p=5: unique max
}};

float greedy_score_value(std::int32_t t, std::int32_t p, std::int32_t c, std::int32_t b) {
    for (const auto& ov : kGreedyOverrides) {
        if (ov.t == t && ov.p == p && ov.c == c && ov.b == b) { return ov.value; }
    }
    return walk_background_score(t, p, c, b);
}

// B=2 scores buffer; side b uses the greedy pattern iff greedy_side[b] is set.
std::vector<float> make_walk_scores(bool greedy_b0, bool greedy_b1) {
    std::vector<float> scores(static_cast<std::size_t>(kWalkK) * kTopK * kTopK * kWalkB);
    for (std::int32_t b = 0; b < kWalkB; ++b) {
        const bool greedy = (b == 0) ? greedy_b0 : greedy_b1;
        for (std::int32_t p = 0; p < kTopK; ++p) {
            for (std::int32_t c = 0; c < kTopK; ++c) {
                for (std::int32_t t = 0; t < kWalkK; ++t) {
                    scores[scores_offset(t, p, c, b, kWalkK)] =
                        greedy ? greedy_score_value(t, p, c, b)
                               : walk_background_score(t, p, c, b);
                }
            }
        }
    }
    return scores;
}

// --- walk CPU reference (bit-exact replica of the kernel's draw convention) ----
std::uint64_t splitmix64_ref(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Exact host replica of sampling_uniform(seed, position, purpose, sub) (pure integer math,
// then the 24-bit float, which is exact).
float sampling_uniform_ref(std::uint64_t seed, int position, int purpose, std::uint32_t sub) {
    std::uint64_t key = seed;
    key = splitmix64_ref(key ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(position)) *
                                0xD1B54A32D192ED03ull));
    key = splitmix64_ref(key ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(purpose)) <<
                                21) ^
                         (static_cast<std::uint64_t>(sub) * 0x2545F4914F6CDD1Dull));
    const std::uint32_t bits = static_cast<std::uint32_t>(key >> 40);  // 24 bits
    return static_cast<float>(bits) * (1.0f / 16777216.0f);
}

// Exact float goal math of sampling_pick_from_support (identity slots 0..n-1, exclude = -1).
int pick_from_support_ref(const float* weight, int n, float u) {
    float mass = 0.0f;
    for (int j = 0; j < n; ++j) { mass += weight[j]; }
    const float goal = u * mass;
    float acc        = 0.0f;
    int picked       = -1;
    for (int j = 0; j < n; ++j) {
        acc += weight[j];
        picked = j;
        if (goal < acc) { return picked; }
    }
    return picked;
}

// One walk row (batch row b): pred starts at 0 (the anchor slot) and follows the chosen slot.
// Greedy rows mirror the kernel's sampling_better argmax (value descending, lower candidate
// id on ties); sampled rows mirror softmax(row/T) + the counter-based draw. Returns the row's
// drafts at flat columns b*k + t.
std::vector<int> walk_row_reference(const std::vector<float>& scores,
                                    const std::vector<std::int32_t>& candidates, std::int32_t k,
                                    std::int32_t b, float temperature, std::uint64_t seed,
                                    int* failures, const std::string& label) {
    std::vector<int> drafts(static_cast<std::size_t>(k));
    int pred = 0;
    for (std::int32_t t = 0; t < k; ++t) {
        const std::int32_t col = b * k + t;
        float row[kTopK];
        for (std::int32_t c = 0; c < kTopK; ++c) {
            row[c] = scores[scores_offset(t, pred, c, b, k)];
        }
        int idx;
        if (temperature <= 0.0f) {
            float best_value = -std::numeric_limits<float>::infinity();
            std::int32_t best_id = std::numeric_limits<std::int32_t>::max();
            idx                  = 0;
            for (std::int32_t c = 0; c < kTopK; ++c) {
                const std::int32_t id = candidates[static_cast<std::size_t>(col) * kTopK + c];
                if (row[c] > best_value || (row[c] == best_value && id < best_id)) {
                    best_value = row[c];
                    best_id    = id;
                    idx        = c;
                }
            }
        } else {
            float scaled[kTopK];
            float max_v = -std::numeric_limits<float>::infinity();
            for (std::int32_t c = 0; c < kTopK; ++c) {
                scaled[c] = row[c] / temperature;
                max_v     = std::fmaxf(max_v, scaled[c]);
            }
            float weight[kTopK];
            for (std::int32_t c = 0; c < kTopK; ++c) { weight[c] = ::expf(scaled[c] - max_v); }
            const float u = sampling_uniform_ref(seed, b, ops::kSamplePurposeDflashSelector,
                                                 static_cast<std::uint32_t>(t));
            // Robustness precondition: no softmax boundary (slots 0..k-2) within 1e-4
            // (relative to the mass) of the draw goal, so a 1-ulp CPU/GPU expf difference can
            // never flip the chosen slot. The slot-k-1 comparison cannot change the result.
            double mass_d = 0.0, cum_d = 0.0;
            for (std::int32_t c = 0; c < kTopK; ++c) {
                mass_d += static_cast<double>(weight[c]);
            }
            const double goal_d = static_cast<double>(u) * mass_d;
            for (std::int32_t c = 0; c < kTopK - 1; ++c) {
                cum_d += static_cast<double>(weight[c]);
                if (std::fabs(cum_d - goal_d) <= 1e-4 * mass_d) {
                    std::cerr << label << ": row b=" << b << " t=" << t
                              << " draw goal within 1e-4 of a softmax boundary; fixture is not "
                                 "robust\n";
                    ++*failures;
                }
            }
            idx = pick_from_support_ref(weight, kTopK, u);
        }
        drafts[static_cast<std::size_t>(t)] =
            candidates[static_cast<std::size_t>(col) * kTopK + idx];
        pred = idx;
    }
    return drafts;
}

struct WalkRun {
    std::vector<int> drafts;
    int failures = 0;
};

// Run the walk op over one batch (rows: (temperature, seed) per batch row).
WalkRun run_walk(const std::vector<float>& scores, const std::vector<std::int32_t>& candidates,
                 std::int32_t k, const std::vector<std::pair<float, std::uint64_t>>& rows,
                 const std::string& label) {
    const std::int32_t b_count = static_cast<std::int32_t>(rows.size());
    std::vector<ops::SamplingConfig> configs(static_cast<std::size_t>(b_count));
    for (std::int32_t b = 0; b < b_count; ++b) {
        configs[static_cast<std::size_t>(b)].temperature = rows[static_cast<std::size_t>(b)].first;
        configs[static_cast<std::size_t>(b)].seed        = rows[static_cast<std::size_t>(b)].second;
    }

    GuardedDeviceBuffer device_scores(scores.size() * sizeof(float));
    GuardedDeviceBuffer device_candidates(candidates.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_drafts(static_cast<std::size_t>(k * b_count) * sizeof(std::int32_t));
    device_scores.copy_from_host(scores.data(), device_scores.bytes());
    device_candidates.copy_from_host(candidates.data(), device_candidates.bytes());
    device_drafts.fill(0xcd);

    Tensor scores_tensor(device_scores.data(), DType::FP32, {k, kTopK, kTopK, b_count});
    Tensor candidates_tensor(device_candidates.data(), DType::I32, {kTopK, k * b_count});
    Tensor drafts_tensor(device_drafts.data(), DType::I32, {k * b_count, 1});
    ops::dflash_selector_walk(scores_tensor, candidates_tensor, configs.data(), drafts_tensor,
                              nullptr);
    cuda_synchronize();

    WalkRun run;
    run.drafts   = from_device<std::int32_t>(device_drafts.data(), static_cast<std::size_t>(k) * b_count);
    run.failures = device_scores.verify_guards((label + " scores").c_str());
    run.failures += device_candidates.verify_guards((label + " candidates").c_str());
    run.failures += device_drafts.verify_guards((label + " drafts").c_str());
    return run;
}

// Reference drafts for a whole batch (flat batch-major columns b*k + t).
std::vector<int> expected_walk_drafts(const std::vector<float>& scores,
                                      const std::vector<std::int32_t>& candidates, std::int32_t k,
                                      const std::vector<std::pair<float, std::uint64_t>>& rows,
                                      const std::string& label, int& failures) {
    std::vector<int> expected;
    expected.reserve(static_cast<std::size_t>(k) * rows.size());
    for (std::int32_t b = 0; b < static_cast<std::int32_t>(rows.size()); ++b) {
        const auto row = walk_row_reference(scores, candidates, k, b, rows[static_cast<std::size_t>(b)].first,
                                            rows[static_cast<std::size_t>(b)].second, &failures, label);
        expected.insert(expected.end(), row.begin(), row.end());
    }
    return expected;
}

int run_walks() {
    const auto candidates = make_walk_candidates();
    const auto greedy_scores   = make_walk_scores(true, true);
    const auto sampled_scores  = make_walk_scores(false, false);
    const auto mixed_scores    = make_walk_scores(true, false);
    // The B=1 invariance batch: the same row-0 data in a single-row batch (the first k*256
    // elements of the B=2 buffer are exactly the b=0 scores slice).
    const auto greedy_b1_scores =
        std::vector<float>(greedy_scores.begin(), greedy_scores.begin() + kWalkK * kTopK * kTopK);

    int failures = 0;
    // Greedy-only batch: both rows T = 0 (no draws; the seeds are irrelevant).
    const std::vector<std::pair<float, std::uint64_t>> greedy_rows = {{0.0f, 1u}, {0.0f, 2u}};
    const WalkRun greedy = run_walk(greedy_scores, candidates, kWalkK, greedy_rows,
                                    "selector walk greedy");
    failures += greedy.failures;
    const auto greedy_expected = expected_walk_drafts(greedy_scores, candidates, kWalkK,
                                                      greedy_rows, "selector walk greedy", failures);
    failures += verify_exact("selector walk greedy drafts", greedy.drafts, greedy_expected);

    // Sampled-only batch: fixed seeds, temperature-only sampling.
    const std::vector<std::pair<float, std::uint64_t>> sampled_rows = {
        {0.7f, 20260824u}, {1.3f, 20260825u}};
    const WalkRun sampled = run_walk(sampled_scores, candidates, kWalkK, sampled_rows,
                                     "selector walk sampled");
    failures += sampled.failures;
    const auto sampled_expected = expected_walk_drafts(sampled_scores, candidates, kWalkK,
                                                       sampled_rows, "selector walk sampled",
                                                       failures);
    failures += verify_exact("selector walk sampled drafts", sampled.drafts, sampled_expected);

    // Mixed batch (spec section 9 edge case): row 0 greedy, row 1 sampled. The greedy row must
    // be identical to its greedy-batch output (no RNG draws) and the sampled row identical to
    // its sampled-batch output (same (seed, row, purpose, position) subkeys).
    const std::vector<std::pair<float, std::uint64_t>> mixed_rows = {{0.0f, 1u},
                                                                     {1.3f, 20260825u}};
    const WalkRun mixed  = run_walk(mixed_scores, candidates, kWalkK, mixed_rows,
                                    "selector walk mixed");
    const WalkRun mixed2 = run_walk(mixed_scores, candidates, kWalkK, mixed_rows,
                                    "selector walk mixed rerun");
    failures += mixed.failures + mixed2.failures;
    failures += verify_exact("selector walk mixed determinism (rerun identical)",
                             mixed2.drafts, mixed.drafts);
    const auto mixed_expected = expected_walk_drafts(mixed_scores, candidates, kWalkK, mixed_rows,
                                                     "selector walk mixed", failures);
    failures += verify_exact("selector walk mixed drafts", mixed.drafts, mixed_expected);
    failures += verify_exact("selector walk greedy batch invariance (B=2 mixed row 0)",
                             std::vector<int>(mixed.drafts.begin(),
                                              mixed.drafts.begin() + kWalkK),
                             std::vector<int>(greedy.drafts.begin(),
                                              greedy.drafts.begin() + kWalkK));
    failures += verify_exact("selector walk sampled batch invariance (B=2 mixed row 1)",
                             std::vector<int>(mixed.drafts.begin() + kWalkK, mixed.drafts.end()),
                             std::vector<int>(sampled.drafts.begin() + kWalkK,
                                              sampled.drafts.end()));

    // Greedy batch invariance at B=1: the same row 0 in a single-row batch.
    const std::vector<std::pair<float, std::uint64_t>> b1_rows = {{0.0f, 1u}};
    const WalkRun b1 = run_walk(greedy_b1_scores, std::vector<std::int32_t>(candidates.begin(),
                                                                            candidates.begin() + 2 * kTopK),
                                kWalkK, b1_rows, "selector walk greedy B=1");
    failures += b1.failures;
    const auto b1_expected =
        expected_walk_drafts(greedy_b1_scores,
                             std::vector<std::int32_t>(candidates.begin(),
                                                       candidates.begin() + 2 * kTopK),
                             kWalkK, b1_rows, "selector walk greedy B=1", failures);
    failures += verify_exact("selector walk greedy B=1 drafts", b1.drafts, b1_expected);
    failures += verify_exact("selector walk greedy B=1 invariance (row 0 == B=2 row 0)", b1.drafts,
                             std::vector<int>(greedy.drafts.begin(),
                                              greedy.drafts.begin() + kWalkK));
    return failures;
}

// --- wrapper validation contract ----------------------------------------------
int rejection_cases() {
    int failures = 0;
    // topk: dummy storage beyond the throw point (the wrapper inspects metadata only before
    // it rejects; the linear_add tests use the same small-storage idiom).
    GuardedDeviceBuffer topk_logits(static_cast<std::size_t>(kVocab) * kTopkColumns * 2);
    GuardedDeviceBuffer topk_logits11(static_cast<std::size_t>(kVocab) * 11 * 2);
    GuardedDeviceBuffer topk_candidates(kTopK * kTopkColumns * sizeof(std::int32_t));
    GuardedDeviceBuffer topk_unary(kTopK * kTopkColumns * sizeof(float));
    topk_logits.fill(0x11);
    topk_logits11.fill(0x22);
    topk_candidates.fill(0x33);
    topk_unary.fill(0x44);
    Tensor topk_logits_tensor(topk_logits.data(), DType::BF16, {kVocab, kTopkColumns});
    Tensor topk_candidates_tensor(topk_candidates.data(), DType::I32, {kTopK, kTopkColumns});
    Tensor topk_unary_tensor(topk_unary.data(), DType::FP32, {kTopK, kTopkColumns});
    failures += expect_invalid_argument(
        [&] {
            Tensor logits11(topk_logits11.data(), DType::BF16, {kVocab, 11});
            ops::dflash_selector_topk(logits11, topk_candidates_tensor, topk_unary_tensor,
                                      nullptr);
        },
        "selector topk rejection C=11",
        "dflash_selector_topk: logits must be BF16 [248320,C] with C = k*B, k in [1,7], B in [1,8]");
    failures += expect_invalid_argument(
        [&] {
            Tensor logits_small(topk_logits.data(), DType::BF16, {1000, kTopkColumns});
            ops::dflash_selector_topk(logits_small, topk_candidates_tensor, topk_unary_tensor,
                                      nullptr);
        },
        "selector topk rejection vocab rows",
        "dflash_selector_topk: logits must be BF16 [248320,C] with C = k*B, k in [1,7], B in [1,8]");
    failures += expect_invalid_argument(
        [&] {
            Tensor candidates_f32(topk_candidates.data(), DType::FP32, {kTopK, kTopkColumns});
            ops::dflash_selector_topk(topk_logits_tensor, candidates_f32, topk_unary_tensor,
                                      nullptr);
        },
        "selector topk rejection candidates dtype",
        "dflash_selector_topk: candidates must be contiguous with the exact registered shape");
    failures += expect_invalid_argument(
        [&] {
            Tensor candidates_alias(topk_logits.data(), DType::I32, {kTopK, kTopkColumns});
            ops::dflash_selector_topk(topk_logits_tensor, candidates_alias, topk_unary_tensor,
                                      nullptr);
        },
        "selector topk rejection logits/candidates overlap",
        "dflash_selector_topk: inputs and outputs must not overlap");

    // scores.
    GuardedDeviceBuffer sc_candidates(kTopK * kScoresCols * sizeof(std::int32_t));
    GuardedDeviceBuffer sc_hidden(kRank * kScoresCols * sizeof(std::uint16_t));
    GuardedDeviceBuffer sc_anchors(9 * sizeof(std::int32_t));
    GuardedDeviceBuffer sc_unary(kTopK * kScoresCols * sizeof(float));
    GuardedDeviceBuffer sc_codebook(4096);
    GuardedDeviceBuffer sc_scores(kScoresK * kTopK * kTopK * kScoresB * sizeof(float));
    sc_candidates.fill(0x55);
    sc_hidden.fill(0x66);
    sc_anchors.fill(0x77);
    sc_unary.fill(0x88);
    sc_codebook.fill(0x99);
    sc_scores.fill(0xaa);
    Tensor sc_candidates_tensor(sc_candidates.data(), DType::I32, {kTopK, kScoresCols});
    Tensor sc_hidden_tensor(sc_hidden.data(), DType::BF16, {kRank, kScoresCols});
    Tensor sc_anchors_tensor(sc_anchors.data(), DType::I32, {kScoresB});
    Tensor sc_unary_tensor(sc_unary.data(), DType::FP32, {kTopK, kScoresCols});
    Tensor sc_pred_tensor(sc_codebook.data(), DType::BF16, {kVocab, kRank});
    Tensor sc_succ_tensor(sc_codebook.data(), DType::BF16, {kVocab, kRank});
    Tensor sc_scores_tensor(sc_scores.data(), DType::FP32, {kScoresK, kTopK, kTopK, kScoresB});
    failures += expect_invalid_argument(
        [&] {
            Tensor anchors9(sc_anchors.data(), DType::I32, {9});
            Tensor scores9(sc_scores.data(), DType::FP32, {kScoresK, kTopK, kTopK, 9});
            ops::dflash_selector_scores(sc_candidates_tensor, sc_hidden_tensor, anchors9,
                                        sc_unary_tensor, sc_pred_tensor, sc_succ_tensor, scores9,
                                        nullptr);
        },
        "selector scores rejection B=9",
        "dflash_selector_scores: anchors must be I32 [B] with B in [1,8]");
    failures += expect_invalid_argument(
        [&] {
            Tensor candidates6(sc_candidates.data(), DType::I32, {kTopK, 6});
            ops::dflash_selector_scores(candidates6, sc_hidden_tensor, sc_anchors_tensor,
                                        sc_unary_tensor, sc_pred_tensor, sc_succ_tensor,
                                        sc_scores_tensor, nullptr);
        },
        "selector scores rejection k*B mismatch",
        "dflash_selector_scores: k*B draft columns must match the registered domain");
    failures += expect_invalid_argument(
        [&] {
            Tensor pred_small(sc_codebook.data(), DType::BF16, {100, kRank});
            ops::dflash_selector_scores(sc_candidates_tensor, sc_hidden_tensor, sc_anchors_tensor,
                                        sc_unary_tensor, pred_small, sc_succ_tensor,
                                        sc_scores_tensor, nullptr);
        },
        "selector scores rejection pred_codebook shape",
        "dflash_selector_scores: pred_codebook must be contiguous with the exact registered shape");
    failures += expect_invalid_argument(
        [&] {
            Tensor scores_bf16(sc_scores.data(), DType::BF16, {kScoresK, kTopK, kTopK, kScoresB});
            ops::dflash_selector_scores(sc_candidates_tensor, sc_hidden_tensor, sc_anchors_tensor,
                                        sc_unary_tensor, sc_pred_tensor, sc_succ_tensor,
                                        scores_bf16, nullptr);
        },
        "selector scores rejection scores dtype",
        "dflash_selector_scores: scores must be contiguous FP32 [k,16,16,B]");

    // walk.
    GuardedDeviceBuffer wk_scores(kWalkK * kTopK * kTopK * kWalkB * sizeof(float));
    GuardedDeviceBuffer wk_candidates(kTopK * kWalkCols * sizeof(std::int32_t));
    GuardedDeviceBuffer wk_drafts(kWalkCols * sizeof(std::int32_t));
    wk_scores.fill(0xbb);
    wk_candidates.fill(0xcc);
    wk_drafts.fill(0xdd);
    Tensor wk_scores_tensor(wk_scores.data(), DType::FP32, {kWalkK, kTopK, kTopK, kWalkB});
    Tensor wk_candidates_tensor(wk_candidates.data(), DType::I32, {kTopK, kWalkCols});
    Tensor wk_drafts_tensor(wk_drafts.data(), DType::I32, {kWalkCols, 1});
    ops::SamplingConfig config;
    failures += expect_invalid_argument(
        [&] {
            ops::dflash_selector_walk(wk_scores_tensor, wk_candidates_tensor, nullptr,
                                      wk_drafts_tensor, nullptr);
        },
        "selector walk rejection null configs",
        "dflash_selector_walk: configs must be a non-null host pointer");
    failures += expect_invalid_argument(
        [&] {
            Tensor scores_k8(wk_scores.data(), DType::FP32, {8, kTopK, kTopK, 1});
            ops::dflash_selector_walk(scores_k8, wk_candidates_tensor, &config,
                                      wk_drafts_tensor, nullptr);
        },
        "selector walk rejection k=8",
        "dflash_selector_walk: scores must be FP32 [k,16,16,B] with k in [1,7] and B in [1,8]");
    failures += expect_invalid_argument(
        [&] {
            Tensor candidates3(wk_candidates.data(), DType::I32, {kTopK, 3});
            ops::dflash_selector_walk(wk_scores_tensor, candidates3, &config, wk_drafts_tensor,
                                      nullptr);
        },
        "selector walk rejection candidates shape",
        "dflash_selector_walk: candidates must be contiguous with the exact registered shape");
    failures += expect_invalid_argument(
        [&] {
            Tensor drafts_alias(wk_scores.data(), DType::I32, {kWalkCols, 1});
            ops::dflash_selector_walk(wk_scores_tensor, wk_candidates_tensor, &config,
                                      drafts_alias, nullptr);
        },
        "selector walk rejection scores/drafts overlap",
        "dflash_selector_walk: inputs and outputs must not overlap");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_topk();
    failures += run_scores();
    failures += run_walks();
    failures += rejection_cases();

    std::cout << (failures == 0 ? "OK" : "FAIL") << " dflash_selector\n";
    return failures == 0 ? 0 : 1;
}