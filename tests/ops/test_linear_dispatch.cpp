// CPU-only shape-coverage contract for the w8 and bf16 linear dispatch select functions
// (spec doc 06, sections 5 and 10). The DFlash 2 27B wave added four registered linear
// classes:
//   W8:   {n=5120,k=25600} dflash feature projection (fc.weight)
//         {n=5120,k=4096}  dflash attention output  (o_proj)
//   BF16: {n=1280,k=5120}  conv kernel_projection
//         {n=256, k=5120}  selector hidden_projection
// This test pins:
//   1. the two new W8 classes do not throw anywhere on T = 1..262144 (dense 1..4096, then a
//      deterministic stride-128 tail up to the 262144 native context, plus every landed
//      ladder boundary at T-1/T/T+1) and never leave the shape-generic launcher set;
//   2. on the DFlash 2 batch domain T = (k_draft+1)*B, k_draft = 1..7, B = 1..8 (the 29
//      distinct column counts), both new W8 classes return shape-generic launchers;
//   3. the pre-existing registered (n,k) pairs keep exactly the HEAD routing (the expected
//      routes below were extracted from the pre-wave w8_dispatch.cpp), so the new classes
//      did not perturb the existing classes;
//   4. the four bf16 classes route t==1 to the decode launcher, 2..small_t_end to the
//      small-t launcher (small_t_end = 32 for n==5120, kBf16SmallTMaxTokens; 27 otherwise,
//      kBf16LinearSmallTDispatchEnd), and above that to the mma launcher, sampled over
//      T = 1..100000 including every boundary; (256,5120) at T=1 is the selector k=1,B=1
//      case and must be the decode launcher.
//
// The select functions are pure host routing: no CUDA calls and no device allocation are
// made here, so the test runs unconditionally (no artifact, no skip). It links ninfer_ops
// only for the select definitions.

#include "ops/linear/bf16/bf16_dispatch.h"
#include "ops/linear/w8/w8_dispatch.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using ninfer::ops::detail::Bf16Launch;
using ninfer::ops::detail::W8Launch;

// Shape-generic W8 launchers declared by ops/linear/w8/w8_launch.h: the SIMT tier plus the
// generic MMA tier. The shape-specific entries (launch_w8_decode_r4, launch_w8_exact_t_*,
// launch_w8_dflash_medium, launch_w8_medium_splitk_c144, launch_w8_exact_mma_*) are reserved
// for the 35B {2048,16384} dflash feature class; the new 27B classes must use shape-generic
// launchers only (w8_dispatch.cpp: "Shape-generic launchers only").
const std::vector<W8Launch>& shape_generic_w8_launchers() {
    static const std::vector<W8Launch> launchers{
        &ninfer::ops::detail::launch_w8_simt_r8_c4,
        &ninfer::ops::detail::launch_w8_simt_r8_c8,
        &ninfer::ops::detail::launch_w8_mma_r32_c64,
        &ninfer::ops::detail::launch_w8_mma_r32_c96,
        &ninfer::ops::detail::launch_w8_mma_r32_c128,
        &ninfer::ops::detail::launch_w8_mma_r48_c64,
        &ninfer::ops::detail::launch_w8_mma_r48_c96,
        &ninfer::ops::detail::launch_w8_mma_r48_c112,
        &ninfer::ops::detail::launch_w8_mma_r48_c128,
        &ninfer::ops::detail::launch_w8_mma_r64_c96,
        &ninfer::ops::detail::launch_w8_mma_r64_c112,
        &ninfer::ops::detail::launch_w8_mma_r64_c128,
        &ninfer::ops::detail::launch_w8_mma_r96_c96,
        &ninfer::ops::detail::launch_w8_mma_r128_c64,
        &ninfer::ops::detail::launch_w8_mma_r128_c80,
    };
    return launchers;
}

bool in_shape_generic_set(W8Launch launch) {
    return std::any_of(shape_generic_w8_launchers().begin(), shape_generic_w8_launchers().end(),
                       [launch](W8Launch entry) { return entry == launch; });
}

// Deterministic T sweep over [1,262144]: every T in 1..4096 (decode + small-prefill domain,
// which contains every landed ladder boundary and its neighborhood), then the stride-128
// tail 4224..262144 (the prefill-chunk alignment domain up to the 27B native context).
std::vector<std::int32_t> sweep_t_values() {
    std::vector<std::int32_t> values;
    values.reserve(4096 + 1921);
    for (std::int32_t t = 1; t <= 4096; ++t) { values.push_back(t); }
    for (std::int32_t t = 4224; t <= 262144; t += 128) { values.push_back(t); }
    return values;
}

int w8_no_throw_generic(std::int32_t n, std::int32_t k) {
    for (const std::int32_t t : sweep_t_values()) {
        W8Launch launch = nullptr;
        try {
            launch = ninfer::ops::detail::select_w8_a16_launch(n, k, t);
        } catch (const std::exception& error) {
            std::cerr << "FAIL: w8 " << n << "x" << k << " threw at T=" << t << ": "
                      << error.what() << '\n';
            return 1;
        }
        if (!in_shape_generic_set(launch)) {
            std::cerr << "FAIL: w8 " << n << "x" << k << " left the shape-generic launcher set "
                         "at T=" << t << '\n';
            return 1;
        }
    }
    return 0;
}

// Exact routing at every landed boundary (and its T-1/T+1 neighbors) for the two new W8
// classes, derived from the landed ladders in w8_dispatch.cpp (case 25600; the
// {n=5120,k=4096} block in case 4096).
struct W8Route {
    std::int32_t n;
    std::int32_t k;
    std::int32_t t;
    W8Launch launch;
};

const std::vector<W8Route> kNewW8Routes{
    // {5120,25600} feature projection: simt 1..16, then the generic mma tier ladder.
    {5120, 25600, 1, &ninfer::ops::detail::launch_w8_simt_r8_c4},
    {5120, 25600, 3, &ninfer::ops::detail::launch_w8_simt_r8_c4},
    {5120, 25600, 4, &ninfer::ops::detail::launch_w8_simt_r8_c4},
    {5120, 25600, 5, &ninfer::ops::detail::launch_w8_simt_r8_c8},
    {5120, 25600, 15, &ninfer::ops::detail::launch_w8_simt_r8_c8},
    {5120, 25600, 16, &ninfer::ops::detail::launch_w8_simt_r8_c8},
    {5120, 25600, 17, &ninfer::ops::detail::launch_w8_mma_r32_c128},
    {5120, 25600, 254, &ninfer::ops::detail::launch_w8_mma_r32_c128},
    {5120, 25600, 255, &ninfer::ops::detail::launch_w8_mma_r32_c128},
    {5120, 25600, 256, &ninfer::ops::detail::launch_w8_mma_r32_c64},
    {5120, 25600, 383, &ninfer::ops::detail::launch_w8_mma_r32_c64},
    {5120, 25600, 384, &ninfer::ops::detail::launch_w8_mma_r32_c64},
    {5120, 25600, 385, &ninfer::ops::detail::launch_w8_mma_r32_c96},
    {5120, 25600, 479, &ninfer::ops::detail::launch_w8_mma_r32_c96},
    {5120, 25600, 480, &ninfer::ops::detail::launch_w8_mma_r32_c96},
    {5120, 25600, 481, &ninfer::ops::detail::launch_w8_mma_r32_c128},
    {5120, 25600, 639, &ninfer::ops::detail::launch_w8_mma_r32_c128},
    {5120, 25600, 640, &ninfer::ops::detail::launch_w8_mma_r32_c128},
    {5120, 25600, 641, &ninfer::ops::detail::launch_w8_mma_r48_c96},
    {5120, 25600, 671, &ninfer::ops::detail::launch_w8_mma_r48_c96},
    {5120, 25600, 672, &ninfer::ops::detail::launch_w8_mma_r48_c96},
    {5120, 25600, 673, &ninfer::ops::detail::launch_w8_mma_r48_c64},
    {5120, 25600, 703, &ninfer::ops::detail::launch_w8_mma_r48_c64},
    {5120, 25600, 704, &ninfer::ops::detail::launch_w8_mma_r48_c64},
    {5120, 25600, 705, &ninfer::ops::detail::launch_w8_mma_r48_c112},
    {5120, 25600, 783, &ninfer::ops::detail::launch_w8_mma_r48_c112},
    {5120, 25600, 784, &ninfer::ops::detail::launch_w8_mma_r48_c112},
    {5120, 25600, 785, &ninfer::ops::detail::launch_w8_mma_r48_c128},
    {5120, 25600, 895, &ninfer::ops::detail::launch_w8_mma_r48_c128},
    {5120, 25600, 896, &ninfer::ops::detail::launch_w8_mma_r48_c128},
    {5120, 25600, 897, &ninfer::ops::detail::launch_w8_mma_r64_c96},
    {5120, 25600, 959, &ninfer::ops::detail::launch_w8_mma_r64_c96},
    {5120, 25600, 960, &ninfer::ops::detail::launch_w8_mma_r64_c96},
    {5120, 25600, 961, &ninfer::ops::detail::launch_w8_mma_r64_c112},
    {5120, 25600, 1007, &ninfer::ops::detail::launch_w8_mma_r64_c112},
    {5120, 25600, 1008, &ninfer::ops::detail::launch_w8_mma_r64_c112},
    {5120, 25600, 1009, &ninfer::ops::detail::launch_w8_mma_r64_c128},
    {5120, 25600, 1118, &ninfer::ops::detail::launch_w8_mma_r64_c128},
    {5120, 25600, 1119, &ninfer::ops::detail::launch_w8_mma_r64_c128},
    {5120, 25600, 1120, &ninfer::ops::detail::launch_w8_mma_r64_c112},
    {5120, 25600, 1121, &ninfer::ops::detail::launch_w8_mma_r64_c128},
    {5120, 25600, 1279, &ninfer::ops::detail::launch_w8_mma_r64_c128},
    {5120, 25600, 1280, &ninfer::ops::detail::launch_w8_mma_r64_c128},
    {5120, 25600, 1281, &ninfer::ops::detail::launch_w8_mma_r128_c64},
    {5120, 25600, 1343, &ninfer::ops::detail::launch_w8_mma_r128_c64},
    {5120, 25600, 1344, &ninfer::ops::detail::launch_w8_mma_r128_c64},
    {5120, 25600, 1345, &ninfer::ops::detail::launch_w8_mma_r96_c96},
    {5120, 25600, 1439, &ninfer::ops::detail::launch_w8_mma_r96_c96},
    {5120, 25600, 1440, &ninfer::ops::detail::launch_w8_mma_r96_c96},
    {5120, 25600, 1441, &ninfer::ops::detail::launch_w8_mma_r128_c80},
    {5120, 25600, 1679, &ninfer::ops::detail::launch_w8_mma_r128_c80},
    {5120, 25600, 1680, &ninfer::ops::detail::launch_w8_mma_r128_c80},
    {5120, 25600, 1681, &ninfer::ops::detail::launch_w8_mma_r48_c128},
    {5120, 25600, 1790, &ninfer::ops::detail::launch_w8_mma_r48_c128},
    {5120, 25600, 1791, &ninfer::ops::detail::launch_w8_mma_r48_c128},
    {5120, 25600, 1792, &ninfer::ops::detail::launch_w8_mma_r64_c128},
    {5120, 25600, 1793, &ninfer::ops::detail::launch_w8_mma_r48_c128},
    {5120, 25600, 1918, &ninfer::ops::detail::launch_w8_mma_r48_c128},
    {5120, 25600, 1919, &ninfer::ops::detail::launch_w8_mma_r48_c128},
    {5120, 25600, 1920, &ninfer::ops::detail::launch_w8_mma_r64_c128},
    {5120, 25600, 1921, &ninfer::ops::detail::launch_w8_mma_r64_c96},
    {5120, 25600, 2015, &ninfer::ops::detail::launch_w8_mma_r64_c96},
    {5120, 25600, 2016, &ninfer::ops::detail::launch_w8_mma_r64_c96},
    {5120, 25600, 2017, &ninfer::ops::detail::launch_w8_mma_r96_c96},
    {5120, 25600, 2111, &ninfer::ops::detail::launch_w8_mma_r96_c96},
    {5120, 25600, 2112, &ninfer::ops::detail::launch_w8_mma_r96_c96},
    {5120, 25600, 2113, &ninfer::ops::detail::launch_w8_mma_r64_c128},
    {5120, 25600, 262144, &ninfer::ops::detail::launch_w8_mma_r64_c128},
    // {5120,4096} attention output: simt 1..16, generic mma 17..48, unbounded mma fallback.
    {5120, 4096, 1, &ninfer::ops::detail::launch_w8_simt_r8_c4},
    {5120, 4096, 4, &ninfer::ops::detail::launch_w8_simt_r8_c4},
    {5120, 4096, 5, &ninfer::ops::detail::launch_w8_simt_r8_c8},
    {5120, 4096, 16, &ninfer::ops::detail::launch_w8_simt_r8_c8},
    {5120, 4096, 17, &ninfer::ops::detail::launch_w8_mma_r32_c128},
    {5120, 4096, 48, &ninfer::ops::detail::launch_w8_mma_r32_c128},
    {5120, 4096, 49, &ninfer::ops::detail::launch_w8_mma_r64_c128},
    {5120, 4096, 262144, &ninfer::ops::detail::launch_w8_mma_r64_c128},
};

int w8_new_class_boundaries() {
    for (const W8Route& route : kNewW8Routes) {
        W8Launch launch = nullptr;
        try {
            launch = ninfer::ops::detail::select_w8_a16_launch(route.n, route.k, route.t);
        } catch (const std::exception& error) {
            std::cerr << "FAIL: w8 " << route.n << "x" << route.k << " threw at boundary T="
                      << route.t << ": " << error.what() << '\n';
            return 1;
        }
        if (launch != route.launch) {
            std::cerr << "FAIL: w8 " << route.n << "x" << route.k << " routed to an unexpected "
                         "launcher at boundary T=" << route.t << '\n';
            return 1;
        }
    }
    return 0;
}

// DFlash 2 conv/selector batch domain: T = (k_draft+1)*B over k_draft = 1..7 (the 27B
// kMaximumDFlashDraftTokens) and B = 1..8 (kMaximumDFlashConcurrency) — 29 distinct values.
int w8_batch_domain() {
    std::vector<std::int32_t> columns;
    for (std::int32_t k_draft = 1; k_draft <= 7; ++k_draft) {
        for (std::int32_t batch = 1; batch <= 8; ++batch) {
            columns.push_back((k_draft + 1) * batch);
        }
    }
    std::sort(columns.begin(), columns.end());
    columns.erase(std::unique(columns.begin(), columns.end()), columns.end());
    if (columns.size() != 29) {
        std::cerr << "FAIL: expected 29 distinct batch columns (k+1)*B, got " << columns.size()
                  << '\n';
        return 1;
    }
    for (const std::int32_t t : columns) {
        W8Launch feature = nullptr;
        W8Launch attention = nullptr;
        try {
            feature   = ninfer::ops::detail::select_w8_a16_launch(5120, 25600, t);
            attention = ninfer::ops::detail::select_w8_a16_launch(5120, 4096, t);
        } catch (const std::exception& error) {
            std::cerr << "FAIL: w8 27B batch-domain select threw at T=" << t << ": "
                      << error.what() << '\n';
            return 1;
        }
        if (!in_shape_generic_set(feature) || !in_shape_generic_set(attention)) {
            std::cerr << "FAIL: w8 27B batch-domain T=" << t
                      << " left the shape-generic launcher set\n";
            return 1;
        }
    }
    return 0;
}

// HEAD regression: the pre-existing registered (n,k) pairs must keep exactly the pre-wave
// routing (extracted from the HEAD w8_dispatch.cpp before the 27B classes were added).
// The sampled T set covers every ladder boundary (and neighbors) plus far-field points,
// including the k=4096 case the wave extended and the k=4608 T upper bound.
const std::vector<W8Route> kHeadW8Routes{
    // (n=6144,k=5120) qkv: simt tiers then mma fallback.
    {6144, 5120, 1, &ninfer::ops::detail::launch_w8_simt_r8_c4},
    {6144, 5120, 4, &ninfer::ops::detail::launch_w8_simt_r8_c4},
    {6144, 5120, 5, &ninfer::ops::detail::launch_w8_simt_r8_c8},
    {6144, 5120, 16, &ninfer::ops::detail::launch_w8_simt_r8_c8},
    {6144, 5120, 17, &ninfer::ops::detail::launch_w8_mma_r64_c128},
    {6144, 5120, 100, &ninfer::ops::detail::launch_w8_mma_r64_c128},
    {6144, 5120, 1000, &ninfer::ops::detail::launch_w8_mma_r64_c128},
    // (n=14336,k=5120): small-t tier to 48, then mma.
    {14336, 5120, 1, &ninfer::ops::detail::launch_w8_small_t},
    {14336, 5120, 47, &ninfer::ops::detail::launch_w8_small_t},
    {14336, 5120, 48, &ninfer::ops::detail::launch_w8_small_t},
    {14336, 5120, 49, &ninfer::ops::detail::launch_w8_mma_r64_c128},
    {14336, 5120, 5000, &ninfer::ops::detail::launch_w8_mma_r64_c128},
    // (n=34816,k=5120) gate_up: small-t to 40, r64x16 to 48, then mma.
    {34816, 5120, 1, &ninfer::ops::detail::launch_w8_small_t},
    {34816, 5120, 40, &ninfer::ops::detail::launch_w8_small_t},
    {34816, 5120, 41, &ninfer::ops::detail::launch_w8_mma_r64x16_c48_k128_a1},
    {34816, 5120, 48, &ninfer::ops::detail::launch_w8_mma_r64x16_c48_k128_a1},
    {34816, 5120, 49, &ninfer::ops::detail::launch_w8_mma_r64_c128},
    {34816, 5120, 1000, &ninfer::ops::detail::launch_w8_mma_r64_c128},
    // (n=248320,k=5120) full output head: small-t to 33, r64x16 to 48, r32_c64 to 64, mma.
    {248320, 5120, 1, &ninfer::ops::detail::launch_w8_small_t},
    {248320, 5120, 32, &ninfer::ops::detail::launch_w8_small_t},
    {248320, 5120, 33, &ninfer::ops::detail::launch_w8_small_t},
    {248320, 5120, 34, &ninfer::ops::detail::launch_w8_mma_r64x16_c48_k128_a1},
    {248320, 5120, 48, &ninfer::ops::detail::launch_w8_mma_r64x16_c48_k128_a1},
    {248320, 5120, 49, &ninfer::ops::detail::launch_w8_mma_r32_c64},
    {248320, 5120, 64, &ninfer::ops::detail::launch_w8_mma_r32_c64},
    {248320, 5120, 65, &ninfer::ops::detail::launch_w8_mma_r64_c128},
    {248320, 5120, 2000, &ninfer::ops::detail::launch_w8_mma_r64_c128},
    // (n=5120,k=17408) down projection: small-t to 48, then mma.
    {5120, 17408, 1, &ninfer::ops::detail::launch_w8_small_t},
    {5120, 17408, 47, &ninfer::ops::detail::launch_w8_small_t},
    {5120, 17408, 48, &ninfer::ops::detail::launch_w8_small_t},
    {5120, 17408, 49, &ninfer::ops::detail::launch_w8_mma_r64_c128},
    {5120, 17408, 1000, &ninfer::ops::detail::launch_w8_mma_r64_c128},
    // (n=2048,k=4096): the case the wave extended for {5120,4096}; the existing n=2048
    // sub-case (small-t to 48, simt to 56, mma_r32 to 895, mma fallback) must be untouched.
    {2048, 4096, 1, &ninfer::ops::detail::launch_w8_small_t},
    {2048, 4096, 47, &ninfer::ops::detail::launch_w8_small_t},
    {2048, 4096, 48, &ninfer::ops::detail::launch_w8_small_t},
    {2048, 4096, 49, &ninfer::ops::detail::launch_w8_simt_r8_c4},
    {2048, 4096, 55, &ninfer::ops::detail::launch_w8_simt_r8_c4},
    {2048, 4096, 56, &ninfer::ops::detail::launch_w8_simt_r8_c4},
    {2048, 4096, 57, &ninfer::ops::detail::launch_w8_mma_r32_c128},
    {2048, 4096, 894, &ninfer::ops::detail::launch_w8_mma_r32_c128},
    {2048, 4096, 895, &ninfer::ops::detail::launch_w8_mma_r32_c128},
    {2048, 4096, 896, &ninfer::ops::detail::launch_w8_mma_r64_c128},
    {2048, 4096, 5000, &ninfer::ops::detail::launch_w8_mma_r64_c128},
    // (n=5120,k=4608): sparse exact-t entries with the 32768 T upper bound.
    {5120, 4608, 1, &ninfer::ops::detail::launch_w8_simt_r8_c4},
    {5120, 4608, 4, &ninfer::ops::detail::launch_w8_simt_r8_c4},
    {5120, 4608, 5, &ninfer::ops::detail::launch_w8_simt_r8_c8},
    {5120, 4608, 6, &ninfer::ops::detail::launch_w8_mma_r64_c128},
    {5120, 4608, 32768, &ninfer::ops::detail::launch_w8_mma_r64_c128},
    // (n=2048,k=4608): the exact-t list ladder (t<=14 or t in {16,20,24,28,32}).
    {2048, 4608, 1, &ninfer::ops::detail::launch_w8_simt_r8_c4},
    {2048, 4608, 14, &ninfer::ops::detail::launch_w8_simt_r8_c4},
    {2048, 4608, 15, &ninfer::ops::detail::launch_w8_mma_r32_c128},
    {2048, 4608, 16, &ninfer::ops::detail::launch_w8_simt_r8_c4},
    {2048, 4608, 17, &ninfer::ops::detail::launch_w8_mma_r32_c128},
    {2048, 4608, 20, &ninfer::ops::detail::launch_w8_simt_r8_c4},
    {2048, 4608, 21, &ninfer::ops::detail::launch_w8_mma_r32_c128},
    {2048, 4608, 871, &ninfer::ops::detail::launch_w8_mma_r32_c128},
    {2048, 4608, 872, &ninfer::ops::detail::launch_w8_mma_r64_c128},
    {2048, 4608, 32768, &ninfer::ops::detail::launch_w8_mma_r64_c128},
    // (n=2048,k=16384) 35B dflash feature class: the shape-specific ladder, pinned in full
    // so the new 27B classes cannot perturb it.
    {2048, 16384, 1, &ninfer::ops::detail::launch_w8_decode_r4},
    {2048, 16384, 48, &ninfer::ops::detail::launch_w8_exact_t_splitk},
    {2048, 16384, 49, &ninfer::ops::detail::launch_w8_dflash_medium},
    {2048, 16384, 128, &ninfer::ops::detail::launch_w8_dflash_medium},
    {2048, 16384, 129, &ninfer::ops::detail::launch_w8_medium_splitk_c144},
    {2048, 16384, 144, &ninfer::ops::detail::launch_w8_medium_splitk_c144},
    {2048, 16384, 145, &ninfer::ops::detail::launch_w8_mma_r32_c128},
    {2048, 16384, 255, &ninfer::ops::detail::launch_w8_mma_r32_c128},
    {2048, 16384, 256, &ninfer::ops::detail::launch_w8_mma_r32_c64},
    {2048, 16384, 384, &ninfer::ops::detail::launch_w8_mma_r32_c64},
    {2048, 16384, 385, &ninfer::ops::detail::launch_w8_mma_r32_c96},
    {2048, 16384, 480, &ninfer::ops::detail::launch_w8_mma_r32_c96},
    {2048, 16384, 481, &ninfer::ops::detail::launch_w8_exact_mma_r32_c96},
    {2048, 16384, 482, &ninfer::ops::detail::launch_w8_mma_r32_c128},
    {2048, 16384, 640, &ninfer::ops::detail::launch_w8_mma_r32_c128},
    {2048, 16384, 641, &ninfer::ops::detail::launch_w8_exact_mma_r32_c128},
    {2048, 16384, 668, &ninfer::ops::detail::launch_w8_exact_mma_r32_c128},
    {2048, 16384, 669, &ninfer::ops::detail::launch_w8_mma_r48_c96},
    {2048, 16384, 672, &ninfer::ops::detail::launch_w8_mma_r48_c96},
    {2048, 16384, 673, &ninfer::ops::detail::launch_w8_exact_mma_r48_c96},
    {2048, 16384, 674, &ninfer::ops::detail::launch_w8_mma_r48_c64},
    {2048, 16384, 704, &ninfer::ops::detail::launch_w8_mma_r48_c64},
    {2048, 16384, 705, &ninfer::ops::detail::launch_w8_mma_r48_c112},
    {2048, 16384, 784, &ninfer::ops::detail::launch_w8_mma_r48_c112},
    {2048, 16384, 785, &ninfer::ops::detail::launch_w8_mma_r48_c128},
    {2048, 16384, 896, &ninfer::ops::detail::launch_w8_mma_r48_c128},
    {2048, 16384, 897, &ninfer::ops::detail::launch_w8_exact_mma_r48_c128},
    {2048, 16384, 912, &ninfer::ops::detail::launch_w8_exact_mma_r48_c128},
    {2048, 16384, 913, &ninfer::ops::detail::launch_w8_mma_r64_c96},
    {2048, 16384, 960, &ninfer::ops::detail::launch_w8_mma_r64_c96},
    {2048, 16384, 961, &ninfer::ops::detail::launch_w8_exact_mma_r64_c96},
    {2048, 16384, 1007, &ninfer::ops::detail::launch_w8_exact_mma_r64_c96},
    {2048, 16384, 1008, &ninfer::ops::detail::launch_w8_mma_r64_c112},
    {2048, 16384, 1009, &ninfer::ops::detail::launch_w8_mma_r64_c128},
    {2048, 16384, 1119, &ninfer::ops::detail::launch_w8_mma_r64_c128},
    {2048, 16384, 1120, &ninfer::ops::detail::launch_w8_mma_r64_c112},
    {2048, 16384, 1121, &ninfer::ops::detail::launch_w8_mma_r64_c128},
    {2048, 16384, 1280, &ninfer::ops::detail::launch_w8_mma_r64_c128},
    {2048, 16384, 1281, &ninfer::ops::detail::launch_w8_exact_mma_r64_c128},
    {2048, 16384, 1313, &ninfer::ops::detail::launch_w8_exact_mma_r64_c128},
    {2048, 16384, 1314, &ninfer::ops::detail::launch_w8_mma_r128_c64},
    {2048, 16384, 1344, &ninfer::ops::detail::launch_w8_mma_r128_c64},
    {2048, 16384, 1345, &ninfer::ops::detail::launch_w8_mma_r96_c96},
    {2048, 16384, 1440, &ninfer::ops::detail::launch_w8_mma_r96_c96},
    {2048, 16384, 1441, &ninfer::ops::detail::launch_w8_exact_mma_r96_c96},
    {2048, 16384, 1500, &ninfer::ops::detail::launch_w8_exact_mma_r96_c96},
    {2048, 16384, 1501, &ninfer::ops::detail::launch_w8_mma_r128_c80},
    {2048, 16384, 1680, &ninfer::ops::detail::launch_w8_mma_r128_c80},
    {2048, 16384, 1681, &ninfer::ops::detail::launch_w8_exact_mma_r128_c80},
    {2048, 16384, 1745, &ninfer::ops::detail::launch_w8_exact_mma_r128_c80},
    {2048, 16384, 1746, &ninfer::ops::detail::launch_w8_mma_r48_c128},
    {2048, 16384, 1791, &ninfer::ops::detail::launch_w8_mma_r48_c128},
    {2048, 16384, 1792, &ninfer::ops::detail::launch_w8_mma_r64_c128},
    {2048, 16384, 1793, &ninfer::ops::detail::launch_w8_mma_r48_c128},
    {2048, 16384, 1919, &ninfer::ops::detail::launch_w8_mma_r48_c128},
    {2048, 16384, 1920, &ninfer::ops::detail::launch_w8_mma_r64_c128},
    {2048, 16384, 1921, &ninfer::ops::detail::launch_w8_exact_mma_r64_c128},
    {2048, 16384, 1953, &ninfer::ops::detail::launch_w8_exact_mma_r64_c128},
    {2048, 16384, 1954, &ninfer::ops::detail::launch_w8_mma_r64_c96},
    {2048, 16384, 2016, &ninfer::ops::detail::launch_w8_mma_r64_c96},
    {2048, 16384, 2017, &ninfer::ops::detail::launch_w8_exact_mma_r64_c96},
    {2048, 16384, 2048, &ninfer::ops::detail::launch_w8_exact_mma_r64_c96},
    {2048, 16384, 2049, &ninfer::ops::detail::launch_w8_mma_r96_c96},
    {2048, 16384, 2112, &ninfer::ops::detail::launch_w8_mma_r96_c96},
    {2048, 16384, 2113, &ninfer::ops::detail::launch_w8_mma_r64_c128},
    {2048, 16384, 3000, &ninfer::ops::detail::launch_w8_mma_r64_c128},
};

struct W8ThrowCase {
    std::int32_t n;
    std::int32_t k;
    std::int32_t t;
};

const std::vector<W8ThrowCase> kW8ThrowCases{
    {6144, 5120, 0},     // T <= 0 is invalid for every w8 class.
    {5120, 25600, 0},    // new class: same guard.
    {5120, 4096, -1},    // new class: same guard.
    {5120, 4608, 32769}, // k=4608 T upper bound.
    {2048, 4608, 32769}, // k=4608 T upper bound.
    {999, 999, 1},       // unregistered shape.
    {1, 25600, 1},       // k=25600 is registered only for n=5120.
    {2048, 25600, 1},    // k=25600 is registered only for n=5120.
};

int w8_head_regression() {
    for (const W8Route& route : kHeadW8Routes) {
        W8Launch launch = nullptr;
        try {
            launch = ninfer::ops::detail::select_w8_a16_launch(route.n, route.k, route.t);
        } catch (const std::exception& error) {
            std::cerr << "FAIL: w8 HEAD route " << route.n << "x" << route.k << " at T="
                      << route.t << " threw: " << error.what() << '\n';
            return 1;
        }
        if (launch != route.launch) {
            std::cerr << "FAIL: w8 HEAD route " << route.n << "x" << route.k << " at T="
                      << route.t << " was perturbed by the new classes\n";
            return 1;
        }
    }
    for (const W8ThrowCase& thrown : kW8ThrowCases) {
        bool did_throw = false;
        try {
            (void)ninfer::ops::detail::select_w8_a16_launch(thrown.n, thrown.k, thrown.t);
        } catch (const std::exception&) {
            did_throw = true;
        }
        if (!did_throw) {
            std::cerr << "FAIL: w8 " << thrown.n << "x" << thrown.k << " at T=" << thrown.t
                      << " must throw\n";
            return 1;
        }
    }
    return 0;
}

// BF16 tier routing (bf16_dispatch.cpp + bf16_launch.h constants): t==1 -> decode;
// t <= small_t_end -> small-t, where small_t_end = kBf16SmallTMaxTokens (32) for n==5120
// and kBf16LinearSmallTDispatchEnd (27) otherwise; above that -> mma.
Bf16Launch expected_bf16_launch(std::int32_t n, std::int32_t t) {
    if (t == 1) { return &ninfer::ops::detail::launch_bf16_decode; }
    const std::int32_t small_t_end = n == 5120 ? 32 : 27;
    if (t <= small_t_end) { return &ninfer::ops::detail::launch_bf16_small_t; }
    return &ninfer::ops::detail::launch_bf16_mma;
}

struct Bf16Route {
    std::int32_t n;
    std::int32_t k;
    std::int32_t t;
    Bf16Launch launch;
};

// The four registered bf16 classes: (14336,5120) and (5120,6144) are the pre-wave
// regression classes; (1280,5120) conv kernel_projection and (256,5120) selector
// hidden_projection are new in this wave. Every boundary (1, 27/28, 32/33) plus far-field
// points; (256,5120) at T=1 is the selector k=1,B=1 case and must be the decode launcher.
const std::vector<Bf16Route> kBf16Routes{
    {14336, 5120, 1, &ninfer::ops::detail::launch_bf16_decode},
    {14336, 5120, 2, &ninfer::ops::detail::launch_bf16_small_t},
    {14336, 5120, 26, &ninfer::ops::detail::launch_bf16_small_t},
    {14336, 5120, 27, &ninfer::ops::detail::launch_bf16_small_t},
    {14336, 5120, 28, &ninfer::ops::detail::launch_bf16_mma},
    {14336, 5120, 100000, &ninfer::ops::detail::launch_bf16_mma},
    {5120, 6144, 1, &ninfer::ops::detail::launch_bf16_decode},
    {5120, 6144, 2, &ninfer::ops::detail::launch_bf16_small_t},
    {5120, 6144, 31, &ninfer::ops::detail::launch_bf16_small_t},
    {5120, 6144, 32, &ninfer::ops::detail::launch_bf16_small_t},
    {5120, 6144, 33, &ninfer::ops::detail::launch_bf16_mma},
    {5120, 6144, 100000, &ninfer::ops::detail::launch_bf16_mma},
    {1280, 5120, 1, &ninfer::ops::detail::launch_bf16_decode},
    {1280, 5120, 2, &ninfer::ops::detail::launch_bf16_small_t},
    {1280, 5120, 27, &ninfer::ops::detail::launch_bf16_small_t},
    {1280, 5120, 28, &ninfer::ops::detail::launch_bf16_mma},
    {1280, 5120, 100000, &ninfer::ops::detail::launch_bf16_mma},
    {256, 5120, 1, &ninfer::ops::detail::launch_bf16_decode},
    {256, 5120, 2, &ninfer::ops::detail::launch_bf16_small_t},
    {256, 5120, 27, &ninfer::ops::detail::launch_bf16_small_t},
    {256, 5120, 28, &ninfer::ops::detail::launch_bf16_mma},
    {256, 5120, 100000, &ninfer::ops::detail::launch_bf16_mma},
};

struct Bf16ThrowCase {
    std::int32_t n;
    std::int32_t k;
    std::int32_t t;
};

const std::vector<Bf16ThrowCase> kBf16ThrowCases{
    {256, 5120, 0},      // T <= 0 is invalid.
    {1280, 2048, 1},     // unregistered shape.
    {256, 51200, 1},     // unregistered shape.
    {5120, 5120, 1},     // unregistered shape.
};

int bf16_shapes() {
    // Sampled sweep over T = 1..100000 including every boundary: dense 1..64 (covers all
    // four classes' tier boundaries and neighbors) plus deterministic far-field points.
    const std::vector<std::int32_t> far_field{100, 1000, 10000, 50000, 100000};
    const std::vector<std::pair<std::int32_t, std::int32_t>> shapes{
        {14336, 5120}, {5120, 6144}, {1280, 5120}, {256, 5120},
    };
    for (const auto& [n, k] : shapes) {
        std::vector<std::int32_t> samples;
        for (std::int32_t t = 1; t <= 64; ++t) { samples.push_back(t); }
        for (const std::int32_t t : far_field) { samples.push_back(t); }
        for (const std::int32_t t : samples) {
            Bf16Launch launch = nullptr;
            try {
                launch = ninfer::ops::detail::select_bf16_a16_launch(n, k, t);
            } catch (const std::exception& error) {
                std::cerr << "FAIL: bf16 " << n << "x" << k << " threw at T=" << t << ": "
                          << error.what() << '\n';
                return 1;
            }
            if (launch != expected_bf16_launch(n, t)) {
                std::cerr << "FAIL: bf16 " << n << "x" << k << " tier routing diverged at T="
                          << t << '\n';
                return 1;
            }
        }
    }
    for (const Bf16Route& route : kBf16Routes) {
        Bf16Launch launch = nullptr;
        try {
            launch = ninfer::ops::detail::select_bf16_a16_launch(route.n, route.k, route.t);
        } catch (const std::exception& error) {
            std::cerr << "FAIL: bf16 " << route.n << "x" << route.k << " at T=" << route.t
                      << " threw: " << error.what() << '\n';
            return 1;
        }
        if (launch != route.launch) {
            std::cerr << "FAIL: bf16 " << route.n << "x" << route.k << " routed to an unexpected "
                         "launcher at T=" << route.t << '\n';
            return 1;
        }
    }
    for (const Bf16ThrowCase& thrown : kBf16ThrowCases) {
        bool did_throw = false;
        try {
            (void)ninfer::ops::detail::select_bf16_a16_launch(thrown.n, thrown.k, thrown.t);
        } catch (const std::exception&) {
            did_throw = true;
        }
        if (!did_throw) {
            std::cerr << "FAIL: bf16 " << thrown.n << "x" << thrown.k << " at T=" << thrown.t
                      << " must throw\n";
            return 1;
        }
    }
    return 0;
}

} // namespace

int main() {
    struct Section {
        const char* name;
        int result;
    };
    const std::vector<Section> sections{
        {"w8 27B new classes: T=1..262144 no-throw + shape-generic set + boundary routing",
         w8_no_throw_generic(5120, 25600) + w8_no_throw_generic(5120, 4096) +
             w8_new_class_boundaries()},
        {"w8 27B new classes: (k_draft+1)*B batch domain (29 values) on shape-generic set",
         w8_batch_domain()},
        {"w8 HEAD regression: pre-existing (n,k) pairs keep the exact pre-wave routing",
         w8_head_regression()},
        {"bf16 classes: tier routing over T=1..100000 incl. boundaries, regression + 27B new",
         bf16_shapes()},
    };
    int failures = 0;
    for (const Section& section : sections) {
        std::cout << (section.result == 0 ? "PASS: " : "FAIL: ") << section.name << '\n';
        failures += section.result;
    }
    if (failures != 0) {
        std::cerr << "linear dispatch shape coverage: " << failures << " violation(s)\n";
        return 1;
    }
    std::cout << "ok: linear dispatch shape coverage\n";
    return 0;
}