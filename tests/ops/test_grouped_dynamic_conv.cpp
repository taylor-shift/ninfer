// Op unit tests for grouped_dynamic_causal_conv (DFlash 2 two-tap grouped dynamic causal
// convolution over the full block columns).
//
// Every case is checked against an independent CPU reference computed in FP64 from the exact
// BF16 values the kernel reads, under the engine layout law (element (r, c) of a contiguous
// [R, C] tensor at c*R + r, first dimension fastest):
//
//   hidden   [5120, C]     element (c, t) at t*5120 + c
//   dynamic  [640, C]      element (g, t) at t*640 + g (rows 0..319 = tap 0, 320..639 = tap 1)
//   base     [2, 2, 5120]  element (use, tap, ch) at use + tap*2 + ch*4
//   out      [5120, C]     element (c, t) at t*5120 + c
//
// with C = (k+1)*B batch-major block columns (column m = b*(k+1) + d) and, for every channel c
// and column t:
//
//   out[t,c] = (base[use,0,c] + dyn[t, c/16])         * x[t,c]
//            + (base[use,1,c] + dyn[t, 320 + c/16])   * x[t-1,c],
//
// where x[t-1,c] is read as zero at every row anchor (t % width == 0, width = C/B, including
// t = 0): each batch row is an independent block-diffusion sequence and the conv never wraps
// across rows. The B>1 cases additionally verify that the anchor zeroing is exercised, not
// vacuous: the would-be x_prev term at each interior row anchor must be non-negligible, so a
// kernel that leaks x[t-1] across an anchor would miss the reference by far more than the
// tolerance (e.g. at C=12, B=4, width=3 the column t=3 must equal the anchor formula
// (base0 + dyn0) * x[3] with no x[2] term).
#include "ninfer/ops/grouped_dynamic_conv.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kChannels = 5120;
constexpr std::int32_t kGroups   = 320;
constexpr std::int32_t kDynRows  = 640;
constexpr std::int32_t kUses     = 2;
constexpr std::int32_t kTaps     = 2;

// Single BF16 round-to-nearest-even store after FP32 accumulation: the same ulp-based
// tolerance as the single-store contract of test_residual_add.
constexpr PointwiseCriterion grouped_conv_bf16_criterion() {
    return {/*absolute*/ 0.0, /*relative*/ 3.95e-3};
}

std::size_t hidden_offset(std::int32_t c, std::int32_t t) {
    return static_cast<std::size_t>(t) * kChannels + static_cast<std::size_t>(c);
}

std::size_t dyn_offset(std::int32_t g, std::int32_t t) {
    return static_cast<std::size_t>(t) * kDynRows + static_cast<std::size_t>(g);
}

std::size_t base_offset(std::int32_t use, std::int32_t tap, std::int32_t c) {
    return static_cast<std::size_t>(use) + static_cast<std::size_t>(tap) * 2 +
           static_cast<std::size_t>(c) * 4;
}

// FP64 oracle over the logical BF16 inputs (post round_to_bf16, exactly the values the kernel
// reads). x_prev is zero at every row anchor t % width == 0 (width = columns / batch).
std::vector<double> conv_oracle(const std::vector<float>& hidden,
                                const std::vector<float>& dynamic, const std::vector<float>& base,
                                std::int32_t columns, std::int32_t width, std::int32_t use) {
    std::vector<double> expected(hidden.size(), 0.0);
    for (std::int32_t t = 0; t < columns; ++t) {
        for (std::int32_t c = 0; c < kChannels; ++c) {
            const std::int32_t g = c / 16;
            const double x_cur  = static_cast<double>(hidden[hidden_offset(c, t)]);
            const double x_prev =
                (t % width == 0) ? 0.0 : static_cast<double>(hidden[hidden_offset(c, t - 1)]);
            const double tap0 = static_cast<double>(base[base_offset(use, 0, c)]) +
                                static_cast<double>(dynamic[dyn_offset(g, t)]);
            const double tap1 = static_cast<double>(base[base_offset(use, 1, c)]) +
                                static_cast<double>(dynamic[dyn_offset(kGroups + g, t)]);
            expected[hidden_offset(c, t)] = tap0 * x_cur + tap1 * x_prev;
        }
    }
    return expected;
}

// Anchor-zeroing fixture integrity: with batch > 1 the would-be x_prev term at the interior
// row anchors must be non-negligible somewhere, otherwise the zeroing is not exercised.
bool anchor_term_exercised(const std::vector<float>& hidden, const std::vector<float>& dynamic,
                           const std::vector<float>& base, std::int32_t columns, std::int32_t width,
                           std::int32_t use) {
    double max_term = 0.0;
    for (std::int32_t t = width; t < columns; t += width) {  // interior row anchors only
        for (std::int32_t c = 0; c < kChannels; ++c) {
            const std::int32_t g = c / 16;
            const double term = (static_cast<double>(base[base_offset(use, 1, c)]) +
                                 static_cast<double>(dynamic[dyn_offset(kGroups + g, t)])) *
                                static_cast<double>(hidden[hidden_offset(c, t - 1)]);
            max_term = std::max(max_term, std::abs(term));
        }
    }
    return max_term >= 1e-1;
}

int run_case(std::int32_t columns, std::int32_t batch, std::int32_t use, std::uint32_t seed) {
    const std::int32_t width = columns / batch;
    const std::string label = "grouped_dynamic_causal_conv C=" + std::to_string(columns) +
                              " B=" + std::to_string(batch) + " use=" + std::to_string(use);

    std::vector<float> hidden(static_cast<std::size_t>(kChannels) * columns);
    std::vector<float> dynamic(static_cast<std::size_t>(kDynRows) * columns);
    std::vector<float> base(static_cast<std::size_t>(kUses) * kTaps * kChannels);
    fill_uniform(hidden, seed, -4.0f, 4.0f);
    fill_uniform(dynamic, seed + 1u, -2.0f, 2.0f);
    fill_uniform(base, seed + 2u, -2.0f, 2.0f);
    round_to_bf16(hidden);
    round_to_bf16(dynamic);
    round_to_bf16(base);
    if (batch > 1 && !anchor_term_exercised(hidden, dynamic, base, columns, width, use)) {
        std::cerr << label << ": fixture would make the row-anchor zeroing vacuous\n";
        return 1;
    }

    const auto expected = conv_oracle(hidden, dynamic, base, columns, width, use);
    const auto hidden_bits = [&hidden]() {
        std::vector<std::uint16_t> bits(hidden.size());
        for (std::size_t i = 0; i < hidden.size(); ++i) { bits[i] = f32_to_bf16(hidden[i]); }
        return bits;
    }();
    const auto dynamic_bits = [&dynamic]() {
        std::vector<std::uint16_t> bits(dynamic.size());
        for (std::size_t i = 0; i < dynamic.size(); ++i) { bits[i] = f32_to_bf16(dynamic[i]); }
        return bits;
    }();
    const auto base_bits = [&base]() {
        std::vector<std::uint16_t> bits(base.size());
        for (std::size_t i = 0; i < base.size(); ++i) { bits[i] = f32_to_bf16(base[i]); }
        return bits;
    }();

    GuardedDeviceBuffer device_hidden(hidden_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_dynamic(dynamic_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_base(base_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_out(hidden_bits.size() * sizeof(std::uint16_t));
    device_hidden.copy_from_host(hidden_bits.data(), device_hidden.bytes());
    device_dynamic.copy_from_host(dynamic_bits.data(), device_dynamic.bytes());
    device_base.copy_from_host(base_bits.data(), device_base.bytes());
    device_out.fill(0xcd);

    Tensor hidden_tensor(device_hidden.data(), DType::BF16, {kChannels, columns});
    Tensor dynamic_tensor(device_dynamic.data(), DType::BF16, {kDynRows, columns});
    Tensor base_tensor(device_base.data(), DType::BF16, {kUses, kTaps, kChannels});
    Tensor out_tensor(device_out.data(), DType::BF16, {kChannels, columns});
    ops::grouped_dynamic_causal_conv(hidden_tensor, dynamic_tensor, base_tensor, out_tensor, use,
                                     batch, nullptr);
    cuda_synchronize();

    int failures = 0;
    failures += verify_pointwise(label + " out",
                                 from_device_bf16(device_out.data(), hidden_bits.size()), expected,
                                 grouped_conv_bf16_criterion());
    failures += verify_exact((label + " hidden unchanged").c_str(),
                             from_device<std::uint16_t>(device_hidden.data(), hidden_bits.size()),
                             hidden_bits);
    failures += verify_exact((label + " dynamic unchanged").c_str(),
                             from_device<std::uint16_t>(device_dynamic.data(), dynamic_bits.size()),
                             dynamic_bits);
    failures += verify_exact((label + " base unchanged").c_str(),
                             from_device<std::uint16_t>(device_base.data(), base_bits.size()),
                             base_bits);
    failures += device_hidden.verify_guards((label + " hidden").c_str());
    failures += device_dynamic.verify_guards((label + " dynamic").c_str());
    failures += device_base.verify_guards((label + " base").c_str());
    failures += device_out.verify_guards((label + " out").c_str());
    return failures;
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

// Wrapper validation contract (src/ops/wrapper/grouped_dynamic_conv.cpp): registered domain
// C = (k+1)*B, batch_size factorization, use, contiguous BF16 shapes, hidden/out overlap.
int rejection_cases() {
    constexpr std::int32_t columns = 8;
    constexpr std::int32_t batch   = 4;  // C = (1+1)*4, k = 1
    GuardedDeviceBuffer hidden_buf(static_cast<std::size_t>(kChannels) * columns * 2);
    GuardedDeviceBuffer hidden11_buf(static_cast<std::size_t>(kChannels) * 11 * 2);
    GuardedDeviceBuffer hidden_f32_buf(static_cast<std::size_t>(kChannels) * columns * 4);
    GuardedDeviceBuffer dynamic_buf(static_cast<std::size_t>(kDynRows) * columns * 2);
    GuardedDeviceBuffer base_buf(static_cast<std::size_t>(kUses) * kTaps * kChannels * 2);
    GuardedDeviceBuffer out_buf(static_cast<std::size_t>(kChannels) * columns * 2);
    GuardedDeviceBuffer permute_src_buf(static_cast<std::size_t>(columns) * kChannels * 2);
    hidden_buf.fill(0x11);
    hidden11_buf.fill(0x22);
    hidden_f32_buf.fill(0x33);
    dynamic_buf.fill(0x44);
    base_buf.fill(0x55);
    out_buf.fill(0x66);
    permute_src_buf.fill(0x77);

    Tensor hidden(hidden_buf.data(), DType::BF16, {kChannels, columns});
    Tensor dynamic(dynamic_buf.data(), DType::BF16, {kDynRows, columns});
    Tensor base(base_buf.data(), DType::BF16, {kUses, kTaps, kChannels});
    Tensor out(out_buf.data(), DType::BF16, {kChannels, columns});

    int failures = 0;
    // 11 is outside the registered (k+1)*B domain (prime, k = 10 > 7).
    failures += expect_invalid_argument(
        [&] {
            Tensor hidden11(hidden11_buf.data(), DType::BF16, {kChannels, 11});
            ops::grouped_dynamic_causal_conv(hidden11, dynamic, base, out, 0, 1, nullptr);
        },
        "grouped_dynamic_causal_conv rejection C=11",
        "grouped_dynamic_causal_conv: C must be (k+1)*B for k in 1..7 and B in 1..8");
    failures += expect_invalid_argument(
        [&] { ops::grouped_dynamic_causal_conv(hidden, dynamic, base, out, 0, 0, nullptr); },
        "grouped_dynamic_causal_conv rejection batch_size=0",
        "grouped_dynamic_causal_conv: batch_size must be in 1..8");
    // 8 % 3 != 0: batch_size is not the factor of the domain.
    failures += expect_invalid_argument(
        [&] { ops::grouped_dynamic_causal_conv(hidden, dynamic, base, out, 0, 3, nullptr); },
        "grouped_dynamic_causal_conv rejection batch_size=3",
        "grouped_dynamic_causal_conv: C must equal (k+1)*batch_size for k in 1..7");
    failures += expect_invalid_argument(
        [&] { ops::grouped_dynamic_causal_conv(hidden, dynamic, base, out, 2, batch, nullptr); },
        "grouped_dynamic_causal_conv rejection use=2",
        "grouped_dynamic_causal_conv: use must be 0 or 1");
    failures += expect_invalid_argument(
        [&] {
            Tensor hidden_f32(hidden_f32_buf.data(), DType::FP32, {kChannels, columns});
            ops::grouped_dynamic_causal_conv(hidden_f32, dynamic, base, out, 0, batch, nullptr);
        },
        "grouped_dynamic_causal_conv rejection hidden FP32",
        "grouped_dynamic_causal_conv: hidden must be a contiguous BF16 matrix");
    // permute({1,0}) of [C, 5120] yields a non-contiguous [5120, C] view with the right ne.
    failures += expect_invalid_argument(
        [&] {
            const Tensor hidden_source(permute_src_buf.data(), DType::BF16, {columns, kChannels});
            const Tensor hidden_permuted = hidden_source.permute({1, 0, 2, 3});
            ops::grouped_dynamic_causal_conv(hidden_permuted, dynamic, base, out, 0, batch,
                                             nullptr);
        },
        "grouped_dynamic_causal_conv rejection hidden non-contiguous",
        "grouped_dynamic_causal_conv: hidden must be a contiguous BF16 matrix");
    failures += expect_invalid_argument(
        [&] {
            Tensor out_alias(hidden_buf.data(), DType::BF16, {kChannels, columns});
            ops::grouped_dynamic_causal_conv(hidden, dynamic, base, out_alias, 0, batch,
                                             nullptr);
        },
        "grouped_dynamic_causal_conv rejection hidden/out overlap",
        "grouped_dynamic_causal_conv: hidden and out must not overlap");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    // Registered-domain cases: C = (k+1)*B. B = 1 with k = 7 (max drafts); B = 4 with widths
    // 2/4/3 exercising the per-row anchor zeroing, including the C=12 (k=2, width=3) case
    // whose interior anchors t=3,6,9 must not leak the previous column.
    const std::array<std::pair<std::int32_t, std::int32_t>, 4> cases = {
        {std::pair<std::int32_t, std::int32_t>{8, 4},     // k=1, width 2
         std::pair<std::int32_t, std::int32_t>{16, 4},    // k=3, width 4
         std::pair<std::int32_t, std::int32_t>{12, 4},    // k=2, width 3
         std::pair<std::int32_t, std::int32_t>{8, 1}}};   // k=7, width 8
    std::uint32_t seed = 4001u;
    for (const auto& [columns, batch] : cases) {
        for (const std::int32_t use : {0, 1}) {
            failures += run_case(columns, batch, use, seed++);
        }
    }
    failures += rejection_cases();

    std::cout << (failures == 0 ? "OK" : "FAIL") << " grouped_dynamic_causal_conv\n";
    return failures == 0 ? 0 : 1;
}