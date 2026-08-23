#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    try {
        constexpr std::array<std::int32_t, 2> kA16Cases{1, 2};
        constexpr std::array<std::int32_t, 6> kA8Cases{1, 2, 3, 48, 65, 1024};
        int failures = 0, skipped = 0;
        auto result = run_profile(
            "LinearSwiGLU FP8_A16",
            {QType::FP8_E4M3FN_ROW_BF16S, 34816, 5120, 17408, 1811U, ActivationCompute::A16},
            kA16Cases);
        if (result == 77) { ++skipped; } else { failures += result; }
        result = run_profile(
            "LinearSwiGLU FP8_A8",
            {QType::FP8_E4M3FN_ROW_BF16S, 34816, 5120, 17408, 1813U, ActivationCompute::A8},
            kA8Cases);
        if (result == 77) { ++skipped; } else { failures += result; }
        if (skipped > 0 && failures == 0) {
            // all profiles skipped (no usable device): report skip, not failure.
            std::cout << "SKIP LinearSwiGLU FP8 correctness (no usable CUDA device)\n";
            return 77;
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " LinearSwiGLU FP8 correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "LinearSwiGLU FP8 test failed: " << error.what() << '\n';
        return 1;
    }
}
