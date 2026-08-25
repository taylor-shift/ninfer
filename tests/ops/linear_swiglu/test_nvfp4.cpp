#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    try {
        constexpr std::array<std::int32_t, 4> kA16Cases{1, 4, 8, 16};
        constexpr std::array<std::int32_t, 5> kA4Cases{5, 48, 49, 128, 1024};
        int failures = 0, skipped = 0;
        auto result = run_profile("LinearSwiGLU NVFP4_A16",
                                {QType::NVFP4, 34816, 5120, 17408, 1801U, ActivationCompute::A16},
                                kA16Cases);
        if (result == 77) { ++skipped; } else { failures += result; }
        result = run_profile("LinearSwiGLU NVFP4_A4",
                        {QType::NVFP4, 34816, 5120, 17408, 1803U, ActivationCompute::A4}, kA4Cases);
        if (result == 77) { ++skipped; } else { failures += result; }
        if (skipped > 0 && failures == 0) {
            // all profiles skipped (no usable device): report skip, not failure.
            std::cout << "SKIP LinearSwiGLU NVFP4 correctness (no usable CUDA device)\n";
            return 77;
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " LinearSwiGLU NVFP4 correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "LinearSwiGLU NVFP4 test failed: " << error.what() << '\n';
        return 1;
    }
}
