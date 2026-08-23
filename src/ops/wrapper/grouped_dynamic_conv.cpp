// ninfer::ops - grouped_dynamic_causal_conv wrapper: public api validation and launcher dispatch.
#include "ninfer/ops/grouped_dynamic_conv.h"

#include "ops/launcher/grouped_dynamic_conv.h" // detail::grouped_dynamic_causal_conv_launch

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kChannels    = 5120;
constexpr std::int32_t kDynamicRows = 640;
constexpr std::int32_t kUses        = 2;
constexpr std::int32_t kTaps        = 2;

// Registered domain: C = (k+1)*B for k in 1..7 draft tokens and B in 1..8 batch rows — the exact
// 29-value subset of 2..64, never the whole interval (for example 11 and 13 are absent).
bool in_registered_domain(std::int32_t columns) {
    if (columns < 2 || columns > 64) { return false; }
    for (std::int32_t k = 1; k <= 7; ++k) {
        if (columns % (k + 1) == 0) {
            const std::int32_t b = columns / (k + 1);
            if (b >= 1 && b <= 8) { return true; }
        }
    }
    return false;
}

void require_bf16_matrix(const Tensor& tensor, std::int32_t rows, std::int32_t columns,
                         const char* name) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != rows || tensor.ne[1] != columns ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        tensor.data == nullptr) {
        throw std::invalid_argument(std::string("grouped_dynamic_causal_conv: ") + name +
                                    " must be a contiguous BF16 matrix");
    }
}

void require_bf16_base(const Tensor& tensor) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != kUses || tensor.ne[1] != kTaps ||
        tensor.ne[2] != kChannels || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        tensor.data == nullptr) {
        throw std::invalid_argument(
            "grouped_dynamic_causal_conv: base must be a contiguous BF16 [2,2,5120] "
            "tensor (use, tap, channel)");
    }
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    return lhs_begin < rhs_begin + rhs.bytes() && rhs_begin < lhs_begin + lhs.bytes();
}

} // namespace

void grouped_dynamic_causal_conv(const Tensor& hidden, const Tensor& dynamic, const Tensor& base,
                                 Tensor& out, std::int32_t use, std::int32_t batch_size,
                                 cudaStream_t stream) {
    const std::int32_t columns = hidden.ne[1];
    if (!in_registered_domain(columns)) {
        throw std::invalid_argument(
            "grouped_dynamic_causal_conv: C must be (k+1)*B for k in 1..7 and B in 1..8");
    }
    if (batch_size < 1 || batch_size > 8) {
        throw std::invalid_argument(
            "grouped_dynamic_causal_conv: batch_size must be in 1..8");
    }
    if (columns % batch_size != 0 || columns / batch_size < 2 || columns / batch_size > 8) {
        throw std::invalid_argument(
            "grouped_dynamic_causal_conv: C must equal (k+1)*batch_size for k in 1..7");
    }
    if (use < 0 || use >= kUses) {
        throw std::invalid_argument("grouped_dynamic_causal_conv: use must be 0 or 1");
    }
    require_bf16_matrix(hidden, kChannels, columns, "hidden");
    require_bf16_matrix(dynamic, kDynamicRows, columns, "dynamic");
    require_bf16_base(base);
    require_bf16_matrix(out, kChannels, columns, "out");
    if (overlaps(hidden, out)) {
        throw std::invalid_argument(
            "grouped_dynamic_causal_conv: hidden and out must not overlap");
    }
    detail::grouped_dynamic_causal_conv_launch(hidden, dynamic, base, out, use, batch_size, stream);
}

} // namespace ninfer::ops