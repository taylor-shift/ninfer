#pragma once

// Typed Qwen3.6 phase-root allocation clusters shared by the real schedule and its startup
// WorkspaceLayoutBuilder simulation. Child Op scratch remains owned by each Op capacity query.

#include "core/arena.h"
#include "core/layout.h"

#include <cstdint>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::workspace_recipe {

template <class Allocator>
Tensor matrix(Allocator& allocator, DType dtype, std::int32_t rows, std::int32_t tokens) {
    return allocator.alloc(dtype, {rows, tokens});
}

template <class Allocator>
Tensor vector(Allocator& allocator, DType dtype, std::int32_t elements) {
    return allocator.alloc(dtype, {elements});
}

struct TextPrefillRoots {
    Tensor ids;
    Tensor positions;
    Tensor rope_positions;
    Tensor residual;
    Tensor scatter_indices;
};

template <class Config, class Allocator>
TextPrefillRoots text_prefill_roots(Allocator& allocator, std::int32_t tokens,
                                    std::int32_t rope_axes, std::int32_t scatter_tokens) {
    TextPrefillRoots out;
    out.ids       = vector(allocator, DType::I32, tokens);
    out.positions = vector(allocator, DType::I32, tokens);
    if (rope_axes != 0) { out.rope_positions = matrix(allocator, DType::I32, tokens, rope_axes); }
    out.residual = matrix(allocator, DType::BF16, Config::hidden, tokens);
    if (scatter_tokens != 0) {
        out.scatter_indices = vector(allocator, DType::I32, scatter_tokens);
    }
    return out;
}

template <class Allocator>
Tensor visual_scatter_indices(Allocator& allocator, std::int32_t tokens) {
    return vector(allocator, DType::I32, tokens);
}

struct TextAttentionProjectionRoots {
    Tensor hidden;
    Tensor query;
    Tensor gate;
    Tensor key;
    Tensor value;
};

template <class Config, class Allocator>
TextAttentionProjectionRoots text_attention_projection(Allocator& allocator, std::int32_t tokens) {
    return {
        matrix(allocator, DType::BF16, Config::hidden, tokens),
        matrix(allocator, DType::BF16, Config::query_size, tokens),
        matrix(allocator, DType::BF16, Config::query_size, tokens),
        matrix(allocator, DType::BF16, Config::kv_size, tokens),
        matrix(allocator, DType::BF16, Config::kv_size, tokens),
    };
}

struct TextAttentionResultRoots {
    Tensor normalized_query;
    Tensor normalized_key;
    Tensor attention;
};

template <class Config, class Allocator>
TextAttentionResultRoots text_attention_results(Allocator& allocator, std::int32_t tokens) {
    return {
        matrix(allocator, DType::BF16, Config::query_size, tokens),
        matrix(allocator, DType::BF16, Config::kv_size, tokens),
        matrix(allocator, DType::BF16, Config::query_size, tokens),
    };
}

struct GdnControlRoots {
    Tensor hidden;
    Tensor g;
    Tensor beta;
};

template <class Config, class Allocator>
GdnControlRoots gdn_control(Allocator& allocator, std::int32_t tokens) {
    return {
        matrix(allocator, DType::BF16, Config::hidden, tokens),
        matrix(allocator, DType::FP32, Config::gdn_value_heads, tokens),
        matrix(allocator, DType::FP32, Config::gdn_value_heads, tokens),
    };
}

struct GdnProjectionRoots {
    Tensor output_gate;
    Tensor query;
    Tensor key;
    Tensor value;
};

template <class Config, class Allocator>
GdnProjectionRoots gdn_projection(Allocator& allocator, std::int32_t tokens) {
    return {
        matrix(allocator, DType::BF16, Config::value_dim, tokens),
        matrix(allocator, DType::BF16, Config::key_dim, tokens),
        matrix(allocator, DType::BF16, Config::key_dim, tokens),
        matrix(allocator, DType::BF16, Config::value_dim, tokens),
    };
}

struct GdnPrefillConvRoots {
    Tensor projected;
    Tensor convolved;
};

template <class Config, class Allocator>
GdnPrefillConvRoots gdn_prefill_conv(Allocator& allocator, std::int32_t tokens) {
    return {
        matrix(allocator, DType::BF16, Config::convolution_dim, tokens),
        matrix(allocator, DType::BF16, Config::convolution_dim, tokens),
    };
}

template <class Config, class Allocator>
Tensor gdn_recurrent_output(Allocator& allocator, std::int32_t tokens) {
    return matrix(allocator, DType::BF16, Config::value_dim, tokens);
}

template <class Config, class Allocator>
Tensor gdn_normalized_output(Allocator& allocator, std::int32_t tokens) {
    return matrix(allocator, DType::BF16, Config::value_dim, tokens);
}

template <class Config, class Allocator>
Tensor post_mixer_hidden(Allocator& allocator, std::int32_t tokens) {
    return matrix(allocator, DType::BF16, Config::hidden, tokens);
}

struct MtpStemRoots {
    Tensor embedding;
    Tensor normalized_embedding;
    Tensor normalized_hidden;
    Tensor packed_input;
    Tensor residual;
    Tensor attention_hidden;
};

template <class Config, class Allocator>
MtpStemRoots mtp_stem(Allocator& allocator, std::int32_t tokens, bool allocate_embedding) {
    MtpStemRoots out;
    if (allocate_embedding) {
        out.embedding = matrix(allocator, DType::BF16, Config::hidden, tokens);
    }
    out.normalized_embedding = matrix(allocator, DType::BF16, Config::hidden, tokens);
    out.normalized_hidden    = matrix(allocator, DType::BF16, Config::hidden, tokens);
    out.packed_input         = matrix(allocator, DType::BF16, Config::mtp_input_rows, tokens);
    out.residual             = matrix(allocator, DType::BF16, Config::hidden, tokens);
    out.attention_hidden     = matrix(allocator, DType::BF16, Config::hidden, tokens);
    return out;
}

struct MtpAttentionProjectionRoots {
    Tensor query;
    Tensor key;
    Tensor gate;
    Tensor value;
};

template <class Config, class Allocator>
MtpAttentionProjectionRoots mtp_attention_projection(Allocator& allocator, std::int32_t tokens) {
    return {
        matrix(allocator, DType::BF16, Config::query_size, tokens),
        matrix(allocator, DType::BF16, Config::kv_size, tokens),
        matrix(allocator, DType::BF16, Config::query_size, tokens),
        matrix(allocator, DType::BF16, Config::kv_size, tokens),
    };
}

struct MtpAttentionResultRoots {
    Tensor normalized_query;
    Tensor normalized_key;
    Tensor attention;
};

template <class Config, class Allocator>
MtpAttentionResultRoots mtp_attention_results(Allocator& allocator, std::int32_t tokens) {
    return {
        matrix(allocator, DType::BF16, Config::query_size, tokens),
        matrix(allocator, DType::BF16, Config::kv_size, tokens),
        matrix(allocator, DType::BF16, Config::query_size, tokens),
    };
}

struct MtpPostAttentionRoots {
    Tensor output;
    Tensor post_mixer_hidden;
};

template <class Config, class Allocator>
MtpPostAttentionRoots mtp_post_attention(Allocator& allocator, std::int32_t tokens) {
    return {
        matrix(allocator, DType::BF16, Config::hidden, tokens),
        matrix(allocator, DType::BF16, Config::hidden, tokens),
    };
}

struct DFlashContextRoots {
    Tensor projected;
    Tensor normalized;
};

template <class Config, class Allocator>
DFlashContextRoots dflash_context(Allocator& allocator, std::int32_t tokens) {
    return {
        matrix(allocator, DType::BF16, Config::hidden, tokens),
        matrix(allocator, DType::BF16, Config::hidden, tokens),
    };
}

struct DFlashContextLayerRoots {
    Tensor key_raw;
    Tensor value;
    Tensor key;
};

template <class Config, class Allocator>
DFlashContextLayerRoots dflash_context_layer(Allocator& allocator, std::int32_t tokens) {
    return {
        matrix(allocator, DType::BF16, Config::kv_size, tokens),
        matrix(allocator, DType::BF16, Config::kv_size, tokens),
        matrix(allocator, DType::BF16, Config::kv_size, tokens),
    };
}

struct DFlashProposalRoots {
    Tensor ids;
    Tensor positions;
    Tensor residual;
};

template <class Config, class Allocator>
DFlashProposalRoots dflash_proposal(Allocator& allocator, std::int32_t tokens) {
    return {
        vector(allocator, DType::I32, tokens),
        vector(allocator, DType::I32, tokens),
        matrix(allocator, DType::BF16, Config::hidden, tokens),
    };
}

struct DFlashAttentionRoots {
    Tensor hidden;
    Tensor query_raw;
    Tensor key_raw;
    Tensor value;
    Tensor query;
    Tensor key;
    Tensor attention;
};

template <class Config, class Allocator>
DFlashAttentionRoots dflash_attention(Allocator& allocator, std::int32_t tokens) {
    return {
        matrix(allocator, DType::BF16, Config::hidden, tokens),
        matrix(allocator, DType::BF16, Config::query_size, tokens),
        matrix(allocator, DType::BF16, Config::kv_size, tokens),
        matrix(allocator, DType::BF16, Config::kv_size, tokens),
        matrix(allocator, DType::BF16, Config::query_size, tokens),
        matrix(allocator, DType::BF16, Config::kv_size, tokens),
        matrix(allocator, DType::BF16, Config::query_size, tokens),
    };
}

struct DFlashMlpRoots {
    Tensor hidden;
    Tensor intermediate;
};

template <class Config, class Allocator>
DFlashMlpRoots dflash_mlp(Allocator& allocator, std::int32_t tokens) {
    return {
        matrix(allocator, DType::BF16, Config::hidden, tokens),
        matrix(allocator, DType::BF16, Config::intermediate, tokens),
    };
}

// DFlash 2 path selector (spec doc 06 section 8): draft columns cols = drafts*batch, drafts in
// [1,7], batch in [1,8]. scores is the full 4-D [k,16,16,B] edge-score table the walk reads.
struct DFlashSelectorRoots {
    Tensor candidates;
    Tensor unary;
    Tensor hidden_proj;
    Tensor scores;
};

template <class Config, class Allocator>
DFlashSelectorRoots dflash_selector(Allocator& allocator, std::int32_t drafts, std::int32_t batch) {
    const std::int32_t cols = drafts * batch;
    return {
        matrix(allocator, DType::I32, Config::selector_top_k, cols),
        matrix(allocator, DType::FP32, Config::selector_top_k, cols),
        matrix(allocator, DType::BF16, Config::selector_rank, cols),
        allocator.alloc(DType::FP32, {drafts, Config::selector_top_k, Config::selector_top_k,
                                      batch}),
    };
}

// DFlash 2 grouped dynamic convs (spec doc 06 decision D7), per layer over the full block columns
// tokens = (drafts+1)*batch. Each *_dynamic is the kernel_projection GEMM output [2*conv_kernel*
// (hidden/conv_group_size), tokens] holding the two tap use-slices (rows [0:640) and [640:1280)
// for the 27B geometry); each *_out is the shared [hidden, tokens] destination of that conv's two
// applications (input-side then output-side; never live together).
struct DFlashConvLayerRoots {
    Tensor attention_dynamic;
    Tensor attention_out;
    Tensor mlp_dynamic;
    Tensor mlp_out;
};

template <class Config, class Allocator>
DFlashConvLayerRoots dflash_conv_layer(Allocator& allocator, std::int32_t tokens) {
    const std::int32_t dynamic_rows =
        2 * Config::conv_kernel_size * (Config::hidden / Config::conv_group_size);
    return {
        matrix(allocator, DType::BF16, dynamic_rows, tokens),
        matrix(allocator, DType::BF16, Config::hidden, tokens),
        matrix(allocator, DType::BF16, dynamic_rows, tokens),
        matrix(allocator, DType::BF16, Config::hidden, tokens),
    };
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::workspace_recipe
