#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"
#include "targets/qwen3_6/impl/runtime/workspace_recipe.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ninfer/ops/argmax.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/bidirectional_gqa_attention.h"
#include "ninfer/ops/dflash_selector.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/grouped_dynamic_conv.h"
#include "ninfer/ops/kv_cache_append_prefix.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/prepare_masked_block.h"
#include "ninfer/ops/prepare_ragged_prefix.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer/ops/swa.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

void require_dflash_state(const PrefillContext& state) {
    if (state.dflash == nullptr || !state.execution.model.dflash.has_value()) {
        throw std::logic_error("DFlash schedule requires DFlash weights and state");
    }
}

DFlashPersistentState& dflash_state(PrefillContext& state) {
    require_dflash_state(state);
    return *state.dflash;
}

DFlashPersistentState& dflash_state(DFlashBatchContext& state) { return state.dflash; }

DFlashPersistentState& dflash_state(DFlashAppendContext& state) { return state.dflash; }

template <class V>
DFlashFeatureSink prefill_feature_sink_impl(PrefillContext& state,
                                            DFlashFeatureSink::PrefillConsumer consume_prefill) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash feature capture is unavailable for this target");
    } else {
        require_dflash_state(state);
        using Config = typename V::DFlashConfig;
        return DFlashFeatureSink{
            .features        = &dflash_state(state).prefill_features,
            .positions       = &dflash_state(state).prefill_positions,
            .layers          = std::span<const int>(Config::target_feature_layers),
            .consume_prefill = std::move(consume_prefill),
        };
    }
}

template <class V>
DFlashFeatureSink batch_feature_sink_impl(DFlashBatchContext& state, const Tensor& lanes,
                                          const Tensor& valid_columns, std::int32_t width,
                                          std::int32_t batch_size) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash feature capture is unavailable for this target");
    } else {
        using Config = typename V::DFlashConfig;
        return DFlashFeatureSink{
            .batch_features      = &dflash_state(state).pending_features,
            .batch_lanes         = &lanes,
            .batch_valid_columns = &valid_columns,
            .batch_width         = width,
            .batch_size          = batch_size,
            .layers              = std::span<const int>(Config::target_feature_layers),
        };
    }
}

template <class V, class Context>
void append_context_impl(Context& state, const Tensor& features, const Tensor& positions,
                         const Tensor& commit_counts, const Tensor& lanes, const Tensor& table_rows,
                         ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash context append is unavailable for this target");
    } else {
        using Config               = typename V::DFlashConfig;
        const std::int32_t width   = features.ne[1];
        const std::int32_t batch   = features.ne[2];
        const std::int32_t columns = width * batch;
        if (width <= 0 || batch <= 0 || features.dtype != DType::BF16 ||
            features.ne[0] != Config::feature_rows || features.ne[3] != 1 ||
            positions.dtype != DType::I32 || positions.ne[0] != width || positions.ne[1] != batch ||
            commit_counts.dtype != DType::I32 || commit_counts.ne[0] != batch ||
            lanes.dtype != DType::I32 || lanes.ne[0] != batch || table_rows.dtype != DType::I32 ||
            table_rows.ne[0] != batch) {
            throw std::invalid_argument("DFlash context append inputs are invalid");
        }
        const bool replace_local_window = batch == 1 && width > Config::local_capacity;
        if (replace_local_window && (envelope.min_count != static_cast<std::uint32_t>(width) ||
                                     envelope.max_count != static_cast<std::uint32_t>(width))) {
            throw std::invalid_argument(
                "DFlash oversized local append requires an exact full-prefix commit");
        }
        const int local_offset = replace_local_window ? width - Config::local_capacity : 0;
        const int local_width  = replace_local_window ? Config::local_capacity : width;
        const ops::KVCacheAppendPrefixExecutionEnvelope local_envelope{
            replace_local_window ? static_cast<std::uint32_t>(Config::local_capacity)
                                 : envelope.min_count,
            replace_local_window ? static_cast<std::uint32_t>(Config::local_capacity)
                                 : envelope.max_count,
        };
        Tensor local_counts = commit_counts;
        if (replace_local_window) {
            if (!state.execution.io.dflash_prefill) {
                throw std::logic_error("DFlash prefill count storage is unavailable");
            }
            local_counts = state.execution.io.dflash_prefill->produced_count;
            ops::set_i32_scalar(local_counts, Config::local_capacity,
                                state.execution.device.stream);
        }

        const auto context_roots =
            workspace_recipe::dflash_context<Config>(state.execution.work, columns);
        Tensor projected = context_roots.projected;
        ops::linear(features.view({Config::feature_rows, columns}),
                    state.execution.model.dflash->feature_projection, projected,
                    state.execution.device.stream);
        Tensor context = context_roots.normalized;
        ops::rmsnorm(projected, state.execution.model.dflash->context_norm, Config::rms_epsilon,
                     false, context, state.execution.device.stream);

        for (int layer = 0; layer < Config::layers; ++layer) {
            auto layer_scope = state.execution.work.scope();
            const auto& weight =
                state.execution.model.dflash->layers.at(static_cast<std::size_t>(layer));
            const bool local_layer  = layer < Config::local_layers;
            const int layer_width   = local_layer ? local_width : width;
            const int layer_columns = layer_width * batch;
            Tensor layer_context    = local_layer && replace_local_window
                                          ? context.slice(1, local_offset, local_width)
                                          : context;
            Tensor layer_positions  = local_layer && replace_local_window
                                          ? positions.slice(0, local_offset, local_width)
                                          : positions;
            auto layer_roots =
                workspace_recipe::dflash_context_layer<Config>(state.execution.work, layer_columns);
            Tensor key_raw =
                layer_roots.key_raw.view({Config::head_dim, Config::kv_heads, layer_columns});
            Tensor value =
                layer_roots.value.view({Config::head_dim, Config::kv_heads, layer_columns});
            Tensor key_flat   = key_raw.view({Config::kv_size, layer_columns});
            Tensor value_flat = value.view({Config::kv_size, layer_columns});
            ops::linear_pair(layer_context, weight.context_key, weight.context_value, key_flat,
                             value_flat, state.execution.device.stream);
            Tensor key = layer_roots.key.view({Config::head_dim, Config::kv_heads, layer_columns});
            ops::rmsnorm(key_raw, weight.key_norm, Config::rms_epsilon, false, key,
                         state.execution.device.stream);
            ops::rope(layer_positions.view({layer_columns}), Config::head_dim, Config::rope_theta,
                      key, state.execution.device.stream);
            Tensor key_batch = key.view({Config::head_dim, Config::kv_heads, layer_width, batch});
            Tensor value_batch =
                value.view({Config::head_dim, Config::kv_heads, layer_width, batch});
            Tensor position_batch = layer_positions.view({layer_width, batch});
            if (local_layer) {
                ops::kv_cache_append_prefix(
                    key_batch, value_batch, position_batch, local_counts, lanes, local_envelope,
                    dflash_state(state).local_layer(static_cast<std::uint32_t>(layer)),
                    state.execution.device.stream);
            } else {
                ops::kv_cache_append_prefix(
                    key_batch, value_batch, position_batch, commit_counts, table_rows, envelope,
                    dflash_state(state).full_batch_layer(0), state.execution.device.stream);
            }
        }
    }
}

template <class V>
void propose_batch_impl(DFlashBatchContext& state, qwen3_6::DFlashDecodeState& frame,
                        std::int32_t batch_size, std::uint32_t k, DFlashEnvelopes envelopes) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash proposal is unavailable for this target");
    } else {
        using Config               = typename V::DFlashConfig;
        if constexpr (Config::selector_top_k > 0) {
            // D4 (spec doc 06): the path selector needs a true top-16 over the full vocabulary,
            // so the subset optimized proposal head is rejected for DFlash 2 targets. This is a
            // host-side check on fixed engine state (no device sync); a violation throws at graph
            // capture and fails startup with the explicit message.
            if (state.execution.proposal_head != ProposalHead::Full) {
                throw std::logic_error(
                    "--spec dflash on this target requires the full output head (--lm-head-draft "
                    "is incompatible)");
            }
        }
        const std::int32_t width   = static_cast<std::int32_t>(k) + 1;
        const std::int32_t columns = width * batch_size;
        Tensor anchors             = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers           = frame.execution_frontiers.slice(0, 0, batch_size);
        Tensor valid_columns       = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor lanes               = frame.lanes.slice(0, 0, batch_size);
        Tensor full_rows           = frame.dflash_kv_table_rows.slice(0, 0, batch_size);
        Tensor ids                 = frame.proposal_ids.slice(1, 0, batch_size);
        Tensor positions           = frame.proposal_positions.slice(1, 0, batch_size);
        Tensor drafts              = frame.draft_tokens.slice(1, 0, batch_size);

        state.execution.work.reset();
        ops::prepare_masked_block(anchors, frontiers, valid_columns, Config::mask_token, ids,
                                  positions, state.execution.device.stream);
        Tensor residual = state.execution.work.alloc(DType::BF16, {Config::hidden, columns});
        ops::embedding(ids.view({columns}), state.execution.model.token_embedding, residual,
                       state.execution.device.stream);

        for (int layer = 0; layer < Config::layers; ++layer) {
            const auto& weight =
                state.execution.model.dflash->layers.at(static_cast<std::size_t>(layer));
            // DFlash 2 grouped dynamic conv roots (spec doc 06 decision D7): the per-layer roots
            // must outlive both sub-scopes below, so they are allocated under their own layer
            // scope and released at the layer boundary. v1 targets (conv_kernel_size == 0)
            // compile the allocation out and the scope stays a no-op marker.
            auto conv_layer_scope = state.execution.work.scope();
            workspace_recipe::DFlashConvLayerRoots conv_roots;
            if constexpr (Config::conv_kernel_size > 0) {
                conv_roots =
                    workspace_recipe::dflash_conv_layer<Config>(state.execution.work, columns);
            }
            {
                auto attention_scope = state.execution.work.scope();
                auto roots =
                    workspace_recipe::dflash_attention<Config>(state.execution.work, columns);
                ops::rmsnorm(residual, weight.input_norm, Config::rms_epsilon, false, roots.hidden,
                             state.execution.device.stream);
                Tensor attention_input = roots.hidden;
                if constexpr (Config::conv_kernel_size > 0) {
                    // DFlash 2 attention-side conv: dyn = linear(h0, attention_conv.
                    // kernel_projection) -> [2*conv_kernel_size*(hidden/conv_group_size),
                    // columns]; the conv runs over ALL block columns, the anchor at t=0
                    // included (zero-padded there). h1 = grouped_dynamic_causal_conv(h0,
                    // dyn, base_kernel, use=0): use selects the 640-row dynamic window and
                    // the base [2,2,5120] use half.
                    ops::linear(roots.hidden, weight.attention_conv.kernel_projection,
                                conv_roots.attention_dynamic, state.execution.device.stream);
                    attention_input = conv_roots.attention_out;
                    ops::grouped_dynamic_causal_conv(
                        roots.hidden, conv_roots.attention_dynamic,
                        weight.attention_conv.base_kernel, attention_input, 0, batch_size,
                        state.execution.device.stream);
                }
                Tensor query_raw =
                    roots.query_raw.view({Config::head_dim, Config::query_heads, columns});
                Tensor key_raw = roots.key_raw.view({Config::head_dim, Config::kv_heads, columns});
                Tensor value   = roots.value.view({Config::head_dim, Config::kv_heads, columns});
                Tensor query_flat = query_raw.view({Config::query_size, columns});
                Tensor key_flat   = key_raw.view({Config::kv_size, columns});
                Tensor value_flat = value.view({Config::kv_size, columns});
                ops::attn_input_proj(attention_input, weight.query_key_value, query_flat, key_flat,
                                     value_flat, state.execution.device.stream);
                Tensor query = roots.query.view({Config::head_dim, Config::query_heads, columns});
                Tensor key   = roots.key.view({Config::head_dim, Config::kv_heads, columns});
                ops::rmsnorm(query_raw, weight.query_norm, Config::rms_epsilon, false, query,
                             state.execution.device.stream);
                ops::rmsnorm(key_raw, weight.key_norm, Config::rms_epsilon, false, key,
                             state.execution.device.stream);
                ops::rope(positions.view({columns}), Config::head_dim, Config::rope_theta, query,
                          key, state.execution.device.stream);
                Tensor query_batch =
                    query.view({Config::head_dim, Config::query_heads, width, batch_size});
                Tensor key_batch =
                    key.view({Config::head_dim, Config::kv_heads, width, batch_size});
                Tensor value_batch =
                    value.view({Config::head_dim, Config::kv_heads, width, batch_size});
                Tensor attention_batch = roots.attention.view(
                    {Config::head_dim, Config::query_heads, width, batch_size});
                if (layer < Config::local_layers) {
                    ops::swa(query_batch, key_batch, value_batch, positions, valid_columns, lanes,
                             Config::attention_scale,
                             dflash_state(state).local_layer(static_cast<std::uint32_t>(layer)),
                             envelopes.local, state.execution.work, attention_batch,
                             state.execution.device.stream);
                } else {
                    ops::bidirectional_gqa_attention(
                        query_batch, key_batch, value_batch, frontiers, valid_columns, full_rows,
                        Config::attention_scale, dflash_state(state).full_batch_layer(0),
                        envelopes.full, state.execution.work, attention_batch,
                        state.execution.device.stream);
                }
                if constexpr (Config::conv_kernel_size > 0) {
                    // DFlash 2: the fused linear_add is unfused under conv_kernel_size > 0:
                    // attn_o = linear(A, attention_output); fin = grouped_dynamic_causal_conv(
                    // attn_o, dyn, base_kernel, use=1) into the shared conv-out buffer (h1 is
                    // fully consumed by the attention before fin is written); residual += fin.
                    Tensor attention_output =
                        state.execution.work.alloc(DType::BF16, {Config::hidden, columns});
                    ops::linear(roots.attention.view({Config::query_size, columns}),
                                weight.attention_output, attention_output,
                                state.execution.device.stream);
                    ops::grouped_dynamic_causal_conv(
                        attention_output, conv_roots.attention_dynamic,
                        weight.attention_conv.base_kernel, conv_roots.attention_out, 1,
                        batch_size, state.execution.device.stream);
                    ops::residual_add(conv_roots.attention_out, residual,
                                      state.execution.device.stream);
                } else {
                    ops::linear_add(roots.attention.view({Config::query_size, columns}),
                                    weight.attention_output, residual, state.execution.work,
                                    state.execution.device.stream);
                }
            }
            {
                auto mlp_scope = state.execution.work.scope();
                auto roots = workspace_recipe::dflash_mlp<Config>(state.execution.work, columns);
                ops::rmsnorm(residual, weight.post_attention_norm, Config::rms_epsilon, false,
                             roots.hidden, state.execution.device.stream);
                Tensor mlp_input = roots.hidden;
                if constexpr (Config::conv_kernel_size > 0) {
                    // DFlash 2 mlp-side conv: dyn2 = linear(h2, mlp_conv.kernel_projection);
                    // h3 = grouped_dynamic_causal_conv(h2, dyn2, mlp base, use=0).
                    ops::linear(roots.hidden, weight.mlp_conv.kernel_projection,
                                conv_roots.mlp_dynamic, state.execution.device.stream);
                    mlp_input = conv_roots.mlp_out;
                    ops::grouped_dynamic_causal_conv(
                        roots.hidden, conv_roots.mlp_dynamic,
                        weight.mlp_conv.base_kernel, mlp_input, 0, batch_size,
                        state.execution.device.stream);
                }
                if constexpr (Config::conv_kernel_size > 0) {
                    // DFlash 2: the fused W8 swiglu plan is tuned to the v1 35B shape
                    // {12288,6144,2048}; the 27B gate_up {34816,5120} runs unfused (W8 linear +
                    // silu_mul, the mtp_post_mixer pattern).
                    Tensor gate_up_out = state.execution.work.alloc(
                        DType::BF16, {2 * Config::intermediate, columns});
                    ops::linear(mlp_input, weight.gate_up, gate_up_out,
                                state.execution.device.stream);
                    ops::silu_mul(gate_up_out.slice(0, 0, Config::intermediate),
                                  gate_up_out.slice(0, Config::intermediate, Config::intermediate),
                                  roots.intermediate, state.execution.device.stream);
                } else {
                    ops::linear_swiglu(mlp_input, weight.gate_up, roots.intermediate,
                                       state.execution.work, state.execution.device.stream);
                }
                if constexpr (Config::conv_kernel_size > 0) {
                    // DFlash 2: mlp_o = linear(inter, down); fin2 = grouped_dynamic_causal_conv(
                    // mlp_o, dyn2, mlp base, use=1) into the shared conv-out buffer;
                    // residual += fin2.
                    Tensor mlp_output =
                        state.execution.work.alloc(DType::BF16, {Config::hidden, columns});
                    ops::linear(roots.intermediate, weight.down, mlp_output,
                                state.execution.device.stream);
                    ops::grouped_dynamic_causal_conv(
                        mlp_output, conv_roots.mlp_dynamic,
                        weight.mlp_conv.base_kernel, conv_roots.mlp_out, 1, batch_size,
                        state.execution.device.stream);
                    ops::residual_add(conv_roots.mlp_out, residual,
                                      state.execution.device.stream);
                } else {
                    ops::linear_add(roots.intermediate, weight.down, residual, state.execution.work,
                                    state.execution.device.stream);
                }
            }
        }

        Tensor packed = state.execution.work.alloc(
            DType::BF16, {Config::hidden, static_cast<std::int32_t>(k) * batch_size});
        const std::size_t element_bytes = dtype_size(DType::BF16);
        const std::size_t row_bytes =
            static_cast<std::size_t>(Config::hidden) * static_cast<std::size_t>(k) * element_bytes;
        const std::size_t source_pitch =
            static_cast<std::size_t>(Config::hidden) * width * element_bytes;
        const auto* source = static_cast<const std::byte*>(residual.data) +
                             static_cast<std::size_t>(Config::hidden) * element_bytes;
        CUDA_CHECK(cudaMemcpy2DAsync(packed.data, row_bytes, source, source_pitch, row_bytes,
                                     static_cast<std::size_t>(batch_size), cudaMemcpyDeviceToDevice,
                                     state.execution.device.stream));
        Tensor proposal_hidden = state.execution.work.alloc(
            DType::BF16, {Config::hidden, static_cast<std::int32_t>(k) * batch_size});
        ops::rmsnorm(packed, state.execution.model.dflash->final_norm, Config::rms_epsilon, false,
                     proposal_hidden, state.execution.device.stream);
        Tensor flat_drafts = drafts.view({static_cast<std::int32_t>(k) * batch_size});
        if (state.execution.proposal_head == ProposalHead::Full) {
            Tensor logits = state.execution.work.alloc(
                DType::BF16, {TextConfig::output_rows, static_cast<std::int32_t>(k) * batch_size});
            ops::linear(proposal_hidden, state.execution.model.output_head, logits,
                        state.execution.device.stream);
            if constexpr (Config::selector_top_k > 0) {
                // D8 (spec doc 06): the v1 argmax proposal is replaced by the path selector:
                // full-vocab top-16, the 256-rank hidden projection, the 16x16 edge-score tables
                // and the temperature-only walk. `logits` is exactly the dflash_selector_topk
                // input domain (draft columns only; the anchor column was never packed). The
                // walk reads the B SamplingConfig rows host-side at launch (graph-legal per the
                // op header); the ingress rows are the per-row sampling configs copied to the
                // frame ingress every round.
                auto sel = workspace_recipe::dflash_selector<Config>(
                    state.execution.work, static_cast<std::int32_t>(k), batch_size);
                ops::dflash_selector_topk(logits, sel.candidates, sel.unary,
                                          state.execution.device.stream);
                ops::linear(proposal_hidden,
                            state.execution.model.dflash->selector.hidden_projection,
                            sel.hidden_proj, state.execution.device.stream);
                ops::dflash_selector_scores(
                    sel.candidates, sel.hidden_proj, anchors, sel.unary,
                    state.execution.model.dflash->selector.predecessor_codebook,
                    state.execution.model.dflash->selector.successor_codebook, sel.scores,
                    state.execution.device.stream);
                ops::dflash_selector_walk(
                    sel.scores, sel.candidates,
                    static_cast<const ops::SamplingConfig*>(&state.host_ingress.sampling[0]),
                    flat_drafts, state.execution.device.stream);
            } else {
                ops::argmax(logits, flat_drafts, TextConfig::token_domain,
                            state.execution.device.stream);
            }
        } else {
            if (!state.execution.model.optimized_proposal.has_value()) {
                throw std::logic_error("optimized DFlash proposal head is unavailable");
            }
            const auto& proposal = *state.execution.model.optimized_proposal;
            Tensor logits        = state.execution.work.alloc(
                DType::BF16, {V::draft_head_rows, static_cast<std::int32_t>(k) * batch_size});
            ops::linear(proposal_hidden, proposal.head, logits, state.execution.device.stream);
            ops::argmax(logits, flat_drafts, V::draft_head_rows, state.execution.device.stream);
            ops::proposal_remap_token_ids(flat_drafts,
                                          static_cast<const std::int32_t*>(proposal.token_ids.data),
                                          V::draft_head_rows, state.execution.device.stream);
        }
        state.execution.work.reset();
    }
}

auto dflash_decode_batch_body(DFlashBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                              DFlashEnvelopes envelopes,
                              ops::GqaExecutionEnvelope target_envelope) {
    return [&state, batch_size, k, envelopes, target_envelope] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency) ||
            k == 0 || k > kDFlashDecodeMaximumDrafts) {
            throw std::logic_error("DFlash decode batch state is incomplete");
        }
        qwen3_6::DFlashDecodeState& frame = state.frame;
        const std::int32_t width          = static_cast<std::int32_t>(k) + 1;
        CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_6::DFlashDecodeIngress), cudaMemcpyHostToDevice,
                                   state.execution.device.stream));

        Tensor anchors          = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers        = frame.execution_frontiers.slice(0, 0, batch_size);
        Tensor context_starts   = frame.context_frontiers.slice(0, 0, batch_size);
        Tensor extents          = frame.proposal_extents.slice(0, 0, batch_size);
        Tensor valid_columns    = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor text_rows        = frame.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor dflash_rows      = frame.dflash_kv_table_rows.slice(0, 0, batch_size);
        Tensor lanes            = frame.lanes.slice(0, 0, batch_size);
        Tensor append_positions = frame.append_positions.slice(1, 0, batch_size);
        Tensor append_counts    = frame.append_counts.slice(0, 0, batch_size);
        Tensor drafts           = frame.draft_tokens.slice(1, 0, batch_size);
        Tensor verify_ids       = frame.verify_ids.slice(1, 0, batch_size);
        Tensor target_positions = frame.proposal_positions.slice(1, 0, batch_size);
        Tensor target_tokens    = frame.target_argmax.slice(1, 0, batch_size);
        Tensor target_logits    = frame.target_logits.slice(2, 0, batch_size);
        Tensor target_hidden    = frame.target_hidden.slice(2, 0, batch_size);
        Tensor selected_hidden  = frame.target_continuation_hidden.slice(1, 0, batch_size);
        Tensor licensed_tokens  = frame.licensed_tokens.slice(1, 0, batch_size);
        Tensor licensed_counts  = frame.licensed_counts.slice(0, 0, batch_size);
        Tensor accepted         = frame.accepted_drafts.slice(0, 0, batch_size);

        state.execution.work.reset();
        Tensor compact_features = state.execution.work.alloc(
            DType::BF16, {Variant::DFlashConfig::feature_rows, width, batch_size});
        ops::prepare_ragged_prefix(dflash_state(state).pending_features, lanes, context_starts,
                                   frontiers, compact_features, append_positions, append_counts,
                                   state.execution.device.stream);
        append_context_impl<Variant>(state, compact_features, append_positions, append_counts,
                                     lanes, dflash_rows, envelopes.append);

        // NINFER_DFLASH_CTXDUMP=1: write the drafter's context inputs for one round so a
        // PyTorch reference run of the same checkpoint can be compared stage by stage.
        // Engine acceptance is ~1.1 drafts/step against a published ~4.8 while every op
        // passes its own unit test, so the open question is whether the drafter's forward
        // reproduces the reference at all on identical inputs.
        {
            static const bool dump = [] {
                const char* value = std::getenv("NINFER_DFLASH_CTXDUMP");
                return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
            }();
            static bool written = false;
            if (dump && !written && batch_size == 1) {
                written = true;
                const std::int32_t rows = Variant::DFlashConfig::feature_rows;
                std::vector<std::uint16_t> host(static_cast<std::size_t>(rows) * width);
                CUDA_CHECK(cudaStreamSynchronize(state.execution.device.stream));
                CUDA_CHECK(cudaMemcpy(host.data(), compact_features.data,
                                      host.size() * sizeof(std::uint16_t),
                                      cudaMemcpyDeviceToHost));
                std::vector<std::int32_t> pos(static_cast<std::size_t>(width));
                CUDA_CHECK(cudaMemcpy(pos.data(), append_positions.data,
                                      pos.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost));
                std::vector<std::int32_t> counts(1);
                CUDA_CHECK(cudaMemcpy(counts.data(), append_counts.data, sizeof(std::int32_t),
                                      cudaMemcpyDeviceToHost));
                std::FILE* f = std::fopen("/mnt/f/dflash2-ctx.bin", "wb");
                if (f != nullptr) {
                    const std::int32_t header[4] = {rows, width, counts[0],
                                                    static_cast<std::int32_t>(k)};
                    std::fwrite(header, sizeof(std::int32_t), 4, f);
                    std::fwrite(pos.data(), sizeof(std::int32_t), pos.size(), f);
                    std::fwrite(host.data(), sizeof(std::uint16_t), host.size(), f);
                    std::fclose(f);
                    std::fprintf(stderr,
                                 "[dflash.ctxdump] rows=%d width=%d count=%d k=%d -> "
                                 "/mnt/f/dflash2-ctx.bin\n",
                                 rows, width, counts[0], static_cast<int>(k));
                }
            }
        }

        propose_batch_impl<Variant>(state, frame, batch_size, k, envelopes);
        ops::speculative_prepare_verify_ids(anchors, drafts, extents, verify_ids,
                                            state.execution.device.stream);

        TextContext card(state.execution.device, state.execution.model, state.execution.work, {},
                         state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk, 0, {},
                         &state.text_cache);
        DFlashFeatureSink sink =
            batch_feature_sink_impl<Variant>(state, lanes, valid_columns, width, batch_size);
        target_verify_accept(state.execution, state.continuation_hidden_store, card,
                             TargetVerifyFrameView{
                                 .ids             = verify_ids,
                                 .cache_positions = target_positions,
                                 .rope_positions  = target_positions,
                                 .valid_columns   = valid_columns,
                                 .kv_table_rows   = text_rows,
                                 .lanes           = lanes,
                                 .target_hidden   = target_hidden,
                                 .target_logits   = target_logits,
                                 .target_tokens   = target_tokens,
                                 .drafts          = drafts,
                                 .current_extents = extents,
                                 .frontiers       = frontiers,
                                 .anchors         = anchors,
                                 .licensed_tokens = licensed_tokens,
                                 .licensed_counts = licensed_counts,
                                 .accepted_drafts = accepted,
                                 .selected_hidden = selected_hidden,
                                 .replay_records  = state.execution.replay_records,
                                 .sampling        = frame.sampling,
                                 .feature_sink    = &sink,
                             },
                             target_envelope);
        CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, frame.egress.data,
                                   sizeof(qwen3_6::DFlashDecodeEgress), cudaMemcpyDeviceToHost,
                                   state.execution.device.stream));
    };
}

} // namespace

DFlashFeatureSink dflash_feature_sink(PrefillContext& state,
                                      DFlashFeatureSink::PrefillConsumer consume_prefill) {
    return prefill_feature_sink_impl<Variant>(state, std::move(consume_prefill));
}

void dflash_append_context(DFlashAppendContext& state, const Tensor& features,
                           const Tensor& positions, const Tensor& commit_counts,
                           const Tensor& lanes, const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    append_context_impl<Variant>(state, features, positions, commit_counts, lanes, table_rows,
                                 envelope);
}

void dflash_append_context(PrefillContext& state, const Tensor& features, const Tensor& positions,
                           const Tensor& commit_counts, const Tensor& lanes,
                           const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    append_context_impl<Variant>(state, features, positions, commit_counts, lanes, table_rows,
                                 envelope);
}

void capture_dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size,
                                 std::uint32_t k, DFlashEnvelopes envelopes,
                                 ops::GqaExecutionEnvelope target_envelope,
                                 DecodeGraphDefinition& definition) {
    auto body = dflash_decode_batch_body(state, batch_size, k, envelopes, target_envelope);
    capture_graph(state, definition, body);
}

void dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                         DFlashEnvelopes envelopes, ops::GqaExecutionEnvelope target_envelope,
                         DecodeGraphExecutable* executable) {
    auto body = dflash_decode_batch_body(state, batch_size, k, envelopes, target_envelope);
    run_prepared(state, executable, body);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
