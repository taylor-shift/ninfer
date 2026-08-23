#include "ninfer/engine.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

// DFlash 2 27B contract (spec doc 06, decisions D1/D4): block 8 = anchor + 7 draft tokens, so
// this engine exercises the 27B target's full draft window (kMaximumDFlashDraftTokens == 7,
// qwen3_6_27b config); the 35B template exercises 3 of its 15.
constexpr std::uint32_t kDraftTokens = 7;
// The 27B drafter is pure-SWA (local_layers == layers) with a 2048-token local cyclic window
// (DFlashConfig::local_capacity). The long-restore fixture must place its reuse boundary
// strictly beyond that window, mirroring the 35B test's 4096-window fixture (4099 > 4096).
constexpr std::uint32_t kDraftLocalWindow = 2048;
// D4: the path selector needs a true top-16 over the full 248320-vocabulary head, so the
// engine rejects dflash + the optimized proposal head with this exact message (dflash_impl.h
// throws it at graph capture, failing startup of the graph engine).
constexpr const char* kFullHeadError =
    "--spec dflash on this target requires the full output head (--lm-head-draft is incompatible)";

ninfer::EngineOptions ordinary_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path  = artifact;
    options.max_context    = 128;
    options.kv_capacity    = ninfer::KvCapacityPolicy::explicit_capacity(128);
    options.prefill_chunk  = 128;
    options.kv_cache       = ninfer::KvCacheStorage::BFloat16;
    options.use_cuda_graph = false;
    options.enable_vision  = false;
    return options;
}

ninfer::EngineOptions dflash_engine_options(const char* artifact, ninfer::ProposalHead proposal,
                                            std::uint32_t max_context) {
    ninfer::EngineOptions options     = ordinary_engine_options(artifact);
    options.max_context               = max_context;
    options.kv_capacity               = ninfer::KvCapacityPolicy::explicit_capacity(max_context);
    options.speculative.backend       = ninfer::SpeculativeBackend::DFlash;
    options.speculative.draft_tokens  = kDraftTokens;
    options.speculative.proposal_head = proposal;
    options.use_cuda_graph            = true;
    return options;
}

ninfer::RequestOptions greedy_options(std::uint32_t outputs, bool reuse) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

ninfer::PromptInput initial_conversation() {
    ninfer::PromptInput input;
    input.options.enable_thinking = false;

    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = "Name one prime number.", .media = {}});
    input.messages.push_back(std::move(user));
    return input;
}

ninfer::PromptInput followup_conversation(const ninfer::GenerationResult& first,
                                          std::string followup) {
    ninfer::PromptInput input = initial_conversation();
    ninfer::ChatMessage assistant;
    assistant.role              = ninfer::ChatRole::Assistant;
    assistant.reasoning_content = first.reasoning;
    assistant.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = first.content, .media = {}});
    input.messages.push_back(std::move(assistant));

    ninfer::ChatMessage next;
    next.role = ninfer::ChatRole::User;
    next.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = std::move(followup), .media = {}});
    input.messages.push_back(std::move(next));
    return input;
}

ninfer::PromptInput altered_history_after_boundary() {
    ninfer::PromptInput input = initial_conversation();
    ninfer::ChatMessage assistant;
    assistant.role = ninfer::ChatRole::Assistant;
    assistant.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = "This history was changed.", .media = {}});
    input.messages.push_back(std::move(assistant));

    ninfer::ChatMessage next;
    next.role = ninfer::ChatRole::User;
    next.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = "Continue briefly.", .media = {}});
    input.messages.push_back(std::move(next));
    return input;
}

int verify_dflash_load(const ninfer::Engine& engine, std::uint32_t max_context) {
    // The qwen3.8-27b fleet NVFP4 artifact (qwen3_8_27b_nvfp4.ninfer) resolves to the 27B
    // target with the nvfp4 weights profile (registry.cpp maps both qwen3.6/qwen3.8 27B model
    // ids to target_key "qwen3_6_27b").
    const ninfer::LoadSummary load = engine.load_summary();
    if (load.target != "qwen3_6_27b" || load.weights_id != "nvfp4" ||
        load.host_to_device_bytes == 0 || load.artifact_bytes_read < load.host_to_device_bytes) {
        std::cerr << "DFlash Engine materialized an invalid artifact payload: target="
                  << load.target << " weights=" << load.weights_id << '\n';
        return 1;
    }
    const ninfer::MemorySummary memory = engine.memory_summary();
    if (memory.max_context != max_context || memory.kv_cache != ninfer::KvCacheStorage::BFloat16 ||
        memory.kv_payload_bytes == 0 || memory.weights.capacity_bytes == 0 ||
        memory.weights.used_bytes == 0 ||
        memory.weights.used_bytes > memory.weights.capacity_bytes ||
        memory.sequence.capacity_bytes == 0 || memory.sequence.used_bytes == 0 ||
        memory.sequence.used_bytes > memory.sequence.capacity_bytes ||
        memory.workspace.capacity_bytes == 0 || memory.request_transient.capacity_bytes != 0 ||
        memory.workspace_logical_peak_bytes != 0 || memory.cuda_graph_allowance_bytes == 0) {
        std::cerr << "DFlash Engine has an invalid frozen memory layout\n";
        return 1;
    }
    return 0;
}

int exercise_partial_terminal(ninfer::Engine& engine, const std::vector<ninfer::TokenId>& prompt,
                              const std::vector<ninfer::TokenId>& baseline) {
    for (std::size_t stop_index = 1; stop_index < baseline.size(); ++stop_index) {
        const ninfer::TokenId stop = baseline[stop_index];
        if (std::find(baseline.begin(), baseline.begin() + static_cast<std::ptrdiff_t>(stop_index),
                      stop) != baseline.begin() + static_cast<std::ptrdiff_t>(stop_index)) {
            continue;
        }
        ninfer::RequestOptions options = greedy_options(24, false);
        options.stop.token_ids.push_back(stop);
        const ninfer::GenerationResult stopped =
            engine.generate(engine.prepare_tokens(prompt), options);
        if (stopped.finish_reason != ninfer::FinishReason::StopToken) { continue; }
        if (stopped.generated_token_ids.size() != stop_index + 1 ||
            !std::equal(stopped.generated_token_ids.begin(), stopped.generated_token_ids.end(),
                        baseline.begin())) {
            std::cerr << "partial DFlash terminal diverged from the target baseline prefix\n";
            return 1;
        }

        const std::uint64_t fully_licensed = 1 + stopped.speculative.rounds +
                                             stopped.speculative.accepted_tokens +
                                             stopped.speculative.fallback_steps;
        if (stopped.generated_token_ids.size() >= fully_licensed) { continue; }

        std::vector<ninfer::TokenId> continuation = prompt;
        continuation.insert(continuation.end(), stopped.generated_token_ids.begin(),
                            stopped.generated_token_ids.end());
        continuation.push_back(198);
        const ninfer::GenerationResult reused = engine.generate(
            engine.prepare_tokens(std::move(continuation)), greedy_options(2, true));
        const std::uint32_t expected_reuse =
            static_cast<std::uint32_t>(prompt.size() + stopped.generated_token_ids.size() - 1);
        if (reused.reused_prompt_tokens != expected_reuse ||
            reused.generated_token_ids.size() != 2) {
            std::cerr << "partial DFlash terminal did not publish its exact context frontier: "
                      << "reused=" << reused.reused_prompt_tokens << " expected=" << expected_reuse
                      << '\n';
            return 1;
        }
        return 0;
    }
    std::cerr << "fixed DFlash fixture exposed no terminal stop inside a licensed batch\n";
    return 1;
}

int exercise_boundary_restore(ninfer::Engine& engine) {
    const ninfer::GenerationResult first =
        engine.generate(engine.prepare(initial_conversation()), greedy_options(2, false));
    if (first.generated_token_ids.size() != 2) {
        std::cerr << "DFlash boundary fixture did not establish resident state\n";
        return 1;
    }

    const ninfer::GenerationResult restored =
        engine.generate(engine.prepare(followup_conversation(first, "Answer with one digit.")),
                        greedy_options(4, true));
    if (restored.reused_prompt_tokens == 0 || restored.generated_token_ids.size() != 4) {
        std::cerr << "DFlash assistant-boundary restore did not reuse its cache snapshot: reused="
                  << restored.reused_prompt_tokens
                  << " outputs=" << restored.generated_token_ids.size() << '\n';
        return 1;
    }
    const ninfer::GenerationResult baseline =
        engine.generate(engine.prepare(followup_conversation(first, "Answer with one digit.")),
                        greedy_options(4, false));
    if (baseline.generated_token_ids != restored.generated_token_ids) {
        std::cerr << "DFlash boundary restore changed greedy target output\n";
        return 1;
    }
    return 0;
}

int exercise_rerun_determinism(ninfer::Engine& engine,
                               const std::vector<ninfer::TokenId>& prompt) {
    // Spec doc 06 section 9 port: the identical greedy request run twice must produce
    // bit-identical output. Greedy rows draw no selector randomness (D8), so the rerun is
    // a pure determinism check on the speculative path.
    const ninfer::GenerationResult first =
        engine.generate(engine.prepare_tokens(prompt), greedy_options(16, false));
    if (first.generated_token_ids.size() != 16) {
        std::cerr << "DFlash determinism fixture did not generate sixteen tokens\n";
        return 1;
    }
    const ninfer::GenerationResult second =
        engine.generate(engine.prepare_tokens(prompt), greedy_options(16, false));
    if (second.generated_token_ids.size() != 16 ||
        first.generated_token_ids != second.generated_token_ids) {
        std::cerr << "DFlash rerun of the identical greedy request diverged\n";
        return 1;
    }
    return 0;
}

int exercise_long_boundary_restore(ninfer::Engine& engine) {
    // Two more generated tokens than the drafter's 2048 local window: the reuse boundary of
    // the altered history sits 2049 tokens behind the resident frontier (2049 > 2048), so the
    // restore must reach strictly beyond one cyclic window — the 27B analog of the 35B
    // fixture's 4099 > 4096.
    constexpr std::uint32_t generated_tokens = kDraftLocalWindow + 2;
    const ninfer::GenerationResult long_run = engine.generate(
        engine.prepare(initial_conversation()), greedy_options(generated_tokens, false));
    if (long_run.generated_token_ids.size() != generated_tokens) {
        std::cerr << "DFlash long-restore fixture did not cross the cyclic-cache window\n";
        return 1;
    }
    const std::uint32_t resident_frontier = long_run.prompt.prompt_tokens + generated_tokens - 1;
    const ninfer::GenerationResult restored =
        engine.generate(engine.prepare(altered_history_after_boundary()), greedy_options(2, true));
    if (restored.reused_prompt_tokens == 0 ||
        resident_frontier - restored.reused_prompt_tokens <= kDraftLocalWindow ||
        restored.generated_token_ids.size() != 2) {
        std::cerr << "DFlash long-distance restore did not use the saved cyclic cache: resident="
                  << resident_frontier << " restored=" << restored.reused_prompt_tokens << '\n';
        return 1;
    }
    const ninfer::GenerationResult baseline =
        engine.generate(engine.prepare(altered_history_after_boundary()), greedy_options(2, false));
    if (baseline.generated_token_ids != restored.generated_token_ids) {
        std::cerr << "DFlash long-distance boundary restore changed greedy target output\n";
        return 1;
    }
    return 0;
}

int exercise_full_head_rejection(const std::string& artifact) {
    // D4 negative case: dflash + the optimized proposal head is illegal on this target and
    // must fail with the exact engine message. With use_cuda_graph the check fires at
    // startup (graph capture); if a future change defers it, the first proposal must throw.
    try {
        ninfer::Engine engine(
            dflash_engine_options(artifact.c_str(), ninfer::ProposalHead::Optimized, 128));
        // If startup validation did not reject the combination, the first proposal must.
        const std::vector<ninfer::TokenId> fixture{198, 198};
        (void)engine.generate(engine.prepare_tokens(fixture), greedy_options(1, false));
    } catch (const std::exception& error) {
        if (std::string(error.what()) == kFullHeadError) { return 0; }
        std::cerr << "DFlash + optimized proposal head failed with the wrong message: "
                  << error.what() << '\n';
        return 1;
    }
    std::cerr << "engine accepted dflash with the optimized proposal head\n";
    return 1;
}

} // namespace

int main() {
    // Artifact contract: NINFER_QWEN3_8_27B_WEIGHTS, falling back to the in-tree NVFP4 fleet
    // artifact (NINFER_SOURCE_DIR is provided by the CMake wiring).
    const char* environment = std::getenv("NINFER_QWEN3_8_27B_WEIGHTS");
    const std::string artifact =
        (environment != nullptr && *environment != '\0')
            ? std::string(environment)
            : (std::filesystem::path(NINFER_SOURCE_DIR) / "out" / "qwen3_8_27b_nvfp4.ninfer")
                  .string();
    if (!std::filesystem::is_regular_file(artifact)) {
        std::cout << "skip: " << artifact << " is not available\n";
        return 77;
    }

    // Same raw-token fixture as the 35B template: both targets share the qwen3.8 tokenizer
    // domain, so the IDs are valid raw input for the 27B target as well.
    const std::vector<ninfer::TokenId> prompt{
        248045, 846,    198, 109266, 3709,  96220, 117443, 97913,
        1710,   248046, 198, 248045, 74455, 198,   248068, 198,
    };
    std::vector<ninfer::TokenId> target_output;
    {
        ninfer::Engine ordinary(ordinary_engine_options(artifact.c_str()));
        target_output =
            ordinary.generate(ordinary.prepare_tokens(prompt), greedy_options(24, false))
                .generated_token_ids;
        if (target_output.size() != 24) {
            std::cerr << "ordinary target baseline did not generate 24 tokens\n";
            return 1;
        }
    }

    {
        // Concurrent DFlash Graph route. The 27B target admits only the full output head
        // under dflash (D4), so the concurrent fixture exercises the full-head engine.
        ninfer::EngineOptions options =
            dflash_engine_options(artifact.c_str(), ninfer::ProposalHead::Full, 128);
        options.max_concurrency = 2;
        ninfer::Engine full(std::move(options));
        auto first  = full.submit(full.prepare_tokens(prompt), greedy_options(17, false));
        auto second = full.submit(full.prepare_tokens(prompt), greedy_options(9, false));
        const ninfer::GenerationResult first_result  = first.wait();
        const ninfer::GenerationResult second_result = second.wait();
        const auto valid = [&](const ninfer::GenerationResult& result, std::size_t count) {
            return result.generated_token_ids.size() == count &&
                   std::equal(result.generated_token_ids.begin(), result.generated_token_ids.end(),
                              target_output.begin(),
                              target_output.begin() + static_cast<std::ptrdiff_t>(count)) &&
                   result.speculative.backend == ninfer::SpeculativeBackend::DFlash &&
                   result.speculative.rounds != 0;
        };
        if (!valid(first_result, 17) || !valid(second_result, 9)) {
            std::cerr << "concurrent full-head DFlash Graph route diverged from ordinary target "
                         "output\n";
            return 1;
        }
    }

    if (const int result = exercise_full_head_rejection(artifact); result != 0) { return result; }

    ninfer::Engine engine(dflash_engine_options(artifact.c_str(), ninfer::ProposalHead::Full, 4352));
    if (const int result = verify_dflash_load(engine, 4352); result != 0) { return result; }
    engine.reset_memory_peaks();
    const ninfer::GenerationResult dflash =
        engine.generate(engine.prepare_tokens(prompt), greedy_options(24, false));
    if (dflash.generated_token_ids != target_output) {
        const auto mismatch =
            std::mismatch(dflash.generated_token_ids.begin(), dflash.generated_token_ids.end(),
                          target_output.begin(), target_output.end());
        std::cerr << "DFlash Graph route diverged from ordinary greedy target output at "
                  << static_cast<std::size_t>(mismatch.first - dflash.generated_token_ids.begin())
                  << ": dflash="
                  << (mismatch.first == dflash.generated_token_ids.end() ? -1 : *mismatch.first)
                  << " target=" << (mismatch.second == target_output.end() ? -1 : *mismatch.second)
                  << '\n';
        return 1;
    }
    const ninfer::MemorySummary memory = engine.memory_summary();
    if (memory.workspace_logical_peak_bytes == 0 ||
        memory.workspace_logical_peak_bytes > memory.workspace.capacity_bytes) {
        std::cerr << "DFlash request did not report a valid planned workspace phase\n";
        return 1;
    }
    if (dflash.speculative.backend != ninfer::SpeculativeBackend::DFlash ||
        dflash.speculative.rounds == 0) {
        std::cerr << "DFlash fixture did not execute speculative decode\n";
        return 1;
    }
    if (const int result = exercise_partial_terminal(engine, prompt, target_output); result != 0) {
        return result;
    }
    if (const int result = exercise_boundary_restore(engine); result != 0) { return result; }
    if (const int result = exercise_rerun_determinism(engine, prompt); result != 0) {
        return result;
    }
    if (const int result = exercise_long_boundary_restore(engine); result != 0) { return result; }

    std::cout << "ok\n";
    return 0;
}