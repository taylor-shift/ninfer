// dflash2 repro v2: replicate the exact run_scores() device sequence through
// the PUBLIC wrapper (ops::dflash_selector_scores) with GuardedDeviceBuffer-
// style allocations (cudaMalloc(payload + 2*256), payload at +256B), the test's
// allocation/copy order, and the test executable's exact link line
// (libninfer_ops.a + libninfer_nvfp4_tma.a + libninfer_core.a + dynamic AND
// static cudart + stubs/libcuda.so).
//
// If this IMCs where the direct-kernel repro (dflash2_minimal_repro.cu) passed,
// the trigger is the wrapper/library path or the guarded-buffer sequence —
// not the kernel code (proven clean) and not the big-copy path (P1 passed).
#include "ninfer/ops/dflash_selector.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace ninfer;

#define CK(x)                                                                     \
    do {                                                                          \
        cudaError_t e_ = (x);                                                     \
        if (e_ != cudaSuccess) {                                                  \
            printf("CUDA ERROR %d: %s\n", __LINE__, cudaGetErrorString(e_));      \
            return 1;                                                             \
        }                                                                         \
    } while (0)

constexpr std::int32_t kVocab = 248320;
constexpr std::int32_t kTopK  = 16;
constexpr std::int32_t kRank  = 256;

std::uint16_t f32_to_bf16(float f) {
    std::uint32_t u;
    __builtin_memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return std::uint16_t((u >> 16) | 0x0040u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return std::uint16_t(u >> 16);
}
std::uint64_t lcg_next(std::uint64_t& state) {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return state;
}
float lcg_value(std::uint64_t& state, float lo, float span) {
    return lo + static_cast<float>(lcg_next(state) >> 40) * (span / 16777216.0f);
}

// GuardedDeviceBuffer layout: cudaMalloc(payload + 2*256), payload at +256.
struct Guarded {
    void* base      = nullptr;
    std::size_t payload = 0;
    void* data() { return static_cast<std::uint8_t*>(base) + 256; }
    const void* data() const { return static_cast<std::uint8_t*>(base) + 256; }
};

int main() {
    const std::int32_t k = 2, B = 2;
    const std::int32_t cols = k * B;

    // ---- host fixtures (exact test values where they matter) ----
    std::vector<std::int32_t> candidates(kTopK * cols);
    const std::int32_t bases[cols] = {1000, 5000, 9000, 13000};
    for (int col = 0; col < cols; ++col)
        for (int rank = 0; rank < kTopK; ++rank)
            candidates[col * kTopK + rank] = bases[col] + rank * 997;
    std::vector<std::uint16_t> hidden(static_cast<std::size_t>(kRank) * cols);
    std::uint64_t s1 = 0x501;
    for (auto& x : hidden) x = f32_to_bf16(lcg_value(s1, -0.5f, 1.0f));
    const std::vector<std::int32_t> anchors{123457, 200123};
    std::vector<float> unary(kTopK * cols);
    std::uint64_t s2 = 0x502;
    for (auto& x : unary) x = lcg_value(s2, -8.0f, 16.0f);
    const std::size_t cb = static_cast<std::size_t>(kVocab) * kRank;
    std::vector<std::uint16_t> pred(cb);
    std::uint64_t s3 = 0x123456789ABCDEF0ull;
    for (auto& x : pred) x = f32_to_bf16(lcg_value(s3, -0.5f, 1.0f));
    std::vector<std::uint16_t> succ(cb);
    std::uint64_t s4 = 0xFEDCBA0987654321ull;
    for (auto& x : succ) x = f32_to_bf16(lcg_value(s4, -0.5f, 1.0f));

    // ---- guarded allocations in the exact test order (run_scores 282-288) ----
    Guarded gc, gh, ga, gu, gp, gs, go;
    gc.payload = candidates.size() * 4;  CK(cudaMalloc(&gc.base, gc.payload + 512));
    gh.payload = hidden.size() * 2;      CK(cudaMalloc(&gh.base, gh.payload + 512));
    ga.payload = anchors.size() * 4;     CK(cudaMalloc(&ga.base, ga.payload + 512));
    gu.payload = unary.size() * 4;       CK(cudaMalloc(&gu.base, gu.payload + 512));
    gp.payload = pred.size() * 2;        CK(cudaMalloc(&gp.base, gp.payload + 512));
    gs.payload = succ.size() * 2;        CK(cudaMalloc(&gs.base, gs.payload + 512));
    go.payload = static_cast<std::size_t>(k) * kTopK * kTopK * B * 4;
    CK(cudaMalloc(&go.base, go.payload + 512));
    printf("allocations ok (codebook %.1f MB each)\n", gp.payload / 1048576.0);

    // ---- copies in the exact test order (289-295) ----
    CK(cudaMemcpy(gc.data(), candidates.data(), gc.payload, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(gh.data(), hidden.data(), gh.payload, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ga.data(), anchors.data(), ga.payload, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(gu.data(), unary.data(), gu.payload, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(gp.data(), pred.data(), gp.payload, cudaMemcpyHostToDevice));
    printf("4 small copies + 127 MB pred copy ok\n");
    CK(cudaMemcpy(gs.data(), succ.data(), gs.payload, cudaMemcpyHostToDevice));
    CK(cudaMemset(go.data(), 0xcd, go.payload));
    printf("127 MB succ copy + memset ok\n");

    // ---- exact test Tensor shapes (297-304) + public wrapper (305) ----
    Tensor candidates_t(gc.data(), DType::I32, {kTopK, cols});
    Tensor hidden_t(gh.data(), DType::BF16, {kRank, cols});
    Tensor anchors_t(ga.data(), DType::I32, {B});
    Tensor unary_t(gu.data(), DType::FP32, {kTopK, cols});
    Tensor pred_t(gp.data(), DType::BF16, {kVocab, kRank});
    Tensor succ_t(gs.data(), DType::BF16, {kVocab, kRank});
    Tensor scores_t(go.data(), DType::FP32, {k, kTopK, kTopK, B});
    ops::dflash_selector_scores(candidates_t, hidden_t, anchors_t, unary_t, pred_t, succ_t,
                                scores_t, nullptr);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    float probe = 0.0f;
    CK(cudaMemcpy(&probe, go.data(), 4, cudaMemcpyDeviceToHost));
    printf("PASS (scores[0]=%.6f)\n", probe);
    return 0;
}