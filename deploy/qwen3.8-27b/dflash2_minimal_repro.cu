// dflash2 minimal repro ladder: isolate the illegal-memory-access trigger of
// dflash_selector_scores on the winbox 5090 (WSL, CUDA 13.1, sm_120a).
//
// Phases (each prints PASS/FAIL + the exact cudaError, then a sticky-error
// check so one bad phase cannot poison the interpretation of the next):
//   P1  2 x 127 MB cudaMalloc + 127 MB H2D cudaMemcpy + sync
//       (isolates the big-copy / WSL BAR1 path from the kernel itself)
//   P2  the real dflash_selector_scores_kernel with the exact test shape
//       (k=2, B=2, full 248320x256 codebooks) — expected to reproduce the IMC
//   P3  same kernel, B=1 (grid 1 x 2) — smaller (b,k) domain
//   P4  stripped kernel: identical grid/block/SHARED/codebook-load pattern,
//       no contraction arithmetic, store a constant — isolates the access
//       pattern from the FP math
//   P5  stripped kernel without the 34 KB shared memory (codebook rows read
//       straight from global per iteration) — isolates the shared-memory size
//   P6  real kernel source with the #pragma unroll removed (runtime loop) —
//       isolates the 256x-unrolled dynamic-index codegen
//
// Build (inside the ninfer-wsl container, needs the repo sources):
//   nvcc -O3 -std=c++20 -arch=sm_120a -lineinfo -I/src/include -I/src/src \
//     dflash2_minimal_repro.cu -o /build/dflash2_minimal_repro
// Run (WSL side, GPU free):
//   LD_LIBRARY_PATH=$REPO/.cuda13-libs $BUILD/dflash2_minimal_repro

#include "ops/kernel/dflash_selector.cuh" // the real kernels

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>

#define CK(x)                                                                     \
    do {                                                                          \
        cudaError_t e_ = (x);                                                     \
        if (e_ != cudaSuccess) {                                                  \
            printf("CUDA ERROR %s:%d: %s\n", __FILE__, __LINE__,                  \
                   cudaGetErrorString(e_));                                       \
            return 1;                                                             \
        }                                                                         \
    } while (0)

// Returns the sticky error if the device context is poisoned (a fault from an
// earlier phase), so each phase reports its own truth.
int sticky() {
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        printf("  (sticky error from earlier phase: %s)\n", cudaGetErrorString(e));
        return 1;
    }
    return 0;
}

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

// The stripped variants (P4/P5/P6) live here, mirroring the real kernel's
// access pattern with the marked piece removed.
__launch_bounds__(256) __global__
void stripped_scores_kernel(const std::int32_t* candidates, const __nv_bfloat16* hidden_proj,
                            const std::int32_t* anchors, const float* unary,
                            const __nv_bfloat16* pred_codebook,
                            const __nv_bfloat16* succ_codebook, float* scores, std::int32_t k,
                            int variant) {
    const int b   = static_cast<int>(blockIdx.x);
    const int t   = static_cast<int>(blockIdx.y);
    const int col = b * k + t;
    const int tid = static_cast<int>(threadIdx.x);

    if (variant == 4) {
        // P4: identical shared pattern, no contraction.
        constexpr int kRowPad = kRank + 2;
        __shared__ float a_sh[kTopK][kRowPad];
        __shared__ float b_sh[kTopK][kRowPad];
        __shared__ float h_sh[kRank];
        h_sh[tid] = __bfloat162float(hidden_proj[static_cast<std::int64_t>(col) * kRank + tid]);
        for (int p = 0; p < kTopK; ++p) {
            const std::int32_t pred_id =
                (t == 0) ? anchors[b] : candidates[static_cast<std::int64_t>(col - 1) * kTopK + p];
            a_sh[p][tid] = __bfloat162float(
                pred_codebook[static_cast<std::int64_t>(pred_id) +
                              static_cast<std::int64_t>(tid) * kVocab]);
        }
        for (int c = 0; c < kTopK; ++c) {
            const std::int32_t cand_id = candidates[static_cast<std::int64_t>(col) * kTopK + c];
            b_sh[c][tid] = __bfloat162float(
                succ_codebook[static_cast<std::int64_t>(cand_id) +
                              static_cast<std::int64_t>(tid) * kVocab]);
        }
        __syncthreads();
        const int p = tid & 15;
        const int c = tid >> 4;
        scores[static_cast<std::int64_t>(t) + static_cast<std::int64_t>(p) * k +
               static_cast<std::int64_t>(c) * k * kTopK +
               static_cast<std::int64_t>(b) * k * kTopK * kTopK] =
            unary[static_cast<std::int64_t>(col) * kTopK + c];
    } else {
        // P5: no shared memory at all — every read straight from global.
        const std::int32_t pred_id =
            (t == 0) ? anchors[b] : candidates[static_cast<std::int64_t>(col - 1) * kTopK + (tid & 15)];
        const std::int32_t cand_id = candidates[static_cast<std::int64_t>(col) * kTopK + (tid >> 4)];
        const int p = tid & 15;
        const int c = tid >> 4;
        float acc = 0.0f;
        for (int d = 0; d < kRank; ++d) { // each thread: full d, no unroll pragma
            const float a = __bfloat162float(
                pred_codebook[static_cast<std::int64_t>(pred_id) +
                              static_cast<std::int64_t>(d) * kVocab]);
            const float h =
                __bfloat162float(hidden_proj[static_cast<std::int64_t>(col) * kRank + d]);
            const float s = __bfloat162float(
                succ_codebook[static_cast<std::int64_t>(cand_id) +
                              static_cast<std::int64_t>(d) * kVocab]);
            acc += a * h * s;
        }
        scores[static_cast<std::int64_t>(t) + static_cast<std::int64_t>(p) * k +
               static_cast<std::int64_t>(c) * k * kTopK +
               static_cast<std::int64_t>(b) * k * kTopK * kTopK] = acc;
    }
}

// P6: the real contraction with a RUNTIME loop (no #pragma unroll).
__launch_bounds__(256) __global__
void real_nounroll_kernel(const std::int32_t* candidates, const __nv_bfloat16* hidden_proj,
                          const std::int32_t* anchors, const float* unary,
                          const __nv_bfloat16* pred_codebook, const __nv_bfloat16* succ_codebook,
                          float* scores, std::int32_t k) {
    constexpr int kRowPad = kRank + 2;
    const int b   = static_cast<int>(blockIdx.x);
    const int t   = static_cast<int>(blockIdx.y);
    const int col = b * k + t;
    const int tid = static_cast<int>(threadIdx.x);
    __shared__ float a_sh[kTopK][kRowPad];
    __shared__ float b_sh[kTopK][kRowPad];
    __shared__ float h_sh[kRank];
    h_sh[tid] = __bfloat162float(hidden_proj[static_cast<std::int64_t>(col) * kRank + tid]);
    for (int p = 0; p < kTopK; ++p) {
        const std::int32_t pred_id =
            (t == 0) ? anchors[b] : candidates[static_cast<std::int64_t>(col - 1) * kTopK + p];
        a_sh[p][tid] = __bfloat162float(
            pred_codebook[static_cast<std::int64_t>(pred_id) +
                          static_cast<std::int64_t>(tid) * kVocab]);
    }
    for (int c = 0; c < kTopK; ++c) {
        const std::int32_t cand_id = candidates[static_cast<std::int64_t>(col) * kTopK + c];
        b_sh[c][tid] = __bfloat162float(
            succ_codebook[static_cast<std::int64_t>(cand_id) +
                          static_cast<std::int64_t>(tid) * kVocab]);
    }
    __syncthreads();
    const int p = tid & 15;
    const int c = tid >> 4;
    float acc = 0.0f;
    for (int d = 0; d < kRank; ++d) { // NOTE: no #pragma unroll
        acc += a_sh[p][d] * h_sh[d] * b_sh[c][d];
    }
    const float unary_c = unary[static_cast<std::int64_t>(col) * kTopK + c];
    scores[static_cast<std::int64_t>(t) + static_cast<std::int64_t>(p) * k +
           static_cast<std::int64_t>(c) * k * kTopK +
           static_cast<std::int64_t>(b) * k * kTopK * kTopK] = unary_c + acc;
}

struct DeviceBuf {
    void* p     = nullptr;
    std::size_t n = 0;
    static bool failed;
    explicit DeviceBuf(std::size_t bytes) : n(bytes) {
        cudaError_t e = cudaMalloc(&p, bytes);
        if (e != cudaSuccess) {
            printf("CUDA ERROR: cudaMalloc: %s\n", cudaGetErrorString(e));
            failed = true;
        }
    }
    ~DeviceBuf() { if (p) cudaFree(p); }
    DeviceBuf(const DeviceBuf&) = delete;
    DeviceBuf& operator=(const DeviceBuf&) = delete;
};
bool DeviceBuf::failed = false;
#define CHECK_ALLOC()                                                        \
    do {                                                                     \
        if (DeviceBuf::failed) { rc = 1; goto done; }                        \
    } while (0)

// Build the exact run_scores() host fixtures for (k, b_count).
struct Fixture {
    std::vector<std::int32_t> candidates;
    std::vector<std::uint16_t> hidden;
    std::vector<std::int32_t> anchors;
    std::vector<float> unary;
    std::vector<std::uint16_t> pred, succ;
    std::size_t scores_elems = 0;
};

Fixture make_fixture(std::int32_t k, std::int32_t b_count) {
    Fixture f;
    const std::int32_t cols = k * b_count;
    f.candidates.resize(kTopK * cols);
    const std::int32_t bases[8] = {1000, 5000, 9000, 13000, 17000, 21000, 25000, 29000};
    for (int col = 0; col < cols; ++col)
        for (int rank = 0; rank < kTopK; ++rank)
            f.candidates[col * kTopK + rank] = bases[col % 8] + rank * 997;
    f.hidden.resize(kRank * cols);
    std::uint64_t s1 = 0x501;
    for (auto& x : f.hidden) x = f32_to_bf16(lcg_value(s1, -0.5f, 1.0f));
    f.anchors = {123457, 200123};
    f.unary.resize(kTopK * cols);
    std::uint64_t s2 = 0x502;
    for (auto& x : f.unary) x = lcg_value(s2, -8.0f, 16.0f);
    std::size_t cb = static_cast<std::size_t>(kVocab) * kRank;
    f.pred.resize(cb);
    std::uint64_t s3 = 0x123456789ABCDEF0ull;
    for (auto& x : f.pred) x = f32_to_bf16(lcg_value(s3, -0.5f, 1.0f));
    f.succ.resize(cb);
    std::uint64_t s4 = 0xFEDCBA0987654321ull;
    for (auto& x : f.succ) x = f32_to_bf16(lcg_value(s4, -0.5f, 1.0f));
    f.scores_elems = static_cast<std::size_t>(k) * kTopK * kTopK * b_count;
    return f;
}

int main() {
    int rc = 0;
    printf("=== dflash2 minimal repro ladder (5090/WSL/CUDA 13.1) ===\n");

    // ---------------- P1: big copy ----------------
    {
        printf("P1: 2x 127MB malloc + 127MB H2D copy + sync ... ");
        std::size_t bytes = static_cast<std::size_t>(kVocab) * kRank * 2; // 127 MB
        DeviceBuf a(bytes), b(bytes);
        CHECK_ALLOC();
        std::vector<std::uint16_t> h(bytes / 2, 0x3f80);
        CK(cudaMemcpy(a.p, h.data(), bytes, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(b.p, h.data(), bytes, cudaMemcpyHostToDevice));
        CK(cudaDeviceSynchronize());
        printf("PASS\n");
    }

    if (sticky()) { rc = 1; goto done; }

    // ---------------- P2: real kernel, k=2 B=2 ----------------
    {
        const std::int32_t k = 2, B = 2;
        printf("P2: real scores kernel k=2 B=2 (exact test shape) ... ");
        const Fixture f = make_fixture(k, B);
        DeviceBuf dc(f.candidates.size() * 4), dh(f.hidden.size() * 2),
            da(f.anchors.size() * 4), du(f.unary.size() * 4), dp(f.pred.size() * 2),
            ds(f.succ.size() * 2), dof(f.scores_elems * 4);
        CHECK_ALLOC();
        CK(cudaMemcpy(dc.p, f.candidates.data(), dc.n, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dh.p, f.hidden.data(), dh.n, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(da.p, f.anchors.data(), da.n, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(du.p, f.unary.data(), du.n, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dp.p, f.pred.data(), dp.n, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(ds.p, f.succ.data(), ds.n, cudaMemcpyHostToDevice));
        CK(cudaMemset(dof.p, 0xcd, dof.n));
        dim3 grid(B, k);
        ninfer::ops::dflash_selector_scores_kernel<<<grid, 256>>>(
            static_cast<const std::int32_t*>(dc.p),
            static_cast<const __nv_bfloat16*>(dh.p),
            static_cast<const std::int32_t*>(da.p),
            static_cast<const float*>(du.p),
            static_cast<const __nv_bfloat16*>(dp.p),
            static_cast<const __nv_bfloat16*>(ds.p),
            static_cast<float*>(dof.p), k);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        float probe = 0.0f;
        CK(cudaMemcpy(&probe, dof.p, 4, cudaMemcpyDeviceToHost));
        printf("PASS (scores[0]=%.6f)\n", probe);
    }

    if (sticky()) { rc = 1; goto done; }

    // ---------------- P3: real kernel, k=2 B=1 ----------------
    {
        const std::int32_t k = 2, B = 1;
        printf("P3: real scores kernel k=2 B=1 ... ");
        const Fixture f = make_fixture(k, B);
        DeviceBuf dc(f.candidates.size() * 4), dh(f.hidden.size() * 2),
            da(f.anchors.size() * 4), du(f.unary.size() * 4), dp(f.pred.size() * 2),
            ds(f.succ.size() * 2), dof(f.scores_elems * 4);
        CHECK_ALLOC();
        CK(cudaMemcpy(dc.p, f.candidates.data(), dc.n, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dh.p, f.hidden.data(), dh.n, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(da.p, f.anchors.data(), da.n, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(du.p, f.unary.data(), du.n, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dp.p, f.pred.data(), dp.n, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(ds.p, f.succ.data(), ds.n, cudaMemcpyHostToDevice));
        CK(cudaMemset(dof.p, 0xcd, dof.n));
        dim3 grid(B, k);
        ninfer::ops::dflash_selector_scores_kernel<<<grid, 256>>>(
            static_cast<const std::int32_t*>(dc.p),
            static_cast<const __nv_bfloat16*>(dh.p),
            static_cast<const std::int32_t*>(da.p),
            static_cast<const float*>(du.p),
            static_cast<const __nv_bfloat16*>(dp.p),
            static_cast<const __nv_bfloat16*>(ds.p),
            static_cast<float*>(dof.p), k);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        printf("PASS\n");
    }

    if (sticky()) { rc = 1; goto done; }

    // ---------------- P4/P5: stripped kernels ----------------
    for (int variant : {4, 5}) {
        const std::int32_t k = 2, B = 2;
        printf("P%d: stripped kernel (variant %d) k=2 B=2 ... ", variant, variant);
        const Fixture f = make_fixture(k, B);
        DeviceBuf dc(f.candidates.size() * 4), dh(f.hidden.size() * 2),
            da(f.anchors.size() * 4), du(f.unary.size() * 4), dp(f.pred.size() * 2),
            ds(f.succ.size() * 2), dof(f.scores_elems * 4);
        CHECK_ALLOC();
        CK(cudaMemcpy(dc.p, f.candidates.data(), dc.n, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dh.p, f.hidden.data(), dh.n, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(da.p, f.anchors.data(), da.n, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(du.p, f.unary.data(), du.n, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dp.p, f.pred.data(), dp.n, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(ds.p, f.succ.data(), ds.n, cudaMemcpyHostToDevice));
        CK(cudaMemset(dof.p, 0xcd, dof.n));
        dim3 grid(B, k);
        stripped_scores_kernel<<<grid, 256>>>(
            static_cast<const std::int32_t*>(dc.p),
            static_cast<const __nv_bfloat16*>(dh.p),
            static_cast<const std::int32_t*>(da.p),
            static_cast<const float*>(du.p),
            static_cast<const __nv_bfloat16*>(dp.p),
            static_cast<const __nv_bfloat16*>(ds.p),
            static_cast<float*>(dof.p), k, variant);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        printf("PASS\n");
    }

    if (sticky()) { rc = 1; goto done; }

    // ---------------- P6: real math, runtime loop ----------------
    {
        const std::int32_t k = 2, B = 2;
        printf("P6: real math, no unroll, k=2 B=2 ... ");
        const Fixture f = make_fixture(k, B);
        DeviceBuf dc(f.candidates.size() * 4), dh(f.hidden.size() * 2),
            da(f.anchors.size() * 4), du(f.unary.size() * 4), dp(f.pred.size() * 2),
            ds(f.succ.size() * 2), dof(f.scores_elems * 4);
        CHECK_ALLOC();
        CK(cudaMemcpy(dc.p, f.candidates.data(), dc.n, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dh.p, f.hidden.data(), dh.n, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(da.p, f.anchors.data(), da.n, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(du.p, f.unary.data(), du.n, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dp.p, f.pred.data(), dp.n, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(ds.p, f.succ.data(), ds.n, cudaMemcpyHostToDevice));
        CK(cudaMemset(dof.p, 0xcd, dof.n));
        dim3 grid(B, k);
        real_nounroll_kernel<<<grid, 256>>>(
            static_cast<const std::int32_t*>(dc.p),
            static_cast<const __nv_bfloat16*>(dh.p),
            static_cast<const std::int32_t*>(da.p),
            static_cast<const float*>(du.p),
            static_cast<const __nv_bfloat16*>(dp.p),
            static_cast<const __nv_bfloat16*>(ds.p),
            static_cast<float*>(dof.p), k);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        printf("PASS\n");
    }

done:
    printf("=== ladder rc=%d ===\n", rc);
    return rc;
}