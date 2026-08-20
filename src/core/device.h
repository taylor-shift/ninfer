#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace ninfer {

void cuda_check(cudaError_t err, const char* expr, const char* file, int line);

#define CUDA_CHECK(expr) ::ninfer::cuda_check((expr), #expr, __FILE__, __LINE__)

// Multiprocessor count of the current device, queried once and cached.
//
// Launch geometry for persistent and cooperative kernels is derived from this value instead of a
// fixed constant, so a wider sm_120 part fills every SM. The result is stable for the process and
// cheap enough to call from a launch path, including one being captured into a CUDA Graph: grid
// dimensions stay deterministic for a given launch configuration.
//
// Both accessors are noexcept so that noexcept schedule-planning paths can derive residency limits
// from them; a failed device query aborts through the same path as CUDA_CHECK.
int sm_count() noexcept;

// Device-wide CTA target for a schedule that admits `ctas_per_sm` resident CTAs per SM.
int resident_cta_target(int ctas_per_sm) noexcept;

struct DeviceContext {
    int device               = 0;
    cudaStream_t stream      = nullptr;
    cudaStream_t load_stream = nullptr;
    cudaDeviceProp props{};

    explicit DeviceContext(int device_id = 0);
    ~DeviceContext();

    DeviceContext(const DeviceContext&)            = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;
    DeviceContext(DeviceContext&& other) noexcept;
    DeviceContext& operator=(DeviceContext&& other) noexcept;

    int sm() const noexcept;
    std::size_t total_vram() const noexcept;
    void synchronize() const;

    // Bind this context's device to the CALLING thread.
    //
    // cudaSetDevice is per-thread state, and the constructor only binds the
    // thread that created the context. Any other thread that reaches CUDA work
    // — the HTTP server hands each request to its own httplib worker — starts
    // on device 0. Launching a kernel on a device-N stream from a device-0
    // thread fails with cudaErrorInvalidValue; on a multi-GPU host that
    // surfaces as `gqa_attention_prefill.cu:58 CUDA_CHECK(cudaGetLastError())
    // failed: cudaErrorInvalidValue` during warmup for every engine except the
    // one that happens to sit on GPU 0.
    //
    // Idempotent and cheap (a no-op once the thread is already bound), so it is
    // safe to call on every entry into engine work.
    void bind_current_thread() const;
};

// RAII helper: binds ctx's device for the current scope. Use at the entry point
// of any function that may run on a thread which did not create the context.
class DeviceGuard {
public:
    explicit DeviceGuard(const DeviceContext& ctx) { ctx.bind_current_thread(); }

    DeviceGuard(const DeviceGuard&)            = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;
};

class CudaEventTimer {
public:
    explicit CudaEventTimer(const DeviceContext& ctx);
    ~CudaEventTimer();

    CudaEventTimer(const CudaEventTimer&)            = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;
    CudaEventTimer(CudaEventTimer&& other) noexcept;
    CudaEventTimer& operator=(CudaEventTimer&& other) noexcept;

    void start();
    void record_stop();
    [[nodiscard]] float elapsed_ms() const;
    float stop_ms();

private:
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_   = nullptr;
    cudaEvent_t stop_    = nullptr;
};

} // namespace ninfer
