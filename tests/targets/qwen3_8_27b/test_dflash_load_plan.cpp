#include "artifact/binder.h"
#include "artifact/reader.h"
#include "targets/qwen3_6_27b/impl/config.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>

namespace {

// D2 (spec 06): the 27B drafter is pure-SWA — all five layers are sliding-window, so
// local_layers == layers and the dflash.full paged pool is compile-time eliminated
// (layouts_impl.h persistent_layout and dflash_context_impl.h gate it on
// local_layers < layers). The pool lives in the frozen runtime layout, not in the
// binder's materialization plan, so its absence is observable here as: the section
// carries exactly the 66 drafter weight objects and the dflash plan's device capacity
// contains no full-cache pool bytes (it is shared text + globals + the 66-object
// section only, displacing the MTP and vision objects the feature turns off).
static_assert(ninfer::targets::qwen3_6_27b::detail::DFlashConfig::local_layers ==
                  ninfer::targets::qwen3_6_27b::detail::DFlashConfig::layers);

std::filesystem::path artifact_path() {
    if (const char* env = std::getenv("NINFER_QWEN3_8_27B_WEIGHTS");
        env != nullptr && *env != '\0') {
        return env;
    }
    return std::filesystem::path(NINFER_SOURCE_DIR) / "out/qwen3_8_27b_nvfp4.ninfer";
}

ninfer::targets::qwen3_6::StartupFeatures load_features(bool dflash) {
    return {
        .vision = !dflash,
        .speculative =
            dflash ? ninfer::SpeculativeBackend::DFlash : ninfer::SpeculativeBackend::Mtp,
        // D4: the 27B path selector needs a true top-16 over the full 248,320-vocabulary
        // output head under dflash. The registered 27B default (StartupFeatures and
        // SpeculativeOptions both default to the full head) is already the full head, so
        // both plans bind it; the optimized draft head (text/draft_head) stays
        // ValidateOnly in both plans (optimized_proposal() is false with the full head).
        .proposal_head = ninfer::ProposalHead::Full,
    };
}

} // namespace

int main() {
    const std::filesystem::path path = artifact_path();
    if (!std::filesystem::is_regular_file(path)) {
        std::cerr << "skip: real 27B artifact is unavailable at " << path << '\n';
        return 77;
    }

    // The qwen3.8-27b fleet NVFP4 artifact (1124 registered objects; the DFlash-augmented
    // image appends the 66 dflash/* objects after the Vision merger objects, artifact doc
    // section 14) resolves to the 27B target with the nvfp4 weights profile.
    const auto profile = ninfer::targets::qwen3_6_27b::detail::WeightsProfile::Qwen38Nvfp4;
    ninfer::artifact::Reader reader(path);
    {
        // No-dflash plan: vision + MTP resident, dflash section validated only.
        // Device objects = 3 text globals + 656 text tensors + 12 MTP + 333 vision.
        // ValidateOnly = 2 draft-head objects + 112 NVFP4 input divisors + 66 dflash.
        // Host = 6 frontend resources. 1004 + 180 + 6 = 1190.
        ninfer::artifact::Binder binder(reader);
        const auto plan =
            ninfer::targets::qwen3_6_27b::detail::bind_artifact(binder, profile, load_features(false));
        if (plan.materialization.object_count != 1190 ||
            plan.materialization.device_objects.size() != 1004 ||
            plan.materialization.host_objects.size() != 6 ||
            plan.materialization.device_capacity_bytes != 21'122'608'640ULL ||
            plan.bindings.dflash.feature_projection.index != 1124 ||
            plan.bindings.dflash.final_norm.index != 1186) {
            std::cerr << "DFlash-disabled materialization plan changed resident weights\n";
            return 1;
        }
    }
    {
        // DFlash plan: the 66 dflash objects (section bytes 2,226,792,960, matching the
        // converter's EXPECTED_PAYLOAD_BYTES) move to the device while the 12 MTP and 333
        // vision objects become validate-only. Device objects = 3 + 656 + 66 = 725.
        // ValidateOnly = 2 + 12 + 333 + 112 = 459. 725 + 459 + 6 = 1190.
        ninfer::artifact::Binder binder(reader);
        const auto plan =
            ninfer::targets::qwen3_6_27b::detail::bind_artifact(binder, profile, load_features(true));
        if (plan.materialization.object_count != 1190 ||
            plan.materialization.device_objects.size() != 725 ||
            plan.materialization.host_objects.size() != 6 ||
            plan.materialization.device_capacity_bytes != 22'602'414'592ULL) {
            std::cerr << "DFlash-enabled materialization plan is incomplete: device_objects="
                      << plan.materialization.device_objects.size()
                      << " device_bytes=" << plan.materialization.device_capacity_bytes << '\n';
            return 1;
        }
    }
    std::cout << "ok\n";
    return 0;
}