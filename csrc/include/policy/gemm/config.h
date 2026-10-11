#pragma once
/* Process-wide GEMM runtime switches and environment seeding. */
#include <atomic>
#include <cstdint>

#include <api/native/gemm_plan.h>

namespace astrai {
namespace gemm {

struct GemmConfig {
    std::atomic<int> planner{-1};      // 0 table, 1 hybrid, 2 fitted model, 3 geometry heuristic
    std::atomic<int> log{-1};          // [gemm-plan] stderr log on/off
    std::atomic<int> tma_disabled{-1}; // cp.async staging forced everywhere
    std::atomic<int> mx_disabled{-1};  // sm_120a block-scale cell knocked out
    std::atomic<int> table_off{-1};    // 1 = "-" (no override, no injected, no builtin rows)
};

inline GemmConfig& gemm_config() {
    static GemmConfig cfg;
    return cfg;
}

void gemm_config_seed_once(); // gemm/plan/planner.cpp: environment defaults

inline bool gemm_plan_log_enabled() {
    gemm_config_seed_once();
    return gemm_config().log.load(std::memory_order_relaxed) > 0;
}
inline bool gemm_tma_staging_disabled() {
    gemm_config_seed_once();
    return gemm_config().tma_disabled.load(std::memory_order_relaxed) > 0;
}
inline bool gemm_mx_cell_disabled() {
    gemm_config_seed_once();
    return gemm_config().mx_disabled.load(std::memory_order_relaxed) > 0;
}

} // namespace gemm
} // namespace astrai
