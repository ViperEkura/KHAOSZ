#pragma once
/* Pure C++ GEMM geometry shared by planning and device policy. */
#include <type_traits>

#include <api/native/gemm.h>

namespace astrai {
namespace gemm {

/*
 * Ring-budget formula, one source for GemmSmem, the launcher's epilogue
 * reclaim check and the host planner's recipe feasibility gate:
 * every operand ring holds kStages+1 buffers of k * (bm*ba + bn*bb) bytes.
 */
constexpr int ring_smem_bytes(int bm, int bn, int k, int k_stages, int ba, int bb) {
    return (k_stages + 1) * k * (bm * ba + bn * bb);
}

/*
 * Shared compiler/planner CTA hint and register budget. The 48 KiB watermark
 * is a preference, not a device limit; occupancy still uses each device's
 * smem_per_sm so larger-memory GPUs can pack additional CTAs.
 */
constexpr int min_ctas_for_ring(int bytes) { return bytes <= 48 * 1024 ? 2 : 1; }

constexpr int tma_smem_bytes(int ring, int k_stages) {
    return ring + 1024 + 2 * (k_stages + 1) * 8;
}

/* Crosswise staging uses ColMajor A or RowMajor B; the other layouts stage as-is. */
template <typename Layout> constexpr bool direct_a() { return std::is_same_v<Layout, ColMajor>; }
template <typename Layout> constexpr bool direct_b() { return std::is_same_v<Layout, RowMajor>; }

} // namespace gemm
} // namespace astrai
