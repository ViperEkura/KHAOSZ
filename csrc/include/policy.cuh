#pragma once
/* One GEMM kernel policy composed from compute traits and a tile recipe. */
#include <policy/manifest.cuh>
#include <policy/traits.cuh>

namespace astrai {
namespace gemm {

struct GemmPolicyOptions {
    using Schedule = MmaSync;
    static constexpr bool kStreamOut = false;
    static constexpr bool kUseTma = false;
    static constexpr bool kUseMx = false;
    static constexpr bool kStoreWriteThrough = false;
};

template <typename Schedule_, bool UseTma_> struct PlannedGemmOptions : GemmPolicyOptions {
    using Schedule = Schedule_;
    static constexpr bool kUseTma = UseTma_;
    static constexpr bool kUseMx = Schedule_::kMx;
};

template <typename ElemA_,
          typename ElemB_,
          typename LayoutA_,
          typename LayoutB_,
          typename Tile_,
          typename LayoutOut_ = RowMajor,
          typename OutT_ = __nv_bfloat16,
          typename Options_ = GemmPolicyOptions>
struct GemmPolicy {
    using Schedule = typename Options_::Schedule;
    using Tile = Tile_;
    using Traits = GemmTraits<ElemA_,
                              ElemB_,
                              typename Tile_::CtaShape,
                              typename Tile_::WarpShape,
                              Tile_::kStages,
                              Options_::kUseMx>;
    using LayoutTagA = LayoutA_;
    using LayoutTagB = LayoutB_;
    /*
     * Output orientation (CUTLASS LayoutC): direction in the type, stride
     * in GemmParams::out_ld. OutT: bf16 (fused-linear convention) or fp32
     * (accumulated outputs, e.g. training dX/dW).
     */
    using LayoutTagOut = LayoutOut_;
    using OutT = OutT_;
    static constexpr bool kStreamOut = Options_::kStreamOut;
    /*
     * __stwt write-through store: the fused-linear output is read-once, so
     * keeping it out of L2 reserves the cache for reused weights/activations.
     * Separate from kStreamOut (__stcs, evict-first); write-through wins
     * when both are set.
     */
    static constexpr bool kStoreWriteThrough = Options_::kStoreWriteThrough;
    /*
     * TMA staging (sm_90+): congruous-only by construction — these policies
     * are instantiated solely for dual-congruous layout pairs with aligned
     * operands; staging layouts and fragment addressing are identical, only
     * the load/wait discipline changes (tma.cuh).
     */
    static constexpr bool kUseTma = Options_::kUseTma;
    static_assert(!Options_::kUseTma || (sizeof(ElemA_) <= 2 && sizeof(ElemB_) <= 2),
                  "TMA staging covers the 1-/2-byte congruous dtypes");
    using Smem = GemmSmem<Traits, LayoutA_, LayoutB_>;
    // The planner and launcher share the same TMA pad/barrier budget.
    // Flattened for __launch_bounds__, which takes no dependent type names.
    static constexpr int kCtaThreads = Traits::kCtaThreads;
    static constexpr int kMinCtas = Smem::kMinCtas;
    static constexpr int kSmemBytes =
        Options_::kUseTma ? tma_smem_bytes(Smem::kBytes, Tile_::kStages) : Smem::kBytes;
};

} // namespace gemm
} // namespace astrai
