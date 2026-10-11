#pragma once
/* GEMM compute traits and shared-memory ring budget. */
#include <cuda_fp8.h>
#include <type_traits>

#include <api/native/gemm.h>
#include <datatype/element.cuh>
#include <policy/geometry.h>
#include <mma/mma.cuh>
#include <utils/tensor.cuh>

namespace astrai {
namespace gemm {

/*
 * MMA compute type per operand pair — the tensor-core input type both
 * operands are brought to before mma.sync. Promotes to bf16 m16n8k16 when
 * int8 rides exactly one side (W8A16, A8W16, the only supported mixed
 * pair); symmetric pairs keep their native mma (bf16 pass-through, fp8 mma,
 * s8 mma with int32 accumulators).
 */
template <typename ElemA, typename ElemB> struct gemm_mma_traits {
    static constexpr bool kI8A = std::is_same_v<ElemA, int8_t>;
    static constexpr bool kI8B = std::is_same_v<ElemB, int8_t>;
    static constexpr bool kPromote = (kI8A != kI8B);
    using MmaT = std::conditional_t<kPromote, __nv_bfloat16, ElemA>;
    static constexpr bool kDequantA = !std::is_same_v<ElemA, MmaT>;
    static constexpr bool kDequantB = !std::is_same_v<ElemB, MmaT>;
};

struct MmaSync {
    static constexpr bool kTma = false, kMx = false;
};
struct TmaMma {
    static constexpr bool kTma = true, kMx = false;
};
struct Sm120Mma {
    static constexpr bool kTma = true, kMx = true;
};

/*
 * Compile-time tile configuration (CTA tile + warp tiling + pipeline depth).
 * ElemA/ElemB are independent operand types; the MMA runs on the promoted
 * MmaT (gemm_mma_traits): W16A16 passes through, symmetric fp8/int8 keep
 * their native mma (fp32/int32 accumulators), a lone 8-bit side against
 * bf16 dequantizes in-register (kDequantA/B mark those inserts per side).
 * UseMx swaps the symmetric-fp8 cell for the sm_120 block_scale cell.
 */
template <typename ElemA_,
          typename ElemB_,
          typename CtaShape_,
          typename WarpShape_,
          int PipelineStages,
          bool UseMx = false>
struct GemmTraits {
    using ElemA = ElemA_;
    using ElemB = ElemB_;
    using MmaPair = gemm_mma_traits<ElemA_, ElemB_>;
    using MmaT = typename MmaPair::MmaT;
    /*
     * The exact mma cell <MmaT, MmaT, shape> (mma/mma.cuh): one type
     * carries the instruction's K extent and accumulator type (fp32 for the
     * float families, s32 for the s8 pair).
     */
    static constexpr bool kMxCell =
        UseMx && (std::is_same_v<MmaT, __nv_fp8_e4m3> || std::is_same_v<MmaT, __nv_fp8_e5m2>);
    using MmaOp =
        std::conditional_t<kMxCell,
                           astrai::MxMmaOp<MmaT>,
                           astrai::MmaOp<MmaT, MmaT, typename astrai::MmaShapeFor<MmaT>::type>>;
    static_assert(!UseMx || kMxCell, "block-scaled MMA requires a symmetric FP8 pair");
    using AccT = typename MmaOp::AccT;
    using ElemTraitsA = astrai::ElemTrait<ElemA_>;
    using ElemTraitsB = astrai::ElemTrait<ElemB_>;

    using CtaShape = CtaShape_;
    using WarpShape = WarpShape_;
    static constexpr int kBlockM = CtaShape_::kM;
    static constexpr int kBlockN = CtaShape_::kN;
    static constexpr int kTile = CtaShape_::kK;
    static constexpr int kStages = PipelineStages;
    static constexpr int kWarpM = WarpShape_::kM;
    static constexpr int kWarpN = WarpShape_::kN;

    static constexpr int kElemBytesA = ElemTraitsA::kBytes;
    static constexpr int kElemBytesB = ElemTraitsB::kBytes;
    /*
     * MMA shape follows the promoted compute type; dequantized fragments
     * are brought to it in-register (dequant.cuh).
     */
    static constexpr int kMmaK = astrai::MmaShapeFor<MmaT>::type::kK;
    static constexpr bool kDequantA = MmaPair::kDequantA;
    static constexpr bool kDequantB = MmaPair::kDequantB;

    /*
     * Derived geometry: warp tiles tile the CTA; the smem budget is
     * layout-aware, so it lives in GemmSmem below.
     */
    static constexpr int kWarpsM = kBlockM / kWarpM;
    static constexpr int kWarpsN = kBlockN / kWarpN;
    static constexpr int kCtaThreads = kWarpsM * kWarpsN * 32;
    static_assert(kWarpsM * kWarpM == kBlockM && kWarpsN * kWarpN == kBlockN,
                  "warp tiles must exactly tile the CTA");
    static_assert(kWarpM % 16 == 0 && kWarpN % 8 == 0,
                  "warp tile must be a multiple of the m16n8 MMA shape");
    /*
     * Warp accumulator geometry, one definition for mainloop and epilogue:
     * m16n8 mma cells on the (kMt, kNt) warp tile grid.
     */
    static constexpr int kMt = kWarpM / 16;
    static constexpr int kNt = kWarpN / 8;
    using AccTensor = Tensor<ArrayEngine<typename MmaOp::CFrag, kMt * kNt>, CellLayout<kNt>>;
};

/* Each kStages+1 ring reuses an already-consumed slot, avoiding a post-compute barrier. */
template <typename Traits, typename LayoutA, typename LayoutB> struct GemmSmem {
    /* Crosswise operands: ColMajor A or RowMajor B. */
    static constexpr bool kDirectA = direct_a<LayoutA>();
    static constexpr bool kDirectB = direct_b<LayoutB>();
    static constexpr int kRingDepth = Traits::kStages + 1;
    static constexpr int kBytes = ring_smem_bytes(Traits::kBlockM,
                                                  Traits::kBlockN,
                                                  Traits::kTile,
                                                  Traits::kStages,
                                                  Traits::kElemBytesA,
                                                  Traits::kElemBytesB);
    static constexpr int kMinCtas = min_ctas_for_ring(kBytes);
};

} // namespace gemm
} // namespace astrai
