/* Compiled-in GEMM tuning rows; generated arrays stay private to this TU. */
#include "table.h"

#include <array>

namespace astrai {
namespace gemm {

/*
 * clang-format off
 * BEGIN GENERATED
 * Rows are the measured DIFF of the model, not full coverage: emitted only
 * where a recipe beat the model's dispatch by >=2% in interleaved A/B
 * (four rounds/point; the sweep's model reference runs last, at the hottest
 * clock — 10-20% underprice on heavy shapes, measured 2026-09-14). Ties and
 * unmeasured bands serve the model. Measured 2026-09-14, this box: sm_120 /
 * RTX 5090 / 170 SMs, M {1..4096} x N {1024..28672} x K {1536,4096,8192},
 * 2% production-semantics holdout.
 *
 * Crosswise F8A8 rows (2026-09-19): the wide CTA joined the byte ladders
 * for crosswise staging, moving the L2 re-read wall — the 64x64 model pick
 * streams 4.8GB of operands through L2 on the qkv cell (84.7% L2 SOL, 48%
 * compute) vs 2.4GB (128-row) and 1.8GB (wide). m=16384, astrai_1b
 * projections, four A/B rounds/point. NN rows band the canonicalized
 * aspect; (16384,1536) splits by exact k band into square (k=1536) and
 * mlp_down (k=6912). Bands are the measured points only. The kK=32 unlock
 * replaced three rows after a four-round confirm: square NN/TT and mlp_down
 * NN run the kK=32 big CTA (the kK=32 narrow lost everywhere, -8..-34%);
 * kK=64 keeps the other nine points.
 *
 * A stale row is worse than none (2026-09-14 +33% lesson): the tier is
 * signature-guarded by kBuiltinPlanMeasuredOn — mismatch serves nothing,
 * default chain runs override -> injected -> [no builtin] -> heuristic.
 * Another part gets its own sweep, never these rows.
 */

static constexpr std::array<TableRow, 29> kBuiltinPlanW16A16 = {{
    /*
     * Prepended so its band wins over the 128x128 kk32 row below, 3.0-3.2x
     * off here: n <= 256 gives that tile 40 blocks over 170 SMs where this
     * fills 160 (interleaved A/B, 2026-09-14, n=256, k {2048,4096},
     * m {1792..2560}; the band is the measured one).
     */
    {TileClass::kSmall64, 1536, 3072, 0, 256, 0, 0, 3, 0, 64},
    {TileClass::kTall64x128, 3072, 0, 5120, 8576, 0, 0, 2, 0, 32},
    {TileClass::kTall64x128, 0, 12, 8576, 19840, 0, 0, 3, 0, 32},
    {TileClass::kTall64x128, 12, 96, 8576, 19840, 0, 0, 2, 0, 32},
    {TileClass::kTall64x128, 0, 4, 19840, 0, 0, 0, 3, 0, 32},
    {TileClass::kSmall64, 0, 768, 0, 1280, 0, 0, 3, 0, 64},
    {TileClass::kNarrow128x64, 768, 1536, 0, 1280, 0, 0, 3, 0, 64},
    {TileClass::kBig128, 1536, 3072, 0, 1280, 0, 0, 3, 0, 32},
    {TileClass::kSmall64, 0, 384, 1280, 2816, 0, 0, 3, 0, 64},
    {TileClass::kNarrow128x64, 384, 768, 1280, 2816, 0, 0, 3, 0, 64},
    {TileClass::kNarrow128x64, 3072, 0, 1280, 2816, 0, 0, 2, 0, 32},
    {TileClass::kSmall64, 0, 192, 2816, 5120, 0, 0, 3, 0, 64},
    {TileClass::kNarrow128x64, 192, 384, 2816, 5120, 0, 0, 3, 0, 64},
    {TileClass::kBig128, 384, 768, 2816, 5120, 0, 0, 2, 0, 64},
    {TileClass::kSmall64, 0, 24, 5120, 8576, 0, 0, 3, 0, 64},
    {TileClass::kSmall64, 24, 48, 5120, 8576, 0, 0, 2, 0, 64},
    {TileClass::kSmall64, 48, 96, 5120, 8576, 0, 0, 3, 0, 64},
    {TileClass::kNarrow128x64, 96, 192, 5120, 8576, 0, 0, 3, 0, 64},
    {TileClass::kSmall64, 192, 384, 5120, 8576, 0, 0, 2, 0, 32},
    {TileClass::kNarrow128x64, 768, 1536, 5120, 8576, 0, 0, 2, 0, 32},
    {TileClass::kSmall64, 0, 4, 8576, 19840, 0, 0, 3, 0, 64},
    {TileClass::kSmall64, 4, 12, 8576, 19840, 0, 0, 2, 0, 32},
    {TileClass::kNarrow128x64, 1536, 3072, 8576, 19840, 0, 0, 2, 0, 64},
    {TileClass::kBig128, 3072, 0, 8576, 19840, 0, 0, 2, 0, 64},
    {TileClass::kSmall64, 0, 96, 19840, 0, 0, 0, 3, 0, 64},
    {TileClass::kNarrow128x64, 96, 192, 19840, 0, 0, 0, 3, 0, 64},
    {TileClass::kBig128, 192, 384, 19840, 0, 0, 0, 2, 0, 64},
    {TileClass::kNarrow128x64, 384, 768, 19840, 0, 0, 0, 2, 0, 64},
    {TileClass::kBig128, 768, 0, 19840, 0, 0, 0, 2, 0, 64},
}};
static constexpr std::array<TableRow, 23> kBuiltinPlanW8A16 = {{
    /*
     * Same narrow-N pathology and fix as the W16A16 row above: the 128x128
     * kk64 row below is 2.9-3.1x off at n=256, m 1792..2560 (A/B, 2026-09-14,
     * k=4096); the band is the measured one.
     */
    {TileClass::kSmall64, 1536, 3072, 0, 256, 1, 0, 3, 0, 64},
    {TileClass::kTall64x128, 0, 96, 8576, 19840, 1, 0, 3, 0, 32},
    {TileClass::kTall64x128, 0, 4, 19840, 0, 1, 0, 2, 0, 32},
    {TileClass::kTall64x128, 96, 192, 19840, 0, 1, 0, 3, 0, 32},
    {TileClass::kNarrow128x64, 768, 1536, 0, 1280, 1, 0, 3, 0, 64},
    {TileClass::kBig128, 1536, 0, 0, 1280, 1, 0, 3, 0, 64},
    {TileClass::kNarrow128x64, 384, 768, 1280, 2816, 1, 0, 3, 0, 64},
    {TileClass::kSmall64, 768, 3072, 1280, 2816, 1, 0, 2, 0, 64},
    {TileClass::kNarrow128x64, 3072, 0, 1280, 2816, 1, 0, 2, 0, 32},
    {TileClass::kNarrow128x64, 192, 384, 2816, 5120, 1, 0, 3, 0, 64},
    {TileClass::kBig128, 384, 1536, 2816, 5120, 1, 0, 3, 0, 64},
    {TileClass::kNarrow128x64, 1536, 0, 2816, 5120, 1, 0, 2, 0, 32},
    {TileClass::kNarrow128x64, 96, 192, 5120, 8576, 1, 0, 3, 0, 64},
    {TileClass::kSmall64, 192, 768, 5120, 8576, 1, 0, 2, 0, 64},
    {TileClass::kNarrow128x64, 768, 0, 5120, 8576, 1, 0, 2, 0, 32},
    {TileClass::kSmall64, 0, 768, 8576, 19840, 1, 0, 2, 0, 64},
    {TileClass::kNarrow128x64, 768, 3072, 8576, 19840, 1, 0, 2, 0, 32},
    {TileClass::kBig128, 3072, 0, 8576, 19840, 1, 0, 3, 0, 64},
    {TileClass::kSmall64, 4, 96, 19840, 0, 1, 0, 2, 0, 64},
    {TileClass::kNarrow128x64, 96, 192, 19840, 0, 1, 0, 3, 0, 64},
    {TileClass::kBig128, 192, 384, 19840, 0, 1, 0, 3, 0, 64},
    {TileClass::kNarrow128x64, 384, 768, 19840, 0, 1, 0, 2, 0, 32},
    {TileClass::kBig128, 768, 0, 19840, 0, 1, 0, 3, 0, 64},
}};
static constexpr std::array<TableRow, 30> kBuiltinPlanW8A8 = {{
    // SM120 W8A8: larger K tile wins at mid-M, wide-N (CUDA Graph ABBA).
    {TileClass::kBig128, 512, 1024, 5120, 8192, 2, 0, 2, 0, 128, 1023, 2048},
    /*
     * The narrow-N pathology of the W16A16/W8A16 rows above, same band and
     * same winner: the 128x128 kk64 row below measured 1.56-1.74x off at n=256,
     * m 1792..2560 (interleaved A/B, 2026-09-14, k=4096).
     */
    {TileClass::kSmall64, 1536, 3072, 0, 256, 2, 0, 3, 0, 64},
    {TileClass::kSmall64, 12, 768, 0, 1280, 2, 0, 3, 0, 64},
    {TileClass::kBig128, 1536, 3072, 0, 1280, 2, 0, 3, 0, 64},
    {TileClass::kWide128x256, 3072, 0, 0, 1280, 2, 0, 2, 0, 64},
    {TileClass::kSmall64, 12, 192, 1280, 2816, 2, 0, 3, 0, 64},
    {TileClass::kSmall64, 192, 384, 1280, 2816, 2, 0, 2, 0, 64},
    {TileClass::kBig128, 768, 1536, 1280, 2816, 2, 0, 3, 0, 64},
    {TileClass::kWide128x256, 1536, 3072, 1280, 2816, 2, 0, 2, 0, 64},
    {TileClass::kBig128, 3072, 0, 1280, 2816, 2, 0, 3, 0, 64},
    {TileClass::kSmall64, 12, 48, 2816, 5120, 2, 0, 3, 0, 64},
    {TileClass::kSmall64, 48, 96, 2816, 5120, 2, 0, 2, 0, 64},
    {TileClass::kSmall64, 96, 192, 2816, 5120, 2, 0, 3, 0, 64},
    {TileClass::kBig128, 384, 768, 2816, 5120, 2, 0, 3, 0, 64},
    {TileClass::kWide128x256, 768, 3072, 2816, 5120, 2, 0, 2, 0, 64},
    {TileClass::kBig128, 3072, 0, 2816, 5120, 2, 0, 2, 0, 64},
    {TileClass::kSmall64, 12, 96, 5120, 8576, 2, 0, 3, 0, 64},
    {TileClass::kBig128, 192, 384, 5120, 8576, 2, 0, 3, 0, 64},
    {TileClass::kWide128x256, 384, 768, 5120, 8576, 2, 0, 2, 0, 64},
    {TileClass::kBig128, 768, 3072, 5120, 8576, 2, 0, 2, 0, 64},
    {TileClass::kWide128x256, 3072, 0, 5120, 8576, 2, 0, 2, 0, 64},
    {TileClass::kSmall64, 12, 24, 8576, 19840, 2, 0, 3, 0, 64},
    {TileClass::kSmall64, 24, 96, 8576, 19840, 2, 0, 2, 0, 64},
    {TileClass::kBig128, 96, 192, 8576, 19840, 2, 0, 2, 0, 64},
    {TileClass::kWide128x256, 192, 384, 8576, 19840, 2, 0, 2, 0, 64},
    {TileClass::kBig128, 384, 768, 8576, 19840, 2, 0, 3, 0, 64},
    {TileClass::kBig128, 768, 3072, 8576, 19840, 2, 0, 2, 0, 64},
    {TileClass::kWide128x256, 3072, 0, 8576, 19840, 2, 0, 2, 0, 64},
    {TileClass::kBig128, 192, 384, 19840, 0, 2, 0, 3, 0, 64},
    {TileClass::kWide128x256, 384, 0, 19840, 0, 2, 0, 2, 0, 64},
}};
static constexpr std::array<TableRow, 41> kBuiltinPlanF8A8 = {{
    /*
     * The same narrow-N pathology, band and winner as above: the 128x128
     * kk64 row below measured 1.72-1.91x off at n=256, m 1792..2560 (A/B,
     * 2026-09-14, k=4096).
     */
    {TileClass::kSmall64, 1536, 3072, 0, 256, 3, 0, 3, 0, 64},
    {TileClass::kSmall64, 12, 768, 0, 1280, 3, 0, 3, 0, 64},
    {TileClass::kBig128, 1536, 3072, 0, 1280, 3, 0, 3, 0, 64},
    {TileClass::kWide128x256, 3072, 0, 0, 1280, 3, 0, 2, 0, 64},
    {TileClass::kSmall64, 12, 384, 1280, 2816, 3, 0, 3, 0, 64},
    {TileClass::kBig128, 768, 1536, 1280, 2816, 3, 0, 3, 0, 64},
    {TileClass::kWide128x256, 1536, 3072, 1280, 2816, 3, 0, 2, 0, 64},
    {TileClass::kBig128, 3072, 0, 1280, 2816, 3, 0, 2, 0, 64},
    {TileClass::kSmall64, 12, 192, 2816, 5120, 3, 0, 3, 0, 64},
    {TileClass::kBig128, 384, 768, 2816, 5120, 3, 0, 3, 0, 64},
    {TileClass::kWide128x256, 768, 1536, 2816, 5120, 3, 0, 2, 0, 64},
    {TileClass::kBig128, 1536, 3072, 2816, 5120, 3, 0, 2, 0, 64},
    {TileClass::kBig128, 3072, 0, 2816, 5120, 3, 0, 3, 0, 64},
    {TileClass::kSmall64, 12, 24, 5120, 8576, 3, 0, 2, 0, 64},
    {TileClass::kSmall64, 24, 96, 5120, 8576, 3, 0, 3, 0, 64},
    {TileClass::kBig128, 192, 384, 5120, 8576, 3, 0, 3, 0, 64},
    {TileClass::kWide128x256, 384, 768, 5120, 8576, 3, 0, 2, 0, 64},
    {TileClass::kBig128, 768, 3072, 5120, 8576, 3, 0, 2, 0, 64},
    {TileClass::kWide128x256, 3072, 0, 5120, 8576, 3, 0, 2, 0, 64},
    {TileClass::kSmall64, 12, 24, 8576, 19840, 3, 0, 3, 0, 64},
    {TileClass::kSmall64, 24, 96, 8576, 19840, 3, 0, 2, 0, 64},
    {TileClass::kBig128, 96, 192, 8576, 19840, 3, 0, 3, 0, 64},
    {TileClass::kWide128x256, 192, 384, 8576, 19840, 3, 0, 2, 0, 64},
    {TileClass::kBig128, 384, 0, 8576, 19840, 3, 0, 2, 0, 64},
    {TileClass::kSmall64, 12, 96, 19840, 0, 3, 0, 2, 0, 64},
    {TileClass::kBig128, 192, 384, 19840, 0, 3, 0, 3, 0, 64},
    {TileClass::kWide128x256, 384, 768, 19840, 0, 3, 0, 2, 0, 64},
    {TileClass::kBig128, 768, 1536, 19840, 0, 3, 0, 2, 0, 64},
    {TileClass::kWide128x256, 1536, 0, 19840, 0, 3, 0, 2, 0, 64},
    /*
     * Crosswise rows, 2026-09-19 (block comment above): qkv. NN bands the
     * swapped aspect (canonicalize_gemm plans NN as 6144x16384).
     */
    {TileClass::kBig128, 6143, 6144, 16383, 16384, 3, 1, 2, 0, 64, 1535, 1536},
    {TileClass::kBig128, 16383, 16384, 6143, 6144, 3, 1, 2, 0, 64, 1535, 1536},
    {TileClass::kWide128x256, 16383, 16384, 6143, 6144, 3, 2, 2, 0, 64, 1535, 1536},
    /*
     * square (k=1536) and mlp_down (k=6912) share the (16384,1536) aspect;
     * the exact k band splits them. kK=32 big CTA carries NN/TT (deeper 32KB
     * ring, same 2.4GB of L2 traffic at higher occupancy), TN keeps wide
     * kK=64.
     */
    {TileClass::kBig128, 1535, 1536, 16383, 16384, 3, 1, 3, 0, 32, 1535, 1536},
    {TileClass::kBig128, 16383, 16384, 1535, 1536, 3, 1, 3, 0, 32, 1535, 1536},
    {TileClass::kWide128x256, 16383, 16384, 1535, 1536, 3, 2, 2, 0, 64, 1535, 1536},
    // mlp_up.
    {TileClass::kBig128, 6911, 6912, 16383, 16384, 3, 1, 2, 0, 64, 1535, 1536},
    {TileClass::kBig128, 16383, 16384, 6911, 6912, 3, 1, 2, 0, 64, 1535, 1536},
    {TileClass::kWide128x256, 16383, 16384, 6911, 6912, 3, 2, 2, 0, 64, 1535, 1536},
    // mlp_down (k=6912).
    {TileClass::kBig128, 1535, 1536, 16383, 16384, 3, 1, 3, 0, 32, 6911, 6912},
    {TileClass::kWide128x256, 16383, 16384, 1535, 1536, 3, 1, 2, 0, 64, 6911, 6912},
    {TileClass::kWide128x256, 16383, 16384, 1535, 1536, 3, 2, 2, 0, 64, 6911, 6912},
}};

// The device the rows above were measured on.
static constexpr DeviceFacts kBuiltinPlanMeasuredOn = {
    /*sms=*/170, /*smem_max=*/101376, /*smem_per_sm=*/102400,
    /*regs_per_sm=*/65536, /*l2_bytes=*/100663296, /*cc=*/120};

bool builtin_rows_match_device(const DeviceFacts& dev) {
    const DeviceFacts& m = kBuiltinPlanMeasuredOn;
    return dev.cc == m.cc && dev.sms == m.sms && dev.smem_max == m.smem_max &&
           dev.smem_per_sm == m.smem_per_sm && dev.regs_per_sm == m.regs_per_sm &&
           dev.l2_bytes == m.l2_bytes;
}

/*
 * END GENERATED
 * clang-format on
 */

/*
 * Builtin table for one dtype class; count receives its row count. An empty
 * table (the shipped default) returns a valid pointer and a zero count, so
 * plan_row_for matches nothing and the default chain falls through to the heuristic.
 */
const TableRow* builtin_plan_table(int perf_class, int& count) {
    switch (perf_class) {
    case 0:
        count = (int)kBuiltinPlanW16A16.size();
        return kBuiltinPlanW16A16.data();
    case 1:
        count = (int)kBuiltinPlanW8A16.size();
        return kBuiltinPlanW8A16.data();
    case 2:
        count = (int)kBuiltinPlanW8A8.size();
        return kBuiltinPlanW8A8.data();
    case 3:
        count = (int)kBuiltinPlanF8A8.size();
        return kBuiltinPlanF8A8.data();
    default:
        count = 0;
        return nullptr;
    }
}

} // namespace gemm
} // namespace astrai
