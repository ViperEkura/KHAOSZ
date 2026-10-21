/*
 * fp8 x fp8 e5m2 instantiation unit. One explicit gemm_dispatch
 * instantiation per unit keeps the heavy template work in parallel nvcc
 * jobs; gemm/entry.cu addresses the specialization through its launcher
 * header.
 */
#include <launcher/gemm_dispatch.cuh>

namespace astrai {
namespace gemm {

ASTRAI_GEMM_INSTANTIATE(__nv_fp8_e5m2, __nv_fp8_e5m2);

} // namespace gemm
} // namespace astrai
