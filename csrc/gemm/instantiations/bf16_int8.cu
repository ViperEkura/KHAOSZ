/*
 * W8A16 instantiation unit: bf16 x int8 (b_scale required). One explicit
 * gemm_dispatch instantiation per unit keeps the heavy template work in
 * parallel nvcc jobs; gemm/entry.cu addresses the specialization through its
 * launcher header.
 */
#include <launcher/gemm_dispatch.cuh>

namespace astrai {
namespace gemm {

ASTRAI_GEMM_INSTANTIATE(__nv_bfloat16, int8_t);

} // namespace gemm
} // namespace astrai
