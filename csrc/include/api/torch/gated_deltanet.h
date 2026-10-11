/*
 * GDN entry points: fwd prepares the chunked forward's operands, bwd is the
 * first stage of the reverse pass (the chunked kernels are not in the tree).
 */

#pragma once

#include <torch/types.h>

#include <vector>

namespace astrai {
namespace gdn {

std::vector<torch::Tensor> gated_deltanet_fwd(torch::Tensor q,
                                              torch::Tensor k,
                                              torch::Tensor v,
                                              torch::Tensor g,
                                              torch::Tensor beta,
                                              double eps,
                                              int64_t chunk);

std::vector<torch::Tensor> gated_deltanet_bwd(torch::Tensor q,
                                              torch::Tensor k,
                                              torch::Tensor v_new,
                                              torch::Tensor h,
                                              torch::Tensor g,
                                              torch::Tensor do_grad,
                                              double scale);

} // namespace gdn
} // namespace astrai
