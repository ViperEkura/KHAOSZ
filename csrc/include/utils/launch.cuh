/*
 * CUDA launch checks shared by native harnesses and Torch extensions.
 * Native code raises a C++ exception; pybind turns it into a Python
 * RuntimeError, while standalone callers choose their own failure policy.
 */
#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace astrai {
namespace cuda {

class Error : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

inline void check(cudaError_t error, const char* operation, const char* file, int line) {
    if (error != cudaSuccess) {
        throw Error(std::string("ASTRAI: ") + operation + " failed: " +
                    cudaGetErrorString(error) + " (" + file + ":" +
                    std::to_string(line) + ")");
    }
}

} // namespace cuda
} // namespace astrai

#define ASTRAI_CUDA_CHECK(expr) ::astrai::cuda::check((expr), #expr, __FILE__, __LINE__)

// Check immediately after a kernel launch so a rejected configuration is
// reported at its source rather than surfacing at an unrelated CUDA call.
#define ASTRAI_LAUNCH_CHECK() ASTRAI_CUDA_CHECK(cudaGetLastError())
