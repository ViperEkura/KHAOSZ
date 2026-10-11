#pragma once
/* Device facts supplied to native planners; no CUDA runtime dependency. */
#include <cstdint>

namespace astrai {

struct DeviceFacts {
    int sms;
    int smem_max;
    int smem_per_sm;
    int regs_per_sm;
    int64_t l2_bytes;
    int cc = 0;
    int ordinal = 0;
    int threads_per_sm = 0;
};

} // namespace astrai
