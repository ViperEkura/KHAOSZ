#pragma once
/* Native launch parameters and ring offsets; no framework or CUDA dependencies. */

namespace astrai {
namespace quant {

/* Scratch lines for the fused amax reduction, reset by the last block. */
inline constexpr int kFoldSlots = 32;
static_assert((kFoldSlots & (kFoldSlots - 1)) == 0,
              "the scratch is addressed by a power-of-two mask");

/* [hist n | scale0 | recip0 | amax | done | scratch | scale1 | recip1] */
struct RingLayout {
    int hist_len;

    /* Relative to the end of history; shared by host views and the kernel. */
    static constexpr int kScale = 0;
    static constexpr int kRecip = 1;
    static constexpr int kAmax = 2;
    static constexpr int kDone = 3;
    static constexpr int kScratch = 4;
    static constexpr int kAuxSlots = kScratch;

    // The composed ring has two pairs; standalone callers may allocate one.
    constexpr int size(int pairs = 2) const {
        return hist_len + kAuxSlots + kFoldSlots + 2 * (pairs - 1);
    }
    constexpr int scale(int pair = 0) const {
        return pair == 0 ? hist_len + kScale : hist_len + kAuxSlots + kFoldSlots;
    }
    constexpr int recip(int pair = 0) const { return scale(pair) + kRecip; }
    constexpr int amax() const { return hist_len + kAmax; }
    constexpr int done() const { return hist_len + kDone; }
    constexpr int scratch() const { return hist_len + kScratch; }
};

static_assert(RingLayout{0}.size(1) == 4 + kFoldSlots,
              "size(1) must stay the pre-double-buffer layout");

struct QuantParams {
    /* Contiguous input matrix extent. */
    const void* input_ptr;
    int rows;
    int cols;

    /* Requested output orientations and primary element strides. */
    void* output_ptr;
    void* output_transposed_ptr;
    int out_row_stride;
    int out_col_stride;

    /* Multiplier element strides in input coordinates; zero broadcasts an axis. */
    const float* inv_scale;
    int inv_scale_row_stride;
    int inv_scale_col_stride;

    /* Null ring skips history publication; publication targets may be outside it. */
    float* ring_state;
    int hist_len;
    int hist_idx;
    float* next_scale;
    float* next_inv_scale;

    /* next_scale = max(max(history) / quant_max / pow2_margin, 1e-12). */
    float quant_max;
    float pow2_margin;
};

} // namespace quant
} // namespace astrai
