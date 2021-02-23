#if defined(__wasm__)

#include "mlasi.h"

// dilations are all 1. pad is 0, 1
// filter 3x3
void
MlasConvDepthwiseFloat_CHW(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    const float* /*Bias*/,
    float* Output,
    const float* Zeros
    )
{
    const float w00 = Filter[0];
    const float w01 = Filter[1];
    const float w02 = Filter[2];
    const float w10 = Filter[3];
    const float w11 = Filter[4];
    const float w12 = Filter[5];
    const float w20 = Filter[6];
    const float w21 = Filter[7];
    const float w22 = Filter[8];

    const size_t pad_top = Parameters->Padding[0];
    const size_t pad_left = Parameters->Padding[1];
    const size_t pad_bottom = Parameters->Padding[2];
    const size_t pad_right = Parameters->Padding[3];
    const size_t H = Parameters->InputShape[0];
    const size_t W = Parameters->InputShape[1];
    const size_t stride_h = Parameters->StrideShape[0];
    const size_t stride_w = Parameters->StrideShape[1];

    const float* row0 = (pad_top > 0) ? Zeros : (Input - pad_left);
    const float* row1 = (Input + (1 - pad_top) * W) - pad_left;
    const float* row2 = (H + pad_top <= 2) ? Zeros : (row1 + W);
    //const float bias = *Bias;

    // Could not hanle input W == 1
    for (size_t h = 0, out_row = Parameters->OutputShape[0]; out_row > 0; --out_row) {
        auto out_col = Parameters->OutputShape[1];

        if (pad_left == 1) {
            *Output++ = 
                w01 * row0[1] + w02 * row0[2] +
                w11 * row1[1] + w12 * row1[2] +
                w21 * row2[1] + w22 * row2[2]; // + *Bias;
            out_col--;
            row0 += stride_w;
            row1 += stride_w;
            row2 += stride_w;
        }

        for (; out_col > pad_right; out_col--) {
            *Output++ =
                w00 * row0[0] + w01 * row0[1] + w02 * row0[2] +
                w10 * row1[0] + w11 * row1[1] + w12 * row1[2] +
                w20 * row2[0] + w21 * row2[1] + w22 * row2[2]; // + *Bias;
            row0 += stride_w;
            row1 += stride_w;
            row2 += stride_w;
        }

        if (out_col == 1 /*pad_right == 1*/) {
            *Output++ =
                w00 * row0[0] + w01 * row0[1] +
                w10 * row1[0] + w11 * row1[1] +
                w20 * row2[0] + w21 * row2[1]; // + *Bias;
        }

        h += stride_h;
        row0 = (Input + (h - pad_top) * W) - pad_left;
        row1 = row0 + W;
        row2 = (h + 2 >= H + pad_top) ? Zeros : (row1 + W);
    }
}


#endif