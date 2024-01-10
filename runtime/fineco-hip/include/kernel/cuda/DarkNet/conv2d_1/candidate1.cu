#include "hip/hip_runtime.h"

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void __launch_bounds__(256) candidate1(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ compute, float* __restrict__ bias) {
  float conv2d_nchw[224];
  __shared__ float pad_temp_shared[6156];
  __shared__ float kernel_shared[864];
  for (int ff_outer_inner_init = 0; ff_outer_inner_init < 2; ++ff_outer_inner_init) {
    for (int ff_inner_init = 0; ff_inner_init < 8; ++ff_inner_init) {
      conv2d_nchw[((ff_outer_inner_init * 56) + (ff_inner_init * 7))] = 0.000000e+00f;
      conv2d_nchw[(((ff_outer_inner_init * 56) + (ff_inner_init * 7)) + 112)] = 0.000000e+00f;
      conv2d_nchw[(((ff_outer_inner_init * 56) + (ff_inner_init * 7)) + 1)] = 0.000000e+00f;
      conv2d_nchw[(((ff_outer_inner_init * 56) + (ff_inner_init * 7)) + 113)] = 0.000000e+00f;
      conv2d_nchw[(((ff_outer_inner_init * 56) + (ff_inner_init * 7)) + 2)] = 0.000000e+00f;
      conv2d_nchw[(((ff_outer_inner_init * 56) + (ff_inner_init * 7)) + 114)] = 0.000000e+00f;
      conv2d_nchw[(((ff_outer_inner_init * 56) + (ff_inner_init * 7)) + 3)] = 0.000000e+00f;
      conv2d_nchw[(((ff_outer_inner_init * 56) + (ff_inner_init * 7)) + 115)] = 0.000000e+00f;
      conv2d_nchw[(((ff_outer_inner_init * 56) + (ff_inner_init * 7)) + 4)] = 0.000000e+00f;
      conv2d_nchw[(((ff_outer_inner_init * 56) + (ff_inner_init * 7)) + 116)] = 0.000000e+00f;
      conv2d_nchw[(((ff_outer_inner_init * 56) + (ff_inner_init * 7)) + 5)] = 0.000000e+00f;
      conv2d_nchw[(((ff_outer_inner_init * 56) + (ff_inner_init * 7)) + 117)] = 0.000000e+00f;
      conv2d_nchw[(((ff_outer_inner_init * 56) + (ff_inner_init * 7)) + 6)] = 0.000000e+00f;
      conv2d_nchw[(((ff_outer_inner_init * 56) + (ff_inner_init * 7)) + 118)] = 0.000000e+00f;
    }
  }
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 25; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
    if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 64) + (((int)threadIdx.x) >> 2)) < 1539) {
      pad_temp_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 256) + ((int)threadIdx.x))] = (((((1 <= (((((int)blockIdx.x) / 14) * 112) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 128) + (((int)threadIdx.x) >> 1)) % 1026) / 9))) && ((((((int)blockIdx.x) / 14) * 112) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 128) + (((int)threadIdx.x) >> 1)) % 1026) / 9)) < 225)) && (1 <= (((((int)blockIdx.x) % 14) * 16) + (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 256) + ((int)threadIdx.x)) % 18)))) && ((((((int)blockIdx.x) % 14) * 16) + (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 256) + ((int)threadIdx.x)) % 18)) < 225)) ? data[(((((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 64) + (((int)threadIdx.x) >> 2)) / 513) * 50176) + ((((int)blockIdx.x) / 14) * 25088)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 128) + (((int)threadIdx.x) >> 1)) % 1026) / 9) * 224)) + ((((int)blockIdx.x) % 14) * 16)) + (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 256) + ((int)threadIdx.x)) % 18)) - 225)] : 0.000000e+00f);
    }
  }
  kernel_shared[((int)threadIdx.x)] = kernel[((int)threadIdx.x)];
  kernel_shared[(((int)threadIdx.x) + 256)] = kernel[(((int)threadIdx.x) + 256)];
  kernel_shared[(((int)threadIdx.x) + 512)] = kernel[(((int)threadIdx.x) + 512)];
  if (((int)threadIdx.x) < 96) {
    kernel_shared[(((int)threadIdx.x) + 768)] = kernel[(((int)threadIdx.x) + 768)];
  }
  __syncthreads();
  for (int rx_outer_inner = 0; rx_outer_inner < 3; ++rx_outer_inner) {
    for (int ff_outer_inner = 0; ff_outer_inner < 2; ++ff_outer_inner) {
      for (int rc_inner = 0; rc_inner < 3; ++rc_inner) {
        for (int ry_inner = 0; ry_inner < 3; ++ry_inner) {
          for (int ff_inner = 0; ff_inner < 8; ++ff_inner) {
            conv2d_nchw[((ff_outer_inner * 56) + (ff_inner * 7))] = (conv2d_nchw[((ff_outer_inner * 56) + (ff_inner * 7))] + (pad_temp_shared[(((((rc_inner * 2052) + ((((int)threadIdx.x) >> 4) * 126)) + (ry_inner * 18)) + rx_outer_inner) + (((int)threadIdx.x) & 15))] * kernel_shared[(((((ff_outer_inner * 216) + (ff_inner * 27)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_outer_inner)]));
            conv2d_nchw[(((ff_outer_inner * 56) + (ff_inner * 7)) + 112)] = (conv2d_nchw[(((ff_outer_inner * 56) + (ff_inner * 7)) + 112)] + (pad_temp_shared[(((((rc_inner * 2052) + ((((int)threadIdx.x) >> 4) * 126)) + (ry_inner * 18)) + rx_outer_inner) + (((int)threadIdx.x) & 15))] * kernel_shared[((((((ff_outer_inner * 216) + (ff_inner * 27)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_outer_inner) + 432)]));
            conv2d_nchw[(((ff_outer_inner * 56) + (ff_inner * 7)) + 1)] = (conv2d_nchw[(((ff_outer_inner * 56) + (ff_inner * 7)) + 1)] + (pad_temp_shared[((((((rc_inner * 2052) + ((((int)threadIdx.x) >> 4) * 126)) + (ry_inner * 18)) + rx_outer_inner) + (((int)threadIdx.x) & 15)) + 18)] * kernel_shared[(((((ff_outer_inner * 216) + (ff_inner * 27)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_outer_inner)]));
            conv2d_nchw[(((ff_outer_inner * 56) + (ff_inner * 7)) + 113)] = (conv2d_nchw[(((ff_outer_inner * 56) + (ff_inner * 7)) + 113)] + (pad_temp_shared[((((((rc_inner * 2052) + ((((int)threadIdx.x) >> 4) * 126)) + (ry_inner * 18)) + rx_outer_inner) + (((int)threadIdx.x) & 15)) + 18)] * kernel_shared[((((((ff_outer_inner * 216) + (ff_inner * 27)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_outer_inner) + 432)]));
            conv2d_nchw[(((ff_outer_inner * 56) + (ff_inner * 7)) + 2)] = (conv2d_nchw[(((ff_outer_inner * 56) + (ff_inner * 7)) + 2)] + (pad_temp_shared[((((((rc_inner * 2052) + ((((int)threadIdx.x) >> 4) * 126)) + (ry_inner * 18)) + rx_outer_inner) + (((int)threadIdx.x) & 15)) + 36)] * kernel_shared[(((((ff_outer_inner * 216) + (ff_inner * 27)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_outer_inner)]));
            conv2d_nchw[(((ff_outer_inner * 56) + (ff_inner * 7)) + 114)] = (conv2d_nchw[(((ff_outer_inner * 56) + (ff_inner * 7)) + 114)] + (pad_temp_shared[((((((rc_inner * 2052) + ((((int)threadIdx.x) >> 4) * 126)) + (ry_inner * 18)) + rx_outer_inner) + (((int)threadIdx.x) & 15)) + 36)] * kernel_shared[((((((ff_outer_inner * 216) + (ff_inner * 27)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_outer_inner) + 432)]));
            conv2d_nchw[(((ff_outer_inner * 56) + (ff_inner * 7)) + 3)] = (conv2d_nchw[(((ff_outer_inner * 56) + (ff_inner * 7)) + 3)] + (pad_temp_shared[((((((rc_inner * 2052) + ((((int)threadIdx.x) >> 4) * 126)) + (ry_inner * 18)) + rx_outer_inner) + (((int)threadIdx.x) & 15)) + 54)] * kernel_shared[(((((ff_outer_inner * 216) + (ff_inner * 27)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_outer_inner)]));
            conv2d_nchw[(((ff_outer_inner * 56) + (ff_inner * 7)) + 115)] = (conv2d_nchw[(((ff_outer_inner * 56) + (ff_inner * 7)) + 115)] + (pad_temp_shared[((((((rc_inner * 2052) + ((((int)threadIdx.x) >> 4) * 126)) + (ry_inner * 18)) + rx_outer_inner) + (((int)threadIdx.x) & 15)) + 54)] * kernel_shared[((((((ff_outer_inner * 216) + (ff_inner * 27)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_outer_inner) + 432)]));
            conv2d_nchw[(((ff_outer_inner * 56) + (ff_inner * 7)) + 4)] = (conv2d_nchw[(((ff_outer_inner * 56) + (ff_inner * 7)) + 4)] + (pad_temp_shared[((((((rc_inner * 2052) + ((((int)threadIdx.x) >> 4) * 126)) + (ry_inner * 18)) + rx_outer_inner) + (((int)threadIdx.x) & 15)) + 72)] * kernel_shared[(((((ff_outer_inner * 216) + (ff_inner * 27)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_outer_inner)]));
            conv2d_nchw[(((ff_outer_inner * 56) + (ff_inner * 7)) + 116)] = (conv2d_nchw[(((ff_outer_inner * 56) + (ff_inner * 7)) + 116)] + (pad_temp_shared[((((((rc_inner * 2052) + ((((int)threadIdx.x) >> 4) * 126)) + (ry_inner * 18)) + rx_outer_inner) + (((int)threadIdx.x) & 15)) + 72)] * kernel_shared[((((((ff_outer_inner * 216) + (ff_inner * 27)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_outer_inner) + 432)]));
            conv2d_nchw[(((ff_outer_inner * 56) + (ff_inner * 7)) + 5)] = (conv2d_nchw[(((ff_outer_inner * 56) + (ff_inner * 7)) + 5)] + (pad_temp_shared[((((((rc_inner * 2052) + ((((int)threadIdx.x) >> 4) * 126)) + (ry_inner * 18)) + rx_outer_inner) + (((int)threadIdx.x) & 15)) + 90)] * kernel_shared[(((((ff_outer_inner * 216) + (ff_inner * 27)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_outer_inner)]));
            conv2d_nchw[(((ff_outer_inner * 56) + (ff_inner * 7)) + 117)] = (conv2d_nchw[(((ff_outer_inner * 56) + (ff_inner * 7)) + 117)] + (pad_temp_shared[((((((rc_inner * 2052) + ((((int)threadIdx.x) >> 4) * 126)) + (ry_inner * 18)) + rx_outer_inner) + (((int)threadIdx.x) & 15)) + 90)] * kernel_shared[((((((ff_outer_inner * 216) + (ff_inner * 27)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_outer_inner) + 432)]));
            conv2d_nchw[(((ff_outer_inner * 56) + (ff_inner * 7)) + 6)] = (conv2d_nchw[(((ff_outer_inner * 56) + (ff_inner * 7)) + 6)] + (pad_temp_shared[((((((rc_inner * 2052) + ((((int)threadIdx.x) >> 4) * 126)) + (ry_inner * 18)) + rx_outer_inner) + (((int)threadIdx.x) & 15)) + 108)] * kernel_shared[(((((ff_outer_inner * 216) + (ff_inner * 27)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_outer_inner)]));
            conv2d_nchw[(((ff_outer_inner * 56) + (ff_inner * 7)) + 118)] = (conv2d_nchw[(((ff_outer_inner * 56) + (ff_inner * 7)) + 118)] + (pad_temp_shared[((((((rc_inner * 2052) + ((((int)threadIdx.x) >> 4) * 126)) + (ry_inner * 18)) + rx_outer_inner) + (((int)threadIdx.x) & 15)) + 108)] * kernel_shared[((((((ff_outer_inner * 216) + (ff_inner * 27)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_outer_inner) + 432)]));
          }
        }
      }
    }
  }
  for (int i1_inner = 0; i1_inner < 16; ++i1_inner) {
    for (int i2_inner = 0; i2_inner < 7; ++i2_inner) {
      compute[((((((i1_inner * 50176) + ((((int)blockIdx.x) / 14) * 25088)) + ((((int)threadIdx.x) >> 4) * 1568)) + (i2_inner * 224)) + ((((int)blockIdx.x) % 14) * 16)) + (((int)threadIdx.x) & 15))] = max((conv2d_nchw[((i1_inner * 7) + i2_inner)] + bias[i1_inner]), 0.000000e+00f);
      compute[(((((((i1_inner * 50176) + ((((int)blockIdx.x) / 14) * 25088)) + ((((int)threadIdx.x) >> 4) * 1568)) + (i2_inner * 224)) + ((((int)blockIdx.x) % 14) * 16)) + (((int)threadIdx.x) & 15)) + 802816)] = max((conv2d_nchw[(((i1_inner * 7) + i2_inner) + 112)] + bias[(i1_inner + 16)]), 0.000000e+00f);
    }
  }
}


