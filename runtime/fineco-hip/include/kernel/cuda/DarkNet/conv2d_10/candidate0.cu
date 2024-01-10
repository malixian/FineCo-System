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
extern "C" __global__ void __launch_bounds__(112) candidate0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ compute, float* __restrict__ bias) {
  float conv2d_nchw[4];
  __shared__ float pad_temp_shared[896];
  __shared__ float kernel_shared[512];
  for (int ff_inner_init = 0; ff_inner_init < 2; ++ff_inner_init) {
    for (int yy_inner_init = 0; yy_inner_init < 2; ++yy_inner_init) {
      conv2d_nchw[((ff_inner_init * 2) + yy_inner_init)] = 0.000000e+00f;
    }
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 16; ++rc_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
      *(float4*)(pad_temp_shared + ((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 448) + (((int)threadIdx.x) * 4))) = *(float4*)(data + (((((rc_outer_outer * 6272) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 3136)) + ((((int)threadIdx.x) / 7) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) % 7) * 4)));
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1) {
      if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 7) + (((int)threadIdx.x) >> 4)) < 32) {
        kernel_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 112) + ((int)threadIdx.x))] = kernel[(((((((int)blockIdx.x) / 7) * 8192) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 7) + (((int)threadIdx.x) >> 4)) >> 1) * 512)) + (rc_outer_outer * 32)) + (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 112) + ((int)threadIdx.x)) & 31))];
      }
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 32; ++rc_outer_inner) {
      for (int ff_inner = 0; ff_inner < 2; ++ff_inner) {
        for (int yy_inner = 0; yy_inner < 2; ++yy_inner) {
          conv2d_nchw[((ff_inner * 2) + yy_inner)] = (conv2d_nchw[((ff_inner * 2) + yy_inner)] + (pad_temp_shared[(((rc_outer_inner * 28) + (yy_inner * 14)) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 14) * 64) + (ff_inner * 32)) + rc_outer_inner)]));
        }
      }
    }
  }
  for (int i1_inner = 0; i1_inner < 2; ++i1_inner) {
    for (int i2_inner = 0; i2_inner < 2; ++i2_inner) {
      compute[(((((((((int)blockIdx.x) / 7) * 3136) + ((((int)threadIdx.x) / 14) * 392)) + (i1_inner * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (i2_inner * 14)) + (((int)threadIdx.x) % 14))] = max((conv2d_nchw[((i1_inner * 2) + i2_inner)] + bias[((((((int)blockIdx.x) / 7) * 16) + ((((int)threadIdx.x) / 14) * 2)) + i1_inner)]), 0.000000e+00f);
    }
  }
}


