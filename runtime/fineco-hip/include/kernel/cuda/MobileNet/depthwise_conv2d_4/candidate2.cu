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
extern "C" __global__ void __launch_bounds__(512) candidate2(float* __restrict__ Input, float* __restrict__ kernel, float* __restrict__ compute) {
  float DepthwiseConv2d[4];
  __shared__ float PaddedInput_shared[10368];
  __shared__ float kernel_shared[1152];
  for (int i_outer_inner_init = 0; i_outer_inner_init < 4; ++i_outer_inner_init) {
    DepthwiseConv2d[i_outer_inner_init] = 0.000000e+00f;
  }
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 21; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
    if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 4) + (((int)threadIdx.x) >> 7)) < 81) {
      PaddedInput_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 512) + ((int)threadIdx.x))] = (((1 <= (((((int)blockIdx.x) / 7) * 8) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 512) + ((int)threadIdx.x)) % 81) / 9))) && (1 <= (((((int)blockIdx.x) % 7) * 8) + (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 512) + ((int)threadIdx.x)) % 9)))) ? Input[(((((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 512) + ((int)threadIdx.x)) / 81) * 3136) + ((((int)blockIdx.x) / 7) * 448)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 512) + ((int)threadIdx.x)) % 81) / 9) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 512) + ((int)threadIdx.x)) % 9)) - 57)] : 0.000000e+00f);
    }
  }
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1) {
    if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 4) + (((int)threadIdx.x) >> 7)) < 9) {
      kernel_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 512) + ((int)threadIdx.x))] = kernel[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 512) + ((int)threadIdx.x))];
    }
  }
  __syncthreads();
  for (int di_outer_inner = 0; di_outer_inner < 3; ++di_outer_inner) {
    for (int i_outer_inner = 0; i_outer_inner < 4; ++i_outer_inner) {
      for (int dj_inner = 0; dj_inner < 3; ++dj_inner) {
        DepthwiseConv2d[i_outer_inner] = (DepthwiseConv2d[i_outer_inner] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 2) * 81) + (i_outer_inner * 18)) + (di_outer_inner * 9)) + ((((int)threadIdx.x) & 3) * 2)) + dj_inner)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 9) + (di_outer_inner * 3)) + dj_inner)]));
      }
    }
  }
  for (int i2_inner = 0; i2_inner < 4; ++i2_inner) {
    compute[((((((((int)threadIdx.x) >> 2) * 784) + ((((int)blockIdx.x) / 7) * 112)) + (i2_inner * 28)) + ((((int)blockIdx.x) % 7) * 4)) + (((int)threadIdx.x) & 3))] = max(DepthwiseConv2d[i2_inner], 0.000000e+00f);
  }
}


