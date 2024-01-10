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
extern "C" __global__ void __launch_bounds__(512) candidate2(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[7];
  __shared__ float pad_temp_shared[112];
  __shared__ float kernel_shared[8192];
  for (int xx_c_inner_init = 0; xx_c_inner_init < 7; ++xx_c_inner_init) {
    conv2d_nchw_local[xx_c_inner_init] = 0.000000e+00f;
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 32; ++rc_outer_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 112) {
      pad_temp_shared[((int)threadIdx.x)] = data[((((rc_outer_outer * 784) + ((((int)threadIdx.x) / 7) * 49)) + ((((int)blockIdx.x) % 7) * 7)) + (((int)threadIdx.x) % 7))];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 16; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
      kernel_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 512) + ((int)threadIdx.x))] = kernel[((((((((int)blockIdx.x) / 7) * 262144) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 16384)) + ((((int)threadIdx.x) >> 4) * 512)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15))];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 16; ++rc_outer_inner) {
      for (int xx_c_inner = 0; xx_c_inner < 7; ++xx_c_inner) {
        conv2d_nchw_local[xx_c_inner] = (conv2d_nchw_local[xx_c_inner] + (pad_temp_shared[((rc_outer_inner * 7) + xx_c_inner)] * kernel_shared[((((int)threadIdx.x) * 16) + rc_outer_inner)]));
      }
    }
  }
  for (int xx_inner = 0; xx_inner < 7; ++xx_inner) {
    conv2d_nchw[(((((((int)blockIdx.x) / 7) * 25088) + (((int)threadIdx.x) * 49)) + ((((int)blockIdx.x) % 7) * 7)) + xx_inner)] = conv2d_nchw_local[xx_inner];
  }
}


