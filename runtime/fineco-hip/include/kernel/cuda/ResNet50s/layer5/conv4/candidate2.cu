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
extern "C" __global__ void __launch_bounds__(128) candidate2(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[7];
  __shared__ float pad_temp_shared[448];
  __shared__ float kernel_shared[8192];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 32; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = data[(((rc_outer_outer * 3136) + (((int)threadIdx.x) * 7)) + (((int)blockIdx.x) % 7))];
    pad_temp_shared[(((int)threadIdx.x) + 128)] = data[((((rc_outer_outer * 3136) + (((int)threadIdx.x) * 7)) + (((int)blockIdx.x) % 7)) + 896)];
    pad_temp_shared[(((int)threadIdx.x) + 256)] = data[((((rc_outer_outer * 3136) + (((int)threadIdx.x) * 7)) + (((int)blockIdx.x) % 7)) + 1792)];
    if (((int)threadIdx.x) < 64) {
      pad_temp_shared[(((int)threadIdx.x) + 384)] = data[((((rc_outer_outer * 3136) + (((int)threadIdx.x) * 7)) + (((int)blockIdx.x) % 7)) + 2688)];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 64; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
      kernel_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 128) + ((int)threadIdx.x))] = kernel[((((((((int)blockIdx.x) / 7) * 262144) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 4096)) + ((((int)threadIdx.x) >> 6) * 2048)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63))];
    }
    __syncthreads();
    for (int yy_c_outer_inner = 0; yy_c_outer_inner < 7; ++yy_c_outer_inner) {
      for (int rc_inner = 0; rc_inner < 64; ++rc_inner) {
        conv2d_nchw_local[yy_c_outer_inner] = (conv2d_nchw_local[yy_c_outer_inner] + (pad_temp_shared[((rc_inner * 7) + yy_c_outer_inner)] * kernel_shared[((((int)threadIdx.x) * 64) + rc_inner)]));
      }
    }
  }
  for (int yy_inner = 0; yy_inner < 7; ++yy_inner) {
    conv2d_nchw[(((((((int)blockIdx.x) / 7) * 6272) + (((int)threadIdx.x) * 49)) + (yy_inner * 7)) + (((int)blockIdx.x) % 7))] = conv2d_nchw_local[yy_inner];
  }
}


