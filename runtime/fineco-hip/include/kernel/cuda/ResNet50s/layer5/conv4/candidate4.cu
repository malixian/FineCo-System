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
extern "C" __global__ void __launch_bounds__(64) candidate4(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[7];
  __shared__ float pad_temp_shared[896];
  __shared__ float kernel_shared[8192];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 16; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = data[((((rc_outer_outer * 6272) + ((((int)threadIdx.x) / 7) * 49)) + ((((int)blockIdx.x) % 7) * 7)) + (((int)threadIdx.x) % 7))];
    pad_temp_shared[(((int)threadIdx.x) + 64)] = data[((((rc_outer_outer * 6272) + (((((int)threadIdx.x) + 64) / 7) * 49)) + ((((int)blockIdx.x) % 7) * 7)) + ((((int)threadIdx.x) + 1) % 7))];
    pad_temp_shared[(((int)threadIdx.x) + 128)] = data[((((rc_outer_outer * 6272) + (((((int)threadIdx.x) + 128) / 7) * 49)) + ((((int)blockIdx.x) % 7) * 7)) + ((((int)threadIdx.x) + 2) % 7))];
    pad_temp_shared[(((int)threadIdx.x) + 192)] = data[((((rc_outer_outer * 6272) + (((((int)threadIdx.x) + 192) / 7) * 49)) + ((((int)blockIdx.x) % 7) * 7)) + ((((int)threadIdx.x) + 3) % 7))];
    pad_temp_shared[(((int)threadIdx.x) + 256)] = data[((((rc_outer_outer * 6272) + (((((int)threadIdx.x) + 256) / 7) * 49)) + ((((int)blockIdx.x) % 7) * 7)) + ((((int)threadIdx.x) + 4) % 7))];
    pad_temp_shared[(((int)threadIdx.x) + 320)] = data[((((rc_outer_outer * 6272) + (((((int)threadIdx.x) + 320) / 7) * 49)) + ((((int)blockIdx.x) % 7) * 7)) + ((((int)threadIdx.x) + 5) % 7))];
    pad_temp_shared[(((int)threadIdx.x) + 384)] = data[((((rc_outer_outer * 6272) + (((((int)threadIdx.x) + 384) / 7) * 49)) + ((((int)blockIdx.x) % 7) * 7)) + ((((int)threadIdx.x) + 6) % 7))];
    pad_temp_shared[(((int)threadIdx.x) + 448)] = data[(((((rc_outer_outer * 6272) + ((((int)threadIdx.x) / 7) * 49)) + ((((int)blockIdx.x) % 7) * 7)) + (((int)threadIdx.x) % 7)) + 3136)];
    pad_temp_shared[(((int)threadIdx.x) + 512)] = data[((((rc_outer_outer * 6272) + (((((int)threadIdx.x) + 512) / 7) * 49)) + ((((int)blockIdx.x) % 7) * 7)) + ((((int)threadIdx.x) + 1) % 7))];
    pad_temp_shared[(((int)threadIdx.x) + 576)] = data[((((rc_outer_outer * 6272) + (((((int)threadIdx.x) + 576) / 7) * 49)) + ((((int)blockIdx.x) % 7) * 7)) + ((((int)threadIdx.x) + 2) % 7))];
    pad_temp_shared[(((int)threadIdx.x) + 640)] = data[((((rc_outer_outer * 6272) + (((((int)threadIdx.x) + 640) / 7) * 49)) + ((((int)blockIdx.x) % 7) * 7)) + ((((int)threadIdx.x) + 3) % 7))];
    pad_temp_shared[(((int)threadIdx.x) + 704)] = data[((((rc_outer_outer * 6272) + (((((int)threadIdx.x) + 704) / 7) * 49)) + ((((int)blockIdx.x) % 7) * 7)) + ((((int)threadIdx.x) + 4) % 7))];
    pad_temp_shared[(((int)threadIdx.x) + 768)] = data[((((rc_outer_outer * 6272) + (((((int)threadIdx.x) + 768) / 7) * 49)) + ((((int)blockIdx.x) % 7) * 7)) + ((((int)threadIdx.x) + 5) % 7))];
    pad_temp_shared[(((int)threadIdx.x) + 832)] = data[((((rc_outer_outer * 6272) + (((((int)threadIdx.x) + 832) / 7) * 49)) + ((((int)blockIdx.x) % 7) * 7)) + ((((int)threadIdx.x) + 6) % 7))];
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 128; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
      kernel_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 64) + ((int)threadIdx.x))] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer >> 1) * 2048)) + (rc_outer_outer * 128)) + ((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer & 1) * 64)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 64; ++rc_outer_inner) {
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(rc_outer_inner * 14)] * kernel_shared[((((int)threadIdx.x) * 128) + (rc_outer_inner * 2))]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((rc_outer_inner * 14) + 1)] * kernel_shared[((((int)threadIdx.x) * 128) + (rc_outer_inner * 2))]));
      conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((rc_outer_inner * 14) + 2)] * kernel_shared[((((int)threadIdx.x) * 128) + (rc_outer_inner * 2))]));
      conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((rc_outer_inner * 14) + 3)] * kernel_shared[((((int)threadIdx.x) * 128) + (rc_outer_inner * 2))]));
      conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[((rc_outer_inner * 14) + 4)] * kernel_shared[((((int)threadIdx.x) * 128) + (rc_outer_inner * 2))]));
      conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[((rc_outer_inner * 14) + 5)] * kernel_shared[((((int)threadIdx.x) * 128) + (rc_outer_inner * 2))]));
      conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[((rc_outer_inner * 14) + 6)] * kernel_shared[((((int)threadIdx.x) * 128) + (rc_outer_inner * 2))]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((rc_outer_inner * 14) + 7)] * kernel_shared[(((((int)threadIdx.x) * 128) + (rc_outer_inner * 2)) + 1)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((rc_outer_inner * 14) + 8)] * kernel_shared[(((((int)threadIdx.x) * 128) + (rc_outer_inner * 2)) + 1)]));
      conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((rc_outer_inner * 14) + 9)] * kernel_shared[(((((int)threadIdx.x) * 128) + (rc_outer_inner * 2)) + 1)]));
      conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((rc_outer_inner * 14) + 10)] * kernel_shared[(((((int)threadIdx.x) * 128) + (rc_outer_inner * 2)) + 1)]));
      conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[((rc_outer_inner * 14) + 11)] * kernel_shared[(((((int)threadIdx.x) * 128) + (rc_outer_inner * 2)) + 1)]));
      conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[((rc_outer_inner * 14) + 12)] * kernel_shared[(((((int)threadIdx.x) * 128) + (rc_outer_inner * 2)) + 1)]));
      conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[((rc_outer_inner * 14) + 13)] * kernel_shared[(((((int)threadIdx.x) * 128) + (rc_outer_inner * 2)) + 1)]));
    }
  }
  for (int xx_inner = 0; xx_inner < 7; ++xx_inner) {
    conv2d_nchw[(((((((int)blockIdx.x) / 7) * 3136) + (((int)threadIdx.x) * 49)) + ((((int)blockIdx.x) % 7) * 7)) + xx_inner)] = conv2d_nchw_local[xx_inner];
  }
}


