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
extern "C" __global__ void __launch_bounds__(512) candidate4(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[4];
  __shared__ float pad_temp_shared[648];
  __shared__ float kernel_shared[9216];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 16; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = (((1 <= (((((int)blockIdx.x) / 7) * 8) + ((((int)threadIdx.x) % 81) / 9))) && (1 <= (((((int)blockIdx.x) % 7) * 8) + (((int)threadIdx.x) % 9)))) ? data[(((((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 81) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((int)threadIdx.x) % 81) / 9) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) % 9)) - 57)] : 0.000000e+00f);
    if (((int)threadIdx.x) < 136) {
      pad_temp_shared[(((int)threadIdx.x) + 512)] = (((1 <= (((((int)blockIdx.x) / 7) * 8) + (((((int)threadIdx.x) + 26) % 81) / 9))) && (1 <= (((((int)blockIdx.x) % 7) * 8) + ((((int)threadIdx.x) + 8) % 9)))) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 512) / 81) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + ((((((int)threadIdx.x) + 26) % 81) / 9) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + ((((int)threadIdx.x) + 8) % 9)) - 57)] : 0.000000e+00f);
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 18; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
      kernel_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 512) + ((int)threadIdx.x))] = kernel[((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 64) + (((int)threadIdx.x) >> 3)) / 9) * 1152) + (rc_outer_outer * 72)) + (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 512) + ((int)threadIdx.x)) % 72))];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 2; ++rc_outer_inner) {
      for (int rc_inner = 0; rc_inner < 4; ++rc_inner) {
        for (int ry_inner = 0; ry_inner < 3; ++ry_inner) {
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((rc_outer_inner * 324) + (rc_inner * 81)) + ((((int)threadIdx.x) & 3) * 18)) + (ry_inner * 9))] * kernel_shared[(((((((int)threadIdx.x) >> 2) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3))]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((rc_outer_inner * 324) + (rc_inner * 81)) + ((((int)threadIdx.x) & 3) * 18)) + (ry_inner * 9)) + 2)] * kernel_shared[(((((((int)threadIdx.x) >> 2) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3))]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((rc_outer_inner * 324) + (rc_inner * 81)) + ((((int)threadIdx.x) & 3) * 18)) + (ry_inner * 9)) + 4)] * kernel_shared[(((((((int)threadIdx.x) >> 2) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3))]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((rc_outer_inner * 324) + (rc_inner * 81)) + ((((int)threadIdx.x) & 3) * 18)) + (ry_inner * 9)) + 6)] * kernel_shared[(((((((int)threadIdx.x) >> 2) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3))]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((rc_outer_inner * 324) + (rc_inner * 81)) + ((((int)threadIdx.x) & 3) * 18)) + (ry_inner * 9)) + 1)] * kernel_shared[((((((((int)threadIdx.x) >> 2) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + 1)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((rc_outer_inner * 324) + (rc_inner * 81)) + ((((int)threadIdx.x) & 3) * 18)) + (ry_inner * 9)) + 3)] * kernel_shared[((((((((int)threadIdx.x) >> 2) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + 1)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((rc_outer_inner * 324) + (rc_inner * 81)) + ((((int)threadIdx.x) & 3) * 18)) + (ry_inner * 9)) + 5)] * kernel_shared[((((((((int)threadIdx.x) >> 2) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + 1)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((rc_outer_inner * 324) + (rc_inner * 81)) + ((((int)threadIdx.x) & 3) * 18)) + (ry_inner * 9)) + 7)] * kernel_shared[((((((((int)threadIdx.x) >> 2) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + 1)]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((rc_outer_inner * 324) + (rc_inner * 81)) + ((((int)threadIdx.x) & 3) * 18)) + (ry_inner * 9)) + 2)] * kernel_shared[((((((((int)threadIdx.x) >> 2) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + 2)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((rc_outer_inner * 324) + (rc_inner * 81)) + ((((int)threadIdx.x) & 3) * 18)) + (ry_inner * 9)) + 4)] * kernel_shared[((((((((int)threadIdx.x) >> 2) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + 2)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((rc_outer_inner * 324) + (rc_inner * 81)) + ((((int)threadIdx.x) & 3) * 18)) + (ry_inner * 9)) + 6)] * kernel_shared[((((((((int)threadIdx.x) >> 2) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + 2)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((rc_outer_inner * 324) + (rc_inner * 81)) + ((((int)threadIdx.x) & 3) * 18)) + (ry_inner * 9)) + 8)] * kernel_shared[((((((((int)threadIdx.x) >> 2) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + 2)]));
        }
      }
    }
  }
  conv2d_nchw[(((((((int)threadIdx.x) >> 2) * 784) + ((((int)blockIdx.x) / 7) * 112)) + ((((int)threadIdx.x) & 3) * 28)) + ((((int)blockIdx.x) % 7) * 4))] = conv2d_nchw_local[0];
  conv2d_nchw[((((((((int)threadIdx.x) >> 2) * 784) + ((((int)blockIdx.x) / 7) * 112)) + ((((int)threadIdx.x) & 3) * 28)) + ((((int)blockIdx.x) % 7) * 4)) + 1)] = conv2d_nchw_local[1];
  conv2d_nchw[((((((((int)threadIdx.x) >> 2) * 784) + ((((int)blockIdx.x) / 7) * 112)) + ((((int)threadIdx.x) & 3) * 28)) + ((((int)blockIdx.x) % 7) * 4)) + 2)] = conv2d_nchw_local[2];
  conv2d_nchw[((((((((int)threadIdx.x) >> 2) * 784) + ((((int)blockIdx.x) / 7) * 112)) + ((((int)threadIdx.x) & 3) * 28)) + ((((int)blockIdx.x) % 7) * 4)) + 3)] = conv2d_nchw_local[3];
}


