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
extern "C" __global__ void __launch_bounds__(196) candidate1(float* __restrict__ Input, float* __restrict__ kernel, float* __restrict__ compute) {
  float conv2d_nchw[128];
  __shared__ float pad_temp_shared[6272];
  __shared__ float kernel_shared[1024];
  for (int ff_outer_inner_init = 0; ff_outer_inner_init < 2; ++ff_outer_inner_init) {
    for (int ff_inner_init = 0; ff_inner_init < 4; ++ff_inner_init) {
      for (int xx_inner_init = 0; xx_inner_init < 4; ++xx_inner_init) {
        conv2d_nchw[(((ff_outer_inner_init * 16) + (ff_inner_init * 4)) + xx_inner_init)] = 0.000000e+00f;
        conv2d_nchw[((((ff_outer_inner_init * 16) + (ff_inner_init * 4)) + xx_inner_init) + 32)] = 0.000000e+00f;
        conv2d_nchw[((((ff_outer_inner_init * 16) + (ff_inner_init * 4)) + xx_inner_init) + 64)] = 0.000000e+00f;
        conv2d_nchw[((((ff_outer_inner_init * 16) + (ff_inner_init * 4)) + xx_inner_init) + 96)] = 0.000000e+00f;
      }
    }
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 8; ++rc_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 32; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
      pad_temp_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 196) + ((int)threadIdx.x))] = Input[(((((rc_outer_outer * 50176) + ((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer >> 1) * 3136)) + ((((int)blockIdx.x) & 7) * 392)) + ((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer & 1) * 196)) + ((int)threadIdx.x))];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 < 6; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1) {
      if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 49) + (((int)threadIdx.x) >> 2)) < 256) {
        kernel_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 196) + ((int)threadIdx.x))] = kernel[(((((((int)blockIdx.x) >> 3) * 8192) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 49) + (((int)threadIdx.x) >> 2)) >> 2) * 128)) + (rc_outer_outer * 16)) + (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 196) + ((int)threadIdx.x)) & 15))];
      }
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 4; ++rc_outer_inner) {
      for (int ff_outer_inner = 0; ff_outer_inner < 2; ++ff_outer_inner) {
        for (int rc_inner = 0; rc_inner < 4; ++rc_inner) {
          for (int ff_inner = 0; ff_inner < 4; ++ff_inner) {
            for (int xx_inner = 0; xx_inner < 4; ++xx_inner) {
              conv2d_nchw[(((ff_outer_inner * 16) + (ff_inner * 4)) + xx_inner)] = (conv2d_nchw[(((ff_outer_inner * 16) + (ff_inner * 4)) + xx_inner)] + (pad_temp_shared[((((rc_outer_inner * 1568) + (rc_inner * 392)) + ((((int)threadIdx.x) % 98) * 4)) + xx_inner)] * kernel_shared[((((((((int)threadIdx.x) / 98) * 128) + (ff_outer_inner * 64)) + (ff_inner * 16)) + (rc_outer_inner * 4)) + rc_inner)]));
              conv2d_nchw[((((ff_outer_inner * 16) + (ff_inner * 4)) + xx_inner) + 32)] = (conv2d_nchw[((((ff_outer_inner * 16) + (ff_inner * 4)) + xx_inner) + 32)] + (pad_temp_shared[((((rc_outer_inner * 1568) + (rc_inner * 392)) + ((((int)threadIdx.x) % 98) * 4)) + xx_inner)] * kernel_shared[(((((((((int)threadIdx.x) / 98) * 128) + (ff_outer_inner * 64)) + (ff_inner * 16)) + (rc_outer_inner * 4)) + rc_inner) + 256)]));
              conv2d_nchw[((((ff_outer_inner * 16) + (ff_inner * 4)) + xx_inner) + 64)] = (conv2d_nchw[((((ff_outer_inner * 16) + (ff_inner * 4)) + xx_inner) + 64)] + (pad_temp_shared[((((rc_outer_inner * 1568) + (rc_inner * 392)) + ((((int)threadIdx.x) % 98) * 4)) + xx_inner)] * kernel_shared[(((((((((int)threadIdx.x) / 98) * 128) + (ff_outer_inner * 64)) + (ff_inner * 16)) + (rc_outer_inner * 4)) + rc_inner) + 512)]));
              conv2d_nchw[((((ff_outer_inner * 16) + (ff_inner * 4)) + xx_inner) + 96)] = (conv2d_nchw[((((ff_outer_inner * 16) + (ff_inner * 4)) + xx_inner) + 96)] + (pad_temp_shared[((((rc_outer_inner * 1568) + (rc_inner * 392)) + ((((int)threadIdx.x) % 98) * 4)) + xx_inner)] * kernel_shared[(((((((((int)threadIdx.x) / 98) * 128) + (ff_outer_inner * 64)) + (ff_inner * 16)) + (rc_outer_inner * 4)) + rc_inner) + 768)]));
            }
          }
        }
      }
    }
  }
  for (int i1_inner = 0; i1_inner < 8; ++i1_inner) {
    for (int i3_inner = 0; i3_inner < 4; ++i3_inner) {
      compute[(((((((((int)blockIdx.x) >> 3) * 200704) + ((((int)threadIdx.x) / 98) * 25088)) + (i1_inner * 3136)) + ((((int)blockIdx.x) & 7) * 392)) + ((((int)threadIdx.x) % 98) * 4)) + i3_inner)] = max(conv2d_nchw[((i1_inner * 4) + i3_inner)], 0.000000e+00f);
      compute[((((((((((int)blockIdx.x) >> 3) * 200704) + ((((int)threadIdx.x) / 98) * 25088)) + (i1_inner * 3136)) + ((((int)blockIdx.x) & 7) * 392)) + ((((int)threadIdx.x) % 98) * 4)) + i3_inner) + 50176)] = max(conv2d_nchw[(((i1_inner * 4) + i3_inner) + 32)], 0.000000e+00f);
      compute[((((((((((int)blockIdx.x) >> 3) * 200704) + ((((int)threadIdx.x) / 98) * 25088)) + (i1_inner * 3136)) + ((((int)blockIdx.x) & 7) * 392)) + ((((int)threadIdx.x) % 98) * 4)) + i3_inner) + 100352)] = max(conv2d_nchw[(((i1_inner * 4) + i3_inner) + 64)], 0.000000e+00f);
      compute[((((((((((int)blockIdx.x) >> 3) * 200704) + ((((int)threadIdx.x) / 98) * 25088)) + (i1_inner * 3136)) + ((((int)blockIdx.x) & 7) * 392)) + ((((int)threadIdx.x) % 98) * 4)) + i3_inner) + 150528)] = max(conv2d_nchw[(((i1_inner * 4) + i3_inner) + 96)], 0.000000e+00f);
    }
  }
}


