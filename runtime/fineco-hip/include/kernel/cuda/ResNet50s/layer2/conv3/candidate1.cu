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
extern "C" __global__ void __launch_bounds__(224) candidate1(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[224];
  __shared__ float pad_temp_shared[6272];
  __shared__ float kernel_shared[512];
  for (int ff_c_outer_inner_init = 0; ff_c_outer_inner_init < 2; ++ff_c_outer_inner_init) {
    for (int ff_c_inner_init = 0; ff_c_inner_init < 2; ++ff_c_inner_init) {
      for (int yy_c_inner_init = 0; yy_c_inner_init < 14; ++yy_c_inner_init) {
        conv2d_nchw_local[(((ff_c_outer_inner_init * 28) + (ff_c_inner_init * 14)) + yy_c_inner_init)] = 0.000000e+00f;
        conv2d_nchw_local[((((ff_c_outer_inner_init * 28) + (ff_c_inner_init * 14)) + yy_c_inner_init) + 56)] = 0.000000e+00f;
        conv2d_nchw_local[((((ff_c_outer_inner_init * 28) + (ff_c_inner_init * 14)) + yy_c_inner_init) + 112)] = 0.000000e+00f;
        conv2d_nchw_local[((((ff_c_outer_inner_init * 28) + (ff_c_inner_init * 14)) + yy_c_inner_init) + 168)] = 0.000000e+00f;
      }
    }
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 8; ++rc_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 28; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
      pad_temp_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 224) + ((int)threadIdx.x))] = data[(((((rc_outer_outer * 25088) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 896)) + ((((int)threadIdx.x) / 14) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14))];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1) {
      if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 7) + (((int)threadIdx.x) >> 5)) < 16) {
        kernel_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 224) + ((int)threadIdx.x))] = kernel[((((((((int)blockIdx.x) >> 2) * 4096) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 1792)) + ((((int)threadIdx.x) >> 3) * 64)) + (rc_outer_outer * 8)) + (((int)threadIdx.x) & 7))];
      }
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 4; ++rc_outer_inner) {
      for (int ff_c_outer_inner = 0; ff_c_outer_inner < 2; ++ff_c_outer_inner) {
        for (int rc_inner = 0; rc_inner < 2; ++rc_inner) {
          for (int ff_c_inner = 0; ff_c_inner < 2; ++ff_c_inner) {
            for (int yy_c_inner = 0; yy_c_inner < 14; ++yy_c_inner) {
              conv2d_nchw_local[(((ff_c_outer_inner * 28) + (ff_c_inner * 14)) + yy_c_inner)] = (conv2d_nchw_local[(((ff_c_outer_inner * 28) + (ff_c_inner * 14)) + yy_c_inner)] + (pad_temp_shared[(((((rc_outer_inner * 1568) + (rc_inner * 784)) + (((((int)threadIdx.x) % 14) / 7) * 196)) + (yy_c_inner * 14)) + (((int)threadIdx.x) % 7))] * kernel_shared[((((((((int)threadIdx.x) / 14) * 32) + (ff_c_outer_inner * 16)) + (ff_c_inner * 8)) + (rc_outer_inner * 2)) + rc_inner)]));
              conv2d_nchw_local[((((ff_c_outer_inner * 28) + (ff_c_inner * 14)) + yy_c_inner) + 56)] = (conv2d_nchw_local[((((ff_c_outer_inner * 28) + (ff_c_inner * 14)) + yy_c_inner) + 56)] + (pad_temp_shared[((((((rc_outer_inner * 1568) + (rc_inner * 784)) + (((((int)threadIdx.x) % 14) / 7) * 196)) + (yy_c_inner * 14)) + (((int)threadIdx.x) % 7)) + 7)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 32) + (ff_c_outer_inner * 16)) + (ff_c_inner * 8)) + (rc_outer_inner * 2)) + rc_inner)]));
              conv2d_nchw_local[((((ff_c_outer_inner * 28) + (ff_c_inner * 14)) + yy_c_inner) + 112)] = (conv2d_nchw_local[((((ff_c_outer_inner * 28) + (ff_c_inner * 14)) + yy_c_inner) + 112)] + (pad_temp_shared[((((((rc_outer_inner * 1568) + (rc_inner * 784)) + (((((int)threadIdx.x) % 14) / 7) * 196)) + (yy_c_inner * 14)) + (((int)threadIdx.x) % 7)) + 392)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 32) + (ff_c_outer_inner * 16)) + (ff_c_inner * 8)) + (rc_outer_inner * 2)) + rc_inner)]));
              conv2d_nchw_local[((((ff_c_outer_inner * 28) + (ff_c_inner * 14)) + yy_c_inner) + 168)] = (conv2d_nchw_local[((((ff_c_outer_inner * 28) + (ff_c_inner * 14)) + yy_c_inner) + 168)] + (pad_temp_shared[((((((rc_outer_inner * 1568) + (rc_inner * 784)) + (((((int)threadIdx.x) % 14) / 7) * 196)) + (yy_c_inner * 14)) + (((int)threadIdx.x) % 7)) + 399)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 32) + (ff_c_outer_inner * 16)) + (ff_c_inner * 8)) + (rc_outer_inner * 2)) + rc_inner)]));
            }
          }
        }
      }
    }
  }
  for (int ff_inner = 0; ff_inner < 4; ++ff_inner) {
    for (int yy_inner = 0; yy_inner < 14; ++yy_inner) {
      conv2d_nchw[((((((((((int)blockIdx.x) >> 2) * 200704) + ((((int)threadIdx.x) / 14) * 12544)) + (ff_inner * 3136)) + (((((int)threadIdx.x) % 14) / 7) * 784)) + (yy_inner * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 7))] = conv2d_nchw_local[((ff_inner * 14) + yy_inner)];
      conv2d_nchw[(((((((((((int)blockIdx.x) >> 2) * 200704) + ((((int)threadIdx.x) / 14) * 12544)) + (ff_inner * 3136)) + (((((int)threadIdx.x) % 14) / 7) * 784)) + (yy_inner * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 7)) + 7)] = conv2d_nchw_local[(((ff_inner * 14) + yy_inner) + 56)];
      conv2d_nchw[(((((((((((int)blockIdx.x) >> 2) * 200704) + ((((int)threadIdx.x) / 14) * 12544)) + (ff_inner * 3136)) + (((((int)threadIdx.x) % 14) / 7) * 784)) + (yy_inner * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 7)) + 1568)] = conv2d_nchw_local[(((ff_inner * 14) + yy_inner) + 112)];
      conv2d_nchw[(((((((((((int)blockIdx.x) >> 2) * 200704) + ((((int)threadIdx.x) / 14) * 12544)) + (ff_inner * 3136)) + (((((int)threadIdx.x) % 14) / 7) * 784)) + (yy_inner * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 7)) + 1575)] = conv2d_nchw_local[(((ff_inner * 14) + yy_inner) + 168)];
    }
  }
}


