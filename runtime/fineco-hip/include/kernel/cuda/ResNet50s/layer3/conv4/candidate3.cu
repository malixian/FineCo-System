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
extern "C" __global__ void __launch_bounds__(98) candidate3(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[32];
  __shared__ float pad_temp_shared[6272];
  __shared__ float kernel_shared[512];
  for (int ff_c_inner_init = 0; ff_c_inner_init < 2; ++ff_c_inner_init) {
    conv2d_nchw_local[(ff_c_inner_init * 2)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_inner_init * 2) + 4)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_inner_init * 2) + 8)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_inner_init * 2) + 12)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_inner_init * 2) + 16)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_inner_init * 2) + 20)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_inner_init * 2) + 24)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_inner_init * 2) + 28)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_inner_init * 2) + 1)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_inner_init * 2) + 5)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_inner_init * 2) + 9)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_inner_init * 2) + 13)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_inner_init * 2) + 17)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_inner_init * 2) + 21)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_inner_init * 2) + 25)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_inner_init * 2) + 29)] = 0.000000e+00f;
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 16; ++rc_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 64; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
      pad_temp_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 98) + ((int)threadIdx.x))] = data[(((((((rc_outer_outer * 25088) + ((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer >> 1) * 784)) + (((((int)blockIdx.x) & 3) >> 1) * 392)) + ((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer & 1) * 196)) + ((((int)threadIdx.x) / 14) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((int)threadIdx.x) % 14))];
    }
    kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) >> 2) * 8192) + ((((int)threadIdx.x) >> 5) * 512)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31))];
    kernel_shared[(((int)threadIdx.x) + 98)] = kernel[(((((((int)blockIdx.x) >> 2) * 8192) + (((((int)threadIdx.x) + 98) >> 5) * 512)) + (rc_outer_outer * 32)) + ((((int)threadIdx.x) + 2) & 31))];
    kernel_shared[(((int)threadIdx.x) + 196)] = kernel[(((((((int)blockIdx.x) >> 2) * 8192) + (((((int)threadIdx.x) + 196) >> 5) * 512)) + (rc_outer_outer * 32)) + ((((int)threadIdx.x) + 4) & 31))];
    kernel_shared[(((int)threadIdx.x) + 294)] = kernel[(((((((int)blockIdx.x) >> 2) * 8192) + (((((int)threadIdx.x) + 294) >> 5) * 512)) + (rc_outer_outer * 32)) + ((((int)threadIdx.x) + 6) & 31))];
    kernel_shared[(((int)threadIdx.x) + 392)] = kernel[(((((((int)blockIdx.x) >> 2) * 8192) + (((((int)threadIdx.x) + 392) >> 5) * 512)) + (rc_outer_outer * 32)) + ((((int)threadIdx.x) + 8) & 31))];
    if (((int)threadIdx.x) < 22) {
      kernel_shared[(((int)threadIdx.x) + 490)] = kernel[(((((((int)blockIdx.x) >> 2) * 8192) + (((((int)threadIdx.x) + 490) >> 5) * 512)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) + 10))];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 8; ++rc_outer_inner) {
      for (int rc_inner = 0; rc_inner < 4; ++rc_inner) {
        for (int ff_c_inner = 0; ff_c_inner < 2; ++ff_c_inner) {
          conv2d_nchw_local[(ff_c_inner * 2)] = (conv2d_nchw_local[(ff_c_inner * 2)] + (pad_temp_shared[(((rc_outer_inner * 784) + (rc_inner * 196)) + (((int)threadIdx.x) * 2))] * kernel_shared[(((ff_c_inner * 32) + (rc_outer_inner * 4)) + rc_inner)]));
          conv2d_nchw_local[((ff_c_inner * 2) + 4)] = (conv2d_nchw_local[((ff_c_inner * 2) + 4)] + (pad_temp_shared[(((rc_outer_inner * 784) + (rc_inner * 196)) + (((int)threadIdx.x) * 2))] * kernel_shared[((((ff_c_inner * 32) + (rc_outer_inner * 4)) + rc_inner) + 64)]));
          conv2d_nchw_local[((ff_c_inner * 2) + 8)] = (conv2d_nchw_local[((ff_c_inner * 2) + 8)] + (pad_temp_shared[(((rc_outer_inner * 784) + (rc_inner * 196)) + (((int)threadIdx.x) * 2))] * kernel_shared[((((ff_c_inner * 32) + (rc_outer_inner * 4)) + rc_inner) + 128)]));
          conv2d_nchw_local[((ff_c_inner * 2) + 12)] = (conv2d_nchw_local[((ff_c_inner * 2) + 12)] + (pad_temp_shared[(((rc_outer_inner * 784) + (rc_inner * 196)) + (((int)threadIdx.x) * 2))] * kernel_shared[((((ff_c_inner * 32) + (rc_outer_inner * 4)) + rc_inner) + 192)]));
          conv2d_nchw_local[((ff_c_inner * 2) + 16)] = (conv2d_nchw_local[((ff_c_inner * 2) + 16)] + (pad_temp_shared[(((rc_outer_inner * 784) + (rc_inner * 196)) + (((int)threadIdx.x) * 2))] * kernel_shared[((((ff_c_inner * 32) + (rc_outer_inner * 4)) + rc_inner) + 256)]));
          conv2d_nchw_local[((ff_c_inner * 2) + 20)] = (conv2d_nchw_local[((ff_c_inner * 2) + 20)] + (pad_temp_shared[(((rc_outer_inner * 784) + (rc_inner * 196)) + (((int)threadIdx.x) * 2))] * kernel_shared[((((ff_c_inner * 32) + (rc_outer_inner * 4)) + rc_inner) + 320)]));
          conv2d_nchw_local[((ff_c_inner * 2) + 24)] = (conv2d_nchw_local[((ff_c_inner * 2) + 24)] + (pad_temp_shared[(((rc_outer_inner * 784) + (rc_inner * 196)) + (((int)threadIdx.x) * 2))] * kernel_shared[((((ff_c_inner * 32) + (rc_outer_inner * 4)) + rc_inner) + 384)]));
          conv2d_nchw_local[((ff_c_inner * 2) + 28)] = (conv2d_nchw_local[((ff_c_inner * 2) + 28)] + (pad_temp_shared[(((rc_outer_inner * 784) + (rc_inner * 196)) + (((int)threadIdx.x) * 2))] * kernel_shared[((((ff_c_inner * 32) + (rc_outer_inner * 4)) + rc_inner) + 448)]));
          conv2d_nchw_local[((ff_c_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_inner * 2) + 1)] + (pad_temp_shared[((((rc_outer_inner * 784) + (rc_inner * 196)) + (((int)threadIdx.x) * 2)) + 1)] * kernel_shared[(((ff_c_inner * 32) + (rc_outer_inner * 4)) + rc_inner)]));
          conv2d_nchw_local[((ff_c_inner * 2) + 5)] = (conv2d_nchw_local[((ff_c_inner * 2) + 5)] + (pad_temp_shared[((((rc_outer_inner * 784) + (rc_inner * 196)) + (((int)threadIdx.x) * 2)) + 1)] * kernel_shared[((((ff_c_inner * 32) + (rc_outer_inner * 4)) + rc_inner) + 64)]));
          conv2d_nchw_local[((ff_c_inner * 2) + 9)] = (conv2d_nchw_local[((ff_c_inner * 2) + 9)] + (pad_temp_shared[((((rc_outer_inner * 784) + (rc_inner * 196)) + (((int)threadIdx.x) * 2)) + 1)] * kernel_shared[((((ff_c_inner * 32) + (rc_outer_inner * 4)) + rc_inner) + 128)]));
          conv2d_nchw_local[((ff_c_inner * 2) + 13)] = (conv2d_nchw_local[((ff_c_inner * 2) + 13)] + (pad_temp_shared[((((rc_outer_inner * 784) + (rc_inner * 196)) + (((int)threadIdx.x) * 2)) + 1)] * kernel_shared[((((ff_c_inner * 32) + (rc_outer_inner * 4)) + rc_inner) + 192)]));
          conv2d_nchw_local[((ff_c_inner * 2) + 17)] = (conv2d_nchw_local[((ff_c_inner * 2) + 17)] + (pad_temp_shared[((((rc_outer_inner * 784) + (rc_inner * 196)) + (((int)threadIdx.x) * 2)) + 1)] * kernel_shared[((((ff_c_inner * 32) + (rc_outer_inner * 4)) + rc_inner) + 256)]));
          conv2d_nchw_local[((ff_c_inner * 2) + 21)] = (conv2d_nchw_local[((ff_c_inner * 2) + 21)] + (pad_temp_shared[((((rc_outer_inner * 784) + (rc_inner * 196)) + (((int)threadIdx.x) * 2)) + 1)] * kernel_shared[((((ff_c_inner * 32) + (rc_outer_inner * 4)) + rc_inner) + 320)]));
          conv2d_nchw_local[((ff_c_inner * 2) + 25)] = (conv2d_nchw_local[((ff_c_inner * 2) + 25)] + (pad_temp_shared[((((rc_outer_inner * 784) + (rc_inner * 196)) + (((int)threadIdx.x) * 2)) + 1)] * kernel_shared[((((ff_c_inner * 32) + (rc_outer_inner * 4)) + rc_inner) + 384)]));
          conv2d_nchw_local[((ff_c_inner * 2) + 29)] = (conv2d_nchw_local[((ff_c_inner * 2) + 29)] + (pad_temp_shared[((((rc_outer_inner * 784) + (rc_inner * 196)) + (((int)threadIdx.x) * 2)) + 1)] * kernel_shared[((((ff_c_inner * 32) + (rc_outer_inner * 4)) + rc_inner) + 448)]));
        }
      }
    }
  }
  for (int ff_inner = 0; ff_inner < 2; ++ff_inner) {
    for (int xx_inner = 0; xx_inner < 2; ++xx_inner) {
      conv2d_nchw[((((((((((int)blockIdx.x) >> 2) * 12544) + (ff_inner * 784)) + (((((int)blockIdx.x) & 3) >> 1) * 392)) + ((((int)threadIdx.x) / 7) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + ((((int)threadIdx.x) % 7) * 2)) + xx_inner)] = conv2d_nchw_local[((ff_inner * 2) + xx_inner)];
      conv2d_nchw[(((((((((((int)blockIdx.x) >> 2) * 12544) + (ff_inner * 784)) + (((((int)blockIdx.x) & 3) >> 1) * 392)) + ((((int)threadIdx.x) / 7) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + ((((int)threadIdx.x) % 7) * 2)) + xx_inner) + 1568)] = conv2d_nchw_local[(((ff_inner * 2) + xx_inner) + 4)];
      conv2d_nchw[(((((((((((int)blockIdx.x) >> 2) * 12544) + (ff_inner * 784)) + (((((int)blockIdx.x) & 3) >> 1) * 392)) + ((((int)threadIdx.x) / 7) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + ((((int)threadIdx.x) % 7) * 2)) + xx_inner) + 3136)] = conv2d_nchw_local[(((ff_inner * 2) + xx_inner) + 8)];
      conv2d_nchw[(((((((((((int)blockIdx.x) >> 2) * 12544) + (ff_inner * 784)) + (((((int)blockIdx.x) & 3) >> 1) * 392)) + ((((int)threadIdx.x) / 7) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + ((((int)threadIdx.x) % 7) * 2)) + xx_inner) + 4704)] = conv2d_nchw_local[(((ff_inner * 2) + xx_inner) + 12)];
      conv2d_nchw[(((((((((((int)blockIdx.x) >> 2) * 12544) + (ff_inner * 784)) + (((((int)blockIdx.x) & 3) >> 1) * 392)) + ((((int)threadIdx.x) / 7) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + ((((int)threadIdx.x) % 7) * 2)) + xx_inner) + 6272)] = conv2d_nchw_local[(((ff_inner * 2) + xx_inner) + 16)];
      conv2d_nchw[(((((((((((int)blockIdx.x) >> 2) * 12544) + (ff_inner * 784)) + (((((int)blockIdx.x) & 3) >> 1) * 392)) + ((((int)threadIdx.x) / 7) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + ((((int)threadIdx.x) % 7) * 2)) + xx_inner) + 7840)] = conv2d_nchw_local[(((ff_inner * 2) + xx_inner) + 20)];
      conv2d_nchw[(((((((((((int)blockIdx.x) >> 2) * 12544) + (ff_inner * 784)) + (((((int)blockIdx.x) & 3) >> 1) * 392)) + ((((int)threadIdx.x) / 7) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + ((((int)threadIdx.x) % 7) * 2)) + xx_inner) + 9408)] = conv2d_nchw_local[(((ff_inner * 2) + xx_inner) + 24)];
      conv2d_nchw[(((((((((((int)blockIdx.x) >> 2) * 12544) + (ff_inner * 784)) + (((((int)blockIdx.x) & 3) >> 1) * 392)) + ((((int)threadIdx.x) / 7) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + ((((int)threadIdx.x) % 7) * 2)) + xx_inner) + 10976)] = conv2d_nchw_local[(((ff_inner * 2) + xx_inner) + 28)];
    }
  }
}


