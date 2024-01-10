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
extern "C" __global__ void __launch_bounds__(128) candidate7(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[16];
  __shared__ float pad_temp_shared[4096];
  __shared__ float kernel_shared[2048];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  conv2d_nchw_local[7] = 0.000000e+00f;
  conv2d_nchw_local[8] = 0.000000e+00f;
  conv2d_nchw_local[9] = 0.000000e+00f;
  conv2d_nchw_local[12] = 0.000000e+00f;
  conv2d_nchw_local[13] = 0.000000e+00f;
  conv2d_nchw_local[10] = 0.000000e+00f;
  conv2d_nchw_local[11] = 0.000000e+00f;
  conv2d_nchw_local[14] = 0.000000e+00f;
  conv2d_nchw_local[15] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 4; ++rc_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 32; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
      pad_temp_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 128) + ((int)threadIdx.x))] = data[(((((((rc_outer_outer * 200704) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 6272)) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7))];
    }
    kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) / 49) * 8192) + ((((int)threadIdx.x) >> 6) * 256)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63))];
    kernel_shared[(((int)threadIdx.x) + 128)] = kernel[((((((((int)blockIdx.x) / 49) * 8192) + ((((int)threadIdx.x) >> 6) * 256)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 512)];
    kernel_shared[(((int)threadIdx.x) + 256)] = kernel[((((((((int)blockIdx.x) / 49) * 8192) + ((((int)threadIdx.x) >> 6) * 256)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 1024)];
    kernel_shared[(((int)threadIdx.x) + 384)] = kernel[((((((((int)blockIdx.x) / 49) * 8192) + ((((int)threadIdx.x) >> 6) * 256)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 1536)];
    kernel_shared[(((int)threadIdx.x) + 512)] = kernel[((((((((int)blockIdx.x) / 49) * 8192) + ((((int)threadIdx.x) >> 6) * 256)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 2048)];
    kernel_shared[(((int)threadIdx.x) + 640)] = kernel[((((((((int)blockIdx.x) / 49) * 8192) + ((((int)threadIdx.x) >> 6) * 256)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 2560)];
    kernel_shared[(((int)threadIdx.x) + 768)] = kernel[((((((((int)blockIdx.x) / 49) * 8192) + ((((int)threadIdx.x) >> 6) * 256)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 3072)];
    kernel_shared[(((int)threadIdx.x) + 896)] = kernel[((((((((int)blockIdx.x) / 49) * 8192) + ((((int)threadIdx.x) >> 6) * 256)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 3584)];
    kernel_shared[(((int)threadIdx.x) + 1024)] = kernel[((((((((int)blockIdx.x) / 49) * 8192) + ((((int)threadIdx.x) >> 6) * 256)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 4096)];
    kernel_shared[(((int)threadIdx.x) + 1152)] = kernel[((((((((int)blockIdx.x) / 49) * 8192) + ((((int)threadIdx.x) >> 6) * 256)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 4608)];
    kernel_shared[(((int)threadIdx.x) + 1280)] = kernel[((((((((int)blockIdx.x) / 49) * 8192) + ((((int)threadIdx.x) >> 6) * 256)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 5120)];
    kernel_shared[(((int)threadIdx.x) + 1408)] = kernel[((((((((int)blockIdx.x) / 49) * 8192) + ((((int)threadIdx.x) >> 6) * 256)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 5632)];
    kernel_shared[(((int)threadIdx.x) + 1536)] = kernel[((((((((int)blockIdx.x) / 49) * 8192) + ((((int)threadIdx.x) >> 6) * 256)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 6144)];
    kernel_shared[(((int)threadIdx.x) + 1664)] = kernel[((((((((int)blockIdx.x) / 49) * 8192) + ((((int)threadIdx.x) >> 6) * 256)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 6656)];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[((((((((int)blockIdx.x) / 49) * 8192) + ((((int)threadIdx.x) >> 6) * 256)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 7168)];
    kernel_shared[(((int)threadIdx.x) + 1920)] = kernel[((((((((int)blockIdx.x) / 49) * 8192) + ((((int)threadIdx.x) >> 6) * 256)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 7680)];
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 64; ++rc_outer_inner) {
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 64) + (((((int)threadIdx.x) & 15) >> 3) * 32)) + (((int)threadIdx.x) & 7))] * kernel_shared[(((((int)threadIdx.x) >> 4) * 256) + rc_outer_inner)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((rc_outer_inner * 64) + (((((int)threadIdx.x) & 15) >> 3) * 32)) + (((int)threadIdx.x) & 7)) + 8)] * kernel_shared[(((((int)threadIdx.x) >> 4) * 256) + rc_outer_inner)]));
      conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((rc_outer_inner * 64) + (((((int)threadIdx.x) & 15) >> 3) * 32)) + (((int)threadIdx.x) & 7))] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + rc_outer_inner) + 64)]));
      conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[((((rc_outer_inner * 64) + (((((int)threadIdx.x) & 15) >> 3) * 32)) + (((int)threadIdx.x) & 7)) + 8)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + rc_outer_inner) + 64)]));
      conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((rc_outer_inner * 64) + (((((int)threadIdx.x) & 15) >> 3) * 32)) + (((int)threadIdx.x) & 7)) + 16)] * kernel_shared[(((((int)threadIdx.x) >> 4) * 256) + rc_outer_inner)]));
      conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((rc_outer_inner * 64) + (((((int)threadIdx.x) & 15) >> 3) * 32)) + (((int)threadIdx.x) & 7)) + 24)] * kernel_shared[(((((int)threadIdx.x) >> 4) * 256) + rc_outer_inner)]));
      conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[((((rc_outer_inner * 64) + (((((int)threadIdx.x) & 15) >> 3) * 32)) + (((int)threadIdx.x) & 7)) + 16)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + rc_outer_inner) + 64)]));
      conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[((((rc_outer_inner * 64) + (((((int)threadIdx.x) & 15) >> 3) * 32)) + (((int)threadIdx.x) & 7)) + 24)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + rc_outer_inner) + 64)]));
      conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[(((rc_outer_inner * 64) + (((((int)threadIdx.x) & 15) >> 3) * 32)) + (((int)threadIdx.x) & 7))] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + rc_outer_inner) + 128)]));
      conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[((((rc_outer_inner * 64) + (((((int)threadIdx.x) & 15) >> 3) * 32)) + (((int)threadIdx.x) & 7)) + 8)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + rc_outer_inner) + 128)]));
      conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[(((rc_outer_inner * 64) + (((((int)threadIdx.x) & 15) >> 3) * 32)) + (((int)threadIdx.x) & 7))] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + rc_outer_inner) + 192)]));
      conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[((((rc_outer_inner * 64) + (((((int)threadIdx.x) & 15) >> 3) * 32)) + (((int)threadIdx.x) & 7)) + 8)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + rc_outer_inner) + 192)]));
      conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[((((rc_outer_inner * 64) + (((((int)threadIdx.x) & 15) >> 3) * 32)) + (((int)threadIdx.x) & 7)) + 16)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + rc_outer_inner) + 128)]));
      conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[((((rc_outer_inner * 64) + (((((int)threadIdx.x) & 15) >> 3) * 32)) + (((int)threadIdx.x) & 7)) + 24)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + rc_outer_inner) + 128)]));
      conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[((((rc_outer_inner * 64) + (((((int)threadIdx.x) & 15) >> 3) * 32)) + (((int)threadIdx.x) & 7)) + 16)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + rc_outer_inner) + 192)]));
      conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[((((rc_outer_inner * 64) + (((((int)threadIdx.x) & 15) >> 3) * 32)) + (((int)threadIdx.x) & 7)) + 24)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + rc_outer_inner) + 192)]));
    }
  }
  for (int ff_inner = 0; ff_inner < 4; ++ff_inner) {
    for (int yy_inner = 0; yy_inner < 4; ++yy_inner) {
      conv2d_nchw[(((((((((((int)blockIdx.x) / 49) * 100352) + ((((int)threadIdx.x) >> 4) * 12544)) + (ff_inner * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 15) >> 3) * 224)) + (yy_inner * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7))] = conv2d_nchw_local[((ff_inner * 4) + yy_inner)];
    }
  }
}


