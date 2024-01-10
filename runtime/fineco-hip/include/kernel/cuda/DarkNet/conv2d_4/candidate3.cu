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
extern "C" __global__ void __launch_bounds__(128) candidate3(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ compute, float* __restrict__ bias) {
  float conv2d_nchw[32];
  __shared__ float pad_temp_shared[2048];
  __shared__ float kernel_shared[2048];
  for (int ff_outer_inner_init = 0; ff_outer_inner_init < 2; ++ff_outer_inner_init) {
    conv2d_nchw[(ff_outer_inner_init * 4)] = 0.000000e+00f;
    conv2d_nchw[((ff_outer_inner_init * 4) + 8)] = 0.000000e+00f;
    conv2d_nchw[((ff_outer_inner_init * 4) + 16)] = 0.000000e+00f;
    conv2d_nchw[((ff_outer_inner_init * 4) + 24)] = 0.000000e+00f;
    conv2d_nchw[((ff_outer_inner_init * 4) + 2)] = 0.000000e+00f;
    conv2d_nchw[((ff_outer_inner_init * 4) + 10)] = 0.000000e+00f;
    conv2d_nchw[((ff_outer_inner_init * 4) + 18)] = 0.000000e+00f;
    conv2d_nchw[((ff_outer_inner_init * 4) + 26)] = 0.000000e+00f;
    conv2d_nchw[((ff_outer_inner_init * 4) + 1)] = 0.000000e+00f;
    conv2d_nchw[((ff_outer_inner_init * 4) + 9)] = 0.000000e+00f;
    conv2d_nchw[((ff_outer_inner_init * 4) + 17)] = 0.000000e+00f;
    conv2d_nchw[((ff_outer_inner_init * 4) + 25)] = 0.000000e+00f;
    conv2d_nchw[((ff_outer_inner_init * 4) + 3)] = 0.000000e+00f;
    conv2d_nchw[((ff_outer_inner_init * 4) + 11)] = 0.000000e+00f;
    conv2d_nchw[((ff_outer_inner_init * 4) + 19)] = 0.000000e+00f;
    conv2d_nchw[((ff_outer_inner_init * 4) + 27)] = 0.000000e+00f;
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 4; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = data[((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7))];
    pad_temp_shared[(((int)threadIdx.x) + 128)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 6272)];
    pad_temp_shared[(((int)threadIdx.x) + 256)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 12544)];
    pad_temp_shared[(((int)threadIdx.x) + 384)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 18816)];
    pad_temp_shared[(((int)threadIdx.x) + 512)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 25088)];
    pad_temp_shared[(((int)threadIdx.x) + 640)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 31360)];
    pad_temp_shared[(((int)threadIdx.x) + 768)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 37632)];
    pad_temp_shared[(((int)threadIdx.x) + 896)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 43904)];
    pad_temp_shared[(((int)threadIdx.x) + 1024)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 50176)];
    pad_temp_shared[(((int)threadIdx.x) + 1152)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 56448)];
    pad_temp_shared[(((int)threadIdx.x) + 1280)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 62720)];
    pad_temp_shared[(((int)threadIdx.x) + 1408)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 68992)];
    pad_temp_shared[(((int)threadIdx.x) + 1536)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 75264)];
    pad_temp_shared[(((int)threadIdx.x) + 1664)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 81536)];
    pad_temp_shared[(((int)threadIdx.x) + 1792)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 87808)];
    pad_temp_shared[(((int)threadIdx.x) + 1920)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 94080)];
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)threadIdx.x) >> 5) * 128) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31))];
    kernel_shared[(((int)threadIdx.x) + 128)] = kernel[(((((((int)threadIdx.x) >> 5) * 128) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 512)];
    kernel_shared[(((int)threadIdx.x) + 256)] = kernel[(((((((int)threadIdx.x) >> 5) * 128) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 1024)];
    kernel_shared[(((int)threadIdx.x) + 384)] = kernel[(((((((int)threadIdx.x) >> 5) * 128) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 1536)];
    kernel_shared[(((int)threadIdx.x) + 512)] = kernel[(((((((int)threadIdx.x) >> 5) * 128) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 2048)];
    kernel_shared[(((int)threadIdx.x) + 640)] = kernel[(((((((int)threadIdx.x) >> 5) * 128) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 2560)];
    kernel_shared[(((int)threadIdx.x) + 768)] = kernel[(((((((int)threadIdx.x) >> 5) * 128) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 3072)];
    kernel_shared[(((int)threadIdx.x) + 896)] = kernel[(((((((int)threadIdx.x) >> 5) * 128) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 3584)];
    kernel_shared[(((int)threadIdx.x) + 1024)] = kernel[(((((((int)threadIdx.x) >> 5) * 128) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 4096)];
    kernel_shared[(((int)threadIdx.x) + 1152)] = kernel[(((((((int)threadIdx.x) >> 5) * 128) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 4608)];
    kernel_shared[(((int)threadIdx.x) + 1280)] = kernel[(((((((int)threadIdx.x) >> 5) * 128) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 5120)];
    kernel_shared[(((int)threadIdx.x) + 1408)] = kernel[(((((((int)threadIdx.x) >> 5) * 128) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 5632)];
    kernel_shared[(((int)threadIdx.x) + 1536)] = kernel[(((((((int)threadIdx.x) >> 5) * 128) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 6144)];
    kernel_shared[(((int)threadIdx.x) + 1664)] = kernel[(((((((int)threadIdx.x) >> 5) * 128) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 6656)];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[(((((((int)threadIdx.x) >> 5) * 128) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 7168)];
    kernel_shared[(((int)threadIdx.x) + 1920)] = kernel[(((((((int)threadIdx.x) >> 5) * 128) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 7680)];
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 8; ++rc_outer_inner) {
      for (int ff_outer_inner = 0; ff_outer_inner < 2; ++ff_outer_inner) {
        for (int yy_outer_inner = 0; yy_outer_inner < 2; ++yy_outer_inner) {
          for (int rc_inner = 0; rc_inner < 4; ++rc_inner) {
            conv2d_nchw[((ff_outer_inner * 4) + yy_outer_inner)] = (conv2d_nchw[((ff_outer_inner * 4) + yy_outer_inner)] + (pad_temp_shared[((((rc_outer_inner * 256) + (rc_inner * 64)) + (yy_outer_inner * 8)) + (((int)threadIdx.x) & 7))] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 4)) + rc_inner)]));
            conv2d_nchw[(((ff_outer_inner * 4) + yy_outer_inner) + 8)] = (conv2d_nchw[(((ff_outer_inner * 4) + yy_outer_inner) + 8)] + (pad_temp_shared[(((((rc_outer_inner * 256) + (rc_inner * 64)) + (yy_outer_inner * 8)) + (((int)threadIdx.x) & 7)) + 16)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 4)) + rc_inner)]));
            conv2d_nchw[(((ff_outer_inner * 4) + yy_outer_inner) + 16)] = (conv2d_nchw[(((ff_outer_inner * 4) + yy_outer_inner) + 16)] + (pad_temp_shared[(((((rc_outer_inner * 256) + (rc_inner * 64)) + (yy_outer_inner * 8)) + (((int)threadIdx.x) & 7)) + 32)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 4)) + rc_inner)]));
            conv2d_nchw[(((ff_outer_inner * 4) + yy_outer_inner) + 24)] = (conv2d_nchw[(((ff_outer_inner * 4) + yy_outer_inner) + 24)] + (pad_temp_shared[(((((rc_outer_inner * 256) + (rc_inner * 64)) + (yy_outer_inner * 8)) + (((int)threadIdx.x) & 7)) + 48)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 4)) + rc_inner)]));
            conv2d_nchw[(((ff_outer_inner * 4) + yy_outer_inner) + 2)] = (conv2d_nchw[(((ff_outer_inner * 4) + yy_outer_inner) + 2)] + (pad_temp_shared[((((rc_outer_inner * 256) + (rc_inner * 64)) + (yy_outer_inner * 8)) + (((int)threadIdx.x) & 7))] * kernel_shared[((((((((int)threadIdx.x) >> 3) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 4)) + rc_inner) + 32)]));
            conv2d_nchw[(((ff_outer_inner * 4) + yy_outer_inner) + 10)] = (conv2d_nchw[(((ff_outer_inner * 4) + yy_outer_inner) + 10)] + (pad_temp_shared[(((((rc_outer_inner * 256) + (rc_inner * 64)) + (yy_outer_inner * 8)) + (((int)threadIdx.x) & 7)) + 16)] * kernel_shared[((((((((int)threadIdx.x) >> 3) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 4)) + rc_inner) + 32)]));
            conv2d_nchw[(((ff_outer_inner * 4) + yy_outer_inner) + 18)] = (conv2d_nchw[(((ff_outer_inner * 4) + yy_outer_inner) + 18)] + (pad_temp_shared[(((((rc_outer_inner * 256) + (rc_inner * 64)) + (yy_outer_inner * 8)) + (((int)threadIdx.x) & 7)) + 32)] * kernel_shared[((((((((int)threadIdx.x) >> 3) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 4)) + rc_inner) + 32)]));
            conv2d_nchw[(((ff_outer_inner * 4) + yy_outer_inner) + 26)] = (conv2d_nchw[(((ff_outer_inner * 4) + yy_outer_inner) + 26)] + (pad_temp_shared[(((((rc_outer_inner * 256) + (rc_inner * 64)) + (yy_outer_inner * 8)) + (((int)threadIdx.x) & 7)) + 48)] * kernel_shared[((((((((int)threadIdx.x) >> 3) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 4)) + rc_inner) + 32)]));
          }
        }
      }
    }
  }
  for (int i1_inner = 0; i1_inner < 4; ++i1_inner) {
    for (int i2_inner = 0; i2_inner < 2; ++i2_inner) {
      compute[(((((((((int)threadIdx.x) >> 3) * 12544) + (i1_inner * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (i2_inner * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7))] = max((conv2d_nchw[((i1_inner * 2) + i2_inner)] + bias[(((((int)threadIdx.x) >> 3) * 4) + i1_inner)]), 0.000000e+00f);
      compute[((((((((((int)threadIdx.x) >> 3) * 12544) + (i1_inner * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (i2_inner * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 112)] = max((conv2d_nchw[(((i1_inner * 2) + i2_inner) + 8)] + bias[(((((int)threadIdx.x) >> 3) * 4) + i1_inner)]), 0.000000e+00f);
      compute[((((((((((int)threadIdx.x) >> 3) * 12544) + (i1_inner * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (i2_inner * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 224)] = max((conv2d_nchw[(((i1_inner * 2) + i2_inner) + 16)] + bias[(((((int)threadIdx.x) >> 3) * 4) + i1_inner)]), 0.000000e+00f);
      compute[((((((((((int)threadIdx.x) >> 3) * 12544) + (i1_inner * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (i2_inner * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 336)] = max((conv2d_nchw[(((i1_inner * 2) + i2_inner) + 24)] + bias[(((((int)threadIdx.x) >> 3) * 4) + i1_inner)]), 0.000000e+00f);
    }
  }
}


