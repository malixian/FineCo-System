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
extern "C" __global__ void __launch_bounds__(128) candidate5(float* __restrict__ Input, float* __restrict__ kernel, float* __restrict__ compute) {
  float conv2d_nchw[32];
  __shared__ float pad_temp_shared[2048];
  __shared__ float kernel_shared[2048];
  conv2d_nchw[0] = 0.000000e+00f;
  conv2d_nchw[16] = 0.000000e+00f;
  conv2d_nchw[1] = 0.000000e+00f;
  conv2d_nchw[17] = 0.000000e+00f;
  conv2d_nchw[2] = 0.000000e+00f;
  conv2d_nchw[18] = 0.000000e+00f;
  conv2d_nchw[3] = 0.000000e+00f;
  conv2d_nchw[19] = 0.000000e+00f;
  conv2d_nchw[4] = 0.000000e+00f;
  conv2d_nchw[20] = 0.000000e+00f;
  conv2d_nchw[5] = 0.000000e+00f;
  conv2d_nchw[21] = 0.000000e+00f;
  conv2d_nchw[6] = 0.000000e+00f;
  conv2d_nchw[22] = 0.000000e+00f;
  conv2d_nchw[7] = 0.000000e+00f;
  conv2d_nchw[23] = 0.000000e+00f;
  conv2d_nchw[8] = 0.000000e+00f;
  conv2d_nchw[24] = 0.000000e+00f;
  conv2d_nchw[9] = 0.000000e+00f;
  conv2d_nchw[25] = 0.000000e+00f;
  conv2d_nchw[10] = 0.000000e+00f;
  conv2d_nchw[26] = 0.000000e+00f;
  conv2d_nchw[11] = 0.000000e+00f;
  conv2d_nchw[27] = 0.000000e+00f;
  conv2d_nchw[12] = 0.000000e+00f;
  conv2d_nchw[28] = 0.000000e+00f;
  conv2d_nchw[13] = 0.000000e+00f;
  conv2d_nchw[29] = 0.000000e+00f;
  conv2d_nchw[14] = 0.000000e+00f;
  conv2d_nchw[30] = 0.000000e+00f;
  conv2d_nchw[15] = 0.000000e+00f;
  conv2d_nchw[31] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 4; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = Input[((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7))];
    pad_temp_shared[(((int)threadIdx.x) + 128)] = Input[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 6272)];
    pad_temp_shared[(((int)threadIdx.x) + 256)] = Input[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 12544)];
    pad_temp_shared[(((int)threadIdx.x) + 384)] = Input[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 18816)];
    pad_temp_shared[(((int)threadIdx.x) + 512)] = Input[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 25088)];
    pad_temp_shared[(((int)threadIdx.x) + 640)] = Input[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 31360)];
    pad_temp_shared[(((int)threadIdx.x) + 768)] = Input[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 37632)];
    pad_temp_shared[(((int)threadIdx.x) + 896)] = Input[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 43904)];
    pad_temp_shared[(((int)threadIdx.x) + 1024)] = Input[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 50176)];
    pad_temp_shared[(((int)threadIdx.x) + 1152)] = Input[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 56448)];
    pad_temp_shared[(((int)threadIdx.x) + 1280)] = Input[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 62720)];
    pad_temp_shared[(((int)threadIdx.x) + 1408)] = Input[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 68992)];
    pad_temp_shared[(((int)threadIdx.x) + 1536)] = Input[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 75264)];
    pad_temp_shared[(((int)threadIdx.x) + 1664)] = Input[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 81536)];
    pad_temp_shared[(((int)threadIdx.x) + 1792)] = Input[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 87808)];
    pad_temp_shared[(((int)threadIdx.x) + 1920)] = Input[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 94080)];
    kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) / 49) * 8192) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31))];
    kernel_shared[(((int)threadIdx.x) + 128)] = kernel[((((((((int)blockIdx.x) / 49) * 8192) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 512)];
    kernel_shared[(((int)threadIdx.x) + 256)] = kernel[((((((((int)blockIdx.x) / 49) * 8192) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 1024)];
    kernel_shared[(((int)threadIdx.x) + 384)] = kernel[((((((((int)blockIdx.x) / 49) * 8192) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 1536)];
    kernel_shared[(((int)threadIdx.x) + 512)] = kernel[((((((((int)blockIdx.x) / 49) * 8192) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 2048)];
    kernel_shared[(((int)threadIdx.x) + 640)] = kernel[((((((((int)blockIdx.x) / 49) * 8192) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 2560)];
    kernel_shared[(((int)threadIdx.x) + 768)] = kernel[((((((((int)blockIdx.x) / 49) * 8192) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 3072)];
    kernel_shared[(((int)threadIdx.x) + 896)] = kernel[((((((((int)blockIdx.x) / 49) * 8192) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 3584)];
    kernel_shared[(((int)threadIdx.x) + 1024)] = kernel[((((((((int)blockIdx.x) / 49) * 8192) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 4096)];
    kernel_shared[(((int)threadIdx.x) + 1152)] = kernel[((((((((int)blockIdx.x) / 49) * 8192) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 4608)];
    kernel_shared[(((int)threadIdx.x) + 1280)] = kernel[((((((((int)blockIdx.x) / 49) * 8192) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 5120)];
    kernel_shared[(((int)threadIdx.x) + 1408)] = kernel[((((((((int)blockIdx.x) / 49) * 8192) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 5632)];
    kernel_shared[(((int)threadIdx.x) + 1536)] = kernel[((((((((int)blockIdx.x) / 49) * 8192) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 6144)];
    kernel_shared[(((int)threadIdx.x) + 1664)] = kernel[((((((((int)blockIdx.x) / 49) * 8192) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 6656)];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[((((((((int)blockIdx.x) / 49) * 8192) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 7168)];
    kernel_shared[(((int)threadIdx.x) + 1920)] = kernel[((((((((int)blockIdx.x) / 49) * 8192) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 7680)];
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 8; ++rc_outer_inner) {
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((rc_outer_inner * 256) + (((int)threadIdx.x) & 7))] * kernel_shared[(((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4))]));
      conv2d_nchw[16] = (conv2d_nchw[16] + (pad_temp_shared[((rc_outer_inner * 256) + (((int)threadIdx.x) & 7))] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1024)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 8)] * kernel_shared[(((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4))]));
      conv2d_nchw[17] = (conv2d_nchw[17] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 8)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1024)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 16)] * kernel_shared[(((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4))]));
      conv2d_nchw[18] = (conv2d_nchw[18] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 16)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1024)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 24)] * kernel_shared[(((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4))]));
      conv2d_nchw[19] = (conv2d_nchw[19] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 24)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1024)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 32)] * kernel_shared[(((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4))]));
      conv2d_nchw[20] = (conv2d_nchw[20] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 32)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1024)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 40)] * kernel_shared[(((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4))]));
      conv2d_nchw[21] = (conv2d_nchw[21] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 40)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1024)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 48)] * kernel_shared[(((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4))]));
      conv2d_nchw[22] = (conv2d_nchw[22] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 48)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1024)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 56)] * kernel_shared[(((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4))]));
      conv2d_nchw[23] = (conv2d_nchw[23] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 56)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1024)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[((rc_outer_inner * 256) + (((int)threadIdx.x) & 7))] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 32)]));
      conv2d_nchw[24] = (conv2d_nchw[24] + (pad_temp_shared[((rc_outer_inner * 256) + (((int)threadIdx.x) & 7))] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1056)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 8)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 32)]));
      conv2d_nchw[25] = (conv2d_nchw[25] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 8)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1056)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 16)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 32)]));
      conv2d_nchw[26] = (conv2d_nchw[26] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 16)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1056)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 24)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 32)]));
      conv2d_nchw[27] = (conv2d_nchw[27] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 24)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1056)]));
      conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 32)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 32)]));
      conv2d_nchw[28] = (conv2d_nchw[28] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 32)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1056)]));
      conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 40)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 32)]));
      conv2d_nchw[29] = (conv2d_nchw[29] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 40)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1056)]));
      conv2d_nchw[14] = (conv2d_nchw[14] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 48)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 32)]));
      conv2d_nchw[30] = (conv2d_nchw[30] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 48)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1056)]));
      conv2d_nchw[15] = (conv2d_nchw[15] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 56)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 32)]));
      conv2d_nchw[31] = (conv2d_nchw[31] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 56)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1056)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 64)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1)]));
      conv2d_nchw[16] = (conv2d_nchw[16] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 64)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1025)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 72)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1)]));
      conv2d_nchw[17] = (conv2d_nchw[17] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 72)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1025)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 80)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1)]));
      conv2d_nchw[18] = (conv2d_nchw[18] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 80)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1025)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 88)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1)]));
      conv2d_nchw[19] = (conv2d_nchw[19] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 88)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1025)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 96)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1)]));
      conv2d_nchw[20] = (conv2d_nchw[20] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 96)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1025)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 104)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1)]));
      conv2d_nchw[21] = (conv2d_nchw[21] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 104)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1025)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 112)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1)]));
      conv2d_nchw[22] = (conv2d_nchw[22] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 112)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1025)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 120)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1)]));
      conv2d_nchw[23] = (conv2d_nchw[23] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 120)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1025)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 64)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 33)]));
      conv2d_nchw[24] = (conv2d_nchw[24] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 64)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1057)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 72)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 33)]));
      conv2d_nchw[25] = (conv2d_nchw[25] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 72)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1057)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 80)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 33)]));
      conv2d_nchw[26] = (conv2d_nchw[26] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 80)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1057)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 88)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 33)]));
      conv2d_nchw[27] = (conv2d_nchw[27] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 88)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1057)]));
      conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 96)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 33)]));
      conv2d_nchw[28] = (conv2d_nchw[28] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 96)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1057)]));
      conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 104)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 33)]));
      conv2d_nchw[29] = (conv2d_nchw[29] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 104)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1057)]));
      conv2d_nchw[14] = (conv2d_nchw[14] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 112)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 33)]));
      conv2d_nchw[30] = (conv2d_nchw[30] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 112)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1057)]));
      conv2d_nchw[15] = (conv2d_nchw[15] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 120)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 33)]));
      conv2d_nchw[31] = (conv2d_nchw[31] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 120)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1057)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 128)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 2)]));
      conv2d_nchw[16] = (conv2d_nchw[16] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 128)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1026)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 136)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 2)]));
      conv2d_nchw[17] = (conv2d_nchw[17] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 136)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1026)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 144)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 2)]));
      conv2d_nchw[18] = (conv2d_nchw[18] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 144)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1026)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 152)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 2)]));
      conv2d_nchw[19] = (conv2d_nchw[19] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 152)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1026)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 160)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 2)]));
      conv2d_nchw[20] = (conv2d_nchw[20] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 160)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1026)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 168)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 2)]));
      conv2d_nchw[21] = (conv2d_nchw[21] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 168)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1026)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 176)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 2)]));
      conv2d_nchw[22] = (conv2d_nchw[22] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 176)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1026)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 184)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 2)]));
      conv2d_nchw[23] = (conv2d_nchw[23] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 184)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1026)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 128)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 34)]));
      conv2d_nchw[24] = (conv2d_nchw[24] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 128)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1058)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 136)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 34)]));
      conv2d_nchw[25] = (conv2d_nchw[25] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 136)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1058)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 144)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 34)]));
      conv2d_nchw[26] = (conv2d_nchw[26] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 144)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1058)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 152)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 34)]));
      conv2d_nchw[27] = (conv2d_nchw[27] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 152)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1058)]));
      conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 160)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 34)]));
      conv2d_nchw[28] = (conv2d_nchw[28] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 160)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1058)]));
      conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 168)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 34)]));
      conv2d_nchw[29] = (conv2d_nchw[29] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 168)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1058)]));
      conv2d_nchw[14] = (conv2d_nchw[14] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 176)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 34)]));
      conv2d_nchw[30] = (conv2d_nchw[30] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 176)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1058)]));
      conv2d_nchw[15] = (conv2d_nchw[15] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 184)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 34)]));
      conv2d_nchw[31] = (conv2d_nchw[31] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 184)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1058)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 192)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 3)]));
      conv2d_nchw[16] = (conv2d_nchw[16] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 192)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1027)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 200)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 3)]));
      conv2d_nchw[17] = (conv2d_nchw[17] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 200)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1027)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 208)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 3)]));
      conv2d_nchw[18] = (conv2d_nchw[18] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 208)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1027)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 216)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 3)]));
      conv2d_nchw[19] = (conv2d_nchw[19] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 216)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1027)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 224)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 3)]));
      conv2d_nchw[20] = (conv2d_nchw[20] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 224)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1027)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 232)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 3)]));
      conv2d_nchw[21] = (conv2d_nchw[21] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 232)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1027)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 240)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 3)]));
      conv2d_nchw[22] = (conv2d_nchw[22] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 240)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1027)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 248)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 3)]));
      conv2d_nchw[23] = (conv2d_nchw[23] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 248)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1027)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 192)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 35)]));
      conv2d_nchw[24] = (conv2d_nchw[24] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 192)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1059)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 200)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 35)]));
      conv2d_nchw[25] = (conv2d_nchw[25] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 200)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1059)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 208)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 35)]));
      conv2d_nchw[26] = (conv2d_nchw[26] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 208)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1059)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 216)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 35)]));
      conv2d_nchw[27] = (conv2d_nchw[27] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 216)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1059)]));
      conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 224)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 35)]));
      conv2d_nchw[28] = (conv2d_nchw[28] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 224)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1059)]));
      conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 232)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 35)]));
      conv2d_nchw[29] = (conv2d_nchw[29] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 232)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1059)]));
      conv2d_nchw[14] = (conv2d_nchw[14] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 240)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 35)]));
      conv2d_nchw[30] = (conv2d_nchw[30] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 240)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1059)]));
      conv2d_nchw[15] = (conv2d_nchw[15] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 248)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 35)]));
      conv2d_nchw[31] = (conv2d_nchw[31] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 7)) + 248)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 64) + (rc_outer_inner * 4)) + 1059)]));
    }
  }
  for (int i1_inner = 0; i1_inner < 2; ++i1_inner) {
    for (int i2_inner = 0; i2_inner < 8; ++i2_inner) {
      compute[((((((((((int)blockIdx.x) / 49) * 200704) + ((((int)threadIdx.x) >> 3) * 6272)) + (i1_inner * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (i2_inner * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7))] = max(conv2d_nchw[((i1_inner * 8) + i2_inner)], 0.000000e+00f);
      compute[(((((((((((int)blockIdx.x) / 49) * 200704) + ((((int)threadIdx.x) >> 3) * 6272)) + (i1_inner * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (i2_inner * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 100352)] = max(conv2d_nchw[(((i1_inner * 8) + i2_inner) + 16)], 0.000000e+00f);
    }
  }
}


