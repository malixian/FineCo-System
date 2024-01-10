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
extern "C" __global__ void __launch_bounds__(128) candidate4(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ compute, float* __restrict__ bias) {
  float conv2d_nchw[16];
  __shared__ float pad_temp_shared[4096];
  __shared__ float kernel_shared[2048];
  conv2d_nchw[0] = 0.000000e+00f;
  conv2d_nchw[8] = 0.000000e+00f;
  conv2d_nchw[1] = 0.000000e+00f;
  conv2d_nchw[9] = 0.000000e+00f;
  conv2d_nchw[2] = 0.000000e+00f;
  conv2d_nchw[10] = 0.000000e+00f;
  conv2d_nchw[3] = 0.000000e+00f;
  conv2d_nchw[11] = 0.000000e+00f;
  conv2d_nchw[4] = 0.000000e+00f;
  conv2d_nchw[12] = 0.000000e+00f;
  conv2d_nchw[5] = 0.000000e+00f;
  conv2d_nchw[13] = 0.000000e+00f;
  conv2d_nchw[6] = 0.000000e+00f;
  conv2d_nchw[14] = 0.000000e+00f;
  conv2d_nchw[7] = 0.000000e+00f;
  conv2d_nchw[15] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 2; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = data[((((((rc_outer_outer * 200704) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7))];
    pad_temp_shared[(((int)threadIdx.x) + 128)] = data[(((((((rc_outer_outer * 200704) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 6272)];
    pad_temp_shared[(((int)threadIdx.x) + 256)] = data[(((((((rc_outer_outer * 200704) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 12544)];
    pad_temp_shared[(((int)threadIdx.x) + 384)] = data[(((((((rc_outer_outer * 200704) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 18816)];
    pad_temp_shared[(((int)threadIdx.x) + 512)] = data[(((((((rc_outer_outer * 200704) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 25088)];
    pad_temp_shared[(((int)threadIdx.x) + 640)] = data[(((((((rc_outer_outer * 200704) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 31360)];
    pad_temp_shared[(((int)threadIdx.x) + 768)] = data[(((((((rc_outer_outer * 200704) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 37632)];
    pad_temp_shared[(((int)threadIdx.x) + 896)] = data[(((((((rc_outer_outer * 200704) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 43904)];
    pad_temp_shared[(((int)threadIdx.x) + 1024)] = data[(((((((rc_outer_outer * 200704) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 50176)];
    pad_temp_shared[(((int)threadIdx.x) + 1152)] = data[(((((((rc_outer_outer * 200704) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 56448)];
    pad_temp_shared[(((int)threadIdx.x) + 1280)] = data[(((((((rc_outer_outer * 200704) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 62720)];
    pad_temp_shared[(((int)threadIdx.x) + 1408)] = data[(((((((rc_outer_outer * 200704) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 68992)];
    pad_temp_shared[(((int)threadIdx.x) + 1536)] = data[(((((((rc_outer_outer * 200704) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 75264)];
    pad_temp_shared[(((int)threadIdx.x) + 1664)] = data[(((((((rc_outer_outer * 200704) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 81536)];
    pad_temp_shared[(((int)threadIdx.x) + 1792)] = data[(((((((rc_outer_outer * 200704) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 87808)];
    pad_temp_shared[(((int)threadIdx.x) + 1920)] = data[(((((((rc_outer_outer * 200704) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 94080)];
    pad_temp_shared[(((int)threadIdx.x) + 2048)] = data[(((((((rc_outer_outer * 200704) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 100352)];
    pad_temp_shared[(((int)threadIdx.x) + 2176)] = data[(((((((rc_outer_outer * 200704) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 106624)];
    pad_temp_shared[(((int)threadIdx.x) + 2304)] = data[(((((((rc_outer_outer * 200704) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 112896)];
    pad_temp_shared[(((int)threadIdx.x) + 2432)] = data[(((((((rc_outer_outer * 200704) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 119168)];
    pad_temp_shared[(((int)threadIdx.x) + 2560)] = data[(((((((rc_outer_outer * 200704) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 125440)];
    pad_temp_shared[(((int)threadIdx.x) + 2688)] = data[(((((((rc_outer_outer * 200704) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 131712)];
    pad_temp_shared[(((int)threadIdx.x) + 2816)] = data[(((((((rc_outer_outer * 200704) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 137984)];
    pad_temp_shared[(((int)threadIdx.x) + 2944)] = data[(((((((rc_outer_outer * 200704) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 144256)];
    pad_temp_shared[(((int)threadIdx.x) + 3072)] = data[(((((((rc_outer_outer * 200704) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 150528)];
    pad_temp_shared[(((int)threadIdx.x) + 3200)] = data[(((((((rc_outer_outer * 200704) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 156800)];
    pad_temp_shared[(((int)threadIdx.x) + 3328)] = data[(((((((rc_outer_outer * 200704) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 163072)];
    pad_temp_shared[(((int)threadIdx.x) + 3456)] = data[(((((((rc_outer_outer * 200704) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 169344)];
    pad_temp_shared[(((int)threadIdx.x) + 3584)] = data[(((((((rc_outer_outer * 200704) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 175616)];
    pad_temp_shared[(((int)threadIdx.x) + 3712)] = data[(((((((rc_outer_outer * 200704) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 181888)];
    pad_temp_shared[(((int)threadIdx.x) + 3840)] = data[(((((((rc_outer_outer * 200704) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 188160)];
    pad_temp_shared[(((int)threadIdx.x) + 3968)] = data[(((((((rc_outer_outer * 200704) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 194432)];
    kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) / 49) * 4096) + ((((int)threadIdx.x) >> 6) * 128)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63))];
    kernel_shared[(((int)threadIdx.x) + 128)] = kernel[((((((((int)blockIdx.x) / 49) * 4096) + ((((int)threadIdx.x) >> 6) * 128)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 256)];
    kernel_shared[(((int)threadIdx.x) + 256)] = kernel[((((((((int)blockIdx.x) / 49) * 4096) + ((((int)threadIdx.x) >> 6) * 128)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 512)];
    kernel_shared[(((int)threadIdx.x) + 384)] = kernel[((((((((int)blockIdx.x) / 49) * 4096) + ((((int)threadIdx.x) >> 6) * 128)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 768)];
    kernel_shared[(((int)threadIdx.x) + 512)] = kernel[((((((((int)blockIdx.x) / 49) * 4096) + ((((int)threadIdx.x) >> 6) * 128)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 1024)];
    kernel_shared[(((int)threadIdx.x) + 640)] = kernel[((((((((int)blockIdx.x) / 49) * 4096) + ((((int)threadIdx.x) >> 6) * 128)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 1280)];
    kernel_shared[(((int)threadIdx.x) + 768)] = kernel[((((((((int)blockIdx.x) / 49) * 4096) + ((((int)threadIdx.x) >> 6) * 128)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 1536)];
    kernel_shared[(((int)threadIdx.x) + 896)] = kernel[((((((((int)blockIdx.x) / 49) * 4096) + ((((int)threadIdx.x) >> 6) * 128)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 1792)];
    kernel_shared[(((int)threadIdx.x) + 1024)] = kernel[((((((((int)blockIdx.x) / 49) * 4096) + ((((int)threadIdx.x) >> 6) * 128)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 2048)];
    kernel_shared[(((int)threadIdx.x) + 1152)] = kernel[((((((((int)blockIdx.x) / 49) * 4096) + ((((int)threadIdx.x) >> 6) * 128)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 2304)];
    kernel_shared[(((int)threadIdx.x) + 1280)] = kernel[((((((((int)blockIdx.x) / 49) * 4096) + ((((int)threadIdx.x) >> 6) * 128)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 2560)];
    kernel_shared[(((int)threadIdx.x) + 1408)] = kernel[((((((((int)blockIdx.x) / 49) * 4096) + ((((int)threadIdx.x) >> 6) * 128)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 2816)];
    kernel_shared[(((int)threadIdx.x) + 1536)] = kernel[((((((((int)blockIdx.x) / 49) * 4096) + ((((int)threadIdx.x) >> 6) * 128)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 3072)];
    kernel_shared[(((int)threadIdx.x) + 1664)] = kernel[((((((((int)blockIdx.x) / 49) * 4096) + ((((int)threadIdx.x) >> 6) * 128)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 3328)];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[((((((((int)blockIdx.x) / 49) * 4096) + ((((int)threadIdx.x) >> 6) * 128)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 3584)];
    kernel_shared[(((int)threadIdx.x) + 1920)] = kernel[((((((((int)blockIdx.x) / 49) * 4096) + ((((int)threadIdx.x) >> 6) * 128)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 3840)];
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 32; ++rc_outer_inner) {
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((rc_outer_inner * 128) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7))] * kernel_shared[(((((int)threadIdx.x) >> 4) * 256) + (rc_outer_inner * 2))]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 32)] * kernel_shared[(((((int)threadIdx.x) >> 4) * 256) + (rc_outer_inner * 2))]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 8)] * kernel_shared[(((((int)threadIdx.x) >> 4) * 256) + (rc_outer_inner * 2))]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 40)] * kernel_shared[(((((int)threadIdx.x) >> 4) * 256) + (rc_outer_inner * 2))]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 64)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + (rc_outer_inner * 2)) + 1)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 96)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + (rc_outer_inner * 2)) + 1)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 72)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + (rc_outer_inner * 2)) + 1)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 104)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + (rc_outer_inner * 2)) + 1)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((rc_outer_inner * 128) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7))] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + (rc_outer_inner * 2)) + 64)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 32)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + (rc_outer_inner * 2)) + 64)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 8)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + (rc_outer_inner * 2)) + 64)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 40)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + (rc_outer_inner * 2)) + 64)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 64)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + (rc_outer_inner * 2)) + 65)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 96)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + (rc_outer_inner * 2)) + 65)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 72)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + (rc_outer_inner * 2)) + 65)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 104)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + (rc_outer_inner * 2)) + 65)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[(((rc_outer_inner * 128) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7))] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + (rc_outer_inner * 2)) + 128)]));
      conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 32)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + (rc_outer_inner * 2)) + 128)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 8)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + (rc_outer_inner * 2)) + 128)]));
      conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 40)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + (rc_outer_inner * 2)) + 128)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 64)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + (rc_outer_inner * 2)) + 129)]));
      conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 96)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + (rc_outer_inner * 2)) + 129)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 72)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + (rc_outer_inner * 2)) + 129)]));
      conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 104)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + (rc_outer_inner * 2)) + 129)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[(((rc_outer_inner * 128) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7))] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + (rc_outer_inner * 2)) + 192)]));
      conv2d_nchw[14] = (conv2d_nchw[14] + (pad_temp_shared[((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 32)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + (rc_outer_inner * 2)) + 192)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 8)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + (rc_outer_inner * 2)) + 192)]));
      conv2d_nchw[15] = (conv2d_nchw[15] + (pad_temp_shared[((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 40)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + (rc_outer_inner * 2)) + 192)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 64)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + (rc_outer_inner * 2)) + 193)]));
      conv2d_nchw[14] = (conv2d_nchw[14] + (pad_temp_shared[((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 96)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + (rc_outer_inner * 2)) + 193)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 72)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + (rc_outer_inner * 2)) + 193)]));
      conv2d_nchw[15] = (conv2d_nchw[15] + (pad_temp_shared[((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 104)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + (rc_outer_inner * 2)) + 193)]));
    }
  }
  for (int i1_inner = 0; i1_inner < 4; ++i1_inner) {
    for (int i2_inner = 0; i2_inner < 2; ++i2_inner) {
      compute[(((((((((((int)blockIdx.x) / 49) * 100352) + ((((int)threadIdx.x) >> 4) * 12544)) + (i1_inner * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (i2_inner * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7))] = max((conv2d_nchw[((i1_inner * 2) + i2_inner)] + bias[((((((int)blockIdx.x) / 49) * 32) + ((((int)threadIdx.x) >> 4) * 4)) + i1_inner)]), 0.000000e+00f);
      compute[((((((((((((int)blockIdx.x) / 49) * 100352) + ((((int)threadIdx.x) >> 4) * 12544)) + (i1_inner * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (i2_inner * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 224)] = max((conv2d_nchw[(((i1_inner * 2) + i2_inner) + 8)] + bias[((((((int)blockIdx.x) / 49) * 32) + ((((int)threadIdx.x) >> 4) * 4)) + i1_inner)]), 0.000000e+00f);
    }
  }
}


