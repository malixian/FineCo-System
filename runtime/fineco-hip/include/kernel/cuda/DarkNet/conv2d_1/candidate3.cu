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
extern "C" __global__ void __launch_bounds__(256) candidate3(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ compute, float* __restrict__ bias) {
  float conv2d_nchw[128];
  __shared__ float pad_temp_shared[3468];
  __shared__ float kernel_shared[864];
  for (int ff_inner_init = 0; ff_inner_init < 4; ++ff_inner_init) {
    conv2d_nchw[(ff_inner_init * 16)] = 0.000000e+00f;
    conv2d_nchw[((ff_inner_init * 16) + 64)] = 0.000000e+00f;
    conv2d_nchw[((ff_inner_init * 16) + 1)] = 0.000000e+00f;
    conv2d_nchw[((ff_inner_init * 16) + 65)] = 0.000000e+00f;
    conv2d_nchw[((ff_inner_init * 16) + 2)] = 0.000000e+00f;
    conv2d_nchw[((ff_inner_init * 16) + 66)] = 0.000000e+00f;
    conv2d_nchw[((ff_inner_init * 16) + 3)] = 0.000000e+00f;
    conv2d_nchw[((ff_inner_init * 16) + 67)] = 0.000000e+00f;
    conv2d_nchw[((ff_inner_init * 16) + 4)] = 0.000000e+00f;
    conv2d_nchw[((ff_inner_init * 16) + 68)] = 0.000000e+00f;
    conv2d_nchw[((ff_inner_init * 16) + 5)] = 0.000000e+00f;
    conv2d_nchw[((ff_inner_init * 16) + 69)] = 0.000000e+00f;
    conv2d_nchw[((ff_inner_init * 16) + 6)] = 0.000000e+00f;
    conv2d_nchw[((ff_inner_init * 16) + 70)] = 0.000000e+00f;
    conv2d_nchw[((ff_inner_init * 16) + 7)] = 0.000000e+00f;
    conv2d_nchw[((ff_inner_init * 16) + 71)] = 0.000000e+00f;
    conv2d_nchw[((ff_inner_init * 16) + 8)] = 0.000000e+00f;
    conv2d_nchw[((ff_inner_init * 16) + 72)] = 0.000000e+00f;
    conv2d_nchw[((ff_inner_init * 16) + 9)] = 0.000000e+00f;
    conv2d_nchw[((ff_inner_init * 16) + 73)] = 0.000000e+00f;
    conv2d_nchw[((ff_inner_init * 16) + 10)] = 0.000000e+00f;
    conv2d_nchw[((ff_inner_init * 16) + 74)] = 0.000000e+00f;
    conv2d_nchw[((ff_inner_init * 16) + 11)] = 0.000000e+00f;
    conv2d_nchw[((ff_inner_init * 16) + 75)] = 0.000000e+00f;
    conv2d_nchw[((ff_inner_init * 16) + 12)] = 0.000000e+00f;
    conv2d_nchw[((ff_inner_init * 16) + 76)] = 0.000000e+00f;
    conv2d_nchw[((ff_inner_init * 16) + 13)] = 0.000000e+00f;
    conv2d_nchw[((ff_inner_init * 16) + 77)] = 0.000000e+00f;
    conv2d_nchw[((ff_inner_init * 16) + 14)] = 0.000000e+00f;
    conv2d_nchw[((ff_inner_init * 16) + 78)] = 0.000000e+00f;
    conv2d_nchw[((ff_inner_init * 16) + 15)] = 0.000000e+00f;
    conv2d_nchw[((ff_inner_init * 16) + 79)] = 0.000000e+00f;
  }
  pad_temp_shared[((int)threadIdx.x)] = ((((1 <= (((((int)blockIdx.x) / 7) * 32) + (((int)threadIdx.x) / 34))) && (1 <= (((((int)blockIdx.x) % 7) * 32) + (((int)threadIdx.x) % 34)))) && ((((((int)blockIdx.x) % 7) * 32) + (((int)threadIdx.x) % 34)) < 225)) ? data[((((((((int)blockIdx.x) / 7) * 7168) + ((((int)threadIdx.x) / 34) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) % 34)) - 225)] : 0.000000e+00f);
  pad_temp_shared[(((int)threadIdx.x) + 256)] = (((1 <= (((((int)blockIdx.x) % 7) * 32) + ((((int)threadIdx.x) + 18) % 34))) && ((((((int)blockIdx.x) % 7) * 32) + ((((int)threadIdx.x) + 18) % 34)) < 225)) ? data[((((((((int)blockIdx.x) / 7) * 7168) + (((((int)threadIdx.x) + 256) / 34) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + ((((int)threadIdx.x) + 18) % 34)) - 225)] : 0.000000e+00f);
  pad_temp_shared[(((int)threadIdx.x) + 512)] = (((1 <= (((((int)blockIdx.x) % 7) * 32) + ((((int)threadIdx.x) + 2) % 34))) && ((((((int)blockIdx.x) % 7) * 32) + ((((int)threadIdx.x) + 2) % 34)) < 225)) ? data[((((((((int)blockIdx.x) / 7) * 7168) + (((((int)threadIdx.x) + 512) / 34) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + ((((int)threadIdx.x) + 2) % 34)) - 225)] : 0.000000e+00f);
  pad_temp_shared[(((int)threadIdx.x) + 768)] = (((1 <= (((((int)blockIdx.x) % 7) * 32) + ((((int)threadIdx.x) + 20) % 34))) && ((((((int)blockIdx.x) % 7) * 32) + ((((int)threadIdx.x) + 20) % 34)) < 225)) ? data[((((((((int)blockIdx.x) / 7) * 7168) + (((((int)threadIdx.x) + 768) / 34) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + ((((int)threadIdx.x) + 20) % 34)) - 225)] : 0.000000e+00f);
  pad_temp_shared[(((int)threadIdx.x) + 1024)] = (((((1 <= (((((int)blockIdx.x) / 7) * 32) + ((((((int)threadIdx.x) >> 1) + 512) % 578) / 17))) && ((((((int)blockIdx.x) / 7) * 32) + ((((((int)threadIdx.x) >> 1) + 512) % 578) / 17)) < 225)) && (1 <= (((((int)blockIdx.x) % 7) * 32) + ((((int)threadIdx.x) + 4) % 34)))) && ((((((int)blockIdx.x) % 7) * 32) + ((((int)threadIdx.x) + 4) % 34)) < 225)) ? data[((((((((((int)threadIdx.x) + 1024) / 1156) * 50176) + ((((int)blockIdx.x) / 7) * 7168)) + (((((((int)threadIdx.x) >> 1) + 512) % 578) / 17) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + ((((int)threadIdx.x) + 4) % 34)) - 225)] : 0.000000e+00f);
  pad_temp_shared[(((int)threadIdx.x) + 1280)] = (((1 <= (((((int)blockIdx.x) % 7) * 32) + ((((int)threadIdx.x) + 22) % 34))) && ((((((int)blockIdx.x) % 7) * 32) + ((((int)threadIdx.x) + 22) % 34)) < 225)) ? data[((((((((((int)threadIdx.x) + 1280) / 1156) * 50176) + ((((int)blockIdx.x) / 7) * 7168)) + (((((((int)threadIdx.x) >> 1) + 62) % 578) / 17) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + ((((int)threadIdx.x) + 22) % 34)) - 225)] : 0.000000e+00f);
  pad_temp_shared[(((int)threadIdx.x) + 1536)] = (((1 <= (((((int)blockIdx.x) % 7) * 32) + ((((int)threadIdx.x) + 6) % 34))) && ((((((int)blockIdx.x) % 7) * 32) + ((((int)threadIdx.x) + 6) % 34)) < 225)) ? data[((((((((((int)threadIdx.x) + 1536) / 1156) * 50176) + ((((int)blockIdx.x) / 7) * 7168)) + (((((((int)threadIdx.x) >> 1) + 190) % 578) / 17) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + ((((int)threadIdx.x) + 6) % 34)) - 225)] : 0.000000e+00f);
  pad_temp_shared[(((int)threadIdx.x) + 1792)] = (((1 <= (((((int)blockIdx.x) % 7) * 32) + ((((int)threadIdx.x) + 24) % 34))) && ((((((int)blockIdx.x) % 7) * 32) + ((((int)threadIdx.x) + 24) % 34)) < 225)) ? data[((((((((((int)threadIdx.x) + 1792) / 1156) * 50176) + ((((int)blockIdx.x) / 7) * 7168)) + (((((((int)threadIdx.x) >> 1) + 318) % 578) / 17) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + ((((int)threadIdx.x) + 24) % 34)) - 225)] : 0.000000e+00f);
  pad_temp_shared[(((int)threadIdx.x) + 2048)] = (((((((((int)blockIdx.x) / 7) * 32) + ((((((int)threadIdx.x) >> 1) + 446) % 578) / 17)) < 225) && (1 <= (((((int)blockIdx.x) % 7) * 32) + ((((int)threadIdx.x) + 8) % 34)))) && ((((((int)blockIdx.x) % 7) * 32) + ((((int)threadIdx.x) + 8) % 34)) < 225)) ? data[((((((((((int)threadIdx.x) + 2048) / 1156) * 50176) + ((((int)blockIdx.x) / 7) * 7168)) + (((((((int)threadIdx.x) >> 1) + 446) % 578) / 17) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + ((((int)threadIdx.x) + 8) % 34)) - 225)] : 0.000000e+00f);
  pad_temp_shared[(((int)threadIdx.x) + 2304)] = (((((1 <= (((((int)blockIdx.x) / 7) * 32) + ((((((int)threadIdx.x) >> 1) + 574) % 578) / 17))) && ((((((int)blockIdx.x) / 7) * 32) + ((((((int)threadIdx.x) >> 1) + 574) % 578) / 17)) < 225)) && (1 <= (((((int)blockIdx.x) % 7) * 32) + ((((int)threadIdx.x) + 26) % 34)))) && ((((((int)blockIdx.x) % 7) * 32) + ((((int)threadIdx.x) + 26) % 34)) < 225)) ? data[((((((((((int)threadIdx.x) + 2304) / 1156) * 50176) + ((((int)blockIdx.x) / 7) * 7168)) + (((((((int)threadIdx.x) >> 1) + 574) % 578) / 17) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + ((((int)threadIdx.x) + 26) % 34)) - 225)] : 0.000000e+00f);
  pad_temp_shared[(((int)threadIdx.x) + 2560)] = (((1 <= (((((int)blockIdx.x) % 7) * 32) + ((((int)threadIdx.x) + 10) % 34))) && ((((((int)blockIdx.x) % 7) * 32) + ((((int)threadIdx.x) + 10) % 34)) < 225)) ? data[((((((((((int)threadIdx.x) + 2560) / 1156) * 50176) + ((((int)blockIdx.x) / 7) * 7168)) + (((((((int)threadIdx.x) >> 1) + 124) % 578) / 17) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + ((((int)threadIdx.x) + 10) % 34)) - 225)] : 0.000000e+00f);
  pad_temp_shared[(((int)threadIdx.x) + 2816)] = (((1 <= (((((int)blockIdx.x) % 7) * 32) + ((((int)threadIdx.x) + 28) % 34))) && ((((((int)blockIdx.x) % 7) * 32) + ((((int)threadIdx.x) + 28) % 34)) < 225)) ? data[((((((((((int)threadIdx.x) + 2816) / 1156) * 50176) + ((((int)blockIdx.x) / 7) * 7168)) + (((((((int)threadIdx.x) >> 1) + 252) % 578) / 17) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + ((((int)threadIdx.x) + 28) % 34)) - 225)] : 0.000000e+00f);
  pad_temp_shared[(((int)threadIdx.x) + 3072)] = (((1 <= (((((int)blockIdx.x) % 7) * 32) + ((((int)threadIdx.x) + 12) % 34))) && ((((((int)blockIdx.x) % 7) * 32) + ((((int)threadIdx.x) + 12) % 34)) < 225)) ? data[((((((((((int)threadIdx.x) + 3072) / 1156) * 50176) + ((((int)blockIdx.x) / 7) * 7168)) + (((((((int)threadIdx.x) >> 1) + 380) % 578) / 17) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + ((((int)threadIdx.x) + 12) % 34)) - 225)] : 0.000000e+00f);
  if (((int)threadIdx.x) < 140) {
    pad_temp_shared[(((int)threadIdx.x) + 3328)] = (((((((((int)blockIdx.x) / 7) * 32) + ((((((int)threadIdx.x) >> 1) + 508) % 578) / 17)) < 225) && (1 <= (((((int)blockIdx.x) % 7) * 32) + ((((int)threadIdx.x) + 30) % 34)))) && ((((((int)blockIdx.x) % 7) * 32) + ((((int)threadIdx.x) + 30) % 34)) < 225)) ? data[((((((((((int)threadIdx.x) + 3328) / 1156) * 50176) + ((((int)blockIdx.x) / 7) * 7168)) + (((((((int)threadIdx.x) >> 1) + 508) % 578) / 17) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + ((((int)threadIdx.x) + 30) % 34)) - 225)] : 0.000000e+00f);
  }
  kernel_shared[((int)threadIdx.x)] = kernel[((int)threadIdx.x)];
  kernel_shared[(((int)threadIdx.x) + 256)] = kernel[(((int)threadIdx.x) + 256)];
  kernel_shared[(((int)threadIdx.x) + 512)] = kernel[(((int)threadIdx.x) + 512)];
  if (((int)threadIdx.x) < 96) {
    kernel_shared[(((int)threadIdx.x) + 768)] = kernel[(((int)threadIdx.x) + 768)];
  }
  __syncthreads();
  for (int rc_outer_inner = 0; rc_outer_inner < 3; ++rc_outer_inner) {
    for (int ry_inner = 0; ry_inner < 3; ++ry_inner) {
      for (int rx_inner = 0; rx_inner < 3; ++rx_inner) {
        for (int ff_inner = 0; ff_inner < 4; ++ff_inner) {
          conv2d_nchw[(ff_inner * 16)] = (conv2d_nchw[(ff_inner * 16)] + (pad_temp_shared[((((rc_outer_inner * 1156) + (ry_inner * 34)) + rx_inner) + (((int)threadIdx.x) & 31))] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 108) + (ff_inner * 27)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
          conv2d_nchw[((ff_inner * 16) + 64)] = (conv2d_nchw[((ff_inner * 16) + 64)] + (pad_temp_shared[(((((rc_outer_inner * 1156) + (ry_inner * 34)) + rx_inner) + (((int)threadIdx.x) & 31)) + 544)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 108) + (ff_inner * 27)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
          conv2d_nchw[((ff_inner * 16) + 1)] = (conv2d_nchw[((ff_inner * 16) + 1)] + (pad_temp_shared[(((((rc_outer_inner * 1156) + (ry_inner * 34)) + rx_inner) + (((int)threadIdx.x) & 31)) + 34)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 108) + (ff_inner * 27)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
          conv2d_nchw[((ff_inner * 16) + 65)] = (conv2d_nchw[((ff_inner * 16) + 65)] + (pad_temp_shared[(((((rc_outer_inner * 1156) + (ry_inner * 34)) + rx_inner) + (((int)threadIdx.x) & 31)) + 578)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 108) + (ff_inner * 27)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
          conv2d_nchw[((ff_inner * 16) + 2)] = (conv2d_nchw[((ff_inner * 16) + 2)] + (pad_temp_shared[(((((rc_outer_inner * 1156) + (ry_inner * 34)) + rx_inner) + (((int)threadIdx.x) & 31)) + 68)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 108) + (ff_inner * 27)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
          conv2d_nchw[((ff_inner * 16) + 66)] = (conv2d_nchw[((ff_inner * 16) + 66)] + (pad_temp_shared[(((((rc_outer_inner * 1156) + (ry_inner * 34)) + rx_inner) + (((int)threadIdx.x) & 31)) + 612)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 108) + (ff_inner * 27)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
          conv2d_nchw[((ff_inner * 16) + 3)] = (conv2d_nchw[((ff_inner * 16) + 3)] + (pad_temp_shared[(((((rc_outer_inner * 1156) + (ry_inner * 34)) + rx_inner) + (((int)threadIdx.x) & 31)) + 102)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 108) + (ff_inner * 27)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
          conv2d_nchw[((ff_inner * 16) + 67)] = (conv2d_nchw[((ff_inner * 16) + 67)] + (pad_temp_shared[(((((rc_outer_inner * 1156) + (ry_inner * 34)) + rx_inner) + (((int)threadIdx.x) & 31)) + 646)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 108) + (ff_inner * 27)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
          conv2d_nchw[((ff_inner * 16) + 4)] = (conv2d_nchw[((ff_inner * 16) + 4)] + (pad_temp_shared[(((((rc_outer_inner * 1156) + (ry_inner * 34)) + rx_inner) + (((int)threadIdx.x) & 31)) + 136)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 108) + (ff_inner * 27)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
          conv2d_nchw[((ff_inner * 16) + 68)] = (conv2d_nchw[((ff_inner * 16) + 68)] + (pad_temp_shared[(((((rc_outer_inner * 1156) + (ry_inner * 34)) + rx_inner) + (((int)threadIdx.x) & 31)) + 680)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 108) + (ff_inner * 27)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
          conv2d_nchw[((ff_inner * 16) + 5)] = (conv2d_nchw[((ff_inner * 16) + 5)] + (pad_temp_shared[(((((rc_outer_inner * 1156) + (ry_inner * 34)) + rx_inner) + (((int)threadIdx.x) & 31)) + 170)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 108) + (ff_inner * 27)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
          conv2d_nchw[((ff_inner * 16) + 69)] = (conv2d_nchw[((ff_inner * 16) + 69)] + (pad_temp_shared[(((((rc_outer_inner * 1156) + (ry_inner * 34)) + rx_inner) + (((int)threadIdx.x) & 31)) + 714)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 108) + (ff_inner * 27)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
          conv2d_nchw[((ff_inner * 16) + 6)] = (conv2d_nchw[((ff_inner * 16) + 6)] + (pad_temp_shared[(((((rc_outer_inner * 1156) + (ry_inner * 34)) + rx_inner) + (((int)threadIdx.x) & 31)) + 204)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 108) + (ff_inner * 27)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
          conv2d_nchw[((ff_inner * 16) + 70)] = (conv2d_nchw[((ff_inner * 16) + 70)] + (pad_temp_shared[(((((rc_outer_inner * 1156) + (ry_inner * 34)) + rx_inner) + (((int)threadIdx.x) & 31)) + 748)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 108) + (ff_inner * 27)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
          conv2d_nchw[((ff_inner * 16) + 7)] = (conv2d_nchw[((ff_inner * 16) + 7)] + (pad_temp_shared[(((((rc_outer_inner * 1156) + (ry_inner * 34)) + rx_inner) + (((int)threadIdx.x) & 31)) + 238)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 108) + (ff_inner * 27)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
          conv2d_nchw[((ff_inner * 16) + 71)] = (conv2d_nchw[((ff_inner * 16) + 71)] + (pad_temp_shared[(((((rc_outer_inner * 1156) + (ry_inner * 34)) + rx_inner) + (((int)threadIdx.x) & 31)) + 782)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 108) + (ff_inner * 27)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
          conv2d_nchw[((ff_inner * 16) + 8)] = (conv2d_nchw[((ff_inner * 16) + 8)] + (pad_temp_shared[(((((rc_outer_inner * 1156) + (ry_inner * 34)) + rx_inner) + (((int)threadIdx.x) & 31)) + 272)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 108) + (ff_inner * 27)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
          conv2d_nchw[((ff_inner * 16) + 72)] = (conv2d_nchw[((ff_inner * 16) + 72)] + (pad_temp_shared[(((((rc_outer_inner * 1156) + (ry_inner * 34)) + rx_inner) + (((int)threadIdx.x) & 31)) + 816)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 108) + (ff_inner * 27)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
          conv2d_nchw[((ff_inner * 16) + 9)] = (conv2d_nchw[((ff_inner * 16) + 9)] + (pad_temp_shared[(((((rc_outer_inner * 1156) + (ry_inner * 34)) + rx_inner) + (((int)threadIdx.x) & 31)) + 306)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 108) + (ff_inner * 27)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
          conv2d_nchw[((ff_inner * 16) + 73)] = (conv2d_nchw[((ff_inner * 16) + 73)] + (pad_temp_shared[(((((rc_outer_inner * 1156) + (ry_inner * 34)) + rx_inner) + (((int)threadIdx.x) & 31)) + 850)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 108) + (ff_inner * 27)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
          conv2d_nchw[((ff_inner * 16) + 10)] = (conv2d_nchw[((ff_inner * 16) + 10)] + (pad_temp_shared[(((((rc_outer_inner * 1156) + (ry_inner * 34)) + rx_inner) + (((int)threadIdx.x) & 31)) + 340)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 108) + (ff_inner * 27)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
          conv2d_nchw[((ff_inner * 16) + 74)] = (conv2d_nchw[((ff_inner * 16) + 74)] + (pad_temp_shared[(((((rc_outer_inner * 1156) + (ry_inner * 34)) + rx_inner) + (((int)threadIdx.x) & 31)) + 884)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 108) + (ff_inner * 27)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
          conv2d_nchw[((ff_inner * 16) + 11)] = (conv2d_nchw[((ff_inner * 16) + 11)] + (pad_temp_shared[(((((rc_outer_inner * 1156) + (ry_inner * 34)) + rx_inner) + (((int)threadIdx.x) & 31)) + 374)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 108) + (ff_inner * 27)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
          conv2d_nchw[((ff_inner * 16) + 75)] = (conv2d_nchw[((ff_inner * 16) + 75)] + (pad_temp_shared[(((((rc_outer_inner * 1156) + (ry_inner * 34)) + rx_inner) + (((int)threadIdx.x) & 31)) + 918)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 108) + (ff_inner * 27)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
          conv2d_nchw[((ff_inner * 16) + 12)] = (conv2d_nchw[((ff_inner * 16) + 12)] + (pad_temp_shared[(((((rc_outer_inner * 1156) + (ry_inner * 34)) + rx_inner) + (((int)threadIdx.x) & 31)) + 408)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 108) + (ff_inner * 27)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
          conv2d_nchw[((ff_inner * 16) + 76)] = (conv2d_nchw[((ff_inner * 16) + 76)] + (pad_temp_shared[(((((rc_outer_inner * 1156) + (ry_inner * 34)) + rx_inner) + (((int)threadIdx.x) & 31)) + 952)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 108) + (ff_inner * 27)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
          conv2d_nchw[((ff_inner * 16) + 13)] = (conv2d_nchw[((ff_inner * 16) + 13)] + (pad_temp_shared[(((((rc_outer_inner * 1156) + (ry_inner * 34)) + rx_inner) + (((int)threadIdx.x) & 31)) + 442)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 108) + (ff_inner * 27)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
          conv2d_nchw[((ff_inner * 16) + 77)] = (conv2d_nchw[((ff_inner * 16) + 77)] + (pad_temp_shared[(((((rc_outer_inner * 1156) + (ry_inner * 34)) + rx_inner) + (((int)threadIdx.x) & 31)) + 986)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 108) + (ff_inner * 27)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
          conv2d_nchw[((ff_inner * 16) + 14)] = (conv2d_nchw[((ff_inner * 16) + 14)] + (pad_temp_shared[(((((rc_outer_inner * 1156) + (ry_inner * 34)) + rx_inner) + (((int)threadIdx.x) & 31)) + 476)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 108) + (ff_inner * 27)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
          conv2d_nchw[((ff_inner * 16) + 78)] = (conv2d_nchw[((ff_inner * 16) + 78)] + (pad_temp_shared[(((((rc_outer_inner * 1156) + (ry_inner * 34)) + rx_inner) + (((int)threadIdx.x) & 31)) + 1020)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 108) + (ff_inner * 27)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
          conv2d_nchw[((ff_inner * 16) + 15)] = (conv2d_nchw[((ff_inner * 16) + 15)] + (pad_temp_shared[(((((rc_outer_inner * 1156) + (ry_inner * 34)) + rx_inner) + (((int)threadIdx.x) & 31)) + 510)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 108) + (ff_inner * 27)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
          conv2d_nchw[((ff_inner * 16) + 79)] = (conv2d_nchw[((ff_inner * 16) + 79)] + (pad_temp_shared[(((((rc_outer_inner * 1156) + (ry_inner * 34)) + rx_inner) + (((int)threadIdx.x) & 31)) + 1054)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 108) + (ff_inner * 27)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
        }
      }
    }
  }
  for (int i1_inner = 0; i1_inner < 4; ++i1_inner) {
    for (int i2_inner = 0; i2_inner < 16; ++i2_inner) {
      compute[(((((((((int)threadIdx.x) >> 5) * 200704) + (i1_inner * 50176)) + ((((int)blockIdx.x) / 7) * 7168)) + (i2_inner * 224)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) & 31))] = max((conv2d_nchw[((i1_inner * 16) + i2_inner)] + bias[(((((int)threadIdx.x) >> 5) * 4) + i1_inner)]), 0.000000e+00f);
      compute[((((((((((int)threadIdx.x) >> 5) * 200704) + (i1_inner * 50176)) + ((((int)blockIdx.x) / 7) * 7168)) + (i2_inner * 224)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) & 31)) + 3584)] = max((conv2d_nchw[(((i1_inner * 16) + i2_inner) + 64)] + bias[(((((int)threadIdx.x) >> 5) * 4) + i1_inner)]), 0.000000e+00f);
    }
  }
}


