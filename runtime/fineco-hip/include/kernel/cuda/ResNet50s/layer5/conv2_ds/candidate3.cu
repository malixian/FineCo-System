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
extern "C" __global__ void __launch_bounds__(56) candidate3(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[14];
  __shared__ float pad_temp_shared[1800];
  __shared__ float kernel_shared[1152];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[7] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[8] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[9] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[10] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[11] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[12] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  conv2d_nchw_local[13] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 64; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = (((15 <= ((int)threadIdx.x)) && (1 <= (((int)threadIdx.x) % 15))) ? data[((((rc_outer_outer * 1568) + ((((int)threadIdx.x) / 15) * 14)) + (((int)threadIdx.x) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 56)] = ((1 <= ((((int)threadIdx.x) + 11) % 15)) ? data[((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 56) / 15) * 14)) + ((((int)threadIdx.x) + 11) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 112)] = ((1 <= ((((int)threadIdx.x) + 7) % 15)) ? data[((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 112) / 15) * 14)) + ((((int)threadIdx.x) + 7) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 168)] = ((1 <= ((((int)threadIdx.x) + 3) % 15)) ? data[((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 168) / 15) * 14)) + ((((int)threadIdx.x) + 3) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 224)] = (((15 <= ((((int)threadIdx.x) + 224) % 225)) && (1 <= ((((int)threadIdx.x) + 14) % 15))) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 224) / 225) * 196)) + ((((((int)threadIdx.x) + 224) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 14) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 280)] = ((1 <= ((((int)threadIdx.x) + 10) % 15)) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 280) / 225) * 196)) + ((((((int)threadIdx.x) + 55) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 10) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 336)] = ((1 <= ((((int)threadIdx.x) + 6) % 15)) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 336) / 225) * 196)) + ((((((int)threadIdx.x) + 111) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 6) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 392)] = ((1 <= ((((int)threadIdx.x) + 2) % 15)) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 392) / 225) * 196)) + ((((((int)threadIdx.x) + 167) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 2) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 448)] = (((15 <= ((((int)threadIdx.x) + 223) % 225)) && (1 <= ((((int)threadIdx.x) + 13) % 15))) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 448) / 225) * 196)) + ((((((int)threadIdx.x) + 223) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 13) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 504)] = ((1 <= ((((int)threadIdx.x) + 9) % 15)) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 504) / 225) * 196)) + ((((((int)threadIdx.x) + 54) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 9) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 560)] = ((1 <= ((((int)threadIdx.x) + 5) % 15)) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 560) / 225) * 196)) + ((((((int)threadIdx.x) + 110) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 5) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 616)] = ((1 <= ((((int)threadIdx.x) + 1) % 15)) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 616) / 225) * 196)) + ((((((int)threadIdx.x) + 166) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 1) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 672)] = (((15 <= ((((int)threadIdx.x) + 222) % 225)) && (1 <= ((((int)threadIdx.x) + 12) % 15))) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 672) / 225) * 196)) + ((((((int)threadIdx.x) + 222) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 12) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 728)] = ((1 <= ((((int)threadIdx.x) + 8) % 15)) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 728) / 225) * 196)) + ((((((int)threadIdx.x) + 53) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 8) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 784)] = ((1 <= ((((int)threadIdx.x) + 4) % 15)) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 784) / 225) * 196)) + ((((((int)threadIdx.x) + 109) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 4) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 840)] = ((1 <= (((int)threadIdx.x) % 15)) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 840) / 225) * 196)) + (((((int)threadIdx.x) / 15) + 11) * 14)) + (((int)threadIdx.x) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 896)] = (((15 <= ((((int)threadIdx.x) + 221) % 225)) && (1 <= ((((int)threadIdx.x) + 11) % 15))) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 896) / 225) * 196)) + ((((((int)threadIdx.x) + 221) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 11) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 952)] = ((1 <= ((((int)threadIdx.x) + 7) % 15)) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 952) / 225) * 196)) + ((((((int)threadIdx.x) + 52) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 7) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1008)] = ((1 <= ((((int)threadIdx.x) + 3) % 15)) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 1008) / 225) * 196)) + ((((((int)threadIdx.x) + 108) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 3) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1064)] = ((1 <= ((((int)threadIdx.x) + 14) % 15)) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 1064) / 225) * 196)) + ((((((int)threadIdx.x) + 164) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 14) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1120)] = (((15 <= ((((int)threadIdx.x) + 220) % 225)) && (1 <= ((((int)threadIdx.x) + 10) % 15))) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 1120) / 225) * 196)) + ((((((int)threadIdx.x) + 220) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 10) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1176)] = ((1 <= ((((int)threadIdx.x) + 6) % 15)) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 1176) / 225) * 196)) + ((((((int)threadIdx.x) + 51) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 6) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1232)] = ((1 <= ((((int)threadIdx.x) + 2) % 15)) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 1232) / 225) * 196)) + ((((((int)threadIdx.x) + 107) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 2) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1288)] = ((1 <= ((((int)threadIdx.x) + 13) % 15)) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 1288) / 225) * 196)) + ((((((int)threadIdx.x) + 163) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 13) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1344)] = (((15 <= ((((int)threadIdx.x) + 219) % 225)) && (1 <= ((((int)threadIdx.x) + 9) % 15))) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 1344) / 225) * 196)) + ((((((int)threadIdx.x) + 219) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 9) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1400)] = ((1 <= ((((int)threadIdx.x) + 5) % 15)) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 1400) / 225) * 196)) + ((((((int)threadIdx.x) + 50) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 5) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1456)] = ((1 <= ((((int)threadIdx.x) + 1) % 15)) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 1456) / 225) * 196)) + ((((((int)threadIdx.x) + 106) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 1) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1512)] = ((1 <= ((((int)threadIdx.x) + 12) % 15)) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 1512) / 225) * 196)) + ((((((int)threadIdx.x) + 162) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 12) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1568)] = (((15 <= ((((int)threadIdx.x) + 218) % 225)) && (1 <= ((((int)threadIdx.x) + 8) % 15))) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 1568) / 225) * 196)) + ((((((int)threadIdx.x) + 218) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 8) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1624)] = ((1 <= ((((int)threadIdx.x) + 4) % 15)) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 1624) / 225) * 196)) + ((((((int)threadIdx.x) + 49) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 4) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1680)] = ((1 <= (((int)threadIdx.x) % 15)) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 1680) / 225) * 196)) + (((((int)threadIdx.x) / 15) + 7) * 14)) + (((int)threadIdx.x) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1736)] = ((1 <= ((((int)threadIdx.x) + 11) % 15)) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 1736) / 225) * 196)) + ((((((int)threadIdx.x) + 161) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 11) % 15)) - 15)] : 0.000000e+00f);
    if (((int)threadIdx.x) < 8) {
      pad_temp_shared[(((int)threadIdx.x) + 1792)] = data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 1792) / 225) * 196)) + ((((((int)threadIdx.x) + 217) % 225) / 15) * 14)) + (((int)threadIdx.x) + 7)) - 15)];
    }
    kernel_shared[((int)threadIdx.x)] = kernel[(((((int)blockIdx.x) * 73728) + (rc_outer_outer * 72)) + ((int)threadIdx.x))];
    kernel_shared[(((int)threadIdx.x) + 56)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 56) / 72) * 4608)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 56) % 72))];
    kernel_shared[(((int)threadIdx.x) + 112)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 112) / 72) * 4608)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 40) % 72))];
    kernel_shared[(((int)threadIdx.x) + 168)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 168) / 72) * 4608)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 24) % 72))];
    kernel_shared[(((int)threadIdx.x) + 224)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 224) / 72) * 4608)) + (rc_outer_outer * 72)) + (((int)threadIdx.x) + 8))];
    kernel_shared[(((int)threadIdx.x) + 280)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 280) / 72) * 4608)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 64) % 72))];
    kernel_shared[(((int)threadIdx.x) + 336)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 336) / 72) * 4608)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 48) % 72))];
    kernel_shared[(((int)threadIdx.x) + 392)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 392) / 72) * 4608)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 32) % 72))];
    kernel_shared[(((int)threadIdx.x) + 448)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 448) / 72) * 4608)) + (rc_outer_outer * 72)) + (((int)threadIdx.x) + 16))];
    kernel_shared[(((int)threadIdx.x) + 504)] = kernel[((((((int)blockIdx.x) * 73728) + (rc_outer_outer * 72)) + ((int)threadIdx.x)) + 32256)];
    kernel_shared[(((int)threadIdx.x) + 560)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 560) / 72) * 4608)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 56) % 72))];
    kernel_shared[(((int)threadIdx.x) + 616)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 616) / 72) * 4608)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 40) % 72))];
    kernel_shared[(((int)threadIdx.x) + 672)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 672) / 72) * 4608)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 24) % 72))];
    kernel_shared[(((int)threadIdx.x) + 728)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 728) / 72) * 4608)) + (rc_outer_outer * 72)) + (((int)threadIdx.x) + 8))];
    kernel_shared[(((int)threadIdx.x) + 784)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 784) / 72) * 4608)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 64) % 72))];
    kernel_shared[(((int)threadIdx.x) + 840)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 840) / 72) * 4608)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 48) % 72))];
    kernel_shared[(((int)threadIdx.x) + 896)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 896) / 72) * 4608)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 32) % 72))];
    kernel_shared[(((int)threadIdx.x) + 952)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 952) / 72) * 4608)) + (rc_outer_outer * 72)) + (((int)threadIdx.x) + 16))];
    kernel_shared[(((int)threadIdx.x) + 1008)] = kernel[((((((int)blockIdx.x) * 73728) + (rc_outer_outer * 72)) + ((int)threadIdx.x)) + 64512)];
    kernel_shared[(((int)threadIdx.x) + 1064)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 1064) / 72) * 4608)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 56) % 72))];
    if (((int)threadIdx.x) < 32) {
      kernel_shared[(((int)threadIdx.x) + 1120)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 1120) / 72) * 4608)) + (rc_outer_outer * 72)) + (((int)threadIdx.x) + 40))];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 2; ++rc_outer_inner) {
      for (int ry_outer_inner = 0; ry_outer_inner < 3; ++ry_outer_inner) {
        for (int yy_c_outer_inner = 0; yy_c_outer_inner < 7; ++yy_c_outer_inner) {
          conv2d_nchw_local[yy_c_outer_inner] = (conv2d_nchw_local[yy_c_outer_inner] + (pad_temp_shared[((((rc_outer_inner * 900) + (yy_c_outer_inner * 30)) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2))] * kernel_shared[((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3))]));
          conv2d_nchw_local[(yy_c_outer_inner + 7)] = (conv2d_nchw_local[(yy_c_outer_inner + 7)] + (pad_temp_shared[((((rc_outer_inner * 900) + (yy_c_outer_inner * 30)) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2))] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + 72)]));
          conv2d_nchw_local[yy_c_outer_inner] = (conv2d_nchw_local[yy_c_outer_inner] + (pad_temp_shared[(((((rc_outer_inner * 900) + (yy_c_outer_inner * 30)) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + 1)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + 1)]));
          conv2d_nchw_local[(yy_c_outer_inner + 7)] = (conv2d_nchw_local[(yy_c_outer_inner + 7)] + (pad_temp_shared[(((((rc_outer_inner * 900) + (yy_c_outer_inner * 30)) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + 1)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + 73)]));
          conv2d_nchw_local[yy_c_outer_inner] = (conv2d_nchw_local[yy_c_outer_inner] + (pad_temp_shared[(((((rc_outer_inner * 900) + (yy_c_outer_inner * 30)) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + 2)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + 2)]));
          conv2d_nchw_local[(yy_c_outer_inner + 7)] = (conv2d_nchw_local[(yy_c_outer_inner + 7)] + (pad_temp_shared[(((((rc_outer_inner * 900) + (yy_c_outer_inner * 30)) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + 2)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + 74)]));
          conv2d_nchw_local[yy_c_outer_inner] = (conv2d_nchw_local[yy_c_outer_inner] + (pad_temp_shared[(((((rc_outer_inner * 900) + (yy_c_outer_inner * 30)) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + 225)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + 9)]));
          conv2d_nchw_local[(yy_c_outer_inner + 7)] = (conv2d_nchw_local[(yy_c_outer_inner + 7)] + (pad_temp_shared[(((((rc_outer_inner * 900) + (yy_c_outer_inner * 30)) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + 225)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + 81)]));
          conv2d_nchw_local[yy_c_outer_inner] = (conv2d_nchw_local[yy_c_outer_inner] + (pad_temp_shared[(((((rc_outer_inner * 900) + (yy_c_outer_inner * 30)) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + 226)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + 10)]));
          conv2d_nchw_local[(yy_c_outer_inner + 7)] = (conv2d_nchw_local[(yy_c_outer_inner + 7)] + (pad_temp_shared[(((((rc_outer_inner * 900) + (yy_c_outer_inner * 30)) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + 226)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + 82)]));
          conv2d_nchw_local[yy_c_outer_inner] = (conv2d_nchw_local[yy_c_outer_inner] + (pad_temp_shared[(((((rc_outer_inner * 900) + (yy_c_outer_inner * 30)) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + 227)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + 11)]));
          conv2d_nchw_local[(yy_c_outer_inner + 7)] = (conv2d_nchw_local[(yy_c_outer_inner + 7)] + (pad_temp_shared[(((((rc_outer_inner * 900) + (yy_c_outer_inner * 30)) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + 227)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + 83)]));
          conv2d_nchw_local[yy_c_outer_inner] = (conv2d_nchw_local[yy_c_outer_inner] + (pad_temp_shared[(((((rc_outer_inner * 900) + (yy_c_outer_inner * 30)) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + 450)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + 18)]));
          conv2d_nchw_local[(yy_c_outer_inner + 7)] = (conv2d_nchw_local[(yy_c_outer_inner + 7)] + (pad_temp_shared[(((((rc_outer_inner * 900) + (yy_c_outer_inner * 30)) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + 450)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + 90)]));
          conv2d_nchw_local[yy_c_outer_inner] = (conv2d_nchw_local[yy_c_outer_inner] + (pad_temp_shared[(((((rc_outer_inner * 900) + (yy_c_outer_inner * 30)) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + 451)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + 19)]));
          conv2d_nchw_local[(yy_c_outer_inner + 7)] = (conv2d_nchw_local[(yy_c_outer_inner + 7)] + (pad_temp_shared[(((((rc_outer_inner * 900) + (yy_c_outer_inner * 30)) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + 451)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + 91)]));
          conv2d_nchw_local[yy_c_outer_inner] = (conv2d_nchw_local[yy_c_outer_inner] + (pad_temp_shared[(((((rc_outer_inner * 900) + (yy_c_outer_inner * 30)) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + 452)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + 20)]));
          conv2d_nchw_local[(yy_c_outer_inner + 7)] = (conv2d_nchw_local[(yy_c_outer_inner + 7)] + (pad_temp_shared[(((((rc_outer_inner * 900) + (yy_c_outer_inner * 30)) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + 452)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + 92)]));
          conv2d_nchw_local[yy_c_outer_inner] = (conv2d_nchw_local[yy_c_outer_inner] + (pad_temp_shared[(((((rc_outer_inner * 900) + (yy_c_outer_inner * 30)) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + 675)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + 27)]));
          conv2d_nchw_local[(yy_c_outer_inner + 7)] = (conv2d_nchw_local[(yy_c_outer_inner + 7)] + (pad_temp_shared[(((((rc_outer_inner * 900) + (yy_c_outer_inner * 30)) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + 675)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + 99)]));
          conv2d_nchw_local[yy_c_outer_inner] = (conv2d_nchw_local[yy_c_outer_inner] + (pad_temp_shared[(((((rc_outer_inner * 900) + (yy_c_outer_inner * 30)) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + 676)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + 28)]));
          conv2d_nchw_local[(yy_c_outer_inner + 7)] = (conv2d_nchw_local[(yy_c_outer_inner + 7)] + (pad_temp_shared[(((((rc_outer_inner * 900) + (yy_c_outer_inner * 30)) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + 676)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + 100)]));
          conv2d_nchw_local[yy_c_outer_inner] = (conv2d_nchw_local[yy_c_outer_inner] + (pad_temp_shared[(((((rc_outer_inner * 900) + (yy_c_outer_inner * 30)) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + 677)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + 29)]));
          conv2d_nchw_local[(yy_c_outer_inner + 7)] = (conv2d_nchw_local[(yy_c_outer_inner + 7)] + (pad_temp_shared[(((((rc_outer_inner * 900) + (yy_c_outer_inner * 30)) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + 677)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + 101)]));
        }
      }
    }
  }
  for (int ff_inner = 0; ff_inner < 2; ++ff_inner) {
    for (int yy_inner = 0; yy_inner < 7; ++yy_inner) {
      conv2d_nchw[(((((((int)blockIdx.x) * 784) + ((((int)threadIdx.x) / 7) * 98)) + (ff_inner * 49)) + (yy_inner * 7)) + (((int)threadIdx.x) % 7))] = conv2d_nchw_local[((ff_inner * 7) + yy_inner)];
    }
  }
}


