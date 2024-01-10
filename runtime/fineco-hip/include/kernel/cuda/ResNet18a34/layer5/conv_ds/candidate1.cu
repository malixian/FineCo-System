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
extern "C" __global__ void __launch_bounds__(112) candidate1(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[14];
  __shared__ float pad_temp_shared[1800];
  __shared__ float kernel_shared[2304];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  conv2d_nchw_local[7] = 0.000000e+00f;
  conv2d_nchw_local[8] = 0.000000e+00f;
  conv2d_nchw_local[9] = 0.000000e+00f;
  conv2d_nchw_local[10] = 0.000000e+00f;
  conv2d_nchw_local[11] = 0.000000e+00f;
  conv2d_nchw_local[12] = 0.000000e+00f;
  conv2d_nchw_local[13] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 32; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = (((15 <= ((int)threadIdx.x)) && (1 <= (((int)threadIdx.x) % 15))) ? data[((((rc_outer_outer * 1568) + ((((int)threadIdx.x) / 15) * 14)) + (((int)threadIdx.x) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 112)] = ((1 <= ((((int)threadIdx.x) + 7) % 15)) ? data[((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 112) / 15) * 14)) + ((((int)threadIdx.x) + 7) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 224)] = (((15 <= ((((int)threadIdx.x) + 224) % 225)) && (1 <= ((((int)threadIdx.x) + 14) % 15))) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 224) / 225) * 196)) + ((((((int)threadIdx.x) + 224) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 14) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 336)] = ((1 <= ((((int)threadIdx.x) + 6) % 15)) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 336) / 225) * 196)) + ((((((int)threadIdx.x) + 111) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 6) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 448)] = (((15 <= ((((int)threadIdx.x) + 223) % 225)) && (1 <= ((((int)threadIdx.x) + 13) % 15))) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 448) / 225) * 196)) + ((((((int)threadIdx.x) + 223) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 13) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 560)] = ((1 <= ((((int)threadIdx.x) + 5) % 15)) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 560) / 225) * 196)) + ((((((int)threadIdx.x) + 110) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 5) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 672)] = (((15 <= ((((int)threadIdx.x) + 222) % 225)) && (1 <= ((((int)threadIdx.x) + 12) % 15))) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 672) / 225) * 196)) + ((((((int)threadIdx.x) + 222) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 12) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 784)] = ((1 <= ((((int)threadIdx.x) + 4) % 15)) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 784) / 225) * 196)) + ((((((int)threadIdx.x) + 109) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 4) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 896)] = (((15 <= ((((int)threadIdx.x) + 221) % 225)) && (1 <= ((((int)threadIdx.x) + 11) % 15))) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 896) / 225) * 196)) + ((((((int)threadIdx.x) + 221) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 11) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1008)] = ((1 <= ((((int)threadIdx.x) + 3) % 15)) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 1008) / 225) * 196)) + ((((((int)threadIdx.x) + 108) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 3) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1120)] = (((15 <= ((((int)threadIdx.x) + 220) % 225)) && (1 <= ((((int)threadIdx.x) + 10) % 15))) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 1120) / 225) * 196)) + ((((((int)threadIdx.x) + 220) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 10) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1232)] = ((1 <= ((((int)threadIdx.x) + 2) % 15)) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 1232) / 225) * 196)) + ((((((int)threadIdx.x) + 107) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 2) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1344)] = (((15 <= ((((int)threadIdx.x) + 219) % 225)) && (1 <= ((((int)threadIdx.x) + 9) % 15))) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 1344) / 225) * 196)) + ((((((int)threadIdx.x) + 219) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 9) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1456)] = ((1 <= ((((int)threadIdx.x) + 1) % 15)) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 1456) / 225) * 196)) + ((((((int)threadIdx.x) + 106) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 1) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1568)] = (((15 <= ((((int)threadIdx.x) + 218) % 225)) && (1 <= ((((int)threadIdx.x) + 8) % 15))) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 1568) / 225) * 196)) + ((((((int)threadIdx.x) + 218) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 8) % 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1680)] = ((1 <= (((int)threadIdx.x) % 15)) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 1680) / 225) * 196)) + (((((int)threadIdx.x) / 15) + 7) * 14)) + (((int)threadIdx.x) % 15)) - 15)] : 0.000000e+00f);
    if (((int)threadIdx.x) < 8) {
      pad_temp_shared[(((int)threadIdx.x) + 1792)] = data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 1792) / 225) * 196)) + ((((((int)threadIdx.x) + 217) % 225) / 15) * 14)) + (((int)threadIdx.x) + 7)) - 15)];
    }
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)blockIdx.x) * 73728) + ((((int)threadIdx.x) / 72) * 2304)) + (rc_outer_outer * 72)) + (((int)threadIdx.x) % 72))];
    kernel_shared[(((int)threadIdx.x) + 112)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 112) / 72) * 2304)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 40) % 72))];
    kernel_shared[(((int)threadIdx.x) + 224)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 224) / 72) * 2304)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 8) % 72))];
    kernel_shared[(((int)threadIdx.x) + 336)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 336) / 72) * 2304)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 48) % 72))];
    kernel_shared[(((int)threadIdx.x) + 448)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 448) / 72) * 2304)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 16) % 72))];
    kernel_shared[(((int)threadIdx.x) + 560)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 560) / 72) * 2304)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 56) % 72))];
    kernel_shared[(((int)threadIdx.x) + 672)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 672) / 72) * 2304)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 24) % 72))];
    kernel_shared[(((int)threadIdx.x) + 784)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 784) / 72) * 2304)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 64) % 72))];
    kernel_shared[(((int)threadIdx.x) + 896)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 896) / 72) * 2304)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 32) % 72))];
    kernel_shared[(((int)threadIdx.x) + 1008)] = kernel[(((((((int)blockIdx.x) * 73728) + ((((int)threadIdx.x) / 72) * 2304)) + (rc_outer_outer * 72)) + (((int)threadIdx.x) % 72)) + 32256)];
    kernel_shared[(((int)threadIdx.x) + 1120)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 1120) / 72) * 2304)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 40) % 72))];
    kernel_shared[(((int)threadIdx.x) + 1232)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 1232) / 72) * 2304)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 8) % 72))];
    kernel_shared[(((int)threadIdx.x) + 1344)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 1344) / 72) * 2304)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 48) % 72))];
    kernel_shared[(((int)threadIdx.x) + 1456)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 1456) / 72) * 2304)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 16) % 72))];
    kernel_shared[(((int)threadIdx.x) + 1568)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 1568) / 72) * 2304)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 56) % 72))];
    kernel_shared[(((int)threadIdx.x) + 1680)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 1680) / 72) * 2304)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 24) % 72))];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 1792) / 72) * 2304)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 64) % 72))];
    kernel_shared[(((int)threadIdx.x) + 1904)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 1904) / 72) * 2304)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 32) % 72))];
    kernel_shared[(((int)threadIdx.x) + 2016)] = kernel[(((((((int)blockIdx.x) * 73728) + ((((int)threadIdx.x) / 72) * 2304)) + (rc_outer_outer * 72)) + (((int)threadIdx.x) % 72)) + 64512)];
    kernel_shared[(((int)threadIdx.x) + 2128)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 2128) / 72) * 2304)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 40) % 72))];
    if (((int)threadIdx.x) < 64) {
      kernel_shared[(((int)threadIdx.x) + 2240)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 2240) / 72) * 2304)) + (rc_outer_outer * 72)) + (((int)threadIdx.x) + 8))];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 2; ++rc_outer_inner) {
      for (int ry_outer_inner = 0; ry_outer_inner < 3; ++ry_outer_inner) {
        for (int rx_outer_inner = 0; rx_outer_inner < 3; ++rx_outer_inner) {
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 30)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 60)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 90)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 120)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 150)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 180)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner)]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 225)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 9)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 255)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 9)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 285)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 9)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 315)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 9)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 345)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 9)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 375)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 9)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 405)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 9)]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 450)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 18)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 480)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 18)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 510)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 18)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 540)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 18)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 570)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 18)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 600)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 18)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 630)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 18)]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 675)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 27)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 705)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 27)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 735)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 27)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 765)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 27)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 795)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 27)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 825)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 27)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 855)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 27)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 72)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 30)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 72)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 60)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 72)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 90)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 72)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 120)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 72)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 150)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 72)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 180)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 72)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 225)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 81)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 255)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 81)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 285)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 81)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 315)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 81)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 345)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 81)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 375)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 81)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 405)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 81)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 450)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 90)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 480)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 90)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 510)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 90)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 540)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 90)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 570)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 90)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 600)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 90)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 630)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 90)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 675)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 99)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 705)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 99)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 735)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 99)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 765)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 99)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 795)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 99)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 825)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 99)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[(((((rc_outer_inner * 900) + (ry_outer_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + rx_outer_inner) + 855)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 36)) + (ry_outer_inner * 3)) + rx_outer_inner) + 99)]));
        }
      }
    }
  }
  for (int ff_inner = 0; ff_inner < 2; ++ff_inner) {
    for (int yy_inner = 0; yy_inner < 7; ++yy_inner) {
      conv2d_nchw[(((((((int)blockIdx.x) * 1568) + ((((int)threadIdx.x) / 7) * 98)) + (ff_inner * 49)) + (yy_inner * 7)) + (((int)threadIdx.x) % 7))] = conv2d_nchw_local[((ff_inner * 7) + yy_inner)];
    }
  }
}


