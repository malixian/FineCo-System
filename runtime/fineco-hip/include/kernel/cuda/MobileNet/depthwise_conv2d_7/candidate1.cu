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
extern "C" __global__ void __launch_bounds__(392) candidate1(float* __restrict__ Input, float* __restrict__ kernel, float* __restrict__ compute) {
  float DepthwiseConv2d[16];
  __shared__ float PaddedInput_shared[8192];
  __shared__ float kernel_shared[288];
  DepthwiseConv2d[0] = 0.000000e+00f;
  DepthwiseConv2d[1] = 0.000000e+00f;
  DepthwiseConv2d[2] = 0.000000e+00f;
  DepthwiseConv2d[3] = 0.000000e+00f;
  DepthwiseConv2d[4] = 0.000000e+00f;
  DepthwiseConv2d[5] = 0.000000e+00f;
  DepthwiseConv2d[6] = 0.000000e+00f;
  DepthwiseConv2d[7] = 0.000000e+00f;
  DepthwiseConv2d[8] = 0.000000e+00f;
  DepthwiseConv2d[9] = 0.000000e+00f;
  DepthwiseConv2d[10] = 0.000000e+00f;
  DepthwiseConv2d[11] = 0.000000e+00f;
  DepthwiseConv2d[12] = 0.000000e+00f;
  DepthwiseConv2d[13] = 0.000000e+00f;
  DepthwiseConv2d[14] = 0.000000e+00f;
  DepthwiseConv2d[15] = 0.000000e+00f;
  PaddedInput_shared[((int)threadIdx.x)] = (((((16 <= (((int)threadIdx.x) & 255)) && ((((int)threadIdx.x) & 255) < 240)) && (1 <= (((int)threadIdx.x) & 15))) && ((((int)threadIdx.x) & 15) < 15)) ? Input[(((((((int)blockIdx.x) * 6272) + ((((int)threadIdx.x) >> 8) * 196)) + (((((int)threadIdx.x) & 255) >> 4) * 14)) + (((int)threadIdx.x) & 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 392)] = (((((2 <= (((((int)threadIdx.x) >> 3) + 17) & 31)) && (((((int)threadIdx.x) + 136) & 255) < 240)) && (1 <= ((((int)threadIdx.x) + 8) & 15))) && (((((int)threadIdx.x) + 8) & 15) < 15)) ? Input[(((((((int)blockIdx.x) * 6272) + (((((int)threadIdx.x) + 392) >> 8) * 196)) + (((((((int)threadIdx.x) >> 3) + 17) & 31) >> 1) * 14)) + ((((int)threadIdx.x) + 8) & 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 784)] = (((((1 <= (((((int)threadIdx.x) >> 4) + 1) & 15)) && (((((int)threadIdx.x) + 16) & 255) < 240)) && (1 <= (((int)threadIdx.x) & 15))) && ((((int)threadIdx.x) & 15) < 15)) ? Input[(((((((int)blockIdx.x) * 6272) + (((((int)threadIdx.x) + 784) >> 8) * 196)) + ((((((int)threadIdx.x) >> 4) + 1) & 15) * 14)) + (((int)threadIdx.x) & 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 1176)] = (((((2 <= (((((int)threadIdx.x) >> 3) + 19) & 31)) && (((((int)threadIdx.x) + 152) & 255) < 240)) && (1 <= ((((int)threadIdx.x) + 8) & 15))) && (((((int)threadIdx.x) + 8) & 15) < 15)) ? Input[(((((((int)blockIdx.x) * 6272) + (((((int)threadIdx.x) + 1176) >> 8) * 196)) + (((((((int)threadIdx.x) >> 3) + 19) & 31) >> 1) * 14)) + ((((int)threadIdx.x) + 8) & 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 1568)] = (((((1 <= (((((int)threadIdx.x) >> 4) + 2) & 15)) && (((((int)threadIdx.x) + 32) & 255) < 240)) && (1 <= (((int)threadIdx.x) & 15))) && ((((int)threadIdx.x) & 15) < 15)) ? Input[(((((((int)blockIdx.x) * 6272) + (((((int)threadIdx.x) + 1568) >> 8) * 196)) + ((((((int)threadIdx.x) >> 4) + 2) & 15) * 14)) + (((int)threadIdx.x) & 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 1960)] = (((((2 <= (((((int)threadIdx.x) >> 3) + 21) & 31)) && (((((int)threadIdx.x) + 168) & 255) < 240)) && (1 <= ((((int)threadIdx.x) + 8) & 15))) && (((((int)threadIdx.x) + 8) & 15) < 15)) ? Input[(((((((int)blockIdx.x) * 6272) + (((((int)threadIdx.x) + 1960) >> 8) * 196)) + (((((((int)threadIdx.x) >> 3) + 21) & 31) >> 1) * 14)) + ((((int)threadIdx.x) + 8) & 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 2352)] = (((((1 <= (((((int)threadIdx.x) >> 4) + 3) & 15)) && (((((int)threadIdx.x) + 48) & 255) < 240)) && (1 <= (((int)threadIdx.x) & 15))) && ((((int)threadIdx.x) & 15) < 15)) ? Input[(((((((int)blockIdx.x) * 6272) + (((((int)threadIdx.x) + 2352) >> 8) * 196)) + ((((((int)threadIdx.x) >> 4) + 3) & 15) * 14)) + (((int)threadIdx.x) & 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 2744)] = (((((2 <= (((((int)threadIdx.x) >> 3) + 23) & 31)) && (((((int)threadIdx.x) + 184) & 255) < 240)) && (1 <= ((((int)threadIdx.x) + 8) & 15))) && (((((int)threadIdx.x) + 8) & 15) < 15)) ? Input[(((((((int)blockIdx.x) * 6272) + (((((int)threadIdx.x) + 2744) >> 8) * 196)) + (((((((int)threadIdx.x) >> 3) + 23) & 31) >> 1) * 14)) + ((((int)threadIdx.x) + 8) & 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 3136)] = (((((1 <= (((((int)threadIdx.x) >> 4) + 4) & 15)) && (((((int)threadIdx.x) + 64) & 255) < 240)) && (1 <= (((int)threadIdx.x) & 15))) && ((((int)threadIdx.x) & 15) < 15)) ? Input[(((((((int)blockIdx.x) * 6272) + (((((int)threadIdx.x) + 3136) >> 8) * 196)) + ((((((int)threadIdx.x) >> 4) + 4) & 15) * 14)) + (((int)threadIdx.x) & 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 3528)] = (((((2 <= (((((int)threadIdx.x) >> 3) + 25) & 31)) && (((((int)threadIdx.x) + 200) & 255) < 240)) && (1 <= ((((int)threadIdx.x) + 8) & 15))) && (((((int)threadIdx.x) + 8) & 15) < 15)) ? Input[(((((((int)blockIdx.x) * 6272) + (((((int)threadIdx.x) + 3528) >> 8) * 196)) + (((((((int)threadIdx.x) >> 3) + 25) & 31) >> 1) * 14)) + ((((int)threadIdx.x) + 8) & 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 3920)] = (((((1 <= (((((int)threadIdx.x) >> 4) + 5) & 15)) && (((((int)threadIdx.x) + 80) & 255) < 240)) && (1 <= (((int)threadIdx.x) & 15))) && ((((int)threadIdx.x) & 15) < 15)) ? Input[(((((((int)blockIdx.x) * 6272) + (((((int)threadIdx.x) + 3920) >> 8) * 196)) + ((((((int)threadIdx.x) >> 4) + 5) & 15) * 14)) + (((int)threadIdx.x) & 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 4312)] = (((((2 <= (((((int)threadIdx.x) >> 3) + 27) & 31)) && (((((int)threadIdx.x) + 216) & 255) < 240)) && (1 <= ((((int)threadIdx.x) + 8) & 15))) && (((((int)threadIdx.x) + 8) & 15) < 15)) ? Input[(((((((int)blockIdx.x) * 6272) + (((((int)threadIdx.x) + 4312) >> 8) * 196)) + (((((((int)threadIdx.x) >> 3) + 27) & 31) >> 1) * 14)) + ((((int)threadIdx.x) + 8) & 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 4704)] = (((((1 <= (((((int)threadIdx.x) >> 4) + 6) & 15)) && (((((int)threadIdx.x) + 96) & 255) < 240)) && (1 <= (((int)threadIdx.x) & 15))) && ((((int)threadIdx.x) & 15) < 15)) ? Input[(((((((int)blockIdx.x) * 6272) + (((((int)threadIdx.x) + 4704) >> 8) * 196)) + ((((((int)threadIdx.x) >> 4) + 6) & 15) * 14)) + (((int)threadIdx.x) & 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 5096)] = (((((2 <= (((((int)threadIdx.x) >> 3) + 29) & 31)) && (((((int)threadIdx.x) + 232) & 255) < 240)) && (1 <= ((((int)threadIdx.x) + 8) & 15))) && (((((int)threadIdx.x) + 8) & 15) < 15)) ? Input[(((((((int)blockIdx.x) * 6272) + (((((int)threadIdx.x) + 5096) >> 8) * 196)) + (((((((int)threadIdx.x) >> 3) + 29) & 31) >> 1) * 14)) + ((((int)threadIdx.x) + 8) & 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 5488)] = (((((1 <= (((((int)threadIdx.x) >> 4) + 7) & 15)) && (((((int)threadIdx.x) + 112) & 255) < 240)) && (1 <= (((int)threadIdx.x) & 15))) && ((((int)threadIdx.x) & 15) < 15)) ? Input[(((((((int)blockIdx.x) * 6272) + (((((int)threadIdx.x) + 5488) >> 8) * 196)) + ((((((int)threadIdx.x) >> 4) + 7) & 15) * 14)) + (((int)threadIdx.x) & 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 5880)] = (((((2 <= (((((int)threadIdx.x) >> 3) + 31) & 31)) && (((((int)threadIdx.x) + 248) & 255) < 240)) && (1 <= ((((int)threadIdx.x) + 8) & 15))) && (((((int)threadIdx.x) + 8) & 15) < 15)) ? Input[(((((((int)blockIdx.x) * 6272) + (((((int)threadIdx.x) + 5880) >> 8) * 196)) + (((((((int)threadIdx.x) >> 3) + 31) & 31) >> 1) * 14)) + ((((int)threadIdx.x) + 8) & 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 6272)] = (((((1 <= (((((int)threadIdx.x) >> 4) + 8) & 15)) && (((((int)threadIdx.x) + 128) & 255) < 240)) && (1 <= (((int)threadIdx.x) & 15))) && ((((int)threadIdx.x) & 15) < 15)) ? Input[(((((((int)blockIdx.x) * 6272) + (((((int)threadIdx.x) + 6272) >> 8) * 196)) + ((((((int)threadIdx.x) >> 4) + 8) & 15) * 14)) + (((int)threadIdx.x) & 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 6664)] = (((((2 <= (((((int)threadIdx.x) >> 3) + 1) & 31)) && (((((int)threadIdx.x) + 8) & 255) < 240)) && (1 <= ((((int)threadIdx.x) + 8) & 15))) && (((((int)threadIdx.x) + 8) & 15) < 15)) ? Input[(((((((int)blockIdx.x) * 6272) + (((((int)threadIdx.x) + 6664) >> 8) * 196)) + (((((((int)threadIdx.x) >> 3) + 1) & 31) >> 1) * 14)) + ((((int)threadIdx.x) + 8) & 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 7056)] = (((((1 <= (((((int)threadIdx.x) >> 4) + 9) & 15)) && (((((int)threadIdx.x) + 144) & 255) < 240)) && (1 <= (((int)threadIdx.x) & 15))) && ((((int)threadIdx.x) & 15) < 15)) ? Input[(((((((int)blockIdx.x) * 6272) + (((((int)threadIdx.x) + 7056) >> 8) * 196)) + ((((((int)threadIdx.x) >> 4) + 9) & 15) * 14)) + (((int)threadIdx.x) & 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 7448)] = (((((2 <= (((((int)threadIdx.x) >> 3) + 3) & 31)) && (((((int)threadIdx.x) + 24) & 255) < 240)) && (1 <= ((((int)threadIdx.x) + 8) & 15))) && (((((int)threadIdx.x) + 8) & 15) < 15)) ? Input[(((((((int)blockIdx.x) * 6272) + (((((int)threadIdx.x) + 7448) >> 8) * 196)) + (((((((int)threadIdx.x) >> 3) + 3) & 31) >> 1) * 14)) + ((((int)threadIdx.x) + 8) & 15)) - 15)] : 0.000000e+00f);
  if (((int)threadIdx.x) < 352) {
    PaddedInput_shared[(((int)threadIdx.x) + 7840)] = (((((1 <= (((((int)threadIdx.x) >> 4) + 10) & 15)) && (((((int)threadIdx.x) + 160) & 255) < 240)) && (1 <= (((int)threadIdx.x) & 15))) && ((((int)threadIdx.x) & 15) < 15)) ? Input[(((((((int)blockIdx.x) * 6272) + (((((int)threadIdx.x) + 7840) >> 8) * 196)) + ((((((int)threadIdx.x) >> 4) + 10) & 15) * 14)) + (((int)threadIdx.x) & 15)) - 15)] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 288) {
    kernel_shared[((int)threadIdx.x)] = kernel[((((int)blockIdx.x) * 288) + ((int)threadIdx.x))];
  }
  __syncthreads();
  for (int c_outer_inner = 0; c_outer_inner < 8; ++c_outer_inner) {
    DepthwiseConv2d[(c_outer_inner * 2)] = (DepthwiseConv2d[(c_outer_inner * 2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 98) * 2048) + (c_outer_inner * 256)) + (((((int)threadIdx.x) % 98) / 7) * 16)) + ((((int)threadIdx.x) % 7) * 2))] * kernel_shared[(((((int)threadIdx.x) / 98) * 72) + (c_outer_inner * 9))]));
    DepthwiseConv2d[((c_outer_inner * 2) + 1)] = (DepthwiseConv2d[((c_outer_inner * 2) + 1)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 98) * 2048) + (c_outer_inner * 256)) + (((((int)threadIdx.x) % 98) / 7) * 16)) + ((((int)threadIdx.x) % 7) * 2)) + 1)] * kernel_shared[(((((int)threadIdx.x) / 98) * 72) + (c_outer_inner * 9))]));
    DepthwiseConv2d[(c_outer_inner * 2)] = (DepthwiseConv2d[(c_outer_inner * 2)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 98) * 2048) + (c_outer_inner * 256)) + (((((int)threadIdx.x) % 98) / 7) * 16)) + ((((int)threadIdx.x) % 7) * 2)) + 1)] * kernel_shared[((((((int)threadIdx.x) / 98) * 72) + (c_outer_inner * 9)) + 1)]));
    DepthwiseConv2d[((c_outer_inner * 2) + 1)] = (DepthwiseConv2d[((c_outer_inner * 2) + 1)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 98) * 2048) + (c_outer_inner * 256)) + (((((int)threadIdx.x) % 98) / 7) * 16)) + ((((int)threadIdx.x) % 7) * 2)) + 2)] * kernel_shared[((((((int)threadIdx.x) / 98) * 72) + (c_outer_inner * 9)) + 1)]));
    DepthwiseConv2d[(c_outer_inner * 2)] = (DepthwiseConv2d[(c_outer_inner * 2)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 98) * 2048) + (c_outer_inner * 256)) + (((((int)threadIdx.x) % 98) / 7) * 16)) + ((((int)threadIdx.x) % 7) * 2)) + 2)] * kernel_shared[((((((int)threadIdx.x) / 98) * 72) + (c_outer_inner * 9)) + 2)]));
    DepthwiseConv2d[((c_outer_inner * 2) + 1)] = (DepthwiseConv2d[((c_outer_inner * 2) + 1)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 98) * 2048) + (c_outer_inner * 256)) + (((((int)threadIdx.x) % 98) / 7) * 16)) + ((((int)threadIdx.x) % 7) * 2)) + 3)] * kernel_shared[((((((int)threadIdx.x) / 98) * 72) + (c_outer_inner * 9)) + 2)]));
    DepthwiseConv2d[(c_outer_inner * 2)] = (DepthwiseConv2d[(c_outer_inner * 2)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 98) * 2048) + (c_outer_inner * 256)) + (((((int)threadIdx.x) % 98) / 7) * 16)) + ((((int)threadIdx.x) % 7) * 2)) + 16)] * kernel_shared[((((((int)threadIdx.x) / 98) * 72) + (c_outer_inner * 9)) + 3)]));
    DepthwiseConv2d[((c_outer_inner * 2) + 1)] = (DepthwiseConv2d[((c_outer_inner * 2) + 1)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 98) * 2048) + (c_outer_inner * 256)) + (((((int)threadIdx.x) % 98) / 7) * 16)) + ((((int)threadIdx.x) % 7) * 2)) + 17)] * kernel_shared[((((((int)threadIdx.x) / 98) * 72) + (c_outer_inner * 9)) + 3)]));
    DepthwiseConv2d[(c_outer_inner * 2)] = (DepthwiseConv2d[(c_outer_inner * 2)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 98) * 2048) + (c_outer_inner * 256)) + (((((int)threadIdx.x) % 98) / 7) * 16)) + ((((int)threadIdx.x) % 7) * 2)) + 17)] * kernel_shared[((((((int)threadIdx.x) / 98) * 72) + (c_outer_inner * 9)) + 4)]));
    DepthwiseConv2d[((c_outer_inner * 2) + 1)] = (DepthwiseConv2d[((c_outer_inner * 2) + 1)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 98) * 2048) + (c_outer_inner * 256)) + (((((int)threadIdx.x) % 98) / 7) * 16)) + ((((int)threadIdx.x) % 7) * 2)) + 18)] * kernel_shared[((((((int)threadIdx.x) / 98) * 72) + (c_outer_inner * 9)) + 4)]));
    DepthwiseConv2d[(c_outer_inner * 2)] = (DepthwiseConv2d[(c_outer_inner * 2)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 98) * 2048) + (c_outer_inner * 256)) + (((((int)threadIdx.x) % 98) / 7) * 16)) + ((((int)threadIdx.x) % 7) * 2)) + 18)] * kernel_shared[((((((int)threadIdx.x) / 98) * 72) + (c_outer_inner * 9)) + 5)]));
    DepthwiseConv2d[((c_outer_inner * 2) + 1)] = (DepthwiseConv2d[((c_outer_inner * 2) + 1)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 98) * 2048) + (c_outer_inner * 256)) + (((((int)threadIdx.x) % 98) / 7) * 16)) + ((((int)threadIdx.x) % 7) * 2)) + 19)] * kernel_shared[((((((int)threadIdx.x) / 98) * 72) + (c_outer_inner * 9)) + 5)]));
    DepthwiseConv2d[(c_outer_inner * 2)] = (DepthwiseConv2d[(c_outer_inner * 2)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 98) * 2048) + (c_outer_inner * 256)) + (((((int)threadIdx.x) % 98) / 7) * 16)) + ((((int)threadIdx.x) % 7) * 2)) + 32)] * kernel_shared[((((((int)threadIdx.x) / 98) * 72) + (c_outer_inner * 9)) + 6)]));
    DepthwiseConv2d[((c_outer_inner * 2) + 1)] = (DepthwiseConv2d[((c_outer_inner * 2) + 1)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 98) * 2048) + (c_outer_inner * 256)) + (((((int)threadIdx.x) % 98) / 7) * 16)) + ((((int)threadIdx.x) % 7) * 2)) + 33)] * kernel_shared[((((((int)threadIdx.x) / 98) * 72) + (c_outer_inner * 9)) + 6)]));
    DepthwiseConv2d[(c_outer_inner * 2)] = (DepthwiseConv2d[(c_outer_inner * 2)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 98) * 2048) + (c_outer_inner * 256)) + (((((int)threadIdx.x) % 98) / 7) * 16)) + ((((int)threadIdx.x) % 7) * 2)) + 33)] * kernel_shared[((((((int)threadIdx.x) / 98) * 72) + (c_outer_inner * 9)) + 7)]));
    DepthwiseConv2d[((c_outer_inner * 2) + 1)] = (DepthwiseConv2d[((c_outer_inner * 2) + 1)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 98) * 2048) + (c_outer_inner * 256)) + (((((int)threadIdx.x) % 98) / 7) * 16)) + ((((int)threadIdx.x) % 7) * 2)) + 34)] * kernel_shared[((((((int)threadIdx.x) / 98) * 72) + (c_outer_inner * 9)) + 7)]));
    DepthwiseConv2d[(c_outer_inner * 2)] = (DepthwiseConv2d[(c_outer_inner * 2)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 98) * 2048) + (c_outer_inner * 256)) + (((((int)threadIdx.x) % 98) / 7) * 16)) + ((((int)threadIdx.x) % 7) * 2)) + 34)] * kernel_shared[((((((int)threadIdx.x) / 98) * 72) + (c_outer_inner * 9)) + 8)]));
    DepthwiseConv2d[((c_outer_inner * 2) + 1)] = (DepthwiseConv2d[((c_outer_inner * 2) + 1)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 98) * 2048) + (c_outer_inner * 256)) + (((((int)threadIdx.x) % 98) / 7) * 16)) + ((((int)threadIdx.x) % 7) * 2)) + 35)] * kernel_shared[((((((int)threadIdx.x) / 98) * 72) + (c_outer_inner * 9)) + 8)]));
  }
  for (int i1_inner = 0; i1_inner < 8; ++i1_inner) {
    for (int i3_inner = 0; i3_inner < 2; ++i3_inner) {
      compute[(((((((int)blockIdx.x) * 6272) + ((((int)threadIdx.x) / 98) * 1568)) + (i1_inner * 196)) + ((((int)threadIdx.x) % 98) * 2)) + i3_inner)] = max(DepthwiseConv2d[((i1_inner * 2) + i3_inner)], 0.000000e+00f);
    }
  }
}


