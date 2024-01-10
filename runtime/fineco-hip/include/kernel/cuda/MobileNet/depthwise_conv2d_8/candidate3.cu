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
extern "C" __global__ void __launch_bounds__(112) candidate3(float* __restrict__ Input, float* __restrict__ kernel, float* __restrict__ compute) {
  float DepthwiseConv2d[7];
  __shared__ float PaddedInput_shared[3600];
  __shared__ float kernel_shared[144];
  DepthwiseConv2d[0] = 0.000000e+00f;
  DepthwiseConv2d[1] = 0.000000e+00f;
  DepthwiseConv2d[2] = 0.000000e+00f;
  DepthwiseConv2d[3] = 0.000000e+00f;
  DepthwiseConv2d[4] = 0.000000e+00f;
  DepthwiseConv2d[5] = 0.000000e+00f;
  DepthwiseConv2d[6] = 0.000000e+00f;
  PaddedInput_shared[((int)threadIdx.x)] = (((15 <= ((int)threadIdx.x)) && (1 <= (((int)threadIdx.x) % 15))) ? Input[((((((int)blockIdx.x) * 3136) + ((((int)threadIdx.x) / 15) * 14)) + (((int)threadIdx.x) % 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 112)] = ((1 <= ((((int)threadIdx.x) + 7) % 15)) ? Input[((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 112) / 15) * 14)) + ((((int)threadIdx.x) + 7) % 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 224)] = (((15 <= ((((int)threadIdx.x) + 224) % 225)) && (1 <= ((((int)threadIdx.x) + 14) % 15))) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 224) / 225) * 196)) + ((((((int)threadIdx.x) + 224) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 14) % 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 336)] = ((1 <= ((((int)threadIdx.x) + 6) % 15)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 336) / 225) * 196)) + ((((((int)threadIdx.x) + 111) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 6) % 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 448)] = (((15 <= ((((int)threadIdx.x) + 223) % 225)) && (1 <= ((((int)threadIdx.x) + 13) % 15))) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 448) / 225) * 196)) + ((((((int)threadIdx.x) + 223) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 13) % 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 560)] = ((1 <= ((((int)threadIdx.x) + 5) % 15)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 560) / 225) * 196)) + ((((((int)threadIdx.x) + 110) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 5) % 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 672)] = (((15 <= ((((int)threadIdx.x) + 222) % 225)) && (1 <= ((((int)threadIdx.x) + 12) % 15))) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 672) / 225) * 196)) + ((((((int)threadIdx.x) + 222) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 12) % 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 784)] = ((1 <= ((((int)threadIdx.x) + 4) % 15)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 784) / 225) * 196)) + ((((((int)threadIdx.x) + 109) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 4) % 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 896)] = (((15 <= ((((int)threadIdx.x) + 221) % 225)) && (1 <= ((((int)threadIdx.x) + 11) % 15))) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 896) / 225) * 196)) + ((((((int)threadIdx.x) + 221) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 11) % 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 1008)] = ((1 <= ((((int)threadIdx.x) + 3) % 15)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 1008) / 225) * 196)) + ((((((int)threadIdx.x) + 108) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 3) % 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 1120)] = (((15 <= ((((int)threadIdx.x) + 220) % 225)) && (1 <= ((((int)threadIdx.x) + 10) % 15))) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 1120) / 225) * 196)) + ((((((int)threadIdx.x) + 220) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 10) % 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 1232)] = ((1 <= ((((int)threadIdx.x) + 2) % 15)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 1232) / 225) * 196)) + ((((((int)threadIdx.x) + 107) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 2) % 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 1344)] = (((15 <= ((((int)threadIdx.x) + 219) % 225)) && (1 <= ((((int)threadIdx.x) + 9) % 15))) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 1344) / 225) * 196)) + ((((((int)threadIdx.x) + 219) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 9) % 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 1456)] = ((1 <= ((((int)threadIdx.x) + 1) % 15)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 1456) / 225) * 196)) + ((((((int)threadIdx.x) + 106) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 1) % 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 1568)] = (((15 <= ((((int)threadIdx.x) + 218) % 225)) && (1 <= ((((int)threadIdx.x) + 8) % 15))) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 1568) / 225) * 196)) + ((((((int)threadIdx.x) + 218) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 8) % 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 1680)] = ((1 <= (((int)threadIdx.x) % 15)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 1680) / 225) * 196)) + (((((int)threadIdx.x) / 15) + 7) * 14)) + (((int)threadIdx.x) % 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 1792)] = (((15 <= ((((int)threadIdx.x) + 217) % 225)) && (1 <= ((((int)threadIdx.x) + 7) % 15))) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 1792) / 225) * 196)) + ((((((int)threadIdx.x) + 217) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 7) % 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 1904)] = ((1 <= ((((int)threadIdx.x) + 14) % 15)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 1904) / 225) * 196)) + ((((((int)threadIdx.x) + 104) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 14) % 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 2016)] = (((15 <= ((((int)threadIdx.x) + 216) % 225)) && (1 <= ((((int)threadIdx.x) + 6) % 15))) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 2016) / 225) * 196)) + ((((((int)threadIdx.x) + 216) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 6) % 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 2128)] = ((1 <= ((((int)threadIdx.x) + 13) % 15)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 2128) / 225) * 196)) + ((((((int)threadIdx.x) + 103) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 13) % 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 2240)] = (((15 <= ((((int)threadIdx.x) + 215) % 225)) && (1 <= ((((int)threadIdx.x) + 5) % 15))) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 2240) / 225) * 196)) + ((((((int)threadIdx.x) + 215) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 5) % 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 2352)] = ((1 <= ((((int)threadIdx.x) + 12) % 15)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 2352) / 225) * 196)) + ((((((int)threadIdx.x) + 102) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 12) % 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 2464)] = (((15 <= ((((int)threadIdx.x) + 214) % 225)) && (1 <= ((((int)threadIdx.x) + 4) % 15))) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 2464) / 225) * 196)) + ((((((int)threadIdx.x) + 214) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 4) % 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 2576)] = ((1 <= ((((int)threadIdx.x) + 11) % 15)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 2576) / 225) * 196)) + ((((((int)threadIdx.x) + 101) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 11) % 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 2688)] = (((15 <= ((((int)threadIdx.x) + 213) % 225)) && (1 <= ((((int)threadIdx.x) + 3) % 15))) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 2688) / 225) * 196)) + ((((((int)threadIdx.x) + 213) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 3) % 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 2800)] = ((1 <= ((((int)threadIdx.x) + 10) % 15)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 2800) / 225) * 196)) + ((((((int)threadIdx.x) + 100) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 10) % 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 2912)] = (((15 <= ((((int)threadIdx.x) + 212) % 225)) && (1 <= ((((int)threadIdx.x) + 2) % 15))) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 2912) / 225) * 196)) + ((((((int)threadIdx.x) + 212) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 2) % 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 3024)] = ((1 <= ((((int)threadIdx.x) + 9) % 15)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 3024) / 225) * 196)) + ((((((int)threadIdx.x) + 99) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 9) % 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 3136)] = (((15 <= ((((int)threadIdx.x) + 211) % 225)) && (1 <= ((((int)threadIdx.x) + 1) % 15))) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 3136) / 225) * 196)) + ((((((int)threadIdx.x) + 211) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 1) % 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 3248)] = ((1 <= ((((int)threadIdx.x) + 8) % 15)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 3248) / 225) * 196)) + ((((((int)threadIdx.x) + 98) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 8) % 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 3360)] = (((1 <= (((((int)threadIdx.x) / 15) + 14) % 15)) && (1 <= (((int)threadIdx.x) % 15))) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 3360) / 225) * 196)) + ((((((int)threadIdx.x) / 15) + 14) % 15) * 14)) + (((int)threadIdx.x) % 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 3472)] = ((1 <= ((((int)threadIdx.x) + 7) % 15)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 3472) / 225) * 196)) + ((((((int)threadIdx.x) + 97) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 7) % 15)) - 15)] : 0.000000e+00f);
  if (((int)threadIdx.x) < 16) {
    PaddedInput_shared[(((int)threadIdx.x) + 3584)] = ((1 <= ((((int)threadIdx.x) + 14) % 15)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 3584) / 225) * 196)) + ((((((int)threadIdx.x) + 209) % 225) / 15) * 14)) + ((((int)threadIdx.x) + 14) % 15)) - 15)] : 0.000000e+00f);
  }
  kernel_shared[((int)threadIdx.x)] = kernel[((((int)blockIdx.x) * 144) + ((int)threadIdx.x))];
  if (((int)threadIdx.x) < 32) {
    kernel_shared[(((int)threadIdx.x) + 112)] = kernel[(((((int)blockIdx.x) * 144) + ((int)threadIdx.x)) + 112)];
  }
  __syncthreads();
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2))] * kernel_shared[((((int)threadIdx.x) / 7) * 9)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 30)] * kernel_shared[((((int)threadIdx.x) / 7) * 9)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 60)] * kernel_shared[((((int)threadIdx.x) / 7) * 9)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 90)] * kernel_shared[((((int)threadIdx.x) / 7) * 9)]));
  DepthwiseConv2d[4] = (DepthwiseConv2d[4] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 120)] * kernel_shared[((((int)threadIdx.x) / 7) * 9)]));
  DepthwiseConv2d[5] = (DepthwiseConv2d[5] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 150)] * kernel_shared[((((int)threadIdx.x) / 7) * 9)]));
  DepthwiseConv2d[6] = (DepthwiseConv2d[6] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 180)] * kernel_shared[((((int)threadIdx.x) / 7) * 9)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 1)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 1)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 31)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 1)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 61)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 1)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 91)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 1)]));
  DepthwiseConv2d[4] = (DepthwiseConv2d[4] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 121)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 1)]));
  DepthwiseConv2d[5] = (DepthwiseConv2d[5] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 151)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 1)]));
  DepthwiseConv2d[6] = (DepthwiseConv2d[6] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 181)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 1)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 2)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 2)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 32)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 2)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 62)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 2)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 92)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 2)]));
  DepthwiseConv2d[4] = (DepthwiseConv2d[4] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 122)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 2)]));
  DepthwiseConv2d[5] = (DepthwiseConv2d[5] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 152)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 2)]));
  DepthwiseConv2d[6] = (DepthwiseConv2d[6] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 182)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 2)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 15)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 3)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 45)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 3)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 75)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 3)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 105)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 3)]));
  DepthwiseConv2d[4] = (DepthwiseConv2d[4] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 135)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 3)]));
  DepthwiseConv2d[5] = (DepthwiseConv2d[5] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 165)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 3)]));
  DepthwiseConv2d[6] = (DepthwiseConv2d[6] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 195)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 3)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 16)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 4)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 46)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 4)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 76)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 4)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 106)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 4)]));
  DepthwiseConv2d[4] = (DepthwiseConv2d[4] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 136)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 4)]));
  DepthwiseConv2d[5] = (DepthwiseConv2d[5] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 166)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 4)]));
  DepthwiseConv2d[6] = (DepthwiseConv2d[6] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 196)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 4)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 17)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 5)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 47)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 5)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 77)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 5)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 107)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 5)]));
  DepthwiseConv2d[4] = (DepthwiseConv2d[4] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 137)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 5)]));
  DepthwiseConv2d[5] = (DepthwiseConv2d[5] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 167)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 5)]));
  DepthwiseConv2d[6] = (DepthwiseConv2d[6] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 197)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 5)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 30)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 6)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 60)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 6)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 90)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 6)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 120)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 6)]));
  DepthwiseConv2d[4] = (DepthwiseConv2d[4] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 150)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 6)]));
  DepthwiseConv2d[5] = (DepthwiseConv2d[5] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 180)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 6)]));
  DepthwiseConv2d[6] = (DepthwiseConv2d[6] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 210)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 6)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 31)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 7)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 61)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 7)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 91)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 7)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 121)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 7)]));
  DepthwiseConv2d[4] = (DepthwiseConv2d[4] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 151)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 7)]));
  DepthwiseConv2d[5] = (DepthwiseConv2d[5] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 181)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 7)]));
  DepthwiseConv2d[6] = (DepthwiseConv2d[6] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 211)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 7)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 32)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 8)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 62)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 8)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 92)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 8)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 122)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 8)]));
  DepthwiseConv2d[4] = (DepthwiseConv2d[4] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 152)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 8)]));
  DepthwiseConv2d[5] = (DepthwiseConv2d[5] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 182)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 8)]));
  DepthwiseConv2d[6] = (DepthwiseConv2d[6] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 225) + ((((int)threadIdx.x) % 7) * 2)) + 212)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 8)]));
  compute[(((((int)blockIdx.x) * 784) + ((((int)threadIdx.x) / 7) * 49)) + (((int)threadIdx.x) % 7))] = max(DepthwiseConv2d[0], 0.000000e+00f);
  compute[((((((int)blockIdx.x) * 784) + ((((int)threadIdx.x) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 7)] = max(DepthwiseConv2d[1], 0.000000e+00f);
  compute[((((((int)blockIdx.x) * 784) + ((((int)threadIdx.x) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 14)] = max(DepthwiseConv2d[2], 0.000000e+00f);
  compute[((((((int)blockIdx.x) * 784) + ((((int)threadIdx.x) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 21)] = max(DepthwiseConv2d[3], 0.000000e+00f);
  compute[((((((int)blockIdx.x) * 784) + ((((int)threadIdx.x) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 28)] = max(DepthwiseConv2d[4], 0.000000e+00f);
  compute[((((((int)blockIdx.x) * 784) + ((((int)threadIdx.x) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 35)] = max(DepthwiseConv2d[5], 0.000000e+00f);
  compute[((((((int)blockIdx.x) * 784) + ((((int)threadIdx.x) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 42)] = max(DepthwiseConv2d[6], 0.000000e+00f);
}


