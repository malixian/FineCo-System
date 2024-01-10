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
extern "C" __global__ void __launch_bounds__(98) candidate0(float* __restrict__ Input, float* __restrict__ kernel, float* __restrict__ compute) {
  float DepthwiseConv2d[4];
  __shared__ float PaddedInput_shared[1800];
  __shared__ float kernel_shared[72];
  DepthwiseConv2d[0] = 0.000000e+00f;
  DepthwiseConv2d[1] = 0.000000e+00f;
  DepthwiseConv2d[2] = 0.000000e+00f;
  DepthwiseConv2d[3] = 0.000000e+00f;
  PaddedInput_shared[(((int)threadIdx.x) * 2)] = (((1 <= ((((((int)blockIdx.x) & 3) >> 1) * 14) + ((((int)threadIdx.x) * 2) / 15))) && (1 <= (((((int)blockIdx.x) & 1) * 14) + ((((int)threadIdx.x) * 2) % 15)))) ? Input[(((((((((int)blockIdx.x) >> 2) * 6272) + (((((int)blockIdx.x) & 3) >> 1) * 392)) + (((((int)threadIdx.x) * 2) / 15) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + ((((int)threadIdx.x) * 2) % 15)) - 29)] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) * 2) + 1)] = (((1 <= ((((((int)blockIdx.x) & 3) >> 1) * 14) + (((((int)threadIdx.x) * 2) + 1) / 15))) && (1 <= (((((int)blockIdx.x) & 1) * 14) + (((((int)threadIdx.x) * 2) + 1) % 15)))) ? Input[(((((((((int)blockIdx.x) >> 2) * 6272) + (((((int)blockIdx.x) & 3) >> 1) * 392)) + ((((((int)threadIdx.x) * 2) + 1) / 15) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((((int)threadIdx.x) * 2) + 1) % 15)) - 29)] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) * 2) + 196)] = (((1 <= ((((((int)blockIdx.x) & 3) >> 1) * 14) + ((((((int)threadIdx.x) * 2) + 196) % 225) / 15))) && (1 <= (((((int)blockIdx.x) & 1) * 14) + (((((int)threadIdx.x) * 2) + 1) % 15)))) ? Input[((((((((((int)blockIdx.x) >> 2) * 6272) + ((((((int)threadIdx.x) * 2) + 196) / 225) * 784)) + (((((int)blockIdx.x) & 3) >> 1) * 392)) + (((((((int)threadIdx.x) * 2) + 196) % 225) / 15) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((((int)threadIdx.x) * 2) + 1) % 15)) - 29)] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) * 2) + 197)] = (((1 <= ((((((int)blockIdx.x) & 3) >> 1) * 14) + ((((((int)threadIdx.x) * 2) + 197) % 225) / 15))) && (1 <= (((((int)blockIdx.x) & 1) * 14) + (((((int)threadIdx.x) * 2) + 2) % 15)))) ? Input[((((((((((int)blockIdx.x) >> 2) * 6272) + ((((((int)threadIdx.x) * 2) + 197) / 225) * 784)) + (((((int)blockIdx.x) & 3) >> 1) * 392)) + (((((((int)threadIdx.x) * 2) + 197) % 225) / 15) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((((int)threadIdx.x) * 2) + 2) % 15)) - 29)] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) * 2) + 392)] = (((1 <= ((((((int)blockIdx.x) & 3) >> 1) * 14) + ((((((int)threadIdx.x) * 2) + 167) % 225) / 15))) && (1 <= (((((int)blockIdx.x) & 1) * 14) + (((((int)threadIdx.x) * 2) + 2) % 15)))) ? Input[((((((((((int)blockIdx.x) >> 2) * 6272) + ((((((int)threadIdx.x) * 2) + 392) / 225) * 784)) + (((((int)blockIdx.x) & 3) >> 1) * 392)) + (((((((int)threadIdx.x) * 2) + 167) % 225) / 15) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((((int)threadIdx.x) * 2) + 2) % 15)) - 29)] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) * 2) + 393)] = (((1 <= ((((((int)blockIdx.x) & 3) >> 1) * 14) + ((((((int)threadIdx.x) * 2) + 168) % 225) / 15))) && (1 <= (((((int)blockIdx.x) & 1) * 14) + (((((int)threadIdx.x) * 2) + 3) % 15)))) ? Input[((((((((((int)blockIdx.x) >> 2) * 6272) + ((((((int)threadIdx.x) * 2) + 393) / 225) * 784)) + (((((int)blockIdx.x) & 3) >> 1) * 392)) + (((((((int)threadIdx.x) * 2) + 168) % 225) / 15) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((((int)threadIdx.x) * 2) + 3) % 15)) - 29)] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) * 2) + 588)] = (((1 <= ((((((int)blockIdx.x) & 3) >> 1) * 14) + ((((((int)threadIdx.x) * 2) + 138) % 225) / 15))) && (1 <= (((((int)blockIdx.x) & 1) * 14) + (((((int)threadIdx.x) * 2) + 3) % 15)))) ? Input[((((((((((int)blockIdx.x) >> 2) * 6272) + ((((((int)threadIdx.x) * 2) + 588) / 225) * 784)) + (((((int)blockIdx.x) & 3) >> 1) * 392)) + (((((((int)threadIdx.x) * 2) + 138) % 225) / 15) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((((int)threadIdx.x) * 2) + 3) % 15)) - 29)] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) * 2) + 589)] = (((1 <= ((((((int)blockIdx.x) & 3) >> 1) * 14) + ((((((int)threadIdx.x) * 2) + 139) % 225) / 15))) && (1 <= (((((int)blockIdx.x) & 1) * 14) + (((((int)threadIdx.x) * 2) + 4) % 15)))) ? Input[((((((((((int)blockIdx.x) >> 2) * 6272) + ((((((int)threadIdx.x) * 2) + 589) / 225) * 784)) + (((((int)blockIdx.x) & 3) >> 1) * 392)) + (((((((int)threadIdx.x) * 2) + 139) % 225) / 15) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((((int)threadIdx.x) * 2) + 4) % 15)) - 29)] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) * 2) + 784)] = (((1 <= ((((((int)blockIdx.x) & 3) >> 1) * 14) + ((((((int)threadIdx.x) * 2) + 109) % 225) / 15))) && (1 <= (((((int)blockIdx.x) & 1) * 14) + (((((int)threadIdx.x) * 2) + 4) % 15)))) ? Input[((((((((((int)blockIdx.x) >> 2) * 6272) + ((((((int)threadIdx.x) * 2) + 784) / 225) * 784)) + (((((int)blockIdx.x) & 3) >> 1) * 392)) + (((((((int)threadIdx.x) * 2) + 109) % 225) / 15) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((((int)threadIdx.x) * 2) + 4) % 15)) - 29)] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) * 2) + 785)] = (((1 <= ((((((int)blockIdx.x) & 3) >> 1) * 14) + ((((((int)threadIdx.x) * 2) + 110) % 225) / 15))) && (1 <= (((((int)blockIdx.x) & 1) * 14) + (((((int)threadIdx.x) * 2) + 5) % 15)))) ? Input[((((((((((int)blockIdx.x) >> 2) * 6272) + ((((((int)threadIdx.x) * 2) + 785) / 225) * 784)) + (((((int)blockIdx.x) & 3) >> 1) * 392)) + (((((((int)threadIdx.x) * 2) + 110) % 225) / 15) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((((int)threadIdx.x) * 2) + 5) % 15)) - 29)] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) * 2) + 980)] = (((1 <= ((((((int)blockIdx.x) & 3) >> 1) * 14) + ((((((int)threadIdx.x) * 2) + 80) % 225) / 15))) && (1 <= (((((int)blockIdx.x) & 1) * 14) + (((((int)threadIdx.x) * 2) + 5) % 15)))) ? Input[((((((((((int)blockIdx.x) >> 2) * 6272) + ((((((int)threadIdx.x) * 2) + 980) / 225) * 784)) + (((((int)blockIdx.x) & 3) >> 1) * 392)) + (((((((int)threadIdx.x) * 2) + 80) % 225) / 15) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((((int)threadIdx.x) * 2) + 5) % 15)) - 29)] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) * 2) + 981)] = (((1 <= ((((((int)blockIdx.x) & 3) >> 1) * 14) + ((((((int)threadIdx.x) * 2) + 81) % 225) / 15))) && (1 <= (((((int)blockIdx.x) & 1) * 14) + (((((int)threadIdx.x) * 2) + 6) % 15)))) ? Input[((((((((((int)blockIdx.x) >> 2) * 6272) + ((((((int)threadIdx.x) * 2) + 981) / 225) * 784)) + (((((int)blockIdx.x) & 3) >> 1) * 392)) + (((((((int)threadIdx.x) * 2) + 81) % 225) / 15) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((((int)threadIdx.x) * 2) + 6) % 15)) - 29)] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) * 2) + 1176)] = (((1 <= ((((((int)blockIdx.x) & 3) >> 1) * 14) + ((((((int)threadIdx.x) * 2) + 51) % 225) / 15))) && (1 <= (((((int)blockIdx.x) & 1) * 14) + (((((int)threadIdx.x) * 2) + 6) % 15)))) ? Input[((((((((((int)blockIdx.x) >> 2) * 6272) + ((((((int)threadIdx.x) * 2) + 1176) / 225) * 784)) + (((((int)blockIdx.x) & 3) >> 1) * 392)) + (((((((int)threadIdx.x) * 2) + 51) % 225) / 15) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((((int)threadIdx.x) * 2) + 6) % 15)) - 29)] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) * 2) + 1177)] = (((1 <= ((((((int)blockIdx.x) & 3) >> 1) * 14) + ((((((int)threadIdx.x) * 2) + 52) % 225) / 15))) && (1 <= (((((int)blockIdx.x) & 1) * 14) + (((((int)threadIdx.x) * 2) + 7) % 15)))) ? Input[((((((((((int)blockIdx.x) >> 2) * 6272) + ((((((int)threadIdx.x) * 2) + 1177) / 225) * 784)) + (((((int)blockIdx.x) & 3) >> 1) * 392)) + (((((((int)threadIdx.x) * 2) + 52) % 225) / 15) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((((int)threadIdx.x) * 2) + 7) % 15)) - 29)] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) * 2) + 1372)] = ((1 <= (((((int)blockIdx.x) & 1) * 14) + (((((int)threadIdx.x) * 2) + 7) % 15))) ? Input[((((((((((int)blockIdx.x) >> 2) * 6272) + ((((((int)threadIdx.x) * 2) + 1372) / 225) * 784)) + (((((int)blockIdx.x) & 3) >> 1) * 392)) + (((((((int)threadIdx.x) * 2) + 22) % 225) / 15) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((((int)threadIdx.x) * 2) + 7) % 15)) - 29)] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) * 2) + 1373)] = ((1 <= (((((int)blockIdx.x) & 1) * 14) + (((((int)threadIdx.x) * 2) + 8) % 15))) ? Input[((((((((((int)blockIdx.x) >> 2) * 6272) + ((((((int)threadIdx.x) * 2) + 1373) / 225) * 784)) + (((((int)blockIdx.x) & 3) >> 1) * 392)) + (((((((int)threadIdx.x) * 2) + 23) % 225) / 15) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((((int)threadIdx.x) * 2) + 8) % 15)) - 29)] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) * 2) + 1568)] = (((1 <= ((((((int)blockIdx.x) & 3) >> 1) * 14) + ((((((int)threadIdx.x) * 2) + 218) % 225) / 15))) && (1 <= (((((int)blockIdx.x) & 1) * 14) + (((((int)threadIdx.x) * 2) + 8) % 15)))) ? Input[((((((((((int)blockIdx.x) >> 2) * 6272) + ((((((int)threadIdx.x) * 2) + 1568) / 225) * 784)) + (((((int)blockIdx.x) & 3) >> 1) * 392)) + (((((((int)threadIdx.x) * 2) + 218) % 225) / 15) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((((int)threadIdx.x) * 2) + 8) % 15)) - 29)] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) * 2) + 1569)] = (((1 <= ((((((int)blockIdx.x) & 3) >> 1) * 14) + ((((((int)threadIdx.x) * 2) + 219) % 225) / 15))) && (1 <= (((((int)blockIdx.x) & 1) * 14) + (((((int)threadIdx.x) * 2) + 9) % 15)))) ? Input[((((((((((int)blockIdx.x) >> 2) * 6272) + ((((((int)threadIdx.x) * 2) + 1569) / 225) * 784)) + (((((int)blockIdx.x) & 3) >> 1) * 392)) + (((((((int)threadIdx.x) * 2) + 219) % 225) / 15) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((((int)threadIdx.x) * 2) + 9) % 15)) - 29)] : 0.000000e+00f);
  if (((int)threadIdx.x) < 18) {
    PaddedInput_shared[((((int)threadIdx.x) * 2) + 1764)] = ((1 <= (((((int)blockIdx.x) & 1) * 14) + (((((int)threadIdx.x) * 2) + 9) % 15))) ? Input[((((((((((int)blockIdx.x) >> 2) * 6272) + ((((((int)threadIdx.x) * 2) + 1764) / 225) * 784)) + (((((int)blockIdx.x) & 3) >> 1) * 392)) + (((((((int)threadIdx.x) * 2) + 189) % 225) / 15) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((((int)threadIdx.x) * 2) + 9) % 15)) - 29)] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 18) {
    PaddedInput_shared[((((int)threadIdx.x) * 2) + 1765)] = ((1 <= (((((int)blockIdx.x) & 1) * 14) + (((((int)threadIdx.x) * 2) + 10) % 15))) ? Input[((((((((((int)blockIdx.x) >> 2) * 6272) + ((((((int)threadIdx.x) * 2) + 1765) / 225) * 784)) + (((((int)blockIdx.x) & 3) >> 1) * 392)) + (((((((int)threadIdx.x) * 2) + 190) % 225) / 15) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((((int)threadIdx.x) * 2) + 10) % 15)) - 29)] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 18) {
    kernel_shared[(((int)threadIdx.x) * 4)] = kernel[(((((int)blockIdx.x) >> 2) * 72) + (((int)threadIdx.x) * 4))];
  }
  if (((int)threadIdx.x) < 18) {
    kernel_shared[((((int)threadIdx.x) * 4) + 1)] = kernel[((((((int)blockIdx.x) >> 2) * 72) + (((int)threadIdx.x) * 4)) + 1)];
  }
  if (((int)threadIdx.x) < 18) {
    kernel_shared[((((int)threadIdx.x) * 4) + 2)] = kernel[((((((int)blockIdx.x) >> 2) * 72) + (((int)threadIdx.x) * 4)) + 2)];
  }
  if (((int)threadIdx.x) < 18) {
    kernel_shared[((((int)threadIdx.x) * 4) + 3)] = kernel[((((((int)blockIdx.x) >> 2) * 72) + (((int)threadIdx.x) * 4)) + 3)];
  }
  __syncthreads();
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 49) * 900) + (((((int)threadIdx.x) % 49) / 7) * 30)) + ((((int)threadIdx.x) % 7) * 2))] * kernel_shared[((((int)threadIdx.x) / 49) * 36)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[(((((((int)threadIdx.x) / 49) * 900) + (((((int)threadIdx.x) % 49) / 7) * 30)) + ((((int)threadIdx.x) % 7) * 2)) + 225)] * kernel_shared[(((((int)threadIdx.x) / 49) * 36) + 9)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[(((((((int)threadIdx.x) / 49) * 900) + (((((int)threadIdx.x) % 49) / 7) * 30)) + ((((int)threadIdx.x) % 7) * 2)) + 450)] * kernel_shared[(((((int)threadIdx.x) / 49) * 36) + 18)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[(((((((int)threadIdx.x) / 49) * 900) + (((((int)threadIdx.x) % 49) / 7) * 30)) + ((((int)threadIdx.x) % 7) * 2)) + 675)] * kernel_shared[(((((int)threadIdx.x) / 49) * 36) + 27)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((((int)threadIdx.x) / 49) * 900) + (((((int)threadIdx.x) % 49) / 7) * 30)) + ((((int)threadIdx.x) % 7) * 2)) + 1)] * kernel_shared[(((((int)threadIdx.x) / 49) * 36) + 1)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[(((((((int)threadIdx.x) / 49) * 900) + (((((int)threadIdx.x) % 49) / 7) * 30)) + ((((int)threadIdx.x) % 7) * 2)) + 226)] * kernel_shared[(((((int)threadIdx.x) / 49) * 36) + 10)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[(((((((int)threadIdx.x) / 49) * 900) + (((((int)threadIdx.x) % 49) / 7) * 30)) + ((((int)threadIdx.x) % 7) * 2)) + 451)] * kernel_shared[(((((int)threadIdx.x) / 49) * 36) + 19)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[(((((((int)threadIdx.x) / 49) * 900) + (((((int)threadIdx.x) % 49) / 7) * 30)) + ((((int)threadIdx.x) % 7) * 2)) + 676)] * kernel_shared[(((((int)threadIdx.x) / 49) * 36) + 28)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((((int)threadIdx.x) / 49) * 900) + (((((int)threadIdx.x) % 49) / 7) * 30)) + ((((int)threadIdx.x) % 7) * 2)) + 2)] * kernel_shared[(((((int)threadIdx.x) / 49) * 36) + 2)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[(((((((int)threadIdx.x) / 49) * 900) + (((((int)threadIdx.x) % 49) / 7) * 30)) + ((((int)threadIdx.x) % 7) * 2)) + 227)] * kernel_shared[(((((int)threadIdx.x) / 49) * 36) + 11)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[(((((((int)threadIdx.x) / 49) * 900) + (((((int)threadIdx.x) % 49) / 7) * 30)) + ((((int)threadIdx.x) % 7) * 2)) + 452)] * kernel_shared[(((((int)threadIdx.x) / 49) * 36) + 20)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[(((((((int)threadIdx.x) / 49) * 900) + (((((int)threadIdx.x) % 49) / 7) * 30)) + ((((int)threadIdx.x) % 7) * 2)) + 677)] * kernel_shared[(((((int)threadIdx.x) / 49) * 36) + 29)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((((int)threadIdx.x) / 49) * 900) + (((((int)threadIdx.x) % 49) / 7) * 30)) + ((((int)threadIdx.x) % 7) * 2)) + 15)] * kernel_shared[(((((int)threadIdx.x) / 49) * 36) + 3)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[(((((((int)threadIdx.x) / 49) * 900) + (((((int)threadIdx.x) % 49) / 7) * 30)) + ((((int)threadIdx.x) % 7) * 2)) + 240)] * kernel_shared[(((((int)threadIdx.x) / 49) * 36) + 12)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[(((((((int)threadIdx.x) / 49) * 900) + (((((int)threadIdx.x) % 49) / 7) * 30)) + ((((int)threadIdx.x) % 7) * 2)) + 465)] * kernel_shared[(((((int)threadIdx.x) / 49) * 36) + 21)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[(((((((int)threadIdx.x) / 49) * 900) + (((((int)threadIdx.x) % 49) / 7) * 30)) + ((((int)threadIdx.x) % 7) * 2)) + 690)] * kernel_shared[(((((int)threadIdx.x) / 49) * 36) + 30)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((((int)threadIdx.x) / 49) * 900) + (((((int)threadIdx.x) % 49) / 7) * 30)) + ((((int)threadIdx.x) % 7) * 2)) + 16)] * kernel_shared[(((((int)threadIdx.x) / 49) * 36) + 4)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[(((((((int)threadIdx.x) / 49) * 900) + (((((int)threadIdx.x) % 49) / 7) * 30)) + ((((int)threadIdx.x) % 7) * 2)) + 241)] * kernel_shared[(((((int)threadIdx.x) / 49) * 36) + 13)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[(((((((int)threadIdx.x) / 49) * 900) + (((((int)threadIdx.x) % 49) / 7) * 30)) + ((((int)threadIdx.x) % 7) * 2)) + 466)] * kernel_shared[(((((int)threadIdx.x) / 49) * 36) + 22)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[(((((((int)threadIdx.x) / 49) * 900) + (((((int)threadIdx.x) % 49) / 7) * 30)) + ((((int)threadIdx.x) % 7) * 2)) + 691)] * kernel_shared[(((((int)threadIdx.x) / 49) * 36) + 31)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((((int)threadIdx.x) / 49) * 900) + (((((int)threadIdx.x) % 49) / 7) * 30)) + ((((int)threadIdx.x) % 7) * 2)) + 17)] * kernel_shared[(((((int)threadIdx.x) / 49) * 36) + 5)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[(((((((int)threadIdx.x) / 49) * 900) + (((((int)threadIdx.x) % 49) / 7) * 30)) + ((((int)threadIdx.x) % 7) * 2)) + 242)] * kernel_shared[(((((int)threadIdx.x) / 49) * 36) + 14)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[(((((((int)threadIdx.x) / 49) * 900) + (((((int)threadIdx.x) % 49) / 7) * 30)) + ((((int)threadIdx.x) % 7) * 2)) + 467)] * kernel_shared[(((((int)threadIdx.x) / 49) * 36) + 23)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[(((((((int)threadIdx.x) / 49) * 900) + (((((int)threadIdx.x) % 49) / 7) * 30)) + ((((int)threadIdx.x) % 7) * 2)) + 692)] * kernel_shared[(((((int)threadIdx.x) / 49) * 36) + 32)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((((int)threadIdx.x) / 49) * 900) + (((((int)threadIdx.x) % 49) / 7) * 30)) + ((((int)threadIdx.x) % 7) * 2)) + 30)] * kernel_shared[(((((int)threadIdx.x) / 49) * 36) + 6)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[(((((((int)threadIdx.x) / 49) * 900) + (((((int)threadIdx.x) % 49) / 7) * 30)) + ((((int)threadIdx.x) % 7) * 2)) + 255)] * kernel_shared[(((((int)threadIdx.x) / 49) * 36) + 15)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[(((((((int)threadIdx.x) / 49) * 900) + (((((int)threadIdx.x) % 49) / 7) * 30)) + ((((int)threadIdx.x) % 7) * 2)) + 480)] * kernel_shared[(((((int)threadIdx.x) / 49) * 36) + 24)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[(((((((int)threadIdx.x) / 49) * 900) + (((((int)threadIdx.x) % 49) / 7) * 30)) + ((((int)threadIdx.x) % 7) * 2)) + 705)] * kernel_shared[(((((int)threadIdx.x) / 49) * 36) + 33)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((((int)threadIdx.x) / 49) * 900) + (((((int)threadIdx.x) % 49) / 7) * 30)) + ((((int)threadIdx.x) % 7) * 2)) + 31)] * kernel_shared[(((((int)threadIdx.x) / 49) * 36) + 7)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[(((((((int)threadIdx.x) / 49) * 900) + (((((int)threadIdx.x) % 49) / 7) * 30)) + ((((int)threadIdx.x) % 7) * 2)) + 256)] * kernel_shared[(((((int)threadIdx.x) / 49) * 36) + 16)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[(((((((int)threadIdx.x) / 49) * 900) + (((((int)threadIdx.x) % 49) / 7) * 30)) + ((((int)threadIdx.x) % 7) * 2)) + 481)] * kernel_shared[(((((int)threadIdx.x) / 49) * 36) + 25)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[(((((((int)threadIdx.x) / 49) * 900) + (((((int)threadIdx.x) % 49) / 7) * 30)) + ((((int)threadIdx.x) % 7) * 2)) + 706)] * kernel_shared[(((((int)threadIdx.x) / 49) * 36) + 34)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((((int)threadIdx.x) / 49) * 900) + (((((int)threadIdx.x) % 49) / 7) * 30)) + ((((int)threadIdx.x) % 7) * 2)) + 32)] * kernel_shared[(((((int)threadIdx.x) / 49) * 36) + 8)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[(((((((int)threadIdx.x) / 49) * 900) + (((((int)threadIdx.x) % 49) / 7) * 30)) + ((((int)threadIdx.x) % 7) * 2)) + 257)] * kernel_shared[(((((int)threadIdx.x) / 49) * 36) + 17)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[(((((((int)threadIdx.x) / 49) * 900) + (((((int)threadIdx.x) % 49) / 7) * 30)) + ((((int)threadIdx.x) % 7) * 2)) + 482)] * kernel_shared[(((((int)threadIdx.x) / 49) * 36) + 26)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[(((((((int)threadIdx.x) / 49) * 900) + (((((int)threadIdx.x) % 49) / 7) * 30)) + ((((int)threadIdx.x) % 7) * 2)) + 707)] * kernel_shared[(((((int)threadIdx.x) / 49) * 36) + 35)]));
  for (int i1_inner = 0; i1_inner < 4; ++i1_inner) {
    compute[((((((((((int)blockIdx.x) >> 2) * 1568) + ((((int)threadIdx.x) / 49) * 784)) + (i1_inner * 196)) + (((((int)blockIdx.x) & 3) >> 1) * 98)) + (((((int)threadIdx.x) % 49) / 7) * 14)) + ((((int)blockIdx.x) & 1) * 7)) + (((int)threadIdx.x) % 7))] = max(DepthwiseConv2d[i1_inner], 0.000000e+00f);
  }
}


