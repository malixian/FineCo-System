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
extern "C" __global__ void __launch_bounds__(224) candidate0(float* __restrict__ Input, float* __restrict__ kernel, float* __restrict__ compute) {
  float DepthwiseConv2d[4];
  __shared__ float PaddedInput_shared[1392];
  __shared__ float kernel_shared[36];
  DepthwiseConv2d[0] = 0.000000e+00f;
  DepthwiseConv2d[2] = 0.000000e+00f;
  DepthwiseConv2d[1] = 0.000000e+00f;
  DepthwiseConv2d[3] = 0.000000e+00f;
  PaddedInput_shared[((int)threadIdx.x)] = ((((1 <= ((((((int)blockIdx.x) % 56) >> 1) * 4) + (((int)threadIdx.x) / 58))) && (1 <= (((((int)blockIdx.x) & 1) * 56) + (((int)threadIdx.x) % 58)))) && ((((((int)blockIdx.x) & 1) * 56) + (((int)threadIdx.x) % 58)) < 113)) ? Input[(((((((((int)blockIdx.x) / 56) * 50176) + (((((int)blockIdx.x) % 56) >> 1) * 448)) + ((((int)threadIdx.x) / 58) * 112)) + ((((int)blockIdx.x) & 1) * 56)) + (((int)threadIdx.x) % 58)) - 113)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 224)] = (((((1 <= ((((((int)blockIdx.x) % 56) >> 1) * 4) + ((((((int)threadIdx.x) >> 1) + 112) % 174) / 29))) && (((((((int)blockIdx.x) % 56) >> 1) * 4) + ((((((int)threadIdx.x) >> 1) + 112) % 174) / 29)) < 113)) && (1 <= (((((int)blockIdx.x) & 1) * 56) + ((((int)threadIdx.x) + 50) % 58)))) && ((((((int)blockIdx.x) & 1) * 56) + ((((int)threadIdx.x) + 50) % 58)) < 113)) ? Input[((((((((((int)blockIdx.x) / 56) * 50176) + (((((int)threadIdx.x) + 224) / 348) * 12544)) + (((((int)blockIdx.x) % 56) >> 1) * 448)) + (((((((int)threadIdx.x) >> 1) + 112) % 174) / 29) * 112)) + ((((int)blockIdx.x) & 1) * 56)) + ((((int)threadIdx.x) + 50) % 58)) - 113)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 448)] = ((((((((((int)blockIdx.x) % 56) >> 1) * 4) + ((((((int)threadIdx.x) >> 1) + 50) % 174) / 29)) < 113) && (1 <= (((((int)blockIdx.x) & 1) * 56) + ((((int)threadIdx.x) + 42) % 58)))) && ((((((int)blockIdx.x) & 1) * 56) + ((((int)threadIdx.x) + 42) % 58)) < 113)) ? Input[((((((((((int)blockIdx.x) / 56) * 50176) + (((((int)threadIdx.x) + 448) / 348) * 12544)) + (((((int)blockIdx.x) % 56) >> 1) * 448)) + (((((((int)threadIdx.x) >> 1) + 50) % 174) / 29) * 112)) + ((((int)blockIdx.x) & 1) * 56)) + ((((int)threadIdx.x) + 42) % 58)) - 113)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 672)] = (((((1 <= ((((((int)blockIdx.x) % 56) >> 1) * 4) + ((((((int)threadIdx.x) >> 1) + 162) % 174) / 29))) && (((((((int)blockIdx.x) % 56) >> 1) * 4) + ((((((int)threadIdx.x) >> 1) + 162) % 174) / 29)) < 113)) && (1 <= (((((int)blockIdx.x) & 1) * 56) + ((((int)threadIdx.x) + 34) % 58)))) && ((((((int)blockIdx.x) & 1) * 56) + ((((int)threadIdx.x) + 34) % 58)) < 113)) ? Input[((((((((((int)blockIdx.x) / 56) * 50176) + (((((int)threadIdx.x) + 672) / 348) * 12544)) + (((((int)blockIdx.x) % 56) >> 1) * 448)) + (((((((int)threadIdx.x) >> 1) + 162) % 174) / 29) * 112)) + ((((int)blockIdx.x) & 1) * 56)) + ((((int)threadIdx.x) + 34) % 58)) - 113)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 896)] = (((((1 <= ((((((int)blockIdx.x) % 56) >> 1) * 4) + ((((((int)threadIdx.x) >> 1) + 100) % 174) / 29))) && (((((((int)blockIdx.x) % 56) >> 1) * 4) + ((((((int)threadIdx.x) >> 1) + 100) % 174) / 29)) < 113)) && (1 <= (((((int)blockIdx.x) & 1) * 56) + ((((int)threadIdx.x) + 26) % 58)))) && ((((((int)blockIdx.x) & 1) * 56) + ((((int)threadIdx.x) + 26) % 58)) < 113)) ? Input[((((((((((int)blockIdx.x) / 56) * 50176) + (((((int)threadIdx.x) + 896) / 348) * 12544)) + (((((int)blockIdx.x) % 56) >> 1) * 448)) + (((((((int)threadIdx.x) >> 1) + 100) % 174) / 29) * 112)) + ((((int)blockIdx.x) & 1) * 56)) + ((((int)threadIdx.x) + 26) % 58)) - 113)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 1120)] = ((((((((((int)blockIdx.x) % 56) >> 1) * 4) + ((((((int)threadIdx.x) >> 1) + 38) % 174) / 29)) < 113) && (1 <= (((((int)blockIdx.x) & 1) * 56) + ((((int)threadIdx.x) + 18) % 58)))) && ((((((int)blockIdx.x) & 1) * 56) + ((((int)threadIdx.x) + 18) % 58)) < 113)) ? Input[((((((((((int)blockIdx.x) / 56) * 50176) + (((((int)threadIdx.x) + 1120) / 348) * 12544)) + (((((int)blockIdx.x) % 56) >> 1) * 448)) + (((((((int)threadIdx.x) >> 1) + 38) % 174) / 29) * 112)) + ((((int)blockIdx.x) & 1) * 56)) + ((((int)threadIdx.x) + 18) % 58)) - 113)] : 0.000000e+00f);
  if (((int)threadIdx.x) < 48) {
    PaddedInput_shared[(((int)threadIdx.x) + 1344)] = (((((((((int)blockIdx.x) % 56) >> 1) * 4) + ((((((int)threadIdx.x) >> 1) + 150) % 174) / 29)) < 113) && ((((((int)blockIdx.x) & 1) * 56) + (((int)threadIdx.x) + 10)) < 113)) ? Input[((((((((((int)blockIdx.x) / 56) * 50176) + (((((int)threadIdx.x) + 1344) / 348) * 12544)) + (((((int)blockIdx.x) % 56) >> 1) * 448)) + (((((((int)threadIdx.x) >> 1) + 150) % 174) / 29) * 112)) + ((((int)blockIdx.x) & 1) * 56)) + (((int)threadIdx.x) + 10)) - 113)] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 36) {
    kernel_shared[((int)threadIdx.x)] = kernel[(((((int)blockIdx.x) / 56) * 36) + ((int)threadIdx.x))];
  }
  __syncthreads();
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 56) * 348) + (((((int)threadIdx.x) % 56) / 28) * 58)) + ((((int)threadIdx.x) % 28) * 2))] * kernel_shared[((((int)threadIdx.x) / 56) * 9)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 348) + (((((int)threadIdx.x) % 56) / 28) * 58)) + ((((int)threadIdx.x) % 28) * 2)) + 116)] * kernel_shared[((((int)threadIdx.x) / 56) * 9)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 348) + (((((int)threadIdx.x) % 56) / 28) * 58)) + ((((int)threadIdx.x) % 28) * 2)) + 1)] * kernel_shared[(((((int)threadIdx.x) / 56) * 9) + 1)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 348) + (((((int)threadIdx.x) % 56) / 28) * 58)) + ((((int)threadIdx.x) % 28) * 2)) + 117)] * kernel_shared[(((((int)threadIdx.x) / 56) * 9) + 1)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 348) + (((((int)threadIdx.x) % 56) / 28) * 58)) + ((((int)threadIdx.x) % 28) * 2)) + 2)] * kernel_shared[(((((int)threadIdx.x) / 56) * 9) + 2)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 348) + (((((int)threadIdx.x) % 56) / 28) * 58)) + ((((int)threadIdx.x) % 28) * 2)) + 118)] * kernel_shared[(((((int)threadIdx.x) / 56) * 9) + 2)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 348) + (((((int)threadIdx.x) % 56) / 28) * 58)) + ((((int)threadIdx.x) % 28) * 2)) + 1)] * kernel_shared[((((int)threadIdx.x) / 56) * 9)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 348) + (((((int)threadIdx.x) % 56) / 28) * 58)) + ((((int)threadIdx.x) % 28) * 2)) + 117)] * kernel_shared[((((int)threadIdx.x) / 56) * 9)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 348) + (((((int)threadIdx.x) % 56) / 28) * 58)) + ((((int)threadIdx.x) % 28) * 2)) + 2)] * kernel_shared[(((((int)threadIdx.x) / 56) * 9) + 1)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 348) + (((((int)threadIdx.x) % 56) / 28) * 58)) + ((((int)threadIdx.x) % 28) * 2)) + 118)] * kernel_shared[(((((int)threadIdx.x) / 56) * 9) + 1)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 348) + (((((int)threadIdx.x) % 56) / 28) * 58)) + ((((int)threadIdx.x) % 28) * 2)) + 3)] * kernel_shared[(((((int)threadIdx.x) / 56) * 9) + 2)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 348) + (((((int)threadIdx.x) % 56) / 28) * 58)) + ((((int)threadIdx.x) % 28) * 2)) + 119)] * kernel_shared[(((((int)threadIdx.x) / 56) * 9) + 2)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 348) + (((((int)threadIdx.x) % 56) / 28) * 58)) + ((((int)threadIdx.x) % 28) * 2)) + 58)] * kernel_shared[(((((int)threadIdx.x) / 56) * 9) + 3)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 348) + (((((int)threadIdx.x) % 56) / 28) * 58)) + ((((int)threadIdx.x) % 28) * 2)) + 174)] * kernel_shared[(((((int)threadIdx.x) / 56) * 9) + 3)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 348) + (((((int)threadIdx.x) % 56) / 28) * 58)) + ((((int)threadIdx.x) % 28) * 2)) + 59)] * kernel_shared[(((((int)threadIdx.x) / 56) * 9) + 4)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 348) + (((((int)threadIdx.x) % 56) / 28) * 58)) + ((((int)threadIdx.x) % 28) * 2)) + 175)] * kernel_shared[(((((int)threadIdx.x) / 56) * 9) + 4)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 348) + (((((int)threadIdx.x) % 56) / 28) * 58)) + ((((int)threadIdx.x) % 28) * 2)) + 60)] * kernel_shared[(((((int)threadIdx.x) / 56) * 9) + 5)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 348) + (((((int)threadIdx.x) % 56) / 28) * 58)) + ((((int)threadIdx.x) % 28) * 2)) + 176)] * kernel_shared[(((((int)threadIdx.x) / 56) * 9) + 5)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 348) + (((((int)threadIdx.x) % 56) / 28) * 58)) + ((((int)threadIdx.x) % 28) * 2)) + 59)] * kernel_shared[(((((int)threadIdx.x) / 56) * 9) + 3)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 348) + (((((int)threadIdx.x) % 56) / 28) * 58)) + ((((int)threadIdx.x) % 28) * 2)) + 175)] * kernel_shared[(((((int)threadIdx.x) / 56) * 9) + 3)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 348) + (((((int)threadIdx.x) % 56) / 28) * 58)) + ((((int)threadIdx.x) % 28) * 2)) + 60)] * kernel_shared[(((((int)threadIdx.x) / 56) * 9) + 4)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 348) + (((((int)threadIdx.x) % 56) / 28) * 58)) + ((((int)threadIdx.x) % 28) * 2)) + 176)] * kernel_shared[(((((int)threadIdx.x) / 56) * 9) + 4)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 348) + (((((int)threadIdx.x) % 56) / 28) * 58)) + ((((int)threadIdx.x) % 28) * 2)) + 61)] * kernel_shared[(((((int)threadIdx.x) / 56) * 9) + 5)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 348) + (((((int)threadIdx.x) % 56) / 28) * 58)) + ((((int)threadIdx.x) % 28) * 2)) + 177)] * kernel_shared[(((((int)threadIdx.x) / 56) * 9) + 5)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 348) + (((((int)threadIdx.x) % 56) / 28) * 58)) + ((((int)threadIdx.x) % 28) * 2)) + 116)] * kernel_shared[(((((int)threadIdx.x) / 56) * 9) + 6)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 348) + (((((int)threadIdx.x) % 56) / 28) * 58)) + ((((int)threadIdx.x) % 28) * 2)) + 232)] * kernel_shared[(((((int)threadIdx.x) / 56) * 9) + 6)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 348) + (((((int)threadIdx.x) % 56) / 28) * 58)) + ((((int)threadIdx.x) % 28) * 2)) + 117)] * kernel_shared[(((((int)threadIdx.x) / 56) * 9) + 7)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 348) + (((((int)threadIdx.x) % 56) / 28) * 58)) + ((((int)threadIdx.x) % 28) * 2)) + 233)] * kernel_shared[(((((int)threadIdx.x) / 56) * 9) + 7)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 348) + (((((int)threadIdx.x) % 56) / 28) * 58)) + ((((int)threadIdx.x) % 28) * 2)) + 118)] * kernel_shared[(((((int)threadIdx.x) / 56) * 9) + 8)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 348) + (((((int)threadIdx.x) % 56) / 28) * 58)) + ((((int)threadIdx.x) % 28) * 2)) + 234)] * kernel_shared[(((((int)threadIdx.x) / 56) * 9) + 8)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 348) + (((((int)threadIdx.x) % 56) / 28) * 58)) + ((((int)threadIdx.x) % 28) * 2)) + 117)] * kernel_shared[(((((int)threadIdx.x) / 56) * 9) + 6)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 348) + (((((int)threadIdx.x) % 56) / 28) * 58)) + ((((int)threadIdx.x) % 28) * 2)) + 233)] * kernel_shared[(((((int)threadIdx.x) / 56) * 9) + 6)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 348) + (((((int)threadIdx.x) % 56) / 28) * 58)) + ((((int)threadIdx.x) % 28) * 2)) + 118)] * kernel_shared[(((((int)threadIdx.x) / 56) * 9) + 7)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 348) + (((((int)threadIdx.x) % 56) / 28) * 58)) + ((((int)threadIdx.x) % 28) * 2)) + 234)] * kernel_shared[(((((int)threadIdx.x) / 56) * 9) + 7)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 348) + (((((int)threadIdx.x) % 56) / 28) * 58)) + ((((int)threadIdx.x) % 28) * 2)) + 119)] * kernel_shared[(((((int)threadIdx.x) / 56) * 9) + 8)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 348) + (((((int)threadIdx.x) % 56) / 28) * 58)) + ((((int)threadIdx.x) % 28) * 2)) + 235)] * kernel_shared[(((((int)threadIdx.x) / 56) * 9) + 8)]));
  for (int i3_inner = 0; i3_inner < 2; ++i3_inner) {
    compute[((((((((((int)blockIdx.x) / 56) * 50176) + ((((int)threadIdx.x) / 56) * 12544)) + (((((int)blockIdx.x) % 56) >> 1) * 448)) + (((((int)threadIdx.x) % 56) / 28) * 112)) + ((((int)blockIdx.x) & 1) * 56)) + ((((int)threadIdx.x) % 28) * 2)) + i3_inner)] = max(DepthwiseConv2d[i3_inner], 0.000000e+00f);
    compute[(((((((((((int)blockIdx.x) / 56) * 50176) + ((((int)threadIdx.x) / 56) * 12544)) + (((((int)blockIdx.x) % 56) >> 1) * 448)) + (((((int)threadIdx.x) % 56) / 28) * 112)) + ((((int)blockIdx.x) & 1) * 56)) + ((((int)threadIdx.x) % 28) * 2)) + i3_inner) + 224)] = max(DepthwiseConv2d[(i3_inner + 2)], 0.000000e+00f);
  }
}


