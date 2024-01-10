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
  float DepthwiseConv2d[2];
  __shared__ float PaddedInput_shared[1938];
  __shared__ float kernel_shared[18];
  DepthwiseConv2d[0] = 0.000000e+00f;
  DepthwiseConv2d[1] = 0.000000e+00f;
  PaddedInput_shared[((int)threadIdx.x)] = (((1 <= ((((((int)blockIdx.x) % 14) / 7) * 56) + (((int)threadIdx.x) / 17))) && (1 <= (((((int)blockIdx.x) % 7) * 16) + (((int)threadIdx.x) % 17)))) ? Input[(((((((((int)blockIdx.x) / 14) * 25088) + (((((int)blockIdx.x) % 14) / 7) * 6272)) + ((((int)threadIdx.x) / 17) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) % 17)) - 113)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 224)] = ((1 <= (((((int)blockIdx.x) % 7) * 16) + ((((int)threadIdx.x) + 3) % 17))) ? Input[(((((((((int)blockIdx.x) / 14) * 25088) + (((((int)blockIdx.x) % 14) / 7) * 6272)) + (((((int)threadIdx.x) + 224) / 17) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + ((((int)threadIdx.x) + 3) % 17)) - 113)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 448)] = ((1 <= (((((int)blockIdx.x) % 7) * 16) + ((((int)threadIdx.x) + 6) % 17))) ? Input[(((((((((int)blockIdx.x) / 14) * 25088) + (((((int)blockIdx.x) % 14) / 7) * 6272)) + (((((int)threadIdx.x) + 448) / 17) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + ((((int)threadIdx.x) + 6) % 17)) - 113)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 672)] = ((1 <= (((((int)blockIdx.x) % 7) * 16) + ((((int)threadIdx.x) + 9) % 17))) ? Input[(((((((((int)blockIdx.x) / 14) * 25088) + (((((int)blockIdx.x) % 14) / 7) * 6272)) + (((((int)threadIdx.x) + 672) / 17) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + ((((int)threadIdx.x) + 9) % 17)) - 113)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 896)] = (((1 <= ((((((int)blockIdx.x) % 14) / 7) * 56) + (((((int)threadIdx.x) + 896) % 969) / 17))) && (1 <= (((((int)blockIdx.x) % 7) * 16) + ((((int)threadIdx.x) + 12) % 17)))) ? Input[((((((((((int)blockIdx.x) / 14) * 25088) + (((((int)threadIdx.x) + 896) / 969) * 12544)) + (((((int)blockIdx.x) % 14) / 7) * 6272)) + ((((((int)threadIdx.x) + 896) % 969) / 17) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + ((((int)threadIdx.x) + 12) % 17)) - 113)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 1120)] = ((1 <= (((((int)blockIdx.x) % 7) * 16) + ((((int)threadIdx.x) + 15) % 17))) ? Input[((((((((((int)blockIdx.x) / 14) * 25088) + (((((int)threadIdx.x) + 1120) / 969) * 12544)) + (((((int)blockIdx.x) % 14) / 7) * 6272)) + ((((((int)threadIdx.x) + 151) % 969) / 17) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + ((((int)threadIdx.x) + 15) % 17)) - 113)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 1344)] = ((1 <= (((((int)blockIdx.x) % 7) * 16) + ((((int)threadIdx.x) + 1) % 17))) ? Input[((((((((((int)blockIdx.x) / 14) * 25088) + (((((int)threadIdx.x) + 1344) / 969) * 12544)) + (((((int)blockIdx.x) % 14) / 7) * 6272)) + ((((((int)threadIdx.x) + 375) % 969) / 17) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + ((((int)threadIdx.x) + 1) % 17)) - 113)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 1568)] = ((1 <= (((((int)blockIdx.x) % 7) * 16) + ((((int)threadIdx.x) + 4) % 17))) ? Input[((((((((((int)blockIdx.x) / 14) * 25088) + (((((int)threadIdx.x) + 1568) / 969) * 12544)) + (((((int)blockIdx.x) % 14) / 7) * 6272)) + ((((((int)threadIdx.x) + 599) % 969) / 17) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + ((((int)threadIdx.x) + 4) % 17)) - 113)] : 0.000000e+00f);
  if (((int)threadIdx.x) < 146) {
    PaddedInput_shared[(((int)threadIdx.x) + 1792)] = ((1 <= (((((int)blockIdx.x) % 7) * 16) + ((((int)threadIdx.x) + 7) % 17))) ? Input[((((((((((int)blockIdx.x) / 14) * 25088) + (((((int)threadIdx.x) + 1792) / 969) * 12544)) + (((((int)blockIdx.x) % 14) / 7) * 6272)) + ((((((int)threadIdx.x) + 823) % 969) / 17) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + ((((int)threadIdx.x) + 7) % 17)) - 113)] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 18) {
    kernel_shared[((int)threadIdx.x)] = kernel[(((((int)blockIdx.x) / 14) * 18) + ((int)threadIdx.x))];
  }
  __syncthreads();
  for (int di_outer_inner = 0; di_outer_inner < 3; ++di_outer_inner) {
    DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((((int)threadIdx.x) / 112) * 969) + (((((int)threadIdx.x) % 112) >> 2) * 34)) + (di_outer_inner * 17)) + ((((int)threadIdx.x) & 3) * 4))] * kernel_shared[(((((int)threadIdx.x) / 112) * 9) + (di_outer_inner * 3))]));
    DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((((int)threadIdx.x) / 112) * 969) + (((((int)threadIdx.x) % 112) >> 2) * 34)) + (di_outer_inner * 17)) + ((((int)threadIdx.x) & 3) * 4)) + 2)] * kernel_shared[(((((int)threadIdx.x) / 112) * 9) + (di_outer_inner * 3))]));
    DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((((int)threadIdx.x) / 112) * 969) + (((((int)threadIdx.x) % 112) >> 2) * 34)) + (di_outer_inner * 17)) + ((((int)threadIdx.x) & 3) * 4)) + 1)] * kernel_shared[((((((int)threadIdx.x) / 112) * 9) + (di_outer_inner * 3)) + 1)]));
    DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((((int)threadIdx.x) / 112) * 969) + (((((int)threadIdx.x) % 112) >> 2) * 34)) + (di_outer_inner * 17)) + ((((int)threadIdx.x) & 3) * 4)) + 3)] * kernel_shared[((((((int)threadIdx.x) / 112) * 9) + (di_outer_inner * 3)) + 1)]));
    DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((((int)threadIdx.x) / 112) * 969) + (((((int)threadIdx.x) % 112) >> 2) * 34)) + (di_outer_inner * 17)) + ((((int)threadIdx.x) & 3) * 4)) + 2)] * kernel_shared[((((((int)threadIdx.x) / 112) * 9) + (di_outer_inner * 3)) + 2)]));
    DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((((int)threadIdx.x) / 112) * 969) + (((((int)threadIdx.x) % 112) >> 2) * 34)) + (di_outer_inner * 17)) + ((((int)threadIdx.x) & 3) * 4)) + 4)] * kernel_shared[((((((int)threadIdx.x) / 112) * 9) + (di_outer_inner * 3)) + 2)]));
  }
  for (int i3_inner = 0; i3_inner < 2; ++i3_inner) {
    compute[((((((((((int)blockIdx.x) / 14) * 6272) + ((((int)threadIdx.x) / 112) * 3136)) + (((((int)blockIdx.x) % 14) / 7) * 1568)) + (((((int)threadIdx.x) % 112) >> 2) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + i3_inner)] = max(DepthwiseConv2d[i3_inner], 0.000000e+00f);
  }
}


