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
  __shared__ float PaddedInput_shared[1440];
  __shared__ float kernel_shared[72];
  DepthwiseConv2d[0] = 0.000000e+00f;
  DepthwiseConv2d[2] = 0.000000e+00f;
  DepthwiseConv2d[1] = 0.000000e+00f;
  DepthwiseConv2d[3] = 0.000000e+00f;
  PaddedInput_shared[((int)threadIdx.x)] = (((((1 <= ((((((int)blockIdx.x) % 28) >> 1) * 4) + ((((int)threadIdx.x) % 180) / 30))) && (((((((int)blockIdx.x) % 28) >> 1) * 4) + ((((int)threadIdx.x) % 180) / 30)) < 57)) && (1 <= (((((int)blockIdx.x) & 1) * 28) + (((int)threadIdx.x) % 30)))) && ((((((int)blockIdx.x) & 1) * 28) + (((int)threadIdx.x) % 30)) < 57)) ? Input[((((((((((int)blockIdx.x) / 28) * 25088) + ((((int)threadIdx.x) / 180) * 3136)) + (((((int)blockIdx.x) % 28) >> 1) * 224)) + (((((int)threadIdx.x) % 180) / 30) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + (((int)threadIdx.x) % 30)) - 57)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 224)] = (((((1 <= ((((((int)blockIdx.x) % 28) >> 1) * 4) + ((((((int)threadIdx.x) >> 1) + 22) % 90) / 15))) && (((((((int)blockIdx.x) % 28) >> 1) * 4) + ((((((int)threadIdx.x) >> 1) + 22) % 90) / 15)) < 57)) && (1 <= (((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 14) % 30)))) && ((((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 14) % 30)) < 57)) ? Input[((((((((((int)blockIdx.x) / 28) * 25088) + (((((int)threadIdx.x) + 224) / 180) * 3136)) + (((((int)blockIdx.x) % 28) >> 1) * 224)) + (((((((int)threadIdx.x) >> 1) + 22) % 90) / 15) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + ((((int)threadIdx.x) + 14) % 30)) - 57)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 448)] = (((((1 <= ((((((int)blockIdx.x) % 28) >> 1) * 4) + ((((((int)threadIdx.x) >> 1) + 44) % 90) / 15))) && (((((((int)blockIdx.x) % 28) >> 1) * 4) + ((((((int)threadIdx.x) >> 1) + 44) % 90) / 15)) < 57)) && (1 <= (((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 28) % 30)))) && ((((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 28) % 30)) < 57)) ? Input[((((((((((int)blockIdx.x) / 28) * 25088) + (((((int)threadIdx.x) + 448) / 180) * 3136)) + (((((int)blockIdx.x) % 28) >> 1) * 224)) + (((((((int)threadIdx.x) >> 1) + 44) % 90) / 15) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + ((((int)threadIdx.x) + 28) % 30)) - 57)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 672)] = (((((1 <= ((((((int)blockIdx.x) % 28) >> 1) * 4) + ((((((int)threadIdx.x) >> 1) + 66) % 90) / 15))) && (((((((int)blockIdx.x) % 28) >> 1) * 4) + ((((((int)threadIdx.x) >> 1) + 66) % 90) / 15)) < 57)) && (1 <= (((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 12) % 30)))) && ((((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 12) % 30)) < 57)) ? Input[((((((((((int)blockIdx.x) / 28) * 25088) + (((((int)threadIdx.x) + 672) / 180) * 3136)) + (((((int)blockIdx.x) % 28) >> 1) * 224)) + (((((((int)threadIdx.x) >> 1) + 66) % 90) / 15) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + ((((int)threadIdx.x) + 12) % 30)) - 57)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 896)] = (((((1 <= ((((((int)blockIdx.x) % 28) >> 1) * 4) + ((((((int)threadIdx.x) >> 1) + 88) % 90) / 15))) && (((((((int)blockIdx.x) % 28) >> 1) * 4) + ((((((int)threadIdx.x) >> 1) + 88) % 90) / 15)) < 57)) && (1 <= (((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 26) % 30)))) && ((((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 26) % 30)) < 57)) ? Input[((((((((((int)blockIdx.x) / 28) * 25088) + (((((int)threadIdx.x) + 896) / 180) * 3136)) + (((((int)blockIdx.x) % 28) >> 1) * 224)) + (((((((int)threadIdx.x) >> 1) + 88) % 90) / 15) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + ((((int)threadIdx.x) + 26) % 30)) - 57)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 1120)] = (((((1 <= ((((((int)blockIdx.x) % 28) >> 1) * 4) + ((((((int)threadIdx.x) >> 1) + 20) % 90) / 15))) && (((((((int)blockIdx.x) % 28) >> 1) * 4) + ((((((int)threadIdx.x) >> 1) + 20) % 90) / 15)) < 57)) && (1 <= (((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 10) % 30)))) && ((((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 10) % 30)) < 57)) ? Input[((((((((((int)blockIdx.x) / 28) * 25088) + (((((int)threadIdx.x) + 1120) / 180) * 3136)) + (((((int)blockIdx.x) % 28) >> 1) * 224)) + (((((((int)threadIdx.x) >> 1) + 20) % 90) / 15) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + ((((int)threadIdx.x) + 10) % 30)) - 57)] : 0.000000e+00f);
  if (((int)threadIdx.x) < 96) {
    PaddedInput_shared[(((int)threadIdx.x) + 1344)] = ((((((((((int)blockIdx.x) % 28) >> 1) * 4) + ((((((int)threadIdx.x) >> 1) + 42) % 90) / 15)) < 57) && (1 <= (((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 24) % 30)))) && ((((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 24) % 30)) < 57)) ? Input[((((((((((int)blockIdx.x) / 28) * 25088) + (((((int)threadIdx.x) + 1344) / 180) * 3136)) + (((((int)blockIdx.x) % 28) >> 1) * 224)) + (((((((int)threadIdx.x) >> 1) + 42) % 90) / 15) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + ((((int)threadIdx.x) + 24) % 30)) - 57)] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 72) {
    kernel_shared[((int)threadIdx.x)] = kernel[(((((int)blockIdx.x) / 28) * 72) + ((int)threadIdx.x))];
  }
  __syncthreads();
  for (int di_outer_inner = 0; di_outer_inner < 3; ++di_outer_inner) {
    DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((((int)threadIdx.x) / 28) * 180) + (((((int)threadIdx.x) % 28) / 14) * 60)) + (di_outer_inner * 30)) + ((((int)threadIdx.x) % 14) * 2))] * kernel_shared[(((((int)threadIdx.x) / 28) * 9) + (di_outer_inner * 3))]));
    DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[((((((((int)threadIdx.x) / 28) * 180) + (((((int)threadIdx.x) % 28) / 14) * 60)) + (di_outer_inner * 30)) + ((((int)threadIdx.x) % 14) * 2)) + 30)] * kernel_shared[(((((int)threadIdx.x) / 28) * 9) + (di_outer_inner * 3))]));
    DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((((int)threadIdx.x) / 28) * 180) + (((((int)threadIdx.x) % 28) / 14) * 60)) + (di_outer_inner * 30)) + ((((int)threadIdx.x) % 14) * 2)) + 1)] * kernel_shared[(((((int)threadIdx.x) / 28) * 9) + (di_outer_inner * 3))]));
    DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[((((((((int)threadIdx.x) / 28) * 180) + (((((int)threadIdx.x) % 28) / 14) * 60)) + (di_outer_inner * 30)) + ((((int)threadIdx.x) % 14) * 2)) + 31)] * kernel_shared[(((((int)threadIdx.x) / 28) * 9) + (di_outer_inner * 3))]));
    DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((((int)threadIdx.x) / 28) * 180) + (((((int)threadIdx.x) % 28) / 14) * 60)) + (di_outer_inner * 30)) + ((((int)threadIdx.x) % 14) * 2)) + 1)] * kernel_shared[((((((int)threadIdx.x) / 28) * 9) + (di_outer_inner * 3)) + 1)]));
    DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[((((((((int)threadIdx.x) / 28) * 180) + (((((int)threadIdx.x) % 28) / 14) * 60)) + (di_outer_inner * 30)) + ((((int)threadIdx.x) % 14) * 2)) + 31)] * kernel_shared[((((((int)threadIdx.x) / 28) * 9) + (di_outer_inner * 3)) + 1)]));
    DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((((int)threadIdx.x) / 28) * 180) + (((((int)threadIdx.x) % 28) / 14) * 60)) + (di_outer_inner * 30)) + ((((int)threadIdx.x) % 14) * 2)) + 2)] * kernel_shared[((((((int)threadIdx.x) / 28) * 9) + (di_outer_inner * 3)) + 1)]));
    DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[((((((((int)threadIdx.x) / 28) * 180) + (((((int)threadIdx.x) % 28) / 14) * 60)) + (di_outer_inner * 30)) + ((((int)threadIdx.x) % 14) * 2)) + 32)] * kernel_shared[((((((int)threadIdx.x) / 28) * 9) + (di_outer_inner * 3)) + 1)]));
    DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((((int)threadIdx.x) / 28) * 180) + (((((int)threadIdx.x) % 28) / 14) * 60)) + (di_outer_inner * 30)) + ((((int)threadIdx.x) % 14) * 2)) + 2)] * kernel_shared[((((((int)threadIdx.x) / 28) * 9) + (di_outer_inner * 3)) + 2)]));
    DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[((((((((int)threadIdx.x) / 28) * 180) + (((((int)threadIdx.x) % 28) / 14) * 60)) + (di_outer_inner * 30)) + ((((int)threadIdx.x) % 14) * 2)) + 32)] * kernel_shared[((((((int)threadIdx.x) / 28) * 9) + (di_outer_inner * 3)) + 2)]));
    DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((((int)threadIdx.x) / 28) * 180) + (((((int)threadIdx.x) % 28) / 14) * 60)) + (di_outer_inner * 30)) + ((((int)threadIdx.x) % 14) * 2)) + 3)] * kernel_shared[((((((int)threadIdx.x) / 28) * 9) + (di_outer_inner * 3)) + 2)]));
    DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[((((((((int)threadIdx.x) / 28) * 180) + (((((int)threadIdx.x) % 28) / 14) * 60)) + (di_outer_inner * 30)) + ((((int)threadIdx.x) % 14) * 2)) + 33)] * kernel_shared[((((((int)threadIdx.x) / 28) * 9) + (di_outer_inner * 3)) + 2)]));
  }
  for (int i2_inner = 0; i2_inner < 2; ++i2_inner) {
    for (int i3_inner = 0; i3_inner < 2; ++i3_inner) {
      compute[(((((((((((int)blockIdx.x) / 28) * 25088) + ((((int)threadIdx.x) / 28) * 3136)) + (((((int)blockIdx.x) % 28) >> 1) * 224)) + (((((int)threadIdx.x) % 28) / 14) * 112)) + (i2_inner * 56)) + ((((int)blockIdx.x) & 1) * 28)) + ((((int)threadIdx.x) % 14) * 2)) + i3_inner)] = max(DepthwiseConv2d[((i2_inner * 2) + i3_inner)], 0.000000e+00f);
    }
  }
}


