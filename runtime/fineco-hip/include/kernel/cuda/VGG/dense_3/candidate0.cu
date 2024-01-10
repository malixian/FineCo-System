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
extern "C" __global__ void __launch_bounds__(250) candidate0(float* __restrict__ data, float* __restrict__ weight, float* __restrict__ compute, float* __restrict__ bias) {
  float T_matmul_NT[1];
  __shared__ float data_shared[8];
  __shared__ float weight_shared[2000];
  T_matmul_NT[0] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 512; ++k_outer_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 2) {
      *(float4*)(data_shared + (((int)threadIdx.x) * 4)) = *(float4*)(data + ((k_outer_outer * 8) + (((int)threadIdx.x) * 4)));
    }
    weight_shared[((int)threadIdx.x)] = weight[((((((int)blockIdx.x) * 1024000) + ((((int)threadIdx.x) >> 3) * 4096)) + (k_outer_outer * 8)) + (((int)threadIdx.x) & 7))];
    weight_shared[(((int)threadIdx.x) + 250)] = weight[((((((int)blockIdx.x) * 1024000) + (((((int)threadIdx.x) + 250) >> 3) * 4096)) + (k_outer_outer * 8)) + ((((int)threadIdx.x) + 2) & 7))];
    weight_shared[(((int)threadIdx.x) + 500)] = weight[((((((int)blockIdx.x) * 1024000) + (((((int)threadIdx.x) + 500) >> 3) * 4096)) + (k_outer_outer * 8)) + ((((int)threadIdx.x) + 4) & 7))];
    weight_shared[(((int)threadIdx.x) + 750)] = weight[((((((int)blockIdx.x) * 1024000) + (((((int)threadIdx.x) + 750) >> 3) * 4096)) + (k_outer_outer * 8)) + ((((int)threadIdx.x) + 6) & 7))];
    weight_shared[(((int)threadIdx.x) + 1000)] = weight[(((((((int)blockIdx.x) * 1024000) + ((((int)threadIdx.x) >> 3) * 4096)) + (k_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 512000)];
    weight_shared[(((int)threadIdx.x) + 1250)] = weight[((((((int)blockIdx.x) * 1024000) + (((((int)threadIdx.x) + 1250) >> 3) * 4096)) + (k_outer_outer * 8)) + ((((int)threadIdx.x) + 2) & 7))];
    weight_shared[(((int)threadIdx.x) + 1500)] = weight[((((((int)blockIdx.x) * 1024000) + (((((int)threadIdx.x) + 1500) >> 3) * 4096)) + (k_outer_outer * 8)) + ((((int)threadIdx.x) + 4) & 7))];
    weight_shared[(((int)threadIdx.x) + 1750)] = weight[((((((int)blockIdx.x) * 1024000) + (((((int)threadIdx.x) + 1750) >> 3) * 4096)) + (k_outer_outer * 8)) + ((((int)threadIdx.x) + 6) & 7))];
    __syncthreads();
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[0] * weight_shared[(((int)threadIdx.x) * 8)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[1] * weight_shared[((((int)threadIdx.x) * 8) + 1)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[2] * weight_shared[((((int)threadIdx.x) * 8) + 2)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[3] * weight_shared[((((int)threadIdx.x) * 8) + 3)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[4] * weight_shared[((((int)threadIdx.x) * 8) + 4)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[5] * weight_shared[((((int)threadIdx.x) * 8) + 5)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[6] * weight_shared[((((int)threadIdx.x) * 8) + 6)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[7] * weight_shared[((((int)threadIdx.x) * 8) + 7)]));
  }
  compute[((((int)blockIdx.x) * 250) + ((int)threadIdx.x))] = (T_matmul_NT[0] + bias[((((int)blockIdx.x) * 250) + ((int)threadIdx.x))]);
}


