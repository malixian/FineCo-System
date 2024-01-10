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
extern "C" __global__ void __launch_bounds__(4) candidate0(float* __restrict__ data, float* __restrict__ weight, float* __restrict__ compute, float* __restrict__ bias) {
  float T_matmul_NT[1];
  __shared__ float data_shared[512];
  __shared__ float weight_shared[2048];
  T_matmul_NT[0] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 9; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 32; ++ax0_ax1_fused_outer_outer) {
      *(float4*)(data_shared + ((ax0_ax1_fused_outer_outer * 16) + (((int)threadIdx.x) * 4))) = *(float4*)(data + (((k_outer_outer * 512) + (ax0_ax1_fused_outer_outer * 16)) + (((int)threadIdx.x) * 4)));
    }
    for (int ax0_ax1_fused_outer_outer1 = 0; ax0_ax1_fused_outer_outer1 < 128; ++ax0_ax1_fused_outer_outer1) {
      *(float4*)(weight_shared + ((ax0_ax1_fused_outer_outer1 * 16) + (((int)threadIdx.x) * 4))) = *(float4*)(weight + (((((((int)blockIdx.x) * 18432) + ((ax0_ax1_fused_outer_outer1 >> 5) * 4608)) + (k_outer_outer * 512)) + ((ax0_ax1_fused_outer_outer1 & 31) * 16)) + (((int)threadIdx.x) * 4)));
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 8; ++k_outer_inner) {
      for (int k_inner = 0; k_inner < 64; ++k_inner) {
        T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + k_inner)] * weight_shared[(((((int)threadIdx.x) * 512) + (k_outer_inner * 64)) + k_inner)]));
      }
    }
  }
  compute[((((int)blockIdx.x) * 4) + ((int)threadIdx.x))] = (T_matmul_NT[0] + bias[((((int)blockIdx.x) * 4) + ((int)threadIdx.x))]);
}


