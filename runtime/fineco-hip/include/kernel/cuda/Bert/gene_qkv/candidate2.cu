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
extern "C" __global__ void __launch_bounds__(32) candidate2(float* __restrict__ I, float* __restrict__ W, float* __restrict__ T_batch_matmul_NT) {
  float T_batch_matmul_NT_local[1];
  __shared__ float I_shared[96];
  __shared__ float W_shared[3072];
  T_batch_matmul_NT_local[0] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 8; ++k_outer_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 24) {
      *(float4*)(I_shared + (((int)threadIdx.x) * 4)) = *(float4*)(I + ((((((int)blockIdx.x) / 24) * 768) + (k_outer_outer * 96)) + (((int)threadIdx.x) * 4)));
    }
    for (int ax0_ax1_fused_ax2_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_outer_outer < 96; ++ax0_ax1_fused_ax2_fused_outer_outer) {
      W_shared[((ax0_ax1_fused_ax2_fused_outer_outer * 32) + ((int)threadIdx.x))] = W[((((((((int)blockIdx.x) % 24) * 24576) + ((ax0_ax1_fused_ax2_fused_outer_outer / 3) * 768)) + (k_outer_outer * 96)) + ((ax0_ax1_fused_ax2_fused_outer_outer % 3) * 32)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 48; ++k_outer_inner) {
      for (int k_inner = 0; k_inner < 2; ++k_inner) {
        T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[((k_outer_inner * 2) + k_inner)] * W_shared[(((((int)threadIdx.x) * 96) + (k_outer_inner * 2)) + k_inner)]));
      }
    }
  }
  T_batch_matmul_NT[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = T_batch_matmul_NT_local[0];
}


