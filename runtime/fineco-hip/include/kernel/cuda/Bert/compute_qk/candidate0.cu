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
extern "C" __global__ void __launch_bounds__(5) candidate0(float* __restrict__ Q, float* __restrict__ K, float* __restrict__ T_divide) {
  float T_batch_matmul_NT[1];
  __shared__ float Q_shared[24];
  __shared__ float K_shared[120];
  T_batch_matmul_NT[0] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 32; ++k_outer_outer) {
    __syncthreads();
    Q_shared[((int)threadIdx.x)] = Q[(((((int)blockIdx.x) * 768) + (k_outer_outer * 24)) + ((int)threadIdx.x))];
    Q_shared[(((int)threadIdx.x) + 5)] = Q[((((((int)blockIdx.x) * 768) + (k_outer_outer * 24)) + ((int)threadIdx.x)) + 5)];
    Q_shared[(((int)threadIdx.x) + 10)] = Q[((((((int)blockIdx.x) * 768) + (k_outer_outer * 24)) + ((int)threadIdx.x)) + 10)];
    Q_shared[(((int)threadIdx.x) + 15)] = Q[((((((int)blockIdx.x) * 768) + (k_outer_outer * 24)) + ((int)threadIdx.x)) + 15)];
    if (((int)threadIdx.x) < 4) {
      Q_shared[(((int)threadIdx.x) + 20)] = Q[((((((int)blockIdx.x) * 768) + (k_outer_outer * 24)) + ((int)threadIdx.x)) + 20)];
    }
    K_shared[((int)threadIdx.x)] = K[((k_outer_outer * 24) + ((int)threadIdx.x))];
    K_shared[(((int)threadIdx.x) + 5)] = K[(((k_outer_outer * 24) + ((int)threadIdx.x)) + 5)];
    K_shared[(((int)threadIdx.x) + 10)] = K[(((k_outer_outer * 24) + ((int)threadIdx.x)) + 10)];
    K_shared[(((int)threadIdx.x) + 15)] = K[(((k_outer_outer * 24) + ((int)threadIdx.x)) + 15)];
    K_shared[(((int)threadIdx.x) + 20)] = K[(((((((int)threadIdx.x) + 20) / 24) * 768) + (k_outer_outer * 24)) + ((((int)threadIdx.x) + 20) % 24))];
    K_shared[(((int)threadIdx.x) + 25)] = K[(((((((int)threadIdx.x) + 25) / 24) * 768) + (k_outer_outer * 24)) + (((int)threadIdx.x) + 1))];
    K_shared[(((int)threadIdx.x) + 30)] = K[(((((((int)threadIdx.x) + 30) / 24) * 768) + (k_outer_outer * 24)) + (((int)threadIdx.x) + 6))];
    K_shared[(((int)threadIdx.x) + 35)] = K[(((((((int)threadIdx.x) + 35) / 24) * 768) + (k_outer_outer * 24)) + (((int)threadIdx.x) + 11))];
    K_shared[(((int)threadIdx.x) + 40)] = K[(((((((int)threadIdx.x) + 40) / 24) * 768) + (k_outer_outer * 24)) + (((int)threadIdx.x) + 16))];
    K_shared[(((int)threadIdx.x) + 45)] = K[(((((((int)threadIdx.x) + 45) / 24) * 768) + (k_outer_outer * 24)) + ((((int)threadIdx.x) + 21) % 24))];
    K_shared[(((int)threadIdx.x) + 50)] = K[(((((((int)threadIdx.x) + 50) / 24) * 768) + (k_outer_outer * 24)) + (((int)threadIdx.x) + 2))];
    K_shared[(((int)threadIdx.x) + 55)] = K[(((((((int)threadIdx.x) + 55) / 24) * 768) + (k_outer_outer * 24)) + (((int)threadIdx.x) + 7))];
    K_shared[(((int)threadIdx.x) + 60)] = K[(((((((int)threadIdx.x) + 60) / 24) * 768) + (k_outer_outer * 24)) + (((int)threadIdx.x) + 12))];
    K_shared[(((int)threadIdx.x) + 65)] = K[(((((((int)threadIdx.x) + 65) / 24) * 768) + (k_outer_outer * 24)) + (((int)threadIdx.x) + 17))];
    K_shared[(((int)threadIdx.x) + 70)] = K[(((((((int)threadIdx.x) + 70) / 24) * 768) + (k_outer_outer * 24)) + ((((int)threadIdx.x) + 22) % 24))];
    K_shared[(((int)threadIdx.x) + 75)] = K[(((((((int)threadIdx.x) + 75) / 24) * 768) + (k_outer_outer * 24)) + (((int)threadIdx.x) + 3))];
    K_shared[(((int)threadIdx.x) + 80)] = K[(((((((int)threadIdx.x) + 80) / 24) * 768) + (k_outer_outer * 24)) + (((int)threadIdx.x) + 8))];
    K_shared[(((int)threadIdx.x) + 85)] = K[(((((((int)threadIdx.x) + 85) / 24) * 768) + (k_outer_outer * 24)) + (((int)threadIdx.x) + 13))];
    K_shared[(((int)threadIdx.x) + 90)] = K[(((((((int)threadIdx.x) + 90) / 24) * 768) + (k_outer_outer * 24)) + (((int)threadIdx.x) + 18))];
    K_shared[(((int)threadIdx.x) + 95)] = K[(((((((int)threadIdx.x) + 95) / 24) * 768) + (k_outer_outer * 24)) + ((((int)threadIdx.x) + 23) % 24))];
    K_shared[(((int)threadIdx.x) + 100)] = K[(((((((int)threadIdx.x) + 100) / 24) * 768) + (k_outer_outer * 24)) + (((int)threadIdx.x) + 4))];
    K_shared[(((int)threadIdx.x) + 105)] = K[(((((((int)threadIdx.x) + 105) / 24) * 768) + (k_outer_outer * 24)) + (((int)threadIdx.x) + 9))];
    K_shared[(((int)threadIdx.x) + 110)] = K[(((((((int)threadIdx.x) + 110) / 24) * 768) + (k_outer_outer * 24)) + (((int)threadIdx.x) + 14))];
    K_shared[(((int)threadIdx.x) + 115)] = K[(((((((int)threadIdx.x) + 115) / 24) * 768) + (k_outer_outer * 24)) + (((int)threadIdx.x) + 19))];
    __syncthreads();
    T_batch_matmul_NT[0] = (T_batch_matmul_NT[0] + (Q_shared[0] * K_shared[(((int)threadIdx.x) * 24)]));
    T_batch_matmul_NT[0] = (T_batch_matmul_NT[0] + (Q_shared[1] * K_shared[((((int)threadIdx.x) * 24) + 1)]));
    T_batch_matmul_NT[0] = (T_batch_matmul_NT[0] + (Q_shared[2] * K_shared[((((int)threadIdx.x) * 24) + 2)]));
    T_batch_matmul_NT[0] = (T_batch_matmul_NT[0] + (Q_shared[3] * K_shared[((((int)threadIdx.x) * 24) + 3)]));
    T_batch_matmul_NT[0] = (T_batch_matmul_NT[0] + (Q_shared[4] * K_shared[((((int)threadIdx.x) * 24) + 4)]));
    T_batch_matmul_NT[0] = (T_batch_matmul_NT[0] + (Q_shared[5] * K_shared[((((int)threadIdx.x) * 24) + 5)]));
    T_batch_matmul_NT[0] = (T_batch_matmul_NT[0] + (Q_shared[6] * K_shared[((((int)threadIdx.x) * 24) + 6)]));
    T_batch_matmul_NT[0] = (T_batch_matmul_NT[0] + (Q_shared[7] * K_shared[((((int)threadIdx.x) * 24) + 7)]));
    T_batch_matmul_NT[0] = (T_batch_matmul_NT[0] + (Q_shared[8] * K_shared[((((int)threadIdx.x) * 24) + 8)]));
    T_batch_matmul_NT[0] = (T_batch_matmul_NT[0] + (Q_shared[9] * K_shared[((((int)threadIdx.x) * 24) + 9)]));
    T_batch_matmul_NT[0] = (T_batch_matmul_NT[0] + (Q_shared[10] * K_shared[((((int)threadIdx.x) * 24) + 10)]));
    T_batch_matmul_NT[0] = (T_batch_matmul_NT[0] + (Q_shared[11] * K_shared[((((int)threadIdx.x) * 24) + 11)]));
    T_batch_matmul_NT[0] = (T_batch_matmul_NT[0] + (Q_shared[12] * K_shared[((((int)threadIdx.x) * 24) + 12)]));
    T_batch_matmul_NT[0] = (T_batch_matmul_NT[0] + (Q_shared[13] * K_shared[((((int)threadIdx.x) * 24) + 13)]));
    T_batch_matmul_NT[0] = (T_batch_matmul_NT[0] + (Q_shared[14] * K_shared[((((int)threadIdx.x) * 24) + 14)]));
    T_batch_matmul_NT[0] = (T_batch_matmul_NT[0] + (Q_shared[15] * K_shared[((((int)threadIdx.x) * 24) + 15)]));
    T_batch_matmul_NT[0] = (T_batch_matmul_NT[0] + (Q_shared[16] * K_shared[((((int)threadIdx.x) * 24) + 16)]));
    T_batch_matmul_NT[0] = (T_batch_matmul_NT[0] + (Q_shared[17] * K_shared[((((int)threadIdx.x) * 24) + 17)]));
    T_batch_matmul_NT[0] = (T_batch_matmul_NT[0] + (Q_shared[18] * K_shared[((((int)threadIdx.x) * 24) + 18)]));
    T_batch_matmul_NT[0] = (T_batch_matmul_NT[0] + (Q_shared[19] * K_shared[((((int)threadIdx.x) * 24) + 19)]));
    T_batch_matmul_NT[0] = (T_batch_matmul_NT[0] + (Q_shared[20] * K_shared[((((int)threadIdx.x) * 24) + 20)]));
    T_batch_matmul_NT[0] = (T_batch_matmul_NT[0] + (Q_shared[21] * K_shared[((((int)threadIdx.x) * 24) + 21)]));
    T_batch_matmul_NT[0] = (T_batch_matmul_NT[0] + (Q_shared[22] * K_shared[((((int)threadIdx.x) * 24) + 22)]));
    T_batch_matmul_NT[0] = (T_batch_matmul_NT[0] + (Q_shared[23] * K_shared[((((int)threadIdx.x) * 24) + 23)]));
  }
  T_divide[((((int)blockIdx.x) * 5) + ((int)threadIdx.x))] = (T_batch_matmul_NT[0] * 1.613743e-02f);
}


