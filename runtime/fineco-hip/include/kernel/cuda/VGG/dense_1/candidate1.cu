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
extern "C" __global__ void __launch_bounds__(128) candidate1(float* __restrict__ data, float* __restrict__ weight, float* __restrict__ compute, float* __restrict__ bias) {
  float T_matmul_NT[1];
  __shared__ float data_shared[32];
  __shared__ float weight_shared[4096];
  T_matmul_NT[0] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 784; ++k_outer_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 32) {
      data_shared[((int)threadIdx.x)] = data[((k_outer_outer * 32) + ((int)threadIdx.x))];
    }
    weight_shared[((int)threadIdx.x)] = weight[((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 5) * 25088)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31))];
    weight_shared[(((int)threadIdx.x) + 128)] = weight[(((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 5) * 25088)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 100352)];
    weight_shared[(((int)threadIdx.x) + 256)] = weight[(((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 5) * 25088)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 200704)];
    weight_shared[(((int)threadIdx.x) + 384)] = weight[(((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 5) * 25088)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 301056)];
    weight_shared[(((int)threadIdx.x) + 512)] = weight[(((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 5) * 25088)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 401408)];
    weight_shared[(((int)threadIdx.x) + 640)] = weight[(((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 5) * 25088)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 501760)];
    weight_shared[(((int)threadIdx.x) + 768)] = weight[(((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 5) * 25088)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 602112)];
    weight_shared[(((int)threadIdx.x) + 896)] = weight[(((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 5) * 25088)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 702464)];
    weight_shared[(((int)threadIdx.x) + 1024)] = weight[(((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 5) * 25088)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 802816)];
    weight_shared[(((int)threadIdx.x) + 1152)] = weight[(((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 5) * 25088)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 903168)];
    weight_shared[(((int)threadIdx.x) + 1280)] = weight[(((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 5) * 25088)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 1003520)];
    weight_shared[(((int)threadIdx.x) + 1408)] = weight[(((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 5) * 25088)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 1103872)];
    weight_shared[(((int)threadIdx.x) + 1536)] = weight[(((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 5) * 25088)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 1204224)];
    weight_shared[(((int)threadIdx.x) + 1664)] = weight[(((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 5) * 25088)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 1304576)];
    weight_shared[(((int)threadIdx.x) + 1792)] = weight[(((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 5) * 25088)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 1404928)];
    weight_shared[(((int)threadIdx.x) + 1920)] = weight[(((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 5) * 25088)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 1505280)];
    weight_shared[(((int)threadIdx.x) + 2048)] = weight[(((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 5) * 25088)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 1605632)];
    weight_shared[(((int)threadIdx.x) + 2176)] = weight[(((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 5) * 25088)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 1705984)];
    weight_shared[(((int)threadIdx.x) + 2304)] = weight[(((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 5) * 25088)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 1806336)];
    weight_shared[(((int)threadIdx.x) + 2432)] = weight[(((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 5) * 25088)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 1906688)];
    weight_shared[(((int)threadIdx.x) + 2560)] = weight[(((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 5) * 25088)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 2007040)];
    weight_shared[(((int)threadIdx.x) + 2688)] = weight[(((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 5) * 25088)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 2107392)];
    weight_shared[(((int)threadIdx.x) + 2816)] = weight[(((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 5) * 25088)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 2207744)];
    weight_shared[(((int)threadIdx.x) + 2944)] = weight[(((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 5) * 25088)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 2308096)];
    weight_shared[(((int)threadIdx.x) + 3072)] = weight[(((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 5) * 25088)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 2408448)];
    weight_shared[(((int)threadIdx.x) + 3200)] = weight[(((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 5) * 25088)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 2508800)];
    weight_shared[(((int)threadIdx.x) + 3328)] = weight[(((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 5) * 25088)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 2609152)];
    weight_shared[(((int)threadIdx.x) + 3456)] = weight[(((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 5) * 25088)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 2709504)];
    weight_shared[(((int)threadIdx.x) + 3584)] = weight[(((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 5) * 25088)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 2809856)];
    weight_shared[(((int)threadIdx.x) + 3712)] = weight[(((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 5) * 25088)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 2910208)];
    weight_shared[(((int)threadIdx.x) + 3840)] = weight[(((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 5) * 25088)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 3010560)];
    weight_shared[(((int)threadIdx.x) + 3968)] = weight[(((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 5) * 25088)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 3110912)];
    __syncthreads();
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[0] * weight_shared[(((int)threadIdx.x) * 32)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[1] * weight_shared[((((int)threadIdx.x) * 32) + 1)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[2] * weight_shared[((((int)threadIdx.x) * 32) + 2)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[3] * weight_shared[((((int)threadIdx.x) * 32) + 3)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[4] * weight_shared[((((int)threadIdx.x) * 32) + 4)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[5] * weight_shared[((((int)threadIdx.x) * 32) + 5)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[6] * weight_shared[((((int)threadIdx.x) * 32) + 6)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[7] * weight_shared[((((int)threadIdx.x) * 32) + 7)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[8] * weight_shared[((((int)threadIdx.x) * 32) + 8)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[9] * weight_shared[((((int)threadIdx.x) * 32) + 9)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[10] * weight_shared[((((int)threadIdx.x) * 32) + 10)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[11] * weight_shared[((((int)threadIdx.x) * 32) + 11)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[12] * weight_shared[((((int)threadIdx.x) * 32) + 12)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[13] * weight_shared[((((int)threadIdx.x) * 32) + 13)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[14] * weight_shared[((((int)threadIdx.x) * 32) + 14)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[15] * weight_shared[((((int)threadIdx.x) * 32) + 15)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[16] * weight_shared[((((int)threadIdx.x) * 32) + 16)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[17] * weight_shared[((((int)threadIdx.x) * 32) + 17)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[18] * weight_shared[((((int)threadIdx.x) * 32) + 18)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[19] * weight_shared[((((int)threadIdx.x) * 32) + 19)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[20] * weight_shared[((((int)threadIdx.x) * 32) + 20)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[21] * weight_shared[((((int)threadIdx.x) * 32) + 21)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[22] * weight_shared[((((int)threadIdx.x) * 32) + 22)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[23] * weight_shared[((((int)threadIdx.x) * 32) + 23)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[24] * weight_shared[((((int)threadIdx.x) * 32) + 24)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[25] * weight_shared[((((int)threadIdx.x) * 32) + 25)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[26] * weight_shared[((((int)threadIdx.x) * 32) + 26)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[27] * weight_shared[((((int)threadIdx.x) * 32) + 27)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[28] * weight_shared[((((int)threadIdx.x) * 32) + 28)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[29] * weight_shared[((((int)threadIdx.x) * 32) + 29)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[30] * weight_shared[((((int)threadIdx.x) * 32) + 30)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[31] * weight_shared[((((int)threadIdx.x) * 32) + 31)]));
  }
  compute[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = (T_matmul_NT[0] + bias[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))]);
}


