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
extern "C" __global__ void __launch_bounds__(32) candidate0(float* __restrict__ I, float* __restrict__ W, float* __restrict__ T_batch_matmul_NT) {
  float T_batch_matmul_NT_local[5];
  __shared__ float I_shared[240];
  __shared__ float W_shared[1536];
  T_batch_matmul_NT_local[0] = 0.000000e+00f;
  T_batch_matmul_NT_local[1] = 0.000000e+00f;
  T_batch_matmul_NT_local[2] = 0.000000e+00f;
  T_batch_matmul_NT_local[3] = 0.000000e+00f;
  T_batch_matmul_NT_local[4] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 64; ++k_outer_outer) {
    __syncthreads();
    I_shared[(((int)threadIdx.x) * 2)] = I[((((((int)threadIdx.x) / 24) * 3072) + (k_outer_outer * 48)) + ((((int)threadIdx.x) % 24) * 2))];
    I_shared[((((int)threadIdx.x) * 2) + 1)] = I[(((((((int)threadIdx.x) / 24) * 3072) + (k_outer_outer * 48)) + ((((int)threadIdx.x) % 24) * 2)) + 1)];
    I_shared[((((int)threadIdx.x) * 2) + 64)] = I[(((((((int)threadIdx.x) + 32) / 24) * 3072) + (k_outer_outer * 48)) + (((((int)threadIdx.x) * 2) + 16) % 48))];
    I_shared[((((int)threadIdx.x) * 2) + 65)] = I[(((((((int)threadIdx.x) + 32) / 24) * 3072) + (k_outer_outer * 48)) + (((((int)threadIdx.x) * 2) + 17) % 48))];
    I_shared[((((int)threadIdx.x) * 2) + 128)] = I[(((((((int)threadIdx.x) + 64) / 24) * 3072) + (k_outer_outer * 48)) + (((((int)threadIdx.x) * 2) + 32) % 48))];
    I_shared[((((int)threadIdx.x) * 2) + 129)] = I[(((((((int)threadIdx.x) + 64) / 24) * 3072) + (k_outer_outer * 48)) + (((((int)threadIdx.x) * 2) + 33) % 48))];
    if (((int)threadIdx.x) < 24) {
      I_shared[((((int)threadIdx.x) * 2) + 192)] = I[(((k_outer_outer * 48) + (((int)threadIdx.x) * 2)) + 12288)];
      I_shared[((((int)threadIdx.x) * 2) + 193)] = I[(((k_outer_outer * 48) + (((int)threadIdx.x) * 2)) + 12289)];
    }
    *(float4*)(W_shared + (((int)threadIdx.x) * 4)) = *(float4*)(W + ((((((int)blockIdx.x) * 98304) + ((((int)threadIdx.x) / 12) * 3072)) + (k_outer_outer * 48)) + ((((int)threadIdx.x) % 12) * 4)));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 128)) = *(float4*)(W + ((((((int)blockIdx.x) * 98304) + ((((((int)threadIdx.x) * 4) + 128) / 48) * 3072)) + (k_outer_outer * 48)) + (((((int)threadIdx.x) * 4) + 32) % 48)));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 256)) = *(float4*)(W + ((((((int)blockIdx.x) * 98304) + ((((((int)threadIdx.x) * 4) + 256) / 48) * 3072)) + (k_outer_outer * 48)) + (((((int)threadIdx.x) * 4) + 16) % 48)));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 384)) = *(float4*)(W + (((((((int)blockIdx.x) * 98304) + ((((int)threadIdx.x) / 12) * 3072)) + (k_outer_outer * 48)) + ((((int)threadIdx.x) % 12) * 4)) + 24576));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 512)) = *(float4*)(W + ((((((int)blockIdx.x) * 98304) + ((((((int)threadIdx.x) * 4) + 512) / 48) * 3072)) + (k_outer_outer * 48)) + (((((int)threadIdx.x) * 4) + 32) % 48)));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 640)) = *(float4*)(W + ((((((int)blockIdx.x) * 98304) + ((((((int)threadIdx.x) * 4) + 640) / 48) * 3072)) + (k_outer_outer * 48)) + (((((int)threadIdx.x) * 4) + 16) % 48)));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 768)) = *(float4*)(W + (((((((int)blockIdx.x) * 98304) + ((((int)threadIdx.x) / 12) * 3072)) + (k_outer_outer * 48)) + ((((int)threadIdx.x) % 12) * 4)) + 49152));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 896)) = *(float4*)(W + ((((((int)blockIdx.x) * 98304) + ((((((int)threadIdx.x) * 4) + 896) / 48) * 3072)) + (k_outer_outer * 48)) + (((((int)threadIdx.x) * 4) + 32) % 48)));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 1024)) = *(float4*)(W + ((((((int)blockIdx.x) * 98304) + ((((((int)threadIdx.x) * 4) + 1024) / 48) * 3072)) + (k_outer_outer * 48)) + (((((int)threadIdx.x) * 4) + 16) % 48)));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 1152)) = *(float4*)(W + (((((((int)blockIdx.x) * 98304) + ((((int)threadIdx.x) / 12) * 3072)) + (k_outer_outer * 48)) + ((((int)threadIdx.x) % 12) * 4)) + 73728));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 1280)) = *(float4*)(W + ((((((int)blockIdx.x) * 98304) + ((((((int)threadIdx.x) * 4) + 1280) / 48) * 3072)) + (k_outer_outer * 48)) + (((((int)threadIdx.x) * 4) + 32) % 48)));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 1408)) = *(float4*)(W + ((((((int)blockIdx.x) * 98304) + ((((((int)threadIdx.x) * 4) + 1408) / 48) * 3072)) + (k_outer_outer * 48)) + (((((int)threadIdx.x) * 4) + 16) % 48)));
    __syncthreads();
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[0] * W_shared[(((int)threadIdx.x) * 48)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[48] * W_shared[(((int)threadIdx.x) * 48)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[96] * W_shared[(((int)threadIdx.x) * 48)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[144] * W_shared[(((int)threadIdx.x) * 48)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[192] * W_shared[(((int)threadIdx.x) * 48)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[1] * W_shared[((((int)threadIdx.x) * 48) + 1)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[49] * W_shared[((((int)threadIdx.x) * 48) + 1)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[97] * W_shared[((((int)threadIdx.x) * 48) + 1)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[145] * W_shared[((((int)threadIdx.x) * 48) + 1)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[193] * W_shared[((((int)threadIdx.x) * 48) + 1)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[2] * W_shared[((((int)threadIdx.x) * 48) + 2)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[50] * W_shared[((((int)threadIdx.x) * 48) + 2)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[98] * W_shared[((((int)threadIdx.x) * 48) + 2)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[146] * W_shared[((((int)threadIdx.x) * 48) + 2)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[194] * W_shared[((((int)threadIdx.x) * 48) + 2)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[3] * W_shared[((((int)threadIdx.x) * 48) + 3)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[51] * W_shared[((((int)threadIdx.x) * 48) + 3)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[99] * W_shared[((((int)threadIdx.x) * 48) + 3)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[147] * W_shared[((((int)threadIdx.x) * 48) + 3)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[195] * W_shared[((((int)threadIdx.x) * 48) + 3)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[4] * W_shared[((((int)threadIdx.x) * 48) + 4)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[52] * W_shared[((((int)threadIdx.x) * 48) + 4)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[100] * W_shared[((((int)threadIdx.x) * 48) + 4)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[148] * W_shared[((((int)threadIdx.x) * 48) + 4)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[196] * W_shared[((((int)threadIdx.x) * 48) + 4)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[5] * W_shared[((((int)threadIdx.x) * 48) + 5)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[53] * W_shared[((((int)threadIdx.x) * 48) + 5)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[101] * W_shared[((((int)threadIdx.x) * 48) + 5)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[149] * W_shared[((((int)threadIdx.x) * 48) + 5)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[197] * W_shared[((((int)threadIdx.x) * 48) + 5)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[6] * W_shared[((((int)threadIdx.x) * 48) + 6)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[54] * W_shared[((((int)threadIdx.x) * 48) + 6)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[102] * W_shared[((((int)threadIdx.x) * 48) + 6)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[150] * W_shared[((((int)threadIdx.x) * 48) + 6)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[198] * W_shared[((((int)threadIdx.x) * 48) + 6)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[7] * W_shared[((((int)threadIdx.x) * 48) + 7)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[55] * W_shared[((((int)threadIdx.x) * 48) + 7)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[103] * W_shared[((((int)threadIdx.x) * 48) + 7)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[151] * W_shared[((((int)threadIdx.x) * 48) + 7)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[199] * W_shared[((((int)threadIdx.x) * 48) + 7)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[8] * W_shared[((((int)threadIdx.x) * 48) + 8)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[56] * W_shared[((((int)threadIdx.x) * 48) + 8)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[104] * W_shared[((((int)threadIdx.x) * 48) + 8)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[152] * W_shared[((((int)threadIdx.x) * 48) + 8)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[200] * W_shared[((((int)threadIdx.x) * 48) + 8)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[9] * W_shared[((((int)threadIdx.x) * 48) + 9)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[57] * W_shared[((((int)threadIdx.x) * 48) + 9)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[105] * W_shared[((((int)threadIdx.x) * 48) + 9)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[153] * W_shared[((((int)threadIdx.x) * 48) + 9)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[201] * W_shared[((((int)threadIdx.x) * 48) + 9)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[10] * W_shared[((((int)threadIdx.x) * 48) + 10)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[58] * W_shared[((((int)threadIdx.x) * 48) + 10)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[106] * W_shared[((((int)threadIdx.x) * 48) + 10)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[154] * W_shared[((((int)threadIdx.x) * 48) + 10)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[202] * W_shared[((((int)threadIdx.x) * 48) + 10)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[11] * W_shared[((((int)threadIdx.x) * 48) + 11)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[59] * W_shared[((((int)threadIdx.x) * 48) + 11)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[107] * W_shared[((((int)threadIdx.x) * 48) + 11)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[155] * W_shared[((((int)threadIdx.x) * 48) + 11)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[203] * W_shared[((((int)threadIdx.x) * 48) + 11)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[12] * W_shared[((((int)threadIdx.x) * 48) + 12)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[60] * W_shared[((((int)threadIdx.x) * 48) + 12)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[108] * W_shared[((((int)threadIdx.x) * 48) + 12)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[156] * W_shared[((((int)threadIdx.x) * 48) + 12)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[204] * W_shared[((((int)threadIdx.x) * 48) + 12)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[13] * W_shared[((((int)threadIdx.x) * 48) + 13)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[61] * W_shared[((((int)threadIdx.x) * 48) + 13)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[109] * W_shared[((((int)threadIdx.x) * 48) + 13)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[157] * W_shared[((((int)threadIdx.x) * 48) + 13)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[205] * W_shared[((((int)threadIdx.x) * 48) + 13)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[14] * W_shared[((((int)threadIdx.x) * 48) + 14)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[62] * W_shared[((((int)threadIdx.x) * 48) + 14)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[110] * W_shared[((((int)threadIdx.x) * 48) + 14)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[158] * W_shared[((((int)threadIdx.x) * 48) + 14)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[206] * W_shared[((((int)threadIdx.x) * 48) + 14)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[15] * W_shared[((((int)threadIdx.x) * 48) + 15)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[63] * W_shared[((((int)threadIdx.x) * 48) + 15)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[111] * W_shared[((((int)threadIdx.x) * 48) + 15)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[159] * W_shared[((((int)threadIdx.x) * 48) + 15)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[207] * W_shared[((((int)threadIdx.x) * 48) + 15)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[16] * W_shared[((((int)threadIdx.x) * 48) + 16)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[64] * W_shared[((((int)threadIdx.x) * 48) + 16)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[112] * W_shared[((((int)threadIdx.x) * 48) + 16)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[160] * W_shared[((((int)threadIdx.x) * 48) + 16)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[208] * W_shared[((((int)threadIdx.x) * 48) + 16)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[17] * W_shared[((((int)threadIdx.x) * 48) + 17)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[65] * W_shared[((((int)threadIdx.x) * 48) + 17)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[113] * W_shared[((((int)threadIdx.x) * 48) + 17)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[161] * W_shared[((((int)threadIdx.x) * 48) + 17)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[209] * W_shared[((((int)threadIdx.x) * 48) + 17)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[18] * W_shared[((((int)threadIdx.x) * 48) + 18)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[66] * W_shared[((((int)threadIdx.x) * 48) + 18)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[114] * W_shared[((((int)threadIdx.x) * 48) + 18)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[162] * W_shared[((((int)threadIdx.x) * 48) + 18)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[210] * W_shared[((((int)threadIdx.x) * 48) + 18)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[19] * W_shared[((((int)threadIdx.x) * 48) + 19)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[67] * W_shared[((((int)threadIdx.x) * 48) + 19)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[115] * W_shared[((((int)threadIdx.x) * 48) + 19)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[163] * W_shared[((((int)threadIdx.x) * 48) + 19)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[211] * W_shared[((((int)threadIdx.x) * 48) + 19)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[20] * W_shared[((((int)threadIdx.x) * 48) + 20)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[68] * W_shared[((((int)threadIdx.x) * 48) + 20)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[116] * W_shared[((((int)threadIdx.x) * 48) + 20)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[164] * W_shared[((((int)threadIdx.x) * 48) + 20)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[212] * W_shared[((((int)threadIdx.x) * 48) + 20)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[21] * W_shared[((((int)threadIdx.x) * 48) + 21)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[69] * W_shared[((((int)threadIdx.x) * 48) + 21)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[117] * W_shared[((((int)threadIdx.x) * 48) + 21)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[165] * W_shared[((((int)threadIdx.x) * 48) + 21)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[213] * W_shared[((((int)threadIdx.x) * 48) + 21)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[22] * W_shared[((((int)threadIdx.x) * 48) + 22)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[70] * W_shared[((((int)threadIdx.x) * 48) + 22)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[118] * W_shared[((((int)threadIdx.x) * 48) + 22)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[166] * W_shared[((((int)threadIdx.x) * 48) + 22)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[214] * W_shared[((((int)threadIdx.x) * 48) + 22)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[23] * W_shared[((((int)threadIdx.x) * 48) + 23)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[71] * W_shared[((((int)threadIdx.x) * 48) + 23)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[119] * W_shared[((((int)threadIdx.x) * 48) + 23)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[167] * W_shared[((((int)threadIdx.x) * 48) + 23)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[215] * W_shared[((((int)threadIdx.x) * 48) + 23)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[24] * W_shared[((((int)threadIdx.x) * 48) + 24)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[72] * W_shared[((((int)threadIdx.x) * 48) + 24)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[120] * W_shared[((((int)threadIdx.x) * 48) + 24)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[168] * W_shared[((((int)threadIdx.x) * 48) + 24)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[216] * W_shared[((((int)threadIdx.x) * 48) + 24)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[25] * W_shared[((((int)threadIdx.x) * 48) + 25)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[73] * W_shared[((((int)threadIdx.x) * 48) + 25)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[121] * W_shared[((((int)threadIdx.x) * 48) + 25)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[169] * W_shared[((((int)threadIdx.x) * 48) + 25)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[217] * W_shared[((((int)threadIdx.x) * 48) + 25)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[26] * W_shared[((((int)threadIdx.x) * 48) + 26)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[74] * W_shared[((((int)threadIdx.x) * 48) + 26)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[122] * W_shared[((((int)threadIdx.x) * 48) + 26)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[170] * W_shared[((((int)threadIdx.x) * 48) + 26)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[218] * W_shared[((((int)threadIdx.x) * 48) + 26)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[27] * W_shared[((((int)threadIdx.x) * 48) + 27)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[75] * W_shared[((((int)threadIdx.x) * 48) + 27)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[123] * W_shared[((((int)threadIdx.x) * 48) + 27)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[171] * W_shared[((((int)threadIdx.x) * 48) + 27)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[219] * W_shared[((((int)threadIdx.x) * 48) + 27)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[28] * W_shared[((((int)threadIdx.x) * 48) + 28)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[76] * W_shared[((((int)threadIdx.x) * 48) + 28)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[124] * W_shared[((((int)threadIdx.x) * 48) + 28)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[172] * W_shared[((((int)threadIdx.x) * 48) + 28)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[220] * W_shared[((((int)threadIdx.x) * 48) + 28)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[29] * W_shared[((((int)threadIdx.x) * 48) + 29)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[77] * W_shared[((((int)threadIdx.x) * 48) + 29)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[125] * W_shared[((((int)threadIdx.x) * 48) + 29)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[173] * W_shared[((((int)threadIdx.x) * 48) + 29)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[221] * W_shared[((((int)threadIdx.x) * 48) + 29)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[30] * W_shared[((((int)threadIdx.x) * 48) + 30)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[78] * W_shared[((((int)threadIdx.x) * 48) + 30)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[126] * W_shared[((((int)threadIdx.x) * 48) + 30)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[174] * W_shared[((((int)threadIdx.x) * 48) + 30)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[222] * W_shared[((((int)threadIdx.x) * 48) + 30)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[31] * W_shared[((((int)threadIdx.x) * 48) + 31)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[79] * W_shared[((((int)threadIdx.x) * 48) + 31)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[127] * W_shared[((((int)threadIdx.x) * 48) + 31)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[175] * W_shared[((((int)threadIdx.x) * 48) + 31)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[223] * W_shared[((((int)threadIdx.x) * 48) + 31)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[32] * W_shared[((((int)threadIdx.x) * 48) + 32)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[80] * W_shared[((((int)threadIdx.x) * 48) + 32)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[128] * W_shared[((((int)threadIdx.x) * 48) + 32)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[176] * W_shared[((((int)threadIdx.x) * 48) + 32)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[224] * W_shared[((((int)threadIdx.x) * 48) + 32)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[33] * W_shared[((((int)threadIdx.x) * 48) + 33)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[81] * W_shared[((((int)threadIdx.x) * 48) + 33)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[129] * W_shared[((((int)threadIdx.x) * 48) + 33)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[177] * W_shared[((((int)threadIdx.x) * 48) + 33)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[225] * W_shared[((((int)threadIdx.x) * 48) + 33)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[34] * W_shared[((((int)threadIdx.x) * 48) + 34)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[82] * W_shared[((((int)threadIdx.x) * 48) + 34)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[130] * W_shared[((((int)threadIdx.x) * 48) + 34)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[178] * W_shared[((((int)threadIdx.x) * 48) + 34)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[226] * W_shared[((((int)threadIdx.x) * 48) + 34)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[35] * W_shared[((((int)threadIdx.x) * 48) + 35)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[83] * W_shared[((((int)threadIdx.x) * 48) + 35)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[131] * W_shared[((((int)threadIdx.x) * 48) + 35)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[179] * W_shared[((((int)threadIdx.x) * 48) + 35)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[227] * W_shared[((((int)threadIdx.x) * 48) + 35)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[36] * W_shared[((((int)threadIdx.x) * 48) + 36)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[84] * W_shared[((((int)threadIdx.x) * 48) + 36)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[132] * W_shared[((((int)threadIdx.x) * 48) + 36)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[180] * W_shared[((((int)threadIdx.x) * 48) + 36)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[228] * W_shared[((((int)threadIdx.x) * 48) + 36)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[37] * W_shared[((((int)threadIdx.x) * 48) + 37)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[85] * W_shared[((((int)threadIdx.x) * 48) + 37)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[133] * W_shared[((((int)threadIdx.x) * 48) + 37)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[181] * W_shared[((((int)threadIdx.x) * 48) + 37)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[229] * W_shared[((((int)threadIdx.x) * 48) + 37)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[38] * W_shared[((((int)threadIdx.x) * 48) + 38)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[86] * W_shared[((((int)threadIdx.x) * 48) + 38)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[134] * W_shared[((((int)threadIdx.x) * 48) + 38)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[182] * W_shared[((((int)threadIdx.x) * 48) + 38)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[230] * W_shared[((((int)threadIdx.x) * 48) + 38)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[39] * W_shared[((((int)threadIdx.x) * 48) + 39)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[87] * W_shared[((((int)threadIdx.x) * 48) + 39)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[135] * W_shared[((((int)threadIdx.x) * 48) + 39)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[183] * W_shared[((((int)threadIdx.x) * 48) + 39)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[231] * W_shared[((((int)threadIdx.x) * 48) + 39)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[40] * W_shared[((((int)threadIdx.x) * 48) + 40)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[88] * W_shared[((((int)threadIdx.x) * 48) + 40)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[136] * W_shared[((((int)threadIdx.x) * 48) + 40)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[184] * W_shared[((((int)threadIdx.x) * 48) + 40)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[232] * W_shared[((((int)threadIdx.x) * 48) + 40)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[41] * W_shared[((((int)threadIdx.x) * 48) + 41)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[89] * W_shared[((((int)threadIdx.x) * 48) + 41)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[137] * W_shared[((((int)threadIdx.x) * 48) + 41)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[185] * W_shared[((((int)threadIdx.x) * 48) + 41)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[233] * W_shared[((((int)threadIdx.x) * 48) + 41)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[42] * W_shared[((((int)threadIdx.x) * 48) + 42)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[90] * W_shared[((((int)threadIdx.x) * 48) + 42)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[138] * W_shared[((((int)threadIdx.x) * 48) + 42)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[186] * W_shared[((((int)threadIdx.x) * 48) + 42)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[234] * W_shared[((((int)threadIdx.x) * 48) + 42)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[43] * W_shared[((((int)threadIdx.x) * 48) + 43)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[91] * W_shared[((((int)threadIdx.x) * 48) + 43)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[139] * W_shared[((((int)threadIdx.x) * 48) + 43)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[187] * W_shared[((((int)threadIdx.x) * 48) + 43)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[235] * W_shared[((((int)threadIdx.x) * 48) + 43)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[44] * W_shared[((((int)threadIdx.x) * 48) + 44)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[92] * W_shared[((((int)threadIdx.x) * 48) + 44)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[140] * W_shared[((((int)threadIdx.x) * 48) + 44)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[188] * W_shared[((((int)threadIdx.x) * 48) + 44)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[236] * W_shared[((((int)threadIdx.x) * 48) + 44)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[45] * W_shared[((((int)threadIdx.x) * 48) + 45)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[93] * W_shared[((((int)threadIdx.x) * 48) + 45)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[141] * W_shared[((((int)threadIdx.x) * 48) + 45)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[189] * W_shared[((((int)threadIdx.x) * 48) + 45)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[237] * W_shared[((((int)threadIdx.x) * 48) + 45)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[46] * W_shared[((((int)threadIdx.x) * 48) + 46)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[94] * W_shared[((((int)threadIdx.x) * 48) + 46)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[142] * W_shared[((((int)threadIdx.x) * 48) + 46)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[190] * W_shared[((((int)threadIdx.x) * 48) + 46)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[238] * W_shared[((((int)threadIdx.x) * 48) + 46)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[47] * W_shared[((((int)threadIdx.x) * 48) + 47)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[95] * W_shared[((((int)threadIdx.x) * 48) + 47)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[143] * W_shared[((((int)threadIdx.x) * 48) + 47)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[191] * W_shared[((((int)threadIdx.x) * 48) + 47)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[239] * W_shared[((((int)threadIdx.x) * 48) + 47)]));
  }
  for (int i_inner = 0; i_inner < 5; ++i_inner) {
    T_batch_matmul_NT[(((i_inner * 768) + (((int)blockIdx.x) * 32)) + ((int)threadIdx.x))] = T_batch_matmul_NT_local[i_inner];
  }
}


