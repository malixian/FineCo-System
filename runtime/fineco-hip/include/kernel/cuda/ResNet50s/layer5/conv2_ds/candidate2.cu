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
extern "C" __global__ void __launch_bounds__(128) candidate2(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[7];
  __shared__ float pad_temp_shared[90];
  __shared__ float kernel_shared[2304];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 256; ++rc_outer_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 90) {
      pad_temp_shared[((int)threadIdx.x)] = (((1 <= (((((int)blockIdx.x) % 7) * 2) + ((((int)threadIdx.x) % 45) / 15))) && (1 <= (((int)threadIdx.x) % 15))) ? data[((((((rc_outer_outer * 392) + ((((int)threadIdx.x) / 45) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((((int)threadIdx.x) % 45) / 15) * 14)) + (((int)threadIdx.x) % 15)) - 15)] : 0.000000e+00f);
    }
    kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) / 7) * 589824) + ((((int)threadIdx.x) / 18) * 4608)) + (rc_outer_outer * 18)) + (((int)threadIdx.x) % 18))];
    kernel_shared[(((int)threadIdx.x) + 128)] = kernel[(((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 128) / 18) * 4608)) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 2) % 18))];
    kernel_shared[(((int)threadIdx.x) + 256)] = kernel[(((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 256) / 18) * 4608)) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 4) % 18))];
    kernel_shared[(((int)threadIdx.x) + 384)] = kernel[(((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 384) / 18) * 4608)) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 6) % 18))];
    kernel_shared[(((int)threadIdx.x) + 512)] = kernel[(((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 512) / 18) * 4608)) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 8) % 18))];
    kernel_shared[(((int)threadIdx.x) + 640)] = kernel[(((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 640) / 18) * 4608)) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 10) % 18))];
    kernel_shared[(((int)threadIdx.x) + 768)] = kernel[(((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 768) / 18) * 4608)) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 12) % 18))];
    kernel_shared[(((int)threadIdx.x) + 896)] = kernel[(((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 896) / 18) * 4608)) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 14) % 18))];
    kernel_shared[(((int)threadIdx.x) + 1024)] = kernel[(((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 1024) / 18) * 4608)) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 16) % 18))];
    kernel_shared[(((int)threadIdx.x) + 1152)] = kernel[((((((((int)blockIdx.x) / 7) * 589824) + ((((int)threadIdx.x) / 18) * 4608)) + (rc_outer_outer * 18)) + (((int)threadIdx.x) % 18)) + 294912)];
    kernel_shared[(((int)threadIdx.x) + 1280)] = kernel[(((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 1280) / 18) * 4608)) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 2) % 18))];
    kernel_shared[(((int)threadIdx.x) + 1408)] = kernel[(((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 1408) / 18) * 4608)) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 4) % 18))];
    kernel_shared[(((int)threadIdx.x) + 1536)] = kernel[(((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 1536) / 18) * 4608)) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 6) % 18))];
    kernel_shared[(((int)threadIdx.x) + 1664)] = kernel[(((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 1664) / 18) * 4608)) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 8) % 18))];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[(((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 1792) / 18) * 4608)) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 10) % 18))];
    kernel_shared[(((int)threadIdx.x) + 1920)] = kernel[(((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 1920) / 18) * 4608)) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 12) % 18))];
    kernel_shared[(((int)threadIdx.x) + 2048)] = kernel[(((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 2048) / 18) * 4608)) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 14) % 18))];
    kernel_shared[(((int)threadIdx.x) + 2176)] = kernel[(((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 2176) / 18) * 4608)) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 16) % 18))];
    __syncthreads();
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[0] * kernel_shared[(((int)threadIdx.x) * 18)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[2] * kernel_shared[(((int)threadIdx.x) * 18)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[4] * kernel_shared[(((int)threadIdx.x) * 18)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[6] * kernel_shared[(((int)threadIdx.x) * 18)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[8] * kernel_shared[(((int)threadIdx.x) * 18)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[10] * kernel_shared[(((int)threadIdx.x) * 18)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[12] * kernel_shared[(((int)threadIdx.x) * 18)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[1] * kernel_shared[((((int)threadIdx.x) * 18) + 1)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[3] * kernel_shared[((((int)threadIdx.x) * 18) + 1)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[5] * kernel_shared[((((int)threadIdx.x) * 18) + 1)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[7] * kernel_shared[((((int)threadIdx.x) * 18) + 1)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.x) * 18) + 1)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.x) * 18) + 1)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.x) * 18) + 1)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx.x) * 18) + 2)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx.x) * 18) + 2)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx.x) * 18) + 2)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx.x) * 18) + 2)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[10] * kernel_shared[((((int)threadIdx.x) * 18) + 2)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[12] * kernel_shared[((((int)threadIdx.x) * 18) + 2)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[14] * kernel_shared[((((int)threadIdx.x) * 18) + 2)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.x) * 18) + 3)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.x) * 18) + 3)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[19] * kernel_shared[((((int)threadIdx.x) * 18) + 3)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[21] * kernel_shared[((((int)threadIdx.x) * 18) + 3)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[23] * kernel_shared[((((int)threadIdx.x) * 18) + 3)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[25] * kernel_shared[((((int)threadIdx.x) * 18) + 3)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[27] * kernel_shared[((((int)threadIdx.x) * 18) + 3)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[16] * kernel_shared[((((int)threadIdx.x) * 18) + 4)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[18] * kernel_shared[((((int)threadIdx.x) * 18) + 4)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[20] * kernel_shared[((((int)threadIdx.x) * 18) + 4)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[22] * kernel_shared[((((int)threadIdx.x) * 18) + 4)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[24] * kernel_shared[((((int)threadIdx.x) * 18) + 4)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[26] * kernel_shared[((((int)threadIdx.x) * 18) + 4)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[28] * kernel_shared[((((int)threadIdx.x) * 18) + 4)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.x) * 18) + 5)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[19] * kernel_shared[((((int)threadIdx.x) * 18) + 5)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[21] * kernel_shared[((((int)threadIdx.x) * 18) + 5)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[23] * kernel_shared[((((int)threadIdx.x) * 18) + 5)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[25] * kernel_shared[((((int)threadIdx.x) * 18) + 5)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[27] * kernel_shared[((((int)threadIdx.x) * 18) + 5)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[29] * kernel_shared[((((int)threadIdx.x) * 18) + 5)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[30] * kernel_shared[((((int)threadIdx.x) * 18) + 6)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[32] * kernel_shared[((((int)threadIdx.x) * 18) + 6)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[34] * kernel_shared[((((int)threadIdx.x) * 18) + 6)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[36] * kernel_shared[((((int)threadIdx.x) * 18) + 6)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[38] * kernel_shared[((((int)threadIdx.x) * 18) + 6)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[40] * kernel_shared[((((int)threadIdx.x) * 18) + 6)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[42] * kernel_shared[((((int)threadIdx.x) * 18) + 6)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[31] * kernel_shared[((((int)threadIdx.x) * 18) + 7)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[33] * kernel_shared[((((int)threadIdx.x) * 18) + 7)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[35] * kernel_shared[((((int)threadIdx.x) * 18) + 7)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[37] * kernel_shared[((((int)threadIdx.x) * 18) + 7)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[39] * kernel_shared[((((int)threadIdx.x) * 18) + 7)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[41] * kernel_shared[((((int)threadIdx.x) * 18) + 7)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[43] * kernel_shared[((((int)threadIdx.x) * 18) + 7)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[32] * kernel_shared[((((int)threadIdx.x) * 18) + 8)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[34] * kernel_shared[((((int)threadIdx.x) * 18) + 8)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[36] * kernel_shared[((((int)threadIdx.x) * 18) + 8)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[38] * kernel_shared[((((int)threadIdx.x) * 18) + 8)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[40] * kernel_shared[((((int)threadIdx.x) * 18) + 8)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[42] * kernel_shared[((((int)threadIdx.x) * 18) + 8)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[44] * kernel_shared[((((int)threadIdx.x) * 18) + 8)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[45] * kernel_shared[((((int)threadIdx.x) * 18) + 9)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[47] * kernel_shared[((((int)threadIdx.x) * 18) + 9)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[49] * kernel_shared[((((int)threadIdx.x) * 18) + 9)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[51] * kernel_shared[((((int)threadIdx.x) * 18) + 9)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[53] * kernel_shared[((((int)threadIdx.x) * 18) + 9)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[55] * kernel_shared[((((int)threadIdx.x) * 18) + 9)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[57] * kernel_shared[((((int)threadIdx.x) * 18) + 9)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[46] * kernel_shared[((((int)threadIdx.x) * 18) + 10)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[48] * kernel_shared[((((int)threadIdx.x) * 18) + 10)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[50] * kernel_shared[((((int)threadIdx.x) * 18) + 10)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[52] * kernel_shared[((((int)threadIdx.x) * 18) + 10)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[54] * kernel_shared[((((int)threadIdx.x) * 18) + 10)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[56] * kernel_shared[((((int)threadIdx.x) * 18) + 10)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[58] * kernel_shared[((((int)threadIdx.x) * 18) + 10)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[47] * kernel_shared[((((int)threadIdx.x) * 18) + 11)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[49] * kernel_shared[((((int)threadIdx.x) * 18) + 11)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[51] * kernel_shared[((((int)threadIdx.x) * 18) + 11)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[53] * kernel_shared[((((int)threadIdx.x) * 18) + 11)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[55] * kernel_shared[((((int)threadIdx.x) * 18) + 11)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[57] * kernel_shared[((((int)threadIdx.x) * 18) + 11)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[59] * kernel_shared[((((int)threadIdx.x) * 18) + 11)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[60] * kernel_shared[((((int)threadIdx.x) * 18) + 12)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[62] * kernel_shared[((((int)threadIdx.x) * 18) + 12)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[64] * kernel_shared[((((int)threadIdx.x) * 18) + 12)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[66] * kernel_shared[((((int)threadIdx.x) * 18) + 12)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[68] * kernel_shared[((((int)threadIdx.x) * 18) + 12)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[70] * kernel_shared[((((int)threadIdx.x) * 18) + 12)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[72] * kernel_shared[((((int)threadIdx.x) * 18) + 12)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[61] * kernel_shared[((((int)threadIdx.x) * 18) + 13)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[63] * kernel_shared[((((int)threadIdx.x) * 18) + 13)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[65] * kernel_shared[((((int)threadIdx.x) * 18) + 13)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[67] * kernel_shared[((((int)threadIdx.x) * 18) + 13)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[69] * kernel_shared[((((int)threadIdx.x) * 18) + 13)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[71] * kernel_shared[((((int)threadIdx.x) * 18) + 13)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[73] * kernel_shared[((((int)threadIdx.x) * 18) + 13)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[62] * kernel_shared[((((int)threadIdx.x) * 18) + 14)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[64] * kernel_shared[((((int)threadIdx.x) * 18) + 14)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[66] * kernel_shared[((((int)threadIdx.x) * 18) + 14)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[68] * kernel_shared[((((int)threadIdx.x) * 18) + 14)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[70] * kernel_shared[((((int)threadIdx.x) * 18) + 14)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[72] * kernel_shared[((((int)threadIdx.x) * 18) + 14)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[74] * kernel_shared[((((int)threadIdx.x) * 18) + 14)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[75] * kernel_shared[((((int)threadIdx.x) * 18) + 15)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[77] * kernel_shared[((((int)threadIdx.x) * 18) + 15)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[79] * kernel_shared[((((int)threadIdx.x) * 18) + 15)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[81] * kernel_shared[((((int)threadIdx.x) * 18) + 15)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[83] * kernel_shared[((((int)threadIdx.x) * 18) + 15)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[85] * kernel_shared[((((int)threadIdx.x) * 18) + 15)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[87] * kernel_shared[((((int)threadIdx.x) * 18) + 15)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[76] * kernel_shared[((((int)threadIdx.x) * 18) + 16)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[78] * kernel_shared[((((int)threadIdx.x) * 18) + 16)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[80] * kernel_shared[((((int)threadIdx.x) * 18) + 16)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[82] * kernel_shared[((((int)threadIdx.x) * 18) + 16)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[84] * kernel_shared[((((int)threadIdx.x) * 18) + 16)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[86] * kernel_shared[((((int)threadIdx.x) * 18) + 16)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[88] * kernel_shared[((((int)threadIdx.x) * 18) + 16)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[77] * kernel_shared[((((int)threadIdx.x) * 18) + 17)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[79] * kernel_shared[((((int)threadIdx.x) * 18) + 17)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[81] * kernel_shared[((((int)threadIdx.x) * 18) + 17)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[83] * kernel_shared[((((int)threadIdx.x) * 18) + 17)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[85] * kernel_shared[((((int)threadIdx.x) * 18) + 17)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[87] * kernel_shared[((((int)threadIdx.x) * 18) + 17)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[89] * kernel_shared[((((int)threadIdx.x) * 18) + 17)]));
  }
  conv2d_nchw[((((((int)blockIdx.x) / 7) * 6272) + (((int)threadIdx.x) * 49)) + ((((int)blockIdx.x) % 7) * 7))] = conv2d_nchw_local[0];
  conv2d_nchw[(((((((int)blockIdx.x) / 7) * 6272) + (((int)threadIdx.x) * 49)) + ((((int)blockIdx.x) % 7) * 7)) + 1)] = conv2d_nchw_local[1];
  conv2d_nchw[(((((((int)blockIdx.x) / 7) * 6272) + (((int)threadIdx.x) * 49)) + ((((int)blockIdx.x) % 7) * 7)) + 2)] = conv2d_nchw_local[2];
  conv2d_nchw[(((((((int)blockIdx.x) / 7) * 6272) + (((int)threadIdx.x) * 49)) + ((((int)blockIdx.x) % 7) * 7)) + 3)] = conv2d_nchw_local[3];
  conv2d_nchw[(((((((int)blockIdx.x) / 7) * 6272) + (((int)threadIdx.x) * 49)) + ((((int)blockIdx.x) % 7) * 7)) + 4)] = conv2d_nchw_local[4];
  conv2d_nchw[(((((((int)blockIdx.x) / 7) * 6272) + (((int)threadIdx.x) * 49)) + ((((int)blockIdx.x) % 7) * 7)) + 5)] = conv2d_nchw_local[5];
  conv2d_nchw[(((((((int)blockIdx.x) / 7) * 6272) + (((int)threadIdx.x) * 49)) + ((((int)blockIdx.x) % 7) * 7)) + 6)] = conv2d_nchw_local[6];
}


