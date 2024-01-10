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
extern "C" __global__ void __launch_bounds__(224) candidate1(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[56];
  __shared__ float pad_temp_shared[3136];
  __shared__ float kernel_shared[1024];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[8] = 0.000000e+00f;
  conv2d_nchw_local[16] = 0.000000e+00f;
  conv2d_nchw_local[24] = 0.000000e+00f;
  conv2d_nchw_local[32] = 0.000000e+00f;
  conv2d_nchw_local[40] = 0.000000e+00f;
  conv2d_nchw_local[48] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[9] = 0.000000e+00f;
  conv2d_nchw_local[17] = 0.000000e+00f;
  conv2d_nchw_local[25] = 0.000000e+00f;
  conv2d_nchw_local[33] = 0.000000e+00f;
  conv2d_nchw_local[41] = 0.000000e+00f;
  conv2d_nchw_local[49] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[10] = 0.000000e+00f;
  conv2d_nchw_local[18] = 0.000000e+00f;
  conv2d_nchw_local[26] = 0.000000e+00f;
  conv2d_nchw_local[34] = 0.000000e+00f;
  conv2d_nchw_local[42] = 0.000000e+00f;
  conv2d_nchw_local[50] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[11] = 0.000000e+00f;
  conv2d_nchw_local[19] = 0.000000e+00f;
  conv2d_nchw_local[27] = 0.000000e+00f;
  conv2d_nchw_local[35] = 0.000000e+00f;
  conv2d_nchw_local[43] = 0.000000e+00f;
  conv2d_nchw_local[51] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[12] = 0.000000e+00f;
  conv2d_nchw_local[20] = 0.000000e+00f;
  conv2d_nchw_local[28] = 0.000000e+00f;
  conv2d_nchw_local[36] = 0.000000e+00f;
  conv2d_nchw_local[44] = 0.000000e+00f;
  conv2d_nchw_local[52] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[13] = 0.000000e+00f;
  conv2d_nchw_local[21] = 0.000000e+00f;
  conv2d_nchw_local[29] = 0.000000e+00f;
  conv2d_nchw_local[37] = 0.000000e+00f;
  conv2d_nchw_local[45] = 0.000000e+00f;
  conv2d_nchw_local[53] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  conv2d_nchw_local[14] = 0.000000e+00f;
  conv2d_nchw_local[22] = 0.000000e+00f;
  conv2d_nchw_local[30] = 0.000000e+00f;
  conv2d_nchw_local[38] = 0.000000e+00f;
  conv2d_nchw_local[46] = 0.000000e+00f;
  conv2d_nchw_local[54] = 0.000000e+00f;
  conv2d_nchw_local[7] = 0.000000e+00f;
  conv2d_nchw_local[15] = 0.000000e+00f;
  conv2d_nchw_local[23] = 0.000000e+00f;
  conv2d_nchw_local[31] = 0.000000e+00f;
  conv2d_nchw_local[39] = 0.000000e+00f;
  conv2d_nchw_local[47] = 0.000000e+00f;
  conv2d_nchw_local[55] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 16; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = data[((rc_outer_outer * 3136) + ((int)threadIdx.x))];
    pad_temp_shared[(((int)threadIdx.x) + 224)] = data[(((rc_outer_outer * 3136) + ((int)threadIdx.x)) + 224)];
    pad_temp_shared[(((int)threadIdx.x) + 448)] = data[(((rc_outer_outer * 3136) + ((int)threadIdx.x)) + 448)];
    pad_temp_shared[(((int)threadIdx.x) + 672)] = data[(((rc_outer_outer * 3136) + ((int)threadIdx.x)) + 672)];
    pad_temp_shared[(((int)threadIdx.x) + 896)] = data[(((rc_outer_outer * 3136) + ((int)threadIdx.x)) + 896)];
    pad_temp_shared[(((int)threadIdx.x) + 1120)] = data[(((rc_outer_outer * 3136) + ((int)threadIdx.x)) + 1120)];
    pad_temp_shared[(((int)threadIdx.x) + 1344)] = data[(((rc_outer_outer * 3136) + ((int)threadIdx.x)) + 1344)];
    pad_temp_shared[(((int)threadIdx.x) + 1568)] = data[(((rc_outer_outer * 3136) + ((int)threadIdx.x)) + 1568)];
    pad_temp_shared[(((int)threadIdx.x) + 1792)] = data[(((rc_outer_outer * 3136) + ((int)threadIdx.x)) + 1792)];
    pad_temp_shared[(((int)threadIdx.x) + 2016)] = data[(((rc_outer_outer * 3136) + ((int)threadIdx.x)) + 2016)];
    pad_temp_shared[(((int)threadIdx.x) + 2240)] = data[(((rc_outer_outer * 3136) + ((int)threadIdx.x)) + 2240)];
    pad_temp_shared[(((int)threadIdx.x) + 2464)] = data[(((rc_outer_outer * 3136) + ((int)threadIdx.x)) + 2464)];
    pad_temp_shared[(((int)threadIdx.x) + 2688)] = data[(((rc_outer_outer * 3136) + ((int)threadIdx.x)) + 2688)];
    pad_temp_shared[(((int)threadIdx.x) + 2912)] = data[(((rc_outer_outer * 3136) + ((int)threadIdx.x)) + 2912)];
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)blockIdx.x) * 16384) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15))];
    kernel_shared[(((int)threadIdx.x) + 224)] = kernel[(((((((int)blockIdx.x) * 16384) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 3584)];
    kernel_shared[(((int)threadIdx.x) + 448)] = kernel[(((((((int)blockIdx.x) * 16384) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 7168)];
    kernel_shared[(((int)threadIdx.x) + 672)] = kernel[(((((((int)blockIdx.x) * 16384) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 10752)];
    if (((int)threadIdx.x) < 128) {
      kernel_shared[(((int)threadIdx.x) + 896)] = kernel[(((((((int)blockIdx.x) * 16384) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 14336)];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 8; ++rc_outer_inner) {
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((rc_outer_inner * 392) + (((int)threadIdx.x) % 28))] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2))]));
      conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 28)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2))]));
      conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 56)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2))]));
      conv2d_nchw_local[24] = (conv2d_nchw_local[24] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 84)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2))]));
      conv2d_nchw_local[32] = (conv2d_nchw_local[32] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 112)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2))]));
      conv2d_nchw_local[40] = (conv2d_nchw_local[40] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 140)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2))]));
      conv2d_nchw_local[48] = (conv2d_nchw_local[48] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 168)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2))]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 196)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 1)]));
      conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 224)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 1)]));
      conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 252)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 1)]));
      conv2d_nchw_local[24] = (conv2d_nchw_local[24] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 280)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 1)]));
      conv2d_nchw_local[32] = (conv2d_nchw_local[32] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 308)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 1)]));
      conv2d_nchw_local[40] = (conv2d_nchw_local[40] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 336)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 1)]));
      conv2d_nchw_local[48] = (conv2d_nchw_local[48] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 364)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 1)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((rc_outer_inner * 392) + (((int)threadIdx.x) % 28))] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 16)]));
      conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 16)]));
      conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 56)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 16)]));
      conv2d_nchw_local[25] = (conv2d_nchw_local[25] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 84)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 16)]));
      conv2d_nchw_local[33] = (conv2d_nchw_local[33] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 112)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 16)]));
      conv2d_nchw_local[41] = (conv2d_nchw_local[41] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 140)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 16)]));
      conv2d_nchw_local[49] = (conv2d_nchw_local[49] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 168)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 16)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 196)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 17)]));
      conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 224)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 17)]));
      conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 252)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 17)]));
      conv2d_nchw_local[25] = (conv2d_nchw_local[25] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 280)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 17)]));
      conv2d_nchw_local[33] = (conv2d_nchw_local[33] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 308)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 17)]));
      conv2d_nchw_local[41] = (conv2d_nchw_local[41] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 336)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 17)]));
      conv2d_nchw_local[49] = (conv2d_nchw_local[49] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 364)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 17)]));
      conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((rc_outer_inner * 392) + (((int)threadIdx.x) % 28))] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 32)]));
      conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 32)]));
      conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 56)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 32)]));
      conv2d_nchw_local[26] = (conv2d_nchw_local[26] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 84)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 32)]));
      conv2d_nchw_local[34] = (conv2d_nchw_local[34] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 112)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 32)]));
      conv2d_nchw_local[42] = (conv2d_nchw_local[42] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 140)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 32)]));
      conv2d_nchw_local[50] = (conv2d_nchw_local[50] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 168)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 32)]));
      conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 196)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 33)]));
      conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 224)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 33)]));
      conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 252)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 33)]));
      conv2d_nchw_local[26] = (conv2d_nchw_local[26] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 280)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 33)]));
      conv2d_nchw_local[34] = (conv2d_nchw_local[34] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 308)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 33)]));
      conv2d_nchw_local[42] = (conv2d_nchw_local[42] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 336)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 33)]));
      conv2d_nchw_local[50] = (conv2d_nchw_local[50] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 364)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 33)]));
      conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((rc_outer_inner * 392) + (((int)threadIdx.x) % 28))] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 48)]));
      conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 48)]));
      conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 56)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 48)]));
      conv2d_nchw_local[27] = (conv2d_nchw_local[27] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 84)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 48)]));
      conv2d_nchw_local[35] = (conv2d_nchw_local[35] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 112)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 48)]));
      conv2d_nchw_local[43] = (conv2d_nchw_local[43] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 140)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 48)]));
      conv2d_nchw_local[51] = (conv2d_nchw_local[51] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 168)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 48)]));
      conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 196)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 49)]));
      conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 224)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 49)]));
      conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 252)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 49)]));
      conv2d_nchw_local[27] = (conv2d_nchw_local[27] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 280)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 49)]));
      conv2d_nchw_local[35] = (conv2d_nchw_local[35] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 308)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 49)]));
      conv2d_nchw_local[43] = (conv2d_nchw_local[43] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 336)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 49)]));
      conv2d_nchw_local[51] = (conv2d_nchw_local[51] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 364)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 49)]));
      conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[((rc_outer_inner * 392) + (((int)threadIdx.x) % 28))] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 64)]));
      conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 64)]));
      conv2d_nchw_local[20] = (conv2d_nchw_local[20] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 56)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 64)]));
      conv2d_nchw_local[28] = (conv2d_nchw_local[28] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 84)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 64)]));
      conv2d_nchw_local[36] = (conv2d_nchw_local[36] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 112)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 64)]));
      conv2d_nchw_local[44] = (conv2d_nchw_local[44] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 140)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 64)]));
      conv2d_nchw_local[52] = (conv2d_nchw_local[52] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 168)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 64)]));
      conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 196)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 65)]));
      conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 224)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 65)]));
      conv2d_nchw_local[20] = (conv2d_nchw_local[20] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 252)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 65)]));
      conv2d_nchw_local[28] = (conv2d_nchw_local[28] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 280)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 65)]));
      conv2d_nchw_local[36] = (conv2d_nchw_local[36] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 308)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 65)]));
      conv2d_nchw_local[44] = (conv2d_nchw_local[44] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 336)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 65)]));
      conv2d_nchw_local[52] = (conv2d_nchw_local[52] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 364)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 65)]));
      conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[((rc_outer_inner * 392) + (((int)threadIdx.x) % 28))] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 80)]));
      conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 80)]));
      conv2d_nchw_local[21] = (conv2d_nchw_local[21] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 56)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 80)]));
      conv2d_nchw_local[29] = (conv2d_nchw_local[29] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 84)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 80)]));
      conv2d_nchw_local[37] = (conv2d_nchw_local[37] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 112)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 80)]));
      conv2d_nchw_local[45] = (conv2d_nchw_local[45] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 140)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 80)]));
      conv2d_nchw_local[53] = (conv2d_nchw_local[53] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 168)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 80)]));
      conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 196)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 81)]));
      conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 224)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 81)]));
      conv2d_nchw_local[21] = (conv2d_nchw_local[21] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 252)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 81)]));
      conv2d_nchw_local[29] = (conv2d_nchw_local[29] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 280)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 81)]));
      conv2d_nchw_local[37] = (conv2d_nchw_local[37] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 308)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 81)]));
      conv2d_nchw_local[45] = (conv2d_nchw_local[45] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 336)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 81)]));
      conv2d_nchw_local[53] = (conv2d_nchw_local[53] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 364)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 81)]));
      conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[((rc_outer_inner * 392) + (((int)threadIdx.x) % 28))] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 96)]));
      conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 96)]));
      conv2d_nchw_local[22] = (conv2d_nchw_local[22] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 56)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 96)]));
      conv2d_nchw_local[30] = (conv2d_nchw_local[30] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 84)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 96)]));
      conv2d_nchw_local[38] = (conv2d_nchw_local[38] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 112)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 96)]));
      conv2d_nchw_local[46] = (conv2d_nchw_local[46] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 140)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 96)]));
      conv2d_nchw_local[54] = (conv2d_nchw_local[54] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 168)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 96)]));
      conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 196)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 97)]));
      conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 224)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 97)]));
      conv2d_nchw_local[22] = (conv2d_nchw_local[22] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 252)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 97)]));
      conv2d_nchw_local[30] = (conv2d_nchw_local[30] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 280)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 97)]));
      conv2d_nchw_local[38] = (conv2d_nchw_local[38] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 308)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 97)]));
      conv2d_nchw_local[46] = (conv2d_nchw_local[46] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 336)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 97)]));
      conv2d_nchw_local[54] = (conv2d_nchw_local[54] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 364)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 97)]));
      conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[((rc_outer_inner * 392) + (((int)threadIdx.x) % 28))] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 112)]));
      conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 112)]));
      conv2d_nchw_local[23] = (conv2d_nchw_local[23] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 56)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 112)]));
      conv2d_nchw_local[31] = (conv2d_nchw_local[31] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 84)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 112)]));
      conv2d_nchw_local[39] = (conv2d_nchw_local[39] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 112)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 112)]));
      conv2d_nchw_local[47] = (conv2d_nchw_local[47] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 140)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 112)]));
      conv2d_nchw_local[55] = (conv2d_nchw_local[55] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 168)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 112)]));
      conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 196)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 113)]));
      conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 224)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 113)]));
      conv2d_nchw_local[23] = (conv2d_nchw_local[23] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 252)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 113)]));
      conv2d_nchw_local[31] = (conv2d_nchw_local[31] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 280)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 113)]));
      conv2d_nchw_local[39] = (conv2d_nchw_local[39] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 308)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 113)]));
      conv2d_nchw_local[47] = (conv2d_nchw_local[47] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 336)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 113)]));
      conv2d_nchw_local[55] = (conv2d_nchw_local[55] + (pad_temp_shared[(((rc_outer_inner * 392) + (((int)threadIdx.x) % 28)) + 364)] * kernel_shared[((((((int)threadIdx.x) / 28) * 128) + (rc_outer_inner * 2)) + 113)]));
    }
  }
  for (int ff_inner = 0; ff_inner < 8; ++ff_inner) {
    conv2d_nchw[((((((int)blockIdx.x) * 12544) + ((((int)threadIdx.x) / 28) * 1568)) + (ff_inner * 196)) + (((int)threadIdx.x) % 28))] = conv2d_nchw_local[ff_inner];
    conv2d_nchw[(((((((int)blockIdx.x) * 12544) + ((((int)threadIdx.x) / 28) * 1568)) + (ff_inner * 196)) + (((int)threadIdx.x) % 28)) + 28)] = conv2d_nchw_local[(ff_inner + 8)];
    conv2d_nchw[(((((((int)blockIdx.x) * 12544) + ((((int)threadIdx.x) / 28) * 1568)) + (ff_inner * 196)) + (((int)threadIdx.x) % 28)) + 56)] = conv2d_nchw_local[(ff_inner + 16)];
    conv2d_nchw[(((((((int)blockIdx.x) * 12544) + ((((int)threadIdx.x) / 28) * 1568)) + (ff_inner * 196)) + (((int)threadIdx.x) % 28)) + 84)] = conv2d_nchw_local[(ff_inner + 24)];
    conv2d_nchw[(((((((int)blockIdx.x) * 12544) + ((((int)threadIdx.x) / 28) * 1568)) + (ff_inner * 196)) + (((int)threadIdx.x) % 28)) + 112)] = conv2d_nchw_local[(ff_inner + 32)];
    conv2d_nchw[(((((((int)blockIdx.x) * 12544) + ((((int)threadIdx.x) / 28) * 1568)) + (ff_inner * 196)) + (((int)threadIdx.x) % 28)) + 140)] = conv2d_nchw_local[(ff_inner + 40)];
    conv2d_nchw[(((((((int)blockIdx.x) * 12544) + ((((int)threadIdx.x) / 28) * 1568)) + (ff_inner * 196)) + (((int)threadIdx.x) % 28)) + 168)] = conv2d_nchw_local[(ff_inner + 48)];
  }
}


