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
extern "C" __global__ void __launch_bounds__(112) candidate1(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[56];
  __shared__ float pad_temp_shared[3480];
  __shared__ float kernel_shared[4608];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[14] = 0.000000e+00f;
  conv2d_nchw_local[28] = 0.000000e+00f;
  conv2d_nchw_local[42] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[15] = 0.000000e+00f;
  conv2d_nchw_local[29] = 0.000000e+00f;
  conv2d_nchw_local[43] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[16] = 0.000000e+00f;
  conv2d_nchw_local[30] = 0.000000e+00f;
  conv2d_nchw_local[44] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[17] = 0.000000e+00f;
  conv2d_nchw_local[31] = 0.000000e+00f;
  conv2d_nchw_local[45] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[18] = 0.000000e+00f;
  conv2d_nchw_local[32] = 0.000000e+00f;
  conv2d_nchw_local[46] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[19] = 0.000000e+00f;
  conv2d_nchw_local[33] = 0.000000e+00f;
  conv2d_nchw_local[47] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  conv2d_nchw_local[20] = 0.000000e+00f;
  conv2d_nchw_local[34] = 0.000000e+00f;
  conv2d_nchw_local[48] = 0.000000e+00f;
  conv2d_nchw_local[7] = 0.000000e+00f;
  conv2d_nchw_local[21] = 0.000000e+00f;
  conv2d_nchw_local[35] = 0.000000e+00f;
  conv2d_nchw_local[49] = 0.000000e+00f;
  conv2d_nchw_local[8] = 0.000000e+00f;
  conv2d_nchw_local[22] = 0.000000e+00f;
  conv2d_nchw_local[36] = 0.000000e+00f;
  conv2d_nchw_local[50] = 0.000000e+00f;
  conv2d_nchw_local[9] = 0.000000e+00f;
  conv2d_nchw_local[23] = 0.000000e+00f;
  conv2d_nchw_local[37] = 0.000000e+00f;
  conv2d_nchw_local[51] = 0.000000e+00f;
  conv2d_nchw_local[10] = 0.000000e+00f;
  conv2d_nchw_local[24] = 0.000000e+00f;
  conv2d_nchw_local[38] = 0.000000e+00f;
  conv2d_nchw_local[52] = 0.000000e+00f;
  conv2d_nchw_local[11] = 0.000000e+00f;
  conv2d_nchw_local[25] = 0.000000e+00f;
  conv2d_nchw_local[39] = 0.000000e+00f;
  conv2d_nchw_local[53] = 0.000000e+00f;
  conv2d_nchw_local[12] = 0.000000e+00f;
  conv2d_nchw_local[26] = 0.000000e+00f;
  conv2d_nchw_local[40] = 0.000000e+00f;
  conv2d_nchw_local[54] = 0.000000e+00f;
  conv2d_nchw_local[13] = 0.000000e+00f;
  conv2d_nchw_local[27] = 0.000000e+00f;
  conv2d_nchw_local[41] = 0.000000e+00f;
  conv2d_nchw_local[55] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 8; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = (((1 <= ((((((int)blockIdx.x) & 7) >> 2) * 28) + (((int)threadIdx.x) / 15))) && (1 <= (((((int)blockIdx.x) & 3) * 14) + (((int)threadIdx.x) % 15)))) ? data[((((((rc_outer_outer * 25088) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((int)threadIdx.x) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 112)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 7) % 15))) ? data[((((((rc_outer_outer * 25088) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + (((((int)threadIdx.x) + 112) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 7) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 224)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 14) % 15))) ? data[((((((rc_outer_outer * 25088) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + (((((int)threadIdx.x) + 224) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 14) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 336)] = (((1 <= ((((((int)blockIdx.x) & 7) >> 2) * 28) + (((((int)threadIdx.x) + 336) % 435) / 15))) && (1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 6) % 15)))) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 336) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 336) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 6) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 448)] = (((1 <= ((((((int)blockIdx.x) & 7) >> 2) * 28) + (((((int)threadIdx.x) + 13) % 435) / 15))) && (1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 13) % 15)))) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 448) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 13) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 13) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 560)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 5) % 15))) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 560) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 125) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 5) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 672)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 12) % 15))) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 672) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 237) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 12) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 784)] = (((1 <= ((((((int)blockIdx.x) & 7) >> 2) * 28) + (((((int)threadIdx.x) + 349) % 435) / 15))) && (1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 4) % 15)))) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 784) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 349) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 4) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 896)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 11) % 15))) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 896) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 26) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 11) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1008)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 3) % 15))) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 1008) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 138) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 3) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1120)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 10) % 15))) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 1120) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 250) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 10) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1232)] = (((1 <= ((((((int)blockIdx.x) & 7) >> 2) * 28) + (((((int)threadIdx.x) + 362) % 435) / 15))) && (1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 2) % 15)))) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 1232) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 362) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 2) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1344)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 9) % 15))) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 1344) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 39) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 9) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1456)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 1) % 15))) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 1456) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 151) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 1) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1568)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 8) % 15))) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 1568) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 263) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 8) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1680)] = (((1 <= ((((((int)blockIdx.x) & 7) >> 2) * 28) + (((((int)threadIdx.x) / 15) + 25) % 29))) && (1 <= (((((int)blockIdx.x) & 3) * 14) + (((int)threadIdx.x) % 15)))) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 1680) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) / 15) + 25) % 29) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1792)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 7) % 15))) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 1792) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 52) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 7) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1904)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 14) % 15))) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 1904) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 164) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 14) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 2016)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 6) % 15))) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 2016) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 276) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 6) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 2128)] = (((1 <= ((((((int)blockIdx.x) & 7) >> 2) * 28) + (((((int)threadIdx.x) + 388) % 435) / 15))) && (1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 13) % 15)))) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 2128) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 388) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 13) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 2240)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 5) % 15))) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 2240) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 65) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 5) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 2352)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 12) % 15))) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 2352) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 177) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 12) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 2464)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 4) % 15))) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 2464) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 289) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 4) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 2576)] = (((1 <= ((((((int)blockIdx.x) & 7) >> 2) * 28) + (((((int)threadIdx.x) + 401) % 435) / 15))) && (1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 11) % 15)))) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 2576) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 401) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 11) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 2688)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 3) % 15))) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 2688) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 78) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 3) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 2800)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 10) % 15))) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 2800) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 190) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 10) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 2912)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 2) % 15))) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 2912) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 302) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 2) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 3024)] = (((1 <= ((((((int)blockIdx.x) & 7) >> 2) * 28) + (((((int)threadIdx.x) + 414) % 435) / 15))) && (1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 9) % 15)))) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 3024) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 414) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 9) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 3136)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 1) % 15))) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 3136) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 91) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 1) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 3248)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 8) % 15))) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 3248) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 203) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 8) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 3360)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + (((int)threadIdx.x) % 15))) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 3360) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + (((((int)threadIdx.x) / 15) + 21) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 15)) - 57)] : 0.000000e+00f);
    if (((int)threadIdx.x) < 8) {
      pad_temp_shared[(((int)threadIdx.x) + 3472)] = data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 3472) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 427) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) + 7)) - 57)];
    }
    kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) >> 3) * 36864) + ((((int)threadIdx.x) / 72) * 576)) + (rc_outer_outer * 72)) + (((int)threadIdx.x) % 72))];
    kernel_shared[(((int)threadIdx.x) + 112)] = kernel[(((((((int)blockIdx.x) >> 3) * 36864) + (((((int)threadIdx.x) + 112) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 40) % 72))];
    kernel_shared[(((int)threadIdx.x) + 224)] = kernel[(((((((int)blockIdx.x) >> 3) * 36864) + (((((int)threadIdx.x) + 224) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 8) % 72))];
    kernel_shared[(((int)threadIdx.x) + 336)] = kernel[(((((((int)blockIdx.x) >> 3) * 36864) + (((((int)threadIdx.x) + 336) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 48) % 72))];
    kernel_shared[(((int)threadIdx.x) + 448)] = kernel[(((((((int)blockIdx.x) >> 3) * 36864) + (((((int)threadIdx.x) + 448) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 16) % 72))];
    kernel_shared[(((int)threadIdx.x) + 560)] = kernel[(((((((int)blockIdx.x) >> 3) * 36864) + (((((int)threadIdx.x) + 560) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 56) % 72))];
    kernel_shared[(((int)threadIdx.x) + 672)] = kernel[(((((((int)blockIdx.x) >> 3) * 36864) + (((((int)threadIdx.x) + 672) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 24) % 72))];
    kernel_shared[(((int)threadIdx.x) + 784)] = kernel[(((((((int)blockIdx.x) >> 3) * 36864) + (((((int)threadIdx.x) + 784) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 64) % 72))];
    kernel_shared[(((int)threadIdx.x) + 896)] = kernel[(((((((int)blockIdx.x) >> 3) * 36864) + (((((int)threadIdx.x) + 896) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 32) % 72))];
    kernel_shared[(((int)threadIdx.x) + 1008)] = kernel[((((((((int)blockIdx.x) >> 3) * 36864) + ((((int)threadIdx.x) / 72) * 576)) + (rc_outer_outer * 72)) + (((int)threadIdx.x) % 72)) + 8064)];
    kernel_shared[(((int)threadIdx.x) + 1120)] = kernel[(((((((int)blockIdx.x) >> 3) * 36864) + (((((int)threadIdx.x) + 1120) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 40) % 72))];
    kernel_shared[(((int)threadIdx.x) + 1232)] = kernel[(((((((int)blockIdx.x) >> 3) * 36864) + (((((int)threadIdx.x) + 1232) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 8) % 72))];
    kernel_shared[(((int)threadIdx.x) + 1344)] = kernel[(((((((int)blockIdx.x) >> 3) * 36864) + (((((int)threadIdx.x) + 1344) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 48) % 72))];
    kernel_shared[(((int)threadIdx.x) + 1456)] = kernel[(((((((int)blockIdx.x) >> 3) * 36864) + (((((int)threadIdx.x) + 1456) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 16) % 72))];
    kernel_shared[(((int)threadIdx.x) + 1568)] = kernel[(((((((int)blockIdx.x) >> 3) * 36864) + (((((int)threadIdx.x) + 1568) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 56) % 72))];
    kernel_shared[(((int)threadIdx.x) + 1680)] = kernel[(((((((int)blockIdx.x) >> 3) * 36864) + (((((int)threadIdx.x) + 1680) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 24) % 72))];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[(((((((int)blockIdx.x) >> 3) * 36864) + (((((int)threadIdx.x) + 1792) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 64) % 72))];
    kernel_shared[(((int)threadIdx.x) + 1904)] = kernel[(((((((int)blockIdx.x) >> 3) * 36864) + (((((int)threadIdx.x) + 1904) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 32) % 72))];
    kernel_shared[(((int)threadIdx.x) + 2016)] = kernel[((((((((int)blockIdx.x) >> 3) * 36864) + ((((int)threadIdx.x) / 72) * 576)) + (rc_outer_outer * 72)) + (((int)threadIdx.x) % 72)) + 16128)];
    kernel_shared[(((int)threadIdx.x) + 2128)] = kernel[(((((((int)blockIdx.x) >> 3) * 36864) + (((((int)threadIdx.x) + 2128) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 40) % 72))];
    kernel_shared[(((int)threadIdx.x) + 2240)] = kernel[(((((((int)blockIdx.x) >> 3) * 36864) + (((((int)threadIdx.x) + 2240) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 8) % 72))];
    kernel_shared[(((int)threadIdx.x) + 2352)] = kernel[(((((((int)blockIdx.x) >> 3) * 36864) + (((((int)threadIdx.x) + 2352) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 48) % 72))];
    kernel_shared[(((int)threadIdx.x) + 2464)] = kernel[(((((((int)blockIdx.x) >> 3) * 36864) + (((((int)threadIdx.x) + 2464) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 16) % 72))];
    kernel_shared[(((int)threadIdx.x) + 2576)] = kernel[(((((((int)blockIdx.x) >> 3) * 36864) + (((((int)threadIdx.x) + 2576) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 56) % 72))];
    kernel_shared[(((int)threadIdx.x) + 2688)] = kernel[(((((((int)blockIdx.x) >> 3) * 36864) + (((((int)threadIdx.x) + 2688) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 24) % 72))];
    kernel_shared[(((int)threadIdx.x) + 2800)] = kernel[(((((((int)blockIdx.x) >> 3) * 36864) + (((((int)threadIdx.x) + 2800) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 64) % 72))];
    kernel_shared[(((int)threadIdx.x) + 2912)] = kernel[(((((((int)blockIdx.x) >> 3) * 36864) + (((((int)threadIdx.x) + 2912) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 32) % 72))];
    kernel_shared[(((int)threadIdx.x) + 3024)] = kernel[((((((((int)blockIdx.x) >> 3) * 36864) + ((((int)threadIdx.x) / 72) * 576)) + (rc_outer_outer * 72)) + (((int)threadIdx.x) % 72)) + 24192)];
    kernel_shared[(((int)threadIdx.x) + 3136)] = kernel[(((((((int)blockIdx.x) >> 3) * 36864) + (((((int)threadIdx.x) + 3136) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 40) % 72))];
    kernel_shared[(((int)threadIdx.x) + 3248)] = kernel[(((((((int)blockIdx.x) >> 3) * 36864) + (((((int)threadIdx.x) + 3248) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 8) % 72))];
    kernel_shared[(((int)threadIdx.x) + 3360)] = kernel[(((((((int)blockIdx.x) >> 3) * 36864) + (((((int)threadIdx.x) + 3360) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 48) % 72))];
    kernel_shared[(((int)threadIdx.x) + 3472)] = kernel[(((((((int)blockIdx.x) >> 3) * 36864) + (((((int)threadIdx.x) + 3472) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 16) % 72))];
    kernel_shared[(((int)threadIdx.x) + 3584)] = kernel[(((((((int)blockIdx.x) >> 3) * 36864) + (((((int)threadIdx.x) + 3584) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 56) % 72))];
    kernel_shared[(((int)threadIdx.x) + 3696)] = kernel[(((((((int)blockIdx.x) >> 3) * 36864) + (((((int)threadIdx.x) + 3696) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 24) % 72))];
    kernel_shared[(((int)threadIdx.x) + 3808)] = kernel[(((((((int)blockIdx.x) >> 3) * 36864) + (((((int)threadIdx.x) + 3808) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 64) % 72))];
    kernel_shared[(((int)threadIdx.x) + 3920)] = kernel[(((((((int)blockIdx.x) >> 3) * 36864) + (((((int)threadIdx.x) + 3920) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 32) % 72))];
    kernel_shared[(((int)threadIdx.x) + 4032)] = kernel[((((((((int)blockIdx.x) >> 3) * 36864) + ((((int)threadIdx.x) / 72) * 576)) + (rc_outer_outer * 72)) + (((int)threadIdx.x) % 72)) + 32256)];
    kernel_shared[(((int)threadIdx.x) + 4144)] = kernel[(((((((int)blockIdx.x) >> 3) * 36864) + (((((int)threadIdx.x) + 4144) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 40) % 72))];
    kernel_shared[(((int)threadIdx.x) + 4256)] = kernel[(((((((int)blockIdx.x) >> 3) * 36864) + (((((int)threadIdx.x) + 4256) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 8) % 72))];
    kernel_shared[(((int)threadIdx.x) + 4368)] = kernel[(((((((int)blockIdx.x) >> 3) * 36864) + (((((int)threadIdx.x) + 4368) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 48) % 72))];
    kernel_shared[(((int)threadIdx.x) + 4480)] = kernel[(((((((int)blockIdx.x) >> 3) * 36864) + (((((int)threadIdx.x) + 4480) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 16) % 72))];
    if (((int)threadIdx.x) < 16) {
      kernel_shared[(((int)threadIdx.x) + 4592)] = kernel[(((((((int)blockIdx.x) >> 3) * 36864) + (((((int)threadIdx.x) + 4592) / 72) * 576)) + (rc_outer_outer * 72)) + (((int)threadIdx.x) + 56))];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 4; ++rc_outer_inner) {
      for (int ry_outer_inner = 0; ry_outer_inner < 3; ++ry_outer_inner) {
        for (int ff_c_outer_inner = 0; ff_c_outer_inner < 2; ++ff_c_outer_inner) {
          for (int rc_inner = 0; rc_inner < 2; ++rc_inner) {
            for (int rx_inner = 0; rx_inner < 3; ++rx_inner) {
              conv2d_nchw_local[(ff_c_outer_inner * 7)] = (conv2d_nchw_local[(ff_c_outer_inner * 7)] + (pad_temp_shared[(((((rc_outer_inner * 870) + (rc_inner * 435)) + ((((int)threadIdx.x) % 14) * 30)) + (ry_outer_inner * 15)) + rx_inner)] * kernel_shared[(((((((((int)threadIdx.x) / 14) * 144) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner)]));
              conv2d_nchw_local[((ff_c_outer_inner * 7) + 14)] = (conv2d_nchw_local[((ff_c_outer_inner * 7) + 14)] + (pad_temp_shared[(((((rc_outer_inner * 870) + (rc_inner * 435)) + ((((int)threadIdx.x) % 14) * 30)) + (ry_outer_inner * 15)) + rx_inner)] * kernel_shared[((((((((((int)threadIdx.x) / 14) * 144) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 1152)]));
              conv2d_nchw_local[((ff_c_outer_inner * 7) + 28)] = (conv2d_nchw_local[((ff_c_outer_inner * 7) + 28)] + (pad_temp_shared[(((((rc_outer_inner * 870) + (rc_inner * 435)) + ((((int)threadIdx.x) % 14) * 30)) + (ry_outer_inner * 15)) + rx_inner)] * kernel_shared[((((((((((int)threadIdx.x) / 14) * 144) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 2304)]));
              conv2d_nchw_local[((ff_c_outer_inner * 7) + 42)] = (conv2d_nchw_local[((ff_c_outer_inner * 7) + 42)] + (pad_temp_shared[(((((rc_outer_inner * 870) + (rc_inner * 435)) + ((((int)threadIdx.x) % 14) * 30)) + (ry_outer_inner * 15)) + rx_inner)] * kernel_shared[((((((((((int)threadIdx.x) / 14) * 144) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 3456)]));
              conv2d_nchw_local[((ff_c_outer_inner * 7) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 7) + 1)] + (pad_temp_shared[((((((rc_outer_inner * 870) + (rc_inner * 435)) + ((((int)threadIdx.x) % 14) * 30)) + (ry_outer_inner * 15)) + rx_inner) + 2)] * kernel_shared[(((((((((int)threadIdx.x) / 14) * 144) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner)]));
              conv2d_nchw_local[((ff_c_outer_inner * 7) + 15)] = (conv2d_nchw_local[((ff_c_outer_inner * 7) + 15)] + (pad_temp_shared[((((((rc_outer_inner * 870) + (rc_inner * 435)) + ((((int)threadIdx.x) % 14) * 30)) + (ry_outer_inner * 15)) + rx_inner) + 2)] * kernel_shared[((((((((((int)threadIdx.x) / 14) * 144) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 1152)]));
              conv2d_nchw_local[((ff_c_outer_inner * 7) + 29)] = (conv2d_nchw_local[((ff_c_outer_inner * 7) + 29)] + (pad_temp_shared[((((((rc_outer_inner * 870) + (rc_inner * 435)) + ((((int)threadIdx.x) % 14) * 30)) + (ry_outer_inner * 15)) + rx_inner) + 2)] * kernel_shared[((((((((((int)threadIdx.x) / 14) * 144) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 2304)]));
              conv2d_nchw_local[((ff_c_outer_inner * 7) + 43)] = (conv2d_nchw_local[((ff_c_outer_inner * 7) + 43)] + (pad_temp_shared[((((((rc_outer_inner * 870) + (rc_inner * 435)) + ((((int)threadIdx.x) % 14) * 30)) + (ry_outer_inner * 15)) + rx_inner) + 2)] * kernel_shared[((((((((((int)threadIdx.x) / 14) * 144) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 3456)]));
              conv2d_nchw_local[((ff_c_outer_inner * 7) + 2)] = (conv2d_nchw_local[((ff_c_outer_inner * 7) + 2)] + (pad_temp_shared[((((((rc_outer_inner * 870) + (rc_inner * 435)) + ((((int)threadIdx.x) % 14) * 30)) + (ry_outer_inner * 15)) + rx_inner) + 4)] * kernel_shared[(((((((((int)threadIdx.x) / 14) * 144) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner)]));
              conv2d_nchw_local[((ff_c_outer_inner * 7) + 16)] = (conv2d_nchw_local[((ff_c_outer_inner * 7) + 16)] + (pad_temp_shared[((((((rc_outer_inner * 870) + (rc_inner * 435)) + ((((int)threadIdx.x) % 14) * 30)) + (ry_outer_inner * 15)) + rx_inner) + 4)] * kernel_shared[((((((((((int)threadIdx.x) / 14) * 144) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 1152)]));
              conv2d_nchw_local[((ff_c_outer_inner * 7) + 30)] = (conv2d_nchw_local[((ff_c_outer_inner * 7) + 30)] + (pad_temp_shared[((((((rc_outer_inner * 870) + (rc_inner * 435)) + ((((int)threadIdx.x) % 14) * 30)) + (ry_outer_inner * 15)) + rx_inner) + 4)] * kernel_shared[((((((((((int)threadIdx.x) / 14) * 144) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 2304)]));
              conv2d_nchw_local[((ff_c_outer_inner * 7) + 44)] = (conv2d_nchw_local[((ff_c_outer_inner * 7) + 44)] + (pad_temp_shared[((((((rc_outer_inner * 870) + (rc_inner * 435)) + ((((int)threadIdx.x) % 14) * 30)) + (ry_outer_inner * 15)) + rx_inner) + 4)] * kernel_shared[((((((((((int)threadIdx.x) / 14) * 144) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 3456)]));
              conv2d_nchw_local[((ff_c_outer_inner * 7) + 3)] = (conv2d_nchw_local[((ff_c_outer_inner * 7) + 3)] + (pad_temp_shared[((((((rc_outer_inner * 870) + (rc_inner * 435)) + ((((int)threadIdx.x) % 14) * 30)) + (ry_outer_inner * 15)) + rx_inner) + 6)] * kernel_shared[(((((((((int)threadIdx.x) / 14) * 144) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner)]));
              conv2d_nchw_local[((ff_c_outer_inner * 7) + 17)] = (conv2d_nchw_local[((ff_c_outer_inner * 7) + 17)] + (pad_temp_shared[((((((rc_outer_inner * 870) + (rc_inner * 435)) + ((((int)threadIdx.x) % 14) * 30)) + (ry_outer_inner * 15)) + rx_inner) + 6)] * kernel_shared[((((((((((int)threadIdx.x) / 14) * 144) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 1152)]));
              conv2d_nchw_local[((ff_c_outer_inner * 7) + 31)] = (conv2d_nchw_local[((ff_c_outer_inner * 7) + 31)] + (pad_temp_shared[((((((rc_outer_inner * 870) + (rc_inner * 435)) + ((((int)threadIdx.x) % 14) * 30)) + (ry_outer_inner * 15)) + rx_inner) + 6)] * kernel_shared[((((((((((int)threadIdx.x) / 14) * 144) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 2304)]));
              conv2d_nchw_local[((ff_c_outer_inner * 7) + 45)] = (conv2d_nchw_local[((ff_c_outer_inner * 7) + 45)] + (pad_temp_shared[((((((rc_outer_inner * 870) + (rc_inner * 435)) + ((((int)threadIdx.x) % 14) * 30)) + (ry_outer_inner * 15)) + rx_inner) + 6)] * kernel_shared[((((((((((int)threadIdx.x) / 14) * 144) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 3456)]));
              conv2d_nchw_local[((ff_c_outer_inner * 7) + 4)] = (conv2d_nchw_local[((ff_c_outer_inner * 7) + 4)] + (pad_temp_shared[((((((rc_outer_inner * 870) + (rc_inner * 435)) + ((((int)threadIdx.x) % 14) * 30)) + (ry_outer_inner * 15)) + rx_inner) + 8)] * kernel_shared[(((((((((int)threadIdx.x) / 14) * 144) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner)]));
              conv2d_nchw_local[((ff_c_outer_inner * 7) + 18)] = (conv2d_nchw_local[((ff_c_outer_inner * 7) + 18)] + (pad_temp_shared[((((((rc_outer_inner * 870) + (rc_inner * 435)) + ((((int)threadIdx.x) % 14) * 30)) + (ry_outer_inner * 15)) + rx_inner) + 8)] * kernel_shared[((((((((((int)threadIdx.x) / 14) * 144) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 1152)]));
              conv2d_nchw_local[((ff_c_outer_inner * 7) + 32)] = (conv2d_nchw_local[((ff_c_outer_inner * 7) + 32)] + (pad_temp_shared[((((((rc_outer_inner * 870) + (rc_inner * 435)) + ((((int)threadIdx.x) % 14) * 30)) + (ry_outer_inner * 15)) + rx_inner) + 8)] * kernel_shared[((((((((((int)threadIdx.x) / 14) * 144) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 2304)]));
              conv2d_nchw_local[((ff_c_outer_inner * 7) + 46)] = (conv2d_nchw_local[((ff_c_outer_inner * 7) + 46)] + (pad_temp_shared[((((((rc_outer_inner * 870) + (rc_inner * 435)) + ((((int)threadIdx.x) % 14) * 30)) + (ry_outer_inner * 15)) + rx_inner) + 8)] * kernel_shared[((((((((((int)threadIdx.x) / 14) * 144) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 3456)]));
              conv2d_nchw_local[((ff_c_outer_inner * 7) + 5)] = (conv2d_nchw_local[((ff_c_outer_inner * 7) + 5)] + (pad_temp_shared[((((((rc_outer_inner * 870) + (rc_inner * 435)) + ((((int)threadIdx.x) % 14) * 30)) + (ry_outer_inner * 15)) + rx_inner) + 10)] * kernel_shared[(((((((((int)threadIdx.x) / 14) * 144) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner)]));
              conv2d_nchw_local[((ff_c_outer_inner * 7) + 19)] = (conv2d_nchw_local[((ff_c_outer_inner * 7) + 19)] + (pad_temp_shared[((((((rc_outer_inner * 870) + (rc_inner * 435)) + ((((int)threadIdx.x) % 14) * 30)) + (ry_outer_inner * 15)) + rx_inner) + 10)] * kernel_shared[((((((((((int)threadIdx.x) / 14) * 144) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 1152)]));
              conv2d_nchw_local[((ff_c_outer_inner * 7) + 33)] = (conv2d_nchw_local[((ff_c_outer_inner * 7) + 33)] + (pad_temp_shared[((((((rc_outer_inner * 870) + (rc_inner * 435)) + ((((int)threadIdx.x) % 14) * 30)) + (ry_outer_inner * 15)) + rx_inner) + 10)] * kernel_shared[((((((((((int)threadIdx.x) / 14) * 144) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 2304)]));
              conv2d_nchw_local[((ff_c_outer_inner * 7) + 47)] = (conv2d_nchw_local[((ff_c_outer_inner * 7) + 47)] + (pad_temp_shared[((((((rc_outer_inner * 870) + (rc_inner * 435)) + ((((int)threadIdx.x) % 14) * 30)) + (ry_outer_inner * 15)) + rx_inner) + 10)] * kernel_shared[((((((((((int)threadIdx.x) / 14) * 144) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 3456)]));
              conv2d_nchw_local[((ff_c_outer_inner * 7) + 6)] = (conv2d_nchw_local[((ff_c_outer_inner * 7) + 6)] + (pad_temp_shared[((((((rc_outer_inner * 870) + (rc_inner * 435)) + ((((int)threadIdx.x) % 14) * 30)) + (ry_outer_inner * 15)) + rx_inner) + 12)] * kernel_shared[(((((((((int)threadIdx.x) / 14) * 144) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner)]));
              conv2d_nchw_local[((ff_c_outer_inner * 7) + 20)] = (conv2d_nchw_local[((ff_c_outer_inner * 7) + 20)] + (pad_temp_shared[((((((rc_outer_inner * 870) + (rc_inner * 435)) + ((((int)threadIdx.x) % 14) * 30)) + (ry_outer_inner * 15)) + rx_inner) + 12)] * kernel_shared[((((((((((int)threadIdx.x) / 14) * 144) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 1152)]));
              conv2d_nchw_local[((ff_c_outer_inner * 7) + 34)] = (conv2d_nchw_local[((ff_c_outer_inner * 7) + 34)] + (pad_temp_shared[((((((rc_outer_inner * 870) + (rc_inner * 435)) + ((((int)threadIdx.x) % 14) * 30)) + (ry_outer_inner * 15)) + rx_inner) + 12)] * kernel_shared[((((((((((int)threadIdx.x) / 14) * 144) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 2304)]));
              conv2d_nchw_local[((ff_c_outer_inner * 7) + 48)] = (conv2d_nchw_local[((ff_c_outer_inner * 7) + 48)] + (pad_temp_shared[((((((rc_outer_inner * 870) + (rc_inner * 435)) + ((((int)threadIdx.x) % 14) * 30)) + (ry_outer_inner * 15)) + rx_inner) + 12)] * kernel_shared[((((((((((int)threadIdx.x) / 14) * 144) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 3456)]));
            }
          }
        }
      }
    }
  }
  for (int ff_inner = 0; ff_inner < 2; ++ff_inner) {
    for (int xx_inner = 0; xx_inner < 7; ++xx_inner) {
      conv2d_nchw[((((((((((int)blockIdx.x) >> 3) * 50176) + ((((int)threadIdx.x) / 14) * 1568)) + (ff_inner * 784)) + (((((int)blockIdx.x) & 7) >> 2) * 392)) + ((((int)threadIdx.x) % 14) * 28)) + ((((int)blockIdx.x) & 3) * 7)) + xx_inner)] = conv2d_nchw_local[((ff_inner * 7) + xx_inner)];
      conv2d_nchw[(((((((((((int)blockIdx.x) >> 3) * 50176) + ((((int)threadIdx.x) / 14) * 1568)) + (ff_inner * 784)) + (((((int)blockIdx.x) & 7) >> 2) * 392)) + ((((int)threadIdx.x) % 14) * 28)) + ((((int)blockIdx.x) & 3) * 7)) + xx_inner) + 12544)] = conv2d_nchw_local[(((ff_inner * 7) + xx_inner) + 14)];
      conv2d_nchw[(((((((((((int)blockIdx.x) >> 3) * 50176) + ((((int)threadIdx.x) / 14) * 1568)) + (ff_inner * 784)) + (((((int)blockIdx.x) & 7) >> 2) * 392)) + ((((int)threadIdx.x) % 14) * 28)) + ((((int)blockIdx.x) & 3) * 7)) + xx_inner) + 25088)] = conv2d_nchw_local[(((ff_inner * 7) + xx_inner) + 28)];
      conv2d_nchw[(((((((((((int)blockIdx.x) >> 3) * 50176) + ((((int)threadIdx.x) / 14) * 1568)) + (ff_inner * 784)) + (((((int)blockIdx.x) & 7) >> 2) * 392)) + ((((int)threadIdx.x) % 14) * 28)) + ((((int)blockIdx.x) & 3) * 7)) + xx_inner) + 37632)] = conv2d_nchw_local[(((ff_inner * 7) + xx_inner) + 42)];
    }
  }
}


