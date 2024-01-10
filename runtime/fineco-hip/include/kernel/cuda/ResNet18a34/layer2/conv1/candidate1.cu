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
  __shared__ float pad_temp_shared[3840];
  __shared__ float kernel_shared[2304];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[14] = 0.000000e+00f;
  conv2d_nchw_local[28] = 0.000000e+00f;
  conv2d_nchw_local[42] = 0.000000e+00f;
  conv2d_nchw_local[7] = 0.000000e+00f;
  conv2d_nchw_local[21] = 0.000000e+00f;
  conv2d_nchw_local[35] = 0.000000e+00f;
  conv2d_nchw_local[49] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[15] = 0.000000e+00f;
  conv2d_nchw_local[29] = 0.000000e+00f;
  conv2d_nchw_local[43] = 0.000000e+00f;
  conv2d_nchw_local[8] = 0.000000e+00f;
  conv2d_nchw_local[22] = 0.000000e+00f;
  conv2d_nchw_local[36] = 0.000000e+00f;
  conv2d_nchw_local[50] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[16] = 0.000000e+00f;
  conv2d_nchw_local[30] = 0.000000e+00f;
  conv2d_nchw_local[44] = 0.000000e+00f;
  conv2d_nchw_local[9] = 0.000000e+00f;
  conv2d_nchw_local[23] = 0.000000e+00f;
  conv2d_nchw_local[37] = 0.000000e+00f;
  conv2d_nchw_local[51] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[17] = 0.000000e+00f;
  conv2d_nchw_local[31] = 0.000000e+00f;
  conv2d_nchw_local[45] = 0.000000e+00f;
  conv2d_nchw_local[10] = 0.000000e+00f;
  conv2d_nchw_local[24] = 0.000000e+00f;
  conv2d_nchw_local[38] = 0.000000e+00f;
  conv2d_nchw_local[52] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[18] = 0.000000e+00f;
  conv2d_nchw_local[32] = 0.000000e+00f;
  conv2d_nchw_local[46] = 0.000000e+00f;
  conv2d_nchw_local[11] = 0.000000e+00f;
  conv2d_nchw_local[25] = 0.000000e+00f;
  conv2d_nchw_local[39] = 0.000000e+00f;
  conv2d_nchw_local[53] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[19] = 0.000000e+00f;
  conv2d_nchw_local[33] = 0.000000e+00f;
  conv2d_nchw_local[47] = 0.000000e+00f;
  conv2d_nchw_local[12] = 0.000000e+00f;
  conv2d_nchw_local[26] = 0.000000e+00f;
  conv2d_nchw_local[40] = 0.000000e+00f;
  conv2d_nchw_local[54] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  conv2d_nchw_local[20] = 0.000000e+00f;
  conv2d_nchw_local[34] = 0.000000e+00f;
  conv2d_nchw_local[48] = 0.000000e+00f;
  conv2d_nchw_local[13] = 0.000000e+00f;
  conv2d_nchw_local[27] = 0.000000e+00f;
  conv2d_nchw_local[41] = 0.000000e+00f;
  conv2d_nchw_local[55] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 8; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = ((((1 <= ((((((int)blockIdx.x) & 7) >> 1) * 14) + (((int)threadIdx.x) / 30))) && (1 <= (((((int)blockIdx.x) & 1) * 28) + (((int)threadIdx.x) % 30)))) && ((((((int)blockIdx.x) & 1) * 28) + (((int)threadIdx.x) % 30)) < 57)) ? data[((((((rc_outer_outer * 25088) + (((((int)blockIdx.x) & 7) >> 1) * 784)) + ((((int)threadIdx.x) / 30) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + (((int)threadIdx.x) % 30)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 224)] = (((1 <= (((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 14) % 30))) && ((((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 14) % 30)) < 57)) ? data[((((((rc_outer_outer * 25088) + (((((int)blockIdx.x) & 7) >> 1) * 784)) + (((((int)threadIdx.x) + 224) / 30) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + ((((int)threadIdx.x) + 14) % 30)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 448)] = (((((1 <= ((((((int)blockIdx.x) & 7) >> 1) * 14) + ((((((int)threadIdx.x) >> 1) + 224) % 240) / 15))) && (((((((int)blockIdx.x) & 7) >> 1) * 14) + ((((((int)threadIdx.x) >> 1) + 224) % 240) / 15)) < 57)) && (1 <= (((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 28) % 30)))) && ((((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 28) % 30)) < 57)) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 448) / 480) * 3136)) + (((((int)blockIdx.x) & 7) >> 1) * 784)) + (((((((int)threadIdx.x) >> 1) + 224) % 240) / 15) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + ((((int)threadIdx.x) + 28) % 30)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 672)] = (((1 <= (((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 12) % 30))) && ((((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 12) % 30)) < 57)) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 672) / 480) * 3136)) + (((((int)blockIdx.x) & 7) >> 1) * 784)) + (((((((int)threadIdx.x) >> 1) + 96) % 240) / 15) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + ((((int)threadIdx.x) + 12) % 30)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 896)] = (((((1 <= ((((((int)blockIdx.x) & 7) >> 1) * 14) + ((((((int)threadIdx.x) >> 1) + 208) % 240) / 15))) && (((((((int)blockIdx.x) & 7) >> 1) * 14) + ((((((int)threadIdx.x) >> 1) + 208) % 240) / 15)) < 57)) && (1 <= (((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 26) % 30)))) && ((((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 26) % 30)) < 57)) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 896) / 480) * 3136)) + (((((int)blockIdx.x) & 7) >> 1) * 784)) + (((((((int)threadIdx.x) >> 1) + 208) % 240) / 15) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + ((((int)threadIdx.x) + 26) % 30)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1120)] = (((1 <= (((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 10) % 30))) && ((((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 10) % 30)) < 57)) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 1120) / 480) * 3136)) + (((((int)blockIdx.x) & 7) >> 1) * 784)) + (((((((int)threadIdx.x) >> 1) + 80) % 240) / 15) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + ((((int)threadIdx.x) + 10) % 30)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1344)] = (((((1 <= ((((((int)blockIdx.x) & 7) >> 1) * 14) + ((((((int)threadIdx.x) >> 1) + 192) % 240) / 15))) && (((((((int)blockIdx.x) & 7) >> 1) * 14) + ((((((int)threadIdx.x) >> 1) + 192) % 240) / 15)) < 57)) && (1 <= (((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 24) % 30)))) && ((((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 24) % 30)) < 57)) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 1344) / 480) * 3136)) + (((((int)blockIdx.x) & 7) >> 1) * 784)) + (((((((int)threadIdx.x) >> 1) + 192) % 240) / 15) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + ((((int)threadIdx.x) + 24) % 30)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1568)] = (((1 <= (((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 8) % 30))) && ((((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 8) % 30)) < 57)) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 1568) / 480) * 3136)) + (((((int)blockIdx.x) & 7) >> 1) * 784)) + (((((((int)threadIdx.x) >> 1) + 64) % 240) / 15) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + ((((int)threadIdx.x) + 8) % 30)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1792)] = (((((1 <= ((((((int)blockIdx.x) & 7) >> 1) * 14) + ((((((int)threadIdx.x) >> 1) + 176) % 240) / 15))) && (((((((int)blockIdx.x) & 7) >> 1) * 14) + ((((((int)threadIdx.x) >> 1) + 176) % 240) / 15)) < 57)) && (1 <= (((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 22) % 30)))) && ((((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 22) % 30)) < 57)) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 1792) / 480) * 3136)) + (((((int)blockIdx.x) & 7) >> 1) * 784)) + (((((((int)threadIdx.x) >> 1) + 176) % 240) / 15) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + ((((int)threadIdx.x) + 22) % 30)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 2016)] = (((1 <= (((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 6) % 30))) && ((((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 6) % 30)) < 57)) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 2016) / 480) * 3136)) + (((((int)blockIdx.x) & 7) >> 1) * 784)) + (((((((int)threadIdx.x) >> 1) + 48) % 240) / 15) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + ((((int)threadIdx.x) + 6) % 30)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 2240)] = (((((1 <= ((((((int)blockIdx.x) & 7) >> 1) * 14) + ((((((int)threadIdx.x) >> 1) + 160) % 240) / 15))) && (((((((int)blockIdx.x) & 7) >> 1) * 14) + ((((((int)threadIdx.x) >> 1) + 160) % 240) / 15)) < 57)) && (1 <= (((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 20) % 30)))) && ((((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 20) % 30)) < 57)) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 2240) / 480) * 3136)) + (((((int)blockIdx.x) & 7) >> 1) * 784)) + (((((((int)threadIdx.x) >> 1) + 160) % 240) / 15) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + ((((int)threadIdx.x) + 20) % 30)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 2464)] = (((1 <= (((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 4) % 30))) && ((((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 4) % 30)) < 57)) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 2464) / 480) * 3136)) + (((((int)blockIdx.x) & 7) >> 1) * 784)) + (((((((int)threadIdx.x) >> 1) + 32) % 240) / 15) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + ((((int)threadIdx.x) + 4) % 30)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 2688)] = (((((1 <= ((((((int)blockIdx.x) & 7) >> 1) * 14) + ((((((int)threadIdx.x) >> 1) + 144) % 240) / 15))) && (((((((int)blockIdx.x) & 7) >> 1) * 14) + ((((((int)threadIdx.x) >> 1) + 144) % 240) / 15)) < 57)) && (1 <= (((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 18) % 30)))) && ((((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 18) % 30)) < 57)) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 2688) / 480) * 3136)) + (((((int)blockIdx.x) & 7) >> 1) * 784)) + (((((((int)threadIdx.x) >> 1) + 144) % 240) / 15) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + ((((int)threadIdx.x) + 18) % 30)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 2912)] = (((1 <= (((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 2) % 30))) && ((((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 2) % 30)) < 57)) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 2912) / 480) * 3136)) + (((((int)blockIdx.x) & 7) >> 1) * 784)) + (((((((int)threadIdx.x) >> 1) + 16) % 240) / 15) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + ((((int)threadIdx.x) + 2) % 30)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 3136)] = ((((((((((int)blockIdx.x) & 7) >> 1) * 14) + ((((((int)threadIdx.x) >> 1) + 128) % 240) / 15)) < 57) && (1 <= (((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 16) % 30)))) && ((((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 16) % 30)) < 57)) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 3136) / 480) * 3136)) + (((((int)blockIdx.x) & 7) >> 1) * 784)) + (((((((int)threadIdx.x) >> 1) + 128) % 240) / 15) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + ((((int)threadIdx.x) + 16) % 30)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 3360)] = ((((1 <= ((((((int)blockIdx.x) & 7) >> 1) * 14) + (((int)threadIdx.x) / 30))) && (1 <= (((((int)blockIdx.x) & 1) * 28) + (((int)threadIdx.x) % 30)))) && ((((((int)blockIdx.x) & 1) * 28) + (((int)threadIdx.x) % 30)) < 57)) ? data[((((((rc_outer_outer * 25088) + (((((int)blockIdx.x) & 7) >> 1) * 784)) + ((((int)threadIdx.x) / 30) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + (((int)threadIdx.x) % 30)) + 21895)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 3584)] = (((1 <= (((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 14) % 30))) && ((((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 14) % 30)) < 57)) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 3584) / 480) * 3136)) + (((((int)blockIdx.x) & 7) >> 1) * 784)) + (((((((int)threadIdx.x) >> 1) + 112) % 240) / 15) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + ((((int)threadIdx.x) + 14) % 30)) - 57)] : 0.000000e+00f);
    if (((int)threadIdx.x) < 32) {
      pad_temp_shared[(((int)threadIdx.x) + 3808)] = ((((((((((int)blockIdx.x) & 7) >> 1) * 14) + ((((((int)threadIdx.x) >> 1) + 224) % 240) / 15)) < 57) && (1 <= (((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 28) % 30)))) && ((((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 28) % 30)) < 57)) ? data[((((((rc_outer_outer * 25088) + (((((int)blockIdx.x) & 7) >> 1) * 784)) + (((((((int)threadIdx.x) >> 1) + 224) % 240) / 15) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + ((((int)threadIdx.x) + 28) % 30)) + 21895)] : 0.000000e+00f);
    }
    kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) >> 3) * 18432) + ((((int)threadIdx.x) / 72) * 576)) + (rc_outer_outer * 72)) + (((int)threadIdx.x) % 72))];
    kernel_shared[(((int)threadIdx.x) + 224)] = kernel[(((((((int)blockIdx.x) >> 3) * 18432) + (((((int)threadIdx.x) + 224) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 8) % 72))];
    kernel_shared[(((int)threadIdx.x) + 448)] = kernel[(((((((int)blockIdx.x) >> 3) * 18432) + (((((int)threadIdx.x) + 448) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 16) % 72))];
    kernel_shared[(((int)threadIdx.x) + 672)] = kernel[(((((((int)blockIdx.x) >> 3) * 18432) + (((((int)threadIdx.x) + 672) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 24) % 72))];
    kernel_shared[(((int)threadIdx.x) + 896)] = kernel[(((((((int)blockIdx.x) >> 3) * 18432) + (((((int)threadIdx.x) + 896) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 32) % 72))];
    kernel_shared[(((int)threadIdx.x) + 1120)] = kernel[(((((((int)blockIdx.x) >> 3) * 18432) + (((((int)threadIdx.x) + 1120) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 40) % 72))];
    kernel_shared[(((int)threadIdx.x) + 1344)] = kernel[(((((((int)blockIdx.x) >> 3) * 18432) + (((((int)threadIdx.x) + 1344) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 48) % 72))];
    kernel_shared[(((int)threadIdx.x) + 1568)] = kernel[(((((((int)blockIdx.x) >> 3) * 18432) + (((((int)threadIdx.x) + 1568) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 56) % 72))];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[(((((((int)blockIdx.x) >> 3) * 18432) + (((((int)threadIdx.x) + 1792) / 72) * 576)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 64) % 72))];
    kernel_shared[(((int)threadIdx.x) + 2016)] = kernel[((((((((int)blockIdx.x) >> 3) * 18432) + ((((int)threadIdx.x) / 72) * 576)) + (rc_outer_outer * 72)) + (((int)threadIdx.x) % 72)) + 16128)];
    if (((int)threadIdx.x) < 64) {
      kernel_shared[(((int)threadIdx.x) + 2240)] = kernel[(((((((int)blockIdx.x) >> 3) * 18432) + (((((int)threadIdx.x) + 2240) / 72) * 576)) + (rc_outer_outer * 72)) + (((int)threadIdx.x) + 8))];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 4; ++rc_outer_inner) {
      for (int ry_outer_inner = 0; ry_outer_inner < 3; ++ry_outer_inner) {
        for (int rx_outer_inner = 0; rx_outer_inner < 3; ++rx_outer_inner) {
          for (int xx_c_outer_inner = 0; xx_c_outer_inner < 7; ++xx_c_outer_inner) {
            conv2d_nchw_local[xx_c_outer_inner] = (conv2d_nchw_local[xx_c_outer_inner] + (pad_temp_shared[((((((rc_outer_inner * 960) + (((((int)threadIdx.x) % 14) >> 1) * 60)) + (ry_outer_inner * 30)) + ((((int)threadIdx.x) & 1) * 7)) + xx_c_outer_inner) + rx_outer_inner)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 72) + (rc_outer_inner * 18)) + (ry_outer_inner * 3)) + rx_outer_inner)]));
            conv2d_nchw_local[(xx_c_outer_inner + 14)] = (conv2d_nchw_local[(xx_c_outer_inner + 14)] + (pad_temp_shared[(((((((rc_outer_inner * 960) + (((((int)threadIdx.x) % 14) >> 1) * 60)) + (ry_outer_inner * 30)) + ((((int)threadIdx.x) & 1) * 7)) + xx_c_outer_inner) + rx_outer_inner) + 14)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 72) + (rc_outer_inner * 18)) + (ry_outer_inner * 3)) + rx_outer_inner)]));
            conv2d_nchw_local[(xx_c_outer_inner + 28)] = (conv2d_nchw_local[(xx_c_outer_inner + 28)] + (pad_temp_shared[((((((rc_outer_inner * 960) + (((((int)threadIdx.x) % 14) >> 1) * 60)) + (ry_outer_inner * 30)) + ((((int)threadIdx.x) & 1) * 7)) + xx_c_outer_inner) + rx_outer_inner)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 72) + (rc_outer_inner * 18)) + (ry_outer_inner * 3)) + rx_outer_inner) + 1152)]));
            conv2d_nchw_local[(xx_c_outer_inner + 42)] = (conv2d_nchw_local[(xx_c_outer_inner + 42)] + (pad_temp_shared[(((((((rc_outer_inner * 960) + (((((int)threadIdx.x) % 14) >> 1) * 60)) + (ry_outer_inner * 30)) + ((((int)threadIdx.x) & 1) * 7)) + xx_c_outer_inner) + rx_outer_inner) + 14)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 72) + (rc_outer_inner * 18)) + (ry_outer_inner * 3)) + rx_outer_inner) + 1152)]));
            conv2d_nchw_local[(xx_c_outer_inner + 7)] = (conv2d_nchw_local[(xx_c_outer_inner + 7)] + (pad_temp_shared[(((((((rc_outer_inner * 960) + (((((int)threadIdx.x) % 14) >> 1) * 60)) + (ry_outer_inner * 30)) + ((((int)threadIdx.x) & 1) * 7)) + xx_c_outer_inner) + rx_outer_inner) + 30)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 72) + (rc_outer_inner * 18)) + (ry_outer_inner * 3)) + rx_outer_inner)]));
            conv2d_nchw_local[(xx_c_outer_inner + 21)] = (conv2d_nchw_local[(xx_c_outer_inner + 21)] + (pad_temp_shared[(((((((rc_outer_inner * 960) + (((((int)threadIdx.x) % 14) >> 1) * 60)) + (ry_outer_inner * 30)) + ((((int)threadIdx.x) & 1) * 7)) + xx_c_outer_inner) + rx_outer_inner) + 44)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 72) + (rc_outer_inner * 18)) + (ry_outer_inner * 3)) + rx_outer_inner)]));
            conv2d_nchw_local[(xx_c_outer_inner + 35)] = (conv2d_nchw_local[(xx_c_outer_inner + 35)] + (pad_temp_shared[(((((((rc_outer_inner * 960) + (((((int)threadIdx.x) % 14) >> 1) * 60)) + (ry_outer_inner * 30)) + ((((int)threadIdx.x) & 1) * 7)) + xx_c_outer_inner) + rx_outer_inner) + 30)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 72) + (rc_outer_inner * 18)) + (ry_outer_inner * 3)) + rx_outer_inner) + 1152)]));
            conv2d_nchw_local[(xx_c_outer_inner + 49)] = (conv2d_nchw_local[(xx_c_outer_inner + 49)] + (pad_temp_shared[(((((((rc_outer_inner * 960) + (((((int)threadIdx.x) % 14) >> 1) * 60)) + (ry_outer_inner * 30)) + ((((int)threadIdx.x) & 1) * 7)) + xx_c_outer_inner) + rx_outer_inner) + 44)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 72) + (rc_outer_inner * 18)) + (ry_outer_inner * 3)) + rx_outer_inner) + 1152)]));
            conv2d_nchw_local[xx_c_outer_inner] = (conv2d_nchw_local[xx_c_outer_inner] + (pad_temp_shared[(((((((rc_outer_inner * 960) + (((((int)threadIdx.x) % 14) >> 1) * 60)) + (ry_outer_inner * 30)) + ((((int)threadIdx.x) & 1) * 7)) + xx_c_outer_inner) + rx_outer_inner) + 480)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 72) + (rc_outer_inner * 18)) + (ry_outer_inner * 3)) + rx_outer_inner) + 9)]));
            conv2d_nchw_local[(xx_c_outer_inner + 14)] = (conv2d_nchw_local[(xx_c_outer_inner + 14)] + (pad_temp_shared[(((((((rc_outer_inner * 960) + (((((int)threadIdx.x) % 14) >> 1) * 60)) + (ry_outer_inner * 30)) + ((((int)threadIdx.x) & 1) * 7)) + xx_c_outer_inner) + rx_outer_inner) + 494)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 72) + (rc_outer_inner * 18)) + (ry_outer_inner * 3)) + rx_outer_inner) + 9)]));
            conv2d_nchw_local[(xx_c_outer_inner + 28)] = (conv2d_nchw_local[(xx_c_outer_inner + 28)] + (pad_temp_shared[(((((((rc_outer_inner * 960) + (((((int)threadIdx.x) % 14) >> 1) * 60)) + (ry_outer_inner * 30)) + ((((int)threadIdx.x) & 1) * 7)) + xx_c_outer_inner) + rx_outer_inner) + 480)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 72) + (rc_outer_inner * 18)) + (ry_outer_inner * 3)) + rx_outer_inner) + 1161)]));
            conv2d_nchw_local[(xx_c_outer_inner + 42)] = (conv2d_nchw_local[(xx_c_outer_inner + 42)] + (pad_temp_shared[(((((((rc_outer_inner * 960) + (((((int)threadIdx.x) % 14) >> 1) * 60)) + (ry_outer_inner * 30)) + ((((int)threadIdx.x) & 1) * 7)) + xx_c_outer_inner) + rx_outer_inner) + 494)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 72) + (rc_outer_inner * 18)) + (ry_outer_inner * 3)) + rx_outer_inner) + 1161)]));
            conv2d_nchw_local[(xx_c_outer_inner + 7)] = (conv2d_nchw_local[(xx_c_outer_inner + 7)] + (pad_temp_shared[(((((((rc_outer_inner * 960) + (((((int)threadIdx.x) % 14) >> 1) * 60)) + (ry_outer_inner * 30)) + ((((int)threadIdx.x) & 1) * 7)) + xx_c_outer_inner) + rx_outer_inner) + 510)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 72) + (rc_outer_inner * 18)) + (ry_outer_inner * 3)) + rx_outer_inner) + 9)]));
            conv2d_nchw_local[(xx_c_outer_inner + 21)] = (conv2d_nchw_local[(xx_c_outer_inner + 21)] + (pad_temp_shared[(((((((rc_outer_inner * 960) + (((((int)threadIdx.x) % 14) >> 1) * 60)) + (ry_outer_inner * 30)) + ((((int)threadIdx.x) & 1) * 7)) + xx_c_outer_inner) + rx_outer_inner) + 524)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 72) + (rc_outer_inner * 18)) + (ry_outer_inner * 3)) + rx_outer_inner) + 9)]));
            conv2d_nchw_local[(xx_c_outer_inner + 35)] = (conv2d_nchw_local[(xx_c_outer_inner + 35)] + (pad_temp_shared[(((((((rc_outer_inner * 960) + (((((int)threadIdx.x) % 14) >> 1) * 60)) + (ry_outer_inner * 30)) + ((((int)threadIdx.x) & 1) * 7)) + xx_c_outer_inner) + rx_outer_inner) + 510)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 72) + (rc_outer_inner * 18)) + (ry_outer_inner * 3)) + rx_outer_inner) + 1161)]));
            conv2d_nchw_local[(xx_c_outer_inner + 49)] = (conv2d_nchw_local[(xx_c_outer_inner + 49)] + (pad_temp_shared[(((((((rc_outer_inner * 960) + (((((int)threadIdx.x) % 14) >> 1) * 60)) + (ry_outer_inner * 30)) + ((((int)threadIdx.x) & 1) * 7)) + xx_c_outer_inner) + rx_outer_inner) + 524)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 72) + (rc_outer_inner * 18)) + (ry_outer_inner * 3)) + rx_outer_inner) + 1161)]));
          }
        }
      }
    }
  }
  for (int yy_inner = 0; yy_inner < 2; ++yy_inner) {
    for (int xx_inner = 0; xx_inner < 7; ++xx_inner) {
      conv2d_nchw[(((((((((((int)blockIdx.x) >> 3) * 100352) + ((((int)threadIdx.x) / 14) * 3136)) + (((((int)blockIdx.x) & 7) >> 1) * 784)) + (((((int)threadIdx.x) % 14) >> 1) * 112)) + (yy_inner * 56)) + ((((int)blockIdx.x) & 1) * 28)) + ((((int)threadIdx.x) & 1) * 7)) + xx_inner)] = conv2d_nchw_local[((yy_inner * 7) + xx_inner)];
      conv2d_nchw[((((((((((((int)blockIdx.x) >> 3) * 100352) + ((((int)threadIdx.x) / 14) * 3136)) + (((((int)blockIdx.x) & 7) >> 1) * 784)) + (((((int)threadIdx.x) % 14) >> 1) * 112)) + (yy_inner * 56)) + ((((int)blockIdx.x) & 1) * 28)) + ((((int)threadIdx.x) & 1) * 7)) + xx_inner) + 14)] = conv2d_nchw_local[(((yy_inner * 7) + xx_inner) + 14)];
      conv2d_nchw[((((((((((((int)blockIdx.x) >> 3) * 100352) + ((((int)threadIdx.x) / 14) * 3136)) + (((((int)blockIdx.x) & 7) >> 1) * 784)) + (((((int)threadIdx.x) % 14) >> 1) * 112)) + (yy_inner * 56)) + ((((int)blockIdx.x) & 1) * 28)) + ((((int)threadIdx.x) & 1) * 7)) + xx_inner) + 50176)] = conv2d_nchw_local[(((yy_inner * 7) + xx_inner) + 28)];
      conv2d_nchw[((((((((((((int)blockIdx.x) >> 3) * 100352) + ((((int)threadIdx.x) / 14) * 3136)) + (((((int)blockIdx.x) & 7) >> 1) * 784)) + (((((int)threadIdx.x) % 14) >> 1) * 112)) + (yy_inner * 56)) + ((((int)blockIdx.x) & 1) * 28)) + ((((int)threadIdx.x) & 1) * 7)) + xx_inner) + 50190)] = conv2d_nchw_local[(((yy_inner * 7) + xx_inner) + 42)];
    }
  }
}


