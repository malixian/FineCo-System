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
extern "C" __global__ void __launch_bounds__(224) candidate2(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[64];
  __shared__ float pad_temp_shared[3584];
  __shared__ float kernel_shared[4096];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[16] = 0.000000e+00f;
  conv2d_nchw_local[32] = 0.000000e+00f;
  conv2d_nchw_local[48] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[17] = 0.000000e+00f;
  conv2d_nchw_local[33] = 0.000000e+00f;
  conv2d_nchw_local[49] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[18] = 0.000000e+00f;
  conv2d_nchw_local[34] = 0.000000e+00f;
  conv2d_nchw_local[50] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[19] = 0.000000e+00f;
  conv2d_nchw_local[35] = 0.000000e+00f;
  conv2d_nchw_local[51] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[20] = 0.000000e+00f;
  conv2d_nchw_local[36] = 0.000000e+00f;
  conv2d_nchw_local[52] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[21] = 0.000000e+00f;
  conv2d_nchw_local[37] = 0.000000e+00f;
  conv2d_nchw_local[53] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  conv2d_nchw_local[22] = 0.000000e+00f;
  conv2d_nchw_local[38] = 0.000000e+00f;
  conv2d_nchw_local[54] = 0.000000e+00f;
  conv2d_nchw_local[7] = 0.000000e+00f;
  conv2d_nchw_local[23] = 0.000000e+00f;
  conv2d_nchw_local[39] = 0.000000e+00f;
  conv2d_nchw_local[55] = 0.000000e+00f;
  conv2d_nchw_local[8] = 0.000000e+00f;
  conv2d_nchw_local[24] = 0.000000e+00f;
  conv2d_nchw_local[40] = 0.000000e+00f;
  conv2d_nchw_local[56] = 0.000000e+00f;
  conv2d_nchw_local[9] = 0.000000e+00f;
  conv2d_nchw_local[25] = 0.000000e+00f;
  conv2d_nchw_local[41] = 0.000000e+00f;
  conv2d_nchw_local[57] = 0.000000e+00f;
  conv2d_nchw_local[10] = 0.000000e+00f;
  conv2d_nchw_local[26] = 0.000000e+00f;
  conv2d_nchw_local[42] = 0.000000e+00f;
  conv2d_nchw_local[58] = 0.000000e+00f;
  conv2d_nchw_local[11] = 0.000000e+00f;
  conv2d_nchw_local[27] = 0.000000e+00f;
  conv2d_nchw_local[43] = 0.000000e+00f;
  conv2d_nchw_local[59] = 0.000000e+00f;
  conv2d_nchw_local[12] = 0.000000e+00f;
  conv2d_nchw_local[28] = 0.000000e+00f;
  conv2d_nchw_local[44] = 0.000000e+00f;
  conv2d_nchw_local[60] = 0.000000e+00f;
  conv2d_nchw_local[13] = 0.000000e+00f;
  conv2d_nchw_local[29] = 0.000000e+00f;
  conv2d_nchw_local[45] = 0.000000e+00f;
  conv2d_nchw_local[61] = 0.000000e+00f;
  conv2d_nchw_local[14] = 0.000000e+00f;
  conv2d_nchw_local[30] = 0.000000e+00f;
  conv2d_nchw_local[46] = 0.000000e+00f;
  conv2d_nchw_local[62] = 0.000000e+00f;
  conv2d_nchw_local[15] = 0.000000e+00f;
  conv2d_nchw_local[31] = 0.000000e+00f;
  conv2d_nchw_local[47] = 0.000000e+00f;
  conv2d_nchw_local[63] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 4; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = data[((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112))];
    pad_temp_shared[(((int)threadIdx.x) + 224)] = data[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 1568)];
    pad_temp_shared[(((int)threadIdx.x) + 448)] = data[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 3136)];
    pad_temp_shared[(((int)threadIdx.x) + 672)] = data[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 4704)];
    pad_temp_shared[(((int)threadIdx.x) + 896)] = data[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 6272)];
    pad_temp_shared[(((int)threadIdx.x) + 1120)] = data[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 7840)];
    pad_temp_shared[(((int)threadIdx.x) + 1344)] = data[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 9408)];
    pad_temp_shared[(((int)threadIdx.x) + 1568)] = data[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 10976)];
    pad_temp_shared[(((int)threadIdx.x) + 1792)] = data[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 12544)];
    pad_temp_shared[(((int)threadIdx.x) + 2016)] = data[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 14112)];
    pad_temp_shared[(((int)threadIdx.x) + 2240)] = data[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 15680)];
    pad_temp_shared[(((int)threadIdx.x) + 2464)] = data[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 17248)];
    pad_temp_shared[(((int)threadIdx.x) + 2688)] = data[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 18816)];
    pad_temp_shared[(((int)threadIdx.x) + 2912)] = data[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 20384)];
    pad_temp_shared[(((int)threadIdx.x) + 3136)] = data[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 21952)];
    pad_temp_shared[(((int)threadIdx.x) + 3360)] = data[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 23520)];
    kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31))];
    kernel_shared[(((int)threadIdx.x) + 224)] = kernel[((((((((int)blockIdx.x) / 7) * 16384) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 896)];
    kernel_shared[(((int)threadIdx.x) + 448)] = kernel[((((((((int)blockIdx.x) / 7) * 16384) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 1792)];
    kernel_shared[(((int)threadIdx.x) + 672)] = kernel[((((((((int)blockIdx.x) / 7) * 16384) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 2688)];
    kernel_shared[(((int)threadIdx.x) + 896)] = kernel[((((((((int)blockIdx.x) / 7) * 16384) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 3584)];
    kernel_shared[(((int)threadIdx.x) + 1120)] = kernel[((((((((int)blockIdx.x) / 7) * 16384) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 4480)];
    kernel_shared[(((int)threadIdx.x) + 1344)] = kernel[((((((((int)blockIdx.x) / 7) * 16384) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 5376)];
    kernel_shared[(((int)threadIdx.x) + 1568)] = kernel[((((((((int)blockIdx.x) / 7) * 16384) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 6272)];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[((((((((int)blockIdx.x) / 7) * 16384) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 7168)];
    kernel_shared[(((int)threadIdx.x) + 2016)] = kernel[((((((((int)blockIdx.x) / 7) * 16384) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 8064)];
    kernel_shared[(((int)threadIdx.x) + 2240)] = kernel[((((((((int)blockIdx.x) / 7) * 16384) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 8960)];
    kernel_shared[(((int)threadIdx.x) + 2464)] = kernel[((((((((int)blockIdx.x) / 7) * 16384) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 9856)];
    kernel_shared[(((int)threadIdx.x) + 2688)] = kernel[((((((((int)blockIdx.x) / 7) * 16384) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 10752)];
    kernel_shared[(((int)threadIdx.x) + 2912)] = kernel[((((((((int)blockIdx.x) / 7) * 16384) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 11648)];
    kernel_shared[(((int)threadIdx.x) + 3136)] = kernel[((((((((int)blockIdx.x) / 7) * 16384) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 12544)];
    kernel_shared[(((int)threadIdx.x) + 3360)] = kernel[((((((((int)blockIdx.x) / 7) * 16384) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 13440)];
    kernel_shared[(((int)threadIdx.x) + 3584)] = kernel[((((((((int)blockIdx.x) / 7) * 16384) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 14336)];
    kernel_shared[(((int)threadIdx.x) + 3808)] = kernel[((((((((int)blockIdx.x) / 7) * 16384) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 15232)];
    if (((int)threadIdx.x) < 64) {
      kernel_shared[(((int)threadIdx.x) + 4032)] = kernel[((((((((int)blockIdx.x) / 7) * 16384) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 16128)];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 4; ++rc_outer_inner) {
      for (int ff_c_outer_inner = 0; ff_c_outer_inner < 4; ++ff_c_outer_inner) {
        for (int yy_c_outer_inner = 0; yy_c_outer_inner < 2; ++yy_c_outer_inner) {
          conv2d_nchw_local[((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2))] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2))] + (pad_temp_shared[((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2))] * kernel_shared[((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8))]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 16)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 16)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8))]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 32)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 32)] + (pad_temp_shared[((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2))] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 2048)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 48)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 48)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 14)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 2048)]));
          conv2d_nchw_local[((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2))] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2))] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 112)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 1)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 16)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 16)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 126)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 1)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 32)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 32)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 112)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 2049)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 48)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 48)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 126)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 2049)]));
          conv2d_nchw_local[((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2))] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2))] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 224)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 2)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 16)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 16)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 238)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 2)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 32)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 32)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 224)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 2050)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 48)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 48)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 238)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 2050)]));
          conv2d_nchw_local[((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2))] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2))] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 336)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 3)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 16)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 16)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 350)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 3)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 32)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 32)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 336)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 2051)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 48)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 48)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 350)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 2051)]));
          conv2d_nchw_local[((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2))] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2))] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 448)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 4)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 16)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 16)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 462)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 4)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 32)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 32)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 448)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 2052)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 48)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 48)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 462)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 2052)]));
          conv2d_nchw_local[((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2))] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2))] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 560)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 5)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 16)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 16)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 574)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 5)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 32)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 32)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 560)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 2053)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 48)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 48)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 574)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 2053)]));
          conv2d_nchw_local[((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2))] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2))] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 672)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 6)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 16)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 16)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 686)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 6)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 32)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 32)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 672)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 2054)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 48)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 48)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 686)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 2054)]));
          conv2d_nchw_local[((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2))] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2))] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 784)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 7)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 16)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 16)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 798)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 7)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 32)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 32)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 784)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 2055)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 48)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 48)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 798)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 2055)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 1)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 1)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 1)] * kernel_shared[((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8))]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 17)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 17)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 15)] * kernel_shared[((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8))]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 33)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 33)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 1)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 2048)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 49)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 49)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 15)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 2048)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 1)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 1)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 113)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 1)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 17)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 17)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 127)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 1)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 33)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 33)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 113)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 2049)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 49)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 49)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 127)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 2049)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 1)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 1)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 225)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 2)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 17)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 17)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 239)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 2)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 33)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 33)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 225)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 2050)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 49)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 49)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 239)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 2050)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 1)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 1)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 337)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 3)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 17)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 17)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 351)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 3)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 33)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 33)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 337)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 2051)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 49)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 49)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 351)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 2051)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 1)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 1)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 449)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 4)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 17)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 17)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 463)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 4)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 33)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 33)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 449)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 2052)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 49)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 49)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 463)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 2052)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 1)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 1)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 561)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 5)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 17)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 17)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 575)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 5)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 33)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 33)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 561)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 2053)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 49)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 49)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 575)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 2053)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 1)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 1)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 673)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 6)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 17)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 17)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 687)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 6)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 33)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 33)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 673)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 2054)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 49)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 49)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 687)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 2054)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 1)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 1)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 785)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 7)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 17)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 17)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 799)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 7)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 33)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 33)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 785)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 2055)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 49)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (yy_c_outer_inner * 2)) + 49)] + (pad_temp_shared[(((((rc_outer_inner * 896) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + 799)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 8)) + 2055)]));
        }
      }
    }
  }
  for (int ff_inner = 0; ff_inner < 4; ++ff_inner) {
    for (int yy_inner = 0; yy_inner < 2; ++yy_inner) {
      for (int xx_inner = 0; xx_inner < 2; ++xx_inner) {
        conv2d_nchw[(((((((((((int)blockIdx.x) / 7) * 100352) + ((((int)threadIdx.x) / 14) * 3136)) + (ff_inner * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + xx_inner)] = conv2d_nchw_local[(((ff_inner * 4) + (yy_inner * 2)) + xx_inner)];
        conv2d_nchw[((((((((((((int)blockIdx.x) / 7) * 100352) + ((((int)threadIdx.x) / 14) * 3136)) + (ff_inner * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + xx_inner) + 14)] = conv2d_nchw_local[((((ff_inner * 4) + (yy_inner * 2)) + xx_inner) + 16)];
        conv2d_nchw[((((((((((((int)blockIdx.x) / 7) * 100352) + ((((int)threadIdx.x) / 14) * 3136)) + (ff_inner * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + xx_inner) + 50176)] = conv2d_nchw_local[((((ff_inner * 4) + (yy_inner * 2)) + xx_inner) + 32)];
        conv2d_nchw[((((((((((((int)blockIdx.x) / 7) * 100352) + ((((int)threadIdx.x) / 14) * 3136)) + (ff_inner * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((((int)threadIdx.x) % 14) / 7) * 56)) + (yy_inner * 28)) + ((((int)threadIdx.x) % 7) * 2)) + xx_inner) + 50190)] = conv2d_nchw_local[((((ff_inner * 4) + (yy_inner * 2)) + xx_inner) + 48)];
      }
    }
  }
}


