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
extern "C" __global__ void __launch_bounds__(128) candidate4(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[32];
  __shared__ float pad_temp_shared[512];
  __shared__ float kernel_shared[8192];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[8] = 0.000000e+00f;
  conv2d_nchw_local[12] = 0.000000e+00f;
  conv2d_nchw_local[16] = 0.000000e+00f;
  conv2d_nchw_local[20] = 0.000000e+00f;
  conv2d_nchw_local[24] = 0.000000e+00f;
  conv2d_nchw_local[28] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[9] = 0.000000e+00f;
  conv2d_nchw_local[13] = 0.000000e+00f;
  conv2d_nchw_local[17] = 0.000000e+00f;
  conv2d_nchw_local[21] = 0.000000e+00f;
  conv2d_nchw_local[25] = 0.000000e+00f;
  conv2d_nchw_local[29] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  conv2d_nchw_local[10] = 0.000000e+00f;
  conv2d_nchw_local[14] = 0.000000e+00f;
  conv2d_nchw_local[18] = 0.000000e+00f;
  conv2d_nchw_local[22] = 0.000000e+00f;
  conv2d_nchw_local[26] = 0.000000e+00f;
  conv2d_nchw_local[30] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[7] = 0.000000e+00f;
  conv2d_nchw_local[11] = 0.000000e+00f;
  conv2d_nchw_local[15] = 0.000000e+00f;
  conv2d_nchw_local[19] = 0.000000e+00f;
  conv2d_nchw_local[23] = 0.000000e+00f;
  conv2d_nchw_local[27] = 0.000000e+00f;
  conv2d_nchw_local[31] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 16; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = data[((((((rc_outer_outer * 25088) + ((((int)threadIdx.x) >> 4) * 784)) + ((((int)blockIdx.x) / 7) * 112)) + (((((int)threadIdx.x) & 15) >> 2) * 28)) + ((((int)blockIdx.x) % 7) * 4)) + (((int)threadIdx.x) & 3))];
    pad_temp_shared[(((int)threadIdx.x) + 128)] = data[(((((((rc_outer_outer * 25088) + ((((int)threadIdx.x) >> 4) * 784)) + ((((int)blockIdx.x) / 7) * 112)) + (((((int)threadIdx.x) & 15) >> 2) * 28)) + ((((int)blockIdx.x) % 7) * 4)) + (((int)threadIdx.x) & 3)) + 6272)];
    pad_temp_shared[(((int)threadIdx.x) + 256)] = data[(((((((rc_outer_outer * 25088) + ((((int)threadIdx.x) >> 4) * 784)) + ((((int)blockIdx.x) / 7) * 112)) + (((((int)threadIdx.x) & 15) >> 2) * 28)) + ((((int)blockIdx.x) % 7) * 4)) + (((int)threadIdx.x) & 3)) + 12544)];
    pad_temp_shared[(((int)threadIdx.x) + 384)] = data[(((((((rc_outer_outer * 25088) + ((((int)threadIdx.x) >> 4) * 784)) + ((((int)blockIdx.x) / 7) * 112)) + (((((int)threadIdx.x) & 15) >> 2) * 28)) + ((((int)blockIdx.x) % 7) * 4)) + (((int)threadIdx.x) & 3)) + 18816)];
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31))];
    kernel_shared[(((int)threadIdx.x) + 128)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 2048)];
    kernel_shared[(((int)threadIdx.x) + 256)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 4096)];
    kernel_shared[(((int)threadIdx.x) + 384)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 6144)];
    kernel_shared[(((int)threadIdx.x) + 512)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 8192)];
    kernel_shared[(((int)threadIdx.x) + 640)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 10240)];
    kernel_shared[(((int)threadIdx.x) + 768)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 12288)];
    kernel_shared[(((int)threadIdx.x) + 896)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 14336)];
    kernel_shared[(((int)threadIdx.x) + 1024)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 16384)];
    kernel_shared[(((int)threadIdx.x) + 1152)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 18432)];
    kernel_shared[(((int)threadIdx.x) + 1280)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 20480)];
    kernel_shared[(((int)threadIdx.x) + 1408)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 22528)];
    kernel_shared[(((int)threadIdx.x) + 1536)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 24576)];
    kernel_shared[(((int)threadIdx.x) + 1664)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 26624)];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 28672)];
    kernel_shared[(((int)threadIdx.x) + 1920)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 30720)];
    kernel_shared[(((int)threadIdx.x) + 2048)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 32768)];
    kernel_shared[(((int)threadIdx.x) + 2176)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 34816)];
    kernel_shared[(((int)threadIdx.x) + 2304)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 36864)];
    kernel_shared[(((int)threadIdx.x) + 2432)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 38912)];
    kernel_shared[(((int)threadIdx.x) + 2560)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 40960)];
    kernel_shared[(((int)threadIdx.x) + 2688)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 43008)];
    kernel_shared[(((int)threadIdx.x) + 2816)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 45056)];
    kernel_shared[(((int)threadIdx.x) + 2944)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 47104)];
    kernel_shared[(((int)threadIdx.x) + 3072)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 49152)];
    kernel_shared[(((int)threadIdx.x) + 3200)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 51200)];
    kernel_shared[(((int)threadIdx.x) + 3328)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 53248)];
    kernel_shared[(((int)threadIdx.x) + 3456)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 55296)];
    kernel_shared[(((int)threadIdx.x) + 3584)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 57344)];
    kernel_shared[(((int)threadIdx.x) + 3712)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 59392)];
    kernel_shared[(((int)threadIdx.x) + 3840)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 61440)];
    kernel_shared[(((int)threadIdx.x) + 3968)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 63488)];
    kernel_shared[(((int)threadIdx.x) + 4096)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 65536)];
    kernel_shared[(((int)threadIdx.x) + 4224)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 67584)];
    kernel_shared[(((int)threadIdx.x) + 4352)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 69632)];
    kernel_shared[(((int)threadIdx.x) + 4480)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 71680)];
    kernel_shared[(((int)threadIdx.x) + 4608)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 73728)];
    kernel_shared[(((int)threadIdx.x) + 4736)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 75776)];
    kernel_shared[(((int)threadIdx.x) + 4864)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 77824)];
    kernel_shared[(((int)threadIdx.x) + 4992)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 79872)];
    kernel_shared[(((int)threadIdx.x) + 5120)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 81920)];
    kernel_shared[(((int)threadIdx.x) + 5248)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 83968)];
    kernel_shared[(((int)threadIdx.x) + 5376)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 86016)];
    kernel_shared[(((int)threadIdx.x) + 5504)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 88064)];
    kernel_shared[(((int)threadIdx.x) + 5632)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 90112)];
    kernel_shared[(((int)threadIdx.x) + 5760)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 92160)];
    kernel_shared[(((int)threadIdx.x) + 5888)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 94208)];
    kernel_shared[(((int)threadIdx.x) + 6016)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 96256)];
    kernel_shared[(((int)threadIdx.x) + 6144)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 98304)];
    kernel_shared[(((int)threadIdx.x) + 6272)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 100352)];
    kernel_shared[(((int)threadIdx.x) + 6400)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 102400)];
    kernel_shared[(((int)threadIdx.x) + 6528)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 104448)];
    kernel_shared[(((int)threadIdx.x) + 6656)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 106496)];
    kernel_shared[(((int)threadIdx.x) + 6784)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 108544)];
    kernel_shared[(((int)threadIdx.x) + 6912)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 110592)];
    kernel_shared[(((int)threadIdx.x) + 7040)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 112640)];
    kernel_shared[(((int)threadIdx.x) + 7168)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 114688)];
    kernel_shared[(((int)threadIdx.x) + 7296)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 116736)];
    kernel_shared[(((int)threadIdx.x) + 7424)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 118784)];
    kernel_shared[(((int)threadIdx.x) + 7552)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 120832)];
    kernel_shared[(((int)threadIdx.x) + 7680)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 122880)];
    kernel_shared[(((int)threadIdx.x) + 7808)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 124928)];
    kernel_shared[(((int)threadIdx.x) + 7936)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 126976)];
    kernel_shared[(((int)threadIdx.x) + 8064)] = kernel[(((((((int)threadIdx.x) >> 5) * 512) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 129024)];
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 4; ++rc_outer_inner) {
      for (int yy_c_outer_inner = 0; yy_c_outer_inner < 2; ++yy_c_outer_inner) {
        for (int xx_c_outer_inner = 0; xx_c_outer_inner < 2; ++xx_c_outer_inner) {
          conv2d_nchw_local[((yy_c_outer_inner * 2) + xx_c_outer_inner)] = (conv2d_nchw_local[((yy_c_outer_inner * 2) + xx_c_outer_inner)] + (pad_temp_shared[(((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8))]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 4)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 4)] + (pad_temp_shared[(((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 32)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 8)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 8)] + (pad_temp_shared[(((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 64)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 12)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 12)] + (pad_temp_shared[(((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 96)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 16)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 16)] + (pad_temp_shared[(((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 128)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 20)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 20)] + (pad_temp_shared[(((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 160)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 24)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 24)] + (pad_temp_shared[(((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 192)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 28)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 28)] + (pad_temp_shared[(((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 224)]));
          conv2d_nchw_local[((yy_c_outer_inner * 2) + xx_c_outer_inner)] = (conv2d_nchw_local[((yy_c_outer_inner * 2) + xx_c_outer_inner)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 16)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 1)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 4)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 4)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 16)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 33)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 8)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 8)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 16)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 65)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 12)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 12)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 16)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 97)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 16)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 16)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 16)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 129)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 20)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 20)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 16)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 161)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 24)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 24)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 16)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 193)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 28)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 28)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 16)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 225)]));
          conv2d_nchw_local[((yy_c_outer_inner * 2) + xx_c_outer_inner)] = (conv2d_nchw_local[((yy_c_outer_inner * 2) + xx_c_outer_inner)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 32)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 2)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 4)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 4)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 32)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 34)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 8)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 8)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 32)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 66)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 12)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 12)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 32)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 98)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 16)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 16)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 32)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 130)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 20)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 20)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 32)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 162)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 24)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 24)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 32)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 194)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 28)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 28)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 32)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 226)]));
          conv2d_nchw_local[((yy_c_outer_inner * 2) + xx_c_outer_inner)] = (conv2d_nchw_local[((yy_c_outer_inner * 2) + xx_c_outer_inner)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 48)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 3)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 4)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 4)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 48)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 35)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 8)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 8)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 48)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 67)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 12)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 12)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 48)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 99)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 16)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 16)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 48)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 131)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 20)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 20)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 48)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 163)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 24)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 24)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 48)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 195)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 28)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 28)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 48)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 227)]));
          conv2d_nchw_local[((yy_c_outer_inner * 2) + xx_c_outer_inner)] = (conv2d_nchw_local[((yy_c_outer_inner * 2) + xx_c_outer_inner)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 64)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 4)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 4)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 4)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 64)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 36)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 8)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 8)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 64)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 68)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 12)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 12)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 64)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 100)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 16)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 16)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 64)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 132)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 20)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 20)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 64)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 164)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 24)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 24)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 64)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 196)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 28)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 28)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 64)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 228)]));
          conv2d_nchw_local[((yy_c_outer_inner * 2) + xx_c_outer_inner)] = (conv2d_nchw_local[((yy_c_outer_inner * 2) + xx_c_outer_inner)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 80)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 5)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 4)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 4)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 80)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 37)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 8)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 8)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 80)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 69)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 12)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 12)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 80)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 101)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 16)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 16)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 80)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 133)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 20)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 20)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 80)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 165)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 24)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 24)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 80)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 197)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 28)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 28)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 80)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 229)]));
          conv2d_nchw_local[((yy_c_outer_inner * 2) + xx_c_outer_inner)] = (conv2d_nchw_local[((yy_c_outer_inner * 2) + xx_c_outer_inner)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 96)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 6)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 4)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 4)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 96)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 38)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 8)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 8)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 96)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 70)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 12)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 12)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 96)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 102)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 16)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 16)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 96)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 134)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 20)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 20)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 96)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 166)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 24)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 24)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 96)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 198)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 28)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 28)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 96)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 230)]));
          conv2d_nchw_local[((yy_c_outer_inner * 2) + xx_c_outer_inner)] = (conv2d_nchw_local[((yy_c_outer_inner * 2) + xx_c_outer_inner)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 112)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 7)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 4)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 4)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 112)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 39)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 8)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 8)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 112)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 71)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 12)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 12)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 112)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 103)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 16)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 16)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 112)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 135)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 20)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 20)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 112)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 167)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 24)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 24)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 112)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 199)]));
          conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 28)] = (conv2d_nchw_local[(((yy_c_outer_inner * 2) + xx_c_outer_inner) + 28)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (((((int)threadIdx.x) & 3) >> 1) * 8)) + (yy_c_outer_inner * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_outer_inner) + 112)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 256) + (rc_outer_inner * 8)) + 231)]));
        }
      }
    }
  }
  for (int ff_inner = 0; ff_inner < 8; ++ff_inner) {
    for (int yy_inner = 0; yy_inner < 2; ++yy_inner) {
      for (int xx_inner = 0; xx_inner < 2; ++xx_inner) {
        conv2d_nchw[(((((((((((int)threadIdx.x) >> 2) * 6272) + (ff_inner * 784)) + ((((int)blockIdx.x) / 7) * 112)) + (((((int)threadIdx.x) & 3) >> 1) * 56)) + (yy_inner * 28)) + ((((int)blockIdx.x) % 7) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_inner)] = conv2d_nchw_local[(((ff_inner * 4) + (yy_inner * 2)) + xx_inner)];
      }
    }
  }
}


