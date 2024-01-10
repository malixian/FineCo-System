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
extern "C" __global__ void __launch_bounds__(128) candidate0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[8];
  __shared__ float pad_temp_shared[2048];
  __shared__ float kernel_shared[8192];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[7] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 4; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = data[((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 4) * 784)) + (((((int)blockIdx.x) % 49) / 7) * 112)) + (((((int)threadIdx.x) & 15) >> 2) * 28)) + ((((int)blockIdx.x) % 7) * 4)) + (((int)threadIdx.x) & 3))];
    pad_temp_shared[(((int)threadIdx.x) + 128)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 4) * 784)) + (((((int)blockIdx.x) % 49) / 7) * 112)) + (((((int)threadIdx.x) & 15) >> 2) * 28)) + ((((int)blockIdx.x) % 7) * 4)) + (((int)threadIdx.x) & 3)) + 6272)];
    pad_temp_shared[(((int)threadIdx.x) + 256)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 4) * 784)) + (((((int)blockIdx.x) % 49) / 7) * 112)) + (((((int)threadIdx.x) & 15) >> 2) * 28)) + ((((int)blockIdx.x) % 7) * 4)) + (((int)threadIdx.x) & 3)) + 12544)];
    pad_temp_shared[(((int)threadIdx.x) + 384)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 4) * 784)) + (((((int)blockIdx.x) % 49) / 7) * 112)) + (((((int)threadIdx.x) & 15) >> 2) * 28)) + ((((int)blockIdx.x) % 7) * 4)) + (((int)threadIdx.x) & 3)) + 18816)];
    pad_temp_shared[(((int)threadIdx.x) + 512)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 4) * 784)) + (((((int)blockIdx.x) % 49) / 7) * 112)) + (((((int)threadIdx.x) & 15) >> 2) * 28)) + ((((int)blockIdx.x) % 7) * 4)) + (((int)threadIdx.x) & 3)) + 25088)];
    pad_temp_shared[(((int)threadIdx.x) + 640)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 4) * 784)) + (((((int)blockIdx.x) % 49) / 7) * 112)) + (((((int)threadIdx.x) & 15) >> 2) * 28)) + ((((int)blockIdx.x) % 7) * 4)) + (((int)threadIdx.x) & 3)) + 31360)];
    pad_temp_shared[(((int)threadIdx.x) + 768)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 4) * 784)) + (((((int)blockIdx.x) % 49) / 7) * 112)) + (((((int)threadIdx.x) & 15) >> 2) * 28)) + ((((int)blockIdx.x) % 7) * 4)) + (((int)threadIdx.x) & 3)) + 37632)];
    pad_temp_shared[(((int)threadIdx.x) + 896)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 4) * 784)) + (((((int)blockIdx.x) % 49) / 7) * 112)) + (((((int)threadIdx.x) & 15) >> 2) * 28)) + ((((int)blockIdx.x) % 7) * 4)) + (((int)threadIdx.x) & 3)) + 43904)];
    pad_temp_shared[(((int)threadIdx.x) + 1024)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 4) * 784)) + (((((int)blockIdx.x) % 49) / 7) * 112)) + (((((int)threadIdx.x) & 15) >> 2) * 28)) + ((((int)blockIdx.x) % 7) * 4)) + (((int)threadIdx.x) & 3)) + 50176)];
    pad_temp_shared[(((int)threadIdx.x) + 1152)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 4) * 784)) + (((((int)blockIdx.x) % 49) / 7) * 112)) + (((((int)threadIdx.x) & 15) >> 2) * 28)) + ((((int)blockIdx.x) % 7) * 4)) + (((int)threadIdx.x) & 3)) + 56448)];
    pad_temp_shared[(((int)threadIdx.x) + 1280)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 4) * 784)) + (((((int)blockIdx.x) % 49) / 7) * 112)) + (((((int)threadIdx.x) & 15) >> 2) * 28)) + ((((int)blockIdx.x) % 7) * 4)) + (((int)threadIdx.x) & 3)) + 62720)];
    pad_temp_shared[(((int)threadIdx.x) + 1408)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 4) * 784)) + (((((int)blockIdx.x) % 49) / 7) * 112)) + (((((int)threadIdx.x) & 15) >> 2) * 28)) + ((((int)blockIdx.x) % 7) * 4)) + (((int)threadIdx.x) & 3)) + 68992)];
    pad_temp_shared[(((int)threadIdx.x) + 1536)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 4) * 784)) + (((((int)blockIdx.x) % 49) / 7) * 112)) + (((((int)threadIdx.x) & 15) >> 2) * 28)) + ((((int)blockIdx.x) % 7) * 4)) + (((int)threadIdx.x) & 3)) + 75264)];
    pad_temp_shared[(((int)threadIdx.x) + 1664)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 4) * 784)) + (((((int)blockIdx.x) % 49) / 7) * 112)) + (((((int)threadIdx.x) & 15) >> 2) * 28)) + ((((int)blockIdx.x) % 7) * 4)) + (((int)threadIdx.x) & 3)) + 81536)];
    pad_temp_shared[(((int)threadIdx.x) + 1792)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 4) * 784)) + (((((int)blockIdx.x) % 49) / 7) * 112)) + (((((int)threadIdx.x) & 15) >> 2) * 28)) + ((((int)blockIdx.x) % 7) * 4)) + (((int)threadIdx.x) & 3)) + 87808)];
    pad_temp_shared[(((int)threadIdx.x) + 1920)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 4) * 784)) + (((((int)blockIdx.x) % 49) / 7) * 112)) + (((((int)threadIdx.x) & 15) >> 2) * 28)) + ((((int)blockIdx.x) % 7) * 4)) + (((int)threadIdx.x) & 3)) + 94080)];
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x))];
    kernel_shared[(((int)threadIdx.x) + 128)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 512)];
    kernel_shared[(((int)threadIdx.x) + 256)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 1024)];
    kernel_shared[(((int)threadIdx.x) + 384)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 1536)];
    kernel_shared[(((int)threadIdx.x) + 512)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 2048)];
    kernel_shared[(((int)threadIdx.x) + 640)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 2560)];
    kernel_shared[(((int)threadIdx.x) + 768)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 3072)];
    kernel_shared[(((int)threadIdx.x) + 896)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 3584)];
    kernel_shared[(((int)threadIdx.x) + 1024)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 4096)];
    kernel_shared[(((int)threadIdx.x) + 1152)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 4608)];
    kernel_shared[(((int)threadIdx.x) + 1280)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 5120)];
    kernel_shared[(((int)threadIdx.x) + 1408)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 5632)];
    kernel_shared[(((int)threadIdx.x) + 1536)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 6144)];
    kernel_shared[(((int)threadIdx.x) + 1664)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 6656)];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 7168)];
    kernel_shared[(((int)threadIdx.x) + 1920)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 7680)];
    kernel_shared[(((int)threadIdx.x) + 2048)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 8192)];
    kernel_shared[(((int)threadIdx.x) + 2176)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 8704)];
    kernel_shared[(((int)threadIdx.x) + 2304)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 9216)];
    kernel_shared[(((int)threadIdx.x) + 2432)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 9728)];
    kernel_shared[(((int)threadIdx.x) + 2560)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 10240)];
    kernel_shared[(((int)threadIdx.x) + 2688)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 10752)];
    kernel_shared[(((int)threadIdx.x) + 2816)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 11264)];
    kernel_shared[(((int)threadIdx.x) + 2944)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 11776)];
    kernel_shared[(((int)threadIdx.x) + 3072)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 12288)];
    kernel_shared[(((int)threadIdx.x) + 3200)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 12800)];
    kernel_shared[(((int)threadIdx.x) + 3328)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 13312)];
    kernel_shared[(((int)threadIdx.x) + 3456)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 13824)];
    kernel_shared[(((int)threadIdx.x) + 3584)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 14336)];
    kernel_shared[(((int)threadIdx.x) + 3712)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 14848)];
    kernel_shared[(((int)threadIdx.x) + 3840)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 15360)];
    kernel_shared[(((int)threadIdx.x) + 3968)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 15872)];
    kernel_shared[(((int)threadIdx.x) + 4096)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 16384)];
    kernel_shared[(((int)threadIdx.x) + 4224)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 16896)];
    kernel_shared[(((int)threadIdx.x) + 4352)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 17408)];
    kernel_shared[(((int)threadIdx.x) + 4480)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 17920)];
    kernel_shared[(((int)threadIdx.x) + 4608)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 18432)];
    kernel_shared[(((int)threadIdx.x) + 4736)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 18944)];
    kernel_shared[(((int)threadIdx.x) + 4864)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 19456)];
    kernel_shared[(((int)threadIdx.x) + 4992)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 19968)];
    kernel_shared[(((int)threadIdx.x) + 5120)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 20480)];
    kernel_shared[(((int)threadIdx.x) + 5248)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 20992)];
    kernel_shared[(((int)threadIdx.x) + 5376)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 21504)];
    kernel_shared[(((int)threadIdx.x) + 5504)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 22016)];
    kernel_shared[(((int)threadIdx.x) + 5632)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 22528)];
    kernel_shared[(((int)threadIdx.x) + 5760)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 23040)];
    kernel_shared[(((int)threadIdx.x) + 5888)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 23552)];
    kernel_shared[(((int)threadIdx.x) + 6016)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 24064)];
    kernel_shared[(((int)threadIdx.x) + 6144)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 24576)];
    kernel_shared[(((int)threadIdx.x) + 6272)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 25088)];
    kernel_shared[(((int)threadIdx.x) + 6400)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 25600)];
    kernel_shared[(((int)threadIdx.x) + 6528)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 26112)];
    kernel_shared[(((int)threadIdx.x) + 6656)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 26624)];
    kernel_shared[(((int)threadIdx.x) + 6784)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 27136)];
    kernel_shared[(((int)threadIdx.x) + 6912)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 27648)];
    kernel_shared[(((int)threadIdx.x) + 7040)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 28160)];
    kernel_shared[(((int)threadIdx.x) + 7168)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 28672)];
    kernel_shared[(((int)threadIdx.x) + 7296)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 29184)];
    kernel_shared[(((int)threadIdx.x) + 7424)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 29696)];
    kernel_shared[(((int)threadIdx.x) + 7552)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 30208)];
    kernel_shared[(((int)threadIdx.x) + 7680)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 30720)];
    kernel_shared[(((int)threadIdx.x) + 7808)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 31232)];
    kernel_shared[(((int)threadIdx.x) + 7936)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 31744)];
    kernel_shared[(((int)threadIdx.x) + 8064)] = kernel[(((((((int)blockIdx.x) / 49) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 32256)];
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 16; ++rc_outer_inner) {
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((rc_outer_inner * 128) + (((int)threadIdx.x) & 3))] * kernel_shared[(((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8))]));
      conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 8)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8))]));
      conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[((rc_outer_inner * 128) + (((int)threadIdx.x) & 3))] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 4096)]));
      conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 8)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 4096)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 16)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 1)]));
      conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 24)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 1)]));
      conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 16)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 4097)]));
      conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 24)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 4097)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 32)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 2)]));
      conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 40)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 2)]));
      conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 32)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 4098)]));
      conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 40)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 4098)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 48)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 3)]));
      conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 56)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 3)]));
      conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 48)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 4099)]));
      conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 56)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 4099)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 64)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 4)]));
      conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 72)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 4)]));
      conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 64)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 4100)]));
      conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 72)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 4100)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 80)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 5)]));
      conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 88)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 5)]));
      conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 80)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 4101)]));
      conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 88)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 4101)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 96)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 6)]));
      conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 104)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 6)]));
      conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 96)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 4102)]));
      conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 104)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 4102)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 112)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 7)]));
      conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 120)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 7)]));
      conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 112)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 4103)]));
      conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 120)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 4103)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 4)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8))]));
      conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 12)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8))]));
      conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 4)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 4096)]));
      conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 12)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 4096)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 20)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 1)]));
      conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 28)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 1)]));
      conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 20)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 4097)]));
      conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 28)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 4097)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 36)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 2)]));
      conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 44)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 2)]));
      conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 36)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 4098)]));
      conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 44)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 4098)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 52)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 3)]));
      conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 60)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 3)]));
      conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 52)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 4099)]));
      conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 60)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 4099)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 68)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 4)]));
      conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 76)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 4)]));
      conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 68)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 4100)]));
      conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 76)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 4100)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 84)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 5)]));
      conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 92)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 5)]));
      conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 84)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 4101)]));
      conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 92)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 4101)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 100)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 6)]));
      conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 108)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 6)]));
      conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 100)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 4102)]));
      conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 108)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 4102)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 116)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 7)]));
      conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 124)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 7)]));
      conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 116)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 4103)]));
      conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((rc_outer_inner * 128) + (((int)threadIdx.x) & 3)) + 124)] * kernel_shared[((((((int)threadIdx.x) >> 2) * 128) + (rc_outer_inner * 8)) + 4103)]));
    }
  }
  for (int yy_inner = 0; yy_inner < 2; ++yy_inner) {
    conv2d_nchw[(((((((((int)blockIdx.x) / 49) * 50176) + ((((int)threadIdx.x) >> 2) * 784)) + (((((int)blockIdx.x) % 49) / 7) * 112)) + (yy_inner * 28)) + ((((int)blockIdx.x) % 7) * 4)) + (((int)threadIdx.x) & 3))] = conv2d_nchw_local[yy_inner];
    conv2d_nchw[((((((((((int)blockIdx.x) / 49) * 50176) + ((((int)threadIdx.x) >> 2) * 784)) + (((((int)blockIdx.x) % 49) / 7) * 112)) + (yy_inner * 28)) + ((((int)blockIdx.x) % 7) * 4)) + (((int)threadIdx.x) & 3)) + 56)] = conv2d_nchw_local[(yy_inner + 2)];
    conv2d_nchw[((((((((((int)blockIdx.x) / 49) * 50176) + ((((int)threadIdx.x) >> 2) * 784)) + (((((int)blockIdx.x) % 49) / 7) * 112)) + (yy_inner * 28)) + ((((int)blockIdx.x) % 7) * 4)) + (((int)threadIdx.x) & 3)) + 25088)] = conv2d_nchw_local[(yy_inner + 4)];
    conv2d_nchw[((((((((((int)blockIdx.x) / 49) * 50176) + ((((int)threadIdx.x) >> 2) * 784)) + (((((int)blockIdx.x) % 49) / 7) * 112)) + (yy_inner * 28)) + ((((int)blockIdx.x) % 7) * 4)) + (((int)threadIdx.x) & 3)) + 25144)] = conv2d_nchw_local[(yy_inner + 6)];
  }
}


