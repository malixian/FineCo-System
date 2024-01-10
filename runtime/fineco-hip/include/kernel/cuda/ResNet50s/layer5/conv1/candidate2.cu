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
  float conv2d_nchw_local[16];
  __shared__ float pad_temp_shared[1792];
  __shared__ float kernel_shared[8192];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  conv2d_nchw_local[7] = 0.000000e+00f;
  conv2d_nchw_local[8] = 0.000000e+00f;
  conv2d_nchw_local[9] = 0.000000e+00f;
  conv2d_nchw_local[10] = 0.000000e+00f;
  conv2d_nchw_local[11] = 0.000000e+00f;
  conv2d_nchw_local[12] = 0.000000e+00f;
  conv2d_nchw_local[13] = 0.000000e+00f;
  conv2d_nchw_local[14] = 0.000000e+00f;
  conv2d_nchw_local[15] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 16; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = data[((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28))];
    pad_temp_shared[(((int)threadIdx.x) + 224)] = data[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 1568)];
    pad_temp_shared[(((int)threadIdx.x) + 448)] = data[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 3136)];
    pad_temp_shared[(((int)threadIdx.x) + 672)] = data[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 4704)];
    pad_temp_shared[(((int)threadIdx.x) + 896)] = data[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 6272)];
    pad_temp_shared[(((int)threadIdx.x) + 1120)] = data[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 7840)];
    pad_temp_shared[(((int)threadIdx.x) + 1344)] = data[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 9408)];
    pad_temp_shared[(((int)threadIdx.x) + 1568)] = data[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 10976)];
    kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63))];
    kernel_shared[(((int)threadIdx.x) + 224)] = kernel[(((((((int)blockIdx.x) / 7) * 131072) + (((((int)threadIdx.x) + 224) >> 6) * 1024)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 32) & 63))];
    kernel_shared[(((int)threadIdx.x) + 448)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 7168)];
    kernel_shared[(((int)threadIdx.x) + 672)] = kernel[(((((((int)blockIdx.x) / 7) * 131072) + (((((int)threadIdx.x) + 672) >> 6) * 1024)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 32) & 63))];
    kernel_shared[(((int)threadIdx.x) + 896)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 14336)];
    kernel_shared[(((int)threadIdx.x) + 1120)] = kernel[(((((((int)blockIdx.x) / 7) * 131072) + (((((int)threadIdx.x) + 1120) >> 6) * 1024)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 32) & 63))];
    kernel_shared[(((int)threadIdx.x) + 1344)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 21504)];
    kernel_shared[(((int)threadIdx.x) + 1568)] = kernel[(((((((int)blockIdx.x) / 7) * 131072) + (((((int)threadIdx.x) + 1568) >> 6) * 1024)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 32) & 63))];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 28672)];
    kernel_shared[(((int)threadIdx.x) + 2016)] = kernel[(((((((int)blockIdx.x) / 7) * 131072) + (((((int)threadIdx.x) + 2016) >> 6) * 1024)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 32) & 63))];
    kernel_shared[(((int)threadIdx.x) + 2240)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 35840)];
    kernel_shared[(((int)threadIdx.x) + 2464)] = kernel[(((((((int)blockIdx.x) / 7) * 131072) + (((((int)threadIdx.x) + 2464) >> 6) * 1024)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 32) & 63))];
    kernel_shared[(((int)threadIdx.x) + 2688)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 43008)];
    kernel_shared[(((int)threadIdx.x) + 2912)] = kernel[(((((((int)blockIdx.x) / 7) * 131072) + (((((int)threadIdx.x) + 2912) >> 6) * 1024)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 32) & 63))];
    kernel_shared[(((int)threadIdx.x) + 3136)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 50176)];
    kernel_shared[(((int)threadIdx.x) + 3360)] = kernel[(((((((int)blockIdx.x) / 7) * 131072) + (((((int)threadIdx.x) + 3360) >> 6) * 1024)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 32) & 63))];
    kernel_shared[(((int)threadIdx.x) + 3584)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 57344)];
    kernel_shared[(((int)threadIdx.x) + 3808)] = kernel[(((((((int)blockIdx.x) / 7) * 131072) + (((((int)threadIdx.x) + 3808) >> 6) * 1024)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 32) & 63))];
    kernel_shared[(((int)threadIdx.x) + 4032)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 64512)];
    kernel_shared[(((int)threadIdx.x) + 4256)] = kernel[(((((((int)blockIdx.x) / 7) * 131072) + (((((int)threadIdx.x) + 4256) >> 6) * 1024)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 32) & 63))];
    kernel_shared[(((int)threadIdx.x) + 4480)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 71680)];
    kernel_shared[(((int)threadIdx.x) + 4704)] = kernel[(((((((int)blockIdx.x) / 7) * 131072) + (((((int)threadIdx.x) + 4704) >> 6) * 1024)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 32) & 63))];
    kernel_shared[(((int)threadIdx.x) + 4928)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 78848)];
    kernel_shared[(((int)threadIdx.x) + 5152)] = kernel[(((((((int)blockIdx.x) / 7) * 131072) + (((((int)threadIdx.x) + 5152) >> 6) * 1024)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 32) & 63))];
    kernel_shared[(((int)threadIdx.x) + 5376)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 86016)];
    kernel_shared[(((int)threadIdx.x) + 5600)] = kernel[(((((((int)blockIdx.x) / 7) * 131072) + (((((int)threadIdx.x) + 5600) >> 6) * 1024)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 32) & 63))];
    kernel_shared[(((int)threadIdx.x) + 5824)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 93184)];
    kernel_shared[(((int)threadIdx.x) + 6048)] = kernel[(((((((int)blockIdx.x) / 7) * 131072) + (((((int)threadIdx.x) + 6048) >> 6) * 1024)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 32) & 63))];
    kernel_shared[(((int)threadIdx.x) + 6272)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 100352)];
    kernel_shared[(((int)threadIdx.x) + 6496)] = kernel[(((((((int)blockIdx.x) / 7) * 131072) + (((((int)threadIdx.x) + 6496) >> 6) * 1024)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 32) & 63))];
    kernel_shared[(((int)threadIdx.x) + 6720)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 107520)];
    kernel_shared[(((int)threadIdx.x) + 6944)] = kernel[(((((((int)blockIdx.x) / 7) * 131072) + (((((int)threadIdx.x) + 6944) >> 6) * 1024)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 32) & 63))];
    kernel_shared[(((int)threadIdx.x) + 7168)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 114688)];
    kernel_shared[(((int)threadIdx.x) + 7392)] = kernel[(((((((int)blockIdx.x) / 7) * 131072) + (((((int)threadIdx.x) + 7392) >> 6) * 1024)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 32) & 63))];
    kernel_shared[(((int)threadIdx.x) + 7616)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 121856)];
    kernel_shared[(((int)threadIdx.x) + 7840)] = kernel[(((((((int)blockIdx.x) / 7) * 131072) + (((((int)threadIdx.x) + 7840) >> 6) * 1024)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 32) & 63))];
    if (((int)threadIdx.x) < 128) {
      kernel_shared[(((int)threadIdx.x) + 8064)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 129024)];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 8; ++rc_outer_inner) {
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2))] * kernel_shared[(((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8))]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 1)] * kernel_shared[(((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8))]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 1)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 29)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 1)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 56)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 2)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 57)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 2)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 84)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 3)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 85)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 3)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 112)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 4)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 113)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 4)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 140)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 5)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 141)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 5)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 168)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 6)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 169)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 6)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 196)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 7)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 197)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 7)]));
      conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 14)] * kernel_shared[(((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8))]));
      conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 15)] * kernel_shared[(((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8))]));
      conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 42)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 1)]));
      conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 43)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 1)]));
      conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 70)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 2)]));
      conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 71)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 2)]));
      conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 98)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 3)]));
      conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 99)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 3)]));
      conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 126)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 4)]));
      conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 127)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 4)]));
      conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 154)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 5)]));
      conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 155)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 5)]));
      conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 182)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 6)]));
      conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 183)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 6)]));
      conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 210)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 7)]));
      conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 211)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 7)]));
      conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2))] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 64)]));
      conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 1)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 64)]));
      conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 65)]));
      conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 29)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 65)]));
      conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 56)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 66)]));
      conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 57)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 66)]));
      conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 84)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 67)]));
      conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 85)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 67)]));
      conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 112)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 68)]));
      conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 113)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 68)]));
      conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 140)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 69)]));
      conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 141)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 69)]));
      conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 168)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 70)]));
      conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 169)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 70)]));
      conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 196)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 71)]));
      conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 197)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 71)]));
      conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 64)]));
      conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 15)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 64)]));
      conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 42)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 65)]));
      conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 43)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 65)]));
      conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 70)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 66)]));
      conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 71)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 66)]));
      conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 98)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 67)]));
      conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 99)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 67)]));
      conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 126)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 68)]));
      conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 127)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 68)]));
      conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 154)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 69)]));
      conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 155)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 69)]));
      conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 182)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 70)]));
      conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 183)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 70)]));
      conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 210)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 71)]));
      conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 211)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 71)]));
      conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2))] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 128)]));
      conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 1)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 128)]));
      conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 129)]));
      conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 29)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 129)]));
      conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 56)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 130)]));
      conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 57)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 130)]));
      conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 84)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 131)]));
      conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 85)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 131)]));
      conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 112)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 132)]));
      conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 113)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 132)]));
      conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 140)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 133)]));
      conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 141)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 133)]));
      conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 168)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 134)]));
      conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 169)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 134)]));
      conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 196)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 135)]));
      conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 197)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 135)]));
      conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 128)]));
      conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 15)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 128)]));
      conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 42)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 129)]));
      conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 43)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 129)]));
      conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 70)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 130)]));
      conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 71)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 130)]));
      conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 98)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 131)]));
      conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 99)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 131)]));
      conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 126)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 132)]));
      conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 127)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 132)]));
      conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 154)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 133)]));
      conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 155)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 133)]));
      conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 182)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 134)]));
      conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 183)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 134)]));
      conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 210)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 135)]));
      conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 211)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 135)]));
      conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2))] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 192)]));
      conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 1)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 192)]));
      conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 193)]));
      conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 29)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 193)]));
      conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 56)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 194)]));
      conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 57)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 194)]));
      conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 84)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 195)]));
      conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 85)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 195)]));
      conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 112)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 196)]));
      conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 113)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 196)]));
      conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 140)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 197)]));
      conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 141)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 197)]));
      conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 168)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 198)]));
      conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 169)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 198)]));
      conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 196)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 199)]));
      conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 197)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 199)]));
      conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 192)]));
      conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 15)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 192)]));
      conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 42)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 193)]));
      conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 43)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 193)]));
      conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 70)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 194)]));
      conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 71)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 194)]));
      conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 98)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 195)]));
      conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 99)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 195)]));
      conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 126)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 196)]));
      conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 127)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 196)]));
      conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 154)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 197)]));
      conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 155)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 197)]));
      conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 182)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 198)]));
      conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 183)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 198)]));
      conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 210)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 199)]));
      conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[(((rc_outer_inner * 224) + ((((int)threadIdx.x) % 7) * 2)) + 211)] * kernel_shared[((((((int)threadIdx.x) / 7) * 256) + (rc_outer_inner * 8)) + 199)]));
    }
  }
  for (int ff_inner = 0; ff_inner < 4; ++ff_inner) {
    for (int yy_inner = 0; yy_inner < 2; ++yy_inner) {
      for (int xx_inner = 0; xx_inner < 2; ++xx_inner) {
        conv2d_nchw[((((((((((int)blockIdx.x) / 7) * 25088) + ((((int)threadIdx.x) / 7) * 784)) + (ff_inner * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (yy_inner * 14)) + ((((int)threadIdx.x) % 7) * 2)) + xx_inner)] = conv2d_nchw_local[(((ff_inner * 4) + (yy_inner * 2)) + xx_inner)];
      }
    }
  }
}


