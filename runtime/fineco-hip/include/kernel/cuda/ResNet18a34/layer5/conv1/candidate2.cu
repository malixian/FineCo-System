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
extern "C" __global__ void __launch_bounds__(112) candidate2(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[7];
  __shared__ float pad_temp_shared[1296];
  __shared__ float kernel_shared[2304];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 32; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = (((((9 <= (((int)threadIdx.x) % 81)) && ((((int)threadIdx.x) % 81) < 72)) && (1 <= (((int)threadIdx.x) % 9))) && ((((int)threadIdx.x) % 9) < 8)) ? data[(((((rc_outer_outer * 784) + ((((int)threadIdx.x) / 81) * 49)) + (((((int)threadIdx.x) % 81) / 9) * 7)) + (((int)threadIdx.x) % 9)) - 8)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 112)] = (((((9 <= ((((int)threadIdx.x) + 31) % 81)) && (((((int)threadIdx.x) + 31) % 81) < 72)) && (1 <= ((((int)threadIdx.x) + 4) % 9))) && (((((int)threadIdx.x) + 4) % 9) < 8)) ? data[(((((rc_outer_outer * 784) + (((((int)threadIdx.x) + 112) / 81) * 49)) + ((((((int)threadIdx.x) + 31) % 81) / 9) * 7)) + ((((int)threadIdx.x) + 4) % 9)) - 8)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 224)] = (((((9 <= ((((int)threadIdx.x) + 62) % 81)) && (((((int)threadIdx.x) + 62) % 81) < 72)) && (1 <= ((((int)threadIdx.x) + 8) % 9))) && (((((int)threadIdx.x) + 8) % 9) < 8)) ? data[(((((rc_outer_outer * 784) + (((((int)threadIdx.x) + 224) / 81) * 49)) + ((((((int)threadIdx.x) + 62) % 81) / 9) * 7)) + ((((int)threadIdx.x) + 8) % 9)) - 8)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 336)] = (((((9 <= ((((int)threadIdx.x) + 12) % 81)) && (((((int)threadIdx.x) + 12) % 81) < 72)) && (1 <= ((((int)threadIdx.x) + 3) % 9))) && (((((int)threadIdx.x) + 3) % 9) < 8)) ? data[(((((rc_outer_outer * 784) + (((((int)threadIdx.x) + 336) / 81) * 49)) + ((((((int)threadIdx.x) + 12) % 81) / 9) * 7)) + ((((int)threadIdx.x) + 3) % 9)) - 8)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 448)] = (((((9 <= ((((int)threadIdx.x) + 43) % 81)) && (((((int)threadIdx.x) + 43) % 81) < 72)) && (1 <= ((((int)threadIdx.x) + 7) % 9))) && (((((int)threadIdx.x) + 7) % 9) < 8)) ? data[(((((rc_outer_outer * 784) + (((((int)threadIdx.x) + 448) / 81) * 49)) + ((((((int)threadIdx.x) + 43) % 81) / 9) * 7)) + ((((int)threadIdx.x) + 7) % 9)) - 8)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 560)] = (((((9 <= ((((int)threadIdx.x) + 74) % 81)) && (((((int)threadIdx.x) + 74) % 81) < 72)) && (1 <= ((((int)threadIdx.x) + 2) % 9))) && (((((int)threadIdx.x) + 2) % 9) < 8)) ? data[(((((rc_outer_outer * 784) + (((((int)threadIdx.x) + 560) / 81) * 49)) + ((((((int)threadIdx.x) + 74) % 81) / 9) * 7)) + ((((int)threadIdx.x) + 2) % 9)) - 8)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 672)] = (((((9 <= ((((int)threadIdx.x) + 24) % 81)) && (((((int)threadIdx.x) + 24) % 81) < 72)) && (1 <= ((((int)threadIdx.x) + 6) % 9))) && (((((int)threadIdx.x) + 6) % 9) < 8)) ? data[(((((rc_outer_outer * 784) + (((((int)threadIdx.x) + 672) / 81) * 49)) + ((((((int)threadIdx.x) + 24) % 81) / 9) * 7)) + ((((int)threadIdx.x) + 6) % 9)) - 8)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 784)] = (((((9 <= ((((int)threadIdx.x) + 55) % 81)) && (((((int)threadIdx.x) + 55) % 81) < 72)) && (1 <= ((((int)threadIdx.x) + 1) % 9))) && (((((int)threadIdx.x) + 1) % 9) < 8)) ? data[(((((rc_outer_outer * 784) + (((((int)threadIdx.x) + 784) / 81) * 49)) + ((((((int)threadIdx.x) + 55) % 81) / 9) * 7)) + ((((int)threadIdx.x) + 1) % 9)) - 8)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 896)] = (((((9 <= ((((int)threadIdx.x) + 5) % 81)) && (((((int)threadIdx.x) + 5) % 81) < 72)) && (1 <= ((((int)threadIdx.x) + 5) % 9))) && (((((int)threadIdx.x) + 5) % 9) < 8)) ? data[(((((rc_outer_outer * 784) + (((((int)threadIdx.x) + 896) / 81) * 49)) + ((((((int)threadIdx.x) + 5) % 81) / 9) * 7)) + ((((int)threadIdx.x) + 5) % 9)) - 8)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1008)] = (((((1 <= (((((int)threadIdx.x) / 9) + 4) % 9)) && (((((int)threadIdx.x) + 36) % 81) < 72)) && (1 <= (((int)threadIdx.x) % 9))) && ((((int)threadIdx.x) % 9) < 8)) ? data[(((((rc_outer_outer * 784) + (((((int)threadIdx.x) + 1008) / 81) * 49)) + ((((((int)threadIdx.x) / 9) + 4) % 9) * 7)) + (((int)threadIdx.x) % 9)) - 8)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1120)] = (((((9 <= ((((int)threadIdx.x) + 67) % 81)) && (((((int)threadIdx.x) + 67) % 81) < 72)) && (1 <= ((((int)threadIdx.x) + 4) % 9))) && (((((int)threadIdx.x) + 4) % 9) < 8)) ? data[(((((rc_outer_outer * 784) + (((((int)threadIdx.x) + 1120) / 81) * 49)) + ((((((int)threadIdx.x) + 67) % 81) / 9) * 7)) + ((((int)threadIdx.x) + 4) % 9)) - 8)] : 0.000000e+00f);
    if (((int)threadIdx.x) < 64) {
      pad_temp_shared[(((int)threadIdx.x) + 1232)] = ((((((int)threadIdx.x) < 55) && (1 <= ((((int)threadIdx.x) + 8) % 9))) && (((((int)threadIdx.x) + 8) % 9) < 8)) ? data[(((((rc_outer_outer * 784) + (((((int)threadIdx.x) + 1232) / 81) * 49)) + ((((((int)threadIdx.x) + 17) % 81) / 9) * 7)) + ((((int)threadIdx.x) + 8) % 9)) - 8)] : 0.000000e+00f);
    }
    kernel_shared[((int)threadIdx.x)] = kernel[(((((int)blockIdx.x) * 73728) + (rc_outer_outer * 144)) + ((int)threadIdx.x))];
    kernel_shared[(((int)threadIdx.x) + 112)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 112) / 144) * 4608)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 112) % 144))];
    kernel_shared[(((int)threadIdx.x) + 224)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 224) / 144) * 4608)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 80) % 144))];
    kernel_shared[(((int)threadIdx.x) + 336)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 336) / 144) * 4608)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 48) % 144))];
    kernel_shared[(((int)threadIdx.x) + 448)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 448) / 144) * 4608)) + (rc_outer_outer * 144)) + (((int)threadIdx.x) + 16))];
    kernel_shared[(((int)threadIdx.x) + 560)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 560) / 144) * 4608)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 128) % 144))];
    kernel_shared[(((int)threadIdx.x) + 672)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 672) / 144) * 4608)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 96) % 144))];
    kernel_shared[(((int)threadIdx.x) + 784)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 784) / 144) * 4608)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 64) % 144))];
    kernel_shared[(((int)threadIdx.x) + 896)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 896) / 144) * 4608)) + (rc_outer_outer * 144)) + (((int)threadIdx.x) + 32))];
    kernel_shared[(((int)threadIdx.x) + 1008)] = kernel[((((((int)blockIdx.x) * 73728) + (rc_outer_outer * 144)) + ((int)threadIdx.x)) + 32256)];
    kernel_shared[(((int)threadIdx.x) + 1120)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 1120) / 144) * 4608)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 112) % 144))];
    kernel_shared[(((int)threadIdx.x) + 1232)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 1232) / 144) * 4608)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 80) % 144))];
    kernel_shared[(((int)threadIdx.x) + 1344)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 1344) / 144) * 4608)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 48) % 144))];
    kernel_shared[(((int)threadIdx.x) + 1456)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 1456) / 144) * 4608)) + (rc_outer_outer * 144)) + (((int)threadIdx.x) + 16))];
    kernel_shared[(((int)threadIdx.x) + 1568)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 1568) / 144) * 4608)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 128) % 144))];
    kernel_shared[(((int)threadIdx.x) + 1680)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 1680) / 144) * 4608)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 96) % 144))];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 1792) / 144) * 4608)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 64) % 144))];
    kernel_shared[(((int)threadIdx.x) + 1904)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 1904) / 144) * 4608)) + (rc_outer_outer * 144)) + (((int)threadIdx.x) + 32))];
    kernel_shared[(((int)threadIdx.x) + 2016)] = kernel[((((((int)blockIdx.x) * 73728) + (rc_outer_outer * 144)) + ((int)threadIdx.x)) + 64512)];
    kernel_shared[(((int)threadIdx.x) + 2128)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 2128) / 144) * 4608)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 112) % 144))];
    if (((int)threadIdx.x) < 64) {
      kernel_shared[(((int)threadIdx.x) + 2240)] = kernel[((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 2240) / 144) * 4608)) + (rc_outer_outer * 144)) + (((int)threadIdx.x) + 80))];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 2; ++rc_outer_inner) {
      for (int ry_outer_inner = 0; ry_outer_inner < 3; ++ry_outer_inner) {
        for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((rc_outer_inner * 648) + (rc_inner * 81)) + (ry_outer_inner * 9)) + ((((int)threadIdx.x) % 7) * 9))] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_outer_inner * 3))]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((rc_outer_inner * 648) + (rc_inner * 81)) + (ry_outer_inner * 9)) + ((((int)threadIdx.x) % 7) * 9)) + 1)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_outer_inner * 3))]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((rc_outer_inner * 648) + (rc_inner * 81)) + (ry_outer_inner * 9)) + ((((int)threadIdx.x) % 7) * 9)) + 2)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_outer_inner * 3))]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((rc_outer_inner * 648) + (rc_inner * 81)) + (ry_outer_inner * 9)) + ((((int)threadIdx.x) % 7) * 9)) + 3)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_outer_inner * 3))]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((rc_outer_inner * 648) + (rc_inner * 81)) + (ry_outer_inner * 9)) + ((((int)threadIdx.x) % 7) * 9)) + 4)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_outer_inner * 3))]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((rc_outer_inner * 648) + (rc_inner * 81)) + (ry_outer_inner * 9)) + ((((int)threadIdx.x) % 7) * 9)) + 5)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_outer_inner * 3))]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((rc_outer_inner * 648) + (rc_inner * 81)) + (ry_outer_inner * 9)) + ((((int)threadIdx.x) % 7) * 9)) + 6)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_outer_inner * 3))]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((rc_outer_inner * 648) + (rc_inner * 81)) + (ry_outer_inner * 9)) + ((((int)threadIdx.x) % 7) * 9)) + 1)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + 1)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((rc_outer_inner * 648) + (rc_inner * 81)) + (ry_outer_inner * 9)) + ((((int)threadIdx.x) % 7) * 9)) + 2)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + 1)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((rc_outer_inner * 648) + (rc_inner * 81)) + (ry_outer_inner * 9)) + ((((int)threadIdx.x) % 7) * 9)) + 3)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + 1)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((rc_outer_inner * 648) + (rc_inner * 81)) + (ry_outer_inner * 9)) + ((((int)threadIdx.x) % 7) * 9)) + 4)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + 1)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((rc_outer_inner * 648) + (rc_inner * 81)) + (ry_outer_inner * 9)) + ((((int)threadIdx.x) % 7) * 9)) + 5)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + 1)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((rc_outer_inner * 648) + (rc_inner * 81)) + (ry_outer_inner * 9)) + ((((int)threadIdx.x) % 7) * 9)) + 6)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + 1)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((rc_outer_inner * 648) + (rc_inner * 81)) + (ry_outer_inner * 9)) + ((((int)threadIdx.x) % 7) * 9)) + 7)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + 1)]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((rc_outer_inner * 648) + (rc_inner * 81)) + (ry_outer_inner * 9)) + ((((int)threadIdx.x) % 7) * 9)) + 2)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + 2)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((rc_outer_inner * 648) + (rc_inner * 81)) + (ry_outer_inner * 9)) + ((((int)threadIdx.x) % 7) * 9)) + 3)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + 2)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((rc_outer_inner * 648) + (rc_inner * 81)) + (ry_outer_inner * 9)) + ((((int)threadIdx.x) % 7) * 9)) + 4)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + 2)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((rc_outer_inner * 648) + (rc_inner * 81)) + (ry_outer_inner * 9)) + ((((int)threadIdx.x) % 7) * 9)) + 5)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + 2)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((rc_outer_inner * 648) + (rc_inner * 81)) + (ry_outer_inner * 9)) + ((((int)threadIdx.x) % 7) * 9)) + 6)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + 2)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((rc_outer_inner * 648) + (rc_inner * 81)) + (ry_outer_inner * 9)) + ((((int)threadIdx.x) % 7) * 9)) + 7)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + 2)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((rc_outer_inner * 648) + (rc_inner * 81)) + (ry_outer_inner * 9)) + ((((int)threadIdx.x) % 7) * 9)) + 8)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + 2)]));
        }
      }
    }
  }
  for (int xx_inner = 0; xx_inner < 7; ++xx_inner) {
    conv2d_nchw[(((((int)blockIdx.x) * 784) + (((int)threadIdx.x) * 7)) + xx_inner)] = conv2d_nchw_local[xx_inner];
  }
}


