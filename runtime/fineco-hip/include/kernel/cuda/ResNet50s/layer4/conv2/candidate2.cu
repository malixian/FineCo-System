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
  float conv2d_nchw_local[8];
  __shared__ float pad_temp_shared[896];
  __shared__ float kernel_shared[3072];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[7] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 16; ++rc_outer_outer) {
    for (int rx_outer_outer = 0; rx_outer_outer < 3; ++rx_outer_outer) {
      __syncthreads();
      pad_temp_shared[((int)threadIdx.x)] = (((((1 <= (((((int)blockIdx.x) % 7) * 2) + ((((int)threadIdx.x) % 56) / 14))) && ((((((int)blockIdx.x) % 7) * 2) + ((((int)threadIdx.x) % 56) / 14)) < 15)) && (1 <= (rx_outer_outer + (((int)threadIdx.x) % 14)))) && ((rx_outer_outer + (((int)threadIdx.x) % 14)) < 15)) ? data[((((((rc_outer_outer * 3136) + ((((int)threadIdx.x) / 56) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + rx_outer_outer) + (((int)threadIdx.x) % 56)) - 15)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 224)] = (((((1 <= (((((int)blockIdx.x) % 7) * 2) + ((((int)threadIdx.x) % 56) / 14))) && ((((((int)blockIdx.x) % 7) * 2) + ((((int)threadIdx.x) % 56) / 14)) < 15)) && (1 <= (rx_outer_outer + (((int)threadIdx.x) % 14)))) && ((rx_outer_outer + (((int)threadIdx.x) % 14)) < 15)) ? data[((((((rc_outer_outer * 3136) + ((((int)threadIdx.x) / 56) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + rx_outer_outer) + (((int)threadIdx.x) % 56)) + 769)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 448)] = (((((1 <= (((((int)blockIdx.x) % 7) * 2) + ((((int)threadIdx.x) % 56) / 14))) && ((((((int)blockIdx.x) % 7) * 2) + ((((int)threadIdx.x) % 56) / 14)) < 15)) && (1 <= (rx_outer_outer + (((int)threadIdx.x) % 14)))) && ((rx_outer_outer + (((int)threadIdx.x) % 14)) < 15)) ? data[((((((rc_outer_outer * 3136) + ((((int)threadIdx.x) / 56) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + rx_outer_outer) + (((int)threadIdx.x) % 56)) + 1553)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 672)] = (((((1 <= (((((int)blockIdx.x) % 7) * 2) + ((((int)threadIdx.x) % 56) / 14))) && ((((((int)blockIdx.x) % 7) * 2) + ((((int)threadIdx.x) % 56) / 14)) < 15)) && (1 <= (rx_outer_outer + (((int)threadIdx.x) % 14)))) && ((rx_outer_outer + (((int)threadIdx.x) % 14)) < 15)) ? data[((((((rc_outer_outer * 3136) + ((((int)threadIdx.x) / 56) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + rx_outer_outer) + (((int)threadIdx.x) % 56)) + 2337)] : 0.000000e+00f);
      kernel_shared[((int)threadIdx.x)] = kernel[((((((((int)blockIdx.x) / 7) * 147456) + ((((int)threadIdx.x) / 48) * 2304)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) % 48) * 3)) + rx_outer_outer)];
      kernel_shared[(((int)threadIdx.x) + 224)] = kernel[((((((((int)blockIdx.x) / 7) * 147456) + (((((int)threadIdx.x) + 224) / 48) * 2304)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) + 32) % 48) * 3)) + rx_outer_outer)];
      kernel_shared[(((int)threadIdx.x) + 448)] = kernel[((((((((int)blockIdx.x) / 7) * 147456) + (((((int)threadIdx.x) + 448) / 48) * 2304)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) + 16) % 48) * 3)) + rx_outer_outer)];
      kernel_shared[(((int)threadIdx.x) + 672)] = kernel[(((((((((int)blockIdx.x) / 7) * 147456) + ((((int)threadIdx.x) / 48) * 2304)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) % 48) * 3)) + rx_outer_outer) + 32256)];
      kernel_shared[(((int)threadIdx.x) + 896)] = kernel[((((((((int)blockIdx.x) / 7) * 147456) + (((((int)threadIdx.x) + 896) / 48) * 2304)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) + 32) % 48) * 3)) + rx_outer_outer)];
      kernel_shared[(((int)threadIdx.x) + 1120)] = kernel[((((((((int)blockIdx.x) / 7) * 147456) + (((((int)threadIdx.x) + 1120) / 48) * 2304)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) + 16) % 48) * 3)) + rx_outer_outer)];
      kernel_shared[(((int)threadIdx.x) + 1344)] = kernel[(((((((((int)blockIdx.x) / 7) * 147456) + ((((int)threadIdx.x) / 48) * 2304)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) % 48) * 3)) + rx_outer_outer) + 64512)];
      kernel_shared[(((int)threadIdx.x) + 1568)] = kernel[((((((((int)blockIdx.x) / 7) * 147456) + (((((int)threadIdx.x) + 1568) / 48) * 2304)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) + 32) % 48) * 3)) + rx_outer_outer)];
      kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[((((((((int)blockIdx.x) / 7) * 147456) + (((((int)threadIdx.x) + 1792) / 48) * 2304)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) + 16) % 48) * 3)) + rx_outer_outer)];
      kernel_shared[(((int)threadIdx.x) + 2016)] = kernel[(((((((((int)blockIdx.x) / 7) * 147456) + ((((int)threadIdx.x) / 48) * 2304)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) % 48) * 3)) + rx_outer_outer) + 96768)];
      kernel_shared[(((int)threadIdx.x) + 2240)] = kernel[((((((((int)blockIdx.x) / 7) * 147456) + (((((int)threadIdx.x) + 2240) / 48) * 2304)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) + 32) % 48) * 3)) + rx_outer_outer)];
      kernel_shared[(((int)threadIdx.x) + 2464)] = kernel[((((((((int)blockIdx.x) / 7) * 147456) + (((((int)threadIdx.x) + 2464) / 48) * 2304)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) + 16) % 48) * 3)) + rx_outer_outer)];
      kernel_shared[(((int)threadIdx.x) + 2688)] = kernel[(((((((((int)blockIdx.x) / 7) * 147456) + ((((int)threadIdx.x) / 48) * 2304)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) % 48) * 3)) + rx_outer_outer) + 129024)];
      if (((int)threadIdx.x) < 160) {
        kernel_shared[(((int)threadIdx.x) + 2912)] = kernel[((((((((int)blockIdx.x) / 7) * 147456) + (((((int)threadIdx.x) + 2912) / 48) * 2304)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) + 32) % 48) * 3)) + rx_outer_outer)];
      }
      __syncthreads();
      for (int rc_outer_inner = 0; rc_outer_inner < 2; ++rc_outer_inner) {
        for (int ry_outer_inner = 0; ry_outer_inner < 3; ++ry_outer_inner) {
          for (int ff_c_outer_inner = 0; ff_c_outer_inner < 2; ++ff_c_outer_inner) {
            for (int yy_c_outer_inner = 0; yy_c_outer_inner < 2; ++yy_c_outer_inner) {
              conv2d_nchw_local[((ff_c_outer_inner * 2) + yy_c_outer_inner)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + yy_c_outer_inner)] + (pad_temp_shared[((((rc_outer_inner * 448) + (yy_c_outer_inner * 14)) + (ry_outer_inner * 14)) + (((int)threadIdx.x) % 14))] * kernel_shared[(((((((int)threadIdx.x) / 14) * 96) + (ff_c_outer_inner * 48)) + (rc_outer_inner * 24)) + ry_outer_inner)]));
              conv2d_nchw_local[(((ff_c_outer_inner * 2) + yy_c_outer_inner) + 4)] = (conv2d_nchw_local[(((ff_c_outer_inner * 2) + yy_c_outer_inner) + 4)] + (pad_temp_shared[((((rc_outer_inner * 448) + (yy_c_outer_inner * 14)) + (ry_outer_inner * 14)) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((((int)threadIdx.x) / 14) * 96) + (ff_c_outer_inner * 48)) + (rc_outer_inner * 24)) + ry_outer_inner) + 1536)]));
              conv2d_nchw_local[((ff_c_outer_inner * 2) + yy_c_outer_inner)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + yy_c_outer_inner)] + (pad_temp_shared[(((((rc_outer_inner * 448) + (yy_c_outer_inner * 14)) + (ry_outer_inner * 14)) + (((int)threadIdx.x) % 14)) + 56)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 96) + (ff_c_outer_inner * 48)) + (rc_outer_inner * 24)) + ry_outer_inner) + 3)]));
              conv2d_nchw_local[(((ff_c_outer_inner * 2) + yy_c_outer_inner) + 4)] = (conv2d_nchw_local[(((ff_c_outer_inner * 2) + yy_c_outer_inner) + 4)] + (pad_temp_shared[(((((rc_outer_inner * 448) + (yy_c_outer_inner * 14)) + (ry_outer_inner * 14)) + (((int)threadIdx.x) % 14)) + 56)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 96) + (ff_c_outer_inner * 48)) + (rc_outer_inner * 24)) + ry_outer_inner) + 1539)]));
              conv2d_nchw_local[((ff_c_outer_inner * 2) + yy_c_outer_inner)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + yy_c_outer_inner)] + (pad_temp_shared[(((((rc_outer_inner * 448) + (yy_c_outer_inner * 14)) + (ry_outer_inner * 14)) + (((int)threadIdx.x) % 14)) + 112)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 96) + (ff_c_outer_inner * 48)) + (rc_outer_inner * 24)) + ry_outer_inner) + 6)]));
              conv2d_nchw_local[(((ff_c_outer_inner * 2) + yy_c_outer_inner) + 4)] = (conv2d_nchw_local[(((ff_c_outer_inner * 2) + yy_c_outer_inner) + 4)] + (pad_temp_shared[(((((rc_outer_inner * 448) + (yy_c_outer_inner * 14)) + (ry_outer_inner * 14)) + (((int)threadIdx.x) % 14)) + 112)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 96) + (ff_c_outer_inner * 48)) + (rc_outer_inner * 24)) + ry_outer_inner) + 1542)]));
              conv2d_nchw_local[((ff_c_outer_inner * 2) + yy_c_outer_inner)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + yy_c_outer_inner)] + (pad_temp_shared[(((((rc_outer_inner * 448) + (yy_c_outer_inner * 14)) + (ry_outer_inner * 14)) + (((int)threadIdx.x) % 14)) + 168)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 96) + (ff_c_outer_inner * 48)) + (rc_outer_inner * 24)) + ry_outer_inner) + 9)]));
              conv2d_nchw_local[(((ff_c_outer_inner * 2) + yy_c_outer_inner) + 4)] = (conv2d_nchw_local[(((ff_c_outer_inner * 2) + yy_c_outer_inner) + 4)] + (pad_temp_shared[(((((rc_outer_inner * 448) + (yy_c_outer_inner * 14)) + (ry_outer_inner * 14)) + (((int)threadIdx.x) % 14)) + 168)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 96) + (ff_c_outer_inner * 48)) + (rc_outer_inner * 24)) + ry_outer_inner) + 1545)]));
              conv2d_nchw_local[((ff_c_outer_inner * 2) + yy_c_outer_inner)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + yy_c_outer_inner)] + (pad_temp_shared[(((((rc_outer_inner * 448) + (yy_c_outer_inner * 14)) + (ry_outer_inner * 14)) + (((int)threadIdx.x) % 14)) + 224)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 96) + (ff_c_outer_inner * 48)) + (rc_outer_inner * 24)) + ry_outer_inner) + 12)]));
              conv2d_nchw_local[(((ff_c_outer_inner * 2) + yy_c_outer_inner) + 4)] = (conv2d_nchw_local[(((ff_c_outer_inner * 2) + yy_c_outer_inner) + 4)] + (pad_temp_shared[(((((rc_outer_inner * 448) + (yy_c_outer_inner * 14)) + (ry_outer_inner * 14)) + (((int)threadIdx.x) % 14)) + 224)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 96) + (ff_c_outer_inner * 48)) + (rc_outer_inner * 24)) + ry_outer_inner) + 1548)]));
              conv2d_nchw_local[((ff_c_outer_inner * 2) + yy_c_outer_inner)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + yy_c_outer_inner)] + (pad_temp_shared[(((((rc_outer_inner * 448) + (yy_c_outer_inner * 14)) + (ry_outer_inner * 14)) + (((int)threadIdx.x) % 14)) + 280)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 96) + (ff_c_outer_inner * 48)) + (rc_outer_inner * 24)) + ry_outer_inner) + 15)]));
              conv2d_nchw_local[(((ff_c_outer_inner * 2) + yy_c_outer_inner) + 4)] = (conv2d_nchw_local[(((ff_c_outer_inner * 2) + yy_c_outer_inner) + 4)] + (pad_temp_shared[(((((rc_outer_inner * 448) + (yy_c_outer_inner * 14)) + (ry_outer_inner * 14)) + (((int)threadIdx.x) % 14)) + 280)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 96) + (ff_c_outer_inner * 48)) + (rc_outer_inner * 24)) + ry_outer_inner) + 1551)]));
              conv2d_nchw_local[((ff_c_outer_inner * 2) + yy_c_outer_inner)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + yy_c_outer_inner)] + (pad_temp_shared[(((((rc_outer_inner * 448) + (yy_c_outer_inner * 14)) + (ry_outer_inner * 14)) + (((int)threadIdx.x) % 14)) + 336)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 96) + (ff_c_outer_inner * 48)) + (rc_outer_inner * 24)) + ry_outer_inner) + 18)]));
              conv2d_nchw_local[(((ff_c_outer_inner * 2) + yy_c_outer_inner) + 4)] = (conv2d_nchw_local[(((ff_c_outer_inner * 2) + yy_c_outer_inner) + 4)] + (pad_temp_shared[(((((rc_outer_inner * 448) + (yy_c_outer_inner * 14)) + (ry_outer_inner * 14)) + (((int)threadIdx.x) % 14)) + 336)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 96) + (ff_c_outer_inner * 48)) + (rc_outer_inner * 24)) + ry_outer_inner) + 1554)]));
              conv2d_nchw_local[((ff_c_outer_inner * 2) + yy_c_outer_inner)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + yy_c_outer_inner)] + (pad_temp_shared[(((((rc_outer_inner * 448) + (yy_c_outer_inner * 14)) + (ry_outer_inner * 14)) + (((int)threadIdx.x) % 14)) + 392)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 96) + (ff_c_outer_inner * 48)) + (rc_outer_inner * 24)) + ry_outer_inner) + 21)]));
              conv2d_nchw_local[(((ff_c_outer_inner * 2) + yy_c_outer_inner) + 4)] = (conv2d_nchw_local[(((ff_c_outer_inner * 2) + yy_c_outer_inner) + 4)] + (pad_temp_shared[(((((rc_outer_inner * 448) + (yy_c_outer_inner * 14)) + (ry_outer_inner * 14)) + (((int)threadIdx.x) % 14)) + 392)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 96) + (ff_c_outer_inner * 48)) + (rc_outer_inner * 24)) + ry_outer_inner) + 1557)]));
            }
          }
        }
      }
    }
  }
  for (int ff_inner = 0; ff_inner < 2; ++ff_inner) {
    for (int yy_inner = 0; yy_inner < 2; ++yy_inner) {
      conv2d_nchw[(((((((((int)blockIdx.x) / 7) * 12544) + ((((int)threadIdx.x) / 14) * 392)) + (ff_inner * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (yy_inner * 14)) + (((int)threadIdx.x) % 14))] = conv2d_nchw_local[((ff_inner * 2) + yy_inner)];
      conv2d_nchw[((((((((((int)blockIdx.x) / 7) * 12544) + ((((int)threadIdx.x) / 14) * 392)) + (ff_inner * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (yy_inner * 14)) + (((int)threadIdx.x) % 14)) + 6272)] = conv2d_nchw_local[(((ff_inner * 2) + yy_inner) + 4)];
    }
  }
}


