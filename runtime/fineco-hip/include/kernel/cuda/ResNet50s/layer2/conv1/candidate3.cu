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
extern "C" __global__ void __launch_bounds__(128) candidate3(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[32];
  __shared__ float pad_temp_shared[128];
  __shared__ float kernel_shared[128];
  for (int ff_c_outer_inner_init = 0; ff_c_outer_inner_init < 2; ++ff_c_outer_inner_init) {
    for (int xx_c_outer_inner_init = 0; xx_c_outer_inner_init < 2; ++xx_c_outer_inner_init) {
      for (int ff_c_inner_init = 0; ff_c_inner_init < 2; ++ff_c_inner_init) {
        for (int yy_c_inner_init = 0; yy_c_inner_init < 2; ++yy_c_inner_init) {
          for (int xx_c_inner_init = 0; xx_c_inner_init < 2; ++xx_c_inner_init) {
            conv2d_nchw_local[(((((ff_c_outer_inner_init * 16) + (ff_c_inner_init * 8)) + (yy_c_inner_init * 4)) + (xx_c_outer_inner_init * 2)) + xx_c_inner_init)] = 0.000000e+00f;
          }
        }
      }
    }
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 32; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = data[((((((rc_outer_outer * 6272) + ((((int)threadIdx.x) >> 6) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7))];
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)threadIdx.x) >> 1) * 64) + (rc_outer_outer * 2)) + (((int)threadIdx.x) & 1))];
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 2; ++rc_outer_inner) {
      for (int ff_c_outer_inner = 0; ff_c_outer_inner < 2; ++ff_c_outer_inner) {
        for (int xx_c_outer_inner = 0; xx_c_outer_inner < 2; ++xx_c_outer_inner) {
          for (int ff_c_inner = 0; ff_c_inner < 2; ++ff_c_inner) {
            for (int yy_c_inner = 0; yy_c_inner < 2; ++yy_c_inner) {
              for (int xx_c_inner = 0; xx_c_inner < 2; ++xx_c_inner) {
                conv2d_nchw_local[(((((ff_c_outer_inner * 16) + (ff_c_inner * 8)) + (yy_c_inner * 4)) + (xx_c_outer_inner * 2)) + xx_c_inner)] = (conv2d_nchw_local[(((((ff_c_outer_inner * 16) + (ff_c_inner * 8)) + (yy_c_inner * 4)) + (xx_c_outer_inner * 2)) + xx_c_inner)] + (pad_temp_shared[((((((rc_outer_inner * 64) + (((((int)threadIdx.x) & 7) >> 1) * 16)) + (yy_c_inner * 8)) + ((((int)threadIdx.x) & 1) * 4)) + (xx_c_outer_inner * 2)) + xx_c_inner)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 8) + (ff_c_outer_inner * 4)) + (ff_c_inner * 2)) + rc_outer_inner)]));
              }
            }
          }
        }
      }
    }
  }
  for (int ff_inner = 0; ff_inner < 4; ++ff_inner) {
    for (int yy_inner = 0; yy_inner < 2; ++yy_inner) {
      for (int xx_inner = 0; xx_inner < 4; ++xx_inner) {
        conv2d_nchw[(((((((((((int)threadIdx.x) >> 3) * 12544) + (ff_inner * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((int)threadIdx.x) & 7) >> 1) * 112)) + (yy_inner * 56)) + ((((int)blockIdx.x) % 7) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + xx_inner)] = conv2d_nchw_local[(((ff_inner * 8) + (yy_inner * 4)) + xx_inner)];
      }
    }
  }
}


