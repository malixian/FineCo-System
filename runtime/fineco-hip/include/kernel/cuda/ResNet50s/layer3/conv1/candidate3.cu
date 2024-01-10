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
extern "C" __global__ void __launch_bounds__(448) candidate3(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[28];
  __shared__ float pad_temp_shared[1568];
  __shared__ float kernel_shared[2048];
  for (int ff_c_outer_inner_init = 0; ff_c_outer_inner_init < 4; ++ff_c_outer_inner_init) {
    conv2d_nchw_local[ff_c_outer_inner_init] = 0.000000e+00f;
    conv2d_nchw_local[(ff_c_outer_inner_init + 4)] = 0.000000e+00f;
    conv2d_nchw_local[(ff_c_outer_inner_init + 8)] = 0.000000e+00f;
    conv2d_nchw_local[(ff_c_outer_inner_init + 12)] = 0.000000e+00f;
    conv2d_nchw_local[(ff_c_outer_inner_init + 16)] = 0.000000e+00f;
    conv2d_nchw_local[(ff_c_outer_inner_init + 20)] = 0.000000e+00f;
    conv2d_nchw_local[(ff_c_outer_inner_init + 24)] = 0.000000e+00f;
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 16; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = data[((((((rc_outer_outer * 50176) + ((((int)threadIdx.x) / 98) * 3136)) + ((((int)blockIdx.x) >> 2) * 392)) + (((((int)threadIdx.x) % 98) / 14) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14))];
    pad_temp_shared[(((int)threadIdx.x) + 448)] = data[((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 448) / 98) * 3136)) + ((((int)blockIdx.x) >> 2) * 392)) + ((((((int)threadIdx.x) / 14) + 4) % 7) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14))];
    pad_temp_shared[(((int)threadIdx.x) + 896)] = data[((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 896) / 98) * 3136)) + ((((int)blockIdx.x) >> 2) * 392)) + ((((((int)threadIdx.x) / 14) + 1) % 7) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14))];
    if (((int)threadIdx.x) < 224) {
      pad_temp_shared[(((int)threadIdx.x) + 1344)] = data[((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 1344) / 98) * 3136)) + ((((int)blockIdx.x) >> 2) * 392)) + ((((((int)threadIdx.x) / 14) + 5) % 7) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14))];
    }
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)threadIdx.x) >> 4) * 256) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15))];
    kernel_shared[(((int)threadIdx.x) + 448)] = kernel[(((((((int)threadIdx.x) >> 4) * 256) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 7168)];
    kernel_shared[(((int)threadIdx.x) + 896)] = kernel[(((((((int)threadIdx.x) >> 4) * 256) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 14336)];
    kernel_shared[(((int)threadIdx.x) + 1344)] = kernel[(((((((int)threadIdx.x) >> 4) * 256) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 21504)];
    if (((int)threadIdx.x) < 256) {
      kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[(((((((int)threadIdx.x) >> 4) * 256) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 28672)];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 2; ++rc_outer_inner) {
      for (int ff_c_outer_inner = 0; ff_c_outer_inner < 4; ++ff_c_outer_inner) {
        for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
          conv2d_nchw_local[ff_c_outer_inner] = (conv2d_nchw_local[ff_c_outer_inner] + (pad_temp_shared[((((rc_outer_inner * 784) + (rc_inner * 98)) + (((((int)threadIdx.x) % 14) >> 1) * 14)) + (((int)threadIdx.x) & 1))] * kernel_shared[(((((((int)threadIdx.x) / 14) * 64) + (ff_c_outer_inner * 16)) + (rc_outer_inner * 8)) + rc_inner)]));
          conv2d_nchw_local[(ff_c_outer_inner + 4)] = (conv2d_nchw_local[(ff_c_outer_inner + 4)] + (pad_temp_shared[(((((rc_outer_inner * 784) + (rc_inner * 98)) + (((((int)threadIdx.x) % 14) >> 1) * 14)) + (((int)threadIdx.x) & 1)) + 2)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 64) + (ff_c_outer_inner * 16)) + (rc_outer_inner * 8)) + rc_inner)]));
          conv2d_nchw_local[(ff_c_outer_inner + 8)] = (conv2d_nchw_local[(ff_c_outer_inner + 8)] + (pad_temp_shared[(((((rc_outer_inner * 784) + (rc_inner * 98)) + (((((int)threadIdx.x) % 14) >> 1) * 14)) + (((int)threadIdx.x) & 1)) + 4)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 64) + (ff_c_outer_inner * 16)) + (rc_outer_inner * 8)) + rc_inner)]));
          conv2d_nchw_local[(ff_c_outer_inner + 12)] = (conv2d_nchw_local[(ff_c_outer_inner + 12)] + (pad_temp_shared[(((((rc_outer_inner * 784) + (rc_inner * 98)) + (((((int)threadIdx.x) % 14) >> 1) * 14)) + (((int)threadIdx.x) & 1)) + 6)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 64) + (ff_c_outer_inner * 16)) + (rc_outer_inner * 8)) + rc_inner)]));
          conv2d_nchw_local[(ff_c_outer_inner + 16)] = (conv2d_nchw_local[(ff_c_outer_inner + 16)] + (pad_temp_shared[(((((rc_outer_inner * 784) + (rc_inner * 98)) + (((((int)threadIdx.x) % 14) >> 1) * 14)) + (((int)threadIdx.x) & 1)) + 8)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 64) + (ff_c_outer_inner * 16)) + (rc_outer_inner * 8)) + rc_inner)]));
          conv2d_nchw_local[(ff_c_outer_inner + 20)] = (conv2d_nchw_local[(ff_c_outer_inner + 20)] + (pad_temp_shared[(((((rc_outer_inner * 784) + (rc_inner * 98)) + (((((int)threadIdx.x) % 14) >> 1) * 14)) + (((int)threadIdx.x) & 1)) + 10)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 64) + (ff_c_outer_inner * 16)) + (rc_outer_inner * 8)) + rc_inner)]));
          conv2d_nchw_local[(ff_c_outer_inner + 24)] = (conv2d_nchw_local[(ff_c_outer_inner + 24)] + (pad_temp_shared[(((((rc_outer_inner * 784) + (rc_inner * 98)) + (((((int)threadIdx.x) % 14) >> 1) * 14)) + (((int)threadIdx.x) & 1)) + 12)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 64) + (ff_c_outer_inner * 16)) + (rc_outer_inner * 8)) + rc_inner)]));
        }
      }
    }
  }
  for (int ff_inner = 0; ff_inner < 4; ++ff_inner) {
    conv2d_nchw[(((((((((int)threadIdx.x) / 14) * 12544) + (ff_inner * 3136)) + ((((int)blockIdx.x) >> 2) * 392)) + (((((int)threadIdx.x) % 14) >> 1) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) & 1))] = conv2d_nchw_local[ff_inner];
    conv2d_nchw[((((((((((int)threadIdx.x) / 14) * 12544) + (ff_inner * 3136)) + ((((int)blockIdx.x) >> 2) * 392)) + (((((int)threadIdx.x) % 14) >> 1) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) & 1)) + 2)] = conv2d_nchw_local[(ff_inner + 4)];
    conv2d_nchw[((((((((((int)threadIdx.x) / 14) * 12544) + (ff_inner * 3136)) + ((((int)blockIdx.x) >> 2) * 392)) + (((((int)threadIdx.x) % 14) >> 1) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) & 1)) + 4)] = conv2d_nchw_local[(ff_inner + 8)];
    conv2d_nchw[((((((((((int)threadIdx.x) / 14) * 12544) + (ff_inner * 3136)) + ((((int)blockIdx.x) >> 2) * 392)) + (((((int)threadIdx.x) % 14) >> 1) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) & 1)) + 6)] = conv2d_nchw_local[(ff_inner + 12)];
    conv2d_nchw[((((((((((int)threadIdx.x) / 14) * 12544) + (ff_inner * 3136)) + ((((int)blockIdx.x) >> 2) * 392)) + (((((int)threadIdx.x) % 14) >> 1) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) & 1)) + 8)] = conv2d_nchw_local[(ff_inner + 16)];
    conv2d_nchw[((((((((((int)threadIdx.x) / 14) * 12544) + (ff_inner * 3136)) + ((((int)blockIdx.x) >> 2) * 392)) + (((((int)threadIdx.x) % 14) >> 1) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) & 1)) + 10)] = conv2d_nchw_local[(ff_inner + 20)];
    conv2d_nchw[((((((((((int)threadIdx.x) / 14) * 12544) + (ff_inner * 3136)) + ((((int)blockIdx.x) >> 2) * 392)) + (((((int)threadIdx.x) % 14) >> 1) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) & 1)) + 12)] = conv2d_nchw_local[(ff_inner + 24)];
  }
}


