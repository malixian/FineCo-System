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
extern "C" __global__ void __launch_bounds__(448) candidate1(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[28];
  __shared__ float pad_temp_shared[6272];
  __shared__ float kernel_shared[2048];
  for (int yy_c_outer_inner_init = 0; yy_c_outer_inner_init < 7; ++yy_c_outer_inner_init) {
    conv2d_nchw_local[(yy_c_outer_inner_init * 2)] = 0.000000e+00f;
    conv2d_nchw_local[((yy_c_outer_inner_init * 2) + 14)] = 0.000000e+00f;
    conv2d_nchw_local[((yy_c_outer_inner_init * 2) + 1)] = 0.000000e+00f;
    conv2d_nchw_local[((yy_c_outer_inner_init * 2) + 15)] = 0.000000e+00f;
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 16; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = data[((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 196) * 784)) + ((((int)blockIdx.x) & 3) * 196)) + (((int)threadIdx.x) % 196))];
    pad_temp_shared[(((int)threadIdx.x) + 448)] = data[(((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 448) / 196) * 784)) + ((((int)blockIdx.x) & 3) * 196)) + ((((((int)threadIdx.x) / 28) + 2) % 7) * 28)) + (((int)threadIdx.x) % 28))];
    pad_temp_shared[(((int)threadIdx.x) + 896)] = data[(((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 896) / 196) * 784)) + ((((int)blockIdx.x) & 3) * 196)) + ((((((int)threadIdx.x) / 28) + 4) % 7) * 28)) + (((int)threadIdx.x) % 28))];
    pad_temp_shared[(((int)threadIdx.x) + 1344)] = data[(((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 1344) / 196) * 784)) + ((((int)blockIdx.x) & 3) * 196)) + ((((((int)threadIdx.x) / 28) + 6) % 7) * 28)) + (((int)threadIdx.x) % 28))];
    pad_temp_shared[(((int)threadIdx.x) + 1792)] = data[(((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 1792) / 196) * 784)) + ((((int)blockIdx.x) & 3) * 196)) + ((((((int)threadIdx.x) / 28) + 1) % 7) * 28)) + (((int)threadIdx.x) % 28))];
    pad_temp_shared[(((int)threadIdx.x) + 2240)] = data[(((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 2240) / 196) * 784)) + ((((int)blockIdx.x) & 3) * 196)) + ((((((int)threadIdx.x) / 28) + 3) % 7) * 28)) + (((int)threadIdx.x) % 28))];
    pad_temp_shared[(((int)threadIdx.x) + 2688)] = data[(((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 2688) / 196) * 784)) + ((((int)blockIdx.x) & 3) * 196)) + ((((((int)threadIdx.x) / 28) + 5) % 7) * 28)) + (((int)threadIdx.x) % 28))];
    pad_temp_shared[(((int)threadIdx.x) + 3136)] = data[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 196) * 784)) + ((((int)blockIdx.x) & 3) * 196)) + (((int)threadIdx.x) % 196)) + 12544)];
    pad_temp_shared[(((int)threadIdx.x) + 3584)] = data[(((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 3584) / 196) * 784)) + ((((int)blockIdx.x) & 3) * 196)) + ((((((int)threadIdx.x) / 28) + 2) % 7) * 28)) + (((int)threadIdx.x) % 28))];
    pad_temp_shared[(((int)threadIdx.x) + 4032)] = data[(((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 4032) / 196) * 784)) + ((((int)blockIdx.x) & 3) * 196)) + ((((((int)threadIdx.x) / 28) + 4) % 7) * 28)) + (((int)threadIdx.x) % 28))];
    pad_temp_shared[(((int)threadIdx.x) + 4480)] = data[(((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 4480) / 196) * 784)) + ((((int)blockIdx.x) & 3) * 196)) + ((((((int)threadIdx.x) / 28) + 6) % 7) * 28)) + (((int)threadIdx.x) % 28))];
    pad_temp_shared[(((int)threadIdx.x) + 4928)] = data[(((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 4928) / 196) * 784)) + ((((int)blockIdx.x) & 3) * 196)) + ((((((int)threadIdx.x) / 28) + 1) % 7) * 28)) + (((int)threadIdx.x) % 28))];
    pad_temp_shared[(((int)threadIdx.x) + 5376)] = data[(((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 5376) / 196) * 784)) + ((((int)blockIdx.x) & 3) * 196)) + ((((((int)threadIdx.x) / 28) + 3) % 7) * 28)) + (((int)threadIdx.x) % 28))];
    pad_temp_shared[(((int)threadIdx.x) + 5824)] = data[(((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 5824) / 196) * 784)) + ((((int)blockIdx.x) & 3) * 196)) + ((((((int)threadIdx.x) / 28) + 5) % 7) * 28)) + (((int)threadIdx.x) % 28))];
    kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) >> 2) * 32768) + ((((int)threadIdx.x) >> 5) * 512)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31))];
    kernel_shared[(((int)threadIdx.x) + 448)] = kernel[((((((((int)blockIdx.x) >> 2) * 32768) + ((((int)threadIdx.x) >> 5) * 512)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 7168)];
    kernel_shared[(((int)threadIdx.x) + 896)] = kernel[((((((((int)blockIdx.x) >> 2) * 32768) + ((((int)threadIdx.x) >> 5) * 512)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 14336)];
    kernel_shared[(((int)threadIdx.x) + 1344)] = kernel[((((((((int)blockIdx.x) >> 2) * 32768) + ((((int)threadIdx.x) >> 5) * 512)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 21504)];
    if (((int)threadIdx.x) < 256) {
      kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[((((((((int)blockIdx.x) >> 2) * 32768) + ((((int)threadIdx.x) >> 5) * 512)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 28672)];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 32; ++rc_outer_inner) {
      for (int yy_c_outer_inner = 0; yy_c_outer_inner < 7; ++yy_c_outer_inner) {
        conv2d_nchw_local[(yy_c_outer_inner * 2)] = (conv2d_nchw_local[(yy_c_outer_inner * 2)] + (pad_temp_shared[(((rc_outer_inner * 196) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 14) * 2))] * kernel_shared[(((((int)threadIdx.x) / 14) * 64) + rc_outer_inner)]));
        conv2d_nchw_local[((yy_c_outer_inner * 2) + 14)] = (conv2d_nchw_local[((yy_c_outer_inner * 2) + 14)] + (pad_temp_shared[(((rc_outer_inner * 196) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 14) * 2))] * kernel_shared[((((((int)threadIdx.x) / 14) * 64) + rc_outer_inner) + 32)]));
        conv2d_nchw_local[((yy_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((yy_c_outer_inner * 2) + 1)] + (pad_temp_shared[((((rc_outer_inner * 196) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 14) * 2)) + 1)] * kernel_shared[(((((int)threadIdx.x) / 14) * 64) + rc_outer_inner)]));
        conv2d_nchw_local[((yy_c_outer_inner * 2) + 15)] = (conv2d_nchw_local[((yy_c_outer_inner * 2) + 15)] + (pad_temp_shared[((((rc_outer_inner * 196) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) % 14) * 2)) + 1)] * kernel_shared[((((((int)threadIdx.x) / 14) * 64) + rc_outer_inner) + 32)]));
      }
    }
  }
  for (int ff_inner = 0; ff_inner < 2; ++ff_inner) {
    for (int yy_inner = 0; yy_inner < 7; ++yy_inner) {
      for (int xx_inner = 0; xx_inner < 2; ++xx_inner) {
        conv2d_nchw[((((((((((int)blockIdx.x) >> 2) * 50176) + ((((int)threadIdx.x) / 14) * 1568)) + (ff_inner * 784)) + ((((int)blockIdx.x) & 3) * 196)) + (yy_inner * 28)) + ((((int)threadIdx.x) % 14) * 2)) + xx_inner)] = conv2d_nchw_local[(((ff_inner * 14) + (yy_inner * 2)) + xx_inner)];
      }
    }
  }
}


