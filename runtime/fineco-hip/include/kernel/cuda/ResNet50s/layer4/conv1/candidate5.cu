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
extern "C" __global__ void __launch_bounds__(448) candidate5(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[8];
  __shared__ float pad_temp_shared[3584];
  __shared__ float kernel_shared[1024];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[7] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 16; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = data[((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112))];
    pad_temp_shared[(((int)threadIdx.x) + 448)] = data[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 3136)];
    pad_temp_shared[(((int)threadIdx.x) + 896)] = data[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 6272)];
    pad_temp_shared[(((int)threadIdx.x) + 1344)] = data[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 9408)];
    pad_temp_shared[(((int)threadIdx.x) + 1792)] = data[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 12544)];
    pad_temp_shared[(((int)threadIdx.x) + 2240)] = data[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 15680)];
    pad_temp_shared[(((int)threadIdx.x) + 2688)] = data[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 18816)];
    pad_temp_shared[(((int)threadIdx.x) + 3136)] = data[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 21952)];
    kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + ((((int)threadIdx.x) >> 5) * 512)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31))];
    kernel_shared[(((int)threadIdx.x) + 448)] = kernel[((((((((int)blockIdx.x) / 7) * 16384) + ((((int)threadIdx.x) >> 5) * 512)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 7168)];
    if (((int)threadIdx.x) < 128) {
      kernel_shared[(((int)threadIdx.x) + 896)] = kernel[((((((((int)blockIdx.x) / 7) * 16384) + ((((int)threadIdx.x) >> 5) * 512)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 14336)];
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 32; ++rc_inner) {
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((rc_inner * 112) + (((int)threadIdx.x) % 28))] * kernel_shared[(((((int)threadIdx.x) / 28) * 32) + rc_inner)]));
      conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[((rc_inner * 112) + (((int)threadIdx.x) % 28))] * kernel_shared[((((((int)threadIdx.x) / 28) * 32) + rc_inner) + 512)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_inner * 112) + (((int)threadIdx.x) % 28)) + 28)] * kernel_shared[(((((int)threadIdx.x) / 28) * 32) + rc_inner)]));
      conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((rc_inner * 112) + (((int)threadIdx.x) % 28)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 28) * 32) + rc_inner) + 512)]));
      conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((rc_inner * 112) + (((int)threadIdx.x) % 28)) + 56)] * kernel_shared[(((((int)threadIdx.x) / 28) * 32) + rc_inner)]));
      conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((rc_inner * 112) + (((int)threadIdx.x) % 28)) + 56)] * kernel_shared[((((((int)threadIdx.x) / 28) * 32) + rc_inner) + 512)]));
      conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((rc_inner * 112) + (((int)threadIdx.x) % 28)) + 84)] * kernel_shared[(((((int)threadIdx.x) / 28) * 32) + rc_inner)]));
      conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((rc_inner * 112) + (((int)threadIdx.x) % 28)) + 84)] * kernel_shared[((((((int)threadIdx.x) / 28) * 32) + rc_inner) + 512)]));
    }
  }
  for (int yy_inner = 0; yy_inner < 4; ++yy_inner) {
    conv2d_nchw[((((((((int)blockIdx.x) / 7) * 25088) + ((((int)threadIdx.x) / 28) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (yy_inner * 28)) + (((int)threadIdx.x) % 28))] = conv2d_nchw_local[yy_inner];
    conv2d_nchw[(((((((((int)blockIdx.x) / 7) * 25088) + ((((int)threadIdx.x) / 28) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (yy_inner * 28)) + (((int)threadIdx.x) % 28)) + 12544)] = conv2d_nchw_local[(yy_inner + 4)];
  }
}


