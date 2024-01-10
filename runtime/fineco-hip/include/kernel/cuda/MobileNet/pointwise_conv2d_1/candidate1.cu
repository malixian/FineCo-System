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
extern "C" __global__ void __launch_bounds__(224) candidate1(float* __restrict__ Input, float* __restrict__ kernel, float* __restrict__ compute) {
  float conv2d_nchw[224];
  __shared__ float pad_temp_shared[3136];
  __shared__ float kernel_shared[256];
  for (int ff_inner_init = 0; ff_inner_init < 4; ++ff_inner_init) {
    for (int yy_inner_init = 0; yy_inner_init < 14; ++yy_inner_init) {
      conv2d_nchw[((ff_inner_init * 14) + yy_inner_init)] = 0.000000e+00f;
      conv2d_nchw[(((ff_inner_init * 14) + yy_inner_init) + 56)] = 0.000000e+00f;
      conv2d_nchw[(((ff_inner_init * 14) + yy_inner_init) + 112)] = 0.000000e+00f;
      conv2d_nchw[(((ff_inner_init * 14) + yy_inner_init) + 168)] = 0.000000e+00f;
    }
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 8; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = Input[(((((rc_outer_outer * 50176) + ((((int)blockIdx.x) >> 2) * 3136)) + ((((int)threadIdx.x) / 28) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + (((int)threadIdx.x) % 28))];
    pad_temp_shared[(((int)threadIdx.x) + 224)] = Input[((((((rc_outer_outer * 50176) + ((((int)blockIdx.x) >> 2) * 3136)) + ((((int)threadIdx.x) / 28) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + (((int)threadIdx.x) % 28)) + 896)];
    pad_temp_shared[(((int)threadIdx.x) + 448)] = Input[((((((rc_outer_outer * 50176) + ((((int)blockIdx.x) >> 2) * 3136)) + ((((int)threadIdx.x) / 28) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + (((int)threadIdx.x) % 28)) + 1792)];
    pad_temp_shared[(((int)threadIdx.x) + 672)] = Input[((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 672) / 784) * 12544)) + ((((int)blockIdx.x) >> 2) * 3136)) + ((((((int)threadIdx.x) / 28) + 24) % 28) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + (((int)threadIdx.x) % 28))];
    pad_temp_shared[(((int)threadIdx.x) + 896)] = Input[((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 896) / 784) * 12544)) + ((((int)blockIdx.x) >> 2) * 3136)) + (((((int)threadIdx.x) / 28) + 4) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + (((int)threadIdx.x) % 28))];
    pad_temp_shared[(((int)threadIdx.x) + 1120)] = Input[((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 1120) / 784) * 12544)) + ((((int)blockIdx.x) >> 2) * 3136)) + (((((int)threadIdx.x) / 28) + 12) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + (((int)threadIdx.x) % 28))];
    pad_temp_shared[(((int)threadIdx.x) + 1344)] = Input[((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 1344) / 784) * 12544)) + ((((int)blockIdx.x) >> 2) * 3136)) + (((((int)threadIdx.x) / 28) + 20) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + (((int)threadIdx.x) % 28))];
    pad_temp_shared[(((int)threadIdx.x) + 1568)] = Input[((((((rc_outer_outer * 50176) + ((((int)blockIdx.x) >> 2) * 3136)) + ((((int)threadIdx.x) / 28) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + (((int)threadIdx.x) % 28)) + 25088)];
    pad_temp_shared[(((int)threadIdx.x) + 1792)] = Input[((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 1792) / 784) * 12544)) + ((((int)blockIdx.x) >> 2) * 3136)) + (((((int)threadIdx.x) / 28) + 8) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + (((int)threadIdx.x) % 28))];
    pad_temp_shared[(((int)threadIdx.x) + 2016)] = Input[((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 2016) / 784) * 12544)) + ((((int)blockIdx.x) >> 2) * 3136)) + (((((int)threadIdx.x) / 28) + 16) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + (((int)threadIdx.x) % 28))];
    pad_temp_shared[(((int)threadIdx.x) + 2240)] = Input[((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 2240) / 784) * 12544)) + ((((int)blockIdx.x) >> 2) * 3136)) + ((((((int)threadIdx.x) / 28) + 24) % 28) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + (((int)threadIdx.x) % 28))];
    pad_temp_shared[(((int)threadIdx.x) + 2464)] = Input[((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 2464) / 784) * 12544)) + ((((int)blockIdx.x) >> 2) * 3136)) + (((((int)threadIdx.x) / 28) + 4) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + (((int)threadIdx.x) % 28))];
    pad_temp_shared[(((int)threadIdx.x) + 2688)] = Input[((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 2688) / 784) * 12544)) + ((((int)blockIdx.x) >> 2) * 3136)) + (((((int)threadIdx.x) / 28) + 12) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + (((int)threadIdx.x) % 28))];
    pad_temp_shared[(((int)threadIdx.x) + 2912)] = Input[((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 2912) / 784) * 12544)) + ((((int)blockIdx.x) >> 2) * 3136)) + (((((int)threadIdx.x) / 28) + 20) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + (((int)threadIdx.x) % 28))];
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)threadIdx.x) >> 2) * 32) + (rc_outer_outer * 4)) + (((int)threadIdx.x) & 3))];
    if (((int)threadIdx.x) < 32) {
      kernel_shared[(((int)threadIdx.x) + 224)] = kernel[(((((((int)threadIdx.x) >> 2) * 32) + (rc_outer_outer * 4)) + (((int)threadIdx.x) & 3)) + 1792)];
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 4; ++rc_inner) {
      for (int ff_inner = 0; ff_inner < 4; ++ff_inner) {
        for (int yy_inner = 0; yy_inner < 14; ++yy_inner) {
          conv2d_nchw[((ff_inner * 14) + yy_inner)] = (conv2d_nchw[((ff_inner * 14) + yy_inner)] + (pad_temp_shared[((((rc_inner * 784) + (((((int)threadIdx.x) % 56) / 28) * 392)) + (yy_inner * 28)) + (((int)threadIdx.x) % 28))] * kernel_shared[((((((int)threadIdx.x) / 56) * 16) + (ff_inner * 4)) + rc_inner)]));
          conv2d_nchw[(((ff_inner * 14) + yy_inner) + 56)] = (conv2d_nchw[(((ff_inner * 14) + yy_inner) + 56)] + (pad_temp_shared[((((rc_inner * 784) + (((((int)threadIdx.x) % 56) / 28) * 392)) + (yy_inner * 28)) + (((int)threadIdx.x) % 28))] * kernel_shared[(((((((int)threadIdx.x) / 56) * 16) + (ff_inner * 4)) + rc_inner) + 64)]));
          conv2d_nchw[(((ff_inner * 14) + yy_inner) + 112)] = (conv2d_nchw[(((ff_inner * 14) + yy_inner) + 112)] + (pad_temp_shared[((((rc_inner * 784) + (((((int)threadIdx.x) % 56) / 28) * 392)) + (yy_inner * 28)) + (((int)threadIdx.x) % 28))] * kernel_shared[(((((((int)threadIdx.x) / 56) * 16) + (ff_inner * 4)) + rc_inner) + 128)]));
          conv2d_nchw[(((ff_inner * 14) + yy_inner) + 168)] = (conv2d_nchw[(((ff_inner * 14) + yy_inner) + 168)] + (pad_temp_shared[((((rc_inner * 784) + (((((int)threadIdx.x) % 56) / 28) * 392)) + (yy_inner * 28)) + (((int)threadIdx.x) % 28))] * kernel_shared[(((((((int)threadIdx.x) / 56) * 16) + (ff_inner * 4)) + rc_inner) + 192)]));
        }
      }
    }
  }
  for (int i1_inner = 0; i1_inner < 4; ++i1_inner) {
    for (int i2_inner = 0; i2_inner < 14; ++i2_inner) {
      compute[((((((((((int)threadIdx.x) / 56) * 50176) + (i1_inner * 12544)) + ((((int)blockIdx.x) >> 2) * 3136)) + (((((int)threadIdx.x) % 56) / 28) * 1568)) + (i2_inner * 112)) + ((((int)blockIdx.x) & 3) * 28)) + (((int)threadIdx.x) % 28))] = max(conv2d_nchw[((i1_inner * 14) + i2_inner)], 0.000000e+00f);
      compute[(((((((((((int)threadIdx.x) / 56) * 50176) + (i1_inner * 12544)) + ((((int)blockIdx.x) >> 2) * 3136)) + (((((int)threadIdx.x) % 56) / 28) * 1568)) + (i2_inner * 112)) + ((((int)blockIdx.x) & 3) * 28)) + (((int)threadIdx.x) % 28)) + 200704)] = max(conv2d_nchw[(((i1_inner * 14) + i2_inner) + 56)], 0.000000e+00f);
      compute[(((((((((((int)threadIdx.x) / 56) * 50176) + (i1_inner * 12544)) + ((((int)blockIdx.x) >> 2) * 3136)) + (((((int)threadIdx.x) % 56) / 28) * 1568)) + (i2_inner * 112)) + ((((int)blockIdx.x) & 3) * 28)) + (((int)threadIdx.x) % 28)) + 401408)] = max(conv2d_nchw[(((i1_inner * 14) + i2_inner) + 112)], 0.000000e+00f);
      compute[(((((((((((int)threadIdx.x) / 56) * 50176) + (i1_inner * 12544)) + ((((int)blockIdx.x) >> 2) * 3136)) + (((((int)threadIdx.x) % 56) / 28) * 1568)) + (i2_inner * 112)) + ((((int)blockIdx.x) & 3) * 28)) + (((int)threadIdx.x) % 28)) + 602112)] = max(conv2d_nchw[(((i1_inner * 14) + i2_inner) + 168)], 0.000000e+00f);
    }
  }
}


