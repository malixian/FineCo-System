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
  float conv2d_nchw_local[2];
  __shared__ float pad_temp_shared[72];
  __shared__ float kernel_shared[3072];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 64; ++rc_outer_outer) {
    for (int rx_outer_outer = 0; rx_outer_outer < 3; ++rx_outer_outer) {
      __syncthreads();
      if (((int)threadIdx.x) < 72) {
        pad_temp_shared[((int)threadIdx.x)] = (((((1 <= (((int)threadIdx.x) % 9)) && ((((int)threadIdx.x) % 9) < 8)) && (1 <= (rx_outer_outer + (((int)blockIdx.x) % 7)))) && ((rx_outer_outer + (((int)blockIdx.x) % 7)) < 8)) ? data[((((((rc_outer_outer * 392) + ((((int)threadIdx.x) / 9) * 49)) + ((((int)threadIdx.x) % 9) * 7)) + rx_outer_outer) + (((int)blockIdx.x) % 7)) - 8)] : 0.000000e+00f);
      }
      kernel_shared[((int)threadIdx.x)] = kernel[((((((((int)blockIdx.x) / 7) * 589824) + ((((int)threadIdx.x) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) % 24) * 3)) + rx_outer_outer)];
      kernel_shared[(((int)threadIdx.x) + 448)] = kernel[((((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 448) / 24) * 4608)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) + 16) % 24) * 3)) + rx_outer_outer)];
      kernel_shared[(((int)threadIdx.x) + 896)] = kernel[((((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 896) / 24) * 4608)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) + 8) % 24) * 3)) + rx_outer_outer)];
      kernel_shared[(((int)threadIdx.x) + 1344)] = kernel[(((((((((int)blockIdx.x) / 7) * 589824) + ((((int)threadIdx.x) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) % 24) * 3)) + rx_outer_outer) + 258048)];
      kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[((((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 1792) / 24) * 4608)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) + 16) % 24) * 3)) + rx_outer_outer)];
      kernel_shared[(((int)threadIdx.x) + 2240)] = kernel[((((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 2240) / 24) * 4608)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) + 8) % 24) * 3)) + rx_outer_outer)];
      if (((int)threadIdx.x) < 384) {
        kernel_shared[(((int)threadIdx.x) + 2688)] = kernel[(((((((((int)blockIdx.x) / 7) * 589824) + ((((int)threadIdx.x) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) % 24) * 3)) + rx_outer_outer) + 516096)];
      }
      __syncthreads();
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((int)threadIdx.x) % 7)] * kernel_shared[((((int)threadIdx.x) / 7) * 48)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 1)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 1)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 2)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 2)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 9)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 3)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 10)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 4)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 11)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 5)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 18)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 6)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 19)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 7)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 20)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 8)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 27)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 9)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 28)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 10)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 29)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 11)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 36)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 12)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 37)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 13)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 38)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 14)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 45)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 15)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 46)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 16)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 47)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 17)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 54)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 18)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 55)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 19)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 56)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 20)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 63)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 21)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 64)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 22)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 65)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 23)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((int)threadIdx.x) % 7)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 24)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 1)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 25)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 2)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 26)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 9)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 27)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 10)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 28)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 11)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 29)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 18)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 30)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 19)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 31)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 20)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 32)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 27)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 33)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 28)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 34)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 29)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 35)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 36)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 36)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 37)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 37)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 38)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 38)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 45)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 39)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 46)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 40)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 47)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 41)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 54)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 42)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 55)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 43)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 56)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 44)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 63)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 45)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 64)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 46)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) % 7) + 65)] * kernel_shared[(((((int)threadIdx.x) / 7) * 48) + 47)]));
    }
  }
  for (int ff_inner = 0; ff_inner < 2; ++ff_inner) {
    conv2d_nchw[((((((((int)blockIdx.x) / 7) * 6272) + ((((int)threadIdx.x) / 7) * 98)) + (ff_inner * 49)) + ((((int)threadIdx.x) % 7) * 7)) + (((int)blockIdx.x) % 7))] = conv2d_nchw_local[ff_inner];
  }
}


