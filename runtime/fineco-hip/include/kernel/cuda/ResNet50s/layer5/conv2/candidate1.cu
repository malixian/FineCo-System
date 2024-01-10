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
extern "C" __global__ void __launch_bounds__(224) candidate1(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[7];
  __shared__ float pad_temp_shared[648];
  __shared__ float kernel_shared[2304];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 64; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = (((((9 <= (((int)threadIdx.x) % 81)) && ((((int)threadIdx.x) % 81) < 72)) && (1 <= (((int)threadIdx.x) % 9))) && ((((int)threadIdx.x) % 9) < 8)) ? data[(((((rc_outer_outer * 392) + ((((int)threadIdx.x) / 81) * 49)) + (((((int)threadIdx.x) % 81) / 9) * 7)) + (((int)threadIdx.x) % 9)) - 8)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 224)] = (((((9 <= ((((int)threadIdx.x) + 62) % 81)) && (((((int)threadIdx.x) + 62) % 81) < 72)) && (1 <= ((((int)threadIdx.x) + 8) % 9))) && (((((int)threadIdx.x) + 8) % 9) < 8)) ? data[(((((rc_outer_outer * 392) + (((((int)threadIdx.x) + 224) / 81) * 49)) + ((((((int)threadIdx.x) + 62) % 81) / 9) * 7)) + ((((int)threadIdx.x) + 8) % 9)) - 8)] : 0.000000e+00f);
    if (((int)threadIdx.x) < 200) {
      pad_temp_shared[(((int)threadIdx.x) + 448)] = (((((9 <= ((((int)threadIdx.x) + 43) % 81)) && (((((int)threadIdx.x) + 43) % 81) < 72)) && (1 <= ((((int)threadIdx.x) + 7) % 9))) && (((((int)threadIdx.x) + 7) % 9) < 8)) ? data[(((((rc_outer_outer * 392) + (((((int)threadIdx.x) + 448) / 81) * 49)) + ((((((int)threadIdx.x) + 43) % 81) / 9) * 7)) + ((((int)threadIdx.x) + 7) % 9)) - 8)] : 0.000000e+00f);
    }
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)blockIdx.x) * 147456) + ((((int)threadIdx.x) / 72) * 4608)) + (rc_outer_outer * 72)) + (((int)threadIdx.x) % 72))];
    kernel_shared[(((int)threadIdx.x) + 224)] = kernel[((((((int)blockIdx.x) * 147456) + (((((int)threadIdx.x) + 224) / 72) * 4608)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 8) % 72))];
    kernel_shared[(((int)threadIdx.x) + 448)] = kernel[((((((int)blockIdx.x) * 147456) + (((((int)threadIdx.x) + 448) / 72) * 4608)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 16) % 72))];
    kernel_shared[(((int)threadIdx.x) + 672)] = kernel[((((((int)blockIdx.x) * 147456) + (((((int)threadIdx.x) + 672) / 72) * 4608)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 24) % 72))];
    kernel_shared[(((int)threadIdx.x) + 896)] = kernel[((((((int)blockIdx.x) * 147456) + (((((int)threadIdx.x) + 896) / 72) * 4608)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 32) % 72))];
    kernel_shared[(((int)threadIdx.x) + 1120)] = kernel[((((((int)blockIdx.x) * 147456) + (((((int)threadIdx.x) + 1120) / 72) * 4608)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 40) % 72))];
    kernel_shared[(((int)threadIdx.x) + 1344)] = kernel[((((((int)blockIdx.x) * 147456) + (((((int)threadIdx.x) + 1344) / 72) * 4608)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 48) % 72))];
    kernel_shared[(((int)threadIdx.x) + 1568)] = kernel[((((((int)blockIdx.x) * 147456) + (((((int)threadIdx.x) + 1568) / 72) * 4608)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 56) % 72))];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[((((((int)blockIdx.x) * 147456) + (((((int)threadIdx.x) + 1792) / 72) * 4608)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 64) % 72))];
    kernel_shared[(((int)threadIdx.x) + 2016)] = kernel[(((((((int)blockIdx.x) * 147456) + ((((int)threadIdx.x) / 72) * 4608)) + (rc_outer_outer * 72)) + (((int)threadIdx.x) % 72)) + 129024)];
    if (((int)threadIdx.x) < 64) {
      kernel_shared[(((int)threadIdx.x) + 2240)] = kernel[((((((int)blockIdx.x) * 147456) + (((((int)threadIdx.x) + 2240) / 72) * 4608)) + (rc_outer_outer * 72)) + (((int)threadIdx.x) + 8))];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 4; ++rc_outer_inner) {
      for (int rx_outer_inner = 0; rx_outer_inner < 3; ++rx_outer_inner) {
        conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7))] * kernel_shared[((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner)]));
        conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 9)] * kernel_shared[((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner)]));
        conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 18)] * kernel_shared[((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner)]));
        conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 27)] * kernel_shared[((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner)]));
        conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 36)] * kernel_shared[((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner)]));
        conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 45)] * kernel_shared[((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner)]));
        conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 54)] * kernel_shared[((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner)]));
        conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 9)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner) + 3)]));
        conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 18)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner) + 3)]));
        conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 27)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner) + 3)]));
        conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 36)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner) + 3)]));
        conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 45)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner) + 3)]));
        conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 54)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner) + 3)]));
        conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 63)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner) + 3)]));
        conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 18)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner) + 6)]));
        conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 27)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner) + 6)]));
        conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 36)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner) + 6)]));
        conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 45)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner) + 6)]));
        conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 54)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner) + 6)]));
        conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 63)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner) + 6)]));
        conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 72)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner) + 6)]));
        conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 81)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner) + 9)]));
        conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 90)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner) + 9)]));
        conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 99)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner) + 9)]));
        conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 108)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner) + 9)]));
        conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 117)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner) + 9)]));
        conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 126)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner) + 9)]));
        conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 135)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner) + 9)]));
        conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 90)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner) + 12)]));
        conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 99)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner) + 12)]));
        conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 108)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner) + 12)]));
        conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 117)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner) + 12)]));
        conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 126)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner) + 12)]));
        conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 135)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner) + 12)]));
        conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 144)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner) + 12)]));
        conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 99)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner) + 15)]));
        conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 108)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner) + 15)]));
        conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 117)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner) + 15)]));
        conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 126)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner) + 15)]));
        conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 135)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner) + 15)]));
        conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 144)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner) + 15)]));
        conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[((((rc_outer_inner * 162) + rx_outer_inner) + (((int)threadIdx.x) % 7)) + 153)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 72) + (rc_outer_inner * 18)) + rx_outer_inner) + 15)]));
      }
    }
  }
  conv2d_nchw[(((((int)blockIdx.x) * 1568) + ((((int)threadIdx.x) / 7) * 49)) + (((int)threadIdx.x) % 7))] = conv2d_nchw_local[0];
  conv2d_nchw[((((((int)blockIdx.x) * 1568) + ((((int)threadIdx.x) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 7)] = conv2d_nchw_local[1];
  conv2d_nchw[((((((int)blockIdx.x) * 1568) + ((((int)threadIdx.x) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 14)] = conv2d_nchw_local[2];
  conv2d_nchw[((((((int)blockIdx.x) * 1568) + ((((int)threadIdx.x) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 21)] = conv2d_nchw_local[3];
  conv2d_nchw[((((((int)blockIdx.x) * 1568) + ((((int)threadIdx.x) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 28)] = conv2d_nchw_local[4];
  conv2d_nchw[((((((int)blockIdx.x) * 1568) + ((((int)threadIdx.x) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 35)] = conv2d_nchw_local[5];
  conv2d_nchw[((((((int)blockIdx.x) * 1568) + ((((int)threadIdx.x) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 42)] = conv2d_nchw_local[6];
}


