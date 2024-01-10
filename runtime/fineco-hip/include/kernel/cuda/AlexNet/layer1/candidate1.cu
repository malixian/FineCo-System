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
extern "C" __global__ void __launch_bounds__(220) candidate1(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ compute, float* __restrict__ bias) {
  float conv2d_nchw[15];
  __shared__ float pad_temp_shared[6129];
  __shared__ float kernel_shared[1452];
  conv2d_nchw[0] = 0.000000e+00f;
  conv2d_nchw[5] = 0.000000e+00f;
  conv2d_nchw[10] = 0.000000e+00f;
  conv2d_nchw[1] = 0.000000e+00f;
  conv2d_nchw[6] = 0.000000e+00f;
  conv2d_nchw[11] = 0.000000e+00f;
  conv2d_nchw[2] = 0.000000e+00f;
  conv2d_nchw[7] = 0.000000e+00f;
  conv2d_nchw[12] = 0.000000e+00f;
  conv2d_nchw[3] = 0.000000e+00f;
  conv2d_nchw[8] = 0.000000e+00f;
  conv2d_nchw[13] = 0.000000e+00f;
  conv2d_nchw[4] = 0.000000e+00f;
  conv2d_nchw[9] = 0.000000e+00f;
  conv2d_nchw[14] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 3; ++rc_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 28; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
      if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 220) + ((int)threadIdx.x)) < 6129) {
        pad_temp_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 220) + ((int)threadIdx.x))] = (((((1 <= (((((int)blockIdx.x) % 11) * 10) + (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 220) + ((int)threadIdx.x)) / 454))) && ((((((int)blockIdx.x) % 11) * 10) + (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 220) + ((int)threadIdx.x)) / 454)) < 113)) && (2 <= (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 220) + ((int)threadIdx.x)) % 227))) && ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 220) + ((int)threadIdx.x)) % 227) < 226)) ? data[(((((rc_outer_outer * 50176) + ((((int)blockIdx.x) % 11) * 4480)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 220) + ((int)threadIdx.x)) / 227) * 224)) + (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 220) + ((int)threadIdx.x)) % 227)) - 450)] : 0.000000e+00f);
      }
    }
    kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) / 11) * 4356) + ((((int)threadIdx.x) / 121) * 363)) + (rc_outer_outer * 121)) + (((int)threadIdx.x) % 121))];
    kernel_shared[(((int)threadIdx.x) + 220)] = kernel[((((((((int)blockIdx.x) / 11) * 4356) + (((((int)threadIdx.x) + 220) / 121) * 363)) + (rc_outer_outer * 121)) + ((((((int)threadIdx.x) / 11) + 9) % 11) * 11)) + (((int)threadIdx.x) % 11))];
    kernel_shared[(((int)threadIdx.x) + 440)] = kernel[((((((((int)blockIdx.x) / 11) * 4356) + (((((int)threadIdx.x) + 440) / 121) * 363)) + (rc_outer_outer * 121)) + ((((((int)threadIdx.x) / 11) + 7) % 11) * 11)) + (((int)threadIdx.x) % 11))];
    kernel_shared[(((int)threadIdx.x) + 660)] = kernel[((((((((int)blockIdx.x) / 11) * 4356) + (((((int)threadIdx.x) + 660) / 121) * 363)) + (rc_outer_outer * 121)) + ((((((int)threadIdx.x) / 11) + 5) % 11) * 11)) + (((int)threadIdx.x) % 11))];
    kernel_shared[(((int)threadIdx.x) + 880)] = kernel[((((((((int)blockIdx.x) / 11) * 4356) + (((((int)threadIdx.x) + 880) / 121) * 363)) + (rc_outer_outer * 121)) + ((((((int)threadIdx.x) / 11) + 3) % 11) * 11)) + (((int)threadIdx.x) % 11))];
    kernel_shared[(((int)threadIdx.x) + 1100)] = kernel[((((((((int)blockIdx.x) / 11) * 4356) + (((((int)threadIdx.x) + 1100) / 121) * 363)) + (rc_outer_outer * 121)) + ((((((int)threadIdx.x) / 11) + 1) % 11) * 11)) + (((int)threadIdx.x) % 11))];
    if (((int)threadIdx.x) < 132) {
      kernel_shared[(((int)threadIdx.x) + 1320)] = kernel[((((((((int)blockIdx.x) / 11) * 4356) + (((((int)threadIdx.x) + 1320) / 121) * 363)) + (rc_outer_outer * 121)) + ((((((int)threadIdx.x) / 11) + 10) % 11) * 11)) + (((int)threadIdx.x) % 11))];
    }
    __syncthreads();
    for (int rx_outer_inner = 0; rx_outer_inner < 11; ++rx_outer_inner) {
      for (int ry_inner = 0; ry_inner < 11; ++ry_inner) {
        conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((ry_inner * 227) + ((((int)threadIdx.x) % 55) * 4)) + rx_outer_inner)] * kernel_shared[((((((int)threadIdx.x) / 55) * 121) + (ry_inner * 11)) + rx_outer_inner)]));
        conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[(((ry_inner * 227) + ((((int)threadIdx.x) % 55) * 4)) + rx_outer_inner)] * kernel_shared[(((((((int)threadIdx.x) / 55) * 121) + (ry_inner * 11)) + rx_outer_inner) + 484)]));
        conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[(((ry_inner * 227) + ((((int)threadIdx.x) % 55) * 4)) + rx_outer_inner)] * kernel_shared[(((((((int)threadIdx.x) / 55) * 121) + (ry_inner * 11)) + rx_outer_inner) + 968)]));
        conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((ry_inner * 227) + ((((int)threadIdx.x) % 55) * 4)) + rx_outer_inner) + 908)] * kernel_shared[((((((int)threadIdx.x) / 55) * 121) + (ry_inner * 11)) + rx_outer_inner)]));
        conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[((((ry_inner * 227) + ((((int)threadIdx.x) % 55) * 4)) + rx_outer_inner) + 908)] * kernel_shared[(((((((int)threadIdx.x) / 55) * 121) + (ry_inner * 11)) + rx_outer_inner) + 484)]));
        conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[((((ry_inner * 227) + ((((int)threadIdx.x) % 55) * 4)) + rx_outer_inner) + 908)] * kernel_shared[(((((((int)threadIdx.x) / 55) * 121) + (ry_inner * 11)) + rx_outer_inner) + 968)]));
        conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((ry_inner * 227) + ((((int)threadIdx.x) % 55) * 4)) + rx_outer_inner) + 1816)] * kernel_shared[((((((int)threadIdx.x) / 55) * 121) + (ry_inner * 11)) + rx_outer_inner)]));
        conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[((((ry_inner * 227) + ((((int)threadIdx.x) % 55) * 4)) + rx_outer_inner) + 1816)] * kernel_shared[(((((((int)threadIdx.x) / 55) * 121) + (ry_inner * 11)) + rx_outer_inner) + 484)]));
        conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[((((ry_inner * 227) + ((((int)threadIdx.x) % 55) * 4)) + rx_outer_inner) + 1816)] * kernel_shared[(((((((int)threadIdx.x) / 55) * 121) + (ry_inner * 11)) + rx_outer_inner) + 968)]));
        conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((ry_inner * 227) + ((((int)threadIdx.x) % 55) * 4)) + rx_outer_inner) + 2724)] * kernel_shared[((((((int)threadIdx.x) / 55) * 121) + (ry_inner * 11)) + rx_outer_inner)]));
        conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[((((ry_inner * 227) + ((((int)threadIdx.x) % 55) * 4)) + rx_outer_inner) + 2724)] * kernel_shared[(((((((int)threadIdx.x) / 55) * 121) + (ry_inner * 11)) + rx_outer_inner) + 484)]));
        conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[((((ry_inner * 227) + ((((int)threadIdx.x) % 55) * 4)) + rx_outer_inner) + 2724)] * kernel_shared[(((((((int)threadIdx.x) / 55) * 121) + (ry_inner * 11)) + rx_outer_inner) + 968)]));
        conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[((((ry_inner * 227) + ((((int)threadIdx.x) % 55) * 4)) + rx_outer_inner) + 3632)] * kernel_shared[((((((int)threadIdx.x) / 55) * 121) + (ry_inner * 11)) + rx_outer_inner)]));
        conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[((((ry_inner * 227) + ((((int)threadIdx.x) % 55) * 4)) + rx_outer_inner) + 3632)] * kernel_shared[(((((((int)threadIdx.x) / 55) * 121) + (ry_inner * 11)) + rx_outer_inner) + 484)]));
        conv2d_nchw[14] = (conv2d_nchw[14] + (pad_temp_shared[((((ry_inner * 227) + ((((int)threadIdx.x) % 55) * 4)) + rx_outer_inner) + 3632)] * kernel_shared[(((((((int)threadIdx.x) / 55) * 121) + (ry_inner * 11)) + rx_outer_inner) + 968)]));
      }
    }
  }
  for (int i2_inner = 0; i2_inner < 5; ++i2_inner) {
    compute[((((((((int)blockIdx.x) / 11) * 36300) + ((((int)threadIdx.x) / 55) * 3025)) + ((((int)blockIdx.x) % 11) * 275)) + (i2_inner * 55)) + (((int)threadIdx.x) % 55))] = max((conv2d_nchw[i2_inner] + bias[(((((int)blockIdx.x) / 11) * 12) + (((int)threadIdx.x) / 55))]), 0.000000e+00f);
    compute[(((((((((int)blockIdx.x) / 11) * 36300) + ((((int)threadIdx.x) / 55) * 3025)) + ((((int)blockIdx.x) % 11) * 275)) + (i2_inner * 55)) + (((int)threadIdx.x) % 55)) + 12100)] = max((conv2d_nchw[(i2_inner + 5)] + bias[((((((int)blockIdx.x) / 11) * 12) + (((int)threadIdx.x) / 55)) + 4)]), 0.000000e+00f);
    compute[(((((((((int)blockIdx.x) / 11) * 36300) + ((((int)threadIdx.x) / 55) * 3025)) + ((((int)blockIdx.x) % 11) * 275)) + (i2_inner * 55)) + (((int)threadIdx.x) % 55)) + 24200)] = max((conv2d_nchw[(i2_inner + 10)] + bias[((((((int)blockIdx.x) / 11) * 12) + (((int)threadIdx.x) / 55)) + 8)]), 0.000000e+00f);
  }
}


