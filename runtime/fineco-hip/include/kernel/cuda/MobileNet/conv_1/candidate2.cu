#include "hip/hip_runtime.h"

#define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long

extern "C" __global__ void __launch_bounds__(448) candidate2(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ compute, float* __restrict__ bias) {
  float conv2d_nchw[32];
  __shared__ float pad_temp_shared[2025];
  __shared__ float kernel_shared[288];
  for (int ff_outer_inner_init = 0; ff_outer_inner_init < 2; ++ff_outer_inner_init) {
    conv2d_nchw[(ff_outer_inner_init * 4)] = 0.000000e+00f;
    conv2d_nchw[((ff_outer_inner_init * 4) + 8)] = 0.000000e+00f;
    conv2d_nchw[((ff_outer_inner_init * 4) + 16)] = 0.000000e+00f;
    conv2d_nchw[((ff_outer_inner_init * 4) + 24)] = 0.000000e+00f;
    conv2d_nchw[((ff_outer_inner_init * 4) + 1)] = 0.000000e+00f;
    conv2d_nchw[((ff_outer_inner_init * 4) + 9)] = 0.000000e+00f;
    conv2d_nchw[((ff_outer_inner_init * 4) + 17)] = 0.000000e+00f;
    conv2d_nchw[((ff_outer_inner_init * 4) + 25)] = 0.000000e+00f;
    conv2d_nchw[((ff_outer_inner_init * 4) + 2)] = 0.000000e+00f;
    conv2d_nchw[((ff_outer_inner_init * 4) + 10)] = 0.000000e+00f;
    conv2d_nchw[((ff_outer_inner_init * 4) + 18)] = 0.000000e+00f;
    conv2d_nchw[((ff_outer_inner_init * 4) + 26)] = 0.000000e+00f;
    conv2d_nchw[((ff_outer_inner_init * 4) + 3)] = 0.000000e+00f;
    conv2d_nchw[((ff_outer_inner_init * 4) + 11)] = 0.000000e+00f;
    conv2d_nchw[((ff_outer_inner_init * 4) + 19)] = 0.000000e+00f;
    conv2d_nchw[((ff_outer_inner_init * 4) + 27)] = 0.000000e+00f;
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 3; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = (((1 <= ((((int)blockIdx.x) * 8) + (((int)threadIdx.x) / 225))) && (1 <= (((int)threadIdx.x) % 225))) ? data[(((((rc_outer_outer * 50176) + (((int)blockIdx.x) * 1792)) + ((((int)threadIdx.x) / 225) * 224)) + (((int)threadIdx.x) % 225)) - 225)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 448)] = ((1 <= ((((int)threadIdx.x) + 223) % 225)) ? data[(((((rc_outer_outer * 50176) + (((int)blockIdx.x) * 1792)) + (((((int)threadIdx.x) + 448) / 225) * 224)) + ((((int)threadIdx.x) + 223) % 225)) - 225)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 896)] = ((1 <= ((((int)threadIdx.x) + 221) % 225)) ? data[(((((rc_outer_outer * 50176) + (((int)blockIdx.x) * 1792)) + (((((int)threadIdx.x) + 896) / 225) * 224)) + ((((int)threadIdx.x) + 221) % 225)) - 225)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1344)] = ((1 <= ((((int)threadIdx.x) + 219) % 225)) ? data[(((((rc_outer_outer * 50176) + (((int)blockIdx.x) * 1792)) + (((((int)threadIdx.x) + 1344) / 225) * 224)) + ((((int)threadIdx.x) + 219) % 225)) - 225)] : 0.000000e+00f);
    if (((int)threadIdx.x) < 233) {
      pad_temp_shared[(((int)threadIdx.x) + 1792)] = ((1 <= ((((int)threadIdx.x) + 217) % 225)) ? data[(((((rc_outer_outer * 50176) + (((int)blockIdx.x) * 1792)) + (((((int)threadIdx.x) + 1792) / 225) * 224)) + ((((int)threadIdx.x) + 217) % 225)) - 225)] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 288) {
      kernel_shared[((int)threadIdx.x)] = kernel[((((((int)threadIdx.x) / 9) * 27) + (rc_outer_outer * 9)) + (((int)threadIdx.x) % 9))];
    }
    __syncthreads();
    for (int ry_outer_inner = 0; ry_outer_inner < 3; ++ry_outer_inner) {
      for (int ff_outer_inner = 0; ff_outer_inner < 2; ++ff_outer_inner) {
        for (int yy_outer_inner = 0; yy_outer_inner < 2; ++yy_outer_inner) {
          for (int rx_inner = 0; rx_inner < 3; ++rx_inner) {
            conv2d_nchw[((ff_outer_inner * 4) + (yy_outer_inner * 2))] = (conv2d_nchw[((ff_outer_inner * 4) + (yy_outer_inner * 2))] + (pad_temp_shared[((((yy_outer_inner * 900) + (ry_outer_inner * 225)) + ((((int)threadIdx.x) % 112) * 2)) + rx_inner)] * kernel_shared[(((((((int)threadIdx.x) / 112) * 18) + (ff_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner)]));
            conv2d_nchw[(((ff_outer_inner * 4) + (yy_outer_inner * 2)) + 8)] = (conv2d_nchw[(((ff_outer_inner * 4) + (yy_outer_inner * 2)) + 8)] + (pad_temp_shared[((((yy_outer_inner * 900) + (ry_outer_inner * 225)) + ((((int)threadIdx.x) % 112) * 2)) + rx_inner)] * kernel_shared[((((((((int)threadIdx.x) / 112) * 18) + (ff_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 72)]));
            conv2d_nchw[(((ff_outer_inner * 4) + (yy_outer_inner * 2)) + 16)] = (conv2d_nchw[(((ff_outer_inner * 4) + (yy_outer_inner * 2)) + 16)] + (pad_temp_shared[((((yy_outer_inner * 900) + (ry_outer_inner * 225)) + ((((int)threadIdx.x) % 112) * 2)) + rx_inner)] * kernel_shared[((((((((int)threadIdx.x) / 112) * 18) + (ff_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 144)]));
            conv2d_nchw[(((ff_outer_inner * 4) + (yy_outer_inner * 2)) + 24)] = (conv2d_nchw[(((ff_outer_inner * 4) + (yy_outer_inner * 2)) + 24)] + (pad_temp_shared[((((yy_outer_inner * 900) + (ry_outer_inner * 225)) + ((((int)threadIdx.x) % 112) * 2)) + rx_inner)] * kernel_shared[((((((((int)threadIdx.x) / 112) * 18) + (ff_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 216)]));
            conv2d_nchw[(((ff_outer_inner * 4) + (yy_outer_inner * 2)) + 1)] = (conv2d_nchw[(((ff_outer_inner * 4) + (yy_outer_inner * 2)) + 1)] + (pad_temp_shared[(((((yy_outer_inner * 900) + (ry_outer_inner * 225)) + ((((int)threadIdx.x) % 112) * 2)) + rx_inner) + 450)] * kernel_shared[(((((((int)threadIdx.x) / 112) * 18) + (ff_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner)]));
            conv2d_nchw[(((ff_outer_inner * 4) + (yy_outer_inner * 2)) + 9)] = (conv2d_nchw[(((ff_outer_inner * 4) + (yy_outer_inner * 2)) + 9)] + (pad_temp_shared[(((((yy_outer_inner * 900) + (ry_outer_inner * 225)) + ((((int)threadIdx.x) % 112) * 2)) + rx_inner) + 450)] * kernel_shared[((((((((int)threadIdx.x) / 112) * 18) + (ff_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 72)]));
            conv2d_nchw[(((ff_outer_inner * 4) + (yy_outer_inner * 2)) + 17)] = (conv2d_nchw[(((ff_outer_inner * 4) + (yy_outer_inner * 2)) + 17)] + (pad_temp_shared[(((((yy_outer_inner * 900) + (ry_outer_inner * 225)) + ((((int)threadIdx.x) % 112) * 2)) + rx_inner) + 450)] * kernel_shared[((((((((int)threadIdx.x) / 112) * 18) + (ff_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 144)]));
            conv2d_nchw[(((ff_outer_inner * 4) + (yy_outer_inner * 2)) + 25)] = (conv2d_nchw[(((ff_outer_inner * 4) + (yy_outer_inner * 2)) + 25)] + (pad_temp_shared[(((((yy_outer_inner * 900) + (ry_outer_inner * 225)) + ((((int)threadIdx.x) % 112) * 2)) + rx_inner) + 450)] * kernel_shared[((((((((int)threadIdx.x) / 112) * 18) + (ff_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 216)]));
          }
        }
      }
    }
  }
  for (int i1_inner = 0; i1_inner < 2; ++i1_inner) {
    for (int i2_inner = 0; i2_inner < 4; ++i2_inner) {
      compute[((((((((int)threadIdx.x) / 112) * 25088) + (i1_inner * 12544)) + (((int)blockIdx.x) * 448)) + (i2_inner * 112)) + (((int)threadIdx.x) % 112))] = max((conv2d_nchw[((i1_inner * 4) + i2_inner)] + bias[(((((int)threadIdx.x) / 112) * 2) + i1_inner)]), 0.000000e+00f);
      compute[(((((((((int)threadIdx.x) / 112) * 25088) + (i1_inner * 12544)) + (((int)blockIdx.x) * 448)) + (i2_inner * 112)) + (((int)threadIdx.x) % 112)) + 100352)] = max((conv2d_nchw[(((i1_inner * 4) + i2_inner) + 8)] + bias[((((((int)threadIdx.x) / 112) * 2) + i1_inner) + 8)]), 0.000000e+00f);
      compute[(((((((((int)threadIdx.x) / 112) * 25088) + (i1_inner * 12544)) + (((int)blockIdx.x) * 448)) + (i2_inner * 112)) + (((int)threadIdx.x) % 112)) + 200704)] = max((conv2d_nchw[(((i1_inner * 4) + i2_inner) + 16)] + bias[((((((int)threadIdx.x) / 112) * 2) + i1_inner) + 16)]), 0.000000e+00f);
      compute[(((((((((int)threadIdx.x) / 112) * 25088) + (i1_inner * 12544)) + (((int)blockIdx.x) * 448)) + (i2_inner * 112)) + (((int)threadIdx.x) % 112)) + 301056)] = max((conv2d_nchw[(((i1_inner * 4) + i2_inner) + 24)] + bias[((((((int)threadIdx.x) / 112) * 2) + i1_inner) + 24)]), 0.000000e+00f);
    }
  }
}


