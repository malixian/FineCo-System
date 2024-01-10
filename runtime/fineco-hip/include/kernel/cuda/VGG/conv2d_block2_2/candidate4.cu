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
extern "C" __global__ void __launch_bounds__(256) candidate4(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[128];
  __shared__ float pad_temp_shared[1296];
  __shared__ float kernel_shared[4608];
  for (int yy_c_outer_inner_init = 0; yy_c_outer_inner_init < 4; ++yy_c_outer_inner_init) {
    for (int ff_c_inner_init = 0; ff_c_inner_init < 2; ++ff_c_inner_init) {
      conv2d_nchw_local[((ff_c_inner_init * 32) + (yy_c_outer_inner_init * 8))] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_inner_init * 32) + (yy_c_outer_inner_init * 8)) + 64)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_inner_init * 32) + (yy_c_outer_inner_init * 8)) + 1)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_inner_init * 32) + (yy_c_outer_inner_init * 8)) + 65)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_inner_init * 32) + (yy_c_outer_inner_init * 8)) + 2)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_inner_init * 32) + (yy_c_outer_inner_init * 8)) + 66)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_inner_init * 32) + (yy_c_outer_inner_init * 8)) + 3)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_inner_init * 32) + (yy_c_outer_inner_init * 8)) + 67)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_inner_init * 32) + (yy_c_outer_inner_init * 8)) + 4)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_inner_init * 32) + (yy_c_outer_inner_init * 8)) + 68)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_inner_init * 32) + (yy_c_outer_inner_init * 8)) + 5)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_inner_init * 32) + (yy_c_outer_inner_init * 8)) + 69)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_inner_init * 32) + (yy_c_outer_inner_init * 8)) + 6)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_inner_init * 32) + (yy_c_outer_inner_init * 8)) + 70)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_inner_init * 32) + (yy_c_outer_inner_init * 8)) + 7)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_inner_init * 32) + (yy_c_outer_inner_init * 8)) + 71)] = 0.000000e+00f;
    }
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 32; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = ((((1 <= (((((int)blockIdx.x) / 7) * 16) + (((int)threadIdx.x) / 18))) && (1 <= (((((int)blockIdx.x) % 7) * 16) + (((int)threadIdx.x) % 18)))) && ((((((int)blockIdx.x) % 7) * 16) + (((int)threadIdx.x) % 18)) < 113)) ? data[((((((rc_outer_outer * 50176) + ((((int)blockIdx.x) / 7) * 1792)) + ((((int)threadIdx.x) / 18) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) % 18)) - 113)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 256)] = (((((1 <= (((((int)blockIdx.x) / 7) * 16) + ((((((int)threadIdx.x) >> 1) + 128) % 162) / 9))) && ((((((int)blockIdx.x) / 7) * 16) + ((((((int)threadIdx.x) >> 1) + 128) % 162) / 9)) < 113)) && (1 <= (((((int)blockIdx.x) % 7) * 16) + ((((int)threadIdx.x) + 4) % 18)))) && ((((((int)blockIdx.x) % 7) * 16) + ((((int)threadIdx.x) + 4) % 18)) < 113)) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 256) / 324) * 12544)) + ((((int)blockIdx.x) / 7) * 1792)) + (((((((int)threadIdx.x) >> 1) + 128) % 162) / 9) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + ((((int)threadIdx.x) + 4) % 18)) - 113)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 512)] = (((((1 <= (((((int)blockIdx.x) / 7) * 16) + ((((((int)threadIdx.x) >> 1) + 94) % 162) / 9))) && ((((((int)blockIdx.x) / 7) * 16) + ((((((int)threadIdx.x) >> 1) + 94) % 162) / 9)) < 113)) && (1 <= (((((int)blockIdx.x) % 7) * 16) + ((((int)threadIdx.x) + 8) % 18)))) && ((((((int)blockIdx.x) % 7) * 16) + ((((int)threadIdx.x) + 8) % 18)) < 113)) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 512) / 324) * 12544)) + ((((int)blockIdx.x) / 7) * 1792)) + (((((((int)threadIdx.x) >> 1) + 94) % 162) / 9) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + ((((int)threadIdx.x) + 8) % 18)) - 113)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 768)] = (((((1 <= (((((int)blockIdx.x) / 7) * 16) + ((((((int)threadIdx.x) >> 1) + 60) % 162) / 9))) && ((((((int)blockIdx.x) / 7) * 16) + ((((((int)threadIdx.x) >> 1) + 60) % 162) / 9)) < 113)) && (1 <= (((((int)blockIdx.x) % 7) * 16) + ((((int)threadIdx.x) + 12) % 18)))) && ((((((int)blockIdx.x) % 7) * 16) + ((((int)threadIdx.x) + 12) % 18)) < 113)) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 768) / 324) * 12544)) + ((((int)blockIdx.x) / 7) * 1792)) + (((((((int)threadIdx.x) >> 1) + 60) % 162) / 9) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + ((((int)threadIdx.x) + 12) % 18)) - 113)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1024)] = (((((((((int)blockIdx.x) / 7) * 16) + ((((((int)threadIdx.x) >> 1) + 26) % 162) / 9)) < 113) && (1 <= (((((int)blockIdx.x) % 7) * 16) + ((((int)threadIdx.x) + 16) % 18)))) && ((((((int)blockIdx.x) % 7) * 16) + ((((int)threadIdx.x) + 16) % 18)) < 113)) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 1024) / 324) * 12544)) + ((((int)blockIdx.x) / 7) * 1792)) + (((((((int)threadIdx.x) >> 1) + 26) % 162) / 9) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + ((((int)threadIdx.x) + 16) % 18)) - 113)] : 0.000000e+00f);
    if (((int)threadIdx.x) < 16) {
      pad_temp_shared[(((int)threadIdx.x) + 1280)] = ((((((((int)blockIdx.x) / 7) * 16) + ((((((int)threadIdx.x) >> 1) + 154) % 162) / 9)) < 113) && ((((((int)blockIdx.x) % 7) * 16) + (((int)threadIdx.x) + 2)) < 113)) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 1280) / 324) * 12544)) + ((((int)blockIdx.x) / 7) * 1792)) + (((((((int)threadIdx.x) >> 1) + 154) % 162) / 9) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) + 2)) - 113)] : 0.000000e+00f);
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 18; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
      kernel_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 256) + ((int)threadIdx.x))] = kernel[((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 64) + (((int)threadIdx.x) >> 2)) / 9) * 1152) + (rc_outer_outer * 36)) + (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 256) + ((int)threadIdx.x)) % 36))];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 2; ++rc_outer_inner) {
      for (int ry_outer_inner = 0; ry_outer_inner < 3; ++ry_outer_inner) {
        for (int yy_c_outer_inner = 0; yy_c_outer_inner < 4; ++yy_c_outer_inner) {
          for (int rc_inner = 0; rc_inner < 2; ++rc_inner) {
            for (int rx_inner = 0; rx_inner < 3; ++rx_inner) {
              for (int ff_c_inner = 0; ff_c_inner < 2; ++ff_c_inner) {
                conv2d_nchw_local[((ff_c_inner * 32) + (yy_c_outer_inner * 8))] = (conv2d_nchw_local[((ff_c_inner * 32) + (yy_c_outer_inner * 8))] + (pad_temp_shared[(((((((rc_outer_inner * 648) + (rc_inner * 324)) + (((((int)threadIdx.x) & 3) >> 1) * 144)) + (yy_c_outer_inner * 36)) + (ry_outer_inner * 18)) + ((((int)threadIdx.x) & 1) * 4)) + rx_inner)] * kernel_shared[(((((((((int)threadIdx.x) >> 2) * 72) + (ff_c_inner * 36)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner)]));
                conv2d_nchw_local[(((ff_c_inner * 32) + (yy_c_outer_inner * 8)) + 64)] = (conv2d_nchw_local[(((ff_c_inner * 32) + (yy_c_outer_inner * 8)) + 64)] + (pad_temp_shared[((((((((rc_outer_inner * 648) + (rc_inner * 324)) + (((((int)threadIdx.x) & 3) >> 1) * 144)) + (yy_c_outer_inner * 36)) + (ry_outer_inner * 18)) + ((((int)threadIdx.x) & 1) * 4)) + rx_inner) + 8)] * kernel_shared[(((((((((int)threadIdx.x) >> 2) * 72) + (ff_c_inner * 36)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner)]));
                conv2d_nchw_local[(((ff_c_inner * 32) + (yy_c_outer_inner * 8)) + 1)] = (conv2d_nchw_local[(((ff_c_inner * 32) + (yy_c_outer_inner * 8)) + 1)] + (pad_temp_shared[((((((((rc_outer_inner * 648) + (rc_inner * 324)) + (((((int)threadIdx.x) & 3) >> 1) * 144)) + (yy_c_outer_inner * 36)) + (ry_outer_inner * 18)) + ((((int)threadIdx.x) & 1) * 4)) + rx_inner) + 1)] * kernel_shared[(((((((((int)threadIdx.x) >> 2) * 72) + (ff_c_inner * 36)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner)]));
                conv2d_nchw_local[(((ff_c_inner * 32) + (yy_c_outer_inner * 8)) + 65)] = (conv2d_nchw_local[(((ff_c_inner * 32) + (yy_c_outer_inner * 8)) + 65)] + (pad_temp_shared[((((((((rc_outer_inner * 648) + (rc_inner * 324)) + (((((int)threadIdx.x) & 3) >> 1) * 144)) + (yy_c_outer_inner * 36)) + (ry_outer_inner * 18)) + ((((int)threadIdx.x) & 1) * 4)) + rx_inner) + 9)] * kernel_shared[(((((((((int)threadIdx.x) >> 2) * 72) + (ff_c_inner * 36)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner)]));
                conv2d_nchw_local[(((ff_c_inner * 32) + (yy_c_outer_inner * 8)) + 2)] = (conv2d_nchw_local[(((ff_c_inner * 32) + (yy_c_outer_inner * 8)) + 2)] + (pad_temp_shared[((((((((rc_outer_inner * 648) + (rc_inner * 324)) + (((((int)threadIdx.x) & 3) >> 1) * 144)) + (yy_c_outer_inner * 36)) + (ry_outer_inner * 18)) + ((((int)threadIdx.x) & 1) * 4)) + rx_inner) + 2)] * kernel_shared[(((((((((int)threadIdx.x) >> 2) * 72) + (ff_c_inner * 36)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner)]));
                conv2d_nchw_local[(((ff_c_inner * 32) + (yy_c_outer_inner * 8)) + 66)] = (conv2d_nchw_local[(((ff_c_inner * 32) + (yy_c_outer_inner * 8)) + 66)] + (pad_temp_shared[((((((((rc_outer_inner * 648) + (rc_inner * 324)) + (((((int)threadIdx.x) & 3) >> 1) * 144)) + (yy_c_outer_inner * 36)) + (ry_outer_inner * 18)) + ((((int)threadIdx.x) & 1) * 4)) + rx_inner) + 10)] * kernel_shared[(((((((((int)threadIdx.x) >> 2) * 72) + (ff_c_inner * 36)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner)]));
                conv2d_nchw_local[(((ff_c_inner * 32) + (yy_c_outer_inner * 8)) + 3)] = (conv2d_nchw_local[(((ff_c_inner * 32) + (yy_c_outer_inner * 8)) + 3)] + (pad_temp_shared[((((((((rc_outer_inner * 648) + (rc_inner * 324)) + (((((int)threadIdx.x) & 3) >> 1) * 144)) + (yy_c_outer_inner * 36)) + (ry_outer_inner * 18)) + ((((int)threadIdx.x) & 1) * 4)) + rx_inner) + 3)] * kernel_shared[(((((((((int)threadIdx.x) >> 2) * 72) + (ff_c_inner * 36)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner)]));
                conv2d_nchw_local[(((ff_c_inner * 32) + (yy_c_outer_inner * 8)) + 67)] = (conv2d_nchw_local[(((ff_c_inner * 32) + (yy_c_outer_inner * 8)) + 67)] + (pad_temp_shared[((((((((rc_outer_inner * 648) + (rc_inner * 324)) + (((((int)threadIdx.x) & 3) >> 1) * 144)) + (yy_c_outer_inner * 36)) + (ry_outer_inner * 18)) + ((((int)threadIdx.x) & 1) * 4)) + rx_inner) + 11)] * kernel_shared[(((((((((int)threadIdx.x) >> 2) * 72) + (ff_c_inner * 36)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner)]));
                conv2d_nchw_local[(((ff_c_inner * 32) + (yy_c_outer_inner * 8)) + 4)] = (conv2d_nchw_local[(((ff_c_inner * 32) + (yy_c_outer_inner * 8)) + 4)] + (pad_temp_shared[((((((((rc_outer_inner * 648) + (rc_inner * 324)) + (((((int)threadIdx.x) & 3) >> 1) * 144)) + (yy_c_outer_inner * 36)) + (ry_outer_inner * 18)) + ((((int)threadIdx.x) & 1) * 4)) + rx_inner) + 18)] * kernel_shared[(((((((((int)threadIdx.x) >> 2) * 72) + (ff_c_inner * 36)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner)]));
                conv2d_nchw_local[(((ff_c_inner * 32) + (yy_c_outer_inner * 8)) + 68)] = (conv2d_nchw_local[(((ff_c_inner * 32) + (yy_c_outer_inner * 8)) + 68)] + (pad_temp_shared[((((((((rc_outer_inner * 648) + (rc_inner * 324)) + (((((int)threadIdx.x) & 3) >> 1) * 144)) + (yy_c_outer_inner * 36)) + (ry_outer_inner * 18)) + ((((int)threadIdx.x) & 1) * 4)) + rx_inner) + 26)] * kernel_shared[(((((((((int)threadIdx.x) >> 2) * 72) + (ff_c_inner * 36)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner)]));
                conv2d_nchw_local[(((ff_c_inner * 32) + (yy_c_outer_inner * 8)) + 5)] = (conv2d_nchw_local[(((ff_c_inner * 32) + (yy_c_outer_inner * 8)) + 5)] + (pad_temp_shared[((((((((rc_outer_inner * 648) + (rc_inner * 324)) + (((((int)threadIdx.x) & 3) >> 1) * 144)) + (yy_c_outer_inner * 36)) + (ry_outer_inner * 18)) + ((((int)threadIdx.x) & 1) * 4)) + rx_inner) + 19)] * kernel_shared[(((((((((int)threadIdx.x) >> 2) * 72) + (ff_c_inner * 36)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner)]));
                conv2d_nchw_local[(((ff_c_inner * 32) + (yy_c_outer_inner * 8)) + 69)] = (conv2d_nchw_local[(((ff_c_inner * 32) + (yy_c_outer_inner * 8)) + 69)] + (pad_temp_shared[((((((((rc_outer_inner * 648) + (rc_inner * 324)) + (((((int)threadIdx.x) & 3) >> 1) * 144)) + (yy_c_outer_inner * 36)) + (ry_outer_inner * 18)) + ((((int)threadIdx.x) & 1) * 4)) + rx_inner) + 27)] * kernel_shared[(((((((((int)threadIdx.x) >> 2) * 72) + (ff_c_inner * 36)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner)]));
                conv2d_nchw_local[(((ff_c_inner * 32) + (yy_c_outer_inner * 8)) + 6)] = (conv2d_nchw_local[(((ff_c_inner * 32) + (yy_c_outer_inner * 8)) + 6)] + (pad_temp_shared[((((((((rc_outer_inner * 648) + (rc_inner * 324)) + (((((int)threadIdx.x) & 3) >> 1) * 144)) + (yy_c_outer_inner * 36)) + (ry_outer_inner * 18)) + ((((int)threadIdx.x) & 1) * 4)) + rx_inner) + 20)] * kernel_shared[(((((((((int)threadIdx.x) >> 2) * 72) + (ff_c_inner * 36)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner)]));
                conv2d_nchw_local[(((ff_c_inner * 32) + (yy_c_outer_inner * 8)) + 70)] = (conv2d_nchw_local[(((ff_c_inner * 32) + (yy_c_outer_inner * 8)) + 70)] + (pad_temp_shared[((((((((rc_outer_inner * 648) + (rc_inner * 324)) + (((((int)threadIdx.x) & 3) >> 1) * 144)) + (yy_c_outer_inner * 36)) + (ry_outer_inner * 18)) + ((((int)threadIdx.x) & 1) * 4)) + rx_inner) + 28)] * kernel_shared[(((((((((int)threadIdx.x) >> 2) * 72) + (ff_c_inner * 36)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner)]));
                conv2d_nchw_local[(((ff_c_inner * 32) + (yy_c_outer_inner * 8)) + 7)] = (conv2d_nchw_local[(((ff_c_inner * 32) + (yy_c_outer_inner * 8)) + 7)] + (pad_temp_shared[((((((((rc_outer_inner * 648) + (rc_inner * 324)) + (((((int)threadIdx.x) & 3) >> 1) * 144)) + (yy_c_outer_inner * 36)) + (ry_outer_inner * 18)) + ((((int)threadIdx.x) & 1) * 4)) + rx_inner) + 21)] * kernel_shared[(((((((((int)threadIdx.x) >> 2) * 72) + (ff_c_inner * 36)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner)]));
                conv2d_nchw_local[(((ff_c_inner * 32) + (yy_c_outer_inner * 8)) + 71)] = (conv2d_nchw_local[(((ff_c_inner * 32) + (yy_c_outer_inner * 8)) + 71)] + (pad_temp_shared[((((((((rc_outer_inner * 648) + (rc_inner * 324)) + (((((int)threadIdx.x) & 3) >> 1) * 144)) + (yy_c_outer_inner * 36)) + (ry_outer_inner * 18)) + ((((int)threadIdx.x) & 1) * 4)) + rx_inner) + 29)] * kernel_shared[(((((((((int)threadIdx.x) >> 2) * 72) + (ff_c_inner * 36)) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_outer_inner * 3)) + rx_inner)]));
              }
            }
          }
        }
      }
    }
  }
  for (int ff_inner = 0; ff_inner < 2; ++ff_inner) {
    for (int yy_inner = 0; yy_inner < 8; ++yy_inner) {
      for (int xx_inner = 0; xx_inner < 4; ++xx_inner) {
        conv2d_nchw[(((((((((((int)threadIdx.x) >> 2) * 25088) + (ff_inner * 12544)) + ((((int)blockIdx.x) / 7) * 1792)) + (((((int)threadIdx.x) & 3) >> 1) * 896)) + (yy_inner * 112)) + ((((int)blockIdx.x) % 7) * 16)) + ((((int)threadIdx.x) & 1) * 4)) + xx_inner)] = conv2d_nchw_local[(((ff_inner * 32) + (yy_inner * 4)) + xx_inner)];
        conv2d_nchw[((((((((((((int)threadIdx.x) >> 2) * 25088) + (ff_inner * 12544)) + ((((int)blockIdx.x) / 7) * 1792)) + (((((int)threadIdx.x) & 3) >> 1) * 896)) + (yy_inner * 112)) + ((((int)blockIdx.x) % 7) * 16)) + ((((int)threadIdx.x) & 1) * 4)) + xx_inner) + 8)] = conv2d_nchw_local[((((ff_inner * 32) + (yy_inner * 4)) + xx_inner) + 64)];
      }
    }
  }
}


