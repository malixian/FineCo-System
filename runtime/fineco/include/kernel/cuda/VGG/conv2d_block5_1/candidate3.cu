
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
extern "C" __global__ void __launch_bounds__(112) candidate3(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[28];
  __shared__ float pad_temp_shared[7168];
  __shared__ float kernel_shared[1536];
  for (int yy_c_outer_inner_init = 0; yy_c_outer_inner_init < 2; ++yy_c_outer_inner_init) {
    conv2d_nchw_local[(yy_c_outer_inner_init * 7)] = 0.000000e+00f;
    conv2d_nchw_local[((yy_c_outer_inner_init * 7) + 14)] = 0.000000e+00f;
    conv2d_nchw_local[((yy_c_outer_inner_init * 7) + 1)] = 0.000000e+00f;
    conv2d_nchw_local[((yy_c_outer_inner_init * 7) + 15)] = 0.000000e+00f;
    conv2d_nchw_local[((yy_c_outer_inner_init * 7) + 2)] = 0.000000e+00f;
    conv2d_nchw_local[((yy_c_outer_inner_init * 7) + 16)] = 0.000000e+00f;
    conv2d_nchw_local[((yy_c_outer_inner_init * 7) + 3)] = 0.000000e+00f;
    conv2d_nchw_local[((yy_c_outer_inner_init * 7) + 17)] = 0.000000e+00f;
    conv2d_nchw_local[((yy_c_outer_inner_init * 7) + 4)] = 0.000000e+00f;
    conv2d_nchw_local[((yy_c_outer_inner_init * 7) + 18)] = 0.000000e+00f;
    conv2d_nchw_local[((yy_c_outer_inner_init * 7) + 5)] = 0.000000e+00f;
    conv2d_nchw_local[((yy_c_outer_inner_init * 7) + 19)] = 0.000000e+00f;
    conv2d_nchw_local[((yy_c_outer_inner_init * 7) + 6)] = 0.000000e+00f;
    conv2d_nchw_local[((yy_c_outer_inner_init * 7) + 20)] = 0.000000e+00f;
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 16; ++rc_outer_outer) {
    for (int rx_outer_outer = 0; rx_outer_outer < 3; ++rx_outer_outer) {
      __syncthreads();
      for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 64; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
        pad_temp_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 112) + ((int)threadIdx.x))] = (((((1 <= (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer & 1) * 8) + (((int)threadIdx.x) / 14))) && ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer & 1) * 8) + (((int)threadIdx.x) / 14)) < 15)) && (1 <= (rx_outer_outer + (((int)threadIdx.x) % 14)))) && ((rx_outer_outer + (((int)threadIdx.x) % 14)) < 15)) ? data[((((((rc_outer_outer * 6272) + ((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer >> 1) * 196)) + ((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer & 1) * 112)) + ((int)threadIdx.x)) + rx_outer_outer) - 15)] : 0.000000e+00f);
      }
      kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) * 73728) + ((((int)threadIdx.x) / 96) * 4608)) + (rc_outer_outer * 288)) + ((((int)threadIdx.x) % 96) * 3)) + rx_outer_outer)];
      kernel_shared[(((int)threadIdx.x) + 112)] = kernel[(((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 112) / 96) * 4608)) + (rc_outer_outer * 288)) + (((((int)threadIdx.x) + 16) % 96) * 3)) + rx_outer_outer)];
      kernel_shared[(((int)threadIdx.x) + 224)] = kernel[(((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 224) / 96) * 4608)) + (rc_outer_outer * 288)) + (((((int)threadIdx.x) + 32) % 96) * 3)) + rx_outer_outer)];
      kernel_shared[(((int)threadIdx.x) + 336)] = kernel[(((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 336) / 96) * 4608)) + (rc_outer_outer * 288)) + (((((int)threadIdx.x) + 48) % 96) * 3)) + rx_outer_outer)];
      kernel_shared[(((int)threadIdx.x) + 448)] = kernel[(((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 448) / 96) * 4608)) + (rc_outer_outer * 288)) + (((((int)threadIdx.x) + 64) % 96) * 3)) + rx_outer_outer)];
      kernel_shared[(((int)threadIdx.x) + 560)] = kernel[(((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 560) / 96) * 4608)) + (rc_outer_outer * 288)) + (((((int)threadIdx.x) + 80) % 96) * 3)) + rx_outer_outer)];
      kernel_shared[(((int)threadIdx.x) + 672)] = kernel[((((((((int)blockIdx.x) * 73728) + ((((int)threadIdx.x) / 96) * 4608)) + (rc_outer_outer * 288)) + ((((int)threadIdx.x) % 96) * 3)) + rx_outer_outer) + 32256)];
      kernel_shared[(((int)threadIdx.x) + 784)] = kernel[(((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 784) / 96) * 4608)) + (rc_outer_outer * 288)) + (((((int)threadIdx.x) + 16) % 96) * 3)) + rx_outer_outer)];
      kernel_shared[(((int)threadIdx.x) + 896)] = kernel[(((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 896) / 96) * 4608)) + (rc_outer_outer * 288)) + (((((int)threadIdx.x) + 32) % 96) * 3)) + rx_outer_outer)];
      kernel_shared[(((int)threadIdx.x) + 1008)] = kernel[(((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 1008) / 96) * 4608)) + (rc_outer_outer * 288)) + (((((int)threadIdx.x) + 48) % 96) * 3)) + rx_outer_outer)];
      kernel_shared[(((int)threadIdx.x) + 1120)] = kernel[(((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 1120) / 96) * 4608)) + (rc_outer_outer * 288)) + (((((int)threadIdx.x) + 64) % 96) * 3)) + rx_outer_outer)];
      kernel_shared[(((int)threadIdx.x) + 1232)] = kernel[(((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 1232) / 96) * 4608)) + (rc_outer_outer * 288)) + (((((int)threadIdx.x) + 80) % 96) * 3)) + rx_outer_outer)];
      kernel_shared[(((int)threadIdx.x) + 1344)] = kernel[((((((((int)blockIdx.x) * 73728) + ((((int)threadIdx.x) / 96) * 4608)) + (rc_outer_outer * 288)) + ((((int)threadIdx.x) % 96) * 3)) + rx_outer_outer) + 64512)];
      if (((int)threadIdx.x) < 80) {
        kernel_shared[(((int)threadIdx.x) + 1456)] = kernel[(((((((int)blockIdx.x) * 73728) + (((((int)threadIdx.x) + 1456) / 96) * 4608)) + (rc_outer_outer * 288)) + ((((int)threadIdx.x) + 16) * 3)) + rx_outer_outer)];
      }
      __syncthreads();
      for (int rc_outer_inner = 0; rc_outer_inner < 4; ++rc_outer_inner) {
        for (int yy_c_outer_inner = 0; yy_c_outer_inner < 2; ++yy_c_outer_inner) {
          for (int xx_c_outer_inner = 0; xx_c_outer_inner < 7; ++xx_c_outer_inner) {
            for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
              conv2d_nchw_local[((yy_c_outer_inner * 7) + xx_c_outer_inner)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + xx_c_outer_inner)] + (pad_temp_shared[((((((rc_outer_inner * 1792) + (rc_inner * 224)) + (((((int)threadIdx.x) % 14) >> 1) * 28)) + (yy_c_outer_inner * 14)) + ((((int)threadIdx.x) & 1) * 7)) + xx_c_outer_inner)] * kernel_shared[((((((int)threadIdx.x) / 14) * 96) + (rc_outer_inner * 24)) + (rc_inner * 3))]));
              conv2d_nchw_local[(((yy_c_outer_inner * 7) + xx_c_outer_inner) + 14)] = (conv2d_nchw_local[(((yy_c_outer_inner * 7) + xx_c_outer_inner) + 14)] + (pad_temp_shared[((((((rc_outer_inner * 1792) + (rc_inner * 224)) + (((((int)threadIdx.x) % 14) >> 1) * 28)) + (yy_c_outer_inner * 14)) + ((((int)threadIdx.x) & 1) * 7)) + xx_c_outer_inner)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 96) + (rc_outer_inner * 24)) + (rc_inner * 3)) + 768)]));
              conv2d_nchw_local[((yy_c_outer_inner * 7) + xx_c_outer_inner)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + xx_c_outer_inner)] + (pad_temp_shared[(((((((rc_outer_inner * 1792) + (rc_inner * 224)) + (((((int)threadIdx.x) % 14) >> 1) * 28)) + (yy_c_outer_inner * 14)) + ((((int)threadIdx.x) & 1) * 7)) + xx_c_outer_inner) + 14)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 96) + (rc_outer_inner * 24)) + (rc_inner * 3)) + 1)]));
              conv2d_nchw_local[(((yy_c_outer_inner * 7) + xx_c_outer_inner) + 14)] = (conv2d_nchw_local[(((yy_c_outer_inner * 7) + xx_c_outer_inner) + 14)] + (pad_temp_shared[(((((((rc_outer_inner * 1792) + (rc_inner * 224)) + (((((int)threadIdx.x) % 14) >> 1) * 28)) + (yy_c_outer_inner * 14)) + ((((int)threadIdx.x) & 1) * 7)) + xx_c_outer_inner) + 14)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 96) + (rc_outer_inner * 24)) + (rc_inner * 3)) + 769)]));
              conv2d_nchw_local[((yy_c_outer_inner * 7) + xx_c_outer_inner)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + xx_c_outer_inner)] + (pad_temp_shared[(((((((rc_outer_inner * 1792) + (rc_inner * 224)) + (((((int)threadIdx.x) % 14) >> 1) * 28)) + (yy_c_outer_inner * 14)) + ((((int)threadIdx.x) & 1) * 7)) + xx_c_outer_inner) + 28)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 96) + (rc_outer_inner * 24)) + (rc_inner * 3)) + 2)]));
              conv2d_nchw_local[(((yy_c_outer_inner * 7) + xx_c_outer_inner) + 14)] = (conv2d_nchw_local[(((yy_c_outer_inner * 7) + xx_c_outer_inner) + 14)] + (pad_temp_shared[(((((((rc_outer_inner * 1792) + (rc_inner * 224)) + (((((int)threadIdx.x) % 14) >> 1) * 28)) + (yy_c_outer_inner * 14)) + ((((int)threadIdx.x) & 1) * 7)) + xx_c_outer_inner) + 28)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 96) + (rc_outer_inner * 24)) + (rc_inner * 3)) + 770)]));
            }
          }
        }
      }
    }
  }
  for (int yy_inner = 0; yy_inner < 2; ++yy_inner) {
    for (int xx_inner = 0; xx_inner < 7; ++xx_inner) {
      conv2d_nchw[(((((((int)blockIdx.x) * 3136) + ((((int)threadIdx.x) >> 1) * 28)) + (yy_inner * 14)) + ((((int)threadIdx.x) & 1) * 7)) + xx_inner)] = conv2d_nchw_local[((yy_inner * 7) + xx_inner)];
      conv2d_nchw[((((((((int)blockIdx.x) * 3136) + ((((int)threadIdx.x) >> 1) * 28)) + (yy_inner * 14)) + ((((int)threadIdx.x) & 1) * 7)) + xx_inner) + 1568)] = conv2d_nchw_local[(((yy_inner * 7) + xx_inner) + 14)];
    }
  }
}


