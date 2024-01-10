
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
  float conv2d_nchw_local[224];
  __shared__ float pad_temp_shared[3861];
  __shared__ float kernel_shared[3136];
  for (int yy_c_outer_inner_init = 0; yy_c_outer_inner_init < 7; ++yy_c_outer_inner_init) {
    for (int xx_c_outer_inner_init = 0; xx_c_outer_inner_init < 2; ++xx_c_outer_inner_init) {
      conv2d_nchw_local[((yy_c_outer_inner_init * 8) + (xx_c_outer_inner_init * 2))] = 0.000000e+00f;
      conv2d_nchw_local[(((yy_c_outer_inner_init * 8) + (xx_c_outer_inner_init * 2)) + 56)] = 0.000000e+00f;
      conv2d_nchw_local[(((yy_c_outer_inner_init * 8) + (xx_c_outer_inner_init * 2)) + 112)] = 0.000000e+00f;
      conv2d_nchw_local[(((yy_c_outer_inner_init * 8) + (xx_c_outer_inner_init * 2)) + 168)] = 0.000000e+00f;
      conv2d_nchw_local[(((yy_c_outer_inner_init * 8) + (xx_c_outer_inner_init * 2)) + 1)] = 0.000000e+00f;
      conv2d_nchw_local[(((yy_c_outer_inner_init * 8) + (xx_c_outer_inner_init * 2)) + 57)] = 0.000000e+00f;
      conv2d_nchw_local[(((yy_c_outer_inner_init * 8) + (xx_c_outer_inner_init * 2)) + 113)] = 0.000000e+00f;
      conv2d_nchw_local[(((yy_c_outer_inner_init * 8) + (xx_c_outer_inner_init * 2)) + 169)] = 0.000000e+00f;
      conv2d_nchw_local[(((yy_c_outer_inner_init * 8) + (xx_c_outer_inner_init * 2)) + 4)] = 0.000000e+00f;
      conv2d_nchw_local[(((yy_c_outer_inner_init * 8) + (xx_c_outer_inner_init * 2)) + 60)] = 0.000000e+00f;
      conv2d_nchw_local[(((yy_c_outer_inner_init * 8) + (xx_c_outer_inner_init * 2)) + 116)] = 0.000000e+00f;
      conv2d_nchw_local[(((yy_c_outer_inner_init * 8) + (xx_c_outer_inner_init * 2)) + 172)] = 0.000000e+00f;
      conv2d_nchw_local[(((yy_c_outer_inner_init * 8) + (xx_c_outer_inner_init * 2)) + 5)] = 0.000000e+00f;
      conv2d_nchw_local[(((yy_c_outer_inner_init * 8) + (xx_c_outer_inner_init * 2)) + 61)] = 0.000000e+00f;
      conv2d_nchw_local[(((yy_c_outer_inner_init * 8) + (xx_c_outer_inner_init * 2)) + 117)] = 0.000000e+00f;
      conv2d_nchw_local[(((yy_c_outer_inner_init * 8) + (xx_c_outer_inner_init * 2)) + 173)] = 0.000000e+00f;
    }
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 3; ++rc_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 18; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
      if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 224) + ((int)threadIdx.x)) < 3861) {
        pad_temp_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 224) + ((int)threadIdx.x))] = (((((3 <= (((((int)blockIdx.x) >> 1) * 28) + (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 224) + ((int)threadIdx.x)) / 117))) && ((((((int)blockIdx.x) >> 1) * 28) + (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 224) + ((int)threadIdx.x)) / 117)) < 227)) && (3 <= (((((int)blockIdx.x) & 1) * 112) + (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 224) + ((int)threadIdx.x)) % 117)))) && ((((((int)blockIdx.x) & 1) * 112) + (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 224) + ((int)threadIdx.x)) % 117)) < 227)) ? data[((((((rc_outer_outer * 50176) + ((((int)blockIdx.x) >> 1) * 6272)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 224) + ((int)threadIdx.x)) / 117) * 224)) + ((((int)blockIdx.x) & 1) * 112)) + (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 224) + ((int)threadIdx.x)) % 117)) - 675)] : 0.000000e+00f);
      }
    }
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)threadIdx.x) / 49) * 147) + (rc_outer_outer * 49)) + (((int)threadIdx.x) % 49))];
    kernel_shared[(((int)threadIdx.x) + 224)] = kernel[((((((((int)threadIdx.x) + 224) / 49) * 147) + (rc_outer_outer * 49)) + ((((((int)threadIdx.x) / 7) + 4) % 7) * 7)) + (((int)threadIdx.x) % 7))];
    kernel_shared[(((int)threadIdx.x) + 448)] = kernel[((((((((int)threadIdx.x) + 448) / 49) * 147) + (rc_outer_outer * 49)) + ((((((int)threadIdx.x) / 7) + 1) % 7) * 7)) + (((int)threadIdx.x) % 7))];
    kernel_shared[(((int)threadIdx.x) + 672)] = kernel[((((((((int)threadIdx.x) + 672) / 49) * 147) + (rc_outer_outer * 49)) + ((((((int)threadIdx.x) / 7) + 5) % 7) * 7)) + (((int)threadIdx.x) % 7))];
    kernel_shared[(((int)threadIdx.x) + 896)] = kernel[((((((((int)threadIdx.x) + 896) / 49) * 147) + (rc_outer_outer * 49)) + ((((((int)threadIdx.x) / 7) + 2) % 7) * 7)) + (((int)threadIdx.x) % 7))];
    kernel_shared[(((int)threadIdx.x) + 1120)] = kernel[((((((((int)threadIdx.x) + 1120) / 49) * 147) + (rc_outer_outer * 49)) + ((((((int)threadIdx.x) / 7) + 6) % 7) * 7)) + (((int)threadIdx.x) % 7))];
    kernel_shared[(((int)threadIdx.x) + 1344)] = kernel[((((((((int)threadIdx.x) + 1344) / 49) * 147) + (rc_outer_outer * 49)) + ((((((int)threadIdx.x) / 7) + 3) % 7) * 7)) + (((int)threadIdx.x) % 7))];
    kernel_shared[(((int)threadIdx.x) + 1568)] = kernel[(((((((int)threadIdx.x) / 49) * 147) + (rc_outer_outer * 49)) + (((int)threadIdx.x) % 49)) + 4704)];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[((((((((int)threadIdx.x) + 1792) / 49) * 147) + (rc_outer_outer * 49)) + ((((((int)threadIdx.x) / 7) + 4) % 7) * 7)) + (((int)threadIdx.x) % 7))];
    kernel_shared[(((int)threadIdx.x) + 2016)] = kernel[((((((((int)threadIdx.x) + 2016) / 49) * 147) + (rc_outer_outer * 49)) + ((((((int)threadIdx.x) / 7) + 1) % 7) * 7)) + (((int)threadIdx.x) % 7))];
    kernel_shared[(((int)threadIdx.x) + 2240)] = kernel[((((((((int)threadIdx.x) + 2240) / 49) * 147) + (rc_outer_outer * 49)) + ((((((int)threadIdx.x) / 7) + 5) % 7) * 7)) + (((int)threadIdx.x) % 7))];
    kernel_shared[(((int)threadIdx.x) + 2464)] = kernel[((((((((int)threadIdx.x) + 2464) / 49) * 147) + (rc_outer_outer * 49)) + ((((((int)threadIdx.x) / 7) + 2) % 7) * 7)) + (((int)threadIdx.x) % 7))];
    kernel_shared[(((int)threadIdx.x) + 2688)] = kernel[((((((((int)threadIdx.x) + 2688) / 49) * 147) + (rc_outer_outer * 49)) + ((((((int)threadIdx.x) / 7) + 6) % 7) * 7)) + (((int)threadIdx.x) % 7))];
    kernel_shared[(((int)threadIdx.x) + 2912)] = kernel[((((((((int)threadIdx.x) + 2912) / 49) * 147) + (rc_outer_outer * 49)) + ((((((int)threadIdx.x) / 7) + 3) % 7) * 7)) + (((int)threadIdx.x) % 7))];
    __syncthreads();
    for (int rx_outer_inner = 0; rx_outer_inner < 7; ++rx_outer_inner) {
      for (int yy_c_outer_inner = 0; yy_c_outer_inner < 7; ++yy_c_outer_inner) {
        for (int xx_c_outer_inner = 0; xx_c_outer_inner < 2; ++xx_c_outer_inner) {
          for (int ry_inner = 0; ry_inner < 7; ++ry_inner) {
            conv2d_nchw_local[((yy_c_outer_inner * 8) + (xx_c_outer_inner * 2))] = (conv2d_nchw_local[((yy_c_outer_inner * 8) + (xx_c_outer_inner * 2))] + (pad_temp_shared[(((((yy_c_outer_inner * 468) + (ry_inner * 117)) + ((((int)threadIdx.x) % 14) * 8)) + (xx_c_outer_inner * 4)) + rx_outer_inner)] * kernel_shared[((((((int)threadIdx.x) / 14) * 49) + (ry_inner * 7)) + rx_outer_inner)]));
            conv2d_nchw_local[(((yy_c_outer_inner * 8) + (xx_c_outer_inner * 2)) + 56)] = (conv2d_nchw_local[(((yy_c_outer_inner * 8) + (xx_c_outer_inner * 2)) + 56)] + (pad_temp_shared[(((((yy_c_outer_inner * 468) + (ry_inner * 117)) + ((((int)threadIdx.x) % 14) * 8)) + (xx_c_outer_inner * 4)) + rx_outer_inner)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 49) + (ry_inner * 7)) + rx_outer_inner) + 784)]));
            conv2d_nchw_local[(((yy_c_outer_inner * 8) + (xx_c_outer_inner * 2)) + 112)] = (conv2d_nchw_local[(((yy_c_outer_inner * 8) + (xx_c_outer_inner * 2)) + 112)] + (pad_temp_shared[(((((yy_c_outer_inner * 468) + (ry_inner * 117)) + ((((int)threadIdx.x) % 14) * 8)) + (xx_c_outer_inner * 4)) + rx_outer_inner)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 49) + (ry_inner * 7)) + rx_outer_inner) + 1568)]));
            conv2d_nchw_local[(((yy_c_outer_inner * 8) + (xx_c_outer_inner * 2)) + 168)] = (conv2d_nchw_local[(((yy_c_outer_inner * 8) + (xx_c_outer_inner * 2)) + 168)] + (pad_temp_shared[(((((yy_c_outer_inner * 468) + (ry_inner * 117)) + ((((int)threadIdx.x) % 14) * 8)) + (xx_c_outer_inner * 4)) + rx_outer_inner)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 49) + (ry_inner * 7)) + rx_outer_inner) + 2352)]));
            conv2d_nchw_local[(((yy_c_outer_inner * 8) + (xx_c_outer_inner * 2)) + 1)] = (conv2d_nchw_local[(((yy_c_outer_inner * 8) + (xx_c_outer_inner * 2)) + 1)] + (pad_temp_shared[((((((yy_c_outer_inner * 468) + (ry_inner * 117)) + ((((int)threadIdx.x) % 14) * 8)) + (xx_c_outer_inner * 4)) + rx_outer_inner) + 2)] * kernel_shared[((((((int)threadIdx.x) / 14) * 49) + (ry_inner * 7)) + rx_outer_inner)]));
            conv2d_nchw_local[(((yy_c_outer_inner * 8) + (xx_c_outer_inner * 2)) + 57)] = (conv2d_nchw_local[(((yy_c_outer_inner * 8) + (xx_c_outer_inner * 2)) + 57)] + (pad_temp_shared[((((((yy_c_outer_inner * 468) + (ry_inner * 117)) + ((((int)threadIdx.x) % 14) * 8)) + (xx_c_outer_inner * 4)) + rx_outer_inner) + 2)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 49) + (ry_inner * 7)) + rx_outer_inner) + 784)]));
            conv2d_nchw_local[(((yy_c_outer_inner * 8) + (xx_c_outer_inner * 2)) + 113)] = (conv2d_nchw_local[(((yy_c_outer_inner * 8) + (xx_c_outer_inner * 2)) + 113)] + (pad_temp_shared[((((((yy_c_outer_inner * 468) + (ry_inner * 117)) + ((((int)threadIdx.x) % 14) * 8)) + (xx_c_outer_inner * 4)) + rx_outer_inner) + 2)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 49) + (ry_inner * 7)) + rx_outer_inner) + 1568)]));
            conv2d_nchw_local[(((yy_c_outer_inner * 8) + (xx_c_outer_inner * 2)) + 169)] = (conv2d_nchw_local[(((yy_c_outer_inner * 8) + (xx_c_outer_inner * 2)) + 169)] + (pad_temp_shared[((((((yy_c_outer_inner * 468) + (ry_inner * 117)) + ((((int)threadIdx.x) % 14) * 8)) + (xx_c_outer_inner * 4)) + rx_outer_inner) + 2)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 49) + (ry_inner * 7)) + rx_outer_inner) + 2352)]));
            conv2d_nchw_local[(((yy_c_outer_inner * 8) + (xx_c_outer_inner * 2)) + 4)] = (conv2d_nchw_local[(((yy_c_outer_inner * 8) + (xx_c_outer_inner * 2)) + 4)] + (pad_temp_shared[((((((yy_c_outer_inner * 468) + (ry_inner * 117)) + ((((int)threadIdx.x) % 14) * 8)) + (xx_c_outer_inner * 4)) + rx_outer_inner) + 234)] * kernel_shared[((((((int)threadIdx.x) / 14) * 49) + (ry_inner * 7)) + rx_outer_inner)]));
            conv2d_nchw_local[(((yy_c_outer_inner * 8) + (xx_c_outer_inner * 2)) + 60)] = (conv2d_nchw_local[(((yy_c_outer_inner * 8) + (xx_c_outer_inner * 2)) + 60)] + (pad_temp_shared[((((((yy_c_outer_inner * 468) + (ry_inner * 117)) + ((((int)threadIdx.x) % 14) * 8)) + (xx_c_outer_inner * 4)) + rx_outer_inner) + 234)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 49) + (ry_inner * 7)) + rx_outer_inner) + 784)]));
            conv2d_nchw_local[(((yy_c_outer_inner * 8) + (xx_c_outer_inner * 2)) + 116)] = (conv2d_nchw_local[(((yy_c_outer_inner * 8) + (xx_c_outer_inner * 2)) + 116)] + (pad_temp_shared[((((((yy_c_outer_inner * 468) + (ry_inner * 117)) + ((((int)threadIdx.x) % 14) * 8)) + (xx_c_outer_inner * 4)) + rx_outer_inner) + 234)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 49) + (ry_inner * 7)) + rx_outer_inner) + 1568)]));
            conv2d_nchw_local[(((yy_c_outer_inner * 8) + (xx_c_outer_inner * 2)) + 172)] = (conv2d_nchw_local[(((yy_c_outer_inner * 8) + (xx_c_outer_inner * 2)) + 172)] + (pad_temp_shared[((((((yy_c_outer_inner * 468) + (ry_inner * 117)) + ((((int)threadIdx.x) % 14) * 8)) + (xx_c_outer_inner * 4)) + rx_outer_inner) + 234)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 49) + (ry_inner * 7)) + rx_outer_inner) + 2352)]));
            conv2d_nchw_local[(((yy_c_outer_inner * 8) + (xx_c_outer_inner * 2)) + 5)] = (conv2d_nchw_local[(((yy_c_outer_inner * 8) + (xx_c_outer_inner * 2)) + 5)] + (pad_temp_shared[((((((yy_c_outer_inner * 468) + (ry_inner * 117)) + ((((int)threadIdx.x) % 14) * 8)) + (xx_c_outer_inner * 4)) + rx_outer_inner) + 236)] * kernel_shared[((((((int)threadIdx.x) / 14) * 49) + (ry_inner * 7)) + rx_outer_inner)]));
            conv2d_nchw_local[(((yy_c_outer_inner * 8) + (xx_c_outer_inner * 2)) + 61)] = (conv2d_nchw_local[(((yy_c_outer_inner * 8) + (xx_c_outer_inner * 2)) + 61)] + (pad_temp_shared[((((((yy_c_outer_inner * 468) + (ry_inner * 117)) + ((((int)threadIdx.x) % 14) * 8)) + (xx_c_outer_inner * 4)) + rx_outer_inner) + 236)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 49) + (ry_inner * 7)) + rx_outer_inner) + 784)]));
            conv2d_nchw_local[(((yy_c_outer_inner * 8) + (xx_c_outer_inner * 2)) + 117)] = (conv2d_nchw_local[(((yy_c_outer_inner * 8) + (xx_c_outer_inner * 2)) + 117)] + (pad_temp_shared[((((((yy_c_outer_inner * 468) + (ry_inner * 117)) + ((((int)threadIdx.x) % 14) * 8)) + (xx_c_outer_inner * 4)) + rx_outer_inner) + 236)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 49) + (ry_inner * 7)) + rx_outer_inner) + 1568)]));
            conv2d_nchw_local[(((yy_c_outer_inner * 8) + (xx_c_outer_inner * 2)) + 173)] = (conv2d_nchw_local[(((yy_c_outer_inner * 8) + (xx_c_outer_inner * 2)) + 173)] + (pad_temp_shared[((((((yy_c_outer_inner * 468) + (ry_inner * 117)) + ((((int)threadIdx.x) % 14) * 8)) + (xx_c_outer_inner * 4)) + rx_outer_inner) + 236)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 49) + (ry_inner * 7)) + rx_outer_inner) + 2352)]));
          }
        }
      }
    }
  }
  for (int yy_inner = 0; yy_inner < 14; ++yy_inner) {
    for (int xx_inner = 0; xx_inner < 4; ++xx_inner) {
      conv2d_nchw[(((((((((int)threadIdx.x) / 14) * 12544) + ((((int)blockIdx.x) >> 1) * 1568)) + (yy_inner * 112)) + ((((int)blockIdx.x) & 1) * 56)) + ((((int)threadIdx.x) % 14) * 4)) + xx_inner)] = conv2d_nchw_local[((yy_inner * 4) + xx_inner)];
      conv2d_nchw[((((((((((int)threadIdx.x) / 14) * 12544) + ((((int)blockIdx.x) >> 1) * 1568)) + (yy_inner * 112)) + ((((int)blockIdx.x) & 1) * 56)) + ((((int)threadIdx.x) % 14) * 4)) + xx_inner) + 200704)] = conv2d_nchw_local[(((yy_inner * 4) + xx_inner) + 56)];
      conv2d_nchw[((((((((((int)threadIdx.x) / 14) * 12544) + ((((int)blockIdx.x) >> 1) * 1568)) + (yy_inner * 112)) + ((((int)blockIdx.x) & 1) * 56)) + ((((int)threadIdx.x) % 14) * 4)) + xx_inner) + 401408)] = conv2d_nchw_local[(((yy_inner * 4) + xx_inner) + 112)];
      conv2d_nchw[((((((((((int)threadIdx.x) / 14) * 12544) + ((((int)blockIdx.x) >> 1) * 1568)) + (yy_inner * 112)) + ((((int)blockIdx.x) & 1) * 56)) + ((((int)threadIdx.x) % 14) * 4)) + xx_inner) + 602112)] = conv2d_nchw_local[(((yy_inner * 4) + xx_inner) + 168)];
    }
  }
}


