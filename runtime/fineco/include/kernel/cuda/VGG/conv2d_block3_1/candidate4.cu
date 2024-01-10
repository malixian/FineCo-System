
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
extern "C" __global__ void __launch_bounds__(512) candidate4(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[32];
  __shared__ float pad_temp_shared[200];
  __shared__ float kernel_shared[4608];
  for (int ff_c_outer_inner_init = 0; ff_c_outer_inner_init < 2; ++ff_c_outer_inner_init) {
    for (int yy_c_inner_init = 0; yy_c_inner_init < 2; ++yy_c_inner_init) {
      conv2d_nchw_local[((ff_c_outer_inner_init * 2) + yy_c_inner_init)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_outer_inner_init * 2) + yy_c_inner_init) + 4)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_outer_inner_init * 2) + yy_c_inner_init) + 8)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_outer_inner_init * 2) + yy_c_inner_init) + 12)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_outer_inner_init * 2) + yy_c_inner_init) + 16)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_outer_inner_init * 2) + yy_c_inner_init) + 20)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_outer_inner_init * 2) + yy_c_inner_init) + 24)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_outer_inner_init * 2) + yy_c_inner_init) + 28)] = 0.000000e+00f;
    }
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 64; ++rc_outer_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 200) {
      pad_temp_shared[((int)threadIdx.x)] = (((((1 <= (((((int)blockIdx.x) / 7) * 8) + ((((int)threadIdx.x) % 100) / 10))) && ((((((int)blockIdx.x) / 7) * 8) + ((((int)threadIdx.x) % 100) / 10)) < 57)) && (1 <= (((((int)blockIdx.x) % 7) * 8) + (((int)threadIdx.x) % 10)))) && ((((((int)blockIdx.x) % 7) * 8) + (((int)threadIdx.x) % 10)) < 57)) ? data[(((((((rc_outer_outer * 6272) + ((((int)threadIdx.x) / 100) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((int)threadIdx.x) % 100) / 10) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) % 10)) - 57)] : 0.000000e+00f);
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 9; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
      kernel_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 512) + ((int)threadIdx.x))] = kernel[((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 256) + (((int)threadIdx.x) >> 1)) / 9) * 1152) + (rc_outer_outer * 18)) + (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 512) + ((int)threadIdx.x)) % 18))];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 2; ++rc_outer_inner) {
      for (int ry_outer_inner = 0; ry_outer_inner < 3; ++ry_outer_inner) {
        for (int ff_c_outer_inner = 0; ff_c_outer_inner < 2; ++ff_c_outer_inner) {
          for (int rx_inner = 0; rx_inner < 3; ++rx_inner) {
            for (int yy_c_inner = 0; yy_c_inner < 2; ++yy_c_inner) {
              conv2d_nchw_local[((ff_c_outer_inner * 2) + yy_c_inner)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + yy_c_inner)] + (pad_temp_shared[(((((rc_outer_inner * 100) + (yy_c_inner * 10)) + (ry_outer_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7))] * kernel_shared[((((((((int)threadIdx.x) >> 3) * 36) + (ff_c_outer_inner * 18)) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner)]));
              conv2d_nchw_local[(((ff_c_outer_inner * 2) + yy_c_inner) + 4)] = (conv2d_nchw_local[(((ff_c_outer_inner * 2) + yy_c_inner) + 4)] + (pad_temp_shared[((((((rc_outer_inner * 100) + (yy_c_inner * 10)) + (ry_outer_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7)) + 20)] * kernel_shared[((((((((int)threadIdx.x) >> 3) * 36) + (ff_c_outer_inner * 18)) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner)]));
              conv2d_nchw_local[(((ff_c_outer_inner * 2) + yy_c_inner) + 8)] = (conv2d_nchw_local[(((ff_c_outer_inner * 2) + yy_c_inner) + 8)] + (pad_temp_shared[((((((rc_outer_inner * 100) + (yy_c_inner * 10)) + (ry_outer_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7)) + 40)] * kernel_shared[((((((((int)threadIdx.x) >> 3) * 36) + (ff_c_outer_inner * 18)) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner)]));
              conv2d_nchw_local[(((ff_c_outer_inner * 2) + yy_c_inner) + 12)] = (conv2d_nchw_local[(((ff_c_outer_inner * 2) + yy_c_inner) + 12)] + (pad_temp_shared[((((((rc_outer_inner * 100) + (yy_c_inner * 10)) + (ry_outer_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7)) + 60)] * kernel_shared[((((((((int)threadIdx.x) >> 3) * 36) + (ff_c_outer_inner * 18)) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner)]));
              conv2d_nchw_local[(((ff_c_outer_inner * 2) + yy_c_inner) + 16)] = (conv2d_nchw_local[(((ff_c_outer_inner * 2) + yy_c_inner) + 16)] + (pad_temp_shared[(((((rc_outer_inner * 100) + (yy_c_inner * 10)) + (ry_outer_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7))] * kernel_shared[(((((((((int)threadIdx.x) >> 3) * 36) + (ff_c_outer_inner * 18)) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 2304)]));
              conv2d_nchw_local[(((ff_c_outer_inner * 2) + yy_c_inner) + 20)] = (conv2d_nchw_local[(((ff_c_outer_inner * 2) + yy_c_inner) + 20)] + (pad_temp_shared[((((((rc_outer_inner * 100) + (yy_c_inner * 10)) + (ry_outer_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7)) + 20)] * kernel_shared[(((((((((int)threadIdx.x) >> 3) * 36) + (ff_c_outer_inner * 18)) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 2304)]));
              conv2d_nchw_local[(((ff_c_outer_inner * 2) + yy_c_inner) + 24)] = (conv2d_nchw_local[(((ff_c_outer_inner * 2) + yy_c_inner) + 24)] + (pad_temp_shared[((((((rc_outer_inner * 100) + (yy_c_inner * 10)) + (ry_outer_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7)) + 40)] * kernel_shared[(((((((((int)threadIdx.x) >> 3) * 36) + (ff_c_outer_inner * 18)) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 2304)]));
              conv2d_nchw_local[(((ff_c_outer_inner * 2) + yy_c_inner) + 28)] = (conv2d_nchw_local[(((ff_c_outer_inner * 2) + yy_c_inner) + 28)] + (pad_temp_shared[((((((rc_outer_inner * 100) + (yy_c_inner * 10)) + (ry_outer_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7)) + 60)] * kernel_shared[(((((((((int)threadIdx.x) >> 3) * 36) + (ff_c_outer_inner * 18)) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 2304)]));
            }
          }
        }
      }
    }
  }
  for (int ff_inner = 0; ff_inner < 2; ++ff_inner) {
    for (int yy_inner = 0; yy_inner < 2; ++yy_inner) {
      conv2d_nchw[(((((((((int)threadIdx.x) >> 3) * 6272) + (ff_inner * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (yy_inner * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7))] = conv2d_nchw_local[((ff_inner * 2) + yy_inner)];
      conv2d_nchw[((((((((((int)threadIdx.x) >> 3) * 6272) + (ff_inner * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (yy_inner * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 112)] = conv2d_nchw_local[(((ff_inner * 2) + yy_inner) + 4)];
      conv2d_nchw[((((((((((int)threadIdx.x) >> 3) * 6272) + (ff_inner * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (yy_inner * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 224)] = conv2d_nchw_local[(((ff_inner * 2) + yy_inner) + 8)];
      conv2d_nchw[((((((((((int)threadIdx.x) >> 3) * 6272) + (ff_inner * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (yy_inner * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 336)] = conv2d_nchw_local[(((ff_inner * 2) + yy_inner) + 12)];
      conv2d_nchw[((((((((((int)threadIdx.x) >> 3) * 6272) + (ff_inner * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (yy_inner * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 401408)] = conv2d_nchw_local[(((ff_inner * 2) + yy_inner) + 16)];
      conv2d_nchw[((((((((((int)threadIdx.x) >> 3) * 6272) + (ff_inner * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (yy_inner * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 401520)] = conv2d_nchw_local[(((ff_inner * 2) + yy_inner) + 20)];
      conv2d_nchw[((((((((((int)threadIdx.x) >> 3) * 6272) + (ff_inner * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (yy_inner * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 401632)] = conv2d_nchw_local[(((ff_inner * 2) + yy_inner) + 24)];
      conv2d_nchw[((((((((((int)threadIdx.x) >> 3) * 6272) + (ff_inner * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (yy_inner * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 401744)] = conv2d_nchw_local[(((ff_inner * 2) + yy_inner) + 28)];
    }
  }
}


