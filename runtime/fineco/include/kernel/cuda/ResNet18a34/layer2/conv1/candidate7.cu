
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
extern "C" __global__ void __launch_bounds__(128) candidate7(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[16];
  __shared__ float pad_temp_shared[640];
  __shared__ float kernel_shared[3072];
  for (int ff_c_outer_inner_init = 0; ff_c_outer_inner_init < 4; ++ff_c_outer_inner_init) {
    for (int yy_c_outer_inner_init = 0; yy_c_outer_inner_init < 2; ++yy_c_outer_inner_init) {
      for (int ff_c_inner_init = 0; ff_c_inner_init < 2; ++ff_c_inner_init) {
        conv2d_nchw_local[(((ff_c_outer_inner_init * 4) + (ff_c_inner_init * 2)) + yy_c_outer_inner_init)] = 0.000000e+00f;
      }
    }
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 4; ++rc_outer_outer) {
    for (int rx_outer_outer = 0; rx_outer_outer < 3; ++rx_outer_outer) {
      __syncthreads();
      for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
        pad_temp_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 128) + ((int)threadIdx.x))] = (((((1 <= (((((int)blockIdx.x) / 14) * 8) + (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 32) + (((int)threadIdx.x) >> 2)) % 10))) && ((((((int)blockIdx.x) / 14) * 8) + (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 32) + (((int)threadIdx.x) >> 2)) % 10)) < 57)) && (1 <= ((((((int)blockIdx.x) % 14) * 4) + rx_outer_outer) + (((int)threadIdx.x) & 3)))) && (((((((int)blockIdx.x) % 14) * 4) + rx_outer_outer) + (((int)threadIdx.x) & 3)) < 57)) ? data[((((((((rc_outer_outer * 50176) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 16) + (((int)threadIdx.x) >> 3)) / 5) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 32) + (((int)threadIdx.x) >> 2)) % 10) * 56)) + ((((int)blockIdx.x) % 14) * 4)) + rx_outer_outer) + (((int)threadIdx.x) & 3)) - 57)] : 0.000000e+00f);
      }
      for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 < 24; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1) {
        kernel_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 128) + ((int)threadIdx.x))] = kernel[(((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 8) + (((int)threadIdx.x) >> 4)) / 3) * 576) + (rc_outer_outer * 144)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 128) + ((int)threadIdx.x)) % 48) * 3)) + rx_outer_outer)];
      }
      __syncthreads();
      for (int rc_outer_inner = 0; rc_outer_inner < 8; ++rc_outer_inner) {
        for (int ff_c_outer_inner = 0; ff_c_outer_inner < 4; ++ff_c_outer_inner) {
          for (int yy_c_outer_inner = 0; yy_c_outer_inner < 2; ++yy_c_outer_inner) {
            for (int rc_inner = 0; rc_inner < 2; ++rc_inner) {
              for (int ry_inner = 0; ry_inner < 3; ++ry_inner) {
                for (int ff_c_inner = 0; ff_c_inner < 2; ++ff_c_inner) {
                  conv2d_nchw_local[(((ff_c_outer_inner * 4) + (ff_c_inner * 2)) + yy_c_outer_inner)] = (conv2d_nchw_local[(((ff_c_outer_inner * 4) + (ff_c_inner * 2)) + yy_c_outer_inner)] + (pad_temp_shared[((((((rc_outer_inner * 80) + (rc_inner * 40)) + (((((int)threadIdx.x) & 15) >> 2) * 8)) + (yy_c_outer_inner * 4)) + (ry_inner * 4)) + (((int)threadIdx.x) & 3))] * kernel_shared[(((((((((int)threadIdx.x) >> 4) * 384) + (ff_c_outer_inner * 96)) + (ff_c_inner * 48)) + (rc_outer_inner * 6)) + (rc_inner * 3)) + ry_inner)]));
                }
              }
            }
          }
        }
      }
    }
  }
  for (int ff_inner = 0; ff_inner < 8; ++ff_inner) {
    for (int yy_inner = 0; yy_inner < 2; ++yy_inner) {
      conv2d_nchw[((((((((((int)threadIdx.x) >> 4) * 25088) + (ff_inner * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + (((((int)threadIdx.x) & 15) >> 2) * 112)) + (yy_inner * 56)) + ((((int)blockIdx.x) % 14) * 4)) + (((int)threadIdx.x) & 3))] = conv2d_nchw_local[((ff_inner * 2) + yy_inner)];
    }
  }
}


