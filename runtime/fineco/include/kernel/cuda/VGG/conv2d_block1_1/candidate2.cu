
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
extern "C" __global__ void __launch_bounds__(224) candidate2(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[512];
  __shared__ float pad_temp_shared[5916];
  __shared__ float kernel_shared[1728];
  for (int ff_c_outer_inner_init = 0; ff_c_outer_inner_init < 32; ++ff_c_outer_inner_init) {
    for (int yy_c_inner_init = 0; yy_c_inner_init < 8; ++yy_c_inner_init) {
      for (int xx_c_inner_init = 0; xx_c_inner_init < 2; ++xx_c_inner_init) {
        conv2d_nchw_local[(((ff_c_outer_inner_init * 16) + (yy_c_inner_init * 2)) + xx_c_inner_init)] = 0.000000e+00f;
      }
    }
  }
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 27; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
    if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 56) + (((int)threadIdx.x) >> 2)) < 1479) {
      pad_temp_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 224) + ((int)threadIdx.x))] = (((((1 <= (((((int)blockIdx.x) / 7) * 56) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 112) + (((int)threadIdx.x) >> 1)) % 986) / 17))) && ((((((int)blockIdx.x) / 7) * 56) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 112) + (((int)threadIdx.x) >> 1)) % 986) / 17)) < 225)) && (1 <= (((((int)blockIdx.x) % 7) * 32) + (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 224) + ((int)threadIdx.x)) % 34)))) && ((((((int)blockIdx.x) % 7) * 32) + (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 224) + ((int)threadIdx.x)) % 34)) < 225)) ? data[(((((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 56) + (((int)threadIdx.x) >> 2)) / 493) * 50176) + ((((int)blockIdx.x) / 7) * 12544)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 112) + (((int)threadIdx.x) >> 1)) % 986) / 17) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 224) + ((int)threadIdx.x)) % 34)) - 225)] : 0.000000e+00f);
    }
  }
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1) {
    if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 7) + (((int)threadIdx.x) >> 5)) < 54) {
      kernel_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 224) + ((int)threadIdx.x))] = kernel[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 224) + ((int)threadIdx.x))];
    }
  }
  __syncthreads();
  for (int rx_outer_inner = 0; rx_outer_inner < 3; ++rx_outer_inner) {
    for (int ff_c_outer_inner = 0; ff_c_outer_inner < 32; ++ff_c_outer_inner) {
      for (int rc_inner = 0; rc_inner < 3; ++rc_inner) {
        for (int ry_inner = 0; ry_inner < 3; ++ry_inner) {
          for (int yy_c_inner = 0; yy_c_inner < 8; ++yy_c_inner) {
            for (int xx_c_inner = 0; xx_c_inner < 2; ++xx_c_inner) {
              conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_inner * 2)) + xx_c_inner)] = (conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_inner * 2)) + xx_c_inner)] + (pad_temp_shared[(((((((rc_inner * 1972) + (((((int)threadIdx.x) % 112) >> 4) * 272)) + (yy_c_inner * 34)) + (ry_inner * 34)) + ((((int)threadIdx.x) & 15) * 2)) + xx_c_inner) + rx_outer_inner)] * kernel_shared[((((((((int)threadIdx.x) / 112) * 864) + (ff_c_outer_inner * 27)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_outer_inner)]));
            }
          }
        }
      }
    }
  }
  for (int ff_inner = 0; ff_inner < 32; ++ff_inner) {
    for (int yy_inner = 0; yy_inner < 8; ++yy_inner) {
      for (int xx_inner = 0; xx_inner < 2; ++xx_inner) {
        conv2d_nchw[(((((((((((int)threadIdx.x) / 112) * 1605632) + (ff_inner * 50176)) + ((((int)blockIdx.x) / 7) * 12544)) + (((((int)threadIdx.x) % 112) >> 4) * 1792)) + (yy_inner * 224)) + ((((int)blockIdx.x) % 7) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + xx_inner)] = conv2d_nchw_local[(((ff_inner * 16) + (yy_inner * 2)) + xx_inner)];
      }
    }
  }
}


