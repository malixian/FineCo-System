
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
  float conv2d_nchw_local[8];
  __shared__ float pad_temp_shared[4096];
  __shared__ float kernel_shared[4096];
  for (int ff_c_outer_inner_init = 0; ff_c_outer_inner_init < 2; ++ff_c_outer_inner_init) {
    for (int yy_c_outer_inner_init = 0; yy_c_outer_inner_init < 2; ++yy_c_outer_inner_init) {
      conv2d_nchw_local[((ff_c_outer_inner_init * 2) + yy_c_outer_inner_init)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_outer_inner_init * 2) + yy_c_outer_inner_init) + 4)] = 0.000000e+00f;
    }
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 4; ++rc_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
      pad_temp_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 512) + ((int)threadIdx.x))] = data[(((((((rc_outer_outer * 200704) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 25088)) + ((((int)threadIdx.x) >> 6) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7))];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1) {
      kernel_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 512) + ((int)threadIdx.x))] = kernel[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 2048) + ((((int)threadIdx.x) >> 6) * 256)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63))];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 4; ++rc_outer_inner) {
      for (int ff_c_outer_inner = 0; ff_c_outer_inner < 2; ++ff_c_outer_inner) {
        for (int yy_c_outer_inner = 0; yy_c_outer_inner < 2; ++yy_c_outer_inner) {
          for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
            conv2d_nchw_local[((ff_c_outer_inner * 2) + yy_c_outer_inner)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + yy_c_outer_inner)] + (pad_temp_shared[(((((rc_outer_inner * 1024) + (rc_inner * 64)) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (yy_c_outer_inner * 8)) + (((int)threadIdx.x) & 7))] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 128) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 16)) + rc_inner)]));
            conv2d_nchw_local[(((ff_c_outer_inner * 2) + yy_c_outer_inner) + 4)] = (conv2d_nchw_local[(((ff_c_outer_inner * 2) + yy_c_outer_inner) + 4)] + (pad_temp_shared[((((((rc_outer_inner * 1024) + (rc_inner * 64)) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (yy_c_outer_inner * 8)) + (((int)threadIdx.x) & 7)) + 32)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 128) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 16)) + rc_inner)]));
          }
        }
      }
    }
  }
  for (int ff_inner = 0; ff_inner < 2; ++ff_inner) {
    for (int yy_inner = 0; yy_inner < 2; ++yy_inner) {
      conv2d_nchw[((((((((((int)threadIdx.x) >> 4) * 6272) + (ff_inner * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (yy_inner * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7))] = conv2d_nchw_local[((ff_inner * 2) + yy_inner)];
      conv2d_nchw[(((((((((((int)threadIdx.x) >> 4) * 6272) + (ff_inner * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (yy_inner * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 224)] = conv2d_nchw_local[(((ff_inner * 2) + yy_inner) + 4)];
    }
  }
}


