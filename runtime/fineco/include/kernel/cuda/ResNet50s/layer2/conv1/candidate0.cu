
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
extern "C" __global__ void __launch_bounds__(64) candidate0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[32];
  __shared__ float pad_temp_shared[1024];
  __shared__ float kernel_shared[2048];
  for (int ff_c_inner_init = 0; ff_c_inner_init < 4; ++ff_c_inner_init) {
    for (int xx_c_inner_init = 0; xx_c_inner_init < 2; ++xx_c_inner_init) {
      conv2d_nchw_local[((ff_c_inner_init * 2) + xx_c_inner_init)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_inner_init * 2) + xx_c_inner_init) + 8)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_inner_init * 2) + xx_c_inner_init) + 16)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_inner_init * 2) + xx_c_inner_init) + 24)] = 0.000000e+00f;
    }
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 2; ++rc_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 16; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
      pad_temp_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 64) + ((int)threadIdx.x))] = data[(((((((rc_outer_outer * 100352) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 6272)) + ((((int)threadIdx.x) >> 5) * 3136)) + ((((int)blockIdx.x) / 7) * 224)) + (((((int)threadIdx.x) & 31) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7))];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 < 32; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1) {
      kernel_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 64) + ((int)threadIdx.x))] = kernel[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 128) + ((((int)threadIdx.x) >> 5) * 64)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31))];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 8; ++rc_outer_inner) {
      for (int rc_inner = 0; rc_inner < 4; ++rc_inner) {
        for (int ff_c_inner = 0; ff_c_inner < 4; ++ff_c_inner) {
          for (int xx_c_inner = 0; xx_c_inner < 2; ++xx_c_inner) {
            conv2d_nchw_local[((ff_c_inner * 2) + xx_c_inner)] = (conv2d_nchw_local[((ff_c_inner * 2) + xx_c_inner)] + (pad_temp_shared[(((((rc_outer_inner * 128) + (rc_inner * 32)) + (((((int)threadIdx.x) & 7) >> 1) * 8)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_inner)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 128) + (ff_c_inner * 32)) + (rc_outer_inner * 4)) + rc_inner)]));
            conv2d_nchw_local[(((ff_c_inner * 2) + xx_c_inner) + 8)] = (conv2d_nchw_local[(((ff_c_inner * 2) + xx_c_inner) + 8)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (rc_inner * 32)) + (((((int)threadIdx.x) & 7) >> 1) * 8)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_inner) + 4)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 128) + (ff_c_inner * 32)) + (rc_outer_inner * 4)) + rc_inner)]));
            conv2d_nchw_local[(((ff_c_inner * 2) + xx_c_inner) + 16)] = (conv2d_nchw_local[(((ff_c_inner * 2) + xx_c_inner) + 16)] + (pad_temp_shared[(((((rc_outer_inner * 128) + (rc_inner * 32)) + (((((int)threadIdx.x) & 7) >> 1) * 8)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_inner)] * kernel_shared[((((((((int)threadIdx.x) >> 3) * 128) + (ff_c_inner * 32)) + (rc_outer_inner * 4)) + rc_inner) + 1024)]));
            conv2d_nchw_local[(((ff_c_inner * 2) + xx_c_inner) + 24)] = (conv2d_nchw_local[(((ff_c_inner * 2) + xx_c_inner) + 24)] + (pad_temp_shared[((((((rc_outer_inner * 128) + (rc_inner * 32)) + (((((int)threadIdx.x) & 7) >> 1) * 8)) + ((((int)threadIdx.x) & 1) * 2)) + xx_c_inner) + 4)] * kernel_shared[((((((((int)threadIdx.x) >> 3) * 128) + (ff_c_inner * 32)) + (rc_outer_inner * 4)) + rc_inner) + 1024)]));
          }
        }
      }
    }
  }
  for (int ff_inner = 0; ff_inner < 4; ++ff_inner) {
    for (int xx_inner = 0; xx_inner < 2; ++xx_inner) {
      conv2d_nchw[((((((((((int)threadIdx.x) >> 3) * 12544) + (ff_inner * 3136)) + ((((int)blockIdx.x) / 7) * 224)) + (((((int)threadIdx.x) & 7) >> 1) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + ((((int)threadIdx.x) & 1) * 2)) + xx_inner)] = conv2d_nchw_local[((ff_inner * 2) + xx_inner)];
      conv2d_nchw[(((((((((((int)threadIdx.x) >> 3) * 12544) + (ff_inner * 3136)) + ((((int)blockIdx.x) / 7) * 224)) + (((((int)threadIdx.x) & 7) >> 1) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + ((((int)threadIdx.x) & 1) * 2)) + xx_inner) + 4)] = conv2d_nchw_local[(((ff_inner * 2) + xx_inner) + 8)];
      conv2d_nchw[(((((((((((int)threadIdx.x) >> 3) * 12544) + (ff_inner * 3136)) + ((((int)blockIdx.x) / 7) * 224)) + (((((int)threadIdx.x) & 7) >> 1) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + ((((int)threadIdx.x) & 1) * 2)) + xx_inner) + 100352)] = conv2d_nchw_local[(((ff_inner * 2) + xx_inner) + 16)];
      conv2d_nchw[(((((((((((int)threadIdx.x) >> 3) * 12544) + (ff_inner * 3136)) + ((((int)blockIdx.x) / 7) * 224)) + (((((int)threadIdx.x) & 7) >> 1) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + ((((int)threadIdx.x) & 1) * 2)) + xx_inner) + 100356)] = conv2d_nchw_local[(((ff_inner * 2) + xx_inner) + 24)];
    }
  }
}


