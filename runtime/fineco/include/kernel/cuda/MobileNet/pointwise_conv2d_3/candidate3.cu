
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
extern "C" __global__ void __launch_bounds__(256) candidate3(float* __restrict__ Input, float* __restrict__ kernel, float* __restrict__ compute) {
  float conv2d_nchw[32];
  __shared__ float pad_temp_shared[4096];
  __shared__ float kernel_shared[8192];
  for (int ff_outer_inner_init = 0; ff_outer_inner_init < 2; ++ff_outer_inner_init) {
    for (int xx_outer_inner_init = 0; xx_outer_inner_init < 2; ++xx_outer_inner_init) {
      for (int yy_inner_init = 0; yy_inner_init < 2; ++yy_inner_init) {
        conv2d_nchw[(((ff_outer_inner_init * 4) + (yy_inner_init * 2)) + xx_outer_inner_init)] = 0.000000e+00f;
        conv2d_nchw[((((ff_outer_inner_init * 4) + (yy_inner_init * 2)) + xx_outer_inner_init) + 8)] = 0.000000e+00f;
        conv2d_nchw[((((ff_outer_inner_init * 4) + (yy_inner_init * 2)) + xx_outer_inner_init) + 16)] = 0.000000e+00f;
        conv2d_nchw[((((ff_outer_inner_init * 4) + (yy_inner_init * 2)) + xx_outer_inner_init) + 24)] = 0.000000e+00f;
      }
    }
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 2; ++rc_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 16; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
      pad_temp_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 256) + ((int)threadIdx.x))] = Input[(((((((rc_outer_outer * 200704) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 12544)) + ((((int)threadIdx.x) >> 6) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7))];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 < 32; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1) {
      kernel_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 256) + ((int)threadIdx.x))] = kernel[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63))];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 64; ++rc_outer_inner) {
      for (int ff_outer_inner = 0; ff_outer_inner < 2; ++ff_outer_inner) {
        for (int xx_outer_inner = 0; xx_outer_inner < 2; ++xx_outer_inner) {
          for (int yy_inner = 0; yy_inner < 2; ++yy_inner) {
            conv2d_nchw[(((ff_outer_inner * 4) + (yy_inner * 2)) + xx_outer_inner)] = (conv2d_nchw[(((ff_outer_inner * 4) + (yy_inner * 2)) + xx_outer_inner)] + (pad_temp_shared[(((((rc_outer_inner * 64) + (((((int)threadIdx.x) & 15) >> 2) * 16)) + (yy_inner * 8)) + ((((int)threadIdx.x) & 3) * 2)) + xx_outer_inner)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 128) + (ff_outer_inner * 64)) + rc_outer_inner)]));
            conv2d_nchw[((((ff_outer_inner * 4) + (yy_inner * 2)) + xx_outer_inner) + 8)] = (conv2d_nchw[((((ff_outer_inner * 4) + (yy_inner * 2)) + xx_outer_inner) + 8)] + (pad_temp_shared[(((((rc_outer_inner * 64) + (((((int)threadIdx.x) & 15) >> 2) * 16)) + (yy_inner * 8)) + ((((int)threadIdx.x) & 3) * 2)) + xx_outer_inner)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 128) + (ff_outer_inner * 64)) + rc_outer_inner) + 2048)]));
            conv2d_nchw[((((ff_outer_inner * 4) + (yy_inner * 2)) + xx_outer_inner) + 16)] = (conv2d_nchw[((((ff_outer_inner * 4) + (yy_inner * 2)) + xx_outer_inner) + 16)] + (pad_temp_shared[(((((rc_outer_inner * 64) + (((((int)threadIdx.x) & 15) >> 2) * 16)) + (yy_inner * 8)) + ((((int)threadIdx.x) & 3) * 2)) + xx_outer_inner)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 128) + (ff_outer_inner * 64)) + rc_outer_inner) + 4096)]));
            conv2d_nchw[((((ff_outer_inner * 4) + (yy_inner * 2)) + xx_outer_inner) + 24)] = (conv2d_nchw[((((ff_outer_inner * 4) + (yy_inner * 2)) + xx_outer_inner) + 24)] + (pad_temp_shared[(((((rc_outer_inner * 64) + (((((int)threadIdx.x) & 15) >> 2) * 16)) + (yy_inner * 8)) + ((((int)threadIdx.x) & 3) * 2)) + xx_outer_inner)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 128) + (ff_outer_inner * 64)) + rc_outer_inner) + 6144)]));
          }
        }
      }
    }
  }
  for (int i1_inner = 0; i1_inner < 2; ++i1_inner) {
    for (int i2_inner = 0; i2_inner < 2; ++i2_inner) {
      for (int i3_inner = 0; i3_inner < 2; ++i3_inner) {
        compute[(((((((((((int)threadIdx.x) >> 4) * 6272) + (i1_inner * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((int)threadIdx.x) & 15) >> 2) * 112)) + (i2_inner * 56)) + ((((int)blockIdx.x) % 7) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + i3_inner)] = max(conv2d_nchw[(((i1_inner * 4) + (i2_inner * 2)) + i3_inner)], 0.000000e+00f);
        compute[((((((((((((int)threadIdx.x) >> 4) * 6272) + (i1_inner * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((int)threadIdx.x) & 15) >> 2) * 112)) + (i2_inner * 56)) + ((((int)blockIdx.x) % 7) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + i3_inner) + 100352)] = max(conv2d_nchw[((((i1_inner * 4) + (i2_inner * 2)) + i3_inner) + 8)], 0.000000e+00f);
        compute[((((((((((((int)threadIdx.x) >> 4) * 6272) + (i1_inner * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((int)threadIdx.x) & 15) >> 2) * 112)) + (i2_inner * 56)) + ((((int)blockIdx.x) % 7) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + i3_inner) + 200704)] = max(conv2d_nchw[((((i1_inner * 4) + (i2_inner * 2)) + i3_inner) + 16)], 0.000000e+00f);
        compute[((((((((((((int)threadIdx.x) >> 4) * 6272) + (i1_inner * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((int)threadIdx.x) & 15) >> 2) * 112)) + (i2_inner * 56)) + ((((int)blockIdx.x) % 7) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + i3_inner) + 301056)] = max(conv2d_nchw[((((i1_inner * 4) + (i2_inner * 2)) + i3_inner) + 24)], 0.000000e+00f);
      }
    }
  }
}


