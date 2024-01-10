
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
extern "C" __global__ void __launch_bounds__(56) candidate0(float* __restrict__ Input, float* __restrict__ kernel, float* __restrict__ compute) {
  float conv2d_nchw[16];
  __shared__ float pad_temp_shared[1792];
  __shared__ float kernel_shared[512];
  for (int ff_outer_inner_init = 0; ff_outer_inner_init < 2; ++ff_outer_inner_init) {
    for (int ff_inner_init = 0; ff_inner_init < 4; ++ff_inner_init) {
      conv2d_nchw[((ff_outer_inner_init * 4) + ff_inner_init)] = 0.000000e+00f;
      conv2d_nchw[(((ff_outer_inner_init * 4) + ff_inner_init) + 8)] = 0.000000e+00f;
    }
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 4; ++rc_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 32; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
      pad_temp_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 56) + ((int)threadIdx.x))] = Input[((((((rc_outer_outer * 100352) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 3136)) + (((((int)blockIdx.x) % 56) >> 1) * 112)) + ((((int)threadIdx.x) / 28) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + (((int)threadIdx.x) % 28))];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 < 10; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1) {
      if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 7) + (((int)threadIdx.x) >> 3)) < 64) {
        kernel_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 56) + ((int)threadIdx.x))] = kernel[(((((((int)blockIdx.x) / 56) * 2048) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 7) + (((int)threadIdx.x) >> 3)) >> 2) * 128)) + (rc_outer_outer * 32)) + (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 56) + ((int)threadIdx.x)) & 31))];
      }
    }
    __syncthreads();
    for (int ff_outer_inner = 0; ff_outer_inner < 2; ++ff_outer_inner) {
      for (int rc_inner = 0; rc_inner < 32; ++rc_inner) {
        for (int ff_inner = 0; ff_inner < 4; ++ff_inner) {
          conv2d_nchw[((ff_outer_inner * 4) + ff_inner)] = (conv2d_nchw[((ff_outer_inner * 4) + ff_inner)] + (pad_temp_shared[((rc_inner * 56) + ((int)threadIdx.x))] * kernel_shared[(((ff_outer_inner * 128) + (ff_inner * 32)) + rc_inner)]));
          conv2d_nchw[(((ff_outer_inner * 4) + ff_inner) + 8)] = (conv2d_nchw[(((ff_outer_inner * 4) + ff_inner) + 8)] + (pad_temp_shared[((rc_inner * 56) + ((int)threadIdx.x))] * kernel_shared[((((ff_outer_inner * 128) + (ff_inner * 32)) + rc_inner) + 256)]));
        }
      }
    }
  }
  for (int i1_inner = 0; i1_inner < 8; ++i1_inner) {
    compute[(((((((((int)blockIdx.x) / 56) * 50176) + (i1_inner * 3136)) + (((((int)blockIdx.x) % 56) >> 1) * 112)) + ((((int)threadIdx.x) / 28) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + (((int)threadIdx.x) % 28))] = max(conv2d_nchw[i1_inner], 0.000000e+00f);
    compute[((((((((((int)blockIdx.x) / 56) * 50176) + (i1_inner * 3136)) + (((((int)blockIdx.x) % 56) >> 1) * 112)) + ((((int)threadIdx.x) / 28) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + (((int)threadIdx.x) % 28)) + 25088)] = max(conv2d_nchw[(i1_inner + 8)], 0.000000e+00f);
  }
}


