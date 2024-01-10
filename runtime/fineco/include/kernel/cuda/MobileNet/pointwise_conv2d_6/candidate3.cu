
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
extern "C" __global__ void __launch_bounds__(196) candidate3(float* __restrict__ Input, float* __restrict__ kernel, float* __restrict__ compute) {
  float conv2d_nchw[16];
  __shared__ float pad_temp_shared[6272];
  __shared__ float kernel_shared[2048];
  for (int ff_outer_inner_init = 0; ff_outer_inner_init < 4; ++ff_outer_inner_init) {
    for (int yy_inner_init = 0; yy_inner_init < 2; ++yy_inner_init) {
      conv2d_nchw[((ff_outer_inner_init * 2) + yy_inner_init)] = 0.000000e+00f;
      conv2d_nchw[(((ff_outer_inner_init * 2) + yy_inner_init) + 8)] = 0.000000e+00f;
    }
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 4; ++rc_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 32; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
      pad_temp_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 196) + ((int)threadIdx.x))] = Input[(((((rc_outer_outer * 12544) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 392)) + ((((int)threadIdx.x) / 7) * 14)) + ((((int)blockIdx.x) & 1) * 7)) + (((int)threadIdx.x) % 7))];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 < 11; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1) {
      if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 49) + (((int)threadIdx.x) >> 2)) < 512) {
        kernel_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 196) + ((int)threadIdx.x))] = kernel[(((((((int)blockIdx.x) >> 1) * 8192) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 49) + (((int)threadIdx.x) >> 2)) >> 4) * 256)) + (rc_outer_outer * 64)) + (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 196) + ((int)threadIdx.x)) & 63))];
      }
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 4; ++rc_outer_inner) {
      for (int ff_outer_inner = 0; ff_outer_inner < 4; ++ff_outer_inner) {
        for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
          for (int yy_inner = 0; yy_inner < 2; ++yy_inner) {
            conv2d_nchw[((ff_outer_inner * 2) + yy_inner)] = (conv2d_nchw[((ff_outer_inner * 2) + yy_inner)] + (pad_temp_shared[(((((rc_outer_inner * 1568) + (rc_inner * 98)) + (((((int)threadIdx.x) % 49) / 7) * 14)) + (yy_inner * 7)) + (((int)threadIdx.x) % 7))] * kernel_shared[(((((((int)threadIdx.x) / 49) * 256) + (ff_outer_inner * 64)) + (rc_outer_inner * 16)) + rc_inner)]));
            conv2d_nchw[(((ff_outer_inner * 2) + yy_inner) + 8)] = (conv2d_nchw[(((ff_outer_inner * 2) + yy_inner) + 8)] + (pad_temp_shared[(((((rc_outer_inner * 1568) + (rc_inner * 98)) + (((((int)threadIdx.x) % 49) / 7) * 14)) + (yy_inner * 7)) + (((int)threadIdx.x) % 7))] * kernel_shared[((((((((int)threadIdx.x) / 49) * 256) + (ff_outer_inner * 64)) + (rc_outer_inner * 16)) + rc_inner) + 1024)]));
          }
        }
      }
    }
  }
  for (int i1_inner = 0; i1_inner < 4; ++i1_inner) {
    for (int i2_inner = 0; i2_inner < 2; ++i2_inner) {
      compute[((((((((((int)blockIdx.x) >> 1) * 6272) + ((((int)threadIdx.x) / 49) * 784)) + (i1_inner * 196)) + (((((int)threadIdx.x) % 49) / 7) * 28)) + (i2_inner * 14)) + ((((int)blockIdx.x) & 1) * 7)) + (((int)threadIdx.x) % 7))] = max(conv2d_nchw[((i1_inner * 2) + i2_inner)], 0.000000e+00f);
      compute[(((((((((((int)blockIdx.x) >> 1) * 6272) + ((((int)threadIdx.x) / 49) * 784)) + (i1_inner * 196)) + (((((int)threadIdx.x) % 49) / 7) * 28)) + (i2_inner * 14)) + ((((int)blockIdx.x) & 1) * 7)) + (((int)threadIdx.x) % 7)) + 3136)] = max(conv2d_nchw[(((i1_inner * 2) + i2_inner) + 8)], 0.000000e+00f);
    }
  }
}


