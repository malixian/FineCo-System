
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
extern "C" __global__ void __launch_bounds__(16) candidate0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ compute, float* __restrict__ bias) {
  float conv2d_nchw[16];
  __shared__ float pad_temp_shared[2048];
  __shared__ float kernel_shared[8192];
  for (int yy_inner_init = 0; yy_inner_init < 2; ++yy_inner_init) {
    conv2d_nchw[yy_inner_init] = 0.000000e+00f;
    conv2d_nchw[(yy_inner_init + 2)] = 0.000000e+00f;
    conv2d_nchw[(yy_inner_init + 4)] = 0.000000e+00f;
    conv2d_nchw[(yy_inner_init + 6)] = 0.000000e+00f;
    conv2d_nchw[(yy_inner_init + 8)] = 0.000000e+00f;
    conv2d_nchw[(yy_inner_init + 10)] = 0.000000e+00f;
    conv2d_nchw[(yy_inner_init + 12)] = 0.000000e+00f;
    conv2d_nchw[(yy_inner_init + 14)] = 0.000000e+00f;
  }
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 128; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
    pad_temp_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 16) + ((int)threadIdx.x))] = data[((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 1568) + ((((int)threadIdx.x) >> 3) * 784)) + (((((int)blockIdx.x) % 98) / 7) * 56)) + (((((int)threadIdx.x) & 7) >> 2) * 28)) + ((((int)blockIdx.x) % 7) * 4)) + (((int)threadIdx.x) & 3))];
  }
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 < 128; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1) {
    *(float4*)(kernel_shared + ((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 64) + (((int)threadIdx.x) * 4))) = *(float4*)(kernel + ((((((int)blockIdx.x) / 98) * 8192) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 64)) + (((int)threadIdx.x) * 4)));
  }
  __syncthreads();
  for (int rc_outer_inner = 0; rc_outer_inner < 16; ++rc_outer_inner) {
    for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
      for (int yy_inner = 0; yy_inner < 2; ++yy_inner) {
        conv2d_nchw[yy_inner] = (conv2d_nchw[yy_inner] + (pad_temp_shared[(((rc_outer_inner * 128) + (rc_inner * 8)) + (yy_inner * 4))] * kernel_shared[(((((int)threadIdx.x) * 256) + (rc_outer_inner * 16)) + rc_inner)]));
        conv2d_nchw[(yy_inner + 2)] = (conv2d_nchw[(yy_inner + 2)] + (pad_temp_shared[((((rc_outer_inner * 128) + (rc_inner * 8)) + (yy_inner * 4)) + 1)] * kernel_shared[(((((int)threadIdx.x) * 256) + (rc_outer_inner * 16)) + rc_inner)]));
        conv2d_nchw[(yy_inner + 4)] = (conv2d_nchw[(yy_inner + 4)] + (pad_temp_shared[((((rc_outer_inner * 128) + (rc_inner * 8)) + (yy_inner * 4)) + 2)] * kernel_shared[(((((int)threadIdx.x) * 256) + (rc_outer_inner * 16)) + rc_inner)]));
        conv2d_nchw[(yy_inner + 6)] = (conv2d_nchw[(yy_inner + 6)] + (pad_temp_shared[((((rc_outer_inner * 128) + (rc_inner * 8)) + (yy_inner * 4)) + 3)] * kernel_shared[(((((int)threadIdx.x) * 256) + (rc_outer_inner * 16)) + rc_inner)]));
        conv2d_nchw[(yy_inner + 8)] = (conv2d_nchw[(yy_inner + 8)] + (pad_temp_shared[(((rc_outer_inner * 128) + (rc_inner * 8)) + (yy_inner * 4))] * kernel_shared[((((((int)threadIdx.x) * 256) + (rc_outer_inner * 16)) + rc_inner) + 4096)]));
        conv2d_nchw[(yy_inner + 10)] = (conv2d_nchw[(yy_inner + 10)] + (pad_temp_shared[((((rc_outer_inner * 128) + (rc_inner * 8)) + (yy_inner * 4)) + 1)] * kernel_shared[((((((int)threadIdx.x) * 256) + (rc_outer_inner * 16)) + rc_inner) + 4096)]));
        conv2d_nchw[(yy_inner + 12)] = (conv2d_nchw[(yy_inner + 12)] + (pad_temp_shared[((((rc_outer_inner * 128) + (rc_inner * 8)) + (yy_inner * 4)) + 2)] * kernel_shared[((((((int)threadIdx.x) * 256) + (rc_outer_inner * 16)) + rc_inner) + 4096)]));
        conv2d_nchw[(yy_inner + 14)] = (conv2d_nchw[(yy_inner + 14)] + (pad_temp_shared[((((rc_outer_inner * 128) + (rc_inner * 8)) + (yy_inner * 4)) + 3)] * kernel_shared[((((((int)threadIdx.x) * 256) + (rc_outer_inner * 16)) + rc_inner) + 4096)]));
      }
    }
  }
  for (int i2_inner = 0; i2_inner < 2; ++i2_inner) {
    compute[((((((((int)blockIdx.x) / 98) * 25088) + (((int)threadIdx.x) * 784)) + (((((int)blockIdx.x) % 98) / 7) * 56)) + (i2_inner * 28)) + ((((int)blockIdx.x) % 7) * 4))] = max((conv2d_nchw[i2_inner] + bias[(((((int)blockIdx.x) / 98) * 32) + ((int)threadIdx.x))]), 0.000000e+00f);
    compute[(((((((((int)blockIdx.x) / 98) * 25088) + (((int)threadIdx.x) * 784)) + (((((int)blockIdx.x) % 98) / 7) * 56)) + (i2_inner * 28)) + ((((int)blockIdx.x) % 7) * 4)) + 1)] = max((conv2d_nchw[(i2_inner + 2)] + bias[(((((int)blockIdx.x) / 98) * 32) + ((int)threadIdx.x))]), 0.000000e+00f);
    compute[(((((((((int)blockIdx.x) / 98) * 25088) + (((int)threadIdx.x) * 784)) + (((((int)blockIdx.x) % 98) / 7) * 56)) + (i2_inner * 28)) + ((((int)blockIdx.x) % 7) * 4)) + 2)] = max((conv2d_nchw[(i2_inner + 4)] + bias[(((((int)blockIdx.x) / 98) * 32) + ((int)threadIdx.x))]), 0.000000e+00f);
    compute[(((((((((int)blockIdx.x) / 98) * 25088) + (((int)threadIdx.x) * 784)) + (((((int)blockIdx.x) % 98) / 7) * 56)) + (i2_inner * 28)) + ((((int)blockIdx.x) % 7) * 4)) + 3)] = max((conv2d_nchw[(i2_inner + 6)] + bias[(((((int)blockIdx.x) / 98) * 32) + ((int)threadIdx.x))]), 0.000000e+00f);
    compute[(((((((((int)blockIdx.x) / 98) * 25088) + (((int)threadIdx.x) * 784)) + (((((int)blockIdx.x) % 98) / 7) * 56)) + (i2_inner * 28)) + ((((int)blockIdx.x) % 7) * 4)) + 12544)] = max((conv2d_nchw[(i2_inner + 8)] + bias[((((((int)blockIdx.x) / 98) * 32) + ((int)threadIdx.x)) + 16)]), 0.000000e+00f);
    compute[(((((((((int)blockIdx.x) / 98) * 25088) + (((int)threadIdx.x) * 784)) + (((((int)blockIdx.x) % 98) / 7) * 56)) + (i2_inner * 28)) + ((((int)blockIdx.x) % 7) * 4)) + 12545)] = max((conv2d_nchw[(i2_inner + 10)] + bias[((((((int)blockIdx.x) / 98) * 32) + ((int)threadIdx.x)) + 16)]), 0.000000e+00f);
    compute[(((((((((int)blockIdx.x) / 98) * 25088) + (((int)threadIdx.x) * 784)) + (((((int)blockIdx.x) % 98) / 7) * 56)) + (i2_inner * 28)) + ((((int)blockIdx.x) % 7) * 4)) + 12546)] = max((conv2d_nchw[(i2_inner + 12)] + bias[((((((int)blockIdx.x) / 98) * 32) + ((int)threadIdx.x)) + 16)]), 0.000000e+00f);
    compute[(((((((((int)blockIdx.x) / 98) * 25088) + (((int)threadIdx.x) * 784)) + (((((int)blockIdx.x) % 98) / 7) * 56)) + (i2_inner * 28)) + ((((int)blockIdx.x) % 7) * 4)) + 12547)] = max((conv2d_nchw[(i2_inner + 14)] + bias[((((((int)blockIdx.x) / 98) * 32) + ((int)threadIdx.x)) + 16)]), 0.000000e+00f);
  }
}


