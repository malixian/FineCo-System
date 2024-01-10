
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
extern "C" __global__ void __launch_bounds__(224) candidate2(float* __restrict__ Input, float* __restrict__ kernel, float* __restrict__ compute) {
  float conv2d_nchw[32];
  __shared__ float pad_temp_shared[1792];
  __shared__ float kernel_shared[4096];
  for (int yy_outer_inner_init = 0; yy_outer_inner_init < 4; ++yy_outer_inner_init) {
    conv2d_nchw[yy_outer_inner_init] = 0.000000e+00f;
    conv2d_nchw[(yy_outer_inner_init + 16)] = 0.000000e+00f;
    conv2d_nchw[(yy_outer_inner_init + 4)] = 0.000000e+00f;
    conv2d_nchw[(yy_outer_inner_init + 20)] = 0.000000e+00f;
    conv2d_nchw[(yy_outer_inner_init + 8)] = 0.000000e+00f;
    conv2d_nchw[(yy_outer_inner_init + 24)] = 0.000000e+00f;
    conv2d_nchw[(yy_outer_inner_init + 12)] = 0.000000e+00f;
    conv2d_nchw[(yy_outer_inner_init + 28)] = 0.000000e+00f;
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 4; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = Input[((((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 56) * 784)) + (((((int)blockIdx.x) % 14) >> 1) * 112)) + (((((int)threadIdx.x) % 56) / 14) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((int)threadIdx.x) % 14))];
    pad_temp_shared[(((int)threadIdx.x) + 224)] = Input[(((((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 56) * 784)) + (((((int)blockIdx.x) % 14) >> 1) * 112)) + (((((int)threadIdx.x) % 56) / 14) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((int)threadIdx.x) % 14)) + 3136)];
    pad_temp_shared[(((int)threadIdx.x) + 448)] = Input[(((((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 56) * 784)) + (((((int)blockIdx.x) % 14) >> 1) * 112)) + (((((int)threadIdx.x) % 56) / 14) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((int)threadIdx.x) % 14)) + 6272)];
    pad_temp_shared[(((int)threadIdx.x) + 672)] = Input[(((((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 56) * 784)) + (((((int)blockIdx.x) % 14) >> 1) * 112)) + (((((int)threadIdx.x) % 56) / 14) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((int)threadIdx.x) % 14)) + 9408)];
    pad_temp_shared[(((int)threadIdx.x) + 896)] = Input[(((((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 56) * 784)) + (((((int)blockIdx.x) % 14) >> 1) * 112)) + (((((int)threadIdx.x) % 56) / 14) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((int)threadIdx.x) % 14)) + 12544)];
    pad_temp_shared[(((int)threadIdx.x) + 1120)] = Input[(((((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 56) * 784)) + (((((int)blockIdx.x) % 14) >> 1) * 112)) + (((((int)threadIdx.x) % 56) / 14) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((int)threadIdx.x) % 14)) + 15680)];
    pad_temp_shared[(((int)threadIdx.x) + 1344)] = Input[(((((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 56) * 784)) + (((((int)blockIdx.x) % 14) >> 1) * 112)) + (((((int)threadIdx.x) % 56) / 14) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((int)threadIdx.x) % 14)) + 18816)];
    pad_temp_shared[(((int)threadIdx.x) + 1568)] = Input[(((((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 56) * 784)) + (((((int)blockIdx.x) % 14) >> 1) * 112)) + (((((int)threadIdx.x) % 56) / 14) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((int)threadIdx.x) % 14)) + 21952)];
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 19; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
      if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 7) + (((int)threadIdx.x) >> 5)) < 128) {
        kernel_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 224) + ((int)threadIdx.x))] = kernel[((((((((int)blockIdx.x) / 14) * 16384) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 896)) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31))];
      }
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 4; ++rc_outer_inner) {
      for (int yy_outer_inner = 0; yy_outer_inner < 4; ++yy_outer_inner) {
        for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
          conv2d_nchw[yy_outer_inner] = (conv2d_nchw[yy_outer_inner] + (pad_temp_shared[((((rc_outer_inner * 448) + (rc_inner * 56)) + (yy_outer_inner * 14)) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 8)) + rc_inner)]));
          conv2d_nchw[(yy_outer_inner + 16)] = (conv2d_nchw[(yy_outer_inner + 16)] + (pad_temp_shared[((((rc_outer_inner * 448) + (rc_inner * 56)) + (yy_outer_inner * 14)) + (((int)threadIdx.x) % 14))] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 8)) + rc_inner) + 2048)]));
          conv2d_nchw[(yy_outer_inner + 4)] = (conv2d_nchw[(yy_outer_inner + 4)] + (pad_temp_shared[((((rc_outer_inner * 448) + (rc_inner * 56)) + (yy_outer_inner * 14)) + (((int)threadIdx.x) % 14))] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 8)) + rc_inner) + 32)]));
          conv2d_nchw[(yy_outer_inner + 20)] = (conv2d_nchw[(yy_outer_inner + 20)] + (pad_temp_shared[((((rc_outer_inner * 448) + (rc_inner * 56)) + (yy_outer_inner * 14)) + (((int)threadIdx.x) % 14))] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 8)) + rc_inner) + 2080)]));
          conv2d_nchw[(yy_outer_inner + 8)] = (conv2d_nchw[(yy_outer_inner + 8)] + (pad_temp_shared[((((rc_outer_inner * 448) + (rc_inner * 56)) + (yy_outer_inner * 14)) + (((int)threadIdx.x) % 14))] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 8)) + rc_inner) + 64)]));
          conv2d_nchw[(yy_outer_inner + 24)] = (conv2d_nchw[(yy_outer_inner + 24)] + (pad_temp_shared[((((rc_outer_inner * 448) + (rc_inner * 56)) + (yy_outer_inner * 14)) + (((int)threadIdx.x) % 14))] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 8)) + rc_inner) + 2112)]));
          conv2d_nchw[(yy_outer_inner + 12)] = (conv2d_nchw[(yy_outer_inner + 12)] + (pad_temp_shared[((((rc_outer_inner * 448) + (rc_inner * 56)) + (yy_outer_inner * 14)) + (((int)threadIdx.x) % 14))] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 8)) + rc_inner) + 96)]));
          conv2d_nchw[(yy_outer_inner + 28)] = (conv2d_nchw[(yy_outer_inner + 28)] + (pad_temp_shared[((((rc_outer_inner * 448) + (rc_inner * 56)) + (yy_outer_inner * 14)) + (((int)threadIdx.x) % 14))] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 8)) + rc_inner) + 2144)]));
        }
      }
    }
  }
  for (int i1_inner = 0; i1_inner < 4; ++i1_inner) {
    for (int i2_inner = 0; i2_inner < 4; ++i2_inner) {
      compute[((((((((((int)blockIdx.x) / 14) * 100352) + ((((int)threadIdx.x) / 14) * 3136)) + (i1_inner * 784)) + (((((int)blockIdx.x) % 14) >> 1) * 112)) + (i2_inner * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((int)threadIdx.x) % 14))] = max(conv2d_nchw[((i1_inner * 4) + i2_inner)], 0.000000e+00f);
      compute[(((((((((((int)blockIdx.x) / 14) * 100352) + ((((int)threadIdx.x) / 14) * 3136)) + (i1_inner * 784)) + (((((int)blockIdx.x) % 14) >> 1) * 112)) + (i2_inner * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((int)threadIdx.x) % 14)) + 50176)] = max(conv2d_nchw[(((i1_inner * 4) + i2_inner) + 16)], 0.000000e+00f);
    }
  }
}


