
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
extern "C" __global__ void __launch_bounds__(128) candidate6(float* __restrict__ Input, float* __restrict__ kernel, float* __restrict__ compute) {
  float conv2d_nchw[32];
  __shared__ float pad_temp_shared[1024];
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
  for (int rc_outer_outer = 0; rc_outer_outer < 2; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = Input[((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 5) * 3136)) + ((((int)blockIdx.x) / 7) * 224)) + (((((int)threadIdx.x) & 31) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7))];
    pad_temp_shared[(((int)threadIdx.x) + 128)] = Input[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 5) * 3136)) + ((((int)blockIdx.x) / 7) * 224)) + (((((int)threadIdx.x) & 31) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 12544)];
    pad_temp_shared[(((int)threadIdx.x) + 256)] = Input[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 5) * 3136)) + ((((int)blockIdx.x) / 7) * 224)) + (((((int)threadIdx.x) & 31) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 25088)];
    pad_temp_shared[(((int)threadIdx.x) + 384)] = Input[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 5) * 3136)) + ((((int)blockIdx.x) / 7) * 224)) + (((((int)threadIdx.x) & 31) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 37632)];
    pad_temp_shared[(((int)threadIdx.x) + 512)] = Input[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 5) * 3136)) + ((((int)blockIdx.x) / 7) * 224)) + (((((int)threadIdx.x) & 31) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 50176)];
    pad_temp_shared[(((int)threadIdx.x) + 640)] = Input[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 5) * 3136)) + ((((int)blockIdx.x) / 7) * 224)) + (((((int)threadIdx.x) & 31) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 62720)];
    pad_temp_shared[(((int)threadIdx.x) + 768)] = Input[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 5) * 3136)) + ((((int)blockIdx.x) / 7) * 224)) + (((((int)threadIdx.x) & 31) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 75264)];
    pad_temp_shared[(((int)threadIdx.x) + 896)] = Input[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 5) * 3136)) + ((((int)blockIdx.x) / 7) * 224)) + (((((int)threadIdx.x) & 31) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 87808)];
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 32; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
      kernel_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 128) + ((int)threadIdx.x))] = kernel[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31))];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 4; ++rc_outer_inner) {
      for (int yy_outer_inner = 0; yy_outer_inner < 4; ++yy_outer_inner) {
        for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
          conv2d_nchw[yy_outer_inner] = (conv2d_nchw[yy_outer_inner] + (pad_temp_shared[((((rc_outer_inner * 256) + (rc_inner * 32)) + (yy_outer_inner * 8)) + (((int)threadIdx.x) & 7))] * kernel_shared[((((((int)threadIdx.x) >> 3) * 128) + (rc_outer_inner * 8)) + rc_inner)]));
          conv2d_nchw[(yy_outer_inner + 16)] = (conv2d_nchw[(yy_outer_inner + 16)] + (pad_temp_shared[((((rc_outer_inner * 256) + (rc_inner * 32)) + (yy_outer_inner * 8)) + (((int)threadIdx.x) & 7))] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 128) + (rc_outer_inner * 8)) + rc_inner) + 2048)]));
          conv2d_nchw[(yy_outer_inner + 4)] = (conv2d_nchw[(yy_outer_inner + 4)] + (pad_temp_shared[((((rc_outer_inner * 256) + (rc_inner * 32)) + (yy_outer_inner * 8)) + (((int)threadIdx.x) & 7))] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 128) + (rc_outer_inner * 8)) + rc_inner) + 32)]));
          conv2d_nchw[(yy_outer_inner + 20)] = (conv2d_nchw[(yy_outer_inner + 20)] + (pad_temp_shared[((((rc_outer_inner * 256) + (rc_inner * 32)) + (yy_outer_inner * 8)) + (((int)threadIdx.x) & 7))] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 128) + (rc_outer_inner * 8)) + rc_inner) + 2080)]));
          conv2d_nchw[(yy_outer_inner + 8)] = (conv2d_nchw[(yy_outer_inner + 8)] + (pad_temp_shared[((((rc_outer_inner * 256) + (rc_inner * 32)) + (yy_outer_inner * 8)) + (((int)threadIdx.x) & 7))] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 128) + (rc_outer_inner * 8)) + rc_inner) + 64)]));
          conv2d_nchw[(yy_outer_inner + 24)] = (conv2d_nchw[(yy_outer_inner + 24)] + (pad_temp_shared[((((rc_outer_inner * 256) + (rc_inner * 32)) + (yy_outer_inner * 8)) + (((int)threadIdx.x) & 7))] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 128) + (rc_outer_inner * 8)) + rc_inner) + 2112)]));
          conv2d_nchw[(yy_outer_inner + 12)] = (conv2d_nchw[(yy_outer_inner + 12)] + (pad_temp_shared[((((rc_outer_inner * 256) + (rc_inner * 32)) + (yy_outer_inner * 8)) + (((int)threadIdx.x) & 7))] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 128) + (rc_outer_inner * 8)) + rc_inner) + 96)]));
          conv2d_nchw[(yy_outer_inner + 28)] = (conv2d_nchw[(yy_outer_inner + 28)] + (pad_temp_shared[((((rc_outer_inner * 256) + (rc_inner * 32)) + (yy_outer_inner * 8)) + (((int)threadIdx.x) & 7))] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 128) + (rc_outer_inner * 8)) + rc_inner) + 2144)]));
        }
      }
    }
  }
  for (int i1_inner = 0; i1_inner < 4; ++i1_inner) {
    for (int i2_inner = 0; i2_inner < 4; ++i2_inner) {
      compute[(((((((((int)threadIdx.x) >> 3) * 12544) + (i1_inner * 3136)) + ((((int)blockIdx.x) / 7) * 224)) + (i2_inner * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7))] = max(conv2d_nchw[((i1_inner * 4) + i2_inner)], 0.000000e+00f);
      compute[((((((((((int)threadIdx.x) >> 3) * 12544) + (i1_inner * 3136)) + ((((int)blockIdx.x) / 7) * 224)) + (i2_inner * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 200704)] = max(conv2d_nchw[(((i1_inner * 4) + i2_inner) + 16)], 0.000000e+00f);
    }
  }
}


