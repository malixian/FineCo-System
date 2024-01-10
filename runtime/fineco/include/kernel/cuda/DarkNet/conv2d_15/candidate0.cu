
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
extern "C" __global__ void __launch_bounds__(196) candidate0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ compute, float* __restrict__ bias) {
  float conv2d_nchw[2];
  __shared__ float pad_temp_shared[6272];
  __shared__ float kernel_shared[1024];
  for (int ff_outer_inner_init = 0; ff_outer_inner_init < 2; ++ff_outer_inner_init) {
    conv2d_nchw[ff_outer_inner_init] = 0.000000e+00f;
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 8; ++rc_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
      *(float4*)(pad_temp_shared + ((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 784) + (((int)threadIdx.x) * 4))) = *(float4*)(data + (((rc_outer_outer * 6272) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 784)) + (((int)threadIdx.x) * 4)));
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1) {
      if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 49) + (((int)threadIdx.x) >> 2)) < 128) {
        for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_s = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_s < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_s) {
          kernel_shared[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 392) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_s)] = kernel[((((((int)blockIdx.x) * 8192) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 49) + (((int)threadIdx.x) >> 2)) >> 4) * 1024)) + (rc_outer_outer * 128)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 392) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_s) & 127))];
        }
      }
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 2; ++rc_outer_inner) {
      for (int ff_outer_inner = 0; ff_outer_inner < 2; ++ff_outer_inner) {
        for (int rc_inner = 0; rc_inner < 64; ++rc_inner) {
          conv2d_nchw[ff_outer_inner] = (conv2d_nchw[ff_outer_inner] + (pad_temp_shared[(((rc_outer_inner * 3136) + (rc_inner * 49)) + (((int)threadIdx.x) % 49))] * kernel_shared[(((((((int)threadIdx.x) / 49) * 256) + (ff_outer_inner * 128)) + (rc_outer_inner * 64)) + rc_inner)]));
        }
      }
    }
  }
  for (int i1_inner = 0; i1_inner < 2; ++i1_inner) {
    compute[((((((int)blockIdx.x) * 392) + ((((int)threadIdx.x) / 49) * 98)) + (i1_inner * 49)) + (((int)threadIdx.x) % 49))] = max((conv2d_nchw[i1_inner] + bias[(((((int)blockIdx.x) * 8) + ((((int)threadIdx.x) / 49) * 2)) + i1_inner)]), 0.000000e+00f);
  }
}


