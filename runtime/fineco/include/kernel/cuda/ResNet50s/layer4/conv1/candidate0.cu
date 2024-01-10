
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
extern "C" __global__ void __launch_bounds__(256) candidate0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[8];
  __shared__ float pad_temp_shared[1024];
  __shared__ float kernel_shared[8192];
  for (int ff_c_outer_inner_init = 0; ff_c_outer_inner_init < 2; ++ff_c_outer_inner_init) {
    for (int ff_c_inner_init = 0; ff_c_inner_init < 2; ++ff_c_inner_init) {
      conv2d_nchw_local[((ff_c_outer_inner_init * 2) + ff_c_inner_init)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_outer_inner_init * 2) + ff_c_inner_init) + 4)] = 0.000000e+00f;
    }
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 8; ++rc_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
      *(float2*)(pad_temp_shared + ((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 512) + (((int)threadIdx.x) * 2))) = *(float2*)(data + (((((((rc_outer_outer * 50176) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 25088)) + ((((int)threadIdx.x) >> 3) * 784)) + (((((int)blockIdx.x) % 49) / 7) * 112)) + (((((int)threadIdx.x) & 7) >> 1) * 28)) + ((((int)blockIdx.x) % 7) * 4)) + ((((int)threadIdx.x) & 1) * 2)));
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1) {
      *(float4*)(kernel_shared + ((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 1024) + (((int)threadIdx.x) * 4))) = *(float4*)(kernel + ((((((((int)blockIdx.x) / 49) * 65536) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 8192)) + ((((int)threadIdx.x) >> 4) * 512)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)));
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 16; ++rc_outer_inner) {
      for (int ff_c_outer_inner = 0; ff_c_outer_inner < 2; ++ff_c_outer_inner) {
        for (int rc_inner = 0; rc_inner < 4; ++rc_inner) {
          for (int ff_c_inner = 0; ff_c_inner < 2; ++ff_c_inner) {
            conv2d_nchw_local[((ff_c_outer_inner * 2) + ff_c_inner)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + ff_c_inner)] + (pad_temp_shared[(((rc_outer_inner * 64) + (rc_inner * 16)) + (((int)threadIdx.x) & 7))] * kernel_shared[((((((((int)threadIdx.x) >> 3) * 256) + (ff_c_outer_inner * 128)) + (ff_c_inner * 64)) + (rc_outer_inner * 4)) + rc_inner)]));
            conv2d_nchw_local[(((ff_c_outer_inner * 2) + ff_c_inner) + 4)] = (conv2d_nchw_local[(((ff_c_outer_inner * 2) + ff_c_inner) + 4)] + (pad_temp_shared[((((rc_outer_inner * 64) + (rc_inner * 16)) + (((int)threadIdx.x) & 7)) + 8)] * kernel_shared[((((((((int)threadIdx.x) >> 3) * 256) + (ff_c_outer_inner * 128)) + (ff_c_inner * 64)) + (rc_outer_inner * 4)) + rc_inner)]));
          }
        }
      }
    }
  }
  for (int ff_inner = 0; ff_inner < 4; ++ff_inner) {
    conv2d_nchw[((((((((((int)blockIdx.x) / 49) * 100352) + ((((int)threadIdx.x) >> 3) * 3136)) + (ff_inner * 784)) + (((((int)blockIdx.x) % 49) / 7) * 112)) + (((((int)threadIdx.x) & 7) >> 2) * 28)) + ((((int)blockIdx.x) % 7) * 4)) + (((int)threadIdx.x) & 3))] = conv2d_nchw_local[ff_inner];
    conv2d_nchw[(((((((((((int)blockIdx.x) / 49) * 100352) + ((((int)threadIdx.x) >> 3) * 3136)) + (ff_inner * 784)) + (((((int)blockIdx.x) % 49) / 7) * 112)) + (((((int)threadIdx.x) & 7) >> 2) * 28)) + ((((int)blockIdx.x) % 7) * 4)) + (((int)threadIdx.x) & 3)) + 56)] = conv2d_nchw_local[(ff_inner + 4)];
  }
}


