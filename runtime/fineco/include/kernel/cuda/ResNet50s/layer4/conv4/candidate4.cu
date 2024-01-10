
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
extern "C" __global__ void __launch_bounds__(256) candidate4(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[4];
  __shared__ float pad_temp_shared[64];
  __shared__ float kernel_shared[4096];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 64; ++rc_outer_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 64) {
      pad_temp_shared[((int)threadIdx.x)] = data[((((((rc_outer_outer * 3136) + ((((int)threadIdx.x) >> 2) * 196)) + ((((int)blockIdx.x) / 7) * 28)) + (((((int)threadIdx.x) & 3) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1))];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 16; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
      kernel_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 256) + ((int)threadIdx.x))] = kernel[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 16384) + ((((int)threadIdx.x) >> 4) * 1024)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15))];
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((rc_inner * 4) + ((((int)threadIdx.x) & 1) * 2))] * kernel_shared[(((((int)threadIdx.x) >> 1) * 16) + rc_inner)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_inner * 4) + ((((int)threadIdx.x) & 1) * 2)) + 1)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 16) + rc_inner)]));
      conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((rc_inner * 4) + ((((int)threadIdx.x) & 1) * 2))] * kernel_shared[((((((int)threadIdx.x) >> 1) * 16) + rc_inner) + 2048)]));
      conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((rc_inner * 4) + ((((int)threadIdx.x) & 1) * 2)) + 1)] * kernel_shared[((((((int)threadIdx.x) >> 1) * 16) + rc_inner) + 2048)]));
    }
  }
  conv2d_nchw[(((((((int)threadIdx.x) >> 1) * 196) + ((((int)blockIdx.x) / 7) * 28)) + ((((int)threadIdx.x) & 1) * 14)) + ((((int)blockIdx.x) % 7) * 2))] = conv2d_nchw_local[0];
  conv2d_nchw[((((((((int)threadIdx.x) >> 1) * 196) + ((((int)blockIdx.x) / 7) * 28)) + ((((int)threadIdx.x) & 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + 1)] = conv2d_nchw_local[1];
  conv2d_nchw[((((((((int)threadIdx.x) >> 1) * 196) + ((((int)blockIdx.x) / 7) * 28)) + ((((int)threadIdx.x) & 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + 25088)] = conv2d_nchw_local[2];
  conv2d_nchw[((((((((int)threadIdx.x) >> 1) * 196) + ((((int)blockIdx.x) / 7) * 28)) + ((((int)threadIdx.x) & 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + 25089)] = conv2d_nchw_local[3];
}


