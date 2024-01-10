
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
extern "C" __global__ void __launch_bounds__(256) candidate2(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ compute, float* __restrict__ bias) {
  float conv2d_nchw[14];
  __shared__ float pad_temp_shared[128];
  __shared__ float kernel_shared[2304];
  for (int ff_outer_inner_init = 0; ff_outer_inner_init < 2; ++ff_outer_inner_init) {
    for (int xx_outer_inner_init = 0; xx_outer_inner_init < 7; ++xx_outer_inner_init) {
      conv2d_nchw[((ff_outer_inner_init * 7) + xx_outer_inner_init)] = 0.000000e+00f;
    }
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 128; ++rc_outer_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 128) {
      pad_temp_shared[((int)threadIdx.x)] = (((((1 <= (((((int)blockIdx.x) % 7) * 2) + ((((int)threadIdx.x) & 63) >> 4))) && ((((((int)blockIdx.x) % 7) * 2) + ((((int)threadIdx.x) & 63) >> 4)) < 15)) && (1 <= (((int)threadIdx.x) & 15))) && ((((int)threadIdx.x) & 15) < 15)) ? data[((((((rc_outer_outer * 392) + ((((int)threadIdx.x) >> 6) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((((int)threadIdx.x) & 63) >> 4) * 14)) + (((int)threadIdx.x) & 15)) - 15)] : 0.000000e+00f);
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 9; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
      kernel_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 256) + ((int)threadIdx.x))] = kernel[(((((((int)blockIdx.x) / 7) * 294912) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 128) + (((int)threadIdx.x) >> 1)) / 9) * 2304)) + (rc_outer_outer * 18)) + (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 256) + ((int)threadIdx.x)) % 18))];
    }
    __syncthreads();
    for (int ff_outer_inner = 0; ff_outer_inner < 2; ++ff_outer_inner) {
      for (int xx_outer_inner = 0; xx_outer_inner < 7; ++xx_outer_inner) {
        for (int rc_inner = 0; rc_inner < 2; ++rc_inner) {
          for (int ry_inner = 0; ry_inner < 3; ++ry_inner) {
            for (int rx_inner = 0; rx_inner < 3; ++rx_inner) {
              conv2d_nchw[((ff_outer_inner * 7) + xx_outer_inner)] = (conv2d_nchw[((ff_outer_inner * 7) + xx_outer_inner)] + (pad_temp_shared[((((((rc_inner * 64) + (((((int)threadIdx.x) & 3) >> 1) * 16)) + (ry_inner * 16)) + ((((int)threadIdx.x) & 1) * 7)) + xx_outer_inner) + rx_inner)] * kernel_shared[((((((((int)threadIdx.x) >> 2) * 36) + (ff_outer_inner * 18)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
            }
          }
        }
      }
    }
  }
  for (int i1_inner = 0; i1_inner < 2; ++i1_inner) {
    for (int i3_inner = 0; i3_inner < 7; ++i3_inner) {
      compute[(((((((((int)blockIdx.x) / 7) * 25088) + ((((int)threadIdx.x) >> 2) * 392)) + (i1_inner * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) & 3) * 7)) + i3_inner)] = max((conv2d_nchw[((i1_inner * 7) + i3_inner)] + bias[((((((int)blockIdx.x) / 7) * 128) + ((((int)threadIdx.x) >> 2) * 2)) + i1_inner)]), 0.000000e+00f);
    }
  }
}


