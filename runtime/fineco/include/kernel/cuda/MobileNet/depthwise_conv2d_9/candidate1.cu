
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
extern "C" __global__ void __launch_bounds__(448) candidate1(float* __restrict__ Input, float* __restrict__ kernel, float* __restrict__ compute) {
  float DepthwiseConv2d[7];
  __shared__ float PaddedInput_shared[4032];
  __shared__ float kernel_shared[192];
  for (int i_outer_inner_init = 0; i_outer_inner_init < 7; ++i_outer_inner_init) {
    DepthwiseConv2d[i_outer_inner_init] = 0.000000e+00f;
  }
  for (int dj_outer_outer = 0; dj_outer_outer < 3; ++dj_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 9; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
      PaddedInput_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 448) + ((int)threadIdx.x))] = (((((1 <= (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 64) + (((int)threadIdx.x) / 7)) % 9)) && ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 64) + (((int)threadIdx.x) / 7)) % 9) < 8)) && (1 <= (dj_outer_outer + (((int)threadIdx.x) % 7)))) && ((dj_outer_outer + (((int)threadIdx.x) % 7)) < 8)) ? Input[((((((((int)blockIdx.x) * 3136) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 64) + (((int)threadIdx.x) / 7)) / 9) * 49)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 64) + (((int)threadIdx.x) / 7)) % 9) * 7)) + dj_outer_outer) + (((int)threadIdx.x) % 7)) - 8)] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 192) {
      kernel_shared[((int)threadIdx.x)] = kernel[(((((int)blockIdx.x) * 576) + (((int)threadIdx.x) * 3)) + dj_outer_outer)];
    }
    __syncthreads();
    for (int i_outer_inner = 0; i_outer_inner < 7; ++i_outer_inner) {
      for (int di_inner = 0; di_inner < 3; ++di_inner) {
        DepthwiseConv2d[i_outer_inner] = (DepthwiseConv2d[i_outer_inner] + (PaddedInput_shared[(((((((int)threadIdx.x) / 7) * 63) + (i_outer_inner * 7)) + (di_inner * 7)) + (((int)threadIdx.x) % 7))] * kernel_shared[(((((int)threadIdx.x) / 7) * 3) + di_inner)]));
      }
    }
  }
  for (int i2_inner = 0; i2_inner < 7; ++i2_inner) {
    compute[((((((int)blockIdx.x) * 3136) + ((((int)threadIdx.x) / 7) * 49)) + (i2_inner * 7)) + (((int)threadIdx.x) % 7))] = max(DepthwiseConv2d[i2_inner], 0.000000e+00f);
  }
}


