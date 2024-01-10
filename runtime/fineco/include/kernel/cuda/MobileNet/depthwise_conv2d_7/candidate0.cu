
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
extern "C" __global__ void __launch_bounds__(784) candidate0(float* __restrict__ Input, float* __restrict__ kernel, float* __restrict__ compute) {
  float DepthwiseConv2d[4];
  __shared__ float PaddedInput_shared[3136];
  __shared__ float kernel_shared[16];
  for (int c_outer_inner_init = 0; c_outer_inner_init < 2; ++c_outer_inner_init) {
    DepthwiseConv2d[c_outer_inner_init] = 0.000000e+00f;
    DepthwiseConv2d[(c_outer_inner_init + 2)] = 0.000000e+00f;
  }
  for (int di_outer_outer = 0; di_outer_outer < 3; ++di_outer_outer) {
    for (int dj_outer_outer = 0; dj_outer_outer < 3; ++dj_outer_outer) {
      __syncthreads();
      for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
        PaddedInput_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 784) + ((int)threadIdx.x))] = (((((1 <= (((((int)threadIdx.x) % 196) / 14) + di_outer_outer)) && ((((((int)threadIdx.x) % 196) / 14) + di_outer_outer) < 15)) && (1 <= (dj_outer_outer + (((int)threadIdx.x) % 14)))) && ((dj_outer_outer + (((int)threadIdx.x) % 14)) < 15)) ? Input[((((((((int)blockIdx.x) * 3136) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 784)) + (di_outer_outer * 14)) + ((int)threadIdx.x)) + dj_outer_outer) - 15)] : 0.000000e+00f);
      }
      if (((int)threadIdx.x) < 16) {
        kernel_shared[((int)threadIdx.x)] = kernel[((((((int)blockIdx.x) * 144) + (((int)threadIdx.x) * 9)) + (di_outer_outer * 3)) + dj_outer_outer)];
      }
      __syncthreads();
      for (int c_outer_inner = 0; c_outer_inner < 2; ++c_outer_inner) {
        DepthwiseConv2d[c_outer_inner] = (DepthwiseConv2d[c_outer_inner] + (PaddedInput_shared[((((((int)threadIdx.x) / 196) * 392) + (c_outer_inner * 196)) + (((int)threadIdx.x) % 196))] * kernel_shared[(((((int)threadIdx.x) / 196) * 2) + c_outer_inner)]));
        DepthwiseConv2d[(c_outer_inner + 2)] = (DepthwiseConv2d[(c_outer_inner + 2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 196) * 392) + (c_outer_inner * 196)) + (((int)threadIdx.x) % 196)) + 1568)] * kernel_shared[((((((int)threadIdx.x) / 196) * 2) + c_outer_inner) + 8)]));
      }
    }
  }
  for (int i1_inner = 0; i1_inner < 2; ++i1_inner) {
    compute[((((((int)blockIdx.x) * 3136) + ((((int)threadIdx.x) / 196) * 392)) + (i1_inner * 196)) + (((int)threadIdx.x) % 196))] = max(DepthwiseConv2d[i1_inner], 0.000000e+00f);
    compute[(((((((int)blockIdx.x) * 3136) + ((((int)threadIdx.x) / 196) * 392)) + (i1_inner * 196)) + (((int)threadIdx.x) % 196)) + 1568)] = max(DepthwiseConv2d[(i1_inner + 2)], 0.000000e+00f);
  }
}


