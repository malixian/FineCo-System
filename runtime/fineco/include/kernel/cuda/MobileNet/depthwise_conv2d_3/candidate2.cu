
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
extern "C" __global__ void __launch_bounds__(112) candidate2(float* __restrict__ Input, float* __restrict__ kernel, float* __restrict__ compute) {
  float DepthwiseConv2d[64];
  __shared__ float PaddedInput_shared[8960];
  __shared__ float kernel_shared[48];
  DepthwiseConv2d[0] = 0.000000e+00f;
  DepthwiseConv2d[16] = 0.000000e+00f;
  DepthwiseConv2d[32] = 0.000000e+00f;
  DepthwiseConv2d[48] = 0.000000e+00f;
  DepthwiseConv2d[1] = 0.000000e+00f;
  DepthwiseConv2d[17] = 0.000000e+00f;
  DepthwiseConv2d[33] = 0.000000e+00f;
  DepthwiseConv2d[49] = 0.000000e+00f;
  DepthwiseConv2d[2] = 0.000000e+00f;
  DepthwiseConv2d[18] = 0.000000e+00f;
  DepthwiseConv2d[34] = 0.000000e+00f;
  DepthwiseConv2d[50] = 0.000000e+00f;
  DepthwiseConv2d[3] = 0.000000e+00f;
  DepthwiseConv2d[19] = 0.000000e+00f;
  DepthwiseConv2d[35] = 0.000000e+00f;
  DepthwiseConv2d[51] = 0.000000e+00f;
  DepthwiseConv2d[4] = 0.000000e+00f;
  DepthwiseConv2d[20] = 0.000000e+00f;
  DepthwiseConv2d[36] = 0.000000e+00f;
  DepthwiseConv2d[52] = 0.000000e+00f;
  DepthwiseConv2d[5] = 0.000000e+00f;
  DepthwiseConv2d[21] = 0.000000e+00f;
  DepthwiseConv2d[37] = 0.000000e+00f;
  DepthwiseConv2d[53] = 0.000000e+00f;
  DepthwiseConv2d[6] = 0.000000e+00f;
  DepthwiseConv2d[22] = 0.000000e+00f;
  DepthwiseConv2d[38] = 0.000000e+00f;
  DepthwiseConv2d[54] = 0.000000e+00f;
  DepthwiseConv2d[7] = 0.000000e+00f;
  DepthwiseConv2d[23] = 0.000000e+00f;
  DepthwiseConv2d[39] = 0.000000e+00f;
  DepthwiseConv2d[55] = 0.000000e+00f;
  DepthwiseConv2d[8] = 0.000000e+00f;
  DepthwiseConv2d[24] = 0.000000e+00f;
  DepthwiseConv2d[40] = 0.000000e+00f;
  DepthwiseConv2d[56] = 0.000000e+00f;
  DepthwiseConv2d[9] = 0.000000e+00f;
  DepthwiseConv2d[25] = 0.000000e+00f;
  DepthwiseConv2d[41] = 0.000000e+00f;
  DepthwiseConv2d[57] = 0.000000e+00f;
  DepthwiseConv2d[10] = 0.000000e+00f;
  DepthwiseConv2d[26] = 0.000000e+00f;
  DepthwiseConv2d[42] = 0.000000e+00f;
  DepthwiseConv2d[58] = 0.000000e+00f;
  DepthwiseConv2d[11] = 0.000000e+00f;
  DepthwiseConv2d[27] = 0.000000e+00f;
  DepthwiseConv2d[43] = 0.000000e+00f;
  DepthwiseConv2d[59] = 0.000000e+00f;
  DepthwiseConv2d[12] = 0.000000e+00f;
  DepthwiseConv2d[28] = 0.000000e+00f;
  DepthwiseConv2d[44] = 0.000000e+00f;
  DepthwiseConv2d[60] = 0.000000e+00f;
  DepthwiseConv2d[13] = 0.000000e+00f;
  DepthwiseConv2d[29] = 0.000000e+00f;
  DepthwiseConv2d[45] = 0.000000e+00f;
  DepthwiseConv2d[61] = 0.000000e+00f;
  DepthwiseConv2d[14] = 0.000000e+00f;
  DepthwiseConv2d[30] = 0.000000e+00f;
  DepthwiseConv2d[46] = 0.000000e+00f;
  DepthwiseConv2d[62] = 0.000000e+00f;
  DepthwiseConv2d[15] = 0.000000e+00f;
  DepthwiseConv2d[31] = 0.000000e+00f;
  DepthwiseConv2d[47] = 0.000000e+00f;
  DepthwiseConv2d[63] = 0.000000e+00f;
  for (int dj_outer_outer = 0; dj_outer_outer < 3; ++dj_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 80; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
      PaddedInput_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 112) + ((int)threadIdx.x))] = (((((1 <= ((((((int)blockIdx.x) % 7) * 8) + ((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer % 5) * 2)) + (((int)threadIdx.x) / 56))) && (((((((int)blockIdx.x) % 7) * 8) + ((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer % 5) * 2)) + (((int)threadIdx.x) / 56)) < 57)) && (1 <= (dj_outer_outer + (((int)threadIdx.x) % 56)))) && ((dj_outer_outer + (((int)threadIdx.x) % 56)) < 57)) ? Input[((((((((((int)blockIdx.x) / 7) * 50176) + ((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer / 5) * 3136)) + ((((int)blockIdx.x) % 7) * 448)) + ((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer % 5) * 112)) + ((int)threadIdx.x)) + dj_outer_outer) - 57)] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 48) {
      kernel_shared[((int)threadIdx.x)] = kernel[((((((int)blockIdx.x) / 7) * 144) + (((int)threadIdx.x) * 3)) + dj_outer_outer)];
    }
    __syncthreads();
    for (int c_outer_inner = 0; c_outer_inner < 4; ++c_outer_inner) {
      DepthwiseConv2d[(c_outer_inner * 4)] = (DepthwiseConv2d[(c_outer_inner * 4)] + (PaddedInput_shared[((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56))] * kernel_shared[(((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3))]));
      DepthwiseConv2d[((c_outer_inner * 4) + 16)] = (DepthwiseConv2d[((c_outer_inner * 4) + 16)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 224)] * kernel_shared[(((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3))]));
      DepthwiseConv2d[((c_outer_inner * 4) + 32)] = (DepthwiseConv2d[((c_outer_inner * 4) + 32)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 4480)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 24)]));
      DepthwiseConv2d[((c_outer_inner * 4) + 48)] = (DepthwiseConv2d[((c_outer_inner * 4) + 48)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 4704)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 24)]));
      DepthwiseConv2d[((c_outer_inner * 4) + 1)] = (DepthwiseConv2d[((c_outer_inner * 4) + 1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 56)] * kernel_shared[(((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3))]));
      DepthwiseConv2d[((c_outer_inner * 4) + 17)] = (DepthwiseConv2d[((c_outer_inner * 4) + 17)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 280)] * kernel_shared[(((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3))]));
      DepthwiseConv2d[((c_outer_inner * 4) + 33)] = (DepthwiseConv2d[((c_outer_inner * 4) + 33)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 4536)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 24)]));
      DepthwiseConv2d[((c_outer_inner * 4) + 49)] = (DepthwiseConv2d[((c_outer_inner * 4) + 49)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 4760)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 24)]));
      DepthwiseConv2d[(c_outer_inner * 4)] = (DepthwiseConv2d[(c_outer_inner * 4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 56)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 1)]));
      DepthwiseConv2d[((c_outer_inner * 4) + 16)] = (DepthwiseConv2d[((c_outer_inner * 4) + 16)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 280)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 1)]));
      DepthwiseConv2d[((c_outer_inner * 4) + 32)] = (DepthwiseConv2d[((c_outer_inner * 4) + 32)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 4536)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 25)]));
      DepthwiseConv2d[((c_outer_inner * 4) + 48)] = (DepthwiseConv2d[((c_outer_inner * 4) + 48)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 4760)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 25)]));
      DepthwiseConv2d[((c_outer_inner * 4) + 1)] = (DepthwiseConv2d[((c_outer_inner * 4) + 1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 112)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 1)]));
      DepthwiseConv2d[((c_outer_inner * 4) + 17)] = (DepthwiseConv2d[((c_outer_inner * 4) + 17)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 336)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 1)]));
      DepthwiseConv2d[((c_outer_inner * 4) + 33)] = (DepthwiseConv2d[((c_outer_inner * 4) + 33)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 4592)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 25)]));
      DepthwiseConv2d[((c_outer_inner * 4) + 49)] = (DepthwiseConv2d[((c_outer_inner * 4) + 49)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 4816)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 25)]));
      DepthwiseConv2d[(c_outer_inner * 4)] = (DepthwiseConv2d[(c_outer_inner * 4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 112)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 2)]));
      DepthwiseConv2d[((c_outer_inner * 4) + 16)] = (DepthwiseConv2d[((c_outer_inner * 4) + 16)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 336)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 2)]));
      DepthwiseConv2d[((c_outer_inner * 4) + 32)] = (DepthwiseConv2d[((c_outer_inner * 4) + 32)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 4592)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 26)]));
      DepthwiseConv2d[((c_outer_inner * 4) + 48)] = (DepthwiseConv2d[((c_outer_inner * 4) + 48)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 4816)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 26)]));
      DepthwiseConv2d[((c_outer_inner * 4) + 1)] = (DepthwiseConv2d[((c_outer_inner * 4) + 1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 168)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 2)]));
      DepthwiseConv2d[((c_outer_inner * 4) + 17)] = (DepthwiseConv2d[((c_outer_inner * 4) + 17)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 392)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 2)]));
      DepthwiseConv2d[((c_outer_inner * 4) + 33)] = (DepthwiseConv2d[((c_outer_inner * 4) + 33)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 4648)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 26)]));
      DepthwiseConv2d[((c_outer_inner * 4) + 49)] = (DepthwiseConv2d[((c_outer_inner * 4) + 49)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 4872)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 26)]));
      DepthwiseConv2d[((c_outer_inner * 4) + 2)] = (DepthwiseConv2d[((c_outer_inner * 4) + 2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 112)] * kernel_shared[(((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3))]));
      DepthwiseConv2d[((c_outer_inner * 4) + 18)] = (DepthwiseConv2d[((c_outer_inner * 4) + 18)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 336)] * kernel_shared[(((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3))]));
      DepthwiseConv2d[((c_outer_inner * 4) + 34)] = (DepthwiseConv2d[((c_outer_inner * 4) + 34)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 4592)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 24)]));
      DepthwiseConv2d[((c_outer_inner * 4) + 50)] = (DepthwiseConv2d[((c_outer_inner * 4) + 50)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 4816)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 24)]));
      DepthwiseConv2d[((c_outer_inner * 4) + 3)] = (DepthwiseConv2d[((c_outer_inner * 4) + 3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 168)] * kernel_shared[(((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3))]));
      DepthwiseConv2d[((c_outer_inner * 4) + 19)] = (DepthwiseConv2d[((c_outer_inner * 4) + 19)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 392)] * kernel_shared[(((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3))]));
      DepthwiseConv2d[((c_outer_inner * 4) + 35)] = (DepthwiseConv2d[((c_outer_inner * 4) + 35)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 4648)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 24)]));
      DepthwiseConv2d[((c_outer_inner * 4) + 51)] = (DepthwiseConv2d[((c_outer_inner * 4) + 51)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 4872)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 24)]));
      DepthwiseConv2d[((c_outer_inner * 4) + 2)] = (DepthwiseConv2d[((c_outer_inner * 4) + 2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 168)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 1)]));
      DepthwiseConv2d[((c_outer_inner * 4) + 18)] = (DepthwiseConv2d[((c_outer_inner * 4) + 18)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 392)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 1)]));
      DepthwiseConv2d[((c_outer_inner * 4) + 34)] = (DepthwiseConv2d[((c_outer_inner * 4) + 34)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 4648)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 25)]));
      DepthwiseConv2d[((c_outer_inner * 4) + 50)] = (DepthwiseConv2d[((c_outer_inner * 4) + 50)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 4872)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 25)]));
      DepthwiseConv2d[((c_outer_inner * 4) + 3)] = (DepthwiseConv2d[((c_outer_inner * 4) + 3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 224)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 1)]));
      DepthwiseConv2d[((c_outer_inner * 4) + 19)] = (DepthwiseConv2d[((c_outer_inner * 4) + 19)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 448)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 1)]));
      DepthwiseConv2d[((c_outer_inner * 4) + 35)] = (DepthwiseConv2d[((c_outer_inner * 4) + 35)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 4704)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 25)]));
      DepthwiseConv2d[((c_outer_inner * 4) + 51)] = (DepthwiseConv2d[((c_outer_inner * 4) + 51)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 4928)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 25)]));
      DepthwiseConv2d[((c_outer_inner * 4) + 2)] = (DepthwiseConv2d[((c_outer_inner * 4) + 2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 224)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 2)]));
      DepthwiseConv2d[((c_outer_inner * 4) + 18)] = (DepthwiseConv2d[((c_outer_inner * 4) + 18)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 448)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 2)]));
      DepthwiseConv2d[((c_outer_inner * 4) + 34)] = (DepthwiseConv2d[((c_outer_inner * 4) + 34)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 4704)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 26)]));
      DepthwiseConv2d[((c_outer_inner * 4) + 50)] = (DepthwiseConv2d[((c_outer_inner * 4) + 50)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 4928)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 26)]));
      DepthwiseConv2d[((c_outer_inner * 4) + 3)] = (DepthwiseConv2d[((c_outer_inner * 4) + 3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 280)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 2)]));
      DepthwiseConv2d[((c_outer_inner * 4) + 19)] = (DepthwiseConv2d[((c_outer_inner * 4) + 19)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 504)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 2)]));
      DepthwiseConv2d[((c_outer_inner * 4) + 35)] = (DepthwiseConv2d[((c_outer_inner * 4) + 35)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 4760)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 26)]));
      DepthwiseConv2d[((c_outer_inner * 4) + 51)] = (DepthwiseConv2d[((c_outer_inner * 4) + 51)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 2240) + (c_outer_inner * 560)) + (((int)threadIdx.x) % 56)) + 4984)] * kernel_shared[((((((int)threadIdx.x) / 56) * 12) + (c_outer_inner * 3)) + 26)]));
    }
  }
  for (int i1_inner = 0; i1_inner < 4; ++i1_inner) {
    for (int i2_inner = 0; i2_inner < 4; ++i2_inner) {
      compute[(((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 56) * 12544)) + (i1_inner * 3136)) + ((((int)blockIdx.x) % 7) * 448)) + (i2_inner * 56)) + (((int)threadIdx.x) % 56))] = max(DepthwiseConv2d[((i1_inner * 4) + i2_inner)], 0.000000e+00f);
      compute[((((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 56) * 12544)) + (i1_inner * 3136)) + ((((int)blockIdx.x) % 7) * 448)) + (i2_inner * 56)) + (((int)threadIdx.x) % 56)) + 224)] = max(DepthwiseConv2d[(((i1_inner * 4) + i2_inner) + 16)], 0.000000e+00f);
      compute[((((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 56) * 12544)) + (i1_inner * 3136)) + ((((int)blockIdx.x) % 7) * 448)) + (i2_inner * 56)) + (((int)threadIdx.x) % 56)) + 25088)] = max(DepthwiseConv2d[(((i1_inner * 4) + i2_inner) + 32)], 0.000000e+00f);
      compute[((((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 56) * 12544)) + (i1_inner * 3136)) + ((((int)blockIdx.x) % 7) * 448)) + (i2_inner * 56)) + (((int)threadIdx.x) % 56)) + 25312)] = max(DepthwiseConv2d[(((i1_inner * 4) + i2_inner) + 48)], 0.000000e+00f);
    }
  }
}


