
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
extern "C" __global__ void __launch_bounds__(784) candidate1(float* __restrict__ Input, float* __restrict__ kernel, float* __restrict__ compute) {
  float DepthwiseConv2d[4];
  __shared__ float PaddedInput_shared[12064];
  __shared__ float kernel_shared[96];
  DepthwiseConv2d[0] = 0.000000e+00f;
  DepthwiseConv2d[1] = 0.000000e+00f;
  DepthwiseConv2d[2] = 0.000000e+00f;
  DepthwiseConv2d[3] = 0.000000e+00f;
  for (int di_outer_outer = 0; di_outer_outer < 3; ++di_outer_outer) {
    __syncthreads();
    PaddedInput_shared[((int)threadIdx.x)] = (((1 <= ((((((int)blockIdx.x) & 1) * 14) + ((((int)threadIdx.x) % 377) / 29)) + di_outer_outer)) && (1 <= (((int)threadIdx.x) % 29))) ? Input[((((((((((int)blockIdx.x) >> 1) * 25088) + ((((int)threadIdx.x) / 377) * 784)) + ((((int)blockIdx.x) & 1) * 392)) + (((((int)threadIdx.x) % 377) / 29) * 28)) + (di_outer_outer * 28)) + (((int)threadIdx.x) % 29)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 784)] = (((1 <= ((((((int)blockIdx.x) & 1) * 14) + (((((int)threadIdx.x) + 30) % 377) / 29)) + di_outer_outer)) && (1 <= ((((int)threadIdx.x) + 1) % 29))) ? Input[((((((((((int)blockIdx.x) >> 1) * 25088) + (((((int)threadIdx.x) + 784) / 377) * 784)) + ((((int)blockIdx.x) & 1) * 392)) + ((((((int)threadIdx.x) + 30) % 377) / 29) * 28)) + (di_outer_outer * 28)) + ((((int)threadIdx.x) + 1) % 29)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 1568)] = (((1 <= ((((((int)blockIdx.x) & 1) * 14) + (((((int)threadIdx.x) + 60) % 377) / 29)) + di_outer_outer)) && (1 <= ((((int)threadIdx.x) + 2) % 29))) ? Input[((((((((((int)blockIdx.x) >> 1) * 25088) + (((((int)threadIdx.x) + 1568) / 377) * 784)) + ((((int)blockIdx.x) & 1) * 392)) + ((((((int)threadIdx.x) + 60) % 377) / 29) * 28)) + (di_outer_outer * 28)) + ((((int)threadIdx.x) + 2) % 29)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 2352)] = (((1 <= ((((((int)blockIdx.x) & 1) * 14) + (((((int)threadIdx.x) + 90) % 377) / 29)) + di_outer_outer)) && (1 <= ((((int)threadIdx.x) + 3) % 29))) ? Input[((((((((((int)blockIdx.x) >> 1) * 25088) + (((((int)threadIdx.x) + 2352) / 377) * 784)) + ((((int)blockIdx.x) & 1) * 392)) + ((((((int)threadIdx.x) + 90) % 377) / 29) * 28)) + (di_outer_outer * 28)) + ((((int)threadIdx.x) + 3) % 29)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 3136)] = (((1 <= ((((((int)blockIdx.x) & 1) * 14) + (((((int)threadIdx.x) + 120) % 377) / 29)) + di_outer_outer)) && (1 <= ((((int)threadIdx.x) + 4) % 29))) ? Input[((((((((((int)blockIdx.x) >> 1) * 25088) + (((((int)threadIdx.x) + 3136) / 377) * 784)) + ((((int)blockIdx.x) & 1) * 392)) + ((((((int)threadIdx.x) + 120) % 377) / 29) * 28)) + (di_outer_outer * 28)) + ((((int)threadIdx.x) + 4) % 29)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 3920)] = (((1 <= ((((((int)blockIdx.x) & 1) * 14) + (((((int)threadIdx.x) + 150) % 377) / 29)) + di_outer_outer)) && (1 <= ((((int)threadIdx.x) + 5) % 29))) ? Input[((((((((((int)blockIdx.x) >> 1) * 25088) + (((((int)threadIdx.x) + 3920) / 377) * 784)) + ((((int)blockIdx.x) & 1) * 392)) + ((((((int)threadIdx.x) + 150) % 377) / 29) * 28)) + (di_outer_outer * 28)) + ((((int)threadIdx.x) + 5) % 29)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 4704)] = (((1 <= ((((((int)blockIdx.x) & 1) * 14) + (((((int)threadIdx.x) + 180) % 377) / 29)) + di_outer_outer)) && (1 <= ((((int)threadIdx.x) + 6) % 29))) ? Input[((((((((((int)blockIdx.x) >> 1) * 25088) + (((((int)threadIdx.x) + 4704) / 377) * 784)) + ((((int)blockIdx.x) & 1) * 392)) + ((((((int)threadIdx.x) + 180) % 377) / 29) * 28)) + (di_outer_outer * 28)) + ((((int)threadIdx.x) + 6) % 29)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 5488)] = (((1 <= ((((((int)blockIdx.x) & 1) * 14) + (((((int)threadIdx.x) + 210) % 377) / 29)) + di_outer_outer)) && (1 <= ((((int)threadIdx.x) + 7) % 29))) ? Input[((((((((((int)blockIdx.x) >> 1) * 25088) + (((((int)threadIdx.x) + 5488) / 377) * 784)) + ((((int)blockIdx.x) & 1) * 392)) + ((((((int)threadIdx.x) + 210) % 377) / 29) * 28)) + (di_outer_outer * 28)) + ((((int)threadIdx.x) + 7) % 29)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 6272)] = (((1 <= ((((((int)blockIdx.x) & 1) * 14) + (((((int)threadIdx.x) + 240) % 377) / 29)) + di_outer_outer)) && (1 <= ((((int)threadIdx.x) + 8) % 29))) ? Input[((((((((((int)blockIdx.x) >> 1) * 25088) + (((((int)threadIdx.x) + 6272) / 377) * 784)) + ((((int)blockIdx.x) & 1) * 392)) + ((((((int)threadIdx.x) + 240) % 377) / 29) * 28)) + (di_outer_outer * 28)) + ((((int)threadIdx.x) + 8) % 29)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 7056)] = (((1 <= ((((((int)blockIdx.x) & 1) * 14) + (((((int)threadIdx.x) + 270) % 377) / 29)) + di_outer_outer)) && (1 <= ((((int)threadIdx.x) + 9) % 29))) ? Input[((((((((((int)blockIdx.x) >> 1) * 25088) + (((((int)threadIdx.x) + 7056) / 377) * 784)) + ((((int)blockIdx.x) & 1) * 392)) + ((((((int)threadIdx.x) + 270) % 377) / 29) * 28)) + (di_outer_outer * 28)) + ((((int)threadIdx.x) + 9) % 29)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 7840)] = (((1 <= ((((((int)blockIdx.x) & 1) * 14) + (((((int)threadIdx.x) + 300) % 377) / 29)) + di_outer_outer)) && (1 <= ((((int)threadIdx.x) + 10) % 29))) ? Input[((((((((((int)blockIdx.x) >> 1) * 25088) + (((((int)threadIdx.x) + 7840) / 377) * 784)) + ((((int)blockIdx.x) & 1) * 392)) + ((((((int)threadIdx.x) + 300) % 377) / 29) * 28)) + (di_outer_outer * 28)) + ((((int)threadIdx.x) + 10) % 29)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 8624)] = (((1 <= ((((((int)blockIdx.x) & 1) * 14) + (((((int)threadIdx.x) + 330) % 377) / 29)) + di_outer_outer)) && (1 <= ((((int)threadIdx.x) + 11) % 29))) ? Input[((((((((((int)blockIdx.x) >> 1) * 25088) + (((((int)threadIdx.x) + 8624) / 377) * 784)) + ((((int)blockIdx.x) & 1) * 392)) + ((((((int)threadIdx.x) + 330) % 377) / 29) * 28)) + (di_outer_outer * 28)) + ((((int)threadIdx.x) + 11) % 29)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 9408)] = (((1 <= ((((((int)blockIdx.x) & 1) * 14) + (((((int)threadIdx.x) + 360) % 377) / 29)) + di_outer_outer)) && (1 <= ((((int)threadIdx.x) + 12) % 29))) ? Input[((((((((((int)blockIdx.x) >> 1) * 25088) + (((((int)threadIdx.x) + 9408) / 377) * 784)) + ((((int)blockIdx.x) & 1) * 392)) + ((((((int)threadIdx.x) + 360) % 377) / 29) * 28)) + (di_outer_outer * 28)) + ((((int)threadIdx.x) + 12) % 29)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 10192)] = (((1 <= ((((((int)blockIdx.x) & 1) * 14) + (((((int)threadIdx.x) + 13) % 377) / 29)) + di_outer_outer)) && (1 <= ((((int)threadIdx.x) + 13) % 29))) ? Input[((((((((((int)blockIdx.x) >> 1) * 25088) + (((((int)threadIdx.x) + 10192) / 377) * 784)) + ((((int)blockIdx.x) & 1) * 392)) + ((((((int)threadIdx.x) + 13) % 377) / 29) * 28)) + (di_outer_outer * 28)) + ((((int)threadIdx.x) + 13) % 29)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 10976)] = (((1 <= ((((((int)blockIdx.x) & 1) * 14) + (((((int)threadIdx.x) + 43) % 377) / 29)) + di_outer_outer)) && (1 <= ((((int)threadIdx.x) + 14) % 29))) ? Input[((((((((((int)blockIdx.x) >> 1) * 25088) + (((((int)threadIdx.x) + 10976) / 377) * 784)) + ((((int)blockIdx.x) & 1) * 392)) + ((((((int)threadIdx.x) + 43) % 377) / 29) * 28)) + (di_outer_outer * 28)) + ((((int)threadIdx.x) + 14) % 29)) - 29)] : 0.000000e+00f);
    if (((int)threadIdx.x) < 304) {
      PaddedInput_shared[(((int)threadIdx.x) + 11760)] = ((1 <= ((((int)threadIdx.x) + 15) % 29)) ? Input[((((((((((int)blockIdx.x) >> 1) * 25088) + (((((int)threadIdx.x) + 11760) / 377) * 784)) + ((((int)blockIdx.x) & 1) * 392)) + ((((((int)threadIdx.x) + 73) % 377) / 29) * 28)) + (di_outer_outer * 28)) + ((((int)threadIdx.x) + 15) % 29)) - 29)] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 96) {
      kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) >> 1) * 288) + ((((int)threadIdx.x) / 3) * 9)) + (di_outer_outer * 3)) + (((int)threadIdx.x) % 3))];
    }
    __syncthreads();
    DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 98) * 1508) + (((((int)threadIdx.x) % 98) / 14) * 58)) + ((((int)threadIdx.x) % 14) * 2))] * kernel_shared[((((int)threadIdx.x) / 98) * 12)]));
    DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[(((((((int)threadIdx.x) / 98) * 1508) + (((((int)threadIdx.x) % 98) / 14) * 58)) + ((((int)threadIdx.x) % 14) * 2)) + 377)] * kernel_shared[(((((int)threadIdx.x) / 98) * 12) + 3)]));
    DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[(((((((int)threadIdx.x) / 98) * 1508) + (((((int)threadIdx.x) % 98) / 14) * 58)) + ((((int)threadIdx.x) % 14) * 2)) + 754)] * kernel_shared[(((((int)threadIdx.x) / 98) * 12) + 6)]));
    DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[(((((((int)threadIdx.x) / 98) * 1508) + (((((int)threadIdx.x) % 98) / 14) * 58)) + ((((int)threadIdx.x) % 14) * 2)) + 1131)] * kernel_shared[(((((int)threadIdx.x) / 98) * 12) + 9)]));
    DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((((int)threadIdx.x) / 98) * 1508) + (((((int)threadIdx.x) % 98) / 14) * 58)) + ((((int)threadIdx.x) % 14) * 2)) + 1)] * kernel_shared[(((((int)threadIdx.x) / 98) * 12) + 1)]));
    DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[(((((((int)threadIdx.x) / 98) * 1508) + (((((int)threadIdx.x) % 98) / 14) * 58)) + ((((int)threadIdx.x) % 14) * 2)) + 378)] * kernel_shared[(((((int)threadIdx.x) / 98) * 12) + 4)]));
    DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[(((((((int)threadIdx.x) / 98) * 1508) + (((((int)threadIdx.x) % 98) / 14) * 58)) + ((((int)threadIdx.x) % 14) * 2)) + 755)] * kernel_shared[(((((int)threadIdx.x) / 98) * 12) + 7)]));
    DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[(((((((int)threadIdx.x) / 98) * 1508) + (((((int)threadIdx.x) % 98) / 14) * 58)) + ((((int)threadIdx.x) % 14) * 2)) + 1132)] * kernel_shared[(((((int)threadIdx.x) / 98) * 12) + 10)]));
    DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((((int)threadIdx.x) / 98) * 1508) + (((((int)threadIdx.x) % 98) / 14) * 58)) + ((((int)threadIdx.x) % 14) * 2)) + 2)] * kernel_shared[(((((int)threadIdx.x) / 98) * 12) + 2)]));
    DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[(((((((int)threadIdx.x) / 98) * 1508) + (((((int)threadIdx.x) % 98) / 14) * 58)) + ((((int)threadIdx.x) % 14) * 2)) + 379)] * kernel_shared[(((((int)threadIdx.x) / 98) * 12) + 5)]));
    DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[(((((((int)threadIdx.x) / 98) * 1508) + (((((int)threadIdx.x) % 98) / 14) * 58)) + ((((int)threadIdx.x) % 14) * 2)) + 756)] * kernel_shared[(((((int)threadIdx.x) / 98) * 12) + 8)]));
    DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[(((((((int)threadIdx.x) / 98) * 1508) + (((((int)threadIdx.x) % 98) / 14) * 58)) + ((((int)threadIdx.x) % 14) * 2)) + 1133)] * kernel_shared[(((((int)threadIdx.x) / 98) * 12) + 11)]));
  }
  for (int i1_inner = 0; i1_inner < 4; ++i1_inner) {
    compute[((((((((int)blockIdx.x) >> 1) * 6272) + ((((int)threadIdx.x) / 98) * 784)) + (i1_inner * 196)) + ((((int)blockIdx.x) & 1) * 98)) + (((int)threadIdx.x) % 98))] = max(DepthwiseConv2d[i1_inner], 0.000000e+00f);
  }
}


