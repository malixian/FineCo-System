
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
extern "C" __global__ void __launch_bounds__(392) candidate4(float* __restrict__ Input, float* __restrict__ kernel, float* __restrict__ compute) {
  float DepthwiseConv2d[4];
  __shared__ float PaddedInput_shared[6498];
  __shared__ float kernel_shared[18];
  DepthwiseConv2d[0] = 0.000000e+00f;
  DepthwiseConv2d[1] = 0.000000e+00f;
  DepthwiseConv2d[2] = 0.000000e+00f;
  DepthwiseConv2d[3] = 0.000000e+00f;
  PaddedInput_shared[((int)threadIdx.x)] = (((57 <= ((int)threadIdx.x)) && (1 <= (((int)threadIdx.x) % 57))) ? Input[((((((int)blockIdx.x) * 6272) + ((((int)threadIdx.x) / 57) * 56)) + (((int)threadIdx.x) % 57)) - 57)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 392)] = ((1 <= ((((int)threadIdx.x) + 50) % 57)) ? Input[((((((int)blockIdx.x) * 6272) + (((((int)threadIdx.x) + 392) / 57) * 56)) + ((((int)threadIdx.x) + 50) % 57)) - 57)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 784)] = ((1 <= ((((int)threadIdx.x) + 43) % 57)) ? Input[((((((int)blockIdx.x) * 6272) + (((((int)threadIdx.x) + 784) / 57) * 56)) + ((((int)threadIdx.x) + 43) % 57)) - 57)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 1176)] = ((1 <= ((((int)threadIdx.x) + 36) % 57)) ? Input[((((((int)blockIdx.x) * 6272) + (((((int)threadIdx.x) + 1176) / 57) * 56)) + ((((int)threadIdx.x) + 36) % 57)) - 57)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 1568)] = ((1 <= ((((int)threadIdx.x) + 29) % 57)) ? Input[((((((int)blockIdx.x) * 6272) + (((((int)threadIdx.x) + 1568) / 57) * 56)) + ((((int)threadIdx.x) + 29) % 57)) - 57)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 1960)] = ((1 <= ((((int)threadIdx.x) + 22) % 57)) ? Input[((((((int)blockIdx.x) * 6272) + (((((int)threadIdx.x) + 1960) / 57) * 56)) + ((((int)threadIdx.x) + 22) % 57)) - 57)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 2352)] = ((1 <= ((((int)threadIdx.x) + 15) % 57)) ? Input[((((((int)blockIdx.x) * 6272) + (((((int)threadIdx.x) + 2352) / 57) * 56)) + ((((int)threadIdx.x) + 15) % 57)) - 57)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 2744)] = ((1 <= ((((int)threadIdx.x) + 8) % 57)) ? Input[((((((int)blockIdx.x) * 6272) + (((((int)threadIdx.x) + 2744) / 57) * 56)) + ((((int)threadIdx.x) + 8) % 57)) - 57)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 3136)] = (((57 <= ((((int)threadIdx.x) + 3136) % 3249)) && (1 <= ((((int)threadIdx.x) + 1) % 57))) ? Input[(((((((int)blockIdx.x) * 6272) + (((((int)threadIdx.x) + 3136) / 3249) * 3136)) + ((((((int)threadIdx.x) + 3136) % 3249) / 57) * 56)) + ((((int)threadIdx.x) + 1) % 57)) - 57)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 3528)] = ((1 <= ((((int)threadIdx.x) + 51) % 57)) ? Input[(((((((int)blockIdx.x) * 6272) + (((((int)threadIdx.x) + 3528) / 3249) * 3136)) + ((((((int)threadIdx.x) + 279) % 3249) / 57) * 56)) + ((((int)threadIdx.x) + 51) % 57)) - 57)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 3920)] = ((1 <= ((((int)threadIdx.x) + 44) % 57)) ? Input[(((((((int)blockIdx.x) * 6272) + (((((int)threadIdx.x) + 3920) / 3249) * 3136)) + ((((((int)threadIdx.x) + 671) % 3249) / 57) * 56)) + ((((int)threadIdx.x) + 44) % 57)) - 57)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 4312)] = ((1 <= ((((int)threadIdx.x) + 37) % 57)) ? Input[(((((((int)blockIdx.x) * 6272) + (((((int)threadIdx.x) + 4312) / 3249) * 3136)) + ((((((int)threadIdx.x) + 1063) % 3249) / 57) * 56)) + ((((int)threadIdx.x) + 37) % 57)) - 57)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 4704)] = ((1 <= ((((int)threadIdx.x) + 30) % 57)) ? Input[(((((((int)blockIdx.x) * 6272) + (((((int)threadIdx.x) + 4704) / 3249) * 3136)) + ((((((int)threadIdx.x) + 1455) % 3249) / 57) * 56)) + ((((int)threadIdx.x) + 30) % 57)) - 57)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 5096)] = ((1 <= ((((int)threadIdx.x) + 23) % 57)) ? Input[(((((((int)blockIdx.x) * 6272) + (((((int)threadIdx.x) + 5096) / 3249) * 3136)) + ((((((int)threadIdx.x) + 1847) % 3249) / 57) * 56)) + ((((int)threadIdx.x) + 23) % 57)) - 57)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 5488)] = ((1 <= ((((int)threadIdx.x) + 16) % 57)) ? Input[(((((((int)blockIdx.x) * 6272) + (((((int)threadIdx.x) + 5488) / 3249) * 3136)) + ((((((int)threadIdx.x) + 2239) % 3249) / 57) * 56)) + ((((int)threadIdx.x) + 16) % 57)) - 57)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 5880)] = ((1 <= ((((int)threadIdx.x) + 9) % 57)) ? Input[(((((((int)blockIdx.x) * 6272) + (((((int)threadIdx.x) + 5880) / 3249) * 3136)) + ((((((int)threadIdx.x) + 2631) % 3249) / 57) * 56)) + ((((int)threadIdx.x) + 9) % 57)) - 57)] : 0.000000e+00f);
  if (((int)threadIdx.x) < 226) {
    PaddedInput_shared[(((int)threadIdx.x) + 6272)] = ((1 <= ((((int)threadIdx.x) + 2) % 57)) ? Input[(((((((int)blockIdx.x) * 6272) + (((((int)threadIdx.x) + 6272) / 3249) * 3136)) + ((((((int)threadIdx.x) + 3023) % 3249) / 57) * 56)) + ((((int)threadIdx.x) + 2) % 57)) - 57)] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 18) {
    kernel_shared[((int)threadIdx.x)] = kernel[((((int)blockIdx.x) * 18) + ((int)threadIdx.x))];
  }
  __syncthreads();
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 196) * 3249) + (((((int)threadIdx.x) % 196) / 7) * 114)) + ((((int)threadIdx.x) % 7) * 8))] * kernel_shared[((((int)threadIdx.x) / 196) * 9)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((((int)threadIdx.x) / 196) * 3249) + (((((int)threadIdx.x) % 196) / 7) * 114)) + ((((int)threadIdx.x) % 7) * 8)) + 1)] * kernel_shared[(((((int)threadIdx.x) / 196) * 9) + 1)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((((int)threadIdx.x) / 196) * 3249) + (((((int)threadIdx.x) % 196) / 7) * 114)) + ((((int)threadIdx.x) % 7) * 8)) + 2)] * kernel_shared[(((((int)threadIdx.x) / 196) * 9) + 2)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[(((((((int)threadIdx.x) / 196) * 3249) + (((((int)threadIdx.x) % 196) / 7) * 114)) + ((((int)threadIdx.x) % 7) * 8)) + 2)] * kernel_shared[((((int)threadIdx.x) / 196) * 9)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[(((((((int)threadIdx.x) / 196) * 3249) + (((((int)threadIdx.x) % 196) / 7) * 114)) + ((((int)threadIdx.x) % 7) * 8)) + 3)] * kernel_shared[(((((int)threadIdx.x) / 196) * 9) + 1)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[(((((((int)threadIdx.x) / 196) * 3249) + (((((int)threadIdx.x) % 196) / 7) * 114)) + ((((int)threadIdx.x) % 7) * 8)) + 4)] * kernel_shared[(((((int)threadIdx.x) / 196) * 9) + 2)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[(((((((int)threadIdx.x) / 196) * 3249) + (((((int)threadIdx.x) % 196) / 7) * 114)) + ((((int)threadIdx.x) % 7) * 8)) + 4)] * kernel_shared[((((int)threadIdx.x) / 196) * 9)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[(((((((int)threadIdx.x) / 196) * 3249) + (((((int)threadIdx.x) % 196) / 7) * 114)) + ((((int)threadIdx.x) % 7) * 8)) + 5)] * kernel_shared[(((((int)threadIdx.x) / 196) * 9) + 1)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[(((((((int)threadIdx.x) / 196) * 3249) + (((((int)threadIdx.x) % 196) / 7) * 114)) + ((((int)threadIdx.x) % 7) * 8)) + 6)] * kernel_shared[(((((int)threadIdx.x) / 196) * 9) + 2)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[(((((((int)threadIdx.x) / 196) * 3249) + (((((int)threadIdx.x) % 196) / 7) * 114)) + ((((int)threadIdx.x) % 7) * 8)) + 6)] * kernel_shared[((((int)threadIdx.x) / 196) * 9)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[(((((((int)threadIdx.x) / 196) * 3249) + (((((int)threadIdx.x) % 196) / 7) * 114)) + ((((int)threadIdx.x) % 7) * 8)) + 7)] * kernel_shared[(((((int)threadIdx.x) / 196) * 9) + 1)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[(((((((int)threadIdx.x) / 196) * 3249) + (((((int)threadIdx.x) % 196) / 7) * 114)) + ((((int)threadIdx.x) % 7) * 8)) + 8)] * kernel_shared[(((((int)threadIdx.x) / 196) * 9) + 2)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((((int)threadIdx.x) / 196) * 3249) + (((((int)threadIdx.x) % 196) / 7) * 114)) + ((((int)threadIdx.x) % 7) * 8)) + 57)] * kernel_shared[(((((int)threadIdx.x) / 196) * 9) + 3)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((((int)threadIdx.x) / 196) * 3249) + (((((int)threadIdx.x) % 196) / 7) * 114)) + ((((int)threadIdx.x) % 7) * 8)) + 58)] * kernel_shared[(((((int)threadIdx.x) / 196) * 9) + 4)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((((int)threadIdx.x) / 196) * 3249) + (((((int)threadIdx.x) % 196) / 7) * 114)) + ((((int)threadIdx.x) % 7) * 8)) + 59)] * kernel_shared[(((((int)threadIdx.x) / 196) * 9) + 5)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[(((((((int)threadIdx.x) / 196) * 3249) + (((((int)threadIdx.x) % 196) / 7) * 114)) + ((((int)threadIdx.x) % 7) * 8)) + 59)] * kernel_shared[(((((int)threadIdx.x) / 196) * 9) + 3)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[(((((((int)threadIdx.x) / 196) * 3249) + (((((int)threadIdx.x) % 196) / 7) * 114)) + ((((int)threadIdx.x) % 7) * 8)) + 60)] * kernel_shared[(((((int)threadIdx.x) / 196) * 9) + 4)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[(((((((int)threadIdx.x) / 196) * 3249) + (((((int)threadIdx.x) % 196) / 7) * 114)) + ((((int)threadIdx.x) % 7) * 8)) + 61)] * kernel_shared[(((((int)threadIdx.x) / 196) * 9) + 5)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[(((((((int)threadIdx.x) / 196) * 3249) + (((((int)threadIdx.x) % 196) / 7) * 114)) + ((((int)threadIdx.x) % 7) * 8)) + 61)] * kernel_shared[(((((int)threadIdx.x) / 196) * 9) + 3)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[(((((((int)threadIdx.x) / 196) * 3249) + (((((int)threadIdx.x) % 196) / 7) * 114)) + ((((int)threadIdx.x) % 7) * 8)) + 62)] * kernel_shared[(((((int)threadIdx.x) / 196) * 9) + 4)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[(((((((int)threadIdx.x) / 196) * 3249) + (((((int)threadIdx.x) % 196) / 7) * 114)) + ((((int)threadIdx.x) % 7) * 8)) + 63)] * kernel_shared[(((((int)threadIdx.x) / 196) * 9) + 5)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[(((((((int)threadIdx.x) / 196) * 3249) + (((((int)threadIdx.x) % 196) / 7) * 114)) + ((((int)threadIdx.x) % 7) * 8)) + 63)] * kernel_shared[(((((int)threadIdx.x) / 196) * 9) + 3)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[(((((((int)threadIdx.x) / 196) * 3249) + (((((int)threadIdx.x) % 196) / 7) * 114)) + ((((int)threadIdx.x) % 7) * 8)) + 64)] * kernel_shared[(((((int)threadIdx.x) / 196) * 9) + 4)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[(((((((int)threadIdx.x) / 196) * 3249) + (((((int)threadIdx.x) % 196) / 7) * 114)) + ((((int)threadIdx.x) % 7) * 8)) + 65)] * kernel_shared[(((((int)threadIdx.x) / 196) * 9) + 5)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((((int)threadIdx.x) / 196) * 3249) + (((((int)threadIdx.x) % 196) / 7) * 114)) + ((((int)threadIdx.x) % 7) * 8)) + 114)] * kernel_shared[(((((int)threadIdx.x) / 196) * 9) + 6)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((((int)threadIdx.x) / 196) * 3249) + (((((int)threadIdx.x) % 196) / 7) * 114)) + ((((int)threadIdx.x) % 7) * 8)) + 115)] * kernel_shared[(((((int)threadIdx.x) / 196) * 9) + 7)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((((int)threadIdx.x) / 196) * 3249) + (((((int)threadIdx.x) % 196) / 7) * 114)) + ((((int)threadIdx.x) % 7) * 8)) + 116)] * kernel_shared[(((((int)threadIdx.x) / 196) * 9) + 8)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[(((((((int)threadIdx.x) / 196) * 3249) + (((((int)threadIdx.x) % 196) / 7) * 114)) + ((((int)threadIdx.x) % 7) * 8)) + 116)] * kernel_shared[(((((int)threadIdx.x) / 196) * 9) + 6)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[(((((((int)threadIdx.x) / 196) * 3249) + (((((int)threadIdx.x) % 196) / 7) * 114)) + ((((int)threadIdx.x) % 7) * 8)) + 117)] * kernel_shared[(((((int)threadIdx.x) / 196) * 9) + 7)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[(((((((int)threadIdx.x) / 196) * 3249) + (((((int)threadIdx.x) % 196) / 7) * 114)) + ((((int)threadIdx.x) % 7) * 8)) + 118)] * kernel_shared[(((((int)threadIdx.x) / 196) * 9) + 8)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[(((((((int)threadIdx.x) / 196) * 3249) + (((((int)threadIdx.x) % 196) / 7) * 114)) + ((((int)threadIdx.x) % 7) * 8)) + 118)] * kernel_shared[(((((int)threadIdx.x) / 196) * 9) + 6)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[(((((((int)threadIdx.x) / 196) * 3249) + (((((int)threadIdx.x) % 196) / 7) * 114)) + ((((int)threadIdx.x) % 7) * 8)) + 119)] * kernel_shared[(((((int)threadIdx.x) / 196) * 9) + 7)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[(((((((int)threadIdx.x) / 196) * 3249) + (((((int)threadIdx.x) % 196) / 7) * 114)) + ((((int)threadIdx.x) % 7) * 8)) + 120)] * kernel_shared[(((((int)threadIdx.x) / 196) * 9) + 8)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[(((((((int)threadIdx.x) / 196) * 3249) + (((((int)threadIdx.x) % 196) / 7) * 114)) + ((((int)threadIdx.x) % 7) * 8)) + 120)] * kernel_shared[(((((int)threadIdx.x) / 196) * 9) + 6)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[(((((((int)threadIdx.x) / 196) * 3249) + (((((int)threadIdx.x) % 196) / 7) * 114)) + ((((int)threadIdx.x) % 7) * 8)) + 121)] * kernel_shared[(((((int)threadIdx.x) / 196) * 9) + 7)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[(((((((int)threadIdx.x) / 196) * 3249) + (((((int)threadIdx.x) % 196) / 7) * 114)) + ((((int)threadIdx.x) % 7) * 8)) + 122)] * kernel_shared[(((((int)threadIdx.x) / 196) * 9) + 8)]));
  for (int i3_inner = 0; i3_inner < 4; ++i3_inner) {
    compute[(((((int)blockIdx.x) * 1568) + (((int)threadIdx.x) * 4)) + i3_inner)] = max(DepthwiseConv2d[i3_inner], 0.000000e+00f);
  }
}


