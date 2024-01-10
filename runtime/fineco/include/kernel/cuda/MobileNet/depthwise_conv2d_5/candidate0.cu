
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
extern "C" __global__ void __launch_bounds__(98) candidate0(float* __restrict__ Input, float* __restrict__ kernel, float* __restrict__ compute) {
  float DepthwiseConv2d[8];
  __shared__ float PaddedInput_shared[1152];
  __shared__ float kernel_shared[72];
  DepthwiseConv2d[0] = 0.000000e+00f;
  DepthwiseConv2d[2] = 0.000000e+00f;
  DepthwiseConv2d[1] = 0.000000e+00f;
  DepthwiseConv2d[3] = 0.000000e+00f;
  DepthwiseConv2d[4] = 0.000000e+00f;
  DepthwiseConv2d[6] = 0.000000e+00f;
  DepthwiseConv2d[5] = 0.000000e+00f;
  DepthwiseConv2d[7] = 0.000000e+00f;
  if (((int)threadIdx.x) < 64) {
    PaddedInput_shared[(((int)threadIdx.x) * 18)] = (((1 <= ((((((int)blockIdx.x) & 7) >> 2) * 14) + ((((int)threadIdx.x) & 7) * 2))) && (1 <= (((int)blockIdx.x) & 3))) ? Input[(((((((((int)blockIdx.x) >> 3) * 6272) + ((((int)threadIdx.x) >> 3) * 784)) + (((((int)blockIdx.x) & 7) >> 2) * 392)) + ((((int)threadIdx.x) & 7) * 56)) + ((((int)blockIdx.x) & 3) * 7)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) * 18) + 1)] = ((1 <= ((((((int)blockIdx.x) & 7) >> 2) * 14) + ((((int)threadIdx.x) & 7) * 2))) ? Input[(((((((((int)blockIdx.x) >> 3) * 6272) + ((((int)threadIdx.x) >> 3) * 784)) + (((((int)blockIdx.x) & 7) >> 2) * 392)) + ((((int)threadIdx.x) & 7) * 56)) + ((((int)blockIdx.x) & 3) * 7)) - 28)] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) * 18) + 2)] = ((1 <= ((((((int)blockIdx.x) & 7) >> 2) * 14) + ((((int)threadIdx.x) & 7) * 2))) ? Input[(((((((((int)blockIdx.x) >> 3) * 6272) + ((((int)threadIdx.x) >> 3) * 784)) + (((((int)blockIdx.x) & 7) >> 2) * 392)) + ((((int)threadIdx.x) & 7) * 56)) + ((((int)blockIdx.x) & 3) * 7)) - 27)] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) * 18) + 3)] = ((1 <= ((((((int)blockIdx.x) & 7) >> 2) * 14) + ((((int)threadIdx.x) & 7) * 2))) ? Input[(((((((((int)blockIdx.x) >> 3) * 6272) + ((((int)threadIdx.x) >> 3) * 784)) + (((((int)blockIdx.x) & 7) >> 2) * 392)) + ((((int)threadIdx.x) & 7) * 56)) + ((((int)blockIdx.x) & 3) * 7)) - 26)] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) * 18) + 4)] = ((1 <= ((((((int)blockIdx.x) & 7) >> 2) * 14) + ((((int)threadIdx.x) & 7) * 2))) ? Input[(((((((((int)blockIdx.x) >> 3) * 6272) + ((((int)threadIdx.x) >> 3) * 784)) + (((((int)blockIdx.x) & 7) >> 2) * 392)) + ((((int)threadIdx.x) & 7) * 56)) + ((((int)blockIdx.x) & 3) * 7)) - 25)] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) * 18) + 5)] = ((1 <= ((((((int)blockIdx.x) & 7) >> 2) * 14) + ((((int)threadIdx.x) & 7) * 2))) ? Input[(((((((((int)blockIdx.x) >> 3) * 6272) + ((((int)threadIdx.x) >> 3) * 784)) + (((((int)blockIdx.x) & 7) >> 2) * 392)) + ((((int)threadIdx.x) & 7) * 56)) + ((((int)blockIdx.x) & 3) * 7)) - 24)] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) * 18) + 6)] = ((1 <= ((((((int)blockIdx.x) & 7) >> 2) * 14) + ((((int)threadIdx.x) & 7) * 2))) ? Input[(((((((((int)blockIdx.x) >> 3) * 6272) + ((((int)threadIdx.x) >> 3) * 784)) + (((((int)blockIdx.x) & 7) >> 2) * 392)) + ((((int)threadIdx.x) & 7) * 56)) + ((((int)blockIdx.x) & 3) * 7)) - 23)] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) * 18) + 7)] = ((1 <= ((((((int)blockIdx.x) & 7) >> 2) * 14) + ((((int)threadIdx.x) & 7) * 2))) ? Input[(((((((((int)blockIdx.x) >> 3) * 6272) + ((((int)threadIdx.x) >> 3) * 784)) + (((((int)blockIdx.x) & 7) >> 2) * 392)) + ((((int)threadIdx.x) & 7) * 56)) + ((((int)blockIdx.x) & 3) * 7)) - 22)] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) * 18) + 8)] = (((1 <= ((((((int)blockIdx.x) & 7) >> 2) * 14) + ((((int)threadIdx.x) & 7) * 2))) && ((((int)blockIdx.x) & 3) < 3)) ? Input[(((((((((int)blockIdx.x) >> 3) * 6272) + ((((int)threadIdx.x) >> 3) * 784)) + (((((int)blockIdx.x) & 7) >> 2) * 392)) + ((((int)threadIdx.x) & 7) * 56)) + ((((int)blockIdx.x) & 3) * 7)) - 21)] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) * 18) + 9)] = (((((((((int)blockIdx.x) & 7) >> 2) * 14) + ((((int)threadIdx.x) & 7) * 2)) < 28) && (1 <= (((int)blockIdx.x) & 3))) ? Input[(((((((((int)blockIdx.x) >> 3) * 6272) + ((((int)threadIdx.x) >> 3) * 784)) + (((((int)blockIdx.x) & 7) >> 2) * 392)) + ((((int)threadIdx.x) & 7) * 56)) + ((((int)blockIdx.x) & 3) * 7)) - 1)] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) * 18) + 10)] = ((((((((int)blockIdx.x) & 7) >> 2) * 14) + ((((int)threadIdx.x) & 7) * 2)) < 28) ? Input[((((((((int)blockIdx.x) >> 3) * 6272) + ((((int)threadIdx.x) >> 3) * 784)) + (((((int)blockIdx.x) & 7) >> 2) * 392)) + ((((int)threadIdx.x) & 7) * 56)) + ((((int)blockIdx.x) & 3) * 7))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) * 18) + 11)] = ((((((((int)blockIdx.x) & 7) >> 2) * 14) + ((((int)threadIdx.x) & 7) * 2)) < 28) ? Input[(((((((((int)blockIdx.x) >> 3) * 6272) + ((((int)threadIdx.x) >> 3) * 784)) + (((((int)blockIdx.x) & 7) >> 2) * 392)) + ((((int)threadIdx.x) & 7) * 56)) + ((((int)blockIdx.x) & 3) * 7)) + 1)] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) * 18) + 12)] = ((((((((int)blockIdx.x) & 7) >> 2) * 14) + ((((int)threadIdx.x) & 7) * 2)) < 28) ? Input[(((((((((int)blockIdx.x) >> 3) * 6272) + ((((int)threadIdx.x) >> 3) * 784)) + (((((int)blockIdx.x) & 7) >> 2) * 392)) + ((((int)threadIdx.x) & 7) * 56)) + ((((int)blockIdx.x) & 3) * 7)) + 2)] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) * 18) + 13)] = ((((((((int)blockIdx.x) & 7) >> 2) * 14) + ((((int)threadIdx.x) & 7) * 2)) < 28) ? Input[(((((((((int)blockIdx.x) >> 3) * 6272) + ((((int)threadIdx.x) >> 3) * 784)) + (((((int)blockIdx.x) & 7) >> 2) * 392)) + ((((int)threadIdx.x) & 7) * 56)) + ((((int)blockIdx.x) & 3) * 7)) + 3)] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) * 18) + 14)] = ((((((((int)blockIdx.x) & 7) >> 2) * 14) + ((((int)threadIdx.x) & 7) * 2)) < 28) ? Input[(((((((((int)blockIdx.x) >> 3) * 6272) + ((((int)threadIdx.x) >> 3) * 784)) + (((((int)blockIdx.x) & 7) >> 2) * 392)) + ((((int)threadIdx.x) & 7) * 56)) + ((((int)blockIdx.x) & 3) * 7)) + 4)] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) * 18) + 15)] = ((((((((int)blockIdx.x) & 7) >> 2) * 14) + ((((int)threadIdx.x) & 7) * 2)) < 28) ? Input[(((((((((int)blockIdx.x) >> 3) * 6272) + ((((int)threadIdx.x) >> 3) * 784)) + (((((int)blockIdx.x) & 7) >> 2) * 392)) + ((((int)threadIdx.x) & 7) * 56)) + ((((int)blockIdx.x) & 3) * 7)) + 5)] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) * 18) + 16)] = ((((((((int)blockIdx.x) & 7) >> 2) * 14) + ((((int)threadIdx.x) & 7) * 2)) < 28) ? Input[(((((((((int)blockIdx.x) >> 3) * 6272) + ((((int)threadIdx.x) >> 3) * 784)) + (((((int)blockIdx.x) & 7) >> 2) * 392)) + ((((int)threadIdx.x) & 7) * 56)) + ((((int)blockIdx.x) & 3) * 7)) + 6)] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) * 18) + 17)] = (((((((((int)blockIdx.x) & 7) >> 2) * 14) + ((((int)threadIdx.x) & 7) * 2)) < 28) && ((((int)blockIdx.x) & 3) < 3)) ? Input[(((((((((int)blockIdx.x) >> 3) * 6272) + ((((int)threadIdx.x) >> 3) * 784)) + (((((int)blockIdx.x) & 7) >> 2) * 392)) + ((((int)threadIdx.x) & 7) * 56)) + ((((int)blockIdx.x) & 3) * 7)) + 7)] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 72) {
    kernel_shared[((int)threadIdx.x)] = kernel[(((((int)blockIdx.x) >> 3) * 72) + ((int)threadIdx.x))];
  }
  __syncthreads();
  for (int dj_outer_inner = 0; dj_outer_inner < 3; ++dj_outer_inner) {
    DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((((int)threadIdx.x) / 49) * 576) + (((((int)threadIdx.x) % 49) / 7) * 18)) + dj_outer_inner) + (((int)threadIdx.x) % 7))] * kernel_shared[(((((int)threadIdx.x) / 49) * 36) + dj_outer_inner)]));
    DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[((((((((int)threadIdx.x) / 49) * 576) + (((((int)threadIdx.x) % 49) / 7) * 18)) + dj_outer_inner) + (((int)threadIdx.x) % 7)) + 144)] * kernel_shared[((((((int)threadIdx.x) / 49) * 36) + dj_outer_inner) + 9)]));
    DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((((int)threadIdx.x) / 49) * 576) + (((((int)threadIdx.x) % 49) / 7) * 18)) + dj_outer_inner) + (((int)threadIdx.x) % 7)) + 9)] * kernel_shared[((((((int)threadIdx.x) / 49) * 36) + dj_outer_inner) + 3)]));
    DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[((((((((int)threadIdx.x) / 49) * 576) + (((((int)threadIdx.x) % 49) / 7) * 18)) + dj_outer_inner) + (((int)threadIdx.x) % 7)) + 153)] * kernel_shared[((((((int)threadIdx.x) / 49) * 36) + dj_outer_inner) + 12)]));
    DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((((int)threadIdx.x) / 49) * 576) + (((((int)threadIdx.x) % 49) / 7) * 18)) + dj_outer_inner) + (((int)threadIdx.x) % 7)) + 18)] * kernel_shared[((((((int)threadIdx.x) / 49) * 36) + dj_outer_inner) + 6)]));
    DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[((((((((int)threadIdx.x) / 49) * 576) + (((((int)threadIdx.x) % 49) / 7) * 18)) + dj_outer_inner) + (((int)threadIdx.x) % 7)) + 162)] * kernel_shared[((((((int)threadIdx.x) / 49) * 36) + dj_outer_inner) + 15)]));
    DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((((int)threadIdx.x) / 49) * 576) + (((((int)threadIdx.x) % 49) / 7) * 18)) + dj_outer_inner) + (((int)threadIdx.x) % 7)) + 9)] * kernel_shared[(((((int)threadIdx.x) / 49) * 36) + dj_outer_inner)]));
    DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[((((((((int)threadIdx.x) / 49) * 576) + (((((int)threadIdx.x) % 49) / 7) * 18)) + dj_outer_inner) + (((int)threadIdx.x) % 7)) + 153)] * kernel_shared[((((((int)threadIdx.x) / 49) * 36) + dj_outer_inner) + 9)]));
    DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((((int)threadIdx.x) / 49) * 576) + (((((int)threadIdx.x) % 49) / 7) * 18)) + dj_outer_inner) + (((int)threadIdx.x) % 7)) + 18)] * kernel_shared[((((((int)threadIdx.x) / 49) * 36) + dj_outer_inner) + 3)]));
    DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[((((((((int)threadIdx.x) / 49) * 576) + (((((int)threadIdx.x) % 49) / 7) * 18)) + dj_outer_inner) + (((int)threadIdx.x) % 7)) + 162)] * kernel_shared[((((((int)threadIdx.x) / 49) * 36) + dj_outer_inner) + 12)]));
    DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((((int)threadIdx.x) / 49) * 576) + (((((int)threadIdx.x) % 49) / 7) * 18)) + dj_outer_inner) + (((int)threadIdx.x) % 7)) + 27)] * kernel_shared[((((((int)threadIdx.x) / 49) * 36) + dj_outer_inner) + 6)]));
    DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[((((((((int)threadIdx.x) / 49) * 576) + (((((int)threadIdx.x) % 49) / 7) * 18)) + dj_outer_inner) + (((int)threadIdx.x) % 7)) + 171)] * kernel_shared[((((((int)threadIdx.x) / 49) * 36) + dj_outer_inner) + 15)]));
    DepthwiseConv2d[4] = (DepthwiseConv2d[4] + (PaddedInput_shared[((((((((int)threadIdx.x) / 49) * 576) + (((((int)threadIdx.x) % 49) / 7) * 18)) + dj_outer_inner) + (((int)threadIdx.x) % 7)) + 288)] * kernel_shared[((((((int)threadIdx.x) / 49) * 36) + dj_outer_inner) + 18)]));
    DepthwiseConv2d[6] = (DepthwiseConv2d[6] + (PaddedInput_shared[((((((((int)threadIdx.x) / 49) * 576) + (((((int)threadIdx.x) % 49) / 7) * 18)) + dj_outer_inner) + (((int)threadIdx.x) % 7)) + 432)] * kernel_shared[((((((int)threadIdx.x) / 49) * 36) + dj_outer_inner) + 27)]));
    DepthwiseConv2d[4] = (DepthwiseConv2d[4] + (PaddedInput_shared[((((((((int)threadIdx.x) / 49) * 576) + (((((int)threadIdx.x) % 49) / 7) * 18)) + dj_outer_inner) + (((int)threadIdx.x) % 7)) + 297)] * kernel_shared[((((((int)threadIdx.x) / 49) * 36) + dj_outer_inner) + 21)]));
    DepthwiseConv2d[6] = (DepthwiseConv2d[6] + (PaddedInput_shared[((((((((int)threadIdx.x) / 49) * 576) + (((((int)threadIdx.x) % 49) / 7) * 18)) + dj_outer_inner) + (((int)threadIdx.x) % 7)) + 441)] * kernel_shared[((((((int)threadIdx.x) / 49) * 36) + dj_outer_inner) + 30)]));
    DepthwiseConv2d[4] = (DepthwiseConv2d[4] + (PaddedInput_shared[((((((((int)threadIdx.x) / 49) * 576) + (((((int)threadIdx.x) % 49) / 7) * 18)) + dj_outer_inner) + (((int)threadIdx.x) % 7)) + 306)] * kernel_shared[((((((int)threadIdx.x) / 49) * 36) + dj_outer_inner) + 24)]));
    DepthwiseConv2d[6] = (DepthwiseConv2d[6] + (PaddedInput_shared[((((((((int)threadIdx.x) / 49) * 576) + (((((int)threadIdx.x) % 49) / 7) * 18)) + dj_outer_inner) + (((int)threadIdx.x) % 7)) + 450)] * kernel_shared[((((((int)threadIdx.x) / 49) * 36) + dj_outer_inner) + 33)]));
    DepthwiseConv2d[5] = (DepthwiseConv2d[5] + (PaddedInput_shared[((((((((int)threadIdx.x) / 49) * 576) + (((((int)threadIdx.x) % 49) / 7) * 18)) + dj_outer_inner) + (((int)threadIdx.x) % 7)) + 297)] * kernel_shared[((((((int)threadIdx.x) / 49) * 36) + dj_outer_inner) + 18)]));
    DepthwiseConv2d[7] = (DepthwiseConv2d[7] + (PaddedInput_shared[((((((((int)threadIdx.x) / 49) * 576) + (((((int)threadIdx.x) % 49) / 7) * 18)) + dj_outer_inner) + (((int)threadIdx.x) % 7)) + 441)] * kernel_shared[((((((int)threadIdx.x) / 49) * 36) + dj_outer_inner) + 27)]));
    DepthwiseConv2d[5] = (DepthwiseConv2d[5] + (PaddedInput_shared[((((((((int)threadIdx.x) / 49) * 576) + (((((int)threadIdx.x) % 49) / 7) * 18)) + dj_outer_inner) + (((int)threadIdx.x) % 7)) + 306)] * kernel_shared[((((((int)threadIdx.x) / 49) * 36) + dj_outer_inner) + 21)]));
    DepthwiseConv2d[7] = (DepthwiseConv2d[7] + (PaddedInput_shared[((((((((int)threadIdx.x) / 49) * 576) + (((((int)threadIdx.x) % 49) / 7) * 18)) + dj_outer_inner) + (((int)threadIdx.x) % 7)) + 450)] * kernel_shared[((((((int)threadIdx.x) / 49) * 36) + dj_outer_inner) + 30)]));
    DepthwiseConv2d[5] = (DepthwiseConv2d[5] + (PaddedInput_shared[((((((((int)threadIdx.x) / 49) * 576) + (((((int)threadIdx.x) % 49) / 7) * 18)) + dj_outer_inner) + (((int)threadIdx.x) % 7)) + 315)] * kernel_shared[((((((int)threadIdx.x) / 49) * 36) + dj_outer_inner) + 24)]));
    DepthwiseConv2d[7] = (DepthwiseConv2d[7] + (PaddedInput_shared[((((((((int)threadIdx.x) / 49) * 576) + (((((int)threadIdx.x) % 49) / 7) * 18)) + dj_outer_inner) + (((int)threadIdx.x) % 7)) + 459)] * kernel_shared[((((((int)threadIdx.x) / 49) * 36) + dj_outer_inner) + 33)]));
  }
  for (int i1_inner = 0; i1_inner < 4; ++i1_inner) {
    for (int i2_inner = 0; i2_inner < 2; ++i2_inner) {
      compute[(((((((((((int)blockIdx.x) >> 3) * 6272) + ((((int)threadIdx.x) / 49) * 3136)) + (i1_inner * 784)) + (((((int)blockIdx.x) & 7) >> 2) * 392)) + (((((int)threadIdx.x) % 49) / 7) * 56)) + (i2_inner * 28)) + ((((int)blockIdx.x) & 3) * 7)) + (((int)threadIdx.x) % 7))] = max(DepthwiseConv2d[((i1_inner * 2) + i2_inner)], 0.000000e+00f);
    }
  }
}


