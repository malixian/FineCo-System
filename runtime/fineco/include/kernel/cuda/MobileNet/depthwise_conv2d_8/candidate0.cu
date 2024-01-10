
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
extern "C" __global__ void __launch_bounds__(14) candidate0(float* __restrict__ Input, float* __restrict__ kernel, float* __restrict__ compute) {
  float DepthwiseConv2d[1];
  __shared__ float PaddedInput_shared[90];
  __shared__ float kernel_shared[18];
  DepthwiseConv2d[0] = 0.000000e+00f;
  PaddedInput_shared[(((int)threadIdx.x) * 2)] = (((1 <= (((((int)blockIdx.x) % 7) * 2) + ((((int)threadIdx.x) * 2) / 15))) && (1 <= ((((int)threadIdx.x) * 2) % 15))) ? Input[((((((((int)blockIdx.x) / 7) * 392) + ((((int)blockIdx.x) % 7) * 28)) + (((((int)threadIdx.x) * 2) / 15) * 14)) + ((((int)threadIdx.x) * 2) % 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) * 2) + 1)] = (((1 <= (((((int)blockIdx.x) % 7) * 2) + (((((int)threadIdx.x) * 2) + 1) / 15))) && (1 <= (((((int)threadIdx.x) * 2) + 1) % 15))) ? Input[((((((((int)blockIdx.x) / 7) * 392) + ((((int)blockIdx.x) % 7) * 28)) + ((((((int)threadIdx.x) * 2) + 1) / 15) * 14)) + (((((int)threadIdx.x) * 2) + 1) % 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) * 2) + 28)] = (((1 <= (((((int)blockIdx.x) % 7) * 2) + ((((((int)threadIdx.x) * 2) + 28) % 45) / 15))) && (1 <= (((((int)threadIdx.x) * 2) + 13) % 15))) ? Input[(((((((((int)blockIdx.x) / 7) * 392) + ((((((int)threadIdx.x) * 2) + 28) / 45) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((((((int)threadIdx.x) * 2) + 28) % 45) / 15) * 14)) + (((((int)threadIdx.x) * 2) + 13) % 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) * 2) + 29)] = (((1 <= (((((int)blockIdx.x) % 7) * 2) + ((((((int)threadIdx.x) * 2) + 29) % 45) / 15))) && (1 <= (((((int)threadIdx.x) * 2) + 14) % 15))) ? Input[(((((((((int)blockIdx.x) / 7) * 392) + ((((((int)threadIdx.x) * 2) + 29) / 45) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((((((int)threadIdx.x) * 2) + 29) % 45) / 15) * 14)) + (((((int)threadIdx.x) * 2) + 14) % 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) * 2) + 56)] = (((1 <= (((((int)blockIdx.x) % 7) * 2) + ((((((int)threadIdx.x) * 2) + 11) % 45) / 15))) && (1 <= (((((int)threadIdx.x) * 2) + 11) % 15))) ? Input[(((((((((int)blockIdx.x) / 7) * 392) + ((((((int)threadIdx.x) * 2) + 56) / 45) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((((((int)threadIdx.x) * 2) + 11) % 45) / 15) * 14)) + (((((int)threadIdx.x) * 2) + 11) % 15)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) * 2) + 57)] = (((1 <= (((((int)blockIdx.x) % 7) * 2) + ((((((int)threadIdx.x) * 2) + 12) % 45) / 15))) && (1 <= (((((int)threadIdx.x) * 2) + 12) % 15))) ? Input[(((((((((int)blockIdx.x) / 7) * 392) + ((((((int)threadIdx.x) * 2) + 57) / 45) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((((((int)threadIdx.x) * 2) + 12) % 45) / 15) * 14)) + (((((int)threadIdx.x) * 2) + 12) % 15)) - 15)] : 0.000000e+00f);
  if (((int)threadIdx.x) < 3) {
    PaddedInput_shared[((((int)threadIdx.x) * 2) + 84)] = Input[(((((((((int)blockIdx.x) / 7) * 392) + ((((((int)threadIdx.x) * 2) + 84) / 45) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((((((int)threadIdx.x) * 2) + 39) % 45) / 15) * 14)) + ((((int)threadIdx.x) * 2) + 9)) - 15)];
  }
  if (((int)threadIdx.x) < 3) {
    PaddedInput_shared[((((int)threadIdx.x) * 2) + 85)] = Input[(((((((((int)blockIdx.x) / 7) * 392) + ((((((int)threadIdx.x) * 2) + 85) / 45) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((((((int)threadIdx.x) * 2) + 40) % 45) / 15) * 14)) + ((((int)threadIdx.x) * 2) + 10)) - 15)];
  }
  if (((int)threadIdx.x) < 6) {
    *(float3*)(kernel_shared + (((int)threadIdx.x) * 3)) = *(float3*)(kernel + (((((int)blockIdx.x) / 7) * 18) + (((int)threadIdx.x) * 3)));
  }
  __syncthreads();
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((int)threadIdx.x) / 7) * 45) + ((((int)threadIdx.x) % 7) * 2))] * kernel_shared[((((int)threadIdx.x) / 7) * 9)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 45) + ((((int)threadIdx.x) % 7) * 2)) + 1)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 1)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 45) + ((((int)threadIdx.x) % 7) * 2)) + 2)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 2)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 45) + ((((int)threadIdx.x) % 7) * 2)) + 15)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 3)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 45) + ((((int)threadIdx.x) % 7) * 2)) + 16)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 4)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 45) + ((((int)threadIdx.x) % 7) * 2)) + 17)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 5)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 45) + ((((int)threadIdx.x) % 7) * 2)) + 30)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 6)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 45) + ((((int)threadIdx.x) % 7) * 2)) + 31)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 7)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 45) + ((((int)threadIdx.x) % 7) * 2)) + 32)] * kernel_shared[(((((int)threadIdx.x) / 7) * 9) + 8)]));
  compute[(((((((int)blockIdx.x) / 7) * 98) + ((((int)threadIdx.x) / 7) * 49)) + ((((int)blockIdx.x) % 7) * 7)) + (((int)threadIdx.x) % 7))] = max(DepthwiseConv2d[0], 0.000000e+00f);
}


