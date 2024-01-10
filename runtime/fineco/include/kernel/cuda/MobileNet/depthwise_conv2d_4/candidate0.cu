
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
extern "C" __global__ void __launch_bounds__(448) candidate0(float* __restrict__ Input, float* __restrict__ kernel, float* __restrict__ compute) {
  float DepthwiseConv2d[1];
  __shared__ float PaddedInput_shared[2784];
  __shared__ float kernel_shared[288];
  DepthwiseConv2d[0] = 0.000000e+00f;
  PaddedInput_shared[((int)threadIdx.x)] = (((1 <= ((((((int)blockIdx.x) % 56) >> 1) * 2) + ((((int)threadIdx.x) % 87) / 29))) && (1 <= (((((int)blockIdx.x) & 1) * 28) + (((int)threadIdx.x) % 29)))) ? Input[((((((((((int)blockIdx.x) / 56) * 100352) + ((((int)threadIdx.x) / 87) * 3136)) + (((((int)blockIdx.x) % 56) >> 1) * 112)) + (((((int)threadIdx.x) % 87) / 29) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + (((int)threadIdx.x) % 29)) - 57)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 448)] = (((1 <= ((((((int)blockIdx.x) % 56) >> 1) * 2) + (((((int)threadIdx.x) + 13) % 87) / 29))) && (1 <= (((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 13) % 29)))) ? Input[((((((((((int)blockIdx.x) / 56) * 100352) + (((((int)threadIdx.x) + 448) / 87) * 3136)) + (((((int)blockIdx.x) % 56) >> 1) * 112)) + ((((((int)threadIdx.x) + 13) % 87) / 29) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + ((((int)threadIdx.x) + 13) % 29)) - 57)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 896)] = (((1 <= ((((((int)blockIdx.x) % 56) >> 1) * 2) + (((((int)threadIdx.x) + 26) % 87) / 29))) && (1 <= (((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 26) % 29)))) ? Input[((((((((((int)blockIdx.x) / 56) * 100352) + (((((int)threadIdx.x) + 896) / 87) * 3136)) + (((((int)blockIdx.x) % 56) >> 1) * 112)) + ((((((int)threadIdx.x) + 26) % 87) / 29) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + ((((int)threadIdx.x) + 26) % 29)) - 57)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 1344)] = (((1 <= ((((((int)blockIdx.x) % 56) >> 1) * 2) + (((((int)threadIdx.x) + 39) % 87) / 29))) && (1 <= (((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 10) % 29)))) ? Input[((((((((((int)blockIdx.x) / 56) * 100352) + (((((int)threadIdx.x) + 1344) / 87) * 3136)) + (((((int)blockIdx.x) % 56) >> 1) * 112)) + ((((((int)threadIdx.x) + 39) % 87) / 29) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + ((((int)threadIdx.x) + 10) % 29)) - 57)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 1792)] = (((1 <= ((((((int)blockIdx.x) % 56) >> 1) * 2) + (((((int)threadIdx.x) + 52) % 87) / 29))) && (1 <= (((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 23) % 29)))) ? Input[((((((((((int)blockIdx.x) / 56) * 100352) + (((((int)threadIdx.x) + 1792) / 87) * 3136)) + (((((int)blockIdx.x) % 56) >> 1) * 112)) + ((((((int)threadIdx.x) + 52) % 87) / 29) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + ((((int)threadIdx.x) + 23) % 29)) - 57)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 2240)] = (((1 <= ((((((int)blockIdx.x) % 56) >> 1) * 2) + (((((int)threadIdx.x) + 65) % 87) / 29))) && (1 <= (((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 7) % 29)))) ? Input[((((((((((int)blockIdx.x) / 56) * 100352) + (((((int)threadIdx.x) + 2240) / 87) * 3136)) + (((((int)blockIdx.x) % 56) >> 1) * 112)) + ((((((int)threadIdx.x) + 65) % 87) / 29) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + ((((int)threadIdx.x) + 7) % 29)) - 57)] : 0.000000e+00f);
  if (((int)threadIdx.x) < 96) {
    PaddedInput_shared[(((int)threadIdx.x) + 2688)] = (((1 <= ((((((int)blockIdx.x) % 56) >> 1) * 2) + (((((int)threadIdx.x) + 78) % 87) / 29))) && (1 <= (((((int)blockIdx.x) & 1) * 28) + ((((int)threadIdx.x) + 20) % 29)))) ? Input[((((((((((int)blockIdx.x) / 56) * 100352) + (((((int)threadIdx.x) + 2688) / 87) * 3136)) + (((((int)blockIdx.x) % 56) >> 1) * 112)) + ((((((int)threadIdx.x) + 78) % 87) / 29) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + ((((int)threadIdx.x) + 20) % 29)) - 57)] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 288) {
    kernel_shared[((int)threadIdx.x)] = kernel[(((((int)blockIdx.x) / 56) * 288) + ((int)threadIdx.x))];
  }
  __syncthreads();
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((int)threadIdx.x) / 14) * 87) + ((((int)threadIdx.x) % 14) * 2))] * kernel_shared[((((int)threadIdx.x) / 14) * 9)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 14) * 87) + ((((int)threadIdx.x) % 14) * 2)) + 29)] * kernel_shared[(((((int)threadIdx.x) / 14) * 9) + 3)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 14) * 87) + ((((int)threadIdx.x) % 14) * 2)) + 58)] * kernel_shared[(((((int)threadIdx.x) / 14) * 9) + 6)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 14) * 87) + ((((int)threadIdx.x) % 14) * 2)) + 1)] * kernel_shared[(((((int)threadIdx.x) / 14) * 9) + 1)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 14) * 87) + ((((int)threadIdx.x) % 14) * 2)) + 30)] * kernel_shared[(((((int)threadIdx.x) / 14) * 9) + 4)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 14) * 87) + ((((int)threadIdx.x) % 14) * 2)) + 59)] * kernel_shared[(((((int)threadIdx.x) / 14) * 9) + 7)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 14) * 87) + ((((int)threadIdx.x) % 14) * 2)) + 2)] * kernel_shared[(((((int)threadIdx.x) / 14) * 9) + 2)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 14) * 87) + ((((int)threadIdx.x) % 14) * 2)) + 31)] * kernel_shared[(((((int)threadIdx.x) / 14) * 9) + 5)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 14) * 87) + ((((int)threadIdx.x) % 14) * 2)) + 60)] * kernel_shared[(((((int)threadIdx.x) / 14) * 9) + 8)]));
  compute[(((((((int)blockIdx.x) / 56) * 25088) + ((((int)threadIdx.x) / 14) * 784)) + ((((int)blockIdx.x) % 56) * 14)) + (((int)threadIdx.x) % 14))] = max(DepthwiseConv2d[0], 0.000000e+00f);
}


