
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
extern "C" __global__ void __launch_bounds__(896) candidate2(float* __restrict__ Input, float* __restrict__ kernel, float* __restrict__ compute) {
  float DepthwiseConv2d[2];
  __shared__ float PaddedInput_shared[3712];
  __shared__ float kernel_shared[384];
  DepthwiseConv2d[0] = 0.000000e+00f;
  DepthwiseConv2d[1] = 0.000000e+00f;
  PaddedInput_shared[((int)threadIdx.x)] = (((1 <= (((int)blockIdx.x) % 14)) && (1 <= (((int)threadIdx.x) % 29))) ? Input[((((((((int)blockIdx.x) / 14) * 100352) + ((((int)threadIdx.x) / 29) * 784)) + ((((int)blockIdx.x) % 14) * 56)) + (((int)threadIdx.x) % 29)) - 29)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 896)] = (((1 <= (((int)blockIdx.x) % 14)) && (1 <= ((((int)threadIdx.x) + 26) % 29))) ? Input[((((((((int)blockIdx.x) / 14) * 100352) + (((((int)threadIdx.x) + 896) / 29) * 784)) + ((((int)blockIdx.x) % 14) * 56)) + ((((int)threadIdx.x) + 26) % 29)) - 29)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 1792)] = (((1 <= (((int)blockIdx.x) % 14)) && (1 <= ((((int)threadIdx.x) + 23) % 29))) ? Input[((((((((int)blockIdx.x) / 14) * 100352) + (((((int)threadIdx.x) + 1792) / 29) * 784)) + ((((int)blockIdx.x) % 14) * 56)) + ((((int)threadIdx.x) + 23) % 29)) - 29)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 2688)] = (((1 <= (((int)blockIdx.x) % 14)) && (1 <= ((((int)threadIdx.x) + 20) % 29))) ? Input[((((((((int)blockIdx.x) / 14) * 100352) + (((((int)threadIdx.x) + 2688) / 29) * 784)) + ((((int)blockIdx.x) % 14) * 56)) + ((((int)threadIdx.x) + 20) % 29)) - 29)] : 0.000000e+00f);
  if (((int)threadIdx.x) < 128) {
    PaddedInput_shared[(((int)threadIdx.x) + 3584)] = (((1 <= (((int)blockIdx.x) % 14)) && (1 <= ((((int)threadIdx.x) + 17) % 29))) ? Input[((((((((int)blockIdx.x) / 14) * 100352) + (((((int)threadIdx.x) + 3584) / 29) * 784)) + ((((int)blockIdx.x) % 14) * 56)) + ((((int)threadIdx.x) + 17) % 29)) - 29)] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 384) {
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)blockIdx.x) / 14) * 1152) + ((((int)threadIdx.x) / 3) * 9)) + (((int)threadIdx.x) % 3))];
  }
  __syncthreads();
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((int)threadIdx.x) / 7) * 29) + ((((int)threadIdx.x) % 7) * 2))] * kernel_shared[((((int)threadIdx.x) / 7) * 3)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 29) + ((((int)threadIdx.x) % 7) * 2)) + 14)] * kernel_shared[((((int)threadIdx.x) / 7) * 3)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 29) + ((((int)threadIdx.x) % 7) * 2)) + 1)] * kernel_shared[(((((int)threadIdx.x) / 7) * 3) + 1)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 29) + ((((int)threadIdx.x) % 7) * 2)) + 15)] * kernel_shared[(((((int)threadIdx.x) / 7) * 3) + 1)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 29) + ((((int)threadIdx.x) % 7) * 2)) + 2)] * kernel_shared[(((((int)threadIdx.x) / 7) * 3) + 2)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 29) + ((((int)threadIdx.x) % 7) * 2)) + 16)] * kernel_shared[(((((int)threadIdx.x) / 7) * 3) + 2)]));
  __syncthreads();
  PaddedInput_shared[((int)threadIdx.x)] = ((1 <= (((int)threadIdx.x) % 29)) ? Input[((((((((int)blockIdx.x) / 14) * 100352) + ((((int)threadIdx.x) / 29) * 784)) + ((((int)blockIdx.x) % 14) * 56)) + (((int)threadIdx.x) % 29)) - 1)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 896)] = ((1 <= ((((int)threadIdx.x) + 26) % 29)) ? Input[((((((((int)blockIdx.x) / 14) * 100352) + (((((int)threadIdx.x) + 896) / 29) * 784)) + ((((int)blockIdx.x) % 14) * 56)) + ((((int)threadIdx.x) + 26) % 29)) - 1)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 1792)] = ((1 <= ((((int)threadIdx.x) + 23) % 29)) ? Input[((((((((int)blockIdx.x) / 14) * 100352) + (((((int)threadIdx.x) + 1792) / 29) * 784)) + ((((int)blockIdx.x) % 14) * 56)) + ((((int)threadIdx.x) + 23) % 29)) - 1)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 2688)] = ((1 <= ((((int)threadIdx.x) + 20) % 29)) ? Input[((((((((int)blockIdx.x) / 14) * 100352) + (((((int)threadIdx.x) + 2688) / 29) * 784)) + ((((int)blockIdx.x) % 14) * 56)) + ((((int)threadIdx.x) + 20) % 29)) - 1)] : 0.000000e+00f);
  if (((int)threadIdx.x) < 128) {
    PaddedInput_shared[(((int)threadIdx.x) + 3584)] = ((1 <= ((((int)threadIdx.x) + 17) % 29)) ? Input[((((((((int)blockIdx.x) / 14) * 100352) + (((((int)threadIdx.x) + 3584) / 29) * 784)) + ((((int)blockIdx.x) % 14) * 56)) + ((((int)threadIdx.x) + 17) % 29)) - 1)] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 384) {
    kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) / 14) * 1152) + ((((int)threadIdx.x) / 3) * 9)) + (((int)threadIdx.x) % 3)) + 3)];
  }
  __syncthreads();
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((int)threadIdx.x) / 7) * 29) + ((((int)threadIdx.x) % 7) * 2))] * kernel_shared[((((int)threadIdx.x) / 7) * 3)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 29) + ((((int)threadIdx.x) % 7) * 2)) + 14)] * kernel_shared[((((int)threadIdx.x) / 7) * 3)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 29) + ((((int)threadIdx.x) % 7) * 2)) + 1)] * kernel_shared[(((((int)threadIdx.x) / 7) * 3) + 1)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 29) + ((((int)threadIdx.x) % 7) * 2)) + 15)] * kernel_shared[(((((int)threadIdx.x) / 7) * 3) + 1)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 29) + ((((int)threadIdx.x) % 7) * 2)) + 2)] * kernel_shared[(((((int)threadIdx.x) / 7) * 3) + 2)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 29) + ((((int)threadIdx.x) % 7) * 2)) + 16)] * kernel_shared[(((((int)threadIdx.x) / 7) * 3) + 2)]));
  __syncthreads();
  PaddedInput_shared[((int)threadIdx.x)] = ((1 <= (((int)threadIdx.x) % 29)) ? Input[((((((((int)blockIdx.x) / 14) * 100352) + ((((int)threadIdx.x) / 29) * 784)) + ((((int)blockIdx.x) % 14) * 56)) + (((int)threadIdx.x) % 29)) + 27)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 896)] = ((1 <= ((((int)threadIdx.x) + 26) % 29)) ? Input[((((((((int)blockIdx.x) / 14) * 100352) + (((((int)threadIdx.x) + 896) / 29) * 784)) + ((((int)blockIdx.x) % 14) * 56)) + ((((int)threadIdx.x) + 26) % 29)) + 27)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 1792)] = ((1 <= ((((int)threadIdx.x) + 23) % 29)) ? Input[((((((((int)blockIdx.x) / 14) * 100352) + (((((int)threadIdx.x) + 1792) / 29) * 784)) + ((((int)blockIdx.x) % 14) * 56)) + ((((int)threadIdx.x) + 23) % 29)) + 27)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 2688)] = ((1 <= ((((int)threadIdx.x) + 20) % 29)) ? Input[((((((((int)blockIdx.x) / 14) * 100352) + (((((int)threadIdx.x) + 2688) / 29) * 784)) + ((((int)blockIdx.x) % 14) * 56)) + ((((int)threadIdx.x) + 20) % 29)) + 27)] : 0.000000e+00f);
  if (((int)threadIdx.x) < 128) {
    PaddedInput_shared[(((int)threadIdx.x) + 3584)] = ((1 <= ((((int)threadIdx.x) + 17) % 29)) ? Input[((((((((int)blockIdx.x) / 14) * 100352) + (((((int)threadIdx.x) + 3584) / 29) * 784)) + ((((int)blockIdx.x) % 14) * 56)) + ((((int)threadIdx.x) + 17) % 29)) + 27)] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 384) {
    kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) / 14) * 1152) + ((((int)threadIdx.x) / 3) * 9)) + (((int)threadIdx.x) % 3)) + 6)];
  }
  __syncthreads();
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((int)threadIdx.x) / 7) * 29) + ((((int)threadIdx.x) % 7) * 2))] * kernel_shared[((((int)threadIdx.x) / 7) * 3)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 29) + ((((int)threadIdx.x) % 7) * 2)) + 14)] * kernel_shared[((((int)threadIdx.x) / 7) * 3)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 29) + ((((int)threadIdx.x) % 7) * 2)) + 1)] * kernel_shared[(((((int)threadIdx.x) / 7) * 3) + 1)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 29) + ((((int)threadIdx.x) % 7) * 2)) + 15)] * kernel_shared[(((((int)threadIdx.x) / 7) * 3) + 1)]));
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 29) + ((((int)threadIdx.x) % 7) * 2)) + 2)] * kernel_shared[(((((int)threadIdx.x) / 7) * 3) + 2)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 29) + ((((int)threadIdx.x) % 7) * 2)) + 16)] * kernel_shared[(((((int)threadIdx.x) / 7) * 3) + 2)]));
  compute[(((((((int)blockIdx.x) / 14) * 25088) + ((((int)threadIdx.x) / 7) * 196)) + ((((int)blockIdx.x) % 14) * 14)) + (((int)threadIdx.x) % 7))] = max(DepthwiseConv2d[0], 0.000000e+00f);
  compute[((((((((int)blockIdx.x) / 14) * 25088) + ((((int)threadIdx.x) / 7) * 196)) + ((((int)blockIdx.x) % 14) * 14)) + (((int)threadIdx.x) % 7)) + 7)] = max(DepthwiseConv2d[1], 0.000000e+00f);
}


