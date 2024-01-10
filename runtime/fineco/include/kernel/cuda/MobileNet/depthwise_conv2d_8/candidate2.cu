
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
extern "C" __global__ void __launch_bounds__(448) candidate2(float* __restrict__ Input, float* __restrict__ kernel, float* __restrict__ compute) {
  float DepthwiseConv2d[2];
  __shared__ float PaddedInput_shared[1664];
  __shared__ float kernel_shared[128];
  DepthwiseConv2d[0] = 0.000000e+00f;
  DepthwiseConv2d[1] = 0.000000e+00f;
  PaddedInput_shared[((int)threadIdx.x)] = (((1 <= (((int)blockIdx.x) % 7)) && (1 <= (((int)threadIdx.x) % 13))) ? Input[((((((((int)blockIdx.x) / 7) * 25088) + ((((int)threadIdx.x) / 13) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 13)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 448)] = (((1 <= (((int)blockIdx.x) % 7)) && (1 <= ((((int)threadIdx.x) + 6) % 13))) ? Input[((((((((int)blockIdx.x) / 7) * 25088) + (((((int)threadIdx.x) + 448) / 13) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) + 6) % 13)) - 15)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 896)] = (((1 <= (((int)blockIdx.x) % 7)) && (1 <= ((((int)threadIdx.x) + 12) % 13))) ? Input[((((((((int)blockIdx.x) / 7) * 25088) + (((((int)threadIdx.x) + 896) / 13) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) + 12) % 13)) - 15)] : 0.000000e+00f);
  if (((int)threadIdx.x) < 320) {
    PaddedInput_shared[(((int)threadIdx.x) + 1344)] = (((1 <= (((int)blockIdx.x) % 7)) && (1 <= ((((int)threadIdx.x) + 5) % 13))) ? Input[((((((((int)blockIdx.x) / 7) * 25088) + (((((int)threadIdx.x) + 1344) / 13) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) + 5) % 13)) - 15)] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 128) {
    kernel_shared[((int)threadIdx.x)] = kernel[(((((int)blockIdx.x) / 7) * 1152) + (((int)threadIdx.x) * 9))];
  }
  __syncthreads();
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((int)threadIdx.x) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2))] * kernel_shared[((((int)threadIdx.x) / 7) * 2)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2)) + 13)] * kernel_shared[(((((int)threadIdx.x) / 7) * 2) + 1)]));
  __syncthreads();
  PaddedInput_shared[((int)threadIdx.x)] = ((1 <= (((int)blockIdx.x) % 7)) ? Input[((((((((int)blockIdx.x) / 7) * 25088) + ((((int)threadIdx.x) / 13) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 13)) - 14)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 448)] = ((1 <= (((int)blockIdx.x) % 7)) ? Input[((((((((int)blockIdx.x) / 7) * 25088) + (((((int)threadIdx.x) + 448) / 13) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) + 6) % 13)) - 14)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 896)] = ((1 <= (((int)blockIdx.x) % 7)) ? Input[((((((((int)blockIdx.x) / 7) * 25088) + (((((int)threadIdx.x) + 896) / 13) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) + 12) % 13)) - 14)] : 0.000000e+00f);
  if (((int)threadIdx.x) < 320) {
    PaddedInput_shared[(((int)threadIdx.x) + 1344)] = ((1 <= (((int)blockIdx.x) % 7)) ? Input[((((((((int)blockIdx.x) / 7) * 25088) + (((((int)threadIdx.x) + 1344) / 13) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) + 5) % 13)) - 14)] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 128) {
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)blockIdx.x) / 7) * 1152) + (((int)threadIdx.x) * 9)) + 1)];
  }
  __syncthreads();
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((int)threadIdx.x) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2))] * kernel_shared[((((int)threadIdx.x) / 7) * 2)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2)) + 13)] * kernel_shared[(((((int)threadIdx.x) / 7) * 2) + 1)]));
  __syncthreads();
  PaddedInput_shared[((int)threadIdx.x)] = ((1 <= (((int)blockIdx.x) % 7)) ? Input[((((((((int)blockIdx.x) / 7) * 25088) + ((((int)threadIdx.x) / 13) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 13)) - 13)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 448)] = ((1 <= (((int)blockIdx.x) % 7)) ? Input[((((((((int)blockIdx.x) / 7) * 25088) + (((((int)threadIdx.x) + 448) / 13) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) + 6) % 13)) - 13)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 896)] = ((1 <= (((int)blockIdx.x) % 7)) ? Input[((((((((int)blockIdx.x) / 7) * 25088) + (((((int)threadIdx.x) + 896) / 13) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) + 12) % 13)) - 13)] : 0.000000e+00f);
  if (((int)threadIdx.x) < 320) {
    PaddedInput_shared[(((int)threadIdx.x) + 1344)] = ((1 <= (((int)blockIdx.x) % 7)) ? Input[((((((((int)blockIdx.x) / 7) * 25088) + (((((int)threadIdx.x) + 1344) / 13) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) + 5) % 13)) - 13)] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 128) {
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)blockIdx.x) / 7) * 1152) + (((int)threadIdx.x) * 9)) + 2)];
  }
  __syncthreads();
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((int)threadIdx.x) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2))] * kernel_shared[((((int)threadIdx.x) / 7) * 2)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2)) + 13)] * kernel_shared[(((((int)threadIdx.x) / 7) * 2) + 1)]));
  __syncthreads();
  PaddedInput_shared[((int)threadIdx.x)] = ((1 <= (((int)threadIdx.x) % 13)) ? Input[((((((((int)blockIdx.x) / 7) * 25088) + ((((int)threadIdx.x) / 13) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 13)) - 1)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 448)] = ((1 <= ((((int)threadIdx.x) + 6) % 13)) ? Input[((((((((int)blockIdx.x) / 7) * 25088) + (((((int)threadIdx.x) + 448) / 13) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) + 6) % 13)) - 1)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 896)] = ((1 <= ((((int)threadIdx.x) + 12) % 13)) ? Input[((((((((int)blockIdx.x) / 7) * 25088) + (((((int)threadIdx.x) + 896) / 13) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) + 12) % 13)) - 1)] : 0.000000e+00f);
  if (((int)threadIdx.x) < 320) {
    PaddedInput_shared[(((int)threadIdx.x) + 1344)] = ((1 <= ((((int)threadIdx.x) + 5) % 13)) ? Input[((((((((int)blockIdx.x) / 7) * 25088) + (((((int)threadIdx.x) + 1344) / 13) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) + 5) % 13)) - 1)] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 128) {
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)blockIdx.x) / 7) * 1152) + (((int)threadIdx.x) * 9)) + 3)];
  }
  __syncthreads();
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((int)threadIdx.x) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2))] * kernel_shared[((((int)threadIdx.x) / 7) * 2)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2)) + 13)] * kernel_shared[(((((int)threadIdx.x) / 7) * 2) + 1)]));
  __syncthreads();
  PaddedInput_shared[((int)threadIdx.x)] = Input[(((((((int)blockIdx.x) / 7) * 25088) + ((((int)threadIdx.x) / 13) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 13))];
  PaddedInput_shared[(((int)threadIdx.x) + 448)] = Input[(((((((int)blockIdx.x) / 7) * 25088) + (((((int)threadIdx.x) + 448) / 13) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) + 6) % 13))];
  PaddedInput_shared[(((int)threadIdx.x) + 896)] = Input[(((((((int)blockIdx.x) / 7) * 25088) + (((((int)threadIdx.x) + 896) / 13) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) + 12) % 13))];
  if (((int)threadIdx.x) < 320) {
    PaddedInput_shared[(((int)threadIdx.x) + 1344)] = Input[(((((((int)blockIdx.x) / 7) * 25088) + (((((int)threadIdx.x) + 1344) / 13) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) + 5) % 13))];
  }
  if (((int)threadIdx.x) < 128) {
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)blockIdx.x) / 7) * 1152) + (((int)threadIdx.x) * 9)) + 4)];
  }
  __syncthreads();
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((int)threadIdx.x) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2))] * kernel_shared[((((int)threadIdx.x) / 7) * 2)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2)) + 13)] * kernel_shared[(((((int)threadIdx.x) / 7) * 2) + 1)]));
  __syncthreads();
  PaddedInput_shared[((int)threadIdx.x)] = Input[((((((((int)blockIdx.x) / 7) * 25088) + ((((int)threadIdx.x) / 13) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 13)) + 1)];
  PaddedInput_shared[(((int)threadIdx.x) + 448)] = Input[((((((((int)blockIdx.x) / 7) * 25088) + (((((int)threadIdx.x) + 448) / 13) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) + 6) % 13)) + 1)];
  PaddedInput_shared[(((int)threadIdx.x) + 896)] = Input[((((((((int)blockIdx.x) / 7) * 25088) + (((((int)threadIdx.x) + 896) / 13) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) + 12) % 13)) + 1)];
  if (((int)threadIdx.x) < 320) {
    PaddedInput_shared[(((int)threadIdx.x) + 1344)] = Input[((((((((int)blockIdx.x) / 7) * 25088) + (((((int)threadIdx.x) + 1344) / 13) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) + 5) % 13)) + 1)];
  }
  if (((int)threadIdx.x) < 128) {
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)blockIdx.x) / 7) * 1152) + (((int)threadIdx.x) * 9)) + 5)];
  }
  __syncthreads();
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((int)threadIdx.x) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2))] * kernel_shared[((((int)threadIdx.x) / 7) * 2)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2)) + 13)] * kernel_shared[(((((int)threadIdx.x) / 7) * 2) + 1)]));
  __syncthreads();
  PaddedInput_shared[((int)threadIdx.x)] = ((1 <= (((int)threadIdx.x) % 13)) ? Input[((((((((int)blockIdx.x) / 7) * 25088) + ((((int)threadIdx.x) / 13) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 13)) + 13)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 448)] = ((1 <= ((((int)threadIdx.x) + 6) % 13)) ? Input[((((((((int)blockIdx.x) / 7) * 25088) + (((((int)threadIdx.x) + 448) / 13) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) + 6) % 13)) + 13)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 896)] = ((1 <= ((((int)threadIdx.x) + 12) % 13)) ? Input[((((((((int)blockIdx.x) / 7) * 25088) + (((((int)threadIdx.x) + 896) / 13) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) + 12) % 13)) + 13)] : 0.000000e+00f);
  if (((int)threadIdx.x) < 320) {
    PaddedInput_shared[(((int)threadIdx.x) + 1344)] = ((1 <= ((((int)threadIdx.x) + 5) % 13)) ? Input[((((((((int)blockIdx.x) / 7) * 25088) + (((((int)threadIdx.x) + 1344) / 13) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) + 5) % 13)) + 13)] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 128) {
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)blockIdx.x) / 7) * 1152) + (((int)threadIdx.x) * 9)) + 6)];
  }
  __syncthreads();
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((int)threadIdx.x) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2))] * kernel_shared[((((int)threadIdx.x) / 7) * 2)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2)) + 13)] * kernel_shared[(((((int)threadIdx.x) / 7) * 2) + 1)]));
  __syncthreads();
  PaddedInput_shared[((int)threadIdx.x)] = Input[((((((((int)blockIdx.x) / 7) * 25088) + ((((int)threadIdx.x) / 13) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 13)) + 14)];
  PaddedInput_shared[(((int)threadIdx.x) + 448)] = Input[((((((((int)blockIdx.x) / 7) * 25088) + (((((int)threadIdx.x) + 448) / 13) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) + 6) % 13)) + 14)];
  PaddedInput_shared[(((int)threadIdx.x) + 896)] = Input[((((((((int)blockIdx.x) / 7) * 25088) + (((((int)threadIdx.x) + 896) / 13) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) + 12) % 13)) + 14)];
  if (((int)threadIdx.x) < 320) {
    PaddedInput_shared[(((int)threadIdx.x) + 1344)] = Input[((((((((int)blockIdx.x) / 7) * 25088) + (((((int)threadIdx.x) + 1344) / 13) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) + 5) % 13)) + 14)];
  }
  if (((int)threadIdx.x) < 128) {
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)blockIdx.x) / 7) * 1152) + (((int)threadIdx.x) * 9)) + 7)];
  }
  __syncthreads();
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((int)threadIdx.x) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2))] * kernel_shared[((((int)threadIdx.x) / 7) * 2)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2)) + 13)] * kernel_shared[(((((int)threadIdx.x) / 7) * 2) + 1)]));
  __syncthreads();
  PaddedInput_shared[((int)threadIdx.x)] = Input[((((((((int)blockIdx.x) / 7) * 25088) + ((((int)threadIdx.x) / 13) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 13)) + 15)];
  PaddedInput_shared[(((int)threadIdx.x) + 448)] = Input[((((((((int)blockIdx.x) / 7) * 25088) + (((((int)threadIdx.x) + 448) / 13) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) + 6) % 13)) + 15)];
  PaddedInput_shared[(((int)threadIdx.x) + 896)] = Input[((((((((int)blockIdx.x) / 7) * 25088) + (((((int)threadIdx.x) + 896) / 13) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) + 12) % 13)) + 15)];
  if (((int)threadIdx.x) < 320) {
    PaddedInput_shared[(((int)threadIdx.x) + 1344)] = Input[((((((((int)blockIdx.x) / 7) * 25088) + (((((int)threadIdx.x) + 1344) / 13) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) + 5) % 13)) + 15)];
  }
  if (((int)threadIdx.x) < 128) {
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)blockIdx.x) / 7) * 1152) + (((int)threadIdx.x) * 9)) + 8)];
  }
  __syncthreads();
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((int)threadIdx.x) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2))] * kernel_shared[((((int)threadIdx.x) / 7) * 2)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2)) + 13)] * kernel_shared[(((((int)threadIdx.x) / 7) * 2) + 1)]));
  for (int i1_inner = 0; i1_inner < 2; ++i1_inner) {
    compute[((((((((int)blockIdx.x) / 7) * 6272) + ((((int)threadIdx.x) / 7) * 98)) + (i1_inner * 49)) + ((((int)blockIdx.x) % 7) * 7)) + (((int)threadIdx.x) % 7))] = max(DepthwiseConv2d[i1_inner], 0.000000e+00f);
  }
}


