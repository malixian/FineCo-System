
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
extern "C" __global__ void __launch_bounds__(392) candidate0(float* __restrict__ Input, float* __restrict__ kernel, float* __restrict__ compute) {
  float DepthwiseConv2d[2];
  __shared__ float PaddedInput_shared[1296];
  __shared__ float kernel_shared[144];
  DepthwiseConv2d[0] = 0.000000e+00f;
  DepthwiseConv2d[1] = 0.000000e+00f;
  PaddedInput_shared[(((int)threadIdx.x) * 3)] = ((((3 <= (((int)threadIdx.x) % 27)) && ((((int)threadIdx.x) % 27) < 24)) && (1 <= (((int)threadIdx.x) % 3))) ? Input[(((((((int)blockIdx.x) * 784) + ((((int)threadIdx.x) / 27) * 49)) + (((((int)threadIdx.x) % 27) / 3) * 7)) + ((((int)threadIdx.x) % 3) * 3)) - 8)] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) * 3) + 1)] = (((3 <= (((int)threadIdx.x) % 27)) && ((((int)threadIdx.x) % 27) < 24)) ? Input[(((((((int)blockIdx.x) * 784) + ((((int)threadIdx.x) / 27) * 49)) + (((((int)threadIdx.x) % 27) / 3) * 7)) + ((((int)threadIdx.x) % 3) * 3)) - 7)] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) * 3) + 2)] = ((((3 <= (((int)threadIdx.x) % 27)) && ((((int)threadIdx.x) % 27) < 24)) && ((((int)threadIdx.x) % 3) < 2)) ? Input[(((((((int)blockIdx.x) * 784) + ((((int)threadIdx.x) / 27) * 49)) + (((((int)threadIdx.x) % 27) / 3) * 7)) + ((((int)threadIdx.x) % 3) * 3)) - 6)] : 0.000000e+00f);
  if (((int)threadIdx.x) < 40) {
    PaddedInput_shared[((((int)threadIdx.x) * 3) + 1176)] = (((((3 <= ((((int)threadIdx.x) + 14) % 27)) && (((((int)threadIdx.x) + 14) % 27) < 24)) && (1 <= (((((int)threadIdx.x) * 3) + 6) % 9))) && ((((((int)threadIdx.x) * 3) + 6) % 9) < 8)) ? Input[(((((((int)blockIdx.x) * 784) + (((((int)threadIdx.x) + 392) / 27) * 49)) + ((((((int)threadIdx.x) + 14) % 27) / 3) * 7)) + (((((int)threadIdx.x) * 3) + 6) % 9)) - 8)] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) * 3) + 1177)] = (((((3 <= ((((int)threadIdx.x) + 14) % 27)) && (((((int)threadIdx.x) + 14) % 27) < 24)) && (1 <= (((((int)threadIdx.x) * 3) + 7) % 9))) && ((((((int)threadIdx.x) * 3) + 7) % 9) < 8)) ? Input[(((((((int)blockIdx.x) * 784) + (((((int)threadIdx.x) + 392) / 27) * 49)) + ((((((int)threadIdx.x) + 14) % 27) / 3) * 7)) + (((((int)threadIdx.x) * 3) + 7) % 9)) - 8)] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) * 3) + 1178)] = (((((3 <= ((((int)threadIdx.x) + 14) % 27)) && (((((int)threadIdx.x) + 14) % 27) < 24)) && (1 <= (((((int)threadIdx.x) * 3) + 8) % 9))) && ((((((int)threadIdx.x) * 3) + 8) % 9) < 8)) ? Input[(((((((int)blockIdx.x) * 784) + (((((int)threadIdx.x) + 392) / 27) * 49)) + ((((((int)threadIdx.x) + 14) % 27) / 3) * 7)) + (((((int)threadIdx.x) * 3) + 8) % 9)) - 8)] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 144) {
    kernel_shared[((int)threadIdx.x)] = kernel[((((int)blockIdx.x) * 144) + ((int)threadIdx.x))];
  }
  __syncthreads();
  for (int di_inner = 0; di_inner < 3; ++di_inner) {
    DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((((int)threadIdx.x) / 49) * 81) + (((((int)threadIdx.x) % 49) / 7) * 9)) + (di_inner * 9)) + (((int)threadIdx.x) % 7))] * kernel_shared[(((((int)threadIdx.x) / 49) * 9) + (di_inner * 3))]));
    DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((((int)threadIdx.x) / 49) * 81) + (((((int)threadIdx.x) % 49) / 7) * 9)) + (di_inner * 9)) + (((int)threadIdx.x) % 7)) + 648)] * kernel_shared[((((((int)threadIdx.x) / 49) * 9) + (di_inner * 3)) + 72)]));
    DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((((int)threadIdx.x) / 49) * 81) + (((((int)threadIdx.x) % 49) / 7) * 9)) + (di_inner * 9)) + (((int)threadIdx.x) % 7)) + 1)] * kernel_shared[((((((int)threadIdx.x) / 49) * 9) + (di_inner * 3)) + 1)]));
    DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((((int)threadIdx.x) / 49) * 81) + (((((int)threadIdx.x) % 49) / 7) * 9)) + (di_inner * 9)) + (((int)threadIdx.x) % 7)) + 649)] * kernel_shared[((((((int)threadIdx.x) / 49) * 9) + (di_inner * 3)) + 73)]));
    DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((((int)threadIdx.x) / 49) * 81) + (((((int)threadIdx.x) % 49) / 7) * 9)) + (di_inner * 9)) + (((int)threadIdx.x) % 7)) + 2)] * kernel_shared[((((((int)threadIdx.x) / 49) * 9) + (di_inner * 3)) + 2)]));
    DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((((int)threadIdx.x) / 49) * 81) + (((((int)threadIdx.x) % 49) / 7) * 9)) + (di_inner * 9)) + (((int)threadIdx.x) % 7)) + 650)] * kernel_shared[((((((int)threadIdx.x) / 49) * 9) + (di_inner * 3)) + 74)]));
  }
  compute[((((int)blockIdx.x) * 784) + ((int)threadIdx.x))] = max(DepthwiseConv2d[0], 0.000000e+00f);
  compute[(((((int)blockIdx.x) * 784) + ((int)threadIdx.x)) + 392)] = max(DepthwiseConv2d[1], 0.000000e+00f);
}


