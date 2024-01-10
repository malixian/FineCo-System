
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
extern "C" __global__ void __launch_bounds__(896) candidate1(float* __restrict__ Input, float* __restrict__ kernel, float* __restrict__ compute) {
  float DepthwiseConv2d[4];
  __shared__ float PaddedInput_shared[10848];
  __shared__ float kernel_shared[96];
  DepthwiseConv2d[0] = 0.000000e+00f;
  DepthwiseConv2d[1] = 0.000000e+00f;
  DepthwiseConv2d[2] = 0.000000e+00f;
  DepthwiseConv2d[3] = 0.000000e+00f;
  for (int di_outer_outer = 0; di_outer_outer < 3; ++di_outer_outer) {
    __syncthreads();
    PaddedInput_shared[((int)threadIdx.x)] = (((1 <= ((((((int)blockIdx.x) % 28) * 4) + ((((int)threadIdx.x) % 339) / 113)) + di_outer_outer)) && (1 <= (((int)threadIdx.x) % 113))) ? Input[((((((((((int)blockIdx.x) / 28) * 401408) + ((((int)threadIdx.x) / 339) * 12544)) + ((((int)blockIdx.x) % 28) * 448)) + (((((int)threadIdx.x) % 339) / 113) * 112)) + (di_outer_outer * 112)) + (((int)threadIdx.x) % 113)) - 113)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 896)] = (((1 <= ((((((int)blockIdx.x) % 28) * 4) + (((((int)threadIdx.x) + 218) % 339) / 113)) + di_outer_outer)) && (1 <= ((((int)threadIdx.x) + 105) % 113))) ? Input[((((((((((int)blockIdx.x) / 28) * 401408) + (((((int)threadIdx.x) + 896) / 339) * 12544)) + ((((int)blockIdx.x) % 28) * 448)) + ((((((int)threadIdx.x) + 218) % 339) / 113) * 112)) + (di_outer_outer * 112)) + ((((int)threadIdx.x) + 105) % 113)) - 113)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 1792)] = (((1 <= ((((((int)blockIdx.x) % 28) * 4) + (((((int)threadIdx.x) + 97) % 339) / 113)) + di_outer_outer)) && (1 <= ((((int)threadIdx.x) + 97) % 113))) ? Input[((((((((((int)blockIdx.x) / 28) * 401408) + (((((int)threadIdx.x) + 1792) / 339) * 12544)) + ((((int)blockIdx.x) % 28) * 448)) + ((((((int)threadIdx.x) + 97) % 339) / 113) * 112)) + (di_outer_outer * 112)) + ((((int)threadIdx.x) + 97) % 113)) - 113)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 2688)] = (((1 <= ((((((int)blockIdx.x) % 28) * 4) + (((((int)threadIdx.x) + 315) % 339) / 113)) + di_outer_outer)) && (1 <= ((((int)threadIdx.x) + 89) % 113))) ? Input[((((((((((int)blockIdx.x) / 28) * 401408) + (((((int)threadIdx.x) + 2688) / 339) * 12544)) + ((((int)blockIdx.x) % 28) * 448)) + ((((((int)threadIdx.x) + 315) % 339) / 113) * 112)) + (di_outer_outer * 112)) + ((((int)threadIdx.x) + 89) % 113)) - 113)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 3584)] = (((1 <= ((((((int)blockIdx.x) % 28) * 4) + (((((int)threadIdx.x) + 194) % 339) / 113)) + di_outer_outer)) && (1 <= ((((int)threadIdx.x) + 81) % 113))) ? Input[((((((((((int)blockIdx.x) / 28) * 401408) + (((((int)threadIdx.x) + 3584) / 339) * 12544)) + ((((int)blockIdx.x) % 28) * 448)) + ((((((int)threadIdx.x) + 194) % 339) / 113) * 112)) + (di_outer_outer * 112)) + ((((int)threadIdx.x) + 81) % 113)) - 113)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 4480)] = (((1 <= ((((((int)blockIdx.x) % 28) * 4) + (((((int)threadIdx.x) + 73) % 339) / 113)) + di_outer_outer)) && (1 <= ((((int)threadIdx.x) + 73) % 113))) ? Input[((((((((((int)blockIdx.x) / 28) * 401408) + (((((int)threadIdx.x) + 4480) / 339) * 12544)) + ((((int)blockIdx.x) % 28) * 448)) + ((((((int)threadIdx.x) + 73) % 339) / 113) * 112)) + (di_outer_outer * 112)) + ((((int)threadIdx.x) + 73) % 113)) - 113)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 5376)] = (((1 <= ((((((int)blockIdx.x) % 28) * 4) + (((((int)threadIdx.x) + 291) % 339) / 113)) + di_outer_outer)) && (1 <= ((((int)threadIdx.x) + 65) % 113))) ? Input[((((((((((int)blockIdx.x) / 28) * 401408) + (((((int)threadIdx.x) + 5376) / 339) * 12544)) + ((((int)blockIdx.x) % 28) * 448)) + ((((((int)threadIdx.x) + 291) % 339) / 113) * 112)) + (di_outer_outer * 112)) + ((((int)threadIdx.x) + 65) % 113)) - 113)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 6272)] = (((1 <= ((((((int)blockIdx.x) % 28) * 4) + (((((int)threadIdx.x) + 170) % 339) / 113)) + di_outer_outer)) && (1 <= ((((int)threadIdx.x) + 57) % 113))) ? Input[((((((((((int)blockIdx.x) / 28) * 401408) + (((((int)threadIdx.x) + 6272) / 339) * 12544)) + ((((int)blockIdx.x) % 28) * 448)) + ((((((int)threadIdx.x) + 170) % 339) / 113) * 112)) + (di_outer_outer * 112)) + ((((int)threadIdx.x) + 57) % 113)) - 113)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 7168)] = (((1 <= ((((((int)blockIdx.x) % 28) * 4) + (((((int)threadIdx.x) + 49) % 339) / 113)) + di_outer_outer)) && (1 <= ((((int)threadIdx.x) + 49) % 113))) ? Input[((((((((((int)blockIdx.x) / 28) * 401408) + (((((int)threadIdx.x) + 7168) / 339) * 12544)) + ((((int)blockIdx.x) % 28) * 448)) + ((((((int)threadIdx.x) + 49) % 339) / 113) * 112)) + (di_outer_outer * 112)) + ((((int)threadIdx.x) + 49) % 113)) - 113)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 8064)] = (((1 <= ((((((int)blockIdx.x) % 28) * 4) + (((((int)threadIdx.x) + 267) % 339) / 113)) + di_outer_outer)) && (1 <= ((((int)threadIdx.x) + 41) % 113))) ? Input[((((((((((int)blockIdx.x) / 28) * 401408) + (((((int)threadIdx.x) + 8064) / 339) * 12544)) + ((((int)blockIdx.x) % 28) * 448)) + ((((((int)threadIdx.x) + 267) % 339) / 113) * 112)) + (di_outer_outer * 112)) + ((((int)threadIdx.x) + 41) % 113)) - 113)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 8960)] = (((1 <= ((((((int)blockIdx.x) % 28) * 4) + (((((int)threadIdx.x) + 146) % 339) / 113)) + di_outer_outer)) && (1 <= ((((int)threadIdx.x) + 33) % 113))) ? Input[((((((((((int)blockIdx.x) / 28) * 401408) + (((((int)threadIdx.x) + 8960) / 339) * 12544)) + ((((int)blockIdx.x) % 28) * 448)) + ((((((int)threadIdx.x) + 146) % 339) / 113) * 112)) + (di_outer_outer * 112)) + ((((int)threadIdx.x) + 33) % 113)) - 113)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 9856)] = (((1 <= ((((((int)blockIdx.x) % 28) * 4) + (((((int)threadIdx.x) + 25) % 339) / 113)) + di_outer_outer)) && (1 <= ((((int)threadIdx.x) + 25) % 113))) ? Input[((((((((((int)blockIdx.x) / 28) * 401408) + (((((int)threadIdx.x) + 9856) / 339) * 12544)) + ((((int)blockIdx.x) % 28) * 448)) + ((((((int)threadIdx.x) + 25) % 339) / 113) * 112)) + (di_outer_outer * 112)) + ((((int)threadIdx.x) + 25) % 113)) - 113)] : 0.000000e+00f);
    if (((int)threadIdx.x) < 96) {
      PaddedInput_shared[(((int)threadIdx.x) + 10752)] = Input[((((((((((int)blockIdx.x) / 28) * 401408) + (((((int)threadIdx.x) + 10752) / 339) * 12544)) + ((((int)blockIdx.x) % 28) * 448)) + ((((((int)threadIdx.x) + 243) % 339) / 113) * 112)) + (di_outer_outer * 112)) + (((int)threadIdx.x) + 17)) - 113)];
    }
    if (((int)threadIdx.x) < 96) {
      kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) / 28) * 288) + ((((int)threadIdx.x) / 3) * 9)) + (di_outer_outer * 3)) + (((int)threadIdx.x) % 3))];
    }
    __syncthreads();
    DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 56) * 339) + (((((int)threadIdx.x) % 56) / 28) * 226)) + ((((int)threadIdx.x) % 28) * 2))] * kernel_shared[((((int)threadIdx.x) / 56) * 3)]));
    DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 339) + (((((int)threadIdx.x) % 56) / 28) * 226)) + ((((int)threadIdx.x) % 28) * 2)) + 56)] * kernel_shared[((((int)threadIdx.x) / 56) * 3)]));
    DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 339) + (((((int)threadIdx.x) % 56) / 28) * 226)) + ((((int)threadIdx.x) % 28) * 2)) + 5424)] * kernel_shared[(((((int)threadIdx.x) / 56) * 3) + 48)]));
    DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 339) + (((((int)threadIdx.x) % 56) / 28) * 226)) + ((((int)threadIdx.x) % 28) * 2)) + 5480)] * kernel_shared[(((((int)threadIdx.x) / 56) * 3) + 48)]));
    DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 339) + (((((int)threadIdx.x) % 56) / 28) * 226)) + ((((int)threadIdx.x) % 28) * 2)) + 1)] * kernel_shared[(((((int)threadIdx.x) / 56) * 3) + 1)]));
    DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 339) + (((((int)threadIdx.x) % 56) / 28) * 226)) + ((((int)threadIdx.x) % 28) * 2)) + 57)] * kernel_shared[(((((int)threadIdx.x) / 56) * 3) + 1)]));
    DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 339) + (((((int)threadIdx.x) % 56) / 28) * 226)) + ((((int)threadIdx.x) % 28) * 2)) + 5425)] * kernel_shared[(((((int)threadIdx.x) / 56) * 3) + 49)]));
    DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 339) + (((((int)threadIdx.x) % 56) / 28) * 226)) + ((((int)threadIdx.x) % 28) * 2)) + 5481)] * kernel_shared[(((((int)threadIdx.x) / 56) * 3) + 49)]));
    DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 339) + (((((int)threadIdx.x) % 56) / 28) * 226)) + ((((int)threadIdx.x) % 28) * 2)) + 2)] * kernel_shared[(((((int)threadIdx.x) / 56) * 3) + 2)]));
    DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 339) + (((((int)threadIdx.x) % 56) / 28) * 226)) + ((((int)threadIdx.x) % 28) * 2)) + 58)] * kernel_shared[(((((int)threadIdx.x) / 56) * 3) + 2)]));
    DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 339) + (((((int)threadIdx.x) % 56) / 28) * 226)) + ((((int)threadIdx.x) % 28) * 2)) + 5426)] * kernel_shared[(((((int)threadIdx.x) / 56) * 3) + 50)]));
    DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[(((((((int)threadIdx.x) / 56) * 339) + (((((int)threadIdx.x) % 56) / 28) * 226)) + ((((int)threadIdx.x) % 28) * 2)) + 5482)] * kernel_shared[(((((int)threadIdx.x) / 56) * 3) + 50)]));
  }
  compute[((((((((int)blockIdx.x) / 28) * 100352) + ((((int)threadIdx.x) / 56) * 3136)) + ((((int)blockIdx.x) % 28) * 112)) + (((((int)threadIdx.x) % 56) / 28) * 56)) + (((int)threadIdx.x) % 28))] = max(DepthwiseConv2d[0], 0.000000e+00f);
  compute[(((((((((int)blockIdx.x) / 28) * 100352) + ((((int)threadIdx.x) / 56) * 3136)) + ((((int)blockIdx.x) % 28) * 112)) + (((((int)threadIdx.x) % 56) / 28) * 56)) + (((int)threadIdx.x) % 28)) + 28)] = max(DepthwiseConv2d[1], 0.000000e+00f);
  compute[(((((((((int)blockIdx.x) / 28) * 100352) + ((((int)threadIdx.x) / 56) * 3136)) + ((((int)blockIdx.x) % 28) * 112)) + (((((int)threadIdx.x) % 56) / 28) * 56)) + (((int)threadIdx.x) % 28)) + 50176)] = max(DepthwiseConv2d[2], 0.000000e+00f);
  compute[(((((((((int)blockIdx.x) / 28) * 100352) + ((((int)threadIdx.x) / 56) * 3136)) + ((((int)blockIdx.x) % 28) * 112)) + (((((int)threadIdx.x) % 56) / 28) * 56)) + (((int)threadIdx.x) % 28)) + 50204)] = max(DepthwiseConv2d[3], 0.000000e+00f);
}


