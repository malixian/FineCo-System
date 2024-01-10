
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
  float DepthwiseConv2d[8];
  __shared__ float PaddedInput_shared[7168];
  __shared__ float kernel_shared[64];
  DepthwiseConv2d[0] = 0.000000e+00f;
  DepthwiseConv2d[1] = 0.000000e+00f;
  DepthwiseConv2d[2] = 0.000000e+00f;
  DepthwiseConv2d[3] = 0.000000e+00f;
  DepthwiseConv2d[4] = 0.000000e+00f;
  DepthwiseConv2d[5] = 0.000000e+00f;
  DepthwiseConv2d[6] = 0.000000e+00f;
  DepthwiseConv2d[7] = 0.000000e+00f;
  PaddedInput_shared[((int)threadIdx.x)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28))) && (1 <= (((int)threadIdx.x) % 28))) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) - 29)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 896)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28))) && (1 <= (((int)threadIdx.x) % 28))) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 6243)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 1792)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28))) && (1 <= (((int)threadIdx.x) % 28))) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 12515)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 2688)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28))) && (1 <= (((int)threadIdx.x) % 28))) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 18787)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 3584)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28))) && (1 <= (((int)threadIdx.x) % 28))) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 25059)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 4480)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28))) && (1 <= (((int)threadIdx.x) % 28))) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 31331)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 5376)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28))) && (1 <= (((int)threadIdx.x) % 28))) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 37603)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 6272)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28))) && (1 <= (((int)threadIdx.x) % 28))) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 43875)] : 0.000000e+00f);
  if (((int)threadIdx.x) < 64) {
    kernel_shared[((int)threadIdx.x)] = kernel[(((((int)blockIdx.x) / 7) * 576) + (((int)threadIdx.x) * 9))];
  }
  __syncthreads();
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112))] * kernel_shared[((((int)threadIdx.x) / 112) * 8)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 112)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 1)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 224)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 2)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 336)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 3)]));
  DepthwiseConv2d[4] = (DepthwiseConv2d[4] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 448)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 4)]));
  DepthwiseConv2d[5] = (DepthwiseConv2d[5] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 560)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 5)]));
  DepthwiseConv2d[6] = (DepthwiseConv2d[6] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 672)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 6)]));
  DepthwiseConv2d[7] = (DepthwiseConv2d[7] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 784)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 7)]));
  __syncthreads();
  PaddedInput_shared[((int)threadIdx.x)] = ((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28))) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) - 28)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 896)] = ((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28))) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 6244)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 1792)] = ((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28))) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 12516)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 2688)] = ((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28))) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 18788)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 3584)] = ((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28))) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 25060)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 4480)] = ((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28))) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 31332)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 5376)] = ((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28))) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 37604)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 6272)] = ((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28))) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 43876)] : 0.000000e+00f);
  if (((int)threadIdx.x) < 64) {
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)blockIdx.x) / 7) * 576) + (((int)threadIdx.x) * 9)) + 1)];
  }
  __syncthreads();
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112))] * kernel_shared[((((int)threadIdx.x) / 112) * 8)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 112)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 1)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 224)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 2)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 336)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 3)]));
  DepthwiseConv2d[4] = (DepthwiseConv2d[4] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 448)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 4)]));
  DepthwiseConv2d[5] = (DepthwiseConv2d[5] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 560)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 5)]));
  DepthwiseConv2d[6] = (DepthwiseConv2d[6] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 672)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 6)]));
  DepthwiseConv2d[7] = (DepthwiseConv2d[7] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 784)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 7)]));
  __syncthreads();
  PaddedInput_shared[((int)threadIdx.x)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28))) && ((((int)threadIdx.x) % 28) < 27)) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) - 27)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 896)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28))) && ((((int)threadIdx.x) % 28) < 27)) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 6245)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 1792)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28))) && ((((int)threadIdx.x) % 28) < 27)) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 12517)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 2688)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28))) && ((((int)threadIdx.x) % 28) < 27)) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 18789)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 3584)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28))) && ((((int)threadIdx.x) % 28) < 27)) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 25061)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 4480)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28))) && ((((int)threadIdx.x) % 28) < 27)) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 31333)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 5376)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28))) && ((((int)threadIdx.x) % 28) < 27)) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 37605)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 6272)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28))) && ((((int)threadIdx.x) % 28) < 27)) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 43877)] : 0.000000e+00f);
  if (((int)threadIdx.x) < 64) {
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)blockIdx.x) / 7) * 576) + (((int)threadIdx.x) * 9)) + 2)];
  }
  __syncthreads();
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112))] * kernel_shared[((((int)threadIdx.x) / 112) * 8)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 112)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 1)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 224)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 2)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 336)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 3)]));
  DepthwiseConv2d[4] = (DepthwiseConv2d[4] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 448)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 4)]));
  DepthwiseConv2d[5] = (DepthwiseConv2d[5] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 560)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 5)]));
  DepthwiseConv2d[6] = (DepthwiseConv2d[6] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 672)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 6)]));
  DepthwiseConv2d[7] = (DepthwiseConv2d[7] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 784)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 7)]));
  __syncthreads();
  PaddedInput_shared[((int)threadIdx.x)] = ((1 <= (((int)threadIdx.x) % 28)) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) - 1)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 896)] = ((1 <= (((int)threadIdx.x) % 28)) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 6271)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 1792)] = ((1 <= (((int)threadIdx.x) % 28)) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 12543)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 2688)] = ((1 <= (((int)threadIdx.x) % 28)) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 18815)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 3584)] = ((1 <= (((int)threadIdx.x) % 28)) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 25087)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 4480)] = ((1 <= (((int)threadIdx.x) % 28)) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 31359)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 5376)] = ((1 <= (((int)threadIdx.x) % 28)) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 37631)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 6272)] = ((1 <= (((int)threadIdx.x) % 28)) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 43903)] : 0.000000e+00f);
  if (((int)threadIdx.x) < 64) {
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)blockIdx.x) / 7) * 576) + (((int)threadIdx.x) * 9)) + 3)];
  }
  __syncthreads();
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112))] * kernel_shared[((((int)threadIdx.x) / 112) * 8)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 112)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 1)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 224)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 2)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 336)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 3)]));
  DepthwiseConv2d[4] = (DepthwiseConv2d[4] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 448)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 4)]));
  DepthwiseConv2d[5] = (DepthwiseConv2d[5] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 560)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 5)]));
  DepthwiseConv2d[6] = (DepthwiseConv2d[6] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 672)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 6)]));
  DepthwiseConv2d[7] = (DepthwiseConv2d[7] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 784)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 7)]));
  __syncthreads();
  PaddedInput_shared[((int)threadIdx.x)] = Input[(((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112))];
  PaddedInput_shared[(((int)threadIdx.x) + 896)] = Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 6272)];
  PaddedInput_shared[(((int)threadIdx.x) + 1792)] = Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 12544)];
  PaddedInput_shared[(((int)threadIdx.x) + 2688)] = Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 18816)];
  PaddedInput_shared[(((int)threadIdx.x) + 3584)] = Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 25088)];
  PaddedInput_shared[(((int)threadIdx.x) + 4480)] = Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 31360)];
  PaddedInput_shared[(((int)threadIdx.x) + 5376)] = Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 37632)];
  PaddedInput_shared[(((int)threadIdx.x) + 6272)] = Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 43904)];
  if (((int)threadIdx.x) < 64) {
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)blockIdx.x) / 7) * 576) + (((int)threadIdx.x) * 9)) + 4)];
  }
  __syncthreads();
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112))] * kernel_shared[((((int)threadIdx.x) / 112) * 8)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 112)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 1)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 224)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 2)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 336)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 3)]));
  DepthwiseConv2d[4] = (DepthwiseConv2d[4] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 448)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 4)]));
  DepthwiseConv2d[5] = (DepthwiseConv2d[5] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 560)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 5)]));
  DepthwiseConv2d[6] = (DepthwiseConv2d[6] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 672)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 6)]));
  DepthwiseConv2d[7] = (DepthwiseConv2d[7] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 784)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 7)]));
  __syncthreads();
  PaddedInput_shared[((int)threadIdx.x)] = (((((int)threadIdx.x) % 28) < 27) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 1)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 896)] = (((((int)threadIdx.x) % 28) < 27) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 6273)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 1792)] = (((((int)threadIdx.x) % 28) < 27) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 12545)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 2688)] = (((((int)threadIdx.x) % 28) < 27) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 18817)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 3584)] = (((((int)threadIdx.x) % 28) < 27) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 25089)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 4480)] = (((((int)threadIdx.x) % 28) < 27) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 31361)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 5376)] = (((((int)threadIdx.x) % 28) < 27) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 37633)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 6272)] = (((((int)threadIdx.x) % 28) < 27) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 43905)] : 0.000000e+00f);
  if (((int)threadIdx.x) < 64) {
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)blockIdx.x) / 7) * 576) + (((int)threadIdx.x) * 9)) + 5)];
  }
  __syncthreads();
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112))] * kernel_shared[((((int)threadIdx.x) / 112) * 8)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 112)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 1)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 224)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 2)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 336)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 3)]));
  DepthwiseConv2d[4] = (DepthwiseConv2d[4] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 448)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 4)]));
  DepthwiseConv2d[5] = (DepthwiseConv2d[5] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 560)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 5)]));
  DepthwiseConv2d[6] = (DepthwiseConv2d[6] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 672)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 6)]));
  DepthwiseConv2d[7] = (DepthwiseConv2d[7] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 784)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 7)]));
  __syncthreads();
  PaddedInput_shared[((int)threadIdx.x)] = ((((((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28)) < 27) && (1 <= (((int)threadIdx.x) % 28))) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 27)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 896)] = ((((((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28)) < 27) && (1 <= (((int)threadIdx.x) % 28))) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 6299)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 1792)] = ((((((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28)) < 27) && (1 <= (((int)threadIdx.x) % 28))) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 12571)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 2688)] = ((((((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28)) < 27) && (1 <= (((int)threadIdx.x) % 28))) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 18843)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 3584)] = ((((((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28)) < 27) && (1 <= (((int)threadIdx.x) % 28))) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 25115)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 4480)] = ((((((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28)) < 27) && (1 <= (((int)threadIdx.x) % 28))) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 31387)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 5376)] = ((((((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28)) < 27) && (1 <= (((int)threadIdx.x) % 28))) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 37659)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 6272)] = ((((((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28)) < 27) && (1 <= (((int)threadIdx.x) % 28))) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 43931)] : 0.000000e+00f);
  if (((int)threadIdx.x) < 64) {
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)blockIdx.x) / 7) * 576) + (((int)threadIdx.x) * 9)) + 6)];
  }
  __syncthreads();
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112))] * kernel_shared[((((int)threadIdx.x) / 112) * 8)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 112)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 1)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 224)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 2)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 336)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 3)]));
  DepthwiseConv2d[4] = (DepthwiseConv2d[4] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 448)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 4)]));
  DepthwiseConv2d[5] = (DepthwiseConv2d[5] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 560)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 5)]));
  DepthwiseConv2d[6] = (DepthwiseConv2d[6] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 672)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 6)]));
  DepthwiseConv2d[7] = (DepthwiseConv2d[7] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 784)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 7)]));
  __syncthreads();
  PaddedInput_shared[((int)threadIdx.x)] = (((((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28)) < 27) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 28)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 896)] = (((((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28)) < 27) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 6300)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 1792)] = (((((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28)) < 27) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 12572)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 2688)] = (((((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28)) < 27) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 18844)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 3584)] = (((((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28)) < 27) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 25116)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 4480)] = (((((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28)) < 27) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 31388)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 5376)] = (((((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28)) < 27) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 37660)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 6272)] = (((((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28)) < 27) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 43932)] : 0.000000e+00f);
  if (((int)threadIdx.x) < 64) {
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)blockIdx.x) / 7) * 576) + (((int)threadIdx.x) * 9)) + 7)];
  }
  __syncthreads();
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112))] * kernel_shared[((((int)threadIdx.x) / 112) * 8)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 112)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 1)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 224)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 2)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 336)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 3)]));
  DepthwiseConv2d[4] = (DepthwiseConv2d[4] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 448)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 4)]));
  DepthwiseConv2d[5] = (DepthwiseConv2d[5] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 560)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 5)]));
  DepthwiseConv2d[6] = (DepthwiseConv2d[6] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 672)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 6)]));
  DepthwiseConv2d[7] = (DepthwiseConv2d[7] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 784)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 7)]));
  __syncthreads();
  PaddedInput_shared[((int)threadIdx.x)] = ((((((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28)) < 27) && ((((int)threadIdx.x) % 28) < 27)) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 29)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 896)] = ((((((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28)) < 27) && ((((int)threadIdx.x) % 28) < 27)) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 6301)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 1792)] = ((((((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28)) < 27) && ((((int)threadIdx.x) % 28) < 27)) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 12573)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 2688)] = ((((((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28)) < 27) && ((((int)threadIdx.x) % 28) < 27)) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 18845)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 3584)] = ((((((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28)) < 27) && ((((int)threadIdx.x) % 28) < 27)) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 25117)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 4480)] = ((((((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28)) < 27) && ((((int)threadIdx.x) % 28) < 27)) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 31389)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 5376)] = ((((((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28)) < 27) && ((((int)threadIdx.x) % 28) < 27)) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 37661)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 6272)] = ((((((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) % 112) / 28)) < 27) && ((((int)threadIdx.x) % 28) < 27)) ? Input[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112)) + 43933)] : 0.000000e+00f);
  if (((int)threadIdx.x) < 64) {
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)blockIdx.x) / 7) * 576) + (((int)threadIdx.x) * 9)) + 8)];
  }
  __syncthreads();
  DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112))] * kernel_shared[((((int)threadIdx.x) / 112) * 8)]));
  DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 112)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 1)]));
  DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 224)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 2)]));
  DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 336)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 3)]));
  DepthwiseConv2d[4] = (DepthwiseConv2d[4] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 448)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 4)]));
  DepthwiseConv2d[5] = (DepthwiseConv2d[5] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 560)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 5)]));
  DepthwiseConv2d[6] = (DepthwiseConv2d[6] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 672)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 6)]));
  DepthwiseConv2d[7] = (DepthwiseConv2d[7] + (PaddedInput_shared[((((((int)threadIdx.x) / 112) * 896) + (((int)threadIdx.x) % 112)) + 784)] * kernel_shared[(((((int)threadIdx.x) / 112) * 8) + 7)]));
  for (int i1_inner = 0; i1_inner < 8; ++i1_inner) {
    compute[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 6272)) + (i1_inner * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((int)threadIdx.x) % 112))] = max(DepthwiseConv2d[i1_inner], 0.000000e+00f);
  }
}


