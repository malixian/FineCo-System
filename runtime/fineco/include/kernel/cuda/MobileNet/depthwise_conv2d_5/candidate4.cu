
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
extern "C" __global__ void __launch_bounds__(98) candidate4(float* __restrict__ Input, float* __restrict__ kernel, float* __restrict__ compute) {
  float DepthwiseConv2d[32];
  __shared__ float PaddedInput_shared[3360];
  __shared__ float kernel_shared[12];
  DepthwiseConv2d[0] = 0.000000e+00f;
  DepthwiseConv2d[8] = 0.000000e+00f;
  DepthwiseConv2d[16] = 0.000000e+00f;
  DepthwiseConv2d[24] = 0.000000e+00f;
  DepthwiseConv2d[1] = 0.000000e+00f;
  DepthwiseConv2d[9] = 0.000000e+00f;
  DepthwiseConv2d[17] = 0.000000e+00f;
  DepthwiseConv2d[25] = 0.000000e+00f;
  DepthwiseConv2d[2] = 0.000000e+00f;
  DepthwiseConv2d[10] = 0.000000e+00f;
  DepthwiseConv2d[18] = 0.000000e+00f;
  DepthwiseConv2d[26] = 0.000000e+00f;
  DepthwiseConv2d[3] = 0.000000e+00f;
  DepthwiseConv2d[11] = 0.000000e+00f;
  DepthwiseConv2d[19] = 0.000000e+00f;
  DepthwiseConv2d[27] = 0.000000e+00f;
  DepthwiseConv2d[4] = 0.000000e+00f;
  DepthwiseConv2d[12] = 0.000000e+00f;
  DepthwiseConv2d[20] = 0.000000e+00f;
  DepthwiseConv2d[28] = 0.000000e+00f;
  DepthwiseConv2d[5] = 0.000000e+00f;
  DepthwiseConv2d[13] = 0.000000e+00f;
  DepthwiseConv2d[21] = 0.000000e+00f;
  DepthwiseConv2d[29] = 0.000000e+00f;
  DepthwiseConv2d[6] = 0.000000e+00f;
  DepthwiseConv2d[14] = 0.000000e+00f;
  DepthwiseConv2d[22] = 0.000000e+00f;
  DepthwiseConv2d[30] = 0.000000e+00f;
  DepthwiseConv2d[7] = 0.000000e+00f;
  DepthwiseConv2d[15] = 0.000000e+00f;
  DepthwiseConv2d[23] = 0.000000e+00f;
  DepthwiseConv2d[31] = 0.000000e+00f;
  for (int dj_outer_outer = 0; dj_outer_outer < 3; ++dj_outer_outer) {
    __syncthreads();
    PaddedInput_shared[((int)threadIdx.x)] = ((((28 <= ((int)threadIdx.x)) && (1 <= (dj_outer_outer + (((int)threadIdx.x) % 28)))) && ((dj_outer_outer + (((int)threadIdx.x) % 28)) < 29)) ? Input[((((((int)blockIdx.x) * 3136) + dj_outer_outer) + ((int)threadIdx.x)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 98)] = (((1 <= (dj_outer_outer + ((((int)threadIdx.x) + 14) % 28))) && ((dj_outer_outer + ((((int)threadIdx.x) + 14) % 28)) < 29)) ? Input[((((((int)blockIdx.x) * 3136) + dj_outer_outer) + ((int)threadIdx.x)) + 69)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 196)] = (((1 <= (dj_outer_outer + (((int)threadIdx.x) % 28))) && ((dj_outer_outer + (((int)threadIdx.x) % 28)) < 29)) ? Input[((((((int)blockIdx.x) * 3136) + dj_outer_outer) + ((int)threadIdx.x)) + 167)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 294)] = (((1 <= (dj_outer_outer + ((((int)threadIdx.x) + 14) % 28))) && ((dj_outer_outer + ((((int)threadIdx.x) + 14) % 28)) < 29)) ? Input[((((((int)blockIdx.x) * 3136) + dj_outer_outer) + ((int)threadIdx.x)) + 265)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 392)] = (((1 <= (dj_outer_outer + (((int)threadIdx.x) % 28))) && ((dj_outer_outer + (((int)threadIdx.x) % 28)) < 29)) ? Input[((((((int)blockIdx.x) * 3136) + dj_outer_outer) + ((int)threadIdx.x)) + 363)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 490)] = (((1 <= (dj_outer_outer + ((((int)threadIdx.x) + 14) % 28))) && ((dj_outer_outer + ((((int)threadIdx.x) + 14) % 28)) < 29)) ? Input[((((((int)blockIdx.x) * 3136) + dj_outer_outer) + ((int)threadIdx.x)) + 461)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 588)] = (((1 <= (dj_outer_outer + (((int)threadIdx.x) % 28))) && ((dj_outer_outer + (((int)threadIdx.x) % 28)) < 29)) ? Input[((((((int)blockIdx.x) * 3136) + dj_outer_outer) + ((int)threadIdx.x)) + 559)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 686)] = (((1 <= (dj_outer_outer + ((((int)threadIdx.x) + 14) % 28))) && ((dj_outer_outer + ((((int)threadIdx.x) + 14) % 28)) < 29)) ? Input[((((((int)blockIdx.x) * 3136) + dj_outer_outer) + ((int)threadIdx.x)) + 657)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 784)] = (((((1 <= (((((int)threadIdx.x) / 28) + 28) % 30)) && (((((int)threadIdx.x) + 784) % 840) < 812)) && (1 <= (dj_outer_outer + (((int)threadIdx.x) % 28)))) && ((dj_outer_outer + (((int)threadIdx.x) % 28)) < 29)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 784) / 840) * 784)) + dj_outer_outer) + ((((int)threadIdx.x) + 784) % 840)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 882)] = (((1 <= (dj_outer_outer + ((((int)threadIdx.x) + 14) % 28))) && ((dj_outer_outer + ((((int)threadIdx.x) + 14) % 28)) < 29)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 882) / 840) * 784)) + dj_outer_outer) + (((int)threadIdx.x) + 42)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 980)] = (((1 <= (dj_outer_outer + (((int)threadIdx.x) % 28))) && ((dj_outer_outer + (((int)threadIdx.x) % 28)) < 29)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 980) / 840) * 784)) + dj_outer_outer) + (((int)threadIdx.x) + 140)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 1078)] = (((1 <= (dj_outer_outer + ((((int)threadIdx.x) + 14) % 28))) && ((dj_outer_outer + ((((int)threadIdx.x) + 14) % 28)) < 29)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 1078) / 840) * 784)) + dj_outer_outer) + (((int)threadIdx.x) + 238)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 1176)] = (((1 <= (dj_outer_outer + (((int)threadIdx.x) % 28))) && ((dj_outer_outer + (((int)threadIdx.x) % 28)) < 29)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 1176) / 840) * 784)) + dj_outer_outer) + (((int)threadIdx.x) + 336)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 1274)] = (((1 <= (dj_outer_outer + ((((int)threadIdx.x) + 14) % 28))) && ((dj_outer_outer + ((((int)threadIdx.x) + 14) % 28)) < 29)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 1274) / 840) * 784)) + dj_outer_outer) + (((int)threadIdx.x) + 434)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 1372)] = (((1 <= (dj_outer_outer + (((int)threadIdx.x) % 28))) && ((dj_outer_outer + (((int)threadIdx.x) % 28)) < 29)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 1372) / 840) * 784)) + dj_outer_outer) + (((int)threadIdx.x) + 532)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 1470)] = (((1 <= (dj_outer_outer + ((((int)threadIdx.x) + 14) % 28))) && ((dj_outer_outer + ((((int)threadIdx.x) + 14) % 28)) < 29)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 1470) / 840) * 784)) + dj_outer_outer) + (((int)threadIdx.x) + 630)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 1568)] = ((((((int)threadIdx.x) < 84) && (1 <= (dj_outer_outer + (((int)threadIdx.x) % 28)))) && ((dj_outer_outer + (((int)threadIdx.x) % 28)) < 29)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 1568) / 840) * 784)) + dj_outer_outer) + (((int)threadIdx.x) + 728)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 1666)] = (((((2 <= (((((int)threadIdx.x) / 14) + 59) % 60)) && (((((int)threadIdx.x) + 826) % 840) < 812)) && (1 <= (dj_outer_outer + ((((int)threadIdx.x) + 14) % 28)))) && ((dj_outer_outer + ((((int)threadIdx.x) + 14) % 28)) < 29)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 1666) / 840) * 784)) + dj_outer_outer) + ((((int)threadIdx.x) + 826) % 840)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 1764)] = (((1 <= (dj_outer_outer + (((int)threadIdx.x) % 28))) && ((dj_outer_outer + (((int)threadIdx.x) % 28)) < 29)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 1764) / 840) * 784)) + dj_outer_outer) + (((int)threadIdx.x) + 84)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 1862)] = (((1 <= (dj_outer_outer + ((((int)threadIdx.x) + 14) % 28))) && ((dj_outer_outer + ((((int)threadIdx.x) + 14) % 28)) < 29)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 1862) / 840) * 784)) + dj_outer_outer) + (((int)threadIdx.x) + 182)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 1960)] = (((1 <= (dj_outer_outer + (((int)threadIdx.x) % 28))) && ((dj_outer_outer + (((int)threadIdx.x) % 28)) < 29)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 1960) / 840) * 784)) + dj_outer_outer) + (((int)threadIdx.x) + 280)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 2058)] = (((1 <= (dj_outer_outer + ((((int)threadIdx.x) + 14) % 28))) && ((dj_outer_outer + ((((int)threadIdx.x) + 14) % 28)) < 29)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 2058) / 840) * 784)) + dj_outer_outer) + (((int)threadIdx.x) + 378)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 2156)] = (((1 <= (dj_outer_outer + (((int)threadIdx.x) % 28))) && ((dj_outer_outer + (((int)threadIdx.x) % 28)) < 29)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 2156) / 840) * 784)) + dj_outer_outer) + (((int)threadIdx.x) + 476)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 2254)] = (((1 <= (dj_outer_outer + ((((int)threadIdx.x) + 14) % 28))) && ((dj_outer_outer + ((((int)threadIdx.x) + 14) % 28)) < 29)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 2254) / 840) * 784)) + dj_outer_outer) + (((int)threadIdx.x) + 574)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 2352)] = (((1 <= (dj_outer_outer + (((int)threadIdx.x) % 28))) && ((dj_outer_outer + (((int)threadIdx.x) % 28)) < 29)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 2352) / 840) * 784)) + dj_outer_outer) + (((int)threadIdx.x) + 672)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 2450)] = (((((2 <= (((((int)threadIdx.x) / 14) + 55) % 60)) && (((((int)threadIdx.x) + 770) % 840) < 812)) && (1 <= (dj_outer_outer + ((((int)threadIdx.x) + 14) % 28)))) && ((dj_outer_outer + ((((int)threadIdx.x) + 14) % 28)) < 29)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 2450) / 840) * 784)) + dj_outer_outer) + ((((int)threadIdx.x) + 770) % 840)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 2548)] = (((1 <= (dj_outer_outer + (((int)threadIdx.x) % 28))) && ((dj_outer_outer + (((int)threadIdx.x) % 28)) < 29)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 2548) / 840) * 784)) + dj_outer_outer) + (((int)threadIdx.x) + 28)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 2646)] = (((1 <= (dj_outer_outer + ((((int)threadIdx.x) + 14) % 28))) && ((dj_outer_outer + ((((int)threadIdx.x) + 14) % 28)) < 29)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 2646) / 840) * 784)) + dj_outer_outer) + (((int)threadIdx.x) + 126)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 2744)] = (((1 <= (dj_outer_outer + (((int)threadIdx.x) % 28))) && ((dj_outer_outer + (((int)threadIdx.x) % 28)) < 29)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 2744) / 840) * 784)) + dj_outer_outer) + (((int)threadIdx.x) + 224)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 2842)] = (((1 <= (dj_outer_outer + ((((int)threadIdx.x) + 14) % 28))) && ((dj_outer_outer + ((((int)threadIdx.x) + 14) % 28)) < 29)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 2842) / 840) * 784)) + dj_outer_outer) + (((int)threadIdx.x) + 322)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 2940)] = (((1 <= (dj_outer_outer + (((int)threadIdx.x) % 28))) && ((dj_outer_outer + (((int)threadIdx.x) % 28)) < 29)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 2940) / 840) * 784)) + dj_outer_outer) + (((int)threadIdx.x) + 420)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 3038)] = (((1 <= (dj_outer_outer + ((((int)threadIdx.x) + 14) % 28))) && ((dj_outer_outer + ((((int)threadIdx.x) + 14) % 28)) < 29)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 3038) / 840) * 784)) + dj_outer_outer) + (((int)threadIdx.x) + 518)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 3136)] = (((1 <= (dj_outer_outer + (((int)threadIdx.x) % 28))) && ((dj_outer_outer + (((int)threadIdx.x) % 28)) < 29)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 3136) / 840) * 784)) + dj_outer_outer) + (((int)threadIdx.x) + 616)) - 29)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 3234)] = (((1 <= (dj_outer_outer + ((((int)threadIdx.x) + 14) % 28))) && ((dj_outer_outer + ((((int)threadIdx.x) + 14) % 28)) < 29)) ? Input[(((((((int)blockIdx.x) * 3136) + (((((int)threadIdx.x) + 3234) / 840) * 784)) + dj_outer_outer) + (((int)threadIdx.x) + 714)) - 29)] : 0.000000e+00f);
    if (((int)threadIdx.x) < 28) {
      PaddedInput_shared[(((int)threadIdx.x) + 3332)] = 0.000000e+00f;
    }
    if (((int)threadIdx.x) < 12) {
      kernel_shared[((int)threadIdx.x)] = kernel[(((((int)blockIdx.x) * 36) + (((int)threadIdx.x) * 3)) + dj_outer_outer)];
    }
    __syncthreads();
    for (int i_outer_inner = 0; i_outer_inner < 4; ++i_outer_inner) {
      DepthwiseConv2d[(i_outer_inner * 2)] = (DepthwiseConv2d[(i_outer_inner * 2)] + (PaddedInput_shared[((((((int)threadIdx.x) / 14) * 112) + (i_outer_inner * 28)) + ((((int)threadIdx.x) % 14) * 2))] * kernel_shared[0]));
      DepthwiseConv2d[((i_outer_inner * 2) + 8)] = (DepthwiseConv2d[((i_outer_inner * 2) + 8)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 112) + (i_outer_inner * 28)) + ((((int)threadIdx.x) % 14) * 2)) + 840)] * kernel_shared[3]));
      DepthwiseConv2d[((i_outer_inner * 2) + 16)] = (DepthwiseConv2d[((i_outer_inner * 2) + 16)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 112) + (i_outer_inner * 28)) + ((((int)threadIdx.x) % 14) * 2)) + 1680)] * kernel_shared[6]));
      DepthwiseConv2d[((i_outer_inner * 2) + 24)] = (DepthwiseConv2d[((i_outer_inner * 2) + 24)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 112) + (i_outer_inner * 28)) + ((((int)threadIdx.x) % 14) * 2)) + 2520)] * kernel_shared[9]));
      DepthwiseConv2d[(i_outer_inner * 2)] = (DepthwiseConv2d[(i_outer_inner * 2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 112) + (i_outer_inner * 28)) + ((((int)threadIdx.x) % 14) * 2)) + 28)] * kernel_shared[1]));
      DepthwiseConv2d[((i_outer_inner * 2) + 8)] = (DepthwiseConv2d[((i_outer_inner * 2) + 8)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 112) + (i_outer_inner * 28)) + ((((int)threadIdx.x) % 14) * 2)) + 868)] * kernel_shared[4]));
      DepthwiseConv2d[((i_outer_inner * 2) + 16)] = (DepthwiseConv2d[((i_outer_inner * 2) + 16)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 112) + (i_outer_inner * 28)) + ((((int)threadIdx.x) % 14) * 2)) + 1708)] * kernel_shared[7]));
      DepthwiseConv2d[((i_outer_inner * 2) + 24)] = (DepthwiseConv2d[((i_outer_inner * 2) + 24)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 112) + (i_outer_inner * 28)) + ((((int)threadIdx.x) % 14) * 2)) + 2548)] * kernel_shared[10]));
      DepthwiseConv2d[(i_outer_inner * 2)] = (DepthwiseConv2d[(i_outer_inner * 2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 112) + (i_outer_inner * 28)) + ((((int)threadIdx.x) % 14) * 2)) + 56)] * kernel_shared[2]));
      DepthwiseConv2d[((i_outer_inner * 2) + 8)] = (DepthwiseConv2d[((i_outer_inner * 2) + 8)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 112) + (i_outer_inner * 28)) + ((((int)threadIdx.x) % 14) * 2)) + 896)] * kernel_shared[5]));
      DepthwiseConv2d[((i_outer_inner * 2) + 16)] = (DepthwiseConv2d[((i_outer_inner * 2) + 16)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 112) + (i_outer_inner * 28)) + ((((int)threadIdx.x) % 14) * 2)) + 1736)] * kernel_shared[8]));
      DepthwiseConv2d[((i_outer_inner * 2) + 24)] = (DepthwiseConv2d[((i_outer_inner * 2) + 24)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 112) + (i_outer_inner * 28)) + ((((int)threadIdx.x) % 14) * 2)) + 2576)] * kernel_shared[11]));
      DepthwiseConv2d[((i_outer_inner * 2) + 1)] = (DepthwiseConv2d[((i_outer_inner * 2) + 1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 112) + (i_outer_inner * 28)) + ((((int)threadIdx.x) % 14) * 2)) + 1)] * kernel_shared[0]));
      DepthwiseConv2d[((i_outer_inner * 2) + 9)] = (DepthwiseConv2d[((i_outer_inner * 2) + 9)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 112) + (i_outer_inner * 28)) + ((((int)threadIdx.x) % 14) * 2)) + 841)] * kernel_shared[3]));
      DepthwiseConv2d[((i_outer_inner * 2) + 17)] = (DepthwiseConv2d[((i_outer_inner * 2) + 17)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 112) + (i_outer_inner * 28)) + ((((int)threadIdx.x) % 14) * 2)) + 1681)] * kernel_shared[6]));
      DepthwiseConv2d[((i_outer_inner * 2) + 25)] = (DepthwiseConv2d[((i_outer_inner * 2) + 25)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 112) + (i_outer_inner * 28)) + ((((int)threadIdx.x) % 14) * 2)) + 2521)] * kernel_shared[9]));
      DepthwiseConv2d[((i_outer_inner * 2) + 1)] = (DepthwiseConv2d[((i_outer_inner * 2) + 1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 112) + (i_outer_inner * 28)) + ((((int)threadIdx.x) % 14) * 2)) + 29)] * kernel_shared[1]));
      DepthwiseConv2d[((i_outer_inner * 2) + 9)] = (DepthwiseConv2d[((i_outer_inner * 2) + 9)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 112) + (i_outer_inner * 28)) + ((((int)threadIdx.x) % 14) * 2)) + 869)] * kernel_shared[4]));
      DepthwiseConv2d[((i_outer_inner * 2) + 17)] = (DepthwiseConv2d[((i_outer_inner * 2) + 17)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 112) + (i_outer_inner * 28)) + ((((int)threadIdx.x) % 14) * 2)) + 1709)] * kernel_shared[7]));
      DepthwiseConv2d[((i_outer_inner * 2) + 25)] = (DepthwiseConv2d[((i_outer_inner * 2) + 25)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 112) + (i_outer_inner * 28)) + ((((int)threadIdx.x) % 14) * 2)) + 2549)] * kernel_shared[10]));
      DepthwiseConv2d[((i_outer_inner * 2) + 1)] = (DepthwiseConv2d[((i_outer_inner * 2) + 1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 112) + (i_outer_inner * 28)) + ((((int)threadIdx.x) % 14) * 2)) + 57)] * kernel_shared[2]));
      DepthwiseConv2d[((i_outer_inner * 2) + 9)] = (DepthwiseConv2d[((i_outer_inner * 2) + 9)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 112) + (i_outer_inner * 28)) + ((((int)threadIdx.x) % 14) * 2)) + 897)] * kernel_shared[5]));
      DepthwiseConv2d[((i_outer_inner * 2) + 17)] = (DepthwiseConv2d[((i_outer_inner * 2) + 17)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 112) + (i_outer_inner * 28)) + ((((int)threadIdx.x) % 14) * 2)) + 1737)] * kernel_shared[8]));
      DepthwiseConv2d[((i_outer_inner * 2) + 25)] = (DepthwiseConv2d[((i_outer_inner * 2) + 25)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 112) + (i_outer_inner * 28)) + ((((int)threadIdx.x) % 14) * 2)) + 2577)] * kernel_shared[11]));
    }
  }
  for (int i1_inner = 0; i1_inner < 4; ++i1_inner) {
    for (int i2_inner = 0; i2_inner < 4; ++i2_inner) {
      for (int i3_inner = 0; i3_inner < 2; ++i3_inner) {
        compute[((((((((int)blockIdx.x) * 3136) + (i1_inner * 784)) + ((((int)threadIdx.x) / 14) * 112)) + (i2_inner * 28)) + ((((int)threadIdx.x) % 14) * 2)) + i3_inner)] = max(DepthwiseConv2d[(((i1_inner * 8) + (i2_inner * 2)) + i3_inner)], 0.000000e+00f);
      }
    }
  }
}


