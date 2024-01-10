
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
extern "C" __global__ void __launch_bounds__(448) candidate3(float* __restrict__ Input, float* __restrict__ kernel, float* __restrict__ compute) {
  float DepthwiseConv2d[4];
  __shared__ float PaddedInput_shared[7776];
  __shared__ float kernel_shared[96];
  DepthwiseConv2d[0] = 0.000000e+00f;
  DepthwiseConv2d[1] = 0.000000e+00f;
  DepthwiseConv2d[2] = 0.000000e+00f;
  DepthwiseConv2d[3] = 0.000000e+00f;
  for (int dj_outer_outer = 0; dj_outer_outer < 3; ++dj_outer_outer) {
    __syncthreads();
    PaddedInput_shared[((int)threadIdx.x)] = (((1 <= ((((((int)blockIdx.x) % 14) >> 1) * 8) + ((((int)threadIdx.x) % 243) / 27))) && (1 <= ((((((int)blockIdx.x) & 1) * 28) + dj_outer_outer) + (((int)threadIdx.x) % 27)))) ? Input[(((((((((((int)blockIdx.x) / 14) * 100352) + ((((int)threadIdx.x) / 243) * 3136)) + (((((int)blockIdx.x) % 14) >> 1) * 448)) + (((((int)threadIdx.x) % 243) / 27) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + dj_outer_outer) + (((int)threadIdx.x) % 27)) - 57)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 448)] = (((1 <= ((((((int)blockIdx.x) % 14) >> 1) * 8) + (((((int)threadIdx.x) + 205) % 243) / 27))) && (1 <= ((((((int)blockIdx.x) & 1) * 28) + dj_outer_outer) + ((((int)threadIdx.x) + 16) % 27)))) ? Input[(((((((((((int)blockIdx.x) / 14) * 100352) + (((((int)threadIdx.x) + 448) / 243) * 3136)) + (((((int)blockIdx.x) % 14) >> 1) * 448)) + ((((((int)threadIdx.x) + 205) % 243) / 27) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + dj_outer_outer) + ((((int)threadIdx.x) + 16) % 27)) - 57)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 896)] = (((1 <= ((((((int)blockIdx.x) % 14) >> 1) * 8) + (((((int)threadIdx.x) + 167) % 243) / 27))) && (1 <= ((((((int)blockIdx.x) & 1) * 28) + dj_outer_outer) + ((((int)threadIdx.x) + 5) % 27)))) ? Input[(((((((((((int)blockIdx.x) / 14) * 100352) + (((((int)threadIdx.x) + 896) / 243) * 3136)) + (((((int)blockIdx.x) % 14) >> 1) * 448)) + ((((((int)threadIdx.x) + 167) % 243) / 27) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + dj_outer_outer) + ((((int)threadIdx.x) + 5) % 27)) - 57)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 1344)] = (((1 <= ((((((int)blockIdx.x) % 14) >> 1) * 8) + (((((int)threadIdx.x) + 129) % 243) / 27))) && (1 <= ((((((int)blockIdx.x) & 1) * 28) + dj_outer_outer) + ((((int)threadIdx.x) + 21) % 27)))) ? Input[(((((((((((int)blockIdx.x) / 14) * 100352) + (((((int)threadIdx.x) + 1344) / 243) * 3136)) + (((((int)blockIdx.x) % 14) >> 1) * 448)) + ((((((int)threadIdx.x) + 129) % 243) / 27) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + dj_outer_outer) + ((((int)threadIdx.x) + 21) % 27)) - 57)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 1792)] = (((1 <= ((((((int)blockIdx.x) % 14) >> 1) * 8) + (((((int)threadIdx.x) + 91) % 243) / 27))) && (1 <= ((((((int)blockIdx.x) & 1) * 28) + dj_outer_outer) + ((((int)threadIdx.x) + 10) % 27)))) ? Input[(((((((((((int)blockIdx.x) / 14) * 100352) + (((((int)threadIdx.x) + 1792) / 243) * 3136)) + (((((int)blockIdx.x) % 14) >> 1) * 448)) + ((((((int)threadIdx.x) + 91) % 243) / 27) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + dj_outer_outer) + ((((int)threadIdx.x) + 10) % 27)) - 57)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 2240)] = (((1 <= ((((((int)blockIdx.x) % 14) >> 1) * 8) + (((((int)threadIdx.x) + 53) % 243) / 27))) && (1 <= ((((((int)blockIdx.x) & 1) * 28) + dj_outer_outer) + ((((int)threadIdx.x) + 26) % 27)))) ? Input[(((((((((((int)blockIdx.x) / 14) * 100352) + (((((int)threadIdx.x) + 2240) / 243) * 3136)) + (((((int)blockIdx.x) % 14) >> 1) * 448)) + ((((((int)threadIdx.x) + 53) % 243) / 27) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + dj_outer_outer) + ((((int)threadIdx.x) + 26) % 27)) - 57)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 2688)] = (((1 <= ((((((int)blockIdx.x) % 14) >> 1) * 8) + (((((int)threadIdx.x) + 15) % 243) / 27))) && (1 <= ((((((int)blockIdx.x) & 1) * 28) + dj_outer_outer) + ((((int)threadIdx.x) + 15) % 27)))) ? Input[(((((((((((int)blockIdx.x) / 14) * 100352) + (((((int)threadIdx.x) + 2688) / 243) * 3136)) + (((((int)blockIdx.x) % 14) >> 1) * 448)) + ((((((int)threadIdx.x) + 15) % 243) / 27) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + dj_outer_outer) + ((((int)threadIdx.x) + 15) % 27)) - 57)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 3136)] = (((1 <= ((((((int)blockIdx.x) % 14) >> 1) * 8) + (((((int)threadIdx.x) + 220) % 243) / 27))) && (1 <= ((((((int)blockIdx.x) & 1) * 28) + dj_outer_outer) + ((((int)threadIdx.x) + 4) % 27)))) ? Input[(((((((((((int)blockIdx.x) / 14) * 100352) + (((((int)threadIdx.x) + 3136) / 243) * 3136)) + (((((int)blockIdx.x) % 14) >> 1) * 448)) + ((((((int)threadIdx.x) + 220) % 243) / 27) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + dj_outer_outer) + ((((int)threadIdx.x) + 4) % 27)) - 57)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 3584)] = (((1 <= ((((((int)blockIdx.x) % 14) >> 1) * 8) + (((((int)threadIdx.x) + 182) % 243) / 27))) && (1 <= ((((((int)blockIdx.x) & 1) * 28) + dj_outer_outer) + ((((int)threadIdx.x) + 20) % 27)))) ? Input[(((((((((((int)blockIdx.x) / 14) * 100352) + (((((int)threadIdx.x) + 3584) / 243) * 3136)) + (((((int)blockIdx.x) % 14) >> 1) * 448)) + ((((((int)threadIdx.x) + 182) % 243) / 27) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + dj_outer_outer) + ((((int)threadIdx.x) + 20) % 27)) - 57)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 4032)] = (((1 <= ((((((int)blockIdx.x) % 14) >> 1) * 8) + (((((int)threadIdx.x) + 144) % 243) / 27))) && (1 <= ((((((int)blockIdx.x) & 1) * 28) + dj_outer_outer) + ((((int)threadIdx.x) + 9) % 27)))) ? Input[(((((((((((int)blockIdx.x) / 14) * 100352) + (((((int)threadIdx.x) + 4032) / 243) * 3136)) + (((((int)blockIdx.x) % 14) >> 1) * 448)) + ((((((int)threadIdx.x) + 144) % 243) / 27) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + dj_outer_outer) + ((((int)threadIdx.x) + 9) % 27)) - 57)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 4480)] = (((1 <= ((((((int)blockIdx.x) % 14) >> 1) * 8) + (((((int)threadIdx.x) + 106) % 243) / 27))) && (1 <= ((((((int)blockIdx.x) & 1) * 28) + dj_outer_outer) + ((((int)threadIdx.x) + 25) % 27)))) ? Input[(((((((((((int)blockIdx.x) / 14) * 100352) + (((((int)threadIdx.x) + 4480) / 243) * 3136)) + (((((int)blockIdx.x) % 14) >> 1) * 448)) + ((((((int)threadIdx.x) + 106) % 243) / 27) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + dj_outer_outer) + ((((int)threadIdx.x) + 25) % 27)) - 57)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 4928)] = (((1 <= ((((((int)blockIdx.x) % 14) >> 1) * 8) + (((((int)threadIdx.x) + 68) % 243) / 27))) && (1 <= ((((((int)blockIdx.x) & 1) * 28) + dj_outer_outer) + ((((int)threadIdx.x) + 14) % 27)))) ? Input[(((((((((((int)blockIdx.x) / 14) * 100352) + (((((int)threadIdx.x) + 4928) / 243) * 3136)) + (((((int)blockIdx.x) % 14) >> 1) * 448)) + ((((((int)threadIdx.x) + 68) % 243) / 27) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + dj_outer_outer) + ((((int)threadIdx.x) + 14) % 27)) - 57)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 5376)] = (((1 <= ((((((int)blockIdx.x) % 14) >> 1) * 8) + (((((int)threadIdx.x) + 30) % 243) / 27))) && (1 <= ((((((int)blockIdx.x) & 1) * 28) + dj_outer_outer) + ((((int)threadIdx.x) + 3) % 27)))) ? Input[(((((((((((int)blockIdx.x) / 14) * 100352) + (((((int)threadIdx.x) + 5376) / 243) * 3136)) + (((((int)blockIdx.x) % 14) >> 1) * 448)) + ((((((int)threadIdx.x) + 30) % 243) / 27) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + dj_outer_outer) + ((((int)threadIdx.x) + 3) % 27)) - 57)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 5824)] = (((1 <= ((((((int)blockIdx.x) % 14) >> 1) * 8) + (((((int)threadIdx.x) + 235) % 243) / 27))) && (1 <= ((((((int)blockIdx.x) & 1) * 28) + dj_outer_outer) + ((((int)threadIdx.x) + 19) % 27)))) ? Input[(((((((((((int)blockIdx.x) / 14) * 100352) + (((((int)threadIdx.x) + 5824) / 243) * 3136)) + (((((int)blockIdx.x) % 14) >> 1) * 448)) + ((((((int)threadIdx.x) + 235) % 243) / 27) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + dj_outer_outer) + ((((int)threadIdx.x) + 19) % 27)) - 57)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 6272)] = (((1 <= ((((((int)blockIdx.x) % 14) >> 1) * 8) + (((((int)threadIdx.x) + 197) % 243) / 27))) && (1 <= ((((((int)blockIdx.x) & 1) * 28) + dj_outer_outer) + ((((int)threadIdx.x) + 8) % 27)))) ? Input[(((((((((((int)blockIdx.x) / 14) * 100352) + (((((int)threadIdx.x) + 6272) / 243) * 3136)) + (((((int)blockIdx.x) % 14) >> 1) * 448)) + ((((((int)threadIdx.x) + 197) % 243) / 27) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + dj_outer_outer) + ((((int)threadIdx.x) + 8) % 27)) - 57)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 6720)] = (((1 <= ((((((int)blockIdx.x) % 14) >> 1) * 8) + (((((int)threadIdx.x) + 159) % 243) / 27))) && (1 <= ((((((int)blockIdx.x) & 1) * 28) + dj_outer_outer) + ((((int)threadIdx.x) + 24) % 27)))) ? Input[(((((((((((int)blockIdx.x) / 14) * 100352) + (((((int)threadIdx.x) + 6720) / 243) * 3136)) + (((((int)blockIdx.x) % 14) >> 1) * 448)) + ((((((int)threadIdx.x) + 159) % 243) / 27) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + dj_outer_outer) + ((((int)threadIdx.x) + 24) % 27)) - 57)] : 0.000000e+00f);
    PaddedInput_shared[(((int)threadIdx.x) + 7168)] = (((1 <= ((((((int)blockIdx.x) % 14) >> 1) * 8) + (((((int)threadIdx.x) + 121) % 243) / 27))) && (1 <= ((((((int)blockIdx.x) & 1) * 28) + dj_outer_outer) + ((((int)threadIdx.x) + 13) % 27)))) ? Input[(((((((((((int)blockIdx.x) / 14) * 100352) + (((((int)threadIdx.x) + 7168) / 243) * 3136)) + (((((int)blockIdx.x) % 14) >> 1) * 448)) + ((((((int)threadIdx.x) + 121) % 243) / 27) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + dj_outer_outer) + ((((int)threadIdx.x) + 13) % 27)) - 57)] : 0.000000e+00f);
    if (((int)threadIdx.x) < 160) {
      PaddedInput_shared[(((int)threadIdx.x) + 7616)] = ((1 <= ((((((int)blockIdx.x) & 1) * 28) + dj_outer_outer) + ((((int)threadIdx.x) + 2) % 27))) ? Input[(((((((((((int)blockIdx.x) / 14) * 100352) + (((((int)threadIdx.x) + 7616) / 243) * 3136)) + (((((int)blockIdx.x) % 14) >> 1) * 448)) + ((((((int)threadIdx.x) + 83) % 243) / 27) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + dj_outer_outer) + ((((int)threadIdx.x) + 2) % 27)) - 57)] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 96) {
      kernel_shared[((int)threadIdx.x)] = kernel[((((((int)blockIdx.x) / 14) * 288) + (((int)threadIdx.x) * 3)) + dj_outer_outer)];
    }
    __syncthreads();
    DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 28) * 243) + (((((int)threadIdx.x) % 28) / 14) * 54)) + ((((int)threadIdx.x) % 14) * 2))] * kernel_shared[((((int)threadIdx.x) / 28) * 3)]));
    DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[(((((((int)threadIdx.x) / 28) * 243) + (((((int)threadIdx.x) % 28) / 14) * 54)) + ((((int)threadIdx.x) % 14) * 2)) + 108)] * kernel_shared[((((int)threadIdx.x) / 28) * 3)]));
    DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[(((((((int)threadIdx.x) / 28) * 243) + (((((int)threadIdx.x) % 28) / 14) * 54)) + ((((int)threadIdx.x) % 14) * 2)) + 3888)] * kernel_shared[(((((int)threadIdx.x) / 28) * 3) + 48)]));
    DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[(((((((int)threadIdx.x) / 28) * 243) + (((((int)threadIdx.x) % 28) / 14) * 54)) + ((((int)threadIdx.x) % 14) * 2)) + 3996)] * kernel_shared[(((((int)threadIdx.x) / 28) * 3) + 48)]));
    DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((((int)threadIdx.x) / 28) * 243) + (((((int)threadIdx.x) % 28) / 14) * 54)) + ((((int)threadIdx.x) % 14) * 2)) + 27)] * kernel_shared[(((((int)threadIdx.x) / 28) * 3) + 1)]));
    DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[(((((((int)threadIdx.x) / 28) * 243) + (((((int)threadIdx.x) % 28) / 14) * 54)) + ((((int)threadIdx.x) % 14) * 2)) + 135)] * kernel_shared[(((((int)threadIdx.x) / 28) * 3) + 1)]));
    DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[(((((((int)threadIdx.x) / 28) * 243) + (((((int)threadIdx.x) % 28) / 14) * 54)) + ((((int)threadIdx.x) % 14) * 2)) + 3915)] * kernel_shared[(((((int)threadIdx.x) / 28) * 3) + 49)]));
    DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[(((((((int)threadIdx.x) / 28) * 243) + (((((int)threadIdx.x) % 28) / 14) * 54)) + ((((int)threadIdx.x) % 14) * 2)) + 4023)] * kernel_shared[(((((int)threadIdx.x) / 28) * 3) + 49)]));
    DepthwiseConv2d[0] = (DepthwiseConv2d[0] + (PaddedInput_shared[(((((((int)threadIdx.x) / 28) * 243) + (((((int)threadIdx.x) % 28) / 14) * 54)) + ((((int)threadIdx.x) % 14) * 2)) + 54)] * kernel_shared[(((((int)threadIdx.x) / 28) * 3) + 2)]));
    DepthwiseConv2d[1] = (DepthwiseConv2d[1] + (PaddedInput_shared[(((((((int)threadIdx.x) / 28) * 243) + (((((int)threadIdx.x) % 28) / 14) * 54)) + ((((int)threadIdx.x) % 14) * 2)) + 162)] * kernel_shared[(((((int)threadIdx.x) / 28) * 3) + 2)]));
    DepthwiseConv2d[2] = (DepthwiseConv2d[2] + (PaddedInput_shared[(((((((int)threadIdx.x) / 28) * 243) + (((((int)threadIdx.x) % 28) / 14) * 54)) + ((((int)threadIdx.x) % 14) * 2)) + 3942)] * kernel_shared[(((((int)threadIdx.x) / 28) * 3) + 50)]));
    DepthwiseConv2d[3] = (DepthwiseConv2d[3] + (PaddedInput_shared[(((((((int)threadIdx.x) / 28) * 243) + (((((int)threadIdx.x) % 28) / 14) * 54)) + ((((int)threadIdx.x) % 14) * 2)) + 4050)] * kernel_shared[(((((int)threadIdx.x) / 28) * 3) + 50)]));
  }
  compute[(((((((((int)blockIdx.x) / 14) * 25088) + ((((int)threadIdx.x) / 28) * 784)) + (((((int)blockIdx.x) % 14) >> 1) * 112)) + (((((int)threadIdx.x) % 28) / 14) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((int)threadIdx.x) % 14))] = max(DepthwiseConv2d[0], 0.000000e+00f);
  compute[((((((((((int)blockIdx.x) / 14) * 25088) + ((((int)threadIdx.x) / 28) * 784)) + (((((int)blockIdx.x) % 14) >> 1) * 112)) + (((((int)threadIdx.x) % 28) / 14) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((int)threadIdx.x) % 14)) + 56)] = max(DepthwiseConv2d[1], 0.000000e+00f);
  compute[((((((((((int)blockIdx.x) / 14) * 25088) + ((((int)threadIdx.x) / 28) * 784)) + (((((int)blockIdx.x) % 14) >> 1) * 112)) + (((((int)threadIdx.x) % 28) / 14) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((int)threadIdx.x) % 14)) + 12544)] = max(DepthwiseConv2d[2], 0.000000e+00f);
  compute[((((((((((int)blockIdx.x) / 14) * 25088) + ((((int)threadIdx.x) / 28) * 784)) + (((((int)blockIdx.x) % 14) >> 1) * 112)) + (((((int)threadIdx.x) % 28) / 14) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((int)threadIdx.x) % 14)) + 12600)] = max(DepthwiseConv2d[3], 0.000000e+00f);
}

