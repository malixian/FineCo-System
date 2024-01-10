#include "hip/hip_runtime.h"

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
extern "C" __global__ void __launch_bounds__(128) candidate3(float* __restrict__ Input, float* __restrict__ kernel, float* __restrict__ compute) {
  float conv2d_nchw[7];
  __shared__ float pad_temp_shared[448];
  __shared__ float kernel_shared[8192];
  conv2d_nchw[0] = 0.000000e+00f;
  conv2d_nchw[1] = 0.000000e+00f;
  conv2d_nchw[2] = 0.000000e+00f;
  conv2d_nchw[3] = 0.000000e+00f;
  conv2d_nchw[4] = 0.000000e+00f;
  conv2d_nchw[5] = 0.000000e+00f;
  conv2d_nchw[6] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 16; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = Input[((((rc_outer_outer * 3136) + ((((int)threadIdx.x) / 7) * 49)) + ((((int)blockIdx.x) % 7) * 7)) + (((int)threadIdx.x) % 7))];
    pad_temp_shared[(((int)threadIdx.x) + 128)] = Input[((((rc_outer_outer * 3136) + (((((int)threadIdx.x) + 128) / 7) * 49)) + ((((int)blockIdx.x) % 7) * 7)) + ((((int)threadIdx.x) + 2) % 7))];
    pad_temp_shared[(((int)threadIdx.x) + 256)] = Input[((((rc_outer_outer * 3136) + (((((int)threadIdx.x) + 256) / 7) * 49)) + ((((int)blockIdx.x) % 7) * 7)) + ((((int)threadIdx.x) + 4) % 7))];
    if (((int)threadIdx.x) < 64) {
      pad_temp_shared[(((int)threadIdx.x) + 384)] = Input[((((rc_outer_outer * 3136) + (((((int)threadIdx.x) + 384) / 7) * 49)) + ((((int)blockIdx.x) % 7) * 7)) + ((((int)threadIdx.x) + 6) % 7))];
    }
    kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63))];
    kernel_shared[(((int)threadIdx.x) + 128)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 2048)];
    kernel_shared[(((int)threadIdx.x) + 256)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 4096)];
    kernel_shared[(((int)threadIdx.x) + 384)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 6144)];
    kernel_shared[(((int)threadIdx.x) + 512)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 8192)];
    kernel_shared[(((int)threadIdx.x) + 640)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 10240)];
    kernel_shared[(((int)threadIdx.x) + 768)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 12288)];
    kernel_shared[(((int)threadIdx.x) + 896)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 14336)];
    kernel_shared[(((int)threadIdx.x) + 1024)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 16384)];
    kernel_shared[(((int)threadIdx.x) + 1152)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 18432)];
    kernel_shared[(((int)threadIdx.x) + 1280)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 20480)];
    kernel_shared[(((int)threadIdx.x) + 1408)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 22528)];
    kernel_shared[(((int)threadIdx.x) + 1536)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 24576)];
    kernel_shared[(((int)threadIdx.x) + 1664)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 26624)];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 28672)];
    kernel_shared[(((int)threadIdx.x) + 1920)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 30720)];
    kernel_shared[(((int)threadIdx.x) + 2048)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 32768)];
    kernel_shared[(((int)threadIdx.x) + 2176)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 34816)];
    kernel_shared[(((int)threadIdx.x) + 2304)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 36864)];
    kernel_shared[(((int)threadIdx.x) + 2432)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 38912)];
    kernel_shared[(((int)threadIdx.x) + 2560)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 40960)];
    kernel_shared[(((int)threadIdx.x) + 2688)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 43008)];
    kernel_shared[(((int)threadIdx.x) + 2816)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 45056)];
    kernel_shared[(((int)threadIdx.x) + 2944)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 47104)];
    kernel_shared[(((int)threadIdx.x) + 3072)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 49152)];
    kernel_shared[(((int)threadIdx.x) + 3200)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 51200)];
    kernel_shared[(((int)threadIdx.x) + 3328)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 53248)];
    kernel_shared[(((int)threadIdx.x) + 3456)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 55296)];
    kernel_shared[(((int)threadIdx.x) + 3584)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 57344)];
    kernel_shared[(((int)threadIdx.x) + 3712)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 59392)];
    kernel_shared[(((int)threadIdx.x) + 3840)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 61440)];
    kernel_shared[(((int)threadIdx.x) + 3968)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 63488)];
    kernel_shared[(((int)threadIdx.x) + 4096)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 65536)];
    kernel_shared[(((int)threadIdx.x) + 4224)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 67584)];
    kernel_shared[(((int)threadIdx.x) + 4352)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 69632)];
    kernel_shared[(((int)threadIdx.x) + 4480)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 71680)];
    kernel_shared[(((int)threadIdx.x) + 4608)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 73728)];
    kernel_shared[(((int)threadIdx.x) + 4736)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 75776)];
    kernel_shared[(((int)threadIdx.x) + 4864)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 77824)];
    kernel_shared[(((int)threadIdx.x) + 4992)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 79872)];
    kernel_shared[(((int)threadIdx.x) + 5120)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 81920)];
    kernel_shared[(((int)threadIdx.x) + 5248)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 83968)];
    kernel_shared[(((int)threadIdx.x) + 5376)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 86016)];
    kernel_shared[(((int)threadIdx.x) + 5504)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 88064)];
    kernel_shared[(((int)threadIdx.x) + 5632)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 90112)];
    kernel_shared[(((int)threadIdx.x) + 5760)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 92160)];
    kernel_shared[(((int)threadIdx.x) + 5888)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 94208)];
    kernel_shared[(((int)threadIdx.x) + 6016)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 96256)];
    kernel_shared[(((int)threadIdx.x) + 6144)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 98304)];
    kernel_shared[(((int)threadIdx.x) + 6272)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 100352)];
    kernel_shared[(((int)threadIdx.x) + 6400)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 102400)];
    kernel_shared[(((int)threadIdx.x) + 6528)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 104448)];
    kernel_shared[(((int)threadIdx.x) + 6656)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 106496)];
    kernel_shared[(((int)threadIdx.x) + 6784)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 108544)];
    kernel_shared[(((int)threadIdx.x) + 6912)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 110592)];
    kernel_shared[(((int)threadIdx.x) + 7040)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 112640)];
    kernel_shared[(((int)threadIdx.x) + 7168)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 114688)];
    kernel_shared[(((int)threadIdx.x) + 7296)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 116736)];
    kernel_shared[(((int)threadIdx.x) + 7424)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 118784)];
    kernel_shared[(((int)threadIdx.x) + 7552)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 120832)];
    kernel_shared[(((int)threadIdx.x) + 7680)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 122880)];
    kernel_shared[(((int)threadIdx.x) + 7808)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 124928)];
    kernel_shared[(((int)threadIdx.x) + 7936)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 126976)];
    kernel_shared[(((int)threadIdx.x) + 8064)] = kernel[((((((((int)blockIdx.x) / 7) * 131072) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 129024)];
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 4; ++rc_outer_inner) {
      for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
        conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((rc_outer_inner * 112) + (rc_inner * 7))] * kernel_shared[(((((int)threadIdx.x) * 64) + (rc_outer_inner * 16)) + rc_inner)]));
        conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((rc_outer_inner * 112) + (rc_inner * 7)) + 1)] * kernel_shared[(((((int)threadIdx.x) * 64) + (rc_outer_inner * 16)) + rc_inner)]));
        conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((rc_outer_inner * 112) + (rc_inner * 7)) + 2)] * kernel_shared[(((((int)threadIdx.x) * 64) + (rc_outer_inner * 16)) + rc_inner)]));
        conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((rc_outer_inner * 112) + (rc_inner * 7)) + 3)] * kernel_shared[(((((int)threadIdx.x) * 64) + (rc_outer_inner * 16)) + rc_inner)]));
        conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[(((rc_outer_inner * 112) + (rc_inner * 7)) + 4)] * kernel_shared[(((((int)threadIdx.x) * 64) + (rc_outer_inner * 16)) + rc_inner)]));
        conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[(((rc_outer_inner * 112) + (rc_inner * 7)) + 5)] * kernel_shared[(((((int)threadIdx.x) * 64) + (rc_outer_inner * 16)) + rc_inner)]));
        conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[(((rc_outer_inner * 112) + (rc_inner * 7)) + 6)] * kernel_shared[(((((int)threadIdx.x) * 64) + (rc_outer_inner * 16)) + rc_inner)]));
      }
    }
  }
  compute[((((((int)blockIdx.x) / 7) * 6272) + (((int)threadIdx.x) * 49)) + ((((int)blockIdx.x) % 7) * 7))] = max(conv2d_nchw[0], 0.000000e+00f);
  compute[(((((((int)blockIdx.x) / 7) * 6272) + (((int)threadIdx.x) * 49)) + ((((int)blockIdx.x) % 7) * 7)) + 1)] = max(conv2d_nchw[1], 0.000000e+00f);
  compute[(((((((int)blockIdx.x) / 7) * 6272) + (((int)threadIdx.x) * 49)) + ((((int)blockIdx.x) % 7) * 7)) + 2)] = max(conv2d_nchw[2], 0.000000e+00f);
  compute[(((((((int)blockIdx.x) / 7) * 6272) + (((int)threadIdx.x) * 49)) + ((((int)blockIdx.x) % 7) * 7)) + 3)] = max(conv2d_nchw[3], 0.000000e+00f);
  compute[(((((((int)blockIdx.x) / 7) * 6272) + (((int)threadIdx.x) * 49)) + ((((int)blockIdx.x) % 7) * 7)) + 4)] = max(conv2d_nchw[4], 0.000000e+00f);
  compute[(((((((int)blockIdx.x) / 7) * 6272) + (((int)threadIdx.x) * 49)) + ((((int)blockIdx.x) % 7) * 7)) + 5)] = max(conv2d_nchw[5], 0.000000e+00f);
  compute[(((((((int)blockIdx.x) / 7) * 6272) + (((int)threadIdx.x) * 49)) + ((((int)blockIdx.x) % 7) * 7)) + 6)] = max(conv2d_nchw[6], 0.000000e+00f);
}


