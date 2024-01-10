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
extern "C" __global__ void __launch_bounds__(512) candidate7(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[4];
  __shared__ float pad_temp_shared[64];
  __shared__ float kernel_shared[8192];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 16; ++rc_outer_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 64) {
      pad_temp_shared[((int)threadIdx.x)] = data[((((((rc_outer_outer * 3136) + ((((int)threadIdx.x) >> 2) * 196)) + (((((int)blockIdx.x) % 49) / 7) * 28)) + (((((int)threadIdx.x) & 3) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1))];
    }
    kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) / 49) * 131072) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15))];
    kernel_shared[(((int)threadIdx.x) + 512)] = kernel[((((((((int)blockIdx.x) / 49) * 131072) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 8192)];
    kernel_shared[(((int)threadIdx.x) + 1024)] = kernel[((((((((int)blockIdx.x) / 49) * 131072) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 16384)];
    kernel_shared[(((int)threadIdx.x) + 1536)] = kernel[((((((((int)blockIdx.x) / 49) * 131072) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 24576)];
    kernel_shared[(((int)threadIdx.x) + 2048)] = kernel[((((((((int)blockIdx.x) / 49) * 131072) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 32768)];
    kernel_shared[(((int)threadIdx.x) + 2560)] = kernel[((((((((int)blockIdx.x) / 49) * 131072) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 40960)];
    kernel_shared[(((int)threadIdx.x) + 3072)] = kernel[((((((((int)blockIdx.x) / 49) * 131072) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 49152)];
    kernel_shared[(((int)threadIdx.x) + 3584)] = kernel[((((((((int)blockIdx.x) / 49) * 131072) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 57344)];
    kernel_shared[(((int)threadIdx.x) + 4096)] = kernel[((((((((int)blockIdx.x) / 49) * 131072) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 65536)];
    kernel_shared[(((int)threadIdx.x) + 4608)] = kernel[((((((((int)blockIdx.x) / 49) * 131072) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 73728)];
    kernel_shared[(((int)threadIdx.x) + 5120)] = kernel[((((((((int)blockIdx.x) / 49) * 131072) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 81920)];
    kernel_shared[(((int)threadIdx.x) + 5632)] = kernel[((((((((int)blockIdx.x) / 49) * 131072) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 90112)];
    kernel_shared[(((int)threadIdx.x) + 6144)] = kernel[((((((((int)blockIdx.x) / 49) * 131072) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 98304)];
    kernel_shared[(((int)threadIdx.x) + 6656)] = kernel[((((((((int)blockIdx.x) / 49) * 131072) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 106496)];
    kernel_shared[(((int)threadIdx.x) + 7168)] = kernel[((((((((int)blockIdx.x) / 49) * 131072) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 114688)];
    kernel_shared[(((int)threadIdx.x) + 7680)] = kernel[((((((((int)blockIdx.x) / 49) * 131072) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 122880)];
    __syncthreads();
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((int)threadIdx.x) & 3)] * kernel_shared[((((int)threadIdx.x) >> 2) * 16)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((int)threadIdx.x) & 3)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 2048)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((int)threadIdx.x) & 3)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 4096)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((int)threadIdx.x) & 3)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 6144)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 4)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 1)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 4)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 2049)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 4)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 4097)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 4)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 6145)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 8)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 2)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 8)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 2050)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 8)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 4098)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 8)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 6146)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 12)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 3)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 12)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 2051)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 12)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 4099)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 12)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 6147)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 16)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 4)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 16)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 2052)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 16)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 4100)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 16)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 6148)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 20)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 5)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 20)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 2053)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 20)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 4101)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 20)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 6149)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 24)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 6)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 24)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 2054)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 24)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 4102)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 24)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 6150)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 28)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 7)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 28)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 2055)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 28)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 4103)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 28)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 6151)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 32)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 8)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 32)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 2056)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 32)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 4104)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 32)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 6152)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 36)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 9)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 36)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 2057)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 36)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 4105)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 36)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 6153)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 40)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 10)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 40)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 2058)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 40)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 4106)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 40)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 6154)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 44)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 11)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 44)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 2059)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 44)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 4107)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 44)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 6155)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 48)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 12)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 48)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 2060)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 48)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 4108)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 48)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 6156)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 52)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 13)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 52)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 2061)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 52)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 4109)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 52)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 6157)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 56)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 14)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 56)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 2062)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 56)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 4110)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 56)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 6158)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 60)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 15)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 60)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 2063)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 60)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 4111)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 60)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 16) + 6159)]));
  }
  conv2d_nchw[(((((((((int)blockIdx.x) / 49) * 100352) + ((((int)threadIdx.x) >> 2) * 196)) + (((((int)blockIdx.x) % 49) / 7) * 28)) + (((((int)threadIdx.x) & 3) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1))] = conv2d_nchw_local[0];
  conv2d_nchw[((((((((((int)blockIdx.x) / 49) * 100352) + ((((int)threadIdx.x) >> 2) * 196)) + (((((int)blockIdx.x) % 49) / 7) * 28)) + (((((int)threadIdx.x) & 3) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1)) + 25088)] = conv2d_nchw_local[1];
  conv2d_nchw[((((((((((int)blockIdx.x) / 49) * 100352) + ((((int)threadIdx.x) >> 2) * 196)) + (((((int)blockIdx.x) % 49) / 7) * 28)) + (((((int)threadIdx.x) & 3) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1)) + 50176)] = conv2d_nchw_local[2];
  conv2d_nchw[((((((((((int)blockIdx.x) / 49) * 100352) + ((((int)threadIdx.x) >> 2) * 196)) + (((((int)blockIdx.x) % 49) / 7) * 28)) + (((((int)threadIdx.x) & 3) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1)) + 75264)] = conv2d_nchw_local[3];
}


