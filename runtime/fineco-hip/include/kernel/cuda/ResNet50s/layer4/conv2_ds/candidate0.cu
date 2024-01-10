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
extern "C" __global__ void __launch_bounds__(256) candidate0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[2];
  __shared__ float pad_temp_shared[200];
  __shared__ float kernel_shared[9216];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 32; ++rc_outer_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 40) {
      pad_temp_shared[(((int)threadIdx.x) * 5)] = (((1 <= ((((((int)blockIdx.x) % 49) / 7) * 4) + (((int)threadIdx.x) % 5))) && (1 <= (((int)blockIdx.x) % 7))) ? data[((((((rc_outer_outer * 6272) + ((((int)threadIdx.x) / 5) * 784)) + (((((int)blockIdx.x) % 49) / 7) * 112)) + ((((int)threadIdx.x) % 5) * 28)) + ((((int)blockIdx.x) % 7) * 4)) - 29)] : 0.000000e+00f);
      pad_temp_shared[((((int)threadIdx.x) * 5) + 1)] = ((1 <= ((((((int)blockIdx.x) % 49) / 7) * 4) + (((int)threadIdx.x) % 5))) ? data[((((((rc_outer_outer * 6272) + ((((int)threadIdx.x) / 5) * 784)) + (((((int)blockIdx.x) % 49) / 7) * 112)) + ((((int)threadIdx.x) % 5) * 28)) + ((((int)blockIdx.x) % 7) * 4)) - 28)] : 0.000000e+00f);
      pad_temp_shared[((((int)threadIdx.x) * 5) + 2)] = ((1 <= ((((((int)blockIdx.x) % 49) / 7) * 4) + (((int)threadIdx.x) % 5))) ? data[((((((rc_outer_outer * 6272) + ((((int)threadIdx.x) / 5) * 784)) + (((((int)blockIdx.x) % 49) / 7) * 112)) + ((((int)threadIdx.x) % 5) * 28)) + ((((int)blockIdx.x) % 7) * 4)) - 27)] : 0.000000e+00f);
      pad_temp_shared[((((int)threadIdx.x) * 5) + 3)] = ((1 <= ((((((int)blockIdx.x) % 49) / 7) * 4) + (((int)threadIdx.x) % 5))) ? data[((((((rc_outer_outer * 6272) + ((((int)threadIdx.x) / 5) * 784)) + (((((int)blockIdx.x) % 49) / 7) * 112)) + ((((int)threadIdx.x) % 5) * 28)) + ((((int)blockIdx.x) % 7) * 4)) - 26)] : 0.000000e+00f);
      pad_temp_shared[((((int)threadIdx.x) * 5) + 4)] = ((1 <= ((((((int)blockIdx.x) % 49) / 7) * 4) + (((int)threadIdx.x) % 5))) ? data[((((((rc_outer_outer * 6272) + ((((int)threadIdx.x) / 5) * 784)) + (((((int)blockIdx.x) % 49) / 7) * 112)) + ((((int)threadIdx.x) % 5) * 28)) + ((((int)blockIdx.x) % 7) * 4)) - 25)] : 0.000000e+00f);
    }
    *(float4*)(kernel_shared + (((int)threadIdx.x) * 4)) = *(float4*)(kernel + (((((((int)blockIdx.x) / 49) * 294912) + ((((int)threadIdx.x) / 18) * 2304)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) % 18) * 4)));
    *(float4*)(kernel_shared + ((((int)threadIdx.x) * 4) + 1024)) = *(float4*)(kernel + (((((((int)blockIdx.x) / 49) * 294912) + ((((((int)threadIdx.x) * 4) + 1024) / 72) * 2304)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) * 4) + 16) % 72)));
    *(float4*)(kernel_shared + ((((int)threadIdx.x) * 4) + 2048)) = *(float4*)(kernel + (((((((int)blockIdx.x) / 49) * 294912) + ((((((int)threadIdx.x) * 4) + 2048) / 72) * 2304)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) * 4) + 32) % 72)));
    *(float4*)(kernel_shared + ((((int)threadIdx.x) * 4) + 3072)) = *(float4*)(kernel + (((((((int)blockIdx.x) / 49) * 294912) + ((((((int)threadIdx.x) * 4) + 3072) / 72) * 2304)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) * 4) + 48) % 72)));
    *(float4*)(kernel_shared + ((((int)threadIdx.x) * 4) + 4096)) = *(float4*)(kernel + (((((((int)blockIdx.x) / 49) * 294912) + ((((((int)threadIdx.x) * 4) + 4096) / 72) * 2304)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) * 4) + 64) % 72)));
    *(float4*)(kernel_shared + ((((int)threadIdx.x) * 4) + 5120)) = *(float4*)(kernel + (((((((int)blockIdx.x) / 49) * 294912) + ((((((int)threadIdx.x) * 4) + 5120) / 72) * 2304)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) * 4) + 8) % 72)));
    *(float4*)(kernel_shared + ((((int)threadIdx.x) * 4) + 6144)) = *(float4*)(kernel + (((((((int)blockIdx.x) / 49) * 294912) + ((((((int)threadIdx.x) * 4) + 6144) / 72) * 2304)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) * 4) + 24) % 72)));
    *(float4*)(kernel_shared + ((((int)threadIdx.x) * 4) + 7168)) = *(float4*)(kernel + (((((((int)blockIdx.x) / 49) * 294912) + ((((((int)threadIdx.x) * 4) + 7168) / 72) * 2304)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) * 4) + 40) % 72)));
    *(float4*)(kernel_shared + ((((int)threadIdx.x) * 4) + 8192)) = *(float4*)(kernel + (((((((int)blockIdx.x) / 49) * 294912) + ((((((int)threadIdx.x) * 4) + 8192) / 72) * 2304)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) * 4) + 56) % 72)));
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 2; ++rc_outer_inner) {
      for (int rc_inner = 0; rc_inner < 4; ++rc_inner) {
        conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((rc_outer_inner * 100) + (rc_inner * 25)) + (((((int)threadIdx.x) & 3) >> 1) * 10)) + ((((int)threadIdx.x) & 1) * 2))] * kernel_shared[((((((int)threadIdx.x) >> 2) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9))]));
        conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((rc_outer_inner * 100) + (rc_inner * 25)) + (((((int)threadIdx.x) & 3) >> 1) * 10)) + ((((int)threadIdx.x) & 1) * 2))] * kernel_shared[(((((((int)threadIdx.x) >> 2) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + 4608)]));
        conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((rc_outer_inner * 100) + (rc_inner * 25)) + (((((int)threadIdx.x) & 3) >> 1) * 10)) + ((((int)threadIdx.x) & 1) * 2)) + 1)] * kernel_shared[(((((((int)threadIdx.x) >> 2) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + 1)]));
        conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((rc_outer_inner * 100) + (rc_inner * 25)) + (((((int)threadIdx.x) & 3) >> 1) * 10)) + ((((int)threadIdx.x) & 1) * 2)) + 1)] * kernel_shared[(((((((int)threadIdx.x) >> 2) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + 4609)]));
        conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((rc_outer_inner * 100) + (rc_inner * 25)) + (((((int)threadIdx.x) & 3) >> 1) * 10)) + ((((int)threadIdx.x) & 1) * 2)) + 2)] * kernel_shared[(((((((int)threadIdx.x) >> 2) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + 2)]));
        conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((rc_outer_inner * 100) + (rc_inner * 25)) + (((((int)threadIdx.x) & 3) >> 1) * 10)) + ((((int)threadIdx.x) & 1) * 2)) + 2)] * kernel_shared[(((((((int)threadIdx.x) >> 2) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + 4610)]));
        conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((rc_outer_inner * 100) + (rc_inner * 25)) + (((((int)threadIdx.x) & 3) >> 1) * 10)) + ((((int)threadIdx.x) & 1) * 2)) + 5)] * kernel_shared[(((((((int)threadIdx.x) >> 2) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + 3)]));
        conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((rc_outer_inner * 100) + (rc_inner * 25)) + (((((int)threadIdx.x) & 3) >> 1) * 10)) + ((((int)threadIdx.x) & 1) * 2)) + 5)] * kernel_shared[(((((((int)threadIdx.x) >> 2) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + 4611)]));
        conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((rc_outer_inner * 100) + (rc_inner * 25)) + (((((int)threadIdx.x) & 3) >> 1) * 10)) + ((((int)threadIdx.x) & 1) * 2)) + 6)] * kernel_shared[(((((((int)threadIdx.x) >> 2) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + 4)]));
        conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((rc_outer_inner * 100) + (rc_inner * 25)) + (((((int)threadIdx.x) & 3) >> 1) * 10)) + ((((int)threadIdx.x) & 1) * 2)) + 6)] * kernel_shared[(((((((int)threadIdx.x) >> 2) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + 4612)]));
        conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((rc_outer_inner * 100) + (rc_inner * 25)) + (((((int)threadIdx.x) & 3) >> 1) * 10)) + ((((int)threadIdx.x) & 1) * 2)) + 7)] * kernel_shared[(((((((int)threadIdx.x) >> 2) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + 5)]));
        conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((rc_outer_inner * 100) + (rc_inner * 25)) + (((((int)threadIdx.x) & 3) >> 1) * 10)) + ((((int)threadIdx.x) & 1) * 2)) + 7)] * kernel_shared[(((((((int)threadIdx.x) >> 2) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + 4613)]));
        conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((rc_outer_inner * 100) + (rc_inner * 25)) + (((((int)threadIdx.x) & 3) >> 1) * 10)) + ((((int)threadIdx.x) & 1) * 2)) + 10)] * kernel_shared[(((((((int)threadIdx.x) >> 2) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + 6)]));
        conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((rc_outer_inner * 100) + (rc_inner * 25)) + (((((int)threadIdx.x) & 3) >> 1) * 10)) + ((((int)threadIdx.x) & 1) * 2)) + 10)] * kernel_shared[(((((((int)threadIdx.x) >> 2) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + 4614)]));
        conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((rc_outer_inner * 100) + (rc_inner * 25)) + (((((int)threadIdx.x) & 3) >> 1) * 10)) + ((((int)threadIdx.x) & 1) * 2)) + 11)] * kernel_shared[(((((((int)threadIdx.x) >> 2) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + 7)]));
        conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((rc_outer_inner * 100) + (rc_inner * 25)) + (((((int)threadIdx.x) & 3) >> 1) * 10)) + ((((int)threadIdx.x) & 1) * 2)) + 11)] * kernel_shared[(((((((int)threadIdx.x) >> 2) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + 4615)]));
        conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((rc_outer_inner * 100) + (rc_inner * 25)) + (((((int)threadIdx.x) & 3) >> 1) * 10)) + ((((int)threadIdx.x) & 1) * 2)) + 12)] * kernel_shared[(((((((int)threadIdx.x) >> 2) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + 8)]));
        conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((rc_outer_inner * 100) + (rc_inner * 25)) + (((((int)threadIdx.x) & 3) >> 1) * 10)) + ((((int)threadIdx.x) & 1) * 2)) + 12)] * kernel_shared[(((((((int)threadIdx.x) >> 2) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + 4616)]));
      }
    }
  }
  conv2d_nchw[(((((((((int)blockIdx.x) / 49) * 25088) + ((((int)threadIdx.x) >> 2) * 196)) + (((((int)blockIdx.x) % 49) / 7) * 28)) + (((((int)threadIdx.x) & 3) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1))] = conv2d_nchw_local[0];
  conv2d_nchw[((((((((((int)blockIdx.x) / 49) * 25088) + ((((int)threadIdx.x) >> 2) * 196)) + (((((int)blockIdx.x) % 49) / 7) * 28)) + (((((int)threadIdx.x) & 3) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1)) + 12544)] = conv2d_nchw_local[1];
}


