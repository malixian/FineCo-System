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
extern "C" __global__ void __launch_bounds__(64) candidate0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[4];
  __shared__ float pad_temp_shared[96];
  __shared__ float kernel_shared[1152];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 32; ++rc_outer_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 32) {
      pad_temp_shared[(((int)threadIdx.x) * 3)] = ((((1 <= ((((((int)blockIdx.x) % 98) / 7) * 2) + ((((int)threadIdx.x) & 7) >> 1))) && (((((((int)blockIdx.x) % 98) / 7) * 2) + ((((int)threadIdx.x) & 7) >> 1)) < 29)) && (1 <= (((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) & 1) * 3)))) ? data[(((((((rc_outer_outer * 3136) + ((((int)threadIdx.x) >> 3) * 784)) + (((((int)blockIdx.x) % 98) / 7) * 56)) + (((((int)threadIdx.x) & 7) >> 1) * 28)) + ((((int)blockIdx.x) % 7) * 4)) + ((((int)threadIdx.x) & 1) * 3)) - 29)] : 0.000000e+00f);
      pad_temp_shared[((((int)threadIdx.x) * 3) + 1)] = (((1 <= ((((((int)blockIdx.x) % 98) / 7) * 2) + ((((int)threadIdx.x) & 7) >> 1))) && (((((((int)blockIdx.x) % 98) / 7) * 2) + ((((int)threadIdx.x) & 7) >> 1)) < 29)) ? data[(((((((rc_outer_outer * 3136) + ((((int)threadIdx.x) >> 3) * 784)) + (((((int)blockIdx.x) % 98) / 7) * 56)) + (((((int)threadIdx.x) & 7) >> 1) * 28)) + ((((int)blockIdx.x) % 7) * 4)) + ((((int)threadIdx.x) & 1) * 3)) - 28)] : 0.000000e+00f);
      pad_temp_shared[((((int)threadIdx.x) * 3) + 2)] = ((((1 <= ((((((int)blockIdx.x) % 98) / 7) * 2) + ((((int)threadIdx.x) & 7) >> 1))) && (((((((int)blockIdx.x) % 98) / 7) * 2) + ((((int)threadIdx.x) & 7) >> 1)) < 29)) && ((((((int)blockIdx.x) % 7) * 4) + ((((int)threadIdx.x) & 1) * 3)) < 27)) ? data[(((((((rc_outer_outer * 3136) + ((((int)threadIdx.x) >> 3) * 784)) + (((((int)blockIdx.x) % 98) / 7) * 56)) + (((((int)threadIdx.x) & 7) >> 1) * 28)) + ((((int)blockIdx.x) % 7) * 4)) + ((((int)threadIdx.x) & 1) * 3)) - 27)] : 0.000000e+00f);
    }
    *(float2*)(kernel_shared + (((int)threadIdx.x) * 2)) = *(float2*)(kernel + (((((((int)blockIdx.x) / 98) * 36864) + ((((int)threadIdx.x) / 18) * 1152)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) % 18) * 2)));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 128)) = *(float2*)(kernel + (((((((int)blockIdx.x) / 98) * 36864) + ((((((int)threadIdx.x) * 2) + 128) / 36) * 1152)) + (rc_outer_outer * 36)) + (((((int)threadIdx.x) * 2) + 20) % 36)));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 256)) = *(float2*)(kernel + (((((((int)blockIdx.x) / 98) * 36864) + ((((((int)threadIdx.x) * 2) + 256) / 36) * 1152)) + (rc_outer_outer * 36)) + (((((int)threadIdx.x) * 2) + 4) % 36)));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 384)) = *(float2*)(kernel + (((((((int)blockIdx.x) / 98) * 36864) + ((((((int)threadIdx.x) * 2) + 384) / 36) * 1152)) + (rc_outer_outer * 36)) + (((((int)threadIdx.x) * 2) + 24) % 36)));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 512)) = *(float2*)(kernel + (((((((int)blockIdx.x) / 98) * 36864) + ((((((int)threadIdx.x) * 2) + 512) / 36) * 1152)) + (rc_outer_outer * 36)) + (((((int)threadIdx.x) * 2) + 8) % 36)));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 640)) = *(float2*)(kernel + (((((((int)blockIdx.x) / 98) * 36864) + ((((((int)threadIdx.x) * 2) + 640) / 36) * 1152)) + (rc_outer_outer * 36)) + (((((int)threadIdx.x) * 2) + 28) % 36)));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 768)) = *(float2*)(kernel + (((((((int)blockIdx.x) / 98) * 36864) + ((((((int)threadIdx.x) * 2) + 768) / 36) * 1152)) + (rc_outer_outer * 36)) + (((((int)threadIdx.x) * 2) + 12) % 36)));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 896)) = *(float2*)(kernel + (((((((int)blockIdx.x) / 98) * 36864) + ((((((int)threadIdx.x) * 2) + 896) / 36) * 1152)) + (rc_outer_outer * 36)) + (((((int)threadIdx.x) * 2) + 32) % 36)));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 1024)) = *(float2*)(kernel + (((((((int)blockIdx.x) / 98) * 36864) + ((((((int)threadIdx.x) * 2) + 1024) / 36) * 1152)) + (rc_outer_outer * 36)) + (((((int)threadIdx.x) * 2) + 16) % 36)));
    __syncthreads();
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((int)threadIdx.x) & 3)] * kernel_shared[((((int)threadIdx.x) >> 2) * 72)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 6)] * kernel_shared[((((int)threadIdx.x) >> 2) * 72)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 6)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 3)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 12)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 3)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 12)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 6)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 18)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 6)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 24)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 9)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 30)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 9)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 30)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 12)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 36)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 12)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 36)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 15)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 42)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 15)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((int)threadIdx.x) & 3)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 36)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 6)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 36)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 6)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 39)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 12)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 39)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 12)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 42)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 18)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 42)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 24)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 45)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 30)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 45)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 30)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 48)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 36)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 48)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 36)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 51)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 42)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 51)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 1)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 1)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 7)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 1)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 7)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 4)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 13)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 4)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 13)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 7)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 19)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 7)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 25)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 10)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 31)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 10)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 31)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 13)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 37)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 13)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 37)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 16)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 43)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 16)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 1)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 37)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 7)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 37)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 7)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 40)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 13)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 40)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 13)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 43)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 19)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 43)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 25)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 46)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 31)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 46)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 31)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 49)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 37)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 49)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 37)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 52)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 43)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 52)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 2)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 2)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 8)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 2)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 8)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 5)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 14)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 5)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 14)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 8)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 20)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 8)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 26)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 11)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 32)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 11)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 32)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 14)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 38)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 14)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 38)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 17)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 44)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 17)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 2)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 38)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 8)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 38)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 8)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 41)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 14)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 41)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 14)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 44)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 20)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 44)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 26)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 47)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 32)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 47)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 32)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 50)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 38)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 50)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 38)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 53)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 44)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 53)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 48)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 18)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 54)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 18)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 54)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 21)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 60)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 21)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 60)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 24)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 66)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 24)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 72)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 27)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 78)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 27)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 78)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 30)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 84)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 30)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 84)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 33)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 90)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 33)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 48)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 54)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 54)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 54)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 54)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 57)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 60)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 57)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 60)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 60)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 66)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 60)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 72)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 63)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 78)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 63)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 78)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 66)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 84)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 66)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 84)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 69)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 90)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 69)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 49)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 19)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 55)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 19)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 55)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 22)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 61)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 22)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 61)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 25)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 67)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 25)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 73)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 28)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 79)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 28)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 79)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 31)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 85)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 31)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 85)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 34)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 91)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 34)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 49)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 55)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 55)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 55)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 55)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 58)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 61)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 58)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 61)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 61)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 67)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 61)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 73)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 64)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 79)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 64)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 79)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 67)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 85)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 67)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 85)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 70)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 91)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 70)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 50)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 20)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 56)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 20)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 56)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 23)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 62)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 23)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 62)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 26)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 68)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 26)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 74)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 29)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 80)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 29)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 80)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 32)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 86)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 32)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 86)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 35)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 92)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 35)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 50)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 56)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 56)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 56)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 56)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 59)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 62)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 59)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 62)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 62)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 68)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 62)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 74)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 65)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 80)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 65)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 80)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 68)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 86)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 68)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 86)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 71)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)threadIdx.x) & 3) + 92)] * kernel_shared[(((((int)threadIdx.x) >> 2) * 72) + 71)]));
  }
  for (int ff_inner = 0; ff_inner < 2; ++ff_inner) {
    for (int yy_inner = 0; yy_inner < 2; ++yy_inner) {
      conv2d_nchw[((((((((((int)blockIdx.x) / 98) * 25088) + ((((int)threadIdx.x) >> 2) * 1568)) + (ff_inner * 784)) + (((((int)blockIdx.x) % 98) / 7) * 56)) + (yy_inner * 28)) + ((((int)blockIdx.x) % 7) * 4)) + (((int)threadIdx.x) & 3))] = conv2d_nchw_local[((ff_inner * 2) + yy_inner)];
    }
  }
}


