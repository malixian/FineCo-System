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
extern "C" __global__ void __launch_bounds__(128) candidate4(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[8];
  __shared__ float pad_temp_shared[50];
  __shared__ float kernel_shared[4608];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[7] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 128; ++rc_outer_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 50) {
      pad_temp_shared[((int)threadIdx.x)] = (((1 <= (((((int)blockIdx.x) / 7) * 4) + ((((int)threadIdx.x) % 25) / 5))) && (1 <= (((((int)blockIdx.x) % 7) * 4) + (((int)threadIdx.x) % 5)))) ? data[(((((((rc_outer_outer * 1568) + ((((int)threadIdx.x) / 25) * 784)) + ((((int)blockIdx.x) / 7) * 112)) + (((((int)threadIdx.x) % 25) / 5) * 28)) + ((((int)blockIdx.x) % 7) * 4)) + (((int)threadIdx.x) % 5)) - 29)] : 0.000000e+00f);
    }
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)threadIdx.x) / 18) * 2304) + (rc_outer_outer * 18)) + (((int)threadIdx.x) % 18))];
    kernel_shared[(((int)threadIdx.x) + 128)] = kernel[(((((((int)threadIdx.x) + 128) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 2) % 18))];
    kernel_shared[(((int)threadIdx.x) + 256)] = kernel[(((((((int)threadIdx.x) + 256) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 4) % 18))];
    kernel_shared[(((int)threadIdx.x) + 384)] = kernel[(((((((int)threadIdx.x) + 384) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 6) % 18))];
    kernel_shared[(((int)threadIdx.x) + 512)] = kernel[(((((((int)threadIdx.x) + 512) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 8) % 18))];
    kernel_shared[(((int)threadIdx.x) + 640)] = kernel[(((((((int)threadIdx.x) + 640) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 10) % 18))];
    kernel_shared[(((int)threadIdx.x) + 768)] = kernel[(((((((int)threadIdx.x) + 768) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 12) % 18))];
    kernel_shared[(((int)threadIdx.x) + 896)] = kernel[(((((((int)threadIdx.x) + 896) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 14) % 18))];
    kernel_shared[(((int)threadIdx.x) + 1024)] = kernel[(((((((int)threadIdx.x) + 1024) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 16) % 18))];
    kernel_shared[(((int)threadIdx.x) + 1152)] = kernel[(((((((int)threadIdx.x) / 18) * 2304) + (rc_outer_outer * 18)) + (((int)threadIdx.x) % 18)) + 147456)];
    kernel_shared[(((int)threadIdx.x) + 1280)] = kernel[(((((((int)threadIdx.x) + 1280) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 2) % 18))];
    kernel_shared[(((int)threadIdx.x) + 1408)] = kernel[(((((((int)threadIdx.x) + 1408) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 4) % 18))];
    kernel_shared[(((int)threadIdx.x) + 1536)] = kernel[(((((((int)threadIdx.x) + 1536) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 6) % 18))];
    kernel_shared[(((int)threadIdx.x) + 1664)] = kernel[(((((((int)threadIdx.x) + 1664) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 8) % 18))];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[(((((((int)threadIdx.x) + 1792) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 10) % 18))];
    kernel_shared[(((int)threadIdx.x) + 1920)] = kernel[(((((((int)threadIdx.x) + 1920) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 12) % 18))];
    kernel_shared[(((int)threadIdx.x) + 2048)] = kernel[(((((((int)threadIdx.x) + 2048) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 14) % 18))];
    kernel_shared[(((int)threadIdx.x) + 2176)] = kernel[(((((((int)threadIdx.x) + 2176) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 16) % 18))];
    kernel_shared[(((int)threadIdx.x) + 2304)] = kernel[(((((((int)threadIdx.x) / 18) * 2304) + (rc_outer_outer * 18)) + (((int)threadIdx.x) % 18)) + 294912)];
    kernel_shared[(((int)threadIdx.x) + 2432)] = kernel[(((((((int)threadIdx.x) + 2432) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 2) % 18))];
    kernel_shared[(((int)threadIdx.x) + 2560)] = kernel[(((((((int)threadIdx.x) + 2560) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 4) % 18))];
    kernel_shared[(((int)threadIdx.x) + 2688)] = kernel[(((((((int)threadIdx.x) + 2688) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 6) % 18))];
    kernel_shared[(((int)threadIdx.x) + 2816)] = kernel[(((((((int)threadIdx.x) + 2816) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 8) % 18))];
    kernel_shared[(((int)threadIdx.x) + 2944)] = kernel[(((((((int)threadIdx.x) + 2944) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 10) % 18))];
    kernel_shared[(((int)threadIdx.x) + 3072)] = kernel[(((((((int)threadIdx.x) + 3072) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 12) % 18))];
    kernel_shared[(((int)threadIdx.x) + 3200)] = kernel[(((((((int)threadIdx.x) + 3200) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 14) % 18))];
    kernel_shared[(((int)threadIdx.x) + 3328)] = kernel[(((((((int)threadIdx.x) + 3328) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 16) % 18))];
    kernel_shared[(((int)threadIdx.x) + 3456)] = kernel[(((((((int)threadIdx.x) / 18) * 2304) + (rc_outer_outer * 18)) + (((int)threadIdx.x) % 18)) + 442368)];
    kernel_shared[(((int)threadIdx.x) + 3584)] = kernel[(((((((int)threadIdx.x) + 3584) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 2) % 18))];
    kernel_shared[(((int)threadIdx.x) + 3712)] = kernel[(((((((int)threadIdx.x) + 3712) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 4) % 18))];
    kernel_shared[(((int)threadIdx.x) + 3840)] = kernel[(((((((int)threadIdx.x) + 3840) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 6) % 18))];
    kernel_shared[(((int)threadIdx.x) + 3968)] = kernel[(((((((int)threadIdx.x) + 3968) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 8) % 18))];
    kernel_shared[(((int)threadIdx.x) + 4096)] = kernel[(((((((int)threadIdx.x) + 4096) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 10) % 18))];
    kernel_shared[(((int)threadIdx.x) + 4224)] = kernel[(((((((int)threadIdx.x) + 4224) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 12) % 18))];
    kernel_shared[(((int)threadIdx.x) + 4352)] = kernel[(((((((int)threadIdx.x) + 4352) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 14) % 18))];
    kernel_shared[(((int)threadIdx.x) + 4480)] = kernel[(((((((int)threadIdx.x) + 4480) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 16) % 18))];
    __syncthreads();
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 1) * 2)] * kernel_shared[((((int)threadIdx.x) >> 1) * 36)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[((((int)threadIdx.x) & 1) * 2)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2304)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 10)] * kernel_shared[((((int)threadIdx.x) >> 1) * 36)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 10)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2304)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 5)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 3)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 5)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2307)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 15)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 3)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 15)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2307)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 10)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 6)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 10)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2310)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 20)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 6)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 20)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2310)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 25)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 9)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 25)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2313)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 35)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 9)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 35)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2313)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 30)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 12)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 30)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2316)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 40)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 12)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 40)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2316)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 35)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 15)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 35)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2319)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 45)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 15)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 45)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2319)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 1) * 2)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 18)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[((((int)threadIdx.x) & 1) * 2)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2322)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 10)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 18)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 10)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2322)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 5)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 21)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 5)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2325)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 15)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 21)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 15)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2325)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 10)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 24)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 10)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2328)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 20)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 24)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 20)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2328)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 25)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 27)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 25)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2331)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 35)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 27)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 35)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2331)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 30)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 30)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 30)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2334)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 40)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 30)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 40)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2334)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 35)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 33)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 35)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2337)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 45)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 33)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 45)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2337)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 1)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 1)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 1)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2305)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 11)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 1)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 11)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2305)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 6)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 4)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 6)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2308)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 16)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 4)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 16)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2308)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 11)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 7)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 11)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2311)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 21)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 7)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 21)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2311)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 26)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 10)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 26)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2314)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 36)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 10)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 36)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2314)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 31)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 13)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 31)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2317)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 41)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 13)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 41)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2317)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 36)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 16)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 36)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2320)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 46)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 16)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 46)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2320)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 1)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 19)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 1)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2323)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 11)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 19)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 11)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2323)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 6)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 22)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 6)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2326)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 16)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 22)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 16)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2326)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 11)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 25)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 11)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2329)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 21)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 25)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 21)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2329)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 26)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 28)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 26)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2332)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 36)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 28)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 36)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2332)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 31)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 31)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 31)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2335)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 41)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 31)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 41)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2335)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 36)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 34)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 36)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2338)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 46)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 34)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 46)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2338)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 2)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 2)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2306)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 12)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 12)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2306)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 7)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 5)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 7)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2309)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 17)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 5)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 17)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2309)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 12)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 8)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 12)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2312)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 22)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 8)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 22)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2312)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 27)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 11)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 27)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2315)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 37)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 11)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 37)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2315)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 32)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 14)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 32)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2318)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 42)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 14)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 42)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2318)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 37)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 17)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 37)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2321)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 47)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 17)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 47)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2321)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 2)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 20)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 2)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2324)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 12)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 20)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 12)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2324)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 7)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 23)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 7)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2327)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 17)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 23)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 17)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2327)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 12)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 26)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 12)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2330)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 22)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 26)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 22)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2330)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 27)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 29)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 27)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2333)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 37)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 29)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 37)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2333)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 32)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 32)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 32)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2336)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 42)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 32)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 42)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2336)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 37)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 35)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 37)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2339)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 47)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 35)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 47)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + 2339)]));
  }
  for (int ff_inner = 0; ff_inner < 2; ++ff_inner) {
    for (int yy_inner = 0; yy_inner < 2; ++yy_inner) {
      conv2d_nchw[(((((((((int)threadIdx.x) >> 1) * 392) + (ff_inner * 196)) + ((((int)blockIdx.x) / 7) * 28)) + (yy_inner * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1))] = conv2d_nchw_local[((ff_inner * 2) + yy_inner)];
      conv2d_nchw[((((((((((int)threadIdx.x) >> 1) * 392) + (ff_inner * 196)) + ((((int)blockIdx.x) / 7) * 28)) + (yy_inner * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1)) + 25088)] = conv2d_nchw_local[(((ff_inner * 2) + yy_inner) + 4)];
    }
  }
}


