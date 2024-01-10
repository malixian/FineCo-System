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
extern "C" __global__ void __launch_bounds__(224) candidate1(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[224];
  __shared__ float pad_temp_shared[3712];
  __shared__ float kernel_shared[2304];
  for (int ff_c_outer_inner_init = 0; ff_c_outer_inner_init < 2; ++ff_c_outer_inner_init) {
    for (int xx_c_outer_inner_init = 0; xx_c_outer_inner_init < 8; ++xx_c_outer_inner_init) {
      conv2d_nchw_local[((ff_c_outer_inner_init * 56) + (xx_c_outer_inner_init * 7))] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_outer_inner_init * 56) + (xx_c_outer_inner_init * 7)) + 112)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_outer_inner_init * 56) + (xx_c_outer_inner_init * 7)) + 1)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_outer_inner_init * 56) + (xx_c_outer_inner_init * 7)) + 113)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_outer_inner_init * 56) + (xx_c_outer_inner_init * 7)) + 2)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_outer_inner_init * 56) + (xx_c_outer_inner_init * 7)) + 114)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_outer_inner_init * 56) + (xx_c_outer_inner_init * 7)) + 3)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_outer_inner_init * 56) + (xx_c_outer_inner_init * 7)) + 115)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_outer_inner_init * 56) + (xx_c_outer_inner_init * 7)) + 4)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_outer_inner_init * 56) + (xx_c_outer_inner_init * 7)) + 116)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_outer_inner_init * 56) + (xx_c_outer_inner_init * 7)) + 5)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_outer_inner_init * 56) + (xx_c_outer_inner_init * 7)) + 117)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_outer_inner_init * 56) + (xx_c_outer_inner_init * 7)) + 6)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_outer_inner_init * 56) + (xx_c_outer_inner_init * 7)) + 118)] = 0.000000e+00f;
    }
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 64; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = ((((1 <= (((((int)blockIdx.x) & 3) * 14) + (((int)threadIdx.x) / 58))) && (1 <= (((int)threadIdx.x) % 58))) && ((((int)threadIdx.x) % 58) < 57)) ? data[(((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 3) * 784)) + ((((int)threadIdx.x) / 58) * 56)) + (((int)threadIdx.x) % 58)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 224)] = (((1 <= ((((int)threadIdx.x) + 50) % 58)) && (((((int)threadIdx.x) + 50) % 58) < 57)) ? data[(((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 3) * 784)) + (((((int)threadIdx.x) + 224) / 58) * 56)) + ((((int)threadIdx.x) + 50) % 58)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 448)] = (((1 <= ((((int)threadIdx.x) + 42) % 58)) && (((((int)threadIdx.x) + 42) % 58) < 57)) ? data[(((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 3) * 784)) + (((((int)threadIdx.x) + 448) / 58) * 56)) + ((((int)threadIdx.x) + 42) % 58)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 672)] = (((((((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 672) / 58)) < 57) && (1 <= ((((int)threadIdx.x) + 34) % 58))) && (((((int)threadIdx.x) + 34) % 58) < 57)) ? data[(((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 3) * 784)) + (((((int)threadIdx.x) + 672) / 58) * 56)) + ((((int)threadIdx.x) + 34) % 58)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 896)] = (((((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((((int)threadIdx.x) >> 1) + 448) % 464) / 29))) && ((((((int)blockIdx.x) & 3) * 14) + ((((((int)threadIdx.x) >> 1) + 448) % 464) / 29)) < 57)) && (1 <= ((((int)threadIdx.x) + 26) % 58))) && (((((int)threadIdx.x) + 26) % 58) < 57)) ? data[((((((rc_outer_outer * 12544) + (((((int)threadIdx.x) + 896) / 928) * 3136)) + ((((int)blockIdx.x) & 3) * 784)) + (((((((int)threadIdx.x) >> 1) + 448) % 464) / 29) * 56)) + ((((int)threadIdx.x) + 26) % 58)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1120)] = (((1 <= ((((int)threadIdx.x) + 18) % 58)) && (((((int)threadIdx.x) + 18) % 58) < 57)) ? data[((((((rc_outer_outer * 12544) + (((((int)threadIdx.x) + 1120) / 928) * 3136)) + ((((int)blockIdx.x) & 3) * 784)) + (((((((int)threadIdx.x) >> 1) + 96) % 464) / 29) * 56)) + ((((int)threadIdx.x) + 18) % 58)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1344)] = (((1 <= ((((int)threadIdx.x) + 10) % 58)) && (((((int)threadIdx.x) + 10) % 58) < 57)) ? data[((((((rc_outer_outer * 12544) + (((((int)threadIdx.x) + 1344) / 928) * 3136)) + ((((int)blockIdx.x) & 3) * 784)) + (((((((int)threadIdx.x) >> 1) + 208) % 464) / 29) * 56)) + ((((int)threadIdx.x) + 10) % 58)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1568)] = (((1 <= ((((int)threadIdx.x) + 2) % 58)) && (((((int)threadIdx.x) + 2) % 58) < 57)) ? data[((((((rc_outer_outer * 12544) + (((((int)threadIdx.x) + 1568) / 928) * 3136)) + ((((int)blockIdx.x) & 3) * 784)) + (((((((int)threadIdx.x) >> 1) + 320) % 464) / 29) * 56)) + ((((int)threadIdx.x) + 2) % 58)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1792)] = (((((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((((int)threadIdx.x) >> 1) + 432) % 464) / 29))) && ((((((int)blockIdx.x) & 3) * 14) + ((((((int)threadIdx.x) >> 1) + 432) % 464) / 29)) < 57)) && (1 <= ((((int)threadIdx.x) + 52) % 58))) && (((((int)threadIdx.x) + 52) % 58) < 57)) ? data[((((((rc_outer_outer * 12544) + (((((int)threadIdx.x) + 1792) / 928) * 3136)) + ((((int)blockIdx.x) & 3) * 784)) + (((((((int)threadIdx.x) >> 1) + 432) % 464) / 29) * 56)) + ((((int)threadIdx.x) + 52) % 58)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 2016)] = (((1 <= ((((int)threadIdx.x) + 44) % 58)) && (((((int)threadIdx.x) + 44) % 58) < 57)) ? data[((((((rc_outer_outer * 12544) + (((((int)threadIdx.x) + 2016) / 928) * 3136)) + ((((int)blockIdx.x) & 3) * 784)) + (((((((int)threadIdx.x) >> 1) + 80) % 464) / 29) * 56)) + ((((int)threadIdx.x) + 44) % 58)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 2240)] = (((1 <= ((((int)threadIdx.x) + 36) % 58)) && (((((int)threadIdx.x) + 36) % 58) < 57)) ? data[((((((rc_outer_outer * 12544) + (((((int)threadIdx.x) + 2240) / 928) * 3136)) + ((((int)blockIdx.x) & 3) * 784)) + (((((((int)threadIdx.x) >> 1) + 192) % 464) / 29) * 56)) + ((((int)threadIdx.x) + 36) % 58)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 2464)] = (((1 <= ((((int)threadIdx.x) + 28) % 58)) && (((((int)threadIdx.x) + 28) % 58) < 57)) ? data[((((((rc_outer_outer * 12544) + (((((int)threadIdx.x) + 2464) / 928) * 3136)) + ((((int)blockIdx.x) & 3) * 784)) + (((((((int)threadIdx.x) >> 1) + 304) % 464) / 29) * 56)) + ((((int)threadIdx.x) + 28) % 58)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 2688)] = (((((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((((int)threadIdx.x) >> 1) + 416) % 464) / 29))) && ((((((int)blockIdx.x) & 3) * 14) + ((((((int)threadIdx.x) >> 1) + 416) % 464) / 29)) < 57)) && (1 <= ((((int)threadIdx.x) + 20) % 58))) && (((((int)threadIdx.x) + 20) % 58) < 57)) ? data[((((((rc_outer_outer * 12544) + (((((int)threadIdx.x) + 2688) / 928) * 3136)) + ((((int)blockIdx.x) & 3) * 784)) + (((((((int)threadIdx.x) >> 1) + 416) % 464) / 29) * 56)) + ((((int)threadIdx.x) + 20) % 58)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 2912)] = (((1 <= ((((int)threadIdx.x) + 12) % 58)) && (((((int)threadIdx.x) + 12) % 58) < 57)) ? data[((((((rc_outer_outer * 12544) + (((((int)threadIdx.x) + 2912) / 928) * 3136)) + ((((int)blockIdx.x) & 3) * 784)) + (((((((int)threadIdx.x) >> 1) + 64) % 464) / 29) * 56)) + ((((int)threadIdx.x) + 12) % 58)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 3136)] = (((1 <= ((((int)threadIdx.x) + 4) % 58)) && (((((int)threadIdx.x) + 4) % 58) < 57)) ? data[((((((rc_outer_outer * 12544) + (((((int)threadIdx.x) + 3136) / 928) * 3136)) + ((((int)blockIdx.x) & 3) * 784)) + (((((((int)threadIdx.x) >> 1) + 176) % 464) / 29) * 56)) + ((((int)threadIdx.x) + 4) % 58)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 3360)] = (((1 <= ((((int)threadIdx.x) + 54) % 58)) && (((((int)threadIdx.x) + 54) % 58) < 57)) ? data[((((((rc_outer_outer * 12544) + (((((int)threadIdx.x) + 3360) / 928) * 3136)) + ((((int)blockIdx.x) & 3) * 784)) + (((((((int)threadIdx.x) >> 1) + 288) % 464) / 29) * 56)) + ((((int)threadIdx.x) + 54) % 58)) - 57)] : 0.000000e+00f);
    if (((int)threadIdx.x) < 128) {
      pad_temp_shared[(((int)threadIdx.x) + 3584)] = (((((((((int)blockIdx.x) & 3) * 14) + ((((((int)threadIdx.x) >> 1) + 400) % 464) / 29)) < 57) && (1 <= ((((int)threadIdx.x) + 46) % 58))) && (((((int)threadIdx.x) + 46) % 58) < 57)) ? data[((((((rc_outer_outer * 12544) + (((((int)threadIdx.x) + 3584) / 928) * 3136)) + ((((int)blockIdx.x) & 3) * 784)) + (((((((int)threadIdx.x) >> 1) + 400) % 464) / 29) * 56)) + ((((int)threadIdx.x) + 46) % 58)) - 57)] : 0.000000e+00f);
    }
    kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) >> 2) * 147456) + ((((int)threadIdx.x) / 36) * 2304)) + (rc_outer_outer * 36)) + (((int)threadIdx.x) % 36))];
    kernel_shared[(((int)threadIdx.x) + 224)] = kernel[(((((((int)blockIdx.x) >> 2) * 147456) + (((((int)threadIdx.x) + 224) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 8) % 36))];
    kernel_shared[(((int)threadIdx.x) + 448)] = kernel[(((((((int)blockIdx.x) >> 2) * 147456) + (((((int)threadIdx.x) + 448) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 16) % 36))];
    kernel_shared[(((int)threadIdx.x) + 672)] = kernel[(((((((int)blockIdx.x) >> 2) * 147456) + (((((int)threadIdx.x) + 672) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 24) % 36))];
    kernel_shared[(((int)threadIdx.x) + 896)] = kernel[(((((((int)blockIdx.x) >> 2) * 147456) + (((((int)threadIdx.x) + 896) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 32) % 36))];
    kernel_shared[(((int)threadIdx.x) + 1120)] = kernel[(((((((int)blockIdx.x) >> 2) * 147456) + (((((int)threadIdx.x) + 1120) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 4) % 36))];
    kernel_shared[(((int)threadIdx.x) + 1344)] = kernel[(((((((int)blockIdx.x) >> 2) * 147456) + (((((int)threadIdx.x) + 1344) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 12) % 36))];
    kernel_shared[(((int)threadIdx.x) + 1568)] = kernel[(((((((int)blockIdx.x) >> 2) * 147456) + (((((int)threadIdx.x) + 1568) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 20) % 36))];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[(((((((int)blockIdx.x) >> 2) * 147456) + (((((int)threadIdx.x) + 1792) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 28) % 36))];
    kernel_shared[(((int)threadIdx.x) + 2016)] = kernel[((((((((int)blockIdx.x) >> 2) * 147456) + ((((int)threadIdx.x) / 36) * 2304)) + (rc_outer_outer * 36)) + (((int)threadIdx.x) % 36)) + 129024)];
    if (((int)threadIdx.x) < 64) {
      kernel_shared[(((int)threadIdx.x) + 2240)] = kernel[(((((((int)blockIdx.x) >> 2) * 147456) + (((((int)threadIdx.x) + 2240) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 8) % 36))];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 4; ++rc_outer_inner) {
      for (int ry_outer_inner = 0; ry_outer_inner < 3; ++ry_outer_inner) {
        for (int rx_outer_inner = 0; rx_outer_inner < 3; ++rx_outer_inner) {
          for (int ff_c_outer_inner = 0; ff_c_outer_inner < 2; ++ff_c_outer_inner) {
            for (int xx_c_outer_inner = 0; xx_c_outer_inner < 8; ++xx_c_outer_inner) {
              conv2d_nchw_local[((ff_c_outer_inner * 56) + (xx_c_outer_inner * 7))] = (conv2d_nchw_local[((ff_c_outer_inner * 56) + (xx_c_outer_inner * 7))] + (pad_temp_shared[(((((rc_outer_inner * 928) + (ry_outer_inner * 58)) + ((((int)threadIdx.x) % 7) * 58)) + (xx_c_outer_inner * 7)) + rx_outer_inner)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 72) + (ff_c_outer_inner * 36)) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_outer_inner)]));
              conv2d_nchw_local[(((ff_c_outer_inner * 56) + (xx_c_outer_inner * 7)) + 112)] = (conv2d_nchw_local[(((ff_c_outer_inner * 56) + (xx_c_outer_inner * 7)) + 112)] + (pad_temp_shared[((((((rc_outer_inner * 928) + (ry_outer_inner * 58)) + ((((int)threadIdx.x) % 7) * 58)) + (xx_c_outer_inner * 7)) + rx_outer_inner) + 406)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 72) + (ff_c_outer_inner * 36)) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_outer_inner)]));
              conv2d_nchw_local[(((ff_c_outer_inner * 56) + (xx_c_outer_inner * 7)) + 1)] = (conv2d_nchw_local[(((ff_c_outer_inner * 56) + (xx_c_outer_inner * 7)) + 1)] + (pad_temp_shared[((((((rc_outer_inner * 928) + (ry_outer_inner * 58)) + ((((int)threadIdx.x) % 7) * 58)) + (xx_c_outer_inner * 7)) + rx_outer_inner) + 1)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 72) + (ff_c_outer_inner * 36)) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_outer_inner)]));
              conv2d_nchw_local[(((ff_c_outer_inner * 56) + (xx_c_outer_inner * 7)) + 113)] = (conv2d_nchw_local[(((ff_c_outer_inner * 56) + (xx_c_outer_inner * 7)) + 113)] + (pad_temp_shared[((((((rc_outer_inner * 928) + (ry_outer_inner * 58)) + ((((int)threadIdx.x) % 7) * 58)) + (xx_c_outer_inner * 7)) + rx_outer_inner) + 407)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 72) + (ff_c_outer_inner * 36)) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_outer_inner)]));
              conv2d_nchw_local[(((ff_c_outer_inner * 56) + (xx_c_outer_inner * 7)) + 2)] = (conv2d_nchw_local[(((ff_c_outer_inner * 56) + (xx_c_outer_inner * 7)) + 2)] + (pad_temp_shared[((((((rc_outer_inner * 928) + (ry_outer_inner * 58)) + ((((int)threadIdx.x) % 7) * 58)) + (xx_c_outer_inner * 7)) + rx_outer_inner) + 2)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 72) + (ff_c_outer_inner * 36)) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_outer_inner)]));
              conv2d_nchw_local[(((ff_c_outer_inner * 56) + (xx_c_outer_inner * 7)) + 114)] = (conv2d_nchw_local[(((ff_c_outer_inner * 56) + (xx_c_outer_inner * 7)) + 114)] + (pad_temp_shared[((((((rc_outer_inner * 928) + (ry_outer_inner * 58)) + ((((int)threadIdx.x) % 7) * 58)) + (xx_c_outer_inner * 7)) + rx_outer_inner) + 408)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 72) + (ff_c_outer_inner * 36)) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_outer_inner)]));
              conv2d_nchw_local[(((ff_c_outer_inner * 56) + (xx_c_outer_inner * 7)) + 3)] = (conv2d_nchw_local[(((ff_c_outer_inner * 56) + (xx_c_outer_inner * 7)) + 3)] + (pad_temp_shared[((((((rc_outer_inner * 928) + (ry_outer_inner * 58)) + ((((int)threadIdx.x) % 7) * 58)) + (xx_c_outer_inner * 7)) + rx_outer_inner) + 3)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 72) + (ff_c_outer_inner * 36)) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_outer_inner)]));
              conv2d_nchw_local[(((ff_c_outer_inner * 56) + (xx_c_outer_inner * 7)) + 115)] = (conv2d_nchw_local[(((ff_c_outer_inner * 56) + (xx_c_outer_inner * 7)) + 115)] + (pad_temp_shared[((((((rc_outer_inner * 928) + (ry_outer_inner * 58)) + ((((int)threadIdx.x) % 7) * 58)) + (xx_c_outer_inner * 7)) + rx_outer_inner) + 409)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 72) + (ff_c_outer_inner * 36)) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_outer_inner)]));
              conv2d_nchw_local[(((ff_c_outer_inner * 56) + (xx_c_outer_inner * 7)) + 4)] = (conv2d_nchw_local[(((ff_c_outer_inner * 56) + (xx_c_outer_inner * 7)) + 4)] + (pad_temp_shared[((((((rc_outer_inner * 928) + (ry_outer_inner * 58)) + ((((int)threadIdx.x) % 7) * 58)) + (xx_c_outer_inner * 7)) + rx_outer_inner) + 4)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 72) + (ff_c_outer_inner * 36)) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_outer_inner)]));
              conv2d_nchw_local[(((ff_c_outer_inner * 56) + (xx_c_outer_inner * 7)) + 116)] = (conv2d_nchw_local[(((ff_c_outer_inner * 56) + (xx_c_outer_inner * 7)) + 116)] + (pad_temp_shared[((((((rc_outer_inner * 928) + (ry_outer_inner * 58)) + ((((int)threadIdx.x) % 7) * 58)) + (xx_c_outer_inner * 7)) + rx_outer_inner) + 410)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 72) + (ff_c_outer_inner * 36)) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_outer_inner)]));
              conv2d_nchw_local[(((ff_c_outer_inner * 56) + (xx_c_outer_inner * 7)) + 5)] = (conv2d_nchw_local[(((ff_c_outer_inner * 56) + (xx_c_outer_inner * 7)) + 5)] + (pad_temp_shared[((((((rc_outer_inner * 928) + (ry_outer_inner * 58)) + ((((int)threadIdx.x) % 7) * 58)) + (xx_c_outer_inner * 7)) + rx_outer_inner) + 5)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 72) + (ff_c_outer_inner * 36)) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_outer_inner)]));
              conv2d_nchw_local[(((ff_c_outer_inner * 56) + (xx_c_outer_inner * 7)) + 117)] = (conv2d_nchw_local[(((ff_c_outer_inner * 56) + (xx_c_outer_inner * 7)) + 117)] + (pad_temp_shared[((((((rc_outer_inner * 928) + (ry_outer_inner * 58)) + ((((int)threadIdx.x) % 7) * 58)) + (xx_c_outer_inner * 7)) + rx_outer_inner) + 411)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 72) + (ff_c_outer_inner * 36)) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_outer_inner)]));
              conv2d_nchw_local[(((ff_c_outer_inner * 56) + (xx_c_outer_inner * 7)) + 6)] = (conv2d_nchw_local[(((ff_c_outer_inner * 56) + (xx_c_outer_inner * 7)) + 6)] + (pad_temp_shared[((((((rc_outer_inner * 928) + (ry_outer_inner * 58)) + ((((int)threadIdx.x) % 7) * 58)) + (xx_c_outer_inner * 7)) + rx_outer_inner) + 6)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 72) + (ff_c_outer_inner * 36)) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_outer_inner)]));
              conv2d_nchw_local[(((ff_c_outer_inner * 56) + (xx_c_outer_inner * 7)) + 118)] = (conv2d_nchw_local[(((ff_c_outer_inner * 56) + (xx_c_outer_inner * 7)) + 118)] + (pad_temp_shared[((((((rc_outer_inner * 928) + (ry_outer_inner * 58)) + ((((int)threadIdx.x) % 7) * 58)) + (xx_c_outer_inner * 7)) + rx_outer_inner) + 412)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 72) + (ff_c_outer_inner * 36)) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_outer_inner)]));
            }
          }
        }
      }
    }
  }
  for (int ff_inner = 0; ff_inner < 2; ++ff_inner) {
    for (int xx_inner = 0; xx_inner < 56; ++xx_inner) {
      conv2d_nchw[(((((((((int)blockIdx.x) >> 2) * 200704) + ((((int)threadIdx.x) / 7) * 6272)) + (ff_inner * 3136)) + ((((int)blockIdx.x) & 3) * 784)) + ((((int)threadIdx.x) % 7) * 56)) + xx_inner)] = conv2d_nchw_local[((ff_inner * 56) + xx_inner)];
      conv2d_nchw[((((((((((int)blockIdx.x) >> 2) * 200704) + ((((int)threadIdx.x) / 7) * 6272)) + (ff_inner * 3136)) + ((((int)blockIdx.x) & 3) * 784)) + ((((int)threadIdx.x) % 7) * 56)) + xx_inner) + 392)] = conv2d_nchw_local[(((ff_inner * 56) + xx_inner) + 112)];
    }
  }
}


