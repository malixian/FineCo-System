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
extern "C" __global__ void __launch_bounds__(156) candidate2(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ compute, float* __restrict__ bias) {
  float conv2d_nchw[2];
  __shared__ float pad_temp_shared[1080];
  __shared__ float kernel_shared[5184];
  conv2d_nchw[0] = 0.000000e+00f;
  conv2d_nchw[1] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 8; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = (((((1 <= (((((int)threadIdx.x) % 45) / 15) + (((int)blockIdx.x) % 13))) && ((((((int)threadIdx.x) % 45) / 15) + (((int)blockIdx.x) % 13)) < 14)) && (1 <= (((int)threadIdx.x) % 15))) && ((((int)threadIdx.x) % 15) < 14)) ? data[((((((rc_outer_outer * 4056) + ((((int)threadIdx.x) / 45) * 169)) + (((((int)threadIdx.x) % 45) / 15) * 13)) + ((((int)blockIdx.x) % 13) * 13)) + (((int)threadIdx.x) % 15)) - 14)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 156)] = (((((1 <= (((((((int)threadIdx.x) / 3) + 7) % 15) / 5) + (((int)blockIdx.x) % 13))) && ((((((((int)threadIdx.x) / 3) + 7) % 15) / 5) + (((int)blockIdx.x) % 13)) < 14)) && (1 <= ((((int)threadIdx.x) + 6) % 15))) && (((((int)threadIdx.x) + 6) % 15) < 14)) ? data[((((((rc_outer_outer * 4056) + (((((int)threadIdx.x) + 156) / 45) * 169)) + (((((((int)threadIdx.x) / 3) + 7) % 15) / 5) * 13)) + ((((int)blockIdx.x) % 13) * 13)) + ((((int)threadIdx.x) + 6) % 15)) - 14)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 312)] = (((((1 <= (((((((int)threadIdx.x) / 3) + 14) % 15) / 5) + (((int)blockIdx.x) % 13))) && ((((((((int)threadIdx.x) / 3) + 14) % 15) / 5) + (((int)blockIdx.x) % 13)) < 14)) && (1 <= ((((int)threadIdx.x) + 12) % 15))) && (((((int)threadIdx.x) + 12) % 15) < 14)) ? data[((((((rc_outer_outer * 4056) + (((((int)threadIdx.x) + 312) / 45) * 169)) + (((((((int)threadIdx.x) / 3) + 14) % 15) / 5) * 13)) + ((((int)blockIdx.x) % 13) * 13)) + ((((int)threadIdx.x) + 12) % 15)) - 14)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 468)] = (((((1 <= (((((((int)threadIdx.x) / 3) + 6) % 15) / 5) + (((int)blockIdx.x) % 13))) && ((((((((int)threadIdx.x) / 3) + 6) % 15) / 5) + (((int)blockIdx.x) % 13)) < 14)) && (1 <= ((((int)threadIdx.x) + 3) % 15))) && (((((int)threadIdx.x) + 3) % 15) < 14)) ? data[((((((rc_outer_outer * 4056) + (((((int)threadIdx.x) + 468) / 45) * 169)) + (((((((int)threadIdx.x) / 3) + 6) % 15) / 5) * 13)) + ((((int)blockIdx.x) % 13) * 13)) + ((((int)threadIdx.x) + 3) % 15)) - 14)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 624)] = (((((1 <= (((((((int)threadIdx.x) / 3) + 13) % 15) / 5) + (((int)blockIdx.x) % 13))) && ((((((((int)threadIdx.x) / 3) + 13) % 15) / 5) + (((int)blockIdx.x) % 13)) < 14)) && (1 <= ((((int)threadIdx.x) + 9) % 15))) && (((((int)threadIdx.x) + 9) % 15) < 14)) ? data[((((((rc_outer_outer * 4056) + (((((int)threadIdx.x) + 624) / 45) * 169)) + (((((((int)threadIdx.x) / 3) + 13) % 15) / 5) * 13)) + ((((int)blockIdx.x) % 13) * 13)) + ((((int)threadIdx.x) + 9) % 15)) - 14)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 780)] = (((((1 <= ((((int)blockIdx.x) % 13) + (((((int)threadIdx.x) / 15) + 1) % 3))) && (((((int)blockIdx.x) % 13) + (((((int)threadIdx.x) / 15) + 1) % 3)) < 14)) && (1 <= (((int)threadIdx.x) % 15))) && ((((int)threadIdx.x) % 15) < 14)) ? data[((((((rc_outer_outer * 4056) + (((((int)threadIdx.x) + 780) / 45) * 169)) + ((((int)blockIdx.x) % 13) * 13)) + ((((((int)threadIdx.x) / 15) + 1) % 3) * 13)) + (((int)threadIdx.x) % 15)) - 14)] : 0.000000e+00f);
    if (((int)threadIdx.x) < 144) {
      pad_temp_shared[(((int)threadIdx.x) + 936)] = (((((1 <= (((((((int)threadIdx.x) / 3) + 12) % 15) / 5) + (((int)blockIdx.x) % 13))) && ((((((((int)threadIdx.x) / 3) + 12) % 15) / 5) + (((int)blockIdx.x) % 13)) < 14)) && (1 <= ((((int)threadIdx.x) + 6) % 15))) && (((((int)threadIdx.x) + 6) % 15) < 14)) ? data[((((((rc_outer_outer * 4056) + (((((int)threadIdx.x) + 936) / 45) * 169)) + (((((((int)threadIdx.x) / 3) + 12) % 15) / 5) * 13)) + ((((int)blockIdx.x) % 13) * 13)) + ((((int)threadIdx.x) + 6) % 15)) - 14)] : 0.000000e+00f);
    }
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)blockIdx.x) / 13) * 41472) + (rc_outer_outer * 216)) + ((int)threadIdx.x))];
    kernel_shared[(((int)threadIdx.x) + 156)] = kernel[((((((((int)blockIdx.x) / 13) * 41472) + (((((int)threadIdx.x) + 156) / 216) * 1728)) + (rc_outer_outer * 216)) + ((((((int)threadIdx.x) / 3) + 52) % 72) * 3)) + (((int)threadIdx.x) % 3))];
    kernel_shared[(((int)threadIdx.x) + 312)] = kernel[((((((((int)blockIdx.x) / 13) * 41472) + (((((int)threadIdx.x) + 312) / 216) * 1728)) + (rc_outer_outer * 216)) + ((((((int)threadIdx.x) / 3) + 32) % 72) * 3)) + (((int)threadIdx.x) % 3))];
    kernel_shared[(((int)threadIdx.x) + 468)] = kernel[((((((((int)blockIdx.x) / 13) * 41472) + (((((int)threadIdx.x) + 468) / 216) * 1728)) + (rc_outer_outer * 216)) + (((((int)threadIdx.x) / 3) + 12) * 3)) + (((int)threadIdx.x) % 3))];
    kernel_shared[(((int)threadIdx.x) + 624)] = kernel[((((((((int)blockIdx.x) / 13) * 41472) + (((((int)threadIdx.x) + 624) / 216) * 1728)) + (rc_outer_outer * 216)) + ((((((int)threadIdx.x) / 3) + 64) % 72) * 3)) + (((int)threadIdx.x) % 3))];
    kernel_shared[(((int)threadIdx.x) + 780)] = kernel[((((((((int)blockIdx.x) / 13) * 41472) + (((((int)threadIdx.x) + 780) / 216) * 1728)) + (rc_outer_outer * 216)) + ((((((int)threadIdx.x) / 3) + 44) % 72) * 3)) + (((int)threadIdx.x) % 3))];
    kernel_shared[(((int)threadIdx.x) + 936)] = kernel[((((((((int)blockIdx.x) / 13) * 41472) + (((((int)threadIdx.x) + 936) / 216) * 1728)) + (rc_outer_outer * 216)) + ((((((int)threadIdx.x) / 3) + 24) % 72) * 3)) + (((int)threadIdx.x) % 3))];
    kernel_shared[(((int)threadIdx.x) + 1092)] = kernel[((((((((int)blockIdx.x) / 13) * 41472) + (((((int)threadIdx.x) + 1092) / 216) * 1728)) + (rc_outer_outer * 216)) + (((((int)threadIdx.x) / 3) + 4) * 3)) + (((int)threadIdx.x) % 3))];
    kernel_shared[(((int)threadIdx.x) + 1248)] = kernel[((((((((int)blockIdx.x) / 13) * 41472) + (((((int)threadIdx.x) + 1248) / 216) * 1728)) + (rc_outer_outer * 216)) + ((((((int)threadIdx.x) / 3) + 56) % 72) * 3)) + (((int)threadIdx.x) % 3))];
    kernel_shared[(((int)threadIdx.x) + 1404)] = kernel[((((((((int)blockIdx.x) / 13) * 41472) + (((((int)threadIdx.x) + 1404) / 216) * 1728)) + (rc_outer_outer * 216)) + ((((((int)threadIdx.x) / 3) + 36) % 72) * 3)) + (((int)threadIdx.x) % 3))];
    kernel_shared[(((int)threadIdx.x) + 1560)] = kernel[((((((((int)blockIdx.x) / 13) * 41472) + (((((int)threadIdx.x) + 1560) / 216) * 1728)) + (rc_outer_outer * 216)) + (((((int)threadIdx.x) / 3) + 16) * 3)) + (((int)threadIdx.x) % 3))];
    kernel_shared[(((int)threadIdx.x) + 1716)] = kernel[((((((((int)blockIdx.x) / 13) * 41472) + (((((int)threadIdx.x) + 1716) / 216) * 1728)) + (rc_outer_outer * 216)) + ((((((int)threadIdx.x) / 3) + 68) % 72) * 3)) + (((int)threadIdx.x) % 3))];
    kernel_shared[(((int)threadIdx.x) + 1872)] = kernel[((((((((int)blockIdx.x) / 13) * 41472) + (((((int)threadIdx.x) + 1872) / 216) * 1728)) + (rc_outer_outer * 216)) + ((((((int)threadIdx.x) / 3) + 48) % 72) * 3)) + (((int)threadIdx.x) % 3))];
    kernel_shared[(((int)threadIdx.x) + 2028)] = kernel[((((((((int)blockIdx.x) / 13) * 41472) + (((((int)threadIdx.x) + 2028) / 216) * 1728)) + (rc_outer_outer * 216)) + ((((((int)threadIdx.x) / 3) + 28) % 72) * 3)) + (((int)threadIdx.x) % 3))];
    kernel_shared[(((int)threadIdx.x) + 2184)] = kernel[((((((((int)blockIdx.x) / 13) * 41472) + (((((int)threadIdx.x) + 2184) / 216) * 1728)) + (rc_outer_outer * 216)) + (((((int)threadIdx.x) / 3) + 8) * 3)) + (((int)threadIdx.x) % 3))];
    kernel_shared[(((int)threadIdx.x) + 2340)] = kernel[((((((((int)blockIdx.x) / 13) * 41472) + (((((int)threadIdx.x) + 2340) / 216) * 1728)) + (rc_outer_outer * 216)) + ((((((int)threadIdx.x) / 3) + 60) % 72) * 3)) + (((int)threadIdx.x) % 3))];
    kernel_shared[(((int)threadIdx.x) + 2496)] = kernel[((((((((int)blockIdx.x) / 13) * 41472) + (((((int)threadIdx.x) + 2496) / 216) * 1728)) + (rc_outer_outer * 216)) + ((((((int)threadIdx.x) / 3) + 40) % 72) * 3)) + (((int)threadIdx.x) % 3))];
    kernel_shared[(((int)threadIdx.x) + 2652)] = kernel[((((((((int)blockIdx.x) / 13) * 41472) + (((((int)threadIdx.x) + 2652) / 216) * 1728)) + (rc_outer_outer * 216)) + (((((int)threadIdx.x) / 3) + 20) * 3)) + (((int)threadIdx.x) % 3))];
    kernel_shared[(((int)threadIdx.x) + 2808)] = kernel[(((((((int)blockIdx.x) / 13) * 41472) + (rc_outer_outer * 216)) + ((int)threadIdx.x)) + 22464)];
    kernel_shared[(((int)threadIdx.x) + 2964)] = kernel[((((((((int)blockIdx.x) / 13) * 41472) + (((((int)threadIdx.x) + 2964) / 216) * 1728)) + (rc_outer_outer * 216)) + ((((((int)threadIdx.x) / 3) + 52) % 72) * 3)) + (((int)threadIdx.x) % 3))];
    kernel_shared[(((int)threadIdx.x) + 3120)] = kernel[((((((((int)blockIdx.x) / 13) * 41472) + (((((int)threadIdx.x) + 3120) / 216) * 1728)) + (rc_outer_outer * 216)) + ((((((int)threadIdx.x) / 3) + 32) % 72) * 3)) + (((int)threadIdx.x) % 3))];
    kernel_shared[(((int)threadIdx.x) + 3276)] = kernel[((((((((int)blockIdx.x) / 13) * 41472) + (((((int)threadIdx.x) + 3276) / 216) * 1728)) + (rc_outer_outer * 216)) + (((((int)threadIdx.x) / 3) + 12) * 3)) + (((int)threadIdx.x) % 3))];
    kernel_shared[(((int)threadIdx.x) + 3432)] = kernel[((((((((int)blockIdx.x) / 13) * 41472) + (((((int)threadIdx.x) + 3432) / 216) * 1728)) + (rc_outer_outer * 216)) + ((((((int)threadIdx.x) / 3) + 64) % 72) * 3)) + (((int)threadIdx.x) % 3))];
    kernel_shared[(((int)threadIdx.x) + 3588)] = kernel[((((((((int)blockIdx.x) / 13) * 41472) + (((((int)threadIdx.x) + 3588) / 216) * 1728)) + (rc_outer_outer * 216)) + ((((((int)threadIdx.x) / 3) + 44) % 72) * 3)) + (((int)threadIdx.x) % 3))];
    kernel_shared[(((int)threadIdx.x) + 3744)] = kernel[((((((((int)blockIdx.x) / 13) * 41472) + (((((int)threadIdx.x) + 3744) / 216) * 1728)) + (rc_outer_outer * 216)) + ((((((int)threadIdx.x) / 3) + 24) % 72) * 3)) + (((int)threadIdx.x) % 3))];
    kernel_shared[(((int)threadIdx.x) + 3900)] = kernel[((((((((int)blockIdx.x) / 13) * 41472) + (((((int)threadIdx.x) + 3900) / 216) * 1728)) + (rc_outer_outer * 216)) + (((((int)threadIdx.x) / 3) + 4) * 3)) + (((int)threadIdx.x) % 3))];
    kernel_shared[(((int)threadIdx.x) + 4056)] = kernel[((((((((int)blockIdx.x) / 13) * 41472) + (((((int)threadIdx.x) + 4056) / 216) * 1728)) + (rc_outer_outer * 216)) + ((((((int)threadIdx.x) / 3) + 56) % 72) * 3)) + (((int)threadIdx.x) % 3))];
    kernel_shared[(((int)threadIdx.x) + 4212)] = kernel[((((((((int)blockIdx.x) / 13) * 41472) + (((((int)threadIdx.x) + 4212) / 216) * 1728)) + (rc_outer_outer * 216)) + ((((((int)threadIdx.x) / 3) + 36) % 72) * 3)) + (((int)threadIdx.x) % 3))];
    kernel_shared[(((int)threadIdx.x) + 4368)] = kernel[((((((((int)blockIdx.x) / 13) * 41472) + (((((int)threadIdx.x) + 4368) / 216) * 1728)) + (rc_outer_outer * 216)) + (((((int)threadIdx.x) / 3) + 16) * 3)) + (((int)threadIdx.x) % 3))];
    kernel_shared[(((int)threadIdx.x) + 4524)] = kernel[((((((((int)blockIdx.x) / 13) * 41472) + (((((int)threadIdx.x) + 4524) / 216) * 1728)) + (rc_outer_outer * 216)) + ((((((int)threadIdx.x) / 3) + 68) % 72) * 3)) + (((int)threadIdx.x) % 3))];
    kernel_shared[(((int)threadIdx.x) + 4680)] = kernel[((((((((int)blockIdx.x) / 13) * 41472) + (((((int)threadIdx.x) + 4680) / 216) * 1728)) + (rc_outer_outer * 216)) + ((((((int)threadIdx.x) / 3) + 48) % 72) * 3)) + (((int)threadIdx.x) % 3))];
    kernel_shared[(((int)threadIdx.x) + 4836)] = kernel[((((((((int)blockIdx.x) / 13) * 41472) + (((((int)threadIdx.x) + 4836) / 216) * 1728)) + (rc_outer_outer * 216)) + ((((((int)threadIdx.x) / 3) + 28) % 72) * 3)) + (((int)threadIdx.x) % 3))];
    kernel_shared[(((int)threadIdx.x) + 4992)] = kernel[((((((((int)blockIdx.x) / 13) * 41472) + (((((int)threadIdx.x) + 4992) / 216) * 1728)) + (rc_outer_outer * 216)) + (((((int)threadIdx.x) / 3) + 8) * 3)) + (((int)threadIdx.x) % 3))];
    if (((int)threadIdx.x) < 36) {
      kernel_shared[(((int)threadIdx.x) + 5148)] = kernel[((((((((int)blockIdx.x) / 13) * 41472) + (((((int)threadIdx.x) + 5148) / 216) * 1728)) + (rc_outer_outer * 216)) + (((((int)threadIdx.x) / 3) + 60) * 3)) + (((int)threadIdx.x) % 3))];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 6; ++rc_outer_inner) {
      for (int rx_outer_inner = 0; rx_outer_inner < 3; ++rx_outer_inner) {
        conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((rc_outer_inner * 180) + rx_outer_inner) + (((int)threadIdx.x) % 13))] * kernel_shared[((((((int)threadIdx.x) / 13) * 432) + (rc_outer_inner * 36)) + rx_outer_inner)]));
        conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((rc_outer_inner * 180) + rx_outer_inner) + (((int)threadIdx.x) % 13)) + 15)] * kernel_shared[(((((((int)threadIdx.x) / 13) * 432) + (rc_outer_inner * 36)) + rx_outer_inner) + 3)]));
        conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((rc_outer_inner * 180) + rx_outer_inner) + (((int)threadIdx.x) % 13)) + 30)] * kernel_shared[(((((((int)threadIdx.x) / 13) * 432) + (rc_outer_inner * 36)) + rx_outer_inner) + 6)]));
        conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((rc_outer_inner * 180) + rx_outer_inner) + (((int)threadIdx.x) % 13)) + 45)] * kernel_shared[(((((((int)threadIdx.x) / 13) * 432) + (rc_outer_inner * 36)) + rx_outer_inner) + 9)]));
        conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((rc_outer_inner * 180) + rx_outer_inner) + (((int)threadIdx.x) % 13)) + 60)] * kernel_shared[(((((((int)threadIdx.x) / 13) * 432) + (rc_outer_inner * 36)) + rx_outer_inner) + 12)]));
        conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((rc_outer_inner * 180) + rx_outer_inner) + (((int)threadIdx.x) % 13)) + 75)] * kernel_shared[(((((((int)threadIdx.x) / 13) * 432) + (rc_outer_inner * 36)) + rx_outer_inner) + 15)]));
        conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((rc_outer_inner * 180) + rx_outer_inner) + (((int)threadIdx.x) % 13)) + 90)] * kernel_shared[(((((((int)threadIdx.x) / 13) * 432) + (rc_outer_inner * 36)) + rx_outer_inner) + 18)]));
        conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((rc_outer_inner * 180) + rx_outer_inner) + (((int)threadIdx.x) % 13)) + 105)] * kernel_shared[(((((((int)threadIdx.x) / 13) * 432) + (rc_outer_inner * 36)) + rx_outer_inner) + 21)]));
        conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((rc_outer_inner * 180) + rx_outer_inner) + (((int)threadIdx.x) % 13)) + 120)] * kernel_shared[(((((((int)threadIdx.x) / 13) * 432) + (rc_outer_inner * 36)) + rx_outer_inner) + 24)]));
        conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((rc_outer_inner * 180) + rx_outer_inner) + (((int)threadIdx.x) % 13)) + 135)] * kernel_shared[(((((((int)threadIdx.x) / 13) * 432) + (rc_outer_inner * 36)) + rx_outer_inner) + 27)]));
        conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((rc_outer_inner * 180) + rx_outer_inner) + (((int)threadIdx.x) % 13)) + 150)] * kernel_shared[(((((((int)threadIdx.x) / 13) * 432) + (rc_outer_inner * 36)) + rx_outer_inner) + 30)]));
        conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((rc_outer_inner * 180) + rx_outer_inner) + (((int)threadIdx.x) % 13)) + 165)] * kernel_shared[(((((((int)threadIdx.x) / 13) * 432) + (rc_outer_inner * 36)) + rx_outer_inner) + 33)]));
        conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((rc_outer_inner * 180) + rx_outer_inner) + (((int)threadIdx.x) % 13))] * kernel_shared[(((((((int)threadIdx.x) / 13) * 432) + (rc_outer_inner * 36)) + rx_outer_inner) + 216)]));
        conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((rc_outer_inner * 180) + rx_outer_inner) + (((int)threadIdx.x) % 13)) + 15)] * kernel_shared[(((((((int)threadIdx.x) / 13) * 432) + (rc_outer_inner * 36)) + rx_outer_inner) + 219)]));
        conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((rc_outer_inner * 180) + rx_outer_inner) + (((int)threadIdx.x) % 13)) + 30)] * kernel_shared[(((((((int)threadIdx.x) / 13) * 432) + (rc_outer_inner * 36)) + rx_outer_inner) + 222)]));
        conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((rc_outer_inner * 180) + rx_outer_inner) + (((int)threadIdx.x) % 13)) + 45)] * kernel_shared[(((((((int)threadIdx.x) / 13) * 432) + (rc_outer_inner * 36)) + rx_outer_inner) + 225)]));
        conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((rc_outer_inner * 180) + rx_outer_inner) + (((int)threadIdx.x) % 13)) + 60)] * kernel_shared[(((((((int)threadIdx.x) / 13) * 432) + (rc_outer_inner * 36)) + rx_outer_inner) + 228)]));
        conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((rc_outer_inner * 180) + rx_outer_inner) + (((int)threadIdx.x) % 13)) + 75)] * kernel_shared[(((((((int)threadIdx.x) / 13) * 432) + (rc_outer_inner * 36)) + rx_outer_inner) + 231)]));
        conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((rc_outer_inner * 180) + rx_outer_inner) + (((int)threadIdx.x) % 13)) + 90)] * kernel_shared[(((((((int)threadIdx.x) / 13) * 432) + (rc_outer_inner * 36)) + rx_outer_inner) + 234)]));
        conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((rc_outer_inner * 180) + rx_outer_inner) + (((int)threadIdx.x) % 13)) + 105)] * kernel_shared[(((((((int)threadIdx.x) / 13) * 432) + (rc_outer_inner * 36)) + rx_outer_inner) + 237)]));
        conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((rc_outer_inner * 180) + rx_outer_inner) + (((int)threadIdx.x) % 13)) + 120)] * kernel_shared[(((((((int)threadIdx.x) / 13) * 432) + (rc_outer_inner * 36)) + rx_outer_inner) + 240)]));
        conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((rc_outer_inner * 180) + rx_outer_inner) + (((int)threadIdx.x) % 13)) + 135)] * kernel_shared[(((((((int)threadIdx.x) / 13) * 432) + (rc_outer_inner * 36)) + rx_outer_inner) + 243)]));
        conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((rc_outer_inner * 180) + rx_outer_inner) + (((int)threadIdx.x) % 13)) + 150)] * kernel_shared[(((((((int)threadIdx.x) / 13) * 432) + (rc_outer_inner * 36)) + rx_outer_inner) + 246)]));
        conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((rc_outer_inner * 180) + rx_outer_inner) + (((int)threadIdx.x) % 13)) + 165)] * kernel_shared[(((((((int)threadIdx.x) / 13) * 432) + (rc_outer_inner * 36)) + rx_outer_inner) + 249)]));
      }
    }
  }
  for (int i1_inner = 0; i1_inner < 2; ++i1_inner) {
    compute[((((((((int)blockIdx.x) / 13) * 4056) + ((((int)threadIdx.x) / 13) * 338)) + (i1_inner * 169)) + ((((int)blockIdx.x) % 13) * 13)) + (((int)threadIdx.x) % 13))] = max((conv2d_nchw[i1_inner] + bias[((((((int)blockIdx.x) / 13) * 24) + ((((int)threadIdx.x) / 13) * 2)) + i1_inner)]), 0.000000e+00f);
  }
}


