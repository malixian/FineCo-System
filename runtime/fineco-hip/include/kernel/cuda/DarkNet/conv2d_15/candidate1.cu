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
extern "C" __global__ void __launch_bounds__(112) candidate1(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ compute, float* __restrict__ bias) {
  float conv2d_nchw[14];
  __shared__ float pad_temp_shared[6272];
  __shared__ float kernel_shared[4096];
  conv2d_nchw[0] = 0.000000e+00f;
  conv2d_nchw[7] = 0.000000e+00f;
  conv2d_nchw[1] = 0.000000e+00f;
  conv2d_nchw[8] = 0.000000e+00f;
  conv2d_nchw[2] = 0.000000e+00f;
  conv2d_nchw[9] = 0.000000e+00f;
  conv2d_nchw[3] = 0.000000e+00f;
  conv2d_nchw[10] = 0.000000e+00f;
  conv2d_nchw[4] = 0.000000e+00f;
  conv2d_nchw[11] = 0.000000e+00f;
  conv2d_nchw[5] = 0.000000e+00f;
  conv2d_nchw[12] = 0.000000e+00f;
  conv2d_nchw[6] = 0.000000e+00f;
  conv2d_nchw[13] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 8; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = data[((rc_outer_outer * 6272) + ((int)threadIdx.x))];
    pad_temp_shared[(((int)threadIdx.x) + 112)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 112)];
    pad_temp_shared[(((int)threadIdx.x) + 224)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 224)];
    pad_temp_shared[(((int)threadIdx.x) + 336)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 336)];
    pad_temp_shared[(((int)threadIdx.x) + 448)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 448)];
    pad_temp_shared[(((int)threadIdx.x) + 560)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 560)];
    pad_temp_shared[(((int)threadIdx.x) + 672)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 672)];
    pad_temp_shared[(((int)threadIdx.x) + 784)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 784)];
    pad_temp_shared[(((int)threadIdx.x) + 896)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 896)];
    pad_temp_shared[(((int)threadIdx.x) + 1008)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 1008)];
    pad_temp_shared[(((int)threadIdx.x) + 1120)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 1120)];
    pad_temp_shared[(((int)threadIdx.x) + 1232)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 1232)];
    pad_temp_shared[(((int)threadIdx.x) + 1344)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 1344)];
    pad_temp_shared[(((int)threadIdx.x) + 1456)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 1456)];
    pad_temp_shared[(((int)threadIdx.x) + 1568)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 1568)];
    pad_temp_shared[(((int)threadIdx.x) + 1680)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 1680)];
    pad_temp_shared[(((int)threadIdx.x) + 1792)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 1792)];
    pad_temp_shared[(((int)threadIdx.x) + 1904)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 1904)];
    pad_temp_shared[(((int)threadIdx.x) + 2016)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 2016)];
    pad_temp_shared[(((int)threadIdx.x) + 2128)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 2128)];
    pad_temp_shared[(((int)threadIdx.x) + 2240)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 2240)];
    pad_temp_shared[(((int)threadIdx.x) + 2352)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 2352)];
    pad_temp_shared[(((int)threadIdx.x) + 2464)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 2464)];
    pad_temp_shared[(((int)threadIdx.x) + 2576)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 2576)];
    pad_temp_shared[(((int)threadIdx.x) + 2688)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 2688)];
    pad_temp_shared[(((int)threadIdx.x) + 2800)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 2800)];
    pad_temp_shared[(((int)threadIdx.x) + 2912)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 2912)];
    pad_temp_shared[(((int)threadIdx.x) + 3024)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 3024)];
    pad_temp_shared[(((int)threadIdx.x) + 3136)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 3136)];
    pad_temp_shared[(((int)threadIdx.x) + 3248)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 3248)];
    pad_temp_shared[(((int)threadIdx.x) + 3360)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 3360)];
    pad_temp_shared[(((int)threadIdx.x) + 3472)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 3472)];
    pad_temp_shared[(((int)threadIdx.x) + 3584)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 3584)];
    pad_temp_shared[(((int)threadIdx.x) + 3696)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 3696)];
    pad_temp_shared[(((int)threadIdx.x) + 3808)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 3808)];
    pad_temp_shared[(((int)threadIdx.x) + 3920)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 3920)];
    pad_temp_shared[(((int)threadIdx.x) + 4032)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 4032)];
    pad_temp_shared[(((int)threadIdx.x) + 4144)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 4144)];
    pad_temp_shared[(((int)threadIdx.x) + 4256)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 4256)];
    pad_temp_shared[(((int)threadIdx.x) + 4368)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 4368)];
    pad_temp_shared[(((int)threadIdx.x) + 4480)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 4480)];
    pad_temp_shared[(((int)threadIdx.x) + 4592)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 4592)];
    pad_temp_shared[(((int)threadIdx.x) + 4704)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 4704)];
    pad_temp_shared[(((int)threadIdx.x) + 4816)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 4816)];
    pad_temp_shared[(((int)threadIdx.x) + 4928)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 4928)];
    pad_temp_shared[(((int)threadIdx.x) + 5040)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 5040)];
    pad_temp_shared[(((int)threadIdx.x) + 5152)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 5152)];
    pad_temp_shared[(((int)threadIdx.x) + 5264)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 5264)];
    pad_temp_shared[(((int)threadIdx.x) + 5376)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 5376)];
    pad_temp_shared[(((int)threadIdx.x) + 5488)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 5488)];
    pad_temp_shared[(((int)threadIdx.x) + 5600)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 5600)];
    pad_temp_shared[(((int)threadIdx.x) + 5712)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 5712)];
    pad_temp_shared[(((int)threadIdx.x) + 5824)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 5824)];
    pad_temp_shared[(((int)threadIdx.x) + 5936)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 5936)];
    pad_temp_shared[(((int)threadIdx.x) + 6048)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 6048)];
    pad_temp_shared[(((int)threadIdx.x) + 6160)] = data[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 6160)];
    kernel_shared[((int)threadIdx.x)] = kernel[(((((int)blockIdx.x) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x))];
    kernel_shared[(((int)threadIdx.x) + 112)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 112) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 112) & 127))];
    kernel_shared[(((int)threadIdx.x) + 224)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 224) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 96) & 127))];
    kernel_shared[(((int)threadIdx.x) + 336)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 336) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 80) & 127))];
    kernel_shared[(((int)threadIdx.x) + 448)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 448) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 64) & 127))];
    kernel_shared[(((int)threadIdx.x) + 560)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 560) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 48) & 127))];
    kernel_shared[(((int)threadIdx.x) + 672)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 672) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 32) & 127))];
    kernel_shared[(((int)threadIdx.x) + 784)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 784) >> 7) * 1024)) + (rc_outer_outer * 128)) + (((int)threadIdx.x) + 16))];
    kernel_shared[(((int)threadIdx.x) + 896)] = kernel[((((((int)blockIdx.x) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 7168)];
    kernel_shared[(((int)threadIdx.x) + 1008)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 1008) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 112) & 127))];
    kernel_shared[(((int)threadIdx.x) + 1120)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 1120) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 96) & 127))];
    kernel_shared[(((int)threadIdx.x) + 1232)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 1232) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 80) & 127))];
    kernel_shared[(((int)threadIdx.x) + 1344)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 1344) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 64) & 127))];
    kernel_shared[(((int)threadIdx.x) + 1456)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 1456) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 48) & 127))];
    kernel_shared[(((int)threadIdx.x) + 1568)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 1568) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 32) & 127))];
    kernel_shared[(((int)threadIdx.x) + 1680)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 1680) >> 7) * 1024)) + (rc_outer_outer * 128)) + (((int)threadIdx.x) + 16))];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[((((((int)blockIdx.x) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 14336)];
    kernel_shared[(((int)threadIdx.x) + 1904)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 1904) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 112) & 127))];
    kernel_shared[(((int)threadIdx.x) + 2016)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 2016) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 96) & 127))];
    kernel_shared[(((int)threadIdx.x) + 2128)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 2128) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 80) & 127))];
    kernel_shared[(((int)threadIdx.x) + 2240)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 2240) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 64) & 127))];
    kernel_shared[(((int)threadIdx.x) + 2352)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 2352) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 48) & 127))];
    kernel_shared[(((int)threadIdx.x) + 2464)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 2464) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 32) & 127))];
    kernel_shared[(((int)threadIdx.x) + 2576)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 2576) >> 7) * 1024)) + (rc_outer_outer * 128)) + (((int)threadIdx.x) + 16))];
    kernel_shared[(((int)threadIdx.x) + 2688)] = kernel[((((((int)blockIdx.x) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 21504)];
    kernel_shared[(((int)threadIdx.x) + 2800)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 2800) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 112) & 127))];
    kernel_shared[(((int)threadIdx.x) + 2912)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 2912) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 96) & 127))];
    kernel_shared[(((int)threadIdx.x) + 3024)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 3024) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 80) & 127))];
    kernel_shared[(((int)threadIdx.x) + 3136)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 3136) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 64) & 127))];
    kernel_shared[(((int)threadIdx.x) + 3248)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 3248) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 48) & 127))];
    kernel_shared[(((int)threadIdx.x) + 3360)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 3360) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 32) & 127))];
    kernel_shared[(((int)threadIdx.x) + 3472)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 3472) >> 7) * 1024)) + (rc_outer_outer * 128)) + (((int)threadIdx.x) + 16))];
    kernel_shared[(((int)threadIdx.x) + 3584)] = kernel[((((((int)blockIdx.x) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 28672)];
    kernel_shared[(((int)threadIdx.x) + 3696)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 3696) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 112) & 127))];
    kernel_shared[(((int)threadIdx.x) + 3808)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 3808) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 96) & 127))];
    kernel_shared[(((int)threadIdx.x) + 3920)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 3920) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 80) & 127))];
    if (((int)threadIdx.x) < 64) {
      kernel_shared[(((int)threadIdx.x) + 4032)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 4032) >> 7) * 1024)) + (rc_outer_outer * 128)) + (((int)threadIdx.x) + 64))];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 4; ++rc_outer_inner) {
      for (int xx_outer_inner = 0; xx_outer_inner < 7; ++xx_outer_inner) {
        conv2d_nchw[xx_outer_inner] = (conv2d_nchw[xx_outer_inner] + (pad_temp_shared[(((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner)] * kernel_shared[(((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32))]));
        conv2d_nchw[(xx_outer_inner + 7)] = (conv2d_nchw[(xx_outer_inner + 7)] + (pad_temp_shared[(((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 2048)]));
        conv2d_nchw[xx_outer_inner] = (conv2d_nchw[xx_outer_inner] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 49)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 1)]));
        conv2d_nchw[(xx_outer_inner + 7)] = (conv2d_nchw[(xx_outer_inner + 7)] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 49)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 2049)]));
        conv2d_nchw[xx_outer_inner] = (conv2d_nchw[xx_outer_inner] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 98)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 2)]));
        conv2d_nchw[(xx_outer_inner + 7)] = (conv2d_nchw[(xx_outer_inner + 7)] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 98)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 2050)]));
        conv2d_nchw[xx_outer_inner] = (conv2d_nchw[xx_outer_inner] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 147)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 3)]));
        conv2d_nchw[(xx_outer_inner + 7)] = (conv2d_nchw[(xx_outer_inner + 7)] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 147)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 2051)]));
        conv2d_nchw[xx_outer_inner] = (conv2d_nchw[xx_outer_inner] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 196)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 4)]));
        conv2d_nchw[(xx_outer_inner + 7)] = (conv2d_nchw[(xx_outer_inner + 7)] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 196)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 2052)]));
        conv2d_nchw[xx_outer_inner] = (conv2d_nchw[xx_outer_inner] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 245)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 5)]));
        conv2d_nchw[(xx_outer_inner + 7)] = (conv2d_nchw[(xx_outer_inner + 7)] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 245)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 2053)]));
        conv2d_nchw[xx_outer_inner] = (conv2d_nchw[xx_outer_inner] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 294)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 6)]));
        conv2d_nchw[(xx_outer_inner + 7)] = (conv2d_nchw[(xx_outer_inner + 7)] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 294)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 2054)]));
        conv2d_nchw[xx_outer_inner] = (conv2d_nchw[xx_outer_inner] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 343)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 7)]));
        conv2d_nchw[(xx_outer_inner + 7)] = (conv2d_nchw[(xx_outer_inner + 7)] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 343)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 2055)]));
        conv2d_nchw[xx_outer_inner] = (conv2d_nchw[xx_outer_inner] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 392)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 8)]));
        conv2d_nchw[(xx_outer_inner + 7)] = (conv2d_nchw[(xx_outer_inner + 7)] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 392)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 2056)]));
        conv2d_nchw[xx_outer_inner] = (conv2d_nchw[xx_outer_inner] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 441)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 9)]));
        conv2d_nchw[(xx_outer_inner + 7)] = (conv2d_nchw[(xx_outer_inner + 7)] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 441)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 2057)]));
        conv2d_nchw[xx_outer_inner] = (conv2d_nchw[xx_outer_inner] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 490)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 10)]));
        conv2d_nchw[(xx_outer_inner + 7)] = (conv2d_nchw[(xx_outer_inner + 7)] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 490)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 2058)]));
        conv2d_nchw[xx_outer_inner] = (conv2d_nchw[xx_outer_inner] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 539)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 11)]));
        conv2d_nchw[(xx_outer_inner + 7)] = (conv2d_nchw[(xx_outer_inner + 7)] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 539)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 2059)]));
        conv2d_nchw[xx_outer_inner] = (conv2d_nchw[xx_outer_inner] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 588)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 12)]));
        conv2d_nchw[(xx_outer_inner + 7)] = (conv2d_nchw[(xx_outer_inner + 7)] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 588)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 2060)]));
        conv2d_nchw[xx_outer_inner] = (conv2d_nchw[xx_outer_inner] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 637)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 13)]));
        conv2d_nchw[(xx_outer_inner + 7)] = (conv2d_nchw[(xx_outer_inner + 7)] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 637)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 2061)]));
        conv2d_nchw[xx_outer_inner] = (conv2d_nchw[xx_outer_inner] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 686)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 14)]));
        conv2d_nchw[(xx_outer_inner + 7)] = (conv2d_nchw[(xx_outer_inner + 7)] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 686)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 2062)]));
        conv2d_nchw[xx_outer_inner] = (conv2d_nchw[xx_outer_inner] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 735)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 15)]));
        conv2d_nchw[(xx_outer_inner + 7)] = (conv2d_nchw[(xx_outer_inner + 7)] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 735)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 2063)]));
        conv2d_nchw[xx_outer_inner] = (conv2d_nchw[xx_outer_inner] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 784)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 16)]));
        conv2d_nchw[(xx_outer_inner + 7)] = (conv2d_nchw[(xx_outer_inner + 7)] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 784)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 2064)]));
        conv2d_nchw[xx_outer_inner] = (conv2d_nchw[xx_outer_inner] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 833)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 17)]));
        conv2d_nchw[(xx_outer_inner + 7)] = (conv2d_nchw[(xx_outer_inner + 7)] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 833)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 2065)]));
        conv2d_nchw[xx_outer_inner] = (conv2d_nchw[xx_outer_inner] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 882)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 18)]));
        conv2d_nchw[(xx_outer_inner + 7)] = (conv2d_nchw[(xx_outer_inner + 7)] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 882)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 2066)]));
        conv2d_nchw[xx_outer_inner] = (conv2d_nchw[xx_outer_inner] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 931)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 19)]));
        conv2d_nchw[(xx_outer_inner + 7)] = (conv2d_nchw[(xx_outer_inner + 7)] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 931)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 2067)]));
        conv2d_nchw[xx_outer_inner] = (conv2d_nchw[xx_outer_inner] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 980)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 20)]));
        conv2d_nchw[(xx_outer_inner + 7)] = (conv2d_nchw[(xx_outer_inner + 7)] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 980)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 2068)]));
        conv2d_nchw[xx_outer_inner] = (conv2d_nchw[xx_outer_inner] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 1029)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 21)]));
        conv2d_nchw[(xx_outer_inner + 7)] = (conv2d_nchw[(xx_outer_inner + 7)] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 1029)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 2069)]));
        conv2d_nchw[xx_outer_inner] = (conv2d_nchw[xx_outer_inner] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 1078)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 22)]));
        conv2d_nchw[(xx_outer_inner + 7)] = (conv2d_nchw[(xx_outer_inner + 7)] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 1078)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 2070)]));
        conv2d_nchw[xx_outer_inner] = (conv2d_nchw[xx_outer_inner] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 1127)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 23)]));
        conv2d_nchw[(xx_outer_inner + 7)] = (conv2d_nchw[(xx_outer_inner + 7)] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 1127)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 2071)]));
        conv2d_nchw[xx_outer_inner] = (conv2d_nchw[xx_outer_inner] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 1176)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 24)]));
        conv2d_nchw[(xx_outer_inner + 7)] = (conv2d_nchw[(xx_outer_inner + 7)] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 1176)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 2072)]));
        conv2d_nchw[xx_outer_inner] = (conv2d_nchw[xx_outer_inner] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 1225)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 25)]));
        conv2d_nchw[(xx_outer_inner + 7)] = (conv2d_nchw[(xx_outer_inner + 7)] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 1225)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 2073)]));
        conv2d_nchw[xx_outer_inner] = (conv2d_nchw[xx_outer_inner] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 1274)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 26)]));
        conv2d_nchw[(xx_outer_inner + 7)] = (conv2d_nchw[(xx_outer_inner + 7)] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 1274)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 2074)]));
        conv2d_nchw[xx_outer_inner] = (conv2d_nchw[xx_outer_inner] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 1323)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 27)]));
        conv2d_nchw[(xx_outer_inner + 7)] = (conv2d_nchw[(xx_outer_inner + 7)] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 1323)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 2075)]));
        conv2d_nchw[xx_outer_inner] = (conv2d_nchw[xx_outer_inner] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 1372)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 28)]));
        conv2d_nchw[(xx_outer_inner + 7)] = (conv2d_nchw[(xx_outer_inner + 7)] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 1372)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 2076)]));
        conv2d_nchw[xx_outer_inner] = (conv2d_nchw[xx_outer_inner] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 1421)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 29)]));
        conv2d_nchw[(xx_outer_inner + 7)] = (conv2d_nchw[(xx_outer_inner + 7)] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 1421)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 2077)]));
        conv2d_nchw[xx_outer_inner] = (conv2d_nchw[xx_outer_inner] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 1470)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 30)]));
        conv2d_nchw[(xx_outer_inner + 7)] = (conv2d_nchw[(xx_outer_inner + 7)] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 1470)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 2078)]));
        conv2d_nchw[xx_outer_inner] = (conv2d_nchw[xx_outer_inner] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 1519)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 31)]));
        conv2d_nchw[(xx_outer_inner + 7)] = (conv2d_nchw[(xx_outer_inner + 7)] + (pad_temp_shared[((((rc_outer_inner * 1568) + ((((int)threadIdx.x) % 7) * 7)) + xx_outer_inner) + 1519)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 32)) + 2079)]));
      }
    }
  }
  for (int i3_inner = 0; i3_inner < 7; ++i3_inner) {
    compute[(((((int)blockIdx.x) * 1568) + (((int)threadIdx.x) * 7)) + i3_inner)] = max((conv2d_nchw[i3_inner] + bias[((((int)blockIdx.x) * 32) + (((int)threadIdx.x) / 7))]), 0.000000e+00f);
    compute[((((((int)blockIdx.x) * 1568) + (((int)threadIdx.x) * 7)) + i3_inner) + 784)] = max((conv2d_nchw[(i3_inner + 7)] + bias[(((((int)blockIdx.x) * 32) + (((int)threadIdx.x) / 7)) + 16)]), 0.000000e+00f);
  }
}


