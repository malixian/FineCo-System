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
extern "C" __global__ void __launch_bounds__(56) candidate0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[2];
  __shared__ float pad_temp_shared[360];
  __shared__ float kernel_shared[1152];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 64; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[(((int)threadIdx.x) * 2)] = (((3 <= ((((int)threadIdx.x) * 2) % 45)) && (1 <= (((((int)blockIdx.x) % 7) * 2) + ((((int)threadIdx.x) * 2) % 3)))) ? data[((((((rc_outer_outer * 1568) + (((((int)threadIdx.x) * 2) / 45) * 196)) + ((((((int)threadIdx.x) * 2) % 45) / 3) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + ((((int)threadIdx.x) * 2) % 3)) - 15)] : 0.000000e+00f);
    pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] = (((3 <= (((((int)threadIdx.x) * 2) + 1) % 45)) && (1 <= (((((int)blockIdx.x) % 7) * 2) + (((((int)threadIdx.x) * 2) + 1) % 3)))) ? data[((((((rc_outer_outer * 1568) + ((((((int)threadIdx.x) * 2) + 1) / 45) * 196)) + (((((((int)threadIdx.x) * 2) + 1) % 45) / 3) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((((int)threadIdx.x) * 2) + 1) % 3)) - 15)] : 0.000000e+00f);
    pad_temp_shared[((((int)threadIdx.x) * 2) + 112)] = (((3 <= (((((int)threadIdx.x) * 2) + 22) % 45)) && (1 <= (((((int)blockIdx.x) % 7) * 2) + (((((int)threadIdx.x) * 2) + 1) % 3)))) ? data[((((((rc_outer_outer * 1568) + ((((((int)threadIdx.x) * 2) + 112) / 45) * 196)) + (((((((int)threadIdx.x) * 2) + 22) % 45) / 3) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((((int)threadIdx.x) * 2) + 1) % 3)) - 15)] : 0.000000e+00f);
    pad_temp_shared[((((int)threadIdx.x) * 2) + 113)] = (((3 <= (((((int)threadIdx.x) * 2) + 23) % 45)) && (1 <= (((((int)blockIdx.x) % 7) * 2) + (((((int)threadIdx.x) * 2) + 2) % 3)))) ? data[((((((rc_outer_outer * 1568) + ((((((int)threadIdx.x) * 2) + 113) / 45) * 196)) + (((((((int)threadIdx.x) * 2) + 23) % 45) / 3) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((((int)threadIdx.x) * 2) + 2) % 3)) - 15)] : 0.000000e+00f);
    pad_temp_shared[((((int)threadIdx.x) * 2) + 224)] = (((3 <= (((((int)threadIdx.x) * 2) + 44) % 45)) && (1 <= (((((int)blockIdx.x) % 7) * 2) + (((((int)threadIdx.x) * 2) + 2) % 3)))) ? data[((((((rc_outer_outer * 1568) + ((((((int)threadIdx.x) * 2) + 224) / 45) * 196)) + (((((((int)threadIdx.x) * 2) + 44) % 45) / 3) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((((int)threadIdx.x) * 2) + 2) % 3)) - 15)] : 0.000000e+00f);
    pad_temp_shared[((((int)threadIdx.x) * 2) + 225)] = (((3 <= ((((int)threadIdx.x) * 2) % 45)) && (1 <= (((((int)blockIdx.x) % 7) * 2) + ((((int)threadIdx.x) * 2) % 3)))) ? data[((((((rc_outer_outer * 1568) + (((((int)threadIdx.x) * 2) / 45) * 196)) + ((((((int)threadIdx.x) * 2) % 45) / 3) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + ((((int)threadIdx.x) * 2) % 3)) + 965)] : 0.000000e+00f);
    if (((int)threadIdx.x) < 12) {
      pad_temp_shared[((((int)threadIdx.x) * 2) + 336)] = ((1 <= (((((int)blockIdx.x) % 7) * 2) + ((((int)threadIdx.x) * 2) % 3))) ? data[((((((rc_outer_outer * 1568) + ((((((int)threadIdx.x) * 2) + 336) / 45) * 196)) + ((((((int)threadIdx.x) * 2) / 3) + 7) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + ((((int)threadIdx.x) * 2) % 3)) - 15)] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 12) {
      pad_temp_shared[((((int)threadIdx.x) * 2) + 337)] = ((1 <= (((((int)blockIdx.x) % 7) * 2) + (((((int)threadIdx.x) * 2) + 1) % 3))) ? data[((((((rc_outer_outer * 1568) + ((((((int)threadIdx.x) * 2) + 337) / 45) * 196)) + (((((((int)threadIdx.x) * 2) + 22) % 45) / 3) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((((int)threadIdx.x) * 2) + 1) % 3)) - 15)] : 0.000000e+00f);
    }
    kernel_shared[(((int)threadIdx.x) * 3)] = kernel[(((((((int)blockIdx.x) / 7) * 73728) + ((((int)threadIdx.x) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) % 24) * 3))];
    kernel_shared[((((int)threadIdx.x) * 3) + 1)] = kernel[((((((((int)blockIdx.x) / 7) * 73728) + ((((int)threadIdx.x) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) % 24) * 3)) + 1)];
    kernel_shared[((((int)threadIdx.x) * 3) + 2)] = kernel[((((((((int)blockIdx.x) / 7) * 73728) + ((((int)threadIdx.x) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) % 24) * 3)) + 2)];
    kernel_shared[((((int)threadIdx.x) * 3) + 168)] = kernel[(((((((int)blockIdx.x) / 7) * 73728) + (((((int)threadIdx.x) + 56) / 24) * 4608)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) + 8) % 24) * 3))];
    kernel_shared[((((int)threadIdx.x) * 3) + 169)] = kernel[((((((((int)blockIdx.x) / 7) * 73728) + (((((int)threadIdx.x) + 56) / 24) * 4608)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) + 8) % 24) * 3)) + 1)];
    kernel_shared[((((int)threadIdx.x) * 3) + 170)] = kernel[((((((((int)blockIdx.x) / 7) * 73728) + (((((int)threadIdx.x) + 56) / 24) * 4608)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) + 8) % 24) * 3)) + 2)];
    kernel_shared[((((int)threadIdx.x) * 3) + 336)] = kernel[(((((((int)blockIdx.x) / 7) * 73728) + (((((int)threadIdx.x) + 112) / 24) * 4608)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) + 16) % 24) * 3))];
    kernel_shared[((((int)threadIdx.x) * 3) + 337)] = kernel[((((((((int)blockIdx.x) / 7) * 73728) + (((((int)threadIdx.x) + 112) / 24) * 4608)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) + 16) % 24) * 3)) + 1)];
    kernel_shared[((((int)threadIdx.x) * 3) + 338)] = kernel[((((((((int)blockIdx.x) / 7) * 73728) + (((((int)threadIdx.x) + 112) / 24) * 4608)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) + 16) % 24) * 3)) + 2)];
    kernel_shared[((((int)threadIdx.x) * 3) + 504)] = kernel[((((((((int)blockIdx.x) / 7) * 73728) + ((((int)threadIdx.x) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) % 24) * 3)) + 32256)];
    kernel_shared[((((int)threadIdx.x) * 3) + 505)] = kernel[((((((((int)blockIdx.x) / 7) * 73728) + ((((int)threadIdx.x) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) % 24) * 3)) + 32257)];
    kernel_shared[((((int)threadIdx.x) * 3) + 506)] = kernel[((((((((int)blockIdx.x) / 7) * 73728) + ((((int)threadIdx.x) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) % 24) * 3)) + 32258)];
    kernel_shared[((((int)threadIdx.x) * 3) + 672)] = kernel[(((((((int)blockIdx.x) / 7) * 73728) + (((((int)threadIdx.x) + 224) / 24) * 4608)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) + 8) % 24) * 3))];
    kernel_shared[((((int)threadIdx.x) * 3) + 673)] = kernel[((((((((int)blockIdx.x) / 7) * 73728) + (((((int)threadIdx.x) + 224) / 24) * 4608)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) + 8) % 24) * 3)) + 1)];
    kernel_shared[((((int)threadIdx.x) * 3) + 674)] = kernel[((((((((int)blockIdx.x) / 7) * 73728) + (((((int)threadIdx.x) + 224) / 24) * 4608)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) + 8) % 24) * 3)) + 2)];
    kernel_shared[((((int)threadIdx.x) * 3) + 840)] = kernel[(((((((int)blockIdx.x) / 7) * 73728) + (((((int)threadIdx.x) + 280) / 24) * 4608)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) + 16) % 24) * 3))];
    kernel_shared[((((int)threadIdx.x) * 3) + 841)] = kernel[((((((((int)blockIdx.x) / 7) * 73728) + (((((int)threadIdx.x) + 280) / 24) * 4608)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) + 16) % 24) * 3)) + 1)];
    kernel_shared[((((int)threadIdx.x) * 3) + 842)] = kernel[((((((((int)blockIdx.x) / 7) * 73728) + (((((int)threadIdx.x) + 280) / 24) * 4608)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) + 16) % 24) * 3)) + 2)];
    if (((int)threadIdx.x) < 48) {
      kernel_shared[((((int)threadIdx.x) * 3) + 1008)] = kernel[((((((((int)blockIdx.x) / 7) * 73728) + ((((int)threadIdx.x) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) % 24) * 3)) + 64512)];
      kernel_shared[((((int)threadIdx.x) * 3) + 1009)] = kernel[((((((((int)blockIdx.x) / 7) * 73728) + ((((int)threadIdx.x) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) % 24) * 3)) + 64513)];
      kernel_shared[((((int)threadIdx.x) * 3) + 1010)] = kernel[((((((((int)blockIdx.x) / 7) * 73728) + ((((int)threadIdx.x) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) % 24) * 3)) + 64514)];
    }
    __syncthreads();
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) % 7) * 6)] * kernel_shared[((((int)threadIdx.x) / 7) * 144)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 1)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 1)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 2)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 2)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 45)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 9)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 46)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 10)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 47)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 11)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 90)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 18)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 91)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 19)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 92)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 20)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 135)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 27)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 136)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 28)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 137)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 29)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 180)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 36)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 181)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 37)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 182)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 38)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 225)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 45)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 226)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 46)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 227)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 47)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 270)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 54)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 271)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 55)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 272)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 56)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 315)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 63)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 316)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 64)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 317)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 65)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)threadIdx.x) % 7) * 6)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 72)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 1)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 73)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 2)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 74)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 45)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 81)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 46)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 82)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 47)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 83)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 90)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 90)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 91)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 91)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 92)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 92)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 135)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 99)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 136)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 100)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 137)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 101)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 180)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 108)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 181)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 109)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 182)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 110)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 225)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 117)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 226)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 118)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 227)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 119)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 270)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 126)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 271)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 127)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 272)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 128)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 315)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 135)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 316)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 136)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 317)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 137)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 3)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 3)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 4)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 4)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 5)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 5)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 48)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 12)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 49)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 13)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 50)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 14)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 93)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 21)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 94)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 22)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 95)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 23)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 138)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 30)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 139)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 31)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 140)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 32)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 183)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 39)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 184)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 40)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 185)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 41)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 228)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 48)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 229)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 49)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 230)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 50)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 273)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 57)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 274)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 58)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 275)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 59)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 318)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 66)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 319)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 67)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 320)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 68)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 3)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 75)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 4)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 76)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 5)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 77)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 48)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 84)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 49)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 85)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 50)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 86)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 93)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 93)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 94)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 94)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 95)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 95)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 138)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 102)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 139)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 103)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 140)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 104)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 183)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 111)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 184)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 112)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 185)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 113)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 228)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 120)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 229)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 121)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 230)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 122)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 273)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 129)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 274)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 130)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 275)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 131)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 318)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 138)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 319)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 139)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 320)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 140)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 6)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 6)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 7)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 7)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 8)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 8)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 51)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 15)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 52)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 16)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 53)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 17)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 96)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 24)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 97)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 25)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 98)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 26)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 141)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 33)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 142)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 34)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 143)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 35)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 186)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 42)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 187)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 43)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 188)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 44)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 231)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 51)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 232)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 52)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 233)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 53)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 276)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 60)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 277)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 61)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 278)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 62)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 321)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 69)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 322)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 70)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 323)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 71)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 6)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 78)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 7)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 79)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 8)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 80)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 51)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 87)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 52)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 88)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 53)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 89)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 96)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 96)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 97)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 97)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 98)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 98)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 141)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 105)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 142)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 106)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 143)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 107)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 186)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 114)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 187)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 115)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 188)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 116)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 231)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 123)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 232)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 124)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 233)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 125)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 276)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 132)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 277)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 133)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 278)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 134)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 321)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 141)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 322)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 142)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 7) * 6) + 323)] * kernel_shared[(((((int)threadIdx.x) / 7) * 144) + 143)]));
  }
  for (int ff_inner = 0; ff_inner < 2; ++ff_inner) {
    conv2d_nchw[((((((((int)blockIdx.x) / 7) * 784) + ((((int)threadIdx.x) / 7) * 98)) + (ff_inner * 49)) + ((((int)threadIdx.x) % 7) * 7)) + (((int)blockIdx.x) % 7))] = conv2d_nchw_local[ff_inner];
  }
}


