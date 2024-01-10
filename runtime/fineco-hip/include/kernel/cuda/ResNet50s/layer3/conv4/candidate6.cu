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
extern "C" __global__ void __launch_bounds__(196) candidate6(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[8];
  __shared__ float pad_temp_shared[3136];
  __shared__ float kernel_shared[128];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  conv2d_nchw_local[7] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 32; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = data[(((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 3) * 196)) + ((int)threadIdx.x))];
    pad_temp_shared[(((int)threadIdx.x) + 196)] = data[((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 3) * 196)) + ((int)threadIdx.x)) + 784)];
    pad_temp_shared[(((int)threadIdx.x) + 392)] = data[((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 3) * 196)) + ((int)threadIdx.x)) + 1568)];
    pad_temp_shared[(((int)threadIdx.x) + 588)] = data[((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 3) * 196)) + ((int)threadIdx.x)) + 2352)];
    pad_temp_shared[(((int)threadIdx.x) + 784)] = data[((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 3) * 196)) + ((int)threadIdx.x)) + 3136)];
    pad_temp_shared[(((int)threadIdx.x) + 980)] = data[((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 3) * 196)) + ((int)threadIdx.x)) + 3920)];
    pad_temp_shared[(((int)threadIdx.x) + 1176)] = data[((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 3) * 196)) + ((int)threadIdx.x)) + 4704)];
    pad_temp_shared[(((int)threadIdx.x) + 1372)] = data[((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 3) * 196)) + ((int)threadIdx.x)) + 5488)];
    pad_temp_shared[(((int)threadIdx.x) + 1568)] = data[((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 3) * 196)) + ((int)threadIdx.x)) + 6272)];
    pad_temp_shared[(((int)threadIdx.x) + 1764)] = data[((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 3) * 196)) + ((int)threadIdx.x)) + 7056)];
    pad_temp_shared[(((int)threadIdx.x) + 1960)] = data[((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 3) * 196)) + ((int)threadIdx.x)) + 7840)];
    pad_temp_shared[(((int)threadIdx.x) + 2156)] = data[((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 3) * 196)) + ((int)threadIdx.x)) + 8624)];
    pad_temp_shared[(((int)threadIdx.x) + 2352)] = data[((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 3) * 196)) + ((int)threadIdx.x)) + 9408)];
    pad_temp_shared[(((int)threadIdx.x) + 2548)] = data[((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 3) * 196)) + ((int)threadIdx.x)) + 10192)];
    pad_temp_shared[(((int)threadIdx.x) + 2744)] = data[((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 3) * 196)) + ((int)threadIdx.x)) + 10976)];
    pad_temp_shared[(((int)threadIdx.x) + 2940)] = data[((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 3) * 196)) + ((int)threadIdx.x)) + 11760)];
    if (((int)threadIdx.x) < 128) {
      kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 512)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15))];
    }
    __syncthreads();
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) % 98) * 2)] * kernel_shared[((((int)threadIdx.x) / 98) * 64)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1)] * kernel_shared[((((int)threadIdx.x) / 98) * 64)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) % 98) * 2)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 16)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 16)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 196)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 1)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 197)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 1)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 196)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 17)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 197)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 17)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[((((int)threadIdx.x) % 98) * 2)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 32)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 32)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[((((int)threadIdx.x) % 98) * 2)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 48)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 48)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 196)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 33)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 197)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 33)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 196)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 49)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 197)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 49)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 392)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 2)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 393)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 2)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 392)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 18)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 393)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 18)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 588)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 3)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 589)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 3)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 588)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 19)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 589)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 19)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 392)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 34)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 393)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 34)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 392)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 50)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 393)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 50)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 588)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 35)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 589)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 35)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 588)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 51)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 589)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 51)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 784)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 4)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 785)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 4)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 784)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 20)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 785)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 20)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 980)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 5)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 981)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 5)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 980)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 21)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 981)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 21)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 784)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 36)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 785)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 36)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 784)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 52)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 785)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 52)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 980)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 37)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 981)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 37)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 980)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 53)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 981)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 53)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1176)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 6)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1177)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 6)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1176)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 22)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1177)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 22)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1372)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 7)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1373)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 7)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1372)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 23)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1373)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 23)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1176)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 38)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1177)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 38)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1176)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 54)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1177)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 54)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1372)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 39)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1373)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 39)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1372)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 55)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1373)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 55)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1568)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 8)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1569)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 8)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1568)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 24)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1569)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 24)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1764)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 9)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1765)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 9)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1764)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 25)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1765)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 25)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1568)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 40)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1569)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 40)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1568)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 56)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1569)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 56)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1764)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 41)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1765)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 41)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1764)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 57)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1765)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 57)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1960)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 10)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1961)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 10)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1960)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 26)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1961)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 26)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2156)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 11)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2157)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 11)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2156)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 27)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2157)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 27)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1960)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 42)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1961)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 42)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1960)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 58)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 1961)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 58)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2156)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 43)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2157)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 43)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2156)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 59)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2157)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 59)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2352)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 12)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2353)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 12)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2352)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 28)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2353)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 28)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2548)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 13)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2549)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 13)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2548)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 29)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2549)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 29)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2352)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 44)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2353)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 44)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2352)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 60)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2353)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 60)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2548)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 45)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2549)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 45)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2548)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 61)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2549)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 61)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2744)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 14)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2745)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 14)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2744)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 30)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2745)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 30)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2940)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 15)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2941)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 15)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2940)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 31)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2941)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 31)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2744)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 46)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2745)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 46)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2744)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 62)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2745)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 62)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2940)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 47)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2941)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 47)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2940)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 63)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)threadIdx.x) % 98) * 2) + 2941)] * kernel_shared[(((((int)threadIdx.x) / 98) * 64) + 63)]));
  }
  for (int ff_inner = 0; ff_inner < 4; ++ff_inner) {
    for (int xx_inner = 0; xx_inner < 2; ++xx_inner) {
      conv2d_nchw[(((((((((int)blockIdx.x) >> 2) * 6272) + ((((int)threadIdx.x) / 98) * 3136)) + (ff_inner * 784)) + ((((int)blockIdx.x) & 3) * 196)) + ((((int)threadIdx.x) % 98) * 2)) + xx_inner)] = conv2d_nchw_local[((ff_inner * 2) + xx_inner)];
    }
  }
}


