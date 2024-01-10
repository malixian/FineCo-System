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
extern "C" __global__ void __launch_bounds__(28) candidate0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[2];
  __shared__ float pad_temp_shared[448];
  __shared__ float kernel_shared[512];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 32; ++rc_outer_outer) {
    __syncthreads();
    int2 _1 = make_int2(((((rc_outer_outer * 3136) + (((int)threadIdx.x) * 14)) + (((int)blockIdx.x) % 7)))+(7*0), ((((rc_outer_outer * 3136) + (((int)threadIdx.x) * 14)) + (((int)blockIdx.x) % 7)))+(7*1));
    *(float2*)(pad_temp_shared + (((int)threadIdx.x) * 2)) = make_float2(data[_1.x],data[_1.y]);
    int2 _2 = make_int2((((((rc_outer_outer * 3136) + (((int)threadIdx.x) * 14)) + (((int)blockIdx.x) % 7)) + 392))+(7*0), (((((rc_outer_outer * 3136) + (((int)threadIdx.x) * 14)) + (((int)blockIdx.x) % 7)) + 392))+(7*1));
    *(float2*)(pad_temp_shared + ((((int)threadIdx.x) * 2) + 56)) = make_float2(data[_2.x],data[_2.y]);
    int2 _3 = make_int2((((((rc_outer_outer * 3136) + (((int)threadIdx.x) * 14)) + (((int)blockIdx.x) % 7)) + 784))+(7*0), (((((rc_outer_outer * 3136) + (((int)threadIdx.x) * 14)) + (((int)blockIdx.x) % 7)) + 784))+(7*1));
    *(float2*)(pad_temp_shared + ((((int)threadIdx.x) * 2) + 112)) = make_float2(data[_3.x],data[_3.y]);
    int2 _4 = make_int2((((((rc_outer_outer * 3136) + (((int)threadIdx.x) * 14)) + (((int)blockIdx.x) % 7)) + 1176))+(7*0), (((((rc_outer_outer * 3136) + (((int)threadIdx.x) * 14)) + (((int)blockIdx.x) % 7)) + 1176))+(7*1));
    *(float2*)(pad_temp_shared + ((((int)threadIdx.x) * 2) + 168)) = make_float2(data[_4.x],data[_4.y]);
    int2 _5 = make_int2((((((rc_outer_outer * 3136) + (((int)threadIdx.x) * 14)) + (((int)blockIdx.x) % 7)) + 1568))+(7*0), (((((rc_outer_outer * 3136) + (((int)threadIdx.x) * 14)) + (((int)blockIdx.x) % 7)) + 1568))+(7*1));
    *(float2*)(pad_temp_shared + ((((int)threadIdx.x) * 2) + 224)) = make_float2(data[_5.x],data[_5.y]);
    int2 _6 = make_int2((((((rc_outer_outer * 3136) + (((int)threadIdx.x) * 14)) + (((int)blockIdx.x) % 7)) + 1960))+(7*0), (((((rc_outer_outer * 3136) + (((int)threadIdx.x) * 14)) + (((int)blockIdx.x) % 7)) + 1960))+(7*1));
    *(float2*)(pad_temp_shared + ((((int)threadIdx.x) * 2) + 280)) = make_float2(data[_6.x],data[_6.y]);
    int2 _7 = make_int2((((((rc_outer_outer * 3136) + (((int)threadIdx.x) * 14)) + (((int)blockIdx.x) % 7)) + 2352))+(7*0), (((((rc_outer_outer * 3136) + (((int)threadIdx.x) * 14)) + (((int)blockIdx.x) % 7)) + 2352))+(7*1));
    *(float2*)(pad_temp_shared + ((((int)threadIdx.x) * 2) + 336)) = make_float2(data[_7.x],data[_7.y]);
    int2 _8 = make_int2((((((rc_outer_outer * 3136) + (((int)threadIdx.x) * 14)) + (((int)blockIdx.x) % 7)) + 2744))+(7*0), (((((rc_outer_outer * 3136) + (((int)threadIdx.x) * 14)) + (((int)blockIdx.x) % 7)) + 2744))+(7*1));
    *(float2*)(pad_temp_shared + ((((int)threadIdx.x) * 2) + 392)) = make_float2(data[_8.x],data[_8.y]);
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)blockIdx.x) / 7) * 16384) + (rc_outer_outer * 64)) + ((int)threadIdx.x))];
    kernel_shared[(((int)threadIdx.x) + 28)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (rc_outer_outer * 64)) + ((int)threadIdx.x)) + 28)];
    kernel_shared[(((int)threadIdx.x) + 56)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 56) >> 6) * 2048)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 56) & 63))];
    kernel_shared[(((int)threadIdx.x) + 84)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 84) >> 6) * 2048)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) + 20))];
    kernel_shared[(((int)threadIdx.x) + 112)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 112) >> 6) * 2048)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 48) & 63))];
    kernel_shared[(((int)threadIdx.x) + 140)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 140) >> 6) * 2048)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) + 12))];
    kernel_shared[(((int)threadIdx.x) + 168)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 168) >> 6) * 2048)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 40) & 63))];
    kernel_shared[(((int)threadIdx.x) + 196)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 196) >> 6) * 2048)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) + 4))];
    kernel_shared[(((int)threadIdx.x) + 224)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 224) >> 6) * 2048)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) + 32))];
    kernel_shared[(((int)threadIdx.x) + 252)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 252) >> 6) * 2048)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 60) & 63))];
    kernel_shared[(((int)threadIdx.x) + 280)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 280) >> 6) * 2048)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) + 24))];
    kernel_shared[(((int)threadIdx.x) + 308)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 308) >> 6) * 2048)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 52) & 63))];
    kernel_shared[(((int)threadIdx.x) + 336)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 336) >> 6) * 2048)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) + 16))];
    kernel_shared[(((int)threadIdx.x) + 364)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 364) >> 6) * 2048)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 44) & 63))];
    kernel_shared[(((int)threadIdx.x) + 392)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 392) >> 6) * 2048)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) + 8))];
    kernel_shared[(((int)threadIdx.x) + 420)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 420) >> 6) * 2048)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) + 36))];
    kernel_shared[(((int)threadIdx.x) + 448)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (rc_outer_outer * 64)) + ((int)threadIdx.x)) + 14336)];
    kernel_shared[(((int)threadIdx.x) + 476)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 476) >> 6) * 2048)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) + 28))];
    if (((int)threadIdx.x) < 8) {
      kernel_shared[(((int)threadIdx.x) + 504)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 504) >> 6) * 2048)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) + 56))];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 4; ++rc_outer_inner) {
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((rc_outer_inner * 112) + (((int)threadIdx.x) % 7))] * kernel_shared[(((((int)threadIdx.x) / 7) * 64) + (rc_outer_inner * 16))]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((rc_outer_inner * 112) + (((int)threadIdx.x) % 7))] * kernel_shared[((((((int)threadIdx.x) / 7) * 64) + (rc_outer_inner * 16)) + 256)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 112) + (((int)threadIdx.x) % 7)) + 7)] * kernel_shared[((((((int)threadIdx.x) / 7) * 64) + (rc_outer_inner * 16)) + 1)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 112) + (((int)threadIdx.x) % 7)) + 7)] * kernel_shared[((((((int)threadIdx.x) / 7) * 64) + (rc_outer_inner * 16)) + 257)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 112) + (((int)threadIdx.x) % 7)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 7) * 64) + (rc_outer_inner * 16)) + 2)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 112) + (((int)threadIdx.x) % 7)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 7) * 64) + (rc_outer_inner * 16)) + 258)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 112) + (((int)threadIdx.x) % 7)) + 21)] * kernel_shared[((((((int)threadIdx.x) / 7) * 64) + (rc_outer_inner * 16)) + 3)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 112) + (((int)threadIdx.x) % 7)) + 21)] * kernel_shared[((((((int)threadIdx.x) / 7) * 64) + (rc_outer_inner * 16)) + 259)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 112) + (((int)threadIdx.x) % 7)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 7) * 64) + (rc_outer_inner * 16)) + 4)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 112) + (((int)threadIdx.x) % 7)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 7) * 64) + (rc_outer_inner * 16)) + 260)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 112) + (((int)threadIdx.x) % 7)) + 35)] * kernel_shared[((((((int)threadIdx.x) / 7) * 64) + (rc_outer_inner * 16)) + 5)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 112) + (((int)threadIdx.x) % 7)) + 35)] * kernel_shared[((((((int)threadIdx.x) / 7) * 64) + (rc_outer_inner * 16)) + 261)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 112) + (((int)threadIdx.x) % 7)) + 42)] * kernel_shared[((((((int)threadIdx.x) / 7) * 64) + (rc_outer_inner * 16)) + 6)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 112) + (((int)threadIdx.x) % 7)) + 42)] * kernel_shared[((((((int)threadIdx.x) / 7) * 64) + (rc_outer_inner * 16)) + 262)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 112) + (((int)threadIdx.x) % 7)) + 49)] * kernel_shared[((((((int)threadIdx.x) / 7) * 64) + (rc_outer_inner * 16)) + 7)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 112) + (((int)threadIdx.x) % 7)) + 49)] * kernel_shared[((((((int)threadIdx.x) / 7) * 64) + (rc_outer_inner * 16)) + 263)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 112) + (((int)threadIdx.x) % 7)) + 56)] * kernel_shared[((((((int)threadIdx.x) / 7) * 64) + (rc_outer_inner * 16)) + 8)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 112) + (((int)threadIdx.x) % 7)) + 56)] * kernel_shared[((((((int)threadIdx.x) / 7) * 64) + (rc_outer_inner * 16)) + 264)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 112) + (((int)threadIdx.x) % 7)) + 63)] * kernel_shared[((((((int)threadIdx.x) / 7) * 64) + (rc_outer_inner * 16)) + 9)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 112) + (((int)threadIdx.x) % 7)) + 63)] * kernel_shared[((((((int)threadIdx.x) / 7) * 64) + (rc_outer_inner * 16)) + 265)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 112) + (((int)threadIdx.x) % 7)) + 70)] * kernel_shared[((((((int)threadIdx.x) / 7) * 64) + (rc_outer_inner * 16)) + 10)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 112) + (((int)threadIdx.x) % 7)) + 70)] * kernel_shared[((((((int)threadIdx.x) / 7) * 64) + (rc_outer_inner * 16)) + 266)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 112) + (((int)threadIdx.x) % 7)) + 77)] * kernel_shared[((((((int)threadIdx.x) / 7) * 64) + (rc_outer_inner * 16)) + 11)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 112) + (((int)threadIdx.x) % 7)) + 77)] * kernel_shared[((((((int)threadIdx.x) / 7) * 64) + (rc_outer_inner * 16)) + 267)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 112) + (((int)threadIdx.x) % 7)) + 84)] * kernel_shared[((((((int)threadIdx.x) / 7) * 64) + (rc_outer_inner * 16)) + 12)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 112) + (((int)threadIdx.x) % 7)) + 84)] * kernel_shared[((((((int)threadIdx.x) / 7) * 64) + (rc_outer_inner * 16)) + 268)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 112) + (((int)threadIdx.x) % 7)) + 91)] * kernel_shared[((((((int)threadIdx.x) / 7) * 64) + (rc_outer_inner * 16)) + 13)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 112) + (((int)threadIdx.x) % 7)) + 91)] * kernel_shared[((((((int)threadIdx.x) / 7) * 64) + (rc_outer_inner * 16)) + 269)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 112) + (((int)threadIdx.x) % 7)) + 98)] * kernel_shared[((((((int)threadIdx.x) / 7) * 64) + (rc_outer_inner * 16)) + 14)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 112) + (((int)threadIdx.x) % 7)) + 98)] * kernel_shared[((((((int)threadIdx.x) / 7) * 64) + (rc_outer_inner * 16)) + 270)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 112) + (((int)threadIdx.x) % 7)) + 105)] * kernel_shared[((((((int)threadIdx.x) / 7) * 64) + (rc_outer_inner * 16)) + 15)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 112) + (((int)threadIdx.x) % 7)) + 105)] * kernel_shared[((((((int)threadIdx.x) / 7) * 64) + (rc_outer_inner * 16)) + 271)]));
    }
  }
  conv2d_nchw[((((((int)blockIdx.x) / 7) * 392) + (((int)threadIdx.x) * 7)) + (((int)blockIdx.x) % 7))] = conv2d_nchw_local[0];
  conv2d_nchw[(((((((int)blockIdx.x) / 7) * 392) + (((int)threadIdx.x) * 7)) + (((int)blockIdx.x) % 7)) + 196)] = conv2d_nchw_local[1];
}


