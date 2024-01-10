
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
extern "C" __global__ void __launch_bounds__(512) candidate4(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[2];
  __shared__ float pad_temp_shared[64];
  __shared__ float kernel_shared[9216];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 64; ++rc_outer_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 64) {
      pad_temp_shared[((int)threadIdx.x)] = (((((1 <= (((((int)blockIdx.x) / 7) * 2) + ((((int)threadIdx.x) & 15) >> 2))) && ((((((int)blockIdx.x) / 7) * 2) + ((((int)threadIdx.x) & 15) >> 2)) < 15)) && (1 <= (((((int)blockIdx.x) % 7) * 2) + (((int)threadIdx.x) & 3)))) && ((((((int)blockIdx.x) % 7) * 2) + (((int)threadIdx.x) & 3)) < 15)) ? data[(((((((rc_outer_outer * 784) + ((((int)threadIdx.x) >> 4) * 196)) + ((((int)blockIdx.x) / 7) * 28)) + (((((int)threadIdx.x) & 15) >> 2) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 3)) - 15)] : 0.000000e+00f);
    }
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)threadIdx.x) / 36) * 2304) + (rc_outer_outer * 36)) + (((int)threadIdx.x) % 36))];
    kernel_shared[(((int)threadIdx.x) + 512)] = kernel[(((((((int)threadIdx.x) + 512) / 36) * 2304) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 8) % 36))];
    kernel_shared[(((int)threadIdx.x) + 1024)] = kernel[(((((((int)threadIdx.x) + 1024) / 36) * 2304) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 16) % 36))];
    kernel_shared[(((int)threadIdx.x) + 1536)] = kernel[(((((((int)threadIdx.x) + 1536) / 36) * 2304) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 24) % 36))];
    kernel_shared[(((int)threadIdx.x) + 2048)] = kernel[(((((((int)threadIdx.x) + 2048) / 36) * 2304) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 32) % 36))];
    kernel_shared[(((int)threadIdx.x) + 2560)] = kernel[(((((((int)threadIdx.x) + 2560) / 36) * 2304) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 4) % 36))];
    kernel_shared[(((int)threadIdx.x) + 3072)] = kernel[(((((((int)threadIdx.x) + 3072) / 36) * 2304) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 12) % 36))];
    kernel_shared[(((int)threadIdx.x) + 3584)] = kernel[(((((((int)threadIdx.x) + 3584) / 36) * 2304) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 20) % 36))];
    kernel_shared[(((int)threadIdx.x) + 4096)] = kernel[(((((((int)threadIdx.x) + 4096) / 36) * 2304) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 28) % 36))];
    kernel_shared[(((int)threadIdx.x) + 4608)] = kernel[(((((((int)threadIdx.x) / 36) * 2304) + (rc_outer_outer * 36)) + (((int)threadIdx.x) % 36)) + 294912)];
    kernel_shared[(((int)threadIdx.x) + 5120)] = kernel[(((((((int)threadIdx.x) + 5120) / 36) * 2304) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 8) % 36))];
    kernel_shared[(((int)threadIdx.x) + 5632)] = kernel[(((((((int)threadIdx.x) + 5632) / 36) * 2304) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 16) % 36))];
    kernel_shared[(((int)threadIdx.x) + 6144)] = kernel[(((((((int)threadIdx.x) + 6144) / 36) * 2304) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 24) % 36))];
    kernel_shared[(((int)threadIdx.x) + 6656)] = kernel[(((((((int)threadIdx.x) + 6656) / 36) * 2304) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 32) % 36))];
    kernel_shared[(((int)threadIdx.x) + 7168)] = kernel[(((((((int)threadIdx.x) + 7168) / 36) * 2304) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 4) % 36))];
    kernel_shared[(((int)threadIdx.x) + 7680)] = kernel[(((((((int)threadIdx.x) + 7680) / 36) * 2304) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 12) % 36))];
    kernel_shared[(((int)threadIdx.x) + 8192)] = kernel[(((((((int)threadIdx.x) + 8192) / 36) * 2304) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 20) % 36))];
    kernel_shared[(((int)threadIdx.x) + 8704)] = kernel[(((((((int)threadIdx.x) + 8704) / 36) * 2304) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 28) % 36))];
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 4; ++rc_outer_inner) {
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((rc_outer_inner * 16) + ((((int)threadIdx.x) & 1) * 4))] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + (rc_outer_inner * 9))]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 16) + ((((int)threadIdx.x) & 1) * 4)) + 4)] * kernel_shared[((((((int)threadIdx.x) >> 1) * 36) + (rc_outer_inner * 9)) + 3)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 16) + ((((int)threadIdx.x) & 1) * 4)) + 8)] * kernel_shared[((((((int)threadIdx.x) >> 1) * 36) + (rc_outer_inner * 9)) + 6)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 16) + ((((int)threadIdx.x) & 1) * 4)) + 1)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 36) + (rc_outer_inner * 9))]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 16) + ((((int)threadIdx.x) & 1) * 4)) + 5)] * kernel_shared[((((((int)threadIdx.x) >> 1) * 36) + (rc_outer_inner * 9)) + 3)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 16) + ((((int)threadIdx.x) & 1) * 4)) + 9)] * kernel_shared[((((((int)threadIdx.x) >> 1) * 36) + (rc_outer_inner * 9)) + 6)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 16) + ((((int)threadIdx.x) & 1) * 4)) + 1)] * kernel_shared[((((((int)threadIdx.x) >> 1) * 36) + (rc_outer_inner * 9)) + 1)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 16) + ((((int)threadIdx.x) & 1) * 4)) + 5)] * kernel_shared[((((((int)threadIdx.x) >> 1) * 36) + (rc_outer_inner * 9)) + 4)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 16) + ((((int)threadIdx.x) & 1) * 4)) + 9)] * kernel_shared[((((((int)threadIdx.x) >> 1) * 36) + (rc_outer_inner * 9)) + 7)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 16) + ((((int)threadIdx.x) & 1) * 4)) + 2)] * kernel_shared[((((((int)threadIdx.x) >> 1) * 36) + (rc_outer_inner * 9)) + 1)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 16) + ((((int)threadIdx.x) & 1) * 4)) + 6)] * kernel_shared[((((((int)threadIdx.x) >> 1) * 36) + (rc_outer_inner * 9)) + 4)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 16) + ((((int)threadIdx.x) & 1) * 4)) + 10)] * kernel_shared[((((((int)threadIdx.x) >> 1) * 36) + (rc_outer_inner * 9)) + 7)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 16) + ((((int)threadIdx.x) & 1) * 4)) + 2)] * kernel_shared[((((((int)threadIdx.x) >> 1) * 36) + (rc_outer_inner * 9)) + 2)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 16) + ((((int)threadIdx.x) & 1) * 4)) + 6)] * kernel_shared[((((((int)threadIdx.x) >> 1) * 36) + (rc_outer_inner * 9)) + 5)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 16) + ((((int)threadIdx.x) & 1) * 4)) + 10)] * kernel_shared[((((((int)threadIdx.x) >> 1) * 36) + (rc_outer_inner * 9)) + 8)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 16) + ((((int)threadIdx.x) & 1) * 4)) + 3)] * kernel_shared[((((((int)threadIdx.x) >> 1) * 36) + (rc_outer_inner * 9)) + 2)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 16) + ((((int)threadIdx.x) & 1) * 4)) + 7)] * kernel_shared[((((((int)threadIdx.x) >> 1) * 36) + (rc_outer_inner * 9)) + 5)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 16) + ((((int)threadIdx.x) & 1) * 4)) + 11)] * kernel_shared[((((((int)threadIdx.x) >> 1) * 36) + (rc_outer_inner * 9)) + 8)]));
    }
  }
  for (int xx_inner = 0; xx_inner < 2; ++xx_inner) {
    conv2d_nchw[((((((((int)threadIdx.x) >> 1) * 196) + ((((int)blockIdx.x) / 7) * 28)) + ((((int)threadIdx.x) & 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + xx_inner)] = conv2d_nchw_local[xx_inner];
  }
}


