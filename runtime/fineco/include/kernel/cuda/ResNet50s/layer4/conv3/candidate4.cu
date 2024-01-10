
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
  float conv2d_nchw_local[8];
  __shared__ float pad_temp_shared[32];
  __shared__ float kernel_shared[8192];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[7] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 32; ++rc_outer_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 32) {
      pad_temp_shared[((int)threadIdx.x)] = data[((((((rc_outer_outer * 1568) + ((((int)threadIdx.x) >> 2) * 196)) + ((((int)blockIdx.x) / 7) * 28)) + (((((int)threadIdx.x) & 3) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1))];
    }
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)threadIdx.x) >> 3) * 256) + (rc_outer_outer * 8)) + (((int)threadIdx.x) & 7))];
    kernel_shared[(((int)threadIdx.x) + 512)] = kernel[(((((((int)threadIdx.x) >> 3) * 256) + (rc_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 16384)];
    kernel_shared[(((int)threadIdx.x) + 1024)] = kernel[(((((((int)threadIdx.x) >> 3) * 256) + (rc_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 32768)];
    kernel_shared[(((int)threadIdx.x) + 1536)] = kernel[(((((((int)threadIdx.x) >> 3) * 256) + (rc_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 49152)];
    kernel_shared[(((int)threadIdx.x) + 2048)] = kernel[(((((((int)threadIdx.x) >> 3) * 256) + (rc_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 65536)];
    kernel_shared[(((int)threadIdx.x) + 2560)] = kernel[(((((((int)threadIdx.x) >> 3) * 256) + (rc_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 81920)];
    kernel_shared[(((int)threadIdx.x) + 3072)] = kernel[(((((((int)threadIdx.x) >> 3) * 256) + (rc_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 98304)];
    kernel_shared[(((int)threadIdx.x) + 3584)] = kernel[(((((((int)threadIdx.x) >> 3) * 256) + (rc_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 114688)];
    kernel_shared[(((int)threadIdx.x) + 4096)] = kernel[(((((((int)threadIdx.x) >> 3) * 256) + (rc_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 131072)];
    kernel_shared[(((int)threadIdx.x) + 4608)] = kernel[(((((((int)threadIdx.x) >> 3) * 256) + (rc_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 147456)];
    kernel_shared[(((int)threadIdx.x) + 5120)] = kernel[(((((((int)threadIdx.x) >> 3) * 256) + (rc_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 163840)];
    kernel_shared[(((int)threadIdx.x) + 5632)] = kernel[(((((((int)threadIdx.x) >> 3) * 256) + (rc_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 180224)];
    kernel_shared[(((int)threadIdx.x) + 6144)] = kernel[(((((((int)threadIdx.x) >> 3) * 256) + (rc_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 196608)];
    kernel_shared[(((int)threadIdx.x) + 6656)] = kernel[(((((((int)threadIdx.x) >> 3) * 256) + (rc_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 212992)];
    kernel_shared[(((int)threadIdx.x) + 7168)] = kernel[(((((((int)threadIdx.x) >> 3) * 256) + (rc_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 229376)];
    kernel_shared[(((int)threadIdx.x) + 7680)] = kernel[(((((((int)threadIdx.x) >> 3) * 256) + (rc_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 245760)];
    __syncthreads();
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[0] * kernel_shared[(((int)threadIdx.x) * 16)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx.x) * 16) + 8)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx.x) * 16) + 1)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx.x) * 16) + 9)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[1] * kernel_shared[(((int)threadIdx.x) * 16)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[1] * kernel_shared[((((int)threadIdx.x) * 16) + 8)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[5] * kernel_shared[((((int)threadIdx.x) * 16) + 1)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[5] * kernel_shared[((((int)threadIdx.x) * 16) + 9)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[2] * kernel_shared[(((int)threadIdx.x) * 16)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx.x) * 16) + 8)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx.x) * 16) + 1)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx.x) * 16) + 9)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[3] * kernel_shared[(((int)threadIdx.x) * 16)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[3] * kernel_shared[((((int)threadIdx.x) * 16) + 8)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[7] * kernel_shared[((((int)threadIdx.x) * 16) + 1)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[7] * kernel_shared[((((int)threadIdx.x) * 16) + 9)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx.x) * 16) + 2)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx.x) * 16) + 10)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[12] * kernel_shared[((((int)threadIdx.x) * 16) + 3)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[12] * kernel_shared[((((int)threadIdx.x) * 16) + 11)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.x) * 16) + 2)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.x) * 16) + 10)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.x) * 16) + 3)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.x) * 16) + 11)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[10] * kernel_shared[((((int)threadIdx.x) * 16) + 2)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[10] * kernel_shared[((((int)threadIdx.x) * 16) + 10)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[14] * kernel_shared[((((int)threadIdx.x) * 16) + 3)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[14] * kernel_shared[((((int)threadIdx.x) * 16) + 11)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.x) * 16) + 2)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.x) * 16) + 10)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.x) * 16) + 3)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.x) * 16) + 11)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[16] * kernel_shared[((((int)threadIdx.x) * 16) + 4)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[16] * kernel_shared[((((int)threadIdx.x) * 16) + 12)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[20] * kernel_shared[((((int)threadIdx.x) * 16) + 5)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[20] * kernel_shared[((((int)threadIdx.x) * 16) + 13)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.x) * 16) + 4)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.x) * 16) + 12)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[21] * kernel_shared[((((int)threadIdx.x) * 16) + 5)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[21] * kernel_shared[((((int)threadIdx.x) * 16) + 13)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[18] * kernel_shared[((((int)threadIdx.x) * 16) + 4)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[18] * kernel_shared[((((int)threadIdx.x) * 16) + 12)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[22] * kernel_shared[((((int)threadIdx.x) * 16) + 5)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[22] * kernel_shared[((((int)threadIdx.x) * 16) + 13)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[19] * kernel_shared[((((int)threadIdx.x) * 16) + 4)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[19] * kernel_shared[((((int)threadIdx.x) * 16) + 12)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[23] * kernel_shared[((((int)threadIdx.x) * 16) + 5)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[23] * kernel_shared[((((int)threadIdx.x) * 16) + 13)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[24] * kernel_shared[((((int)threadIdx.x) * 16) + 6)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[24] * kernel_shared[((((int)threadIdx.x) * 16) + 14)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[28] * kernel_shared[((((int)threadIdx.x) * 16) + 7)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[28] * kernel_shared[((((int)threadIdx.x) * 16) + 15)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[25] * kernel_shared[((((int)threadIdx.x) * 16) + 6)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[25] * kernel_shared[((((int)threadIdx.x) * 16) + 14)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[29] * kernel_shared[((((int)threadIdx.x) * 16) + 7)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[29] * kernel_shared[((((int)threadIdx.x) * 16) + 15)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[26] * kernel_shared[((((int)threadIdx.x) * 16) + 6)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[26] * kernel_shared[((((int)threadIdx.x) * 16) + 14)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[30] * kernel_shared[((((int)threadIdx.x) * 16) + 7)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[30] * kernel_shared[((((int)threadIdx.x) * 16) + 15)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[27] * kernel_shared[((((int)threadIdx.x) * 16) + 6)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[27] * kernel_shared[((((int)threadIdx.x) * 16) + 14)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[31] * kernel_shared[((((int)threadIdx.x) * 16) + 7)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[31] * kernel_shared[((((int)threadIdx.x) * 16) + 15)]));
  }
  for (int ff_inner = 0; ff_inner < 2; ++ff_inner) {
    for (int yy_inner = 0; yy_inner < 2; ++yy_inner) {
      for (int xx_inner = 0; xx_inner < 2; ++xx_inner) {
        conv2d_nchw[((((((((int)threadIdx.x) * 392) + (ff_inner * 196)) + ((((int)blockIdx.x) / 7) * 28)) + (yy_inner * 14)) + ((((int)blockIdx.x) % 7) * 2)) + xx_inner)] = conv2d_nchw_local[(((ff_inner * 4) + (yy_inner * 2)) + xx_inner)];
      }
    }
  }
}


