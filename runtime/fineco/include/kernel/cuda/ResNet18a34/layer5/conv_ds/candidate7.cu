
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
extern "C" __global__ void __launch_bounds__(256) candidate7(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[1];
  __shared__ float pad_temp_shared[36];
  __shared__ float kernel_shared[9216];
  conv2d_nchw_local[0] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 64; ++rc_outer_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 36) {
      pad_temp_shared[((int)threadIdx.x)] = (((1 <= ((((((int)blockIdx.x) % 49) / 7) * 2) + ((((int)threadIdx.x) % 9) / 3))) && (1 <= (((((int)blockIdx.x) % 7) * 2) + (((int)threadIdx.x) % 3)))) ? data[(((((((rc_outer_outer * 784) + ((((int)threadIdx.x) / 9) * 196)) + (((((int)blockIdx.x) % 49) / 7) * 28)) + (((((int)threadIdx.x) % 9) / 3) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) % 3)) - 15)] : 0.000000e+00f);
    }
    kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) / 49) * 589824) + ((((int)threadIdx.x) / 36) * 2304)) + (rc_outer_outer * 36)) + (((int)threadIdx.x) % 36))];
    kernel_shared[(((int)threadIdx.x) + 256)] = kernel[(((((((int)blockIdx.x) / 49) * 589824) + (((((int)threadIdx.x) + 256) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 4) % 36))];
    kernel_shared[(((int)threadIdx.x) + 512)] = kernel[(((((((int)blockIdx.x) / 49) * 589824) + (((((int)threadIdx.x) + 512) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 8) % 36))];
    kernel_shared[(((int)threadIdx.x) + 768)] = kernel[(((((((int)blockIdx.x) / 49) * 589824) + (((((int)threadIdx.x) + 768) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 12) % 36))];
    kernel_shared[(((int)threadIdx.x) + 1024)] = kernel[(((((((int)blockIdx.x) / 49) * 589824) + (((((int)threadIdx.x) + 1024) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 16) % 36))];
    kernel_shared[(((int)threadIdx.x) + 1280)] = kernel[(((((((int)blockIdx.x) / 49) * 589824) + (((((int)threadIdx.x) + 1280) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 20) % 36))];
    kernel_shared[(((int)threadIdx.x) + 1536)] = kernel[(((((((int)blockIdx.x) / 49) * 589824) + (((((int)threadIdx.x) + 1536) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 24) % 36))];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[(((((((int)blockIdx.x) / 49) * 589824) + (((((int)threadIdx.x) + 1792) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 28) % 36))];
    kernel_shared[(((int)threadIdx.x) + 2048)] = kernel[(((((((int)blockIdx.x) / 49) * 589824) + (((((int)threadIdx.x) + 2048) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 32) % 36))];
    kernel_shared[(((int)threadIdx.x) + 2304)] = kernel[((((((((int)blockIdx.x) / 49) * 589824) + ((((int)threadIdx.x) / 36) * 2304)) + (rc_outer_outer * 36)) + (((int)threadIdx.x) % 36)) + 147456)];
    kernel_shared[(((int)threadIdx.x) + 2560)] = kernel[(((((((int)blockIdx.x) / 49) * 589824) + (((((int)threadIdx.x) + 2560) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 4) % 36))];
    kernel_shared[(((int)threadIdx.x) + 2816)] = kernel[(((((((int)blockIdx.x) / 49) * 589824) + (((((int)threadIdx.x) + 2816) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 8) % 36))];
    kernel_shared[(((int)threadIdx.x) + 3072)] = kernel[(((((((int)blockIdx.x) / 49) * 589824) + (((((int)threadIdx.x) + 3072) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 12) % 36))];
    kernel_shared[(((int)threadIdx.x) + 3328)] = kernel[(((((((int)blockIdx.x) / 49) * 589824) + (((((int)threadIdx.x) + 3328) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 16) % 36))];
    kernel_shared[(((int)threadIdx.x) + 3584)] = kernel[(((((((int)blockIdx.x) / 49) * 589824) + (((((int)threadIdx.x) + 3584) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 20) % 36))];
    kernel_shared[(((int)threadIdx.x) + 3840)] = kernel[(((((((int)blockIdx.x) / 49) * 589824) + (((((int)threadIdx.x) + 3840) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 24) % 36))];
    kernel_shared[(((int)threadIdx.x) + 4096)] = kernel[(((((((int)blockIdx.x) / 49) * 589824) + (((((int)threadIdx.x) + 4096) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 28) % 36))];
    kernel_shared[(((int)threadIdx.x) + 4352)] = kernel[(((((((int)blockIdx.x) / 49) * 589824) + (((((int)threadIdx.x) + 4352) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 32) % 36))];
    kernel_shared[(((int)threadIdx.x) + 4608)] = kernel[((((((((int)blockIdx.x) / 49) * 589824) + ((((int)threadIdx.x) / 36) * 2304)) + (rc_outer_outer * 36)) + (((int)threadIdx.x) % 36)) + 294912)];
    kernel_shared[(((int)threadIdx.x) + 4864)] = kernel[(((((((int)blockIdx.x) / 49) * 589824) + (((((int)threadIdx.x) + 4864) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 4) % 36))];
    kernel_shared[(((int)threadIdx.x) + 5120)] = kernel[(((((((int)blockIdx.x) / 49) * 589824) + (((((int)threadIdx.x) + 5120) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 8) % 36))];
    kernel_shared[(((int)threadIdx.x) + 5376)] = kernel[(((((((int)blockIdx.x) / 49) * 589824) + (((((int)threadIdx.x) + 5376) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 12) % 36))];
    kernel_shared[(((int)threadIdx.x) + 5632)] = kernel[(((((((int)blockIdx.x) / 49) * 589824) + (((((int)threadIdx.x) + 5632) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 16) % 36))];
    kernel_shared[(((int)threadIdx.x) + 5888)] = kernel[(((((((int)blockIdx.x) / 49) * 589824) + (((((int)threadIdx.x) + 5888) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 20) % 36))];
    kernel_shared[(((int)threadIdx.x) + 6144)] = kernel[(((((((int)blockIdx.x) / 49) * 589824) + (((((int)threadIdx.x) + 6144) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 24) % 36))];
    kernel_shared[(((int)threadIdx.x) + 6400)] = kernel[(((((((int)blockIdx.x) / 49) * 589824) + (((((int)threadIdx.x) + 6400) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 28) % 36))];
    kernel_shared[(((int)threadIdx.x) + 6656)] = kernel[(((((((int)blockIdx.x) / 49) * 589824) + (((((int)threadIdx.x) + 6656) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 32) % 36))];
    kernel_shared[(((int)threadIdx.x) + 6912)] = kernel[((((((((int)blockIdx.x) / 49) * 589824) + ((((int)threadIdx.x) / 36) * 2304)) + (rc_outer_outer * 36)) + (((int)threadIdx.x) % 36)) + 442368)];
    kernel_shared[(((int)threadIdx.x) + 7168)] = kernel[(((((((int)blockIdx.x) / 49) * 589824) + (((((int)threadIdx.x) + 7168) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 4) % 36))];
    kernel_shared[(((int)threadIdx.x) + 7424)] = kernel[(((((((int)blockIdx.x) / 49) * 589824) + (((((int)threadIdx.x) + 7424) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 8) % 36))];
    kernel_shared[(((int)threadIdx.x) + 7680)] = kernel[(((((((int)blockIdx.x) / 49) * 589824) + (((((int)threadIdx.x) + 7680) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 12) % 36))];
    kernel_shared[(((int)threadIdx.x) + 7936)] = kernel[(((((((int)blockIdx.x) / 49) * 589824) + (((((int)threadIdx.x) + 7936) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 16) % 36))];
    kernel_shared[(((int)threadIdx.x) + 8192)] = kernel[(((((((int)blockIdx.x) / 49) * 589824) + (((((int)threadIdx.x) + 8192) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 20) % 36))];
    kernel_shared[(((int)threadIdx.x) + 8448)] = kernel[(((((((int)blockIdx.x) / 49) * 589824) + (((((int)threadIdx.x) + 8448) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 24) % 36))];
    kernel_shared[(((int)threadIdx.x) + 8704)] = kernel[(((((((int)blockIdx.x) / 49) * 589824) + (((((int)threadIdx.x) + 8704) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 28) % 36))];
    kernel_shared[(((int)threadIdx.x) + 8960)] = kernel[(((((((int)blockIdx.x) / 49) * 589824) + (((((int)threadIdx.x) + 8960) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 32) % 36))];
    __syncthreads();
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[0] * kernel_shared[(((int)threadIdx.x) * 36)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[1] * kernel_shared[((((int)threadIdx.x) * 36) + 1)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx.x) * 36) + 2)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[3] * kernel_shared[((((int)threadIdx.x) * 36) + 3)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx.x) * 36) + 4)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[5] * kernel_shared[((((int)threadIdx.x) * 36) + 5)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx.x) * 36) + 6)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[7] * kernel_shared[((((int)threadIdx.x) * 36) + 7)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx.x) * 36) + 8)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.x) * 36) + 9)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[10] * kernel_shared[((((int)threadIdx.x) * 36) + 10)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.x) * 36) + 11)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[12] * kernel_shared[((((int)threadIdx.x) * 36) + 12)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.x) * 36) + 13)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[14] * kernel_shared[((((int)threadIdx.x) * 36) + 14)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.x) * 36) + 15)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[16] * kernel_shared[((((int)threadIdx.x) * 36) + 16)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.x) * 36) + 17)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[18] * kernel_shared[((((int)threadIdx.x) * 36) + 18)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[19] * kernel_shared[((((int)threadIdx.x) * 36) + 19)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[20] * kernel_shared[((((int)threadIdx.x) * 36) + 20)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[21] * kernel_shared[((((int)threadIdx.x) * 36) + 21)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[22] * kernel_shared[((((int)threadIdx.x) * 36) + 22)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[23] * kernel_shared[((((int)threadIdx.x) * 36) + 23)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[24] * kernel_shared[((((int)threadIdx.x) * 36) + 24)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[25] * kernel_shared[((((int)threadIdx.x) * 36) + 25)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[26] * kernel_shared[((((int)threadIdx.x) * 36) + 26)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[27] * kernel_shared[((((int)threadIdx.x) * 36) + 27)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[28] * kernel_shared[((((int)threadIdx.x) * 36) + 28)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[29] * kernel_shared[((((int)threadIdx.x) * 36) + 29)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[30] * kernel_shared[((((int)threadIdx.x) * 36) + 30)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[31] * kernel_shared[((((int)threadIdx.x) * 36) + 31)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[32] * kernel_shared[((((int)threadIdx.x) * 36) + 32)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[33] * kernel_shared[((((int)threadIdx.x) * 36) + 33)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[34] * kernel_shared[((((int)threadIdx.x) * 36) + 34)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[35] * kernel_shared[((((int)threadIdx.x) * 36) + 35)]));
  }
  conv2d_nchw[((((((int)blockIdx.x) / 49) * 12544) + (((int)threadIdx.x) * 49)) + (((int)blockIdx.x) % 49))] = conv2d_nchw_local[0];
}


