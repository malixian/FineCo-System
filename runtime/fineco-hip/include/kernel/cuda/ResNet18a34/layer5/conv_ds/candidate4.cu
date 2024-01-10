
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
extern "C" __global__ void __launch_bounds__(256) candidate4(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[2];
  __shared__ float pad_temp_shared[18];
  __shared__ float kernel_shared[9216];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 128; ++rc_outer_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 18) {
      pad_temp_shared[((int)threadIdx.x)] = (((1 <= (((((int)blockIdx.x) / 7) * 2) + ((((int)threadIdx.x) % 9) / 3))) && (1 <= (((((int)blockIdx.x) % 7) * 2) + (((int)threadIdx.x) % 3)))) ? data[(((((((rc_outer_outer * 392) + ((((int)threadIdx.x) / 9) * 196)) + ((((int)blockIdx.x) / 7) * 28)) + (((((int)threadIdx.x) % 9) / 3) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) % 3)) - 15)] : 0.000000e+00f);
    }
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)threadIdx.x) / 18) * 2304) + (rc_outer_outer * 18)) + (((int)threadIdx.x) % 18))];
    kernel_shared[(((int)threadIdx.x) + 256)] = kernel[(((((((int)threadIdx.x) + 256) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 4) % 18))];
    kernel_shared[(((int)threadIdx.x) + 512)] = kernel[(((((((int)threadIdx.x) + 512) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 8) % 18))];
    kernel_shared[(((int)threadIdx.x) + 768)] = kernel[(((((((int)threadIdx.x) + 768) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 12) % 18))];
    kernel_shared[(((int)threadIdx.x) + 1024)] = kernel[(((((((int)threadIdx.x) + 1024) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 16) % 18))];
    kernel_shared[(((int)threadIdx.x) + 1280)] = kernel[(((((((int)threadIdx.x) + 1280) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 2) % 18))];
    kernel_shared[(((int)threadIdx.x) + 1536)] = kernel[(((((((int)threadIdx.x) + 1536) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 6) % 18))];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[(((((((int)threadIdx.x) + 1792) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 10) % 18))];
    kernel_shared[(((int)threadIdx.x) + 2048)] = kernel[(((((((int)threadIdx.x) + 2048) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 14) % 18))];
    kernel_shared[(((int)threadIdx.x) + 2304)] = kernel[(((((((int)threadIdx.x) / 18) * 2304) + (rc_outer_outer * 18)) + (((int)threadIdx.x) % 18)) + 294912)];
    kernel_shared[(((int)threadIdx.x) + 2560)] = kernel[(((((((int)threadIdx.x) + 2560) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 4) % 18))];
    kernel_shared[(((int)threadIdx.x) + 2816)] = kernel[(((((((int)threadIdx.x) + 2816) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 8) % 18))];
    kernel_shared[(((int)threadIdx.x) + 3072)] = kernel[(((((((int)threadIdx.x) + 3072) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 12) % 18))];
    kernel_shared[(((int)threadIdx.x) + 3328)] = kernel[(((((((int)threadIdx.x) + 3328) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 16) % 18))];
    kernel_shared[(((int)threadIdx.x) + 3584)] = kernel[(((((((int)threadIdx.x) + 3584) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 2) % 18))];
    kernel_shared[(((int)threadIdx.x) + 3840)] = kernel[(((((((int)threadIdx.x) + 3840) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 6) % 18))];
    kernel_shared[(((int)threadIdx.x) + 4096)] = kernel[(((((((int)threadIdx.x) + 4096) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 10) % 18))];
    kernel_shared[(((int)threadIdx.x) + 4352)] = kernel[(((((((int)threadIdx.x) + 4352) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 14) % 18))];
    kernel_shared[(((int)threadIdx.x) + 4608)] = kernel[(((((((int)threadIdx.x) / 18) * 2304) + (rc_outer_outer * 18)) + (((int)threadIdx.x) % 18)) + 589824)];
    kernel_shared[(((int)threadIdx.x) + 4864)] = kernel[(((((((int)threadIdx.x) + 4864) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 4) % 18))];
    kernel_shared[(((int)threadIdx.x) + 5120)] = kernel[(((((((int)threadIdx.x) + 5120) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 8) % 18))];
    kernel_shared[(((int)threadIdx.x) + 5376)] = kernel[(((((((int)threadIdx.x) + 5376) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 12) % 18))];
    kernel_shared[(((int)threadIdx.x) + 5632)] = kernel[(((((((int)threadIdx.x) + 5632) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 16) % 18))];
    kernel_shared[(((int)threadIdx.x) + 5888)] = kernel[(((((((int)threadIdx.x) + 5888) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 2) % 18))];
    kernel_shared[(((int)threadIdx.x) + 6144)] = kernel[(((((((int)threadIdx.x) + 6144) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 6) % 18))];
    kernel_shared[(((int)threadIdx.x) + 6400)] = kernel[(((((((int)threadIdx.x) + 6400) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 10) % 18))];
    kernel_shared[(((int)threadIdx.x) + 6656)] = kernel[(((((((int)threadIdx.x) + 6656) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 14) % 18))];
    kernel_shared[(((int)threadIdx.x) + 6912)] = kernel[(((((((int)threadIdx.x) / 18) * 2304) + (rc_outer_outer * 18)) + (((int)threadIdx.x) % 18)) + 884736)];
    kernel_shared[(((int)threadIdx.x) + 7168)] = kernel[(((((((int)threadIdx.x) + 7168) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 4) % 18))];
    kernel_shared[(((int)threadIdx.x) + 7424)] = kernel[(((((((int)threadIdx.x) + 7424) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 8) % 18))];
    kernel_shared[(((int)threadIdx.x) + 7680)] = kernel[(((((((int)threadIdx.x) + 7680) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 12) % 18))];
    kernel_shared[(((int)threadIdx.x) + 7936)] = kernel[(((((((int)threadIdx.x) + 7936) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 16) % 18))];
    kernel_shared[(((int)threadIdx.x) + 8192)] = kernel[(((((((int)threadIdx.x) + 8192) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 2) % 18))];
    kernel_shared[(((int)threadIdx.x) + 8448)] = kernel[(((((((int)threadIdx.x) + 8448) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 6) % 18))];
    kernel_shared[(((int)threadIdx.x) + 8704)] = kernel[(((((((int)threadIdx.x) + 8704) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 10) % 18))];
    kernel_shared[(((int)threadIdx.x) + 8960)] = kernel[(((((((int)threadIdx.x) + 8960) / 18) * 2304) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 14) % 18))];
    __syncthreads();
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[0] * kernel_shared[(((int)threadIdx.x) * 36)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx.x) * 36) + 18)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[1] * kernel_shared[((((int)threadIdx.x) * 36) + 1)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[1] * kernel_shared[((((int)threadIdx.x) * 36) + 19)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx.x) * 36) + 2)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx.x) * 36) + 20)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[3] * kernel_shared[((((int)threadIdx.x) * 36) + 3)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[3] * kernel_shared[((((int)threadIdx.x) * 36) + 21)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx.x) * 36) + 4)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx.x) * 36) + 22)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[5] * kernel_shared[((((int)threadIdx.x) * 36) + 5)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[5] * kernel_shared[((((int)threadIdx.x) * 36) + 23)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx.x) * 36) + 6)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx.x) * 36) + 24)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[7] * kernel_shared[((((int)threadIdx.x) * 36) + 7)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[7] * kernel_shared[((((int)threadIdx.x) * 36) + 25)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx.x) * 36) + 8)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx.x) * 36) + 26)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.x) * 36) + 9)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.x) * 36) + 27)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[10] * kernel_shared[((((int)threadIdx.x) * 36) + 10)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[10] * kernel_shared[((((int)threadIdx.x) * 36) + 28)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.x) * 36) + 11)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.x) * 36) + 29)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[12] * kernel_shared[((((int)threadIdx.x) * 36) + 12)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[12] * kernel_shared[((((int)threadIdx.x) * 36) + 30)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.x) * 36) + 13)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.x) * 36) + 31)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[14] * kernel_shared[((((int)threadIdx.x) * 36) + 14)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[14] * kernel_shared[((((int)threadIdx.x) * 36) + 32)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.x) * 36) + 15)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.x) * 36) + 33)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[16] * kernel_shared[((((int)threadIdx.x) * 36) + 16)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[16] * kernel_shared[((((int)threadIdx.x) * 36) + 34)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.x) * 36) + 17)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.x) * 36) + 35)]));
  }
  for (int ff_inner = 0; ff_inner < 2; ++ff_inner) {
    conv2d_nchw[(((((int)threadIdx.x) * 98) + (ff_inner * 49)) + ((int)blockIdx.x))] = conv2d_nchw_local[ff_inner];
  }
}


