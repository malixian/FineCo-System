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
extern "C" __global__ void __launch_bounds__(256) candidate5(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[56];
  __shared__ float pad_temp_shared[2784];
  __shared__ float kernel_shared[4608];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  conv2d_nchw_local[7] = 0.000000e+00f;
  conv2d_nchw_local[8] = 0.000000e+00f;
  conv2d_nchw_local[9] = 0.000000e+00f;
  conv2d_nchw_local[10] = 0.000000e+00f;
  conv2d_nchw_local[11] = 0.000000e+00f;
  conv2d_nchw_local[12] = 0.000000e+00f;
  conv2d_nchw_local[13] = 0.000000e+00f;
  conv2d_nchw_local[14] = 0.000000e+00f;
  conv2d_nchw_local[15] = 0.000000e+00f;
  conv2d_nchw_local[16] = 0.000000e+00f;
  conv2d_nchw_local[17] = 0.000000e+00f;
  conv2d_nchw_local[18] = 0.000000e+00f;
  conv2d_nchw_local[19] = 0.000000e+00f;
  conv2d_nchw_local[20] = 0.000000e+00f;
  conv2d_nchw_local[21] = 0.000000e+00f;
  conv2d_nchw_local[22] = 0.000000e+00f;
  conv2d_nchw_local[23] = 0.000000e+00f;
  conv2d_nchw_local[24] = 0.000000e+00f;
  conv2d_nchw_local[25] = 0.000000e+00f;
  conv2d_nchw_local[26] = 0.000000e+00f;
  conv2d_nchw_local[27] = 0.000000e+00f;
  conv2d_nchw_local[28] = 0.000000e+00f;
  conv2d_nchw_local[29] = 0.000000e+00f;
  conv2d_nchw_local[30] = 0.000000e+00f;
  conv2d_nchw_local[31] = 0.000000e+00f;
  conv2d_nchw_local[32] = 0.000000e+00f;
  conv2d_nchw_local[33] = 0.000000e+00f;
  conv2d_nchw_local[34] = 0.000000e+00f;
  conv2d_nchw_local[35] = 0.000000e+00f;
  conv2d_nchw_local[36] = 0.000000e+00f;
  conv2d_nchw_local[37] = 0.000000e+00f;
  conv2d_nchw_local[38] = 0.000000e+00f;
  conv2d_nchw_local[39] = 0.000000e+00f;
  conv2d_nchw_local[40] = 0.000000e+00f;
  conv2d_nchw_local[41] = 0.000000e+00f;
  conv2d_nchw_local[42] = 0.000000e+00f;
  conv2d_nchw_local[43] = 0.000000e+00f;
  conv2d_nchw_local[44] = 0.000000e+00f;
  conv2d_nchw_local[45] = 0.000000e+00f;
  conv2d_nchw_local[46] = 0.000000e+00f;
  conv2d_nchw_local[47] = 0.000000e+00f;
  conv2d_nchw_local[48] = 0.000000e+00f;
  conv2d_nchw_local[49] = 0.000000e+00f;
  conv2d_nchw_local[50] = 0.000000e+00f;
  conv2d_nchw_local[51] = 0.000000e+00f;
  conv2d_nchw_local[52] = 0.000000e+00f;
  conv2d_nchw_local[53] = 0.000000e+00f;
  conv2d_nchw_local[54] = 0.000000e+00f;
  conv2d_nchw_local[55] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 16; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = ((((6 <= ((int)threadIdx.x)) && (1 <= (((((int)blockIdx.x) % 14) * 4) + (((int)threadIdx.x) % 6)))) && ((((((int)blockIdx.x) % 14) * 4) + (((int)threadIdx.x) % 6)) < 57)) ? data[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 6) * 56)) + ((((int)blockIdx.x) % 14) * 4)) + (((int)threadIdx.x) % 6)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 256)] = (((((3 <= (((((int)threadIdx.x) >> 1) + 128) % 174)) && (((((int)threadIdx.x) + 256) % 348) < 342)) && (1 <= (((((int)blockIdx.x) % 14) * 4) + ((((int)threadIdx.x) + 4) % 6)))) && ((((((int)blockIdx.x) % 14) * 4) + ((((int)threadIdx.x) + 4) % 6)) < 57)) ? data[((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 256) / 348) * 3136)) + (((((((int)threadIdx.x) >> 1) + 128) % 174) / 3) * 56)) + ((((int)blockIdx.x) % 14) * 4)) + ((((int)threadIdx.x) + 4) % 6)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 512)] = (((((3 <= (((((int)threadIdx.x) >> 1) + 82) % 174)) && (((((int)threadIdx.x) + 164) % 348) < 342)) && (1 <= (((((int)blockIdx.x) % 14) * 4) + ((((int)threadIdx.x) + 2) % 6)))) && ((((((int)blockIdx.x) % 14) * 4) + ((((int)threadIdx.x) + 2) % 6)) < 57)) ? data[((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 512) / 348) * 3136)) + (((((((int)threadIdx.x) >> 1) + 82) % 174) / 3) * 56)) + ((((int)blockIdx.x) % 14) * 4)) + ((((int)threadIdx.x) + 2) % 6)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 768)] = (((1 <= (((((int)blockIdx.x) % 14) * 4) + (((int)threadIdx.x) % 6))) && ((((((int)blockIdx.x) % 14) * 4) + (((int)threadIdx.x) % 6)) < 57)) ? data[((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 768) / 348) * 3136)) + (((((int)threadIdx.x) / 6) + 12) * 56)) + ((((int)blockIdx.x) % 14) * 4)) + (((int)threadIdx.x) % 6)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1024)] = (((((3 <= (((((int)threadIdx.x) >> 1) + 164) % 174)) && (((((int)threadIdx.x) + 328) % 348) < 342)) && (1 <= (((((int)blockIdx.x) % 14) * 4) + ((((int)threadIdx.x) + 4) % 6)))) && ((((((int)blockIdx.x) % 14) * 4) + ((((int)threadIdx.x) + 4) % 6)) < 57)) ? data[((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 1024) / 348) * 3136)) + (((((((int)threadIdx.x) >> 1) + 164) % 174) / 3) * 56)) + ((((int)blockIdx.x) % 14) * 4)) + ((((int)threadIdx.x) + 4) % 6)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1280)] = (((((3 <= (((((int)threadIdx.x) >> 1) + 118) % 174)) && (((((int)threadIdx.x) + 236) % 348) < 342)) && (1 <= (((((int)blockIdx.x) % 14) * 4) + ((((int)threadIdx.x) + 2) % 6)))) && ((((((int)blockIdx.x) % 14) * 4) + ((((int)threadIdx.x) + 2) % 6)) < 57)) ? data[((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 1280) / 348) * 3136)) + (((((((int)threadIdx.x) >> 1) + 118) % 174) / 3) * 56)) + ((((int)blockIdx.x) % 14) * 4)) + ((((int)threadIdx.x) + 2) % 6)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1536)] = (((((1 <= (((((int)threadIdx.x) / 6) + 24) % 58)) && (((((int)threadIdx.x) + 144) % 348) < 342)) && (1 <= (((((int)blockIdx.x) % 14) * 4) + (((int)threadIdx.x) % 6)))) && ((((((int)blockIdx.x) % 14) * 4) + (((int)threadIdx.x) % 6)) < 57)) ? data[((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 1536) / 348) * 3136)) + ((((((int)threadIdx.x) / 6) + 24) % 58) * 56)) + ((((int)blockIdx.x) % 14) * 4)) + (((int)threadIdx.x) % 6)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1792)] = (((1 <= (((((int)blockIdx.x) % 14) * 4) + ((((int)threadIdx.x) + 4) % 6))) && ((((((int)blockIdx.x) % 14) * 4) + ((((int)threadIdx.x) + 4) % 6)) < 57)) ? data[((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 1792) / 348) * 3136)) + (((((((int)threadIdx.x) >> 1) + 26) % 174) / 3) * 56)) + ((((int)blockIdx.x) % 14) * 4)) + ((((int)threadIdx.x) + 4) % 6)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 2048)] = (((((3 <= (((((int)threadIdx.x) >> 1) + 154) % 174)) && (((((int)threadIdx.x) + 308) % 348) < 342)) && (1 <= (((((int)blockIdx.x) % 14) * 4) + ((((int)threadIdx.x) + 2) % 6)))) && ((((((int)blockIdx.x) % 14) * 4) + ((((int)threadIdx.x) + 2) % 6)) < 57)) ? data[((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 2048) / 348) * 3136)) + (((((((int)threadIdx.x) >> 1) + 154) % 174) / 3) * 56)) + ((((int)blockIdx.x) % 14) * 4)) + ((((int)threadIdx.x) + 2) % 6)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 2304)] = (((((1 <= (((((int)threadIdx.x) / 6) + 36) % 58)) && (((((int)threadIdx.x) + 216) % 348) < 342)) && (1 <= (((((int)blockIdx.x) % 14) * 4) + (((int)threadIdx.x) % 6)))) && ((((((int)blockIdx.x) % 14) * 4) + (((int)threadIdx.x) % 6)) < 57)) ? data[((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 2304) / 348) * 3136)) + ((((((int)threadIdx.x) / 6) + 36) % 58) * 56)) + ((((int)blockIdx.x) % 14) * 4)) + (((int)threadIdx.x) % 6)) - 57)] : 0.000000e+00f);
    if (((int)threadIdx.x) < 224) {
      pad_temp_shared[(((int)threadIdx.x) + 2560)] = ((((((int)threadIdx.x) < 218) && (1 <= (((((int)blockIdx.x) % 14) * 4) + ((((int)threadIdx.x) + 4) % 6)))) && ((((((int)blockIdx.x) % 14) * 4) + ((((int)threadIdx.x) + 4) % 6)) < 57)) ? data[((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 2560) / 348) * 3136)) + (((((((int)threadIdx.x) >> 1) + 62) % 174) / 3) * 56)) + ((((int)blockIdx.x) % 14) * 4)) + ((((int)threadIdx.x) + 4) % 6)) - 57)] : 0.000000e+00f);
    }
    kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) / 14) * 73728) + ((((int)threadIdx.x) / 72) * 1152)) + (rc_outer_outer * 72)) + (((int)threadIdx.x) % 72))];
    kernel_shared[(((int)threadIdx.x) + 256)] = kernel[(((((((int)blockIdx.x) / 14) * 73728) + (((((int)threadIdx.x) + 256) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 40) % 72))];
    kernel_shared[(((int)threadIdx.x) + 512)] = kernel[(((((((int)blockIdx.x) / 14) * 73728) + (((((int)threadIdx.x) + 512) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 8) % 72))];
    kernel_shared[(((int)threadIdx.x) + 768)] = kernel[(((((((int)blockIdx.x) / 14) * 73728) + (((((int)threadIdx.x) + 768) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 48) % 72))];
    kernel_shared[(((int)threadIdx.x) + 1024)] = kernel[(((((((int)blockIdx.x) / 14) * 73728) + (((((int)threadIdx.x) + 1024) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 16) % 72))];
    kernel_shared[(((int)threadIdx.x) + 1280)] = kernel[(((((((int)blockIdx.x) / 14) * 73728) + (((((int)threadIdx.x) + 1280) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 56) % 72))];
    kernel_shared[(((int)threadIdx.x) + 1536)] = kernel[(((((((int)blockIdx.x) / 14) * 73728) + (((((int)threadIdx.x) + 1536) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 24) % 72))];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[(((((((int)blockIdx.x) / 14) * 73728) + (((((int)threadIdx.x) + 1792) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 64) % 72))];
    kernel_shared[(((int)threadIdx.x) + 2048)] = kernel[(((((((int)blockIdx.x) / 14) * 73728) + (((((int)threadIdx.x) + 2048) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 32) % 72))];
    kernel_shared[(((int)threadIdx.x) + 2304)] = kernel[((((((((int)blockIdx.x) / 14) * 73728) + ((((int)threadIdx.x) / 72) * 1152)) + (rc_outer_outer * 72)) + (((int)threadIdx.x) % 72)) + 36864)];
    kernel_shared[(((int)threadIdx.x) + 2560)] = kernel[(((((((int)blockIdx.x) / 14) * 73728) + (((((int)threadIdx.x) + 2560) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 40) % 72))];
    kernel_shared[(((int)threadIdx.x) + 2816)] = kernel[(((((((int)blockIdx.x) / 14) * 73728) + (((((int)threadIdx.x) + 2816) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 8) % 72))];
    kernel_shared[(((int)threadIdx.x) + 3072)] = kernel[(((((((int)blockIdx.x) / 14) * 73728) + (((((int)threadIdx.x) + 3072) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 48) % 72))];
    kernel_shared[(((int)threadIdx.x) + 3328)] = kernel[(((((((int)blockIdx.x) / 14) * 73728) + (((((int)threadIdx.x) + 3328) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 16) % 72))];
    kernel_shared[(((int)threadIdx.x) + 3584)] = kernel[(((((((int)blockIdx.x) / 14) * 73728) + (((((int)threadIdx.x) + 3584) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 56) % 72))];
    kernel_shared[(((int)threadIdx.x) + 3840)] = kernel[(((((((int)blockIdx.x) / 14) * 73728) + (((((int)threadIdx.x) + 3840) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 24) % 72))];
    kernel_shared[(((int)threadIdx.x) + 4096)] = kernel[(((((((int)blockIdx.x) / 14) * 73728) + (((((int)threadIdx.x) + 4096) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 64) % 72))];
    kernel_shared[(((int)threadIdx.x) + 4352)] = kernel[(((((((int)blockIdx.x) / 14) * 73728) + (((((int)threadIdx.x) + 4352) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 32) % 72))];
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 8; ++rc_outer_inner) {
      for (int ff_c_outer_inner = 0; ff_c_outer_inner < 4; ++ff_c_outer_inner) {
        for (int ry_inner = 0; ry_inner < 3; ++ry_inner) {
          conv2d_nchw_local[(ff_c_outer_inner * 14)] = (conv2d_nchw_local[(ff_c_outer_inner * 14)] + (pad_temp_shared[((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2))] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3))]));
          conv2d_nchw_local[((ff_c_outer_inner * 14) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 14) + 1)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 1)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3))]));
          conv2d_nchw_local[((ff_c_outer_inner * 14) + 2)] = (conv2d_nchw_local[((ff_c_outer_inner * 14) + 2)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 6)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3))]));
          conv2d_nchw_local[((ff_c_outer_inner * 14) + 3)] = (conv2d_nchw_local[((ff_c_outer_inner * 14) + 3)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 7)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3))]));
          conv2d_nchw_local[((ff_c_outer_inner * 14) + 4)] = (conv2d_nchw_local[((ff_c_outer_inner * 14) + 4)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 12)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3))]));
          conv2d_nchw_local[((ff_c_outer_inner * 14) + 5)] = (conv2d_nchw_local[((ff_c_outer_inner * 14) + 5)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 13)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3))]));
          conv2d_nchw_local[((ff_c_outer_inner * 14) + 6)] = (conv2d_nchw_local[((ff_c_outer_inner * 14) + 6)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 18)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3))]));
          conv2d_nchw_local[((ff_c_outer_inner * 14) + 7)] = (conv2d_nchw_local[((ff_c_outer_inner * 14) + 7)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 19)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3))]));
          conv2d_nchw_local[((ff_c_outer_inner * 14) + 8)] = (conv2d_nchw_local[((ff_c_outer_inner * 14) + 8)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 24)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3))]));
          conv2d_nchw_local[((ff_c_outer_inner * 14) + 9)] = (conv2d_nchw_local[((ff_c_outer_inner * 14) + 9)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 25)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3))]));
          conv2d_nchw_local[((ff_c_outer_inner * 14) + 10)] = (conv2d_nchw_local[((ff_c_outer_inner * 14) + 10)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 30)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3))]));
          conv2d_nchw_local[((ff_c_outer_inner * 14) + 11)] = (conv2d_nchw_local[((ff_c_outer_inner * 14) + 11)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 31)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3))]));
          conv2d_nchw_local[((ff_c_outer_inner * 14) + 12)] = (conv2d_nchw_local[((ff_c_outer_inner * 14) + 12)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 36)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3))]));
          conv2d_nchw_local[((ff_c_outer_inner * 14) + 13)] = (conv2d_nchw_local[((ff_c_outer_inner * 14) + 13)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 37)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3))]));
          conv2d_nchw_local[(ff_c_outer_inner * 14)] = (conv2d_nchw_local[(ff_c_outer_inner * 14)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 1)] * kernel_shared[((((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + 1)]));
          conv2d_nchw_local[((ff_c_outer_inner * 14) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 14) + 1)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 2)] * kernel_shared[((((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + 1)]));
          conv2d_nchw_local[((ff_c_outer_inner * 14) + 2)] = (conv2d_nchw_local[((ff_c_outer_inner * 14) + 2)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 7)] * kernel_shared[((((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + 1)]));
          conv2d_nchw_local[((ff_c_outer_inner * 14) + 3)] = (conv2d_nchw_local[((ff_c_outer_inner * 14) + 3)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 8)] * kernel_shared[((((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + 1)]));
          conv2d_nchw_local[((ff_c_outer_inner * 14) + 4)] = (conv2d_nchw_local[((ff_c_outer_inner * 14) + 4)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 13)] * kernel_shared[((((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + 1)]));
          conv2d_nchw_local[((ff_c_outer_inner * 14) + 5)] = (conv2d_nchw_local[((ff_c_outer_inner * 14) + 5)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 14)] * kernel_shared[((((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + 1)]));
          conv2d_nchw_local[((ff_c_outer_inner * 14) + 6)] = (conv2d_nchw_local[((ff_c_outer_inner * 14) + 6)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 19)] * kernel_shared[((((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + 1)]));
          conv2d_nchw_local[((ff_c_outer_inner * 14) + 7)] = (conv2d_nchw_local[((ff_c_outer_inner * 14) + 7)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 20)] * kernel_shared[((((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + 1)]));
          conv2d_nchw_local[((ff_c_outer_inner * 14) + 8)] = (conv2d_nchw_local[((ff_c_outer_inner * 14) + 8)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 25)] * kernel_shared[((((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + 1)]));
          conv2d_nchw_local[((ff_c_outer_inner * 14) + 9)] = (conv2d_nchw_local[((ff_c_outer_inner * 14) + 9)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 26)] * kernel_shared[((((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + 1)]));
          conv2d_nchw_local[((ff_c_outer_inner * 14) + 10)] = (conv2d_nchw_local[((ff_c_outer_inner * 14) + 10)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 31)] * kernel_shared[((((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + 1)]));
          conv2d_nchw_local[((ff_c_outer_inner * 14) + 11)] = (conv2d_nchw_local[((ff_c_outer_inner * 14) + 11)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 32)] * kernel_shared[((((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + 1)]));
          conv2d_nchw_local[((ff_c_outer_inner * 14) + 12)] = (conv2d_nchw_local[((ff_c_outer_inner * 14) + 12)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 37)] * kernel_shared[((((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + 1)]));
          conv2d_nchw_local[((ff_c_outer_inner * 14) + 13)] = (conv2d_nchw_local[((ff_c_outer_inner * 14) + 13)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 38)] * kernel_shared[((((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + 1)]));
          conv2d_nchw_local[(ff_c_outer_inner * 14)] = (conv2d_nchw_local[(ff_c_outer_inner * 14)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 2)] * kernel_shared[((((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + 2)]));
          conv2d_nchw_local[((ff_c_outer_inner * 14) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 14) + 1)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 3)] * kernel_shared[((((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + 2)]));
          conv2d_nchw_local[((ff_c_outer_inner * 14) + 2)] = (conv2d_nchw_local[((ff_c_outer_inner * 14) + 2)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 8)] * kernel_shared[((((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + 2)]));
          conv2d_nchw_local[((ff_c_outer_inner * 14) + 3)] = (conv2d_nchw_local[((ff_c_outer_inner * 14) + 3)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 9)] * kernel_shared[((((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + 2)]));
          conv2d_nchw_local[((ff_c_outer_inner * 14) + 4)] = (conv2d_nchw_local[((ff_c_outer_inner * 14) + 4)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 14)] * kernel_shared[((((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + 2)]));
          conv2d_nchw_local[((ff_c_outer_inner * 14) + 5)] = (conv2d_nchw_local[((ff_c_outer_inner * 14) + 5)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 15)] * kernel_shared[((((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + 2)]));
          conv2d_nchw_local[((ff_c_outer_inner * 14) + 6)] = (conv2d_nchw_local[((ff_c_outer_inner * 14) + 6)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 20)] * kernel_shared[((((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + 2)]));
          conv2d_nchw_local[((ff_c_outer_inner * 14) + 7)] = (conv2d_nchw_local[((ff_c_outer_inner * 14) + 7)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 21)] * kernel_shared[((((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + 2)]));
          conv2d_nchw_local[((ff_c_outer_inner * 14) + 8)] = (conv2d_nchw_local[((ff_c_outer_inner * 14) + 8)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 26)] * kernel_shared[((((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + 2)]));
          conv2d_nchw_local[((ff_c_outer_inner * 14) + 9)] = (conv2d_nchw_local[((ff_c_outer_inner * 14) + 9)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 27)] * kernel_shared[((((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + 2)]));
          conv2d_nchw_local[((ff_c_outer_inner * 14) + 10)] = (conv2d_nchw_local[((ff_c_outer_inner * 14) + 10)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 32)] * kernel_shared[((((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + 2)]));
          conv2d_nchw_local[((ff_c_outer_inner * 14) + 11)] = (conv2d_nchw_local[((ff_c_outer_inner * 14) + 11)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 33)] * kernel_shared[((((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + 2)]));
          conv2d_nchw_local[((ff_c_outer_inner * 14) + 12)] = (conv2d_nchw_local[((ff_c_outer_inner * 14) + 12)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 38)] * kernel_shared[((((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + 2)]));
          conv2d_nchw_local[((ff_c_outer_inner * 14) + 13)] = (conv2d_nchw_local[((ff_c_outer_inner * 14) + 13)] + (pad_temp_shared[(((((rc_outer_inner * 348) + (((((int)threadIdx.x) & 15) >> 1) * 42)) + (ry_inner * 6)) + ((((int)threadIdx.x) & 1) * 2)) + 39)] * kernel_shared[((((((((int)threadIdx.x) >> 4) * 288) + (ff_c_outer_inner * 72)) + (rc_outer_inner * 9)) + (ry_inner * 3)) + 2)]));
        }
      }
    }
  }
  for (int ff_inner = 0; ff_inner < 4; ++ff_inner) {
    for (int yy_inner = 0; yy_inner < 7; ++yy_inner) {
      for (int xx_inner = 0; xx_inner < 2; ++xx_inner) {
        conv2d_nchw[(((((((((((int)blockIdx.x) / 14) * 200704) + ((((int)threadIdx.x) >> 4) * 12544)) + (ff_inner * 3136)) + (((((int)threadIdx.x) & 15) >> 1) * 392)) + (yy_inner * 56)) + ((((int)blockIdx.x) % 14) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + xx_inner)] = conv2d_nchw_local[(((ff_inner * 14) + (yy_inner * 2)) + xx_inner)];
      }
    }
  }
}


