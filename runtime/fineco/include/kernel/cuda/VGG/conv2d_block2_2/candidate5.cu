
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
extern "C" __global__ void __launch_bounds__(512) candidate5(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[56];
  __shared__ float pad_temp_shared[1024];
  __shared__ float kernel_shared[1536];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[14] = 0.000000e+00f;
  conv2d_nchw_local[28] = 0.000000e+00f;
  conv2d_nchw_local[42] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[15] = 0.000000e+00f;
  conv2d_nchw_local[29] = 0.000000e+00f;
  conv2d_nchw_local[43] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[16] = 0.000000e+00f;
  conv2d_nchw_local[30] = 0.000000e+00f;
  conv2d_nchw_local[44] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[17] = 0.000000e+00f;
  conv2d_nchw_local[31] = 0.000000e+00f;
  conv2d_nchw_local[45] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[18] = 0.000000e+00f;
  conv2d_nchw_local[32] = 0.000000e+00f;
  conv2d_nchw_local[46] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[19] = 0.000000e+00f;
  conv2d_nchw_local[33] = 0.000000e+00f;
  conv2d_nchw_local[47] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  conv2d_nchw_local[20] = 0.000000e+00f;
  conv2d_nchw_local[34] = 0.000000e+00f;
  conv2d_nchw_local[48] = 0.000000e+00f;
  conv2d_nchw_local[7] = 0.000000e+00f;
  conv2d_nchw_local[21] = 0.000000e+00f;
  conv2d_nchw_local[35] = 0.000000e+00f;
  conv2d_nchw_local[49] = 0.000000e+00f;
  conv2d_nchw_local[8] = 0.000000e+00f;
  conv2d_nchw_local[22] = 0.000000e+00f;
  conv2d_nchw_local[36] = 0.000000e+00f;
  conv2d_nchw_local[50] = 0.000000e+00f;
  conv2d_nchw_local[9] = 0.000000e+00f;
  conv2d_nchw_local[23] = 0.000000e+00f;
  conv2d_nchw_local[37] = 0.000000e+00f;
  conv2d_nchw_local[51] = 0.000000e+00f;
  conv2d_nchw_local[10] = 0.000000e+00f;
  conv2d_nchw_local[24] = 0.000000e+00f;
  conv2d_nchw_local[38] = 0.000000e+00f;
  conv2d_nchw_local[52] = 0.000000e+00f;
  conv2d_nchw_local[11] = 0.000000e+00f;
  conv2d_nchw_local[25] = 0.000000e+00f;
  conv2d_nchw_local[39] = 0.000000e+00f;
  conv2d_nchw_local[53] = 0.000000e+00f;
  conv2d_nchw_local[12] = 0.000000e+00f;
  conv2d_nchw_local[26] = 0.000000e+00f;
  conv2d_nchw_local[40] = 0.000000e+00f;
  conv2d_nchw_local[54] = 0.000000e+00f;
  conv2d_nchw_local[13] = 0.000000e+00f;
  conv2d_nchw_local[27] = 0.000000e+00f;
  conv2d_nchw_local[41] = 0.000000e+00f;
  conv2d_nchw_local[55] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 32; ++rc_outer_outer) {
    for (int rx_outer_outer = 0; rx_outer_outer < 3; ++rx_outer_outer) {
      __syncthreads();
      pad_temp_shared[((int)threadIdx.x)] = (((((1 <= (((((int)blockIdx.x) / 7) * 14) + ((((int)threadIdx.x) & 255) >> 4))) && ((((((int)blockIdx.x) / 7) * 14) + ((((int)threadIdx.x) & 255) >> 4)) < 113)) && (1 <= ((((((int)blockIdx.x) % 7) * 16) + rx_outer_outer) + (((int)threadIdx.x) & 15)))) && (((((((int)blockIdx.x) % 7) * 16) + rx_outer_outer) + (((int)threadIdx.x) & 15)) < 113)) ? data[((((((((rc_outer_outer * 50176) + ((((int)threadIdx.x) >> 8) * 12544)) + ((((int)blockIdx.x) / 7) * 1568)) + (((((int)threadIdx.x) & 255) >> 4) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + rx_outer_outer) + (((int)threadIdx.x) & 15)) - 113)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 512)] = (((((1 <= (((((int)blockIdx.x) / 7) * 14) + ((((int)threadIdx.x) & 255) >> 4))) && ((((((int)blockIdx.x) / 7) * 14) + ((((int)threadIdx.x) & 255) >> 4)) < 113)) && (1 <= ((((((int)blockIdx.x) % 7) * 16) + rx_outer_outer) + (((int)threadIdx.x) & 15)))) && (((((((int)blockIdx.x) % 7) * 16) + rx_outer_outer) + (((int)threadIdx.x) & 15)) < 113)) ? data[((((((((rc_outer_outer * 50176) + ((((int)threadIdx.x) >> 8) * 12544)) + ((((int)blockIdx.x) / 7) * 1568)) + (((((int)threadIdx.x) & 255) >> 4) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + rx_outer_outer) + (((int)threadIdx.x) & 15)) + 24975)] : 0.000000e+00f);
      kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)threadIdx.x) / 12) * 1152) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) % 12) * 3)) + rx_outer_outer)];
      kernel_shared[(((int)threadIdx.x) + 512)] = kernel[((((((((int)threadIdx.x) + 512) / 12) * 1152) + (rc_outer_outer * 36)) + (((((int)threadIdx.x) + 8) % 12) * 3)) + rx_outer_outer)];
      kernel_shared[(((int)threadIdx.x) + 1024)] = kernel[((((((((int)threadIdx.x) + 1024) / 12) * 1152) + (rc_outer_outer * 36)) + (((((int)threadIdx.x) + 4) % 12) * 3)) + rx_outer_outer)];
      __syncthreads();
      for (int rc_outer_inner = 0; rc_outer_inner < 4; ++rc_outer_inner) {
        for (int ry_inner = 0; ry_inner < 3; ++ry_inner) {
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7))] * kernel_shared[((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 8)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner)]));
          conv2d_nchw_local[28] = (conv2d_nchw_local[28] + (pad_temp_shared[((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7))] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 768)]));
          conv2d_nchw_local[42] = (conv2d_nchw_local[42] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 8)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 768)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 16)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 24)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner)]));
          conv2d_nchw_local[29] = (conv2d_nchw_local[29] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 16)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 768)]));
          conv2d_nchw_local[43] = (conv2d_nchw_local[43] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 24)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 768)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 32)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 40)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner)]));
          conv2d_nchw_local[30] = (conv2d_nchw_local[30] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 32)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 768)]));
          conv2d_nchw_local[44] = (conv2d_nchw_local[44] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 40)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 768)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 48)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 56)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner)]));
          conv2d_nchw_local[31] = (conv2d_nchw_local[31] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 48)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 768)]));
          conv2d_nchw_local[45] = (conv2d_nchw_local[45] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 56)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 768)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 64)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 72)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner)]));
          conv2d_nchw_local[32] = (conv2d_nchw_local[32] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 64)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 768)]));
          conv2d_nchw_local[46] = (conv2d_nchw_local[46] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 72)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 768)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 80)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 88)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner)]));
          conv2d_nchw_local[33] = (conv2d_nchw_local[33] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 80)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 768)]));
          conv2d_nchw_local[47] = (conv2d_nchw_local[47] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 88)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 768)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 96)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner)]));
          conv2d_nchw_local[20] = (conv2d_nchw_local[20] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 104)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner)]));
          conv2d_nchw_local[34] = (conv2d_nchw_local[34] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 96)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 768)]));
          conv2d_nchw_local[48] = (conv2d_nchw_local[48] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 104)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 768)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7))] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 12)]));
          conv2d_nchw_local[21] = (conv2d_nchw_local[21] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 8)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 12)]));
          conv2d_nchw_local[35] = (conv2d_nchw_local[35] + (pad_temp_shared[((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7))] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 780)]));
          conv2d_nchw_local[49] = (conv2d_nchw_local[49] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 8)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 780)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 16)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 12)]));
          conv2d_nchw_local[22] = (conv2d_nchw_local[22] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 24)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 12)]));
          conv2d_nchw_local[36] = (conv2d_nchw_local[36] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 16)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 780)]));
          conv2d_nchw_local[50] = (conv2d_nchw_local[50] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 24)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 780)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 32)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 12)]));
          conv2d_nchw_local[23] = (conv2d_nchw_local[23] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 40)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 12)]));
          conv2d_nchw_local[37] = (conv2d_nchw_local[37] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 32)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 780)]));
          conv2d_nchw_local[51] = (conv2d_nchw_local[51] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 40)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 780)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 48)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 12)]));
          conv2d_nchw_local[24] = (conv2d_nchw_local[24] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 56)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 12)]));
          conv2d_nchw_local[38] = (conv2d_nchw_local[38] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 48)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 780)]));
          conv2d_nchw_local[52] = (conv2d_nchw_local[52] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 56)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 780)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 64)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 12)]));
          conv2d_nchw_local[25] = (conv2d_nchw_local[25] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 72)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 12)]));
          conv2d_nchw_local[39] = (conv2d_nchw_local[39] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 64)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 780)]));
          conv2d_nchw_local[53] = (conv2d_nchw_local[53] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 72)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 780)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 80)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 12)]));
          conv2d_nchw_local[26] = (conv2d_nchw_local[26] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 88)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 12)]));
          conv2d_nchw_local[40] = (conv2d_nchw_local[40] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 80)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 780)]));
          conv2d_nchw_local[54] = (conv2d_nchw_local[54] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 88)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 780)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 96)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 12)]));
          conv2d_nchw_local[27] = (conv2d_nchw_local[27] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 104)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 12)]));
          conv2d_nchw_local[41] = (conv2d_nchw_local[41] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 96)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 780)]));
          conv2d_nchw_local[55] = (conv2d_nchw_local[55] + (pad_temp_shared[(((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (ry_inner * 16)) + (((int)threadIdx.x) & 7)) + 104)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 24) + (rc_outer_inner * 3)) + ry_inner) + 780)]));
        }
      }
    }
  }
  for (int ff_inner = 0; ff_inner < 2; ++ff_inner) {
    for (int yy_inner = 0; yy_inner < 7; ++yy_inner) {
      conv2d_nchw[((((((((((int)threadIdx.x) >> 4) * 25088) + (ff_inner * 12544)) + ((((int)blockIdx.x) / 7) * 1568)) + (((((int)threadIdx.x) & 15) >> 3) * 784)) + (yy_inner * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) & 7))] = conv2d_nchw_local[((ff_inner * 7) + yy_inner)];
      conv2d_nchw[(((((((((((int)threadIdx.x) >> 4) * 25088) + (ff_inner * 12544)) + ((((int)blockIdx.x) / 7) * 1568)) + (((((int)threadIdx.x) & 15) >> 3) * 784)) + (yy_inner * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) & 7)) + 8)] = conv2d_nchw_local[(((ff_inner * 7) + yy_inner) + 14)];
      conv2d_nchw[(((((((((((int)threadIdx.x) >> 4) * 25088) + (ff_inner * 12544)) + ((((int)blockIdx.x) / 7) * 1568)) + (((((int)threadIdx.x) & 15) >> 3) * 784)) + (yy_inner * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) & 7)) + 802816)] = conv2d_nchw_local[(((ff_inner * 7) + yy_inner) + 28)];
      conv2d_nchw[(((((((((((int)threadIdx.x) >> 4) * 25088) + (ff_inner * 12544)) + ((((int)blockIdx.x) / 7) * 1568)) + (((((int)threadIdx.x) & 15) >> 3) * 784)) + (yy_inner * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) & 7)) + 802824)] = conv2d_nchw_local[(((ff_inner * 7) + yy_inner) + 42)];
    }
  }
}


