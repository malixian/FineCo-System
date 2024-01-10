
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
extern "C" __global__ void __launch_bounds__(96) candidate0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ compute, float* __restrict__ bias) {
  float conv2d_nchw[36];
  __shared__ float pad_temp_shared[312];
  __shared__ float kernel_shared[5120];
  conv2d_nchw[0] = 0.000000e+00f;
  conv2d_nchw[1] = 0.000000e+00f;
  conv2d_nchw[2] = 0.000000e+00f;
  conv2d_nchw[9] = 0.000000e+00f;
  conv2d_nchw[10] = 0.000000e+00f;
  conv2d_nchw[11] = 0.000000e+00f;
  conv2d_nchw[3] = 0.000000e+00f;
  conv2d_nchw[4] = 0.000000e+00f;
  conv2d_nchw[5] = 0.000000e+00f;
  conv2d_nchw[12] = 0.000000e+00f;
  conv2d_nchw[13] = 0.000000e+00f;
  conv2d_nchw[14] = 0.000000e+00f;
  conv2d_nchw[6] = 0.000000e+00f;
  conv2d_nchw[7] = 0.000000e+00f;
  conv2d_nchw[8] = 0.000000e+00f;
  conv2d_nchw[15] = 0.000000e+00f;
  conv2d_nchw[16] = 0.000000e+00f;
  conv2d_nchw[17] = 0.000000e+00f;
  conv2d_nchw[18] = 0.000000e+00f;
  conv2d_nchw[19] = 0.000000e+00f;
  conv2d_nchw[20] = 0.000000e+00f;
  conv2d_nchw[27] = 0.000000e+00f;
  conv2d_nchw[28] = 0.000000e+00f;
  conv2d_nchw[29] = 0.000000e+00f;
  conv2d_nchw[21] = 0.000000e+00f;
  conv2d_nchw[22] = 0.000000e+00f;
  conv2d_nchw[23] = 0.000000e+00f;
  conv2d_nchw[30] = 0.000000e+00f;
  conv2d_nchw[31] = 0.000000e+00f;
  conv2d_nchw[32] = 0.000000e+00f;
  conv2d_nchw[24] = 0.000000e+00f;
  conv2d_nchw[25] = 0.000000e+00f;
  conv2d_nchw[26] = 0.000000e+00f;
  conv2d_nchw[33] = 0.000000e+00f;
  conv2d_nchw[34] = 0.000000e+00f;
  conv2d_nchw[35] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 6; ++rc_outer_outer) {
    for (int ry_outer_outer = 0; ry_outer_outer < 5; ++ry_outer_outer) {
      __syncthreads();
      pad_temp_shared[((int)threadIdx.x)] = (((((2 <= ((((((int)blockIdx.x) / 3) * 3) + ((((int)threadIdx.x) % 39) / 13)) + ry_outer_outer)) && (((((((int)blockIdx.x) / 3) * 3) + ((((int)threadIdx.x) % 39) / 13)) + ry_outer_outer) < 29)) && (2 <= (((((int)blockIdx.x) % 3) * 9) + (((int)threadIdx.x) % 13)))) && ((((((int)blockIdx.x) % 3) * 9) + (((int)threadIdx.x) % 13)) < 29)) ? data[((((((((rc_outer_outer * 5832) + ((((int)threadIdx.x) / 39) * 729)) + ((((int)blockIdx.x) / 3) * 81)) + (((((int)threadIdx.x) % 39) / 13) * 27)) + (ry_outer_outer * 27)) + ((((int)blockIdx.x) % 3) * 9)) + (((int)threadIdx.x) % 13)) - 56)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 96)] = (((((2 <= ((((((int)blockIdx.x) / 3) * 3) + (((((int)threadIdx.x) + 18) % 39) / 13)) + ry_outer_outer)) && (((((((int)blockIdx.x) / 3) * 3) + (((((int)threadIdx.x) + 18) % 39) / 13)) + ry_outer_outer) < 29)) && (2 <= (((((int)blockIdx.x) % 3) * 9) + ((((int)threadIdx.x) + 5) % 13)))) && ((((((int)blockIdx.x) % 3) * 9) + ((((int)threadIdx.x) + 5) % 13)) < 29)) ? data[((((((((rc_outer_outer * 5832) + (((((int)threadIdx.x) + 96) / 39) * 729)) + ((((int)blockIdx.x) / 3) * 81)) + ((((((int)threadIdx.x) + 18) % 39) / 13) * 27)) + (ry_outer_outer * 27)) + ((((int)blockIdx.x) % 3) * 9)) + ((((int)threadIdx.x) + 5) % 13)) - 56)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 192)] = (((((2 <= ((((((int)blockIdx.x) / 3) * 3) + (((((int)threadIdx.x) + 36) % 39) / 13)) + ry_outer_outer)) && (((((((int)blockIdx.x) / 3) * 3) + (((((int)threadIdx.x) + 36) % 39) / 13)) + ry_outer_outer) < 29)) && (2 <= (((((int)blockIdx.x) % 3) * 9) + ((((int)threadIdx.x) + 10) % 13)))) && ((((((int)blockIdx.x) % 3) * 9) + ((((int)threadIdx.x) + 10) % 13)) < 29)) ? data[((((((((rc_outer_outer * 5832) + (((((int)threadIdx.x) + 192) / 39) * 729)) + ((((int)blockIdx.x) / 3) * 81)) + ((((((int)threadIdx.x) + 36) % 39) / 13) * 27)) + (ry_outer_outer * 27)) + ((((int)blockIdx.x) % 3) * 9)) + ((((int)threadIdx.x) + 10) % 13)) - 56)] : 0.000000e+00f);
      if (((int)threadIdx.x) < 24) {
        pad_temp_shared[(((int)threadIdx.x) + 288)] = (((((2 <= ((((((int)blockIdx.x) / 3) * 3) + (((((int)threadIdx.x) + 15) % 39) / 13)) + ry_outer_outer)) && (((((((int)blockIdx.x) / 3) * 3) + (((((int)threadIdx.x) + 15) % 39) / 13)) + ry_outer_outer) < 29)) && (2 <= (((((int)blockIdx.x) % 3) * 9) + ((((int)threadIdx.x) + 2) % 13)))) && ((((((int)blockIdx.x) % 3) * 9) + ((((int)threadIdx.x) + 2) % 13)) < 29)) ? data[((((((((rc_outer_outer * 5832) + (((((int)threadIdx.x) + 288) / 39) * 729)) + ((((int)blockIdx.x) / 3) * 81)) + ((((((int)threadIdx.x) + 15) % 39) / 13) * 27)) + (ry_outer_outer * 27)) + ((((int)blockIdx.x) % 3) * 9)) + ((((int)threadIdx.x) + 2) % 13)) - 56)] : 0.000000e+00f);
      }
      kernel_shared[((int)threadIdx.x)] = kernel[((((((((int)threadIdx.x) / 40) * 1200) + (rc_outer_outer * 200)) + (((((int)threadIdx.x) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + (((int)threadIdx.x) % 5))];
      kernel_shared[(((int)threadIdx.x) + 96)] = kernel[(((((((((int)threadIdx.x) + 96) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 16) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 1) % 5))];
      kernel_shared[(((int)threadIdx.x) + 192)] = kernel[(((((((((int)threadIdx.x) + 192) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 32) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 2) % 5))];
      kernel_shared[(((int)threadIdx.x) + 288)] = kernel[(((((((((int)threadIdx.x) + 288) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 8) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 3) % 5))];
      kernel_shared[(((int)threadIdx.x) + 384)] = kernel[(((((((((int)threadIdx.x) + 384) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 24) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 4) % 5))];
      kernel_shared[(((int)threadIdx.x) + 480)] = kernel[(((((((((int)threadIdx.x) / 40) * 1200) + (rc_outer_outer * 200)) + (((((int)threadIdx.x) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + (((int)threadIdx.x) % 5)) + 14400)];
      kernel_shared[(((int)threadIdx.x) + 576)] = kernel[(((((((((int)threadIdx.x) + 576) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 16) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 1) % 5))];
      kernel_shared[(((int)threadIdx.x) + 672)] = kernel[(((((((((int)threadIdx.x) + 672) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 32) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 2) % 5))];
      kernel_shared[(((int)threadIdx.x) + 768)] = kernel[(((((((((int)threadIdx.x) + 768) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 8) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 3) % 5))];
      kernel_shared[(((int)threadIdx.x) + 864)] = kernel[(((((((((int)threadIdx.x) + 864) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 24) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 4) % 5))];
      kernel_shared[(((int)threadIdx.x) + 960)] = kernel[(((((((((int)threadIdx.x) / 40) * 1200) + (rc_outer_outer * 200)) + (((((int)threadIdx.x) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + (((int)threadIdx.x) % 5)) + 28800)];
      kernel_shared[(((int)threadIdx.x) + 1056)] = kernel[(((((((((int)threadIdx.x) + 1056) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 16) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 1) % 5))];
      kernel_shared[(((int)threadIdx.x) + 1152)] = kernel[(((((((((int)threadIdx.x) + 1152) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 32) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 2) % 5))];
      kernel_shared[(((int)threadIdx.x) + 1248)] = kernel[(((((((((int)threadIdx.x) + 1248) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 8) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 3) % 5))];
      kernel_shared[(((int)threadIdx.x) + 1344)] = kernel[(((((((((int)threadIdx.x) + 1344) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 24) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 4) % 5))];
      kernel_shared[(((int)threadIdx.x) + 1440)] = kernel[(((((((((int)threadIdx.x) / 40) * 1200) + (rc_outer_outer * 200)) + (((((int)threadIdx.x) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + (((int)threadIdx.x) % 5)) + 43200)];
      kernel_shared[(((int)threadIdx.x) + 1536)] = kernel[(((((((((int)threadIdx.x) + 1536) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 16) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 1) % 5))];
      kernel_shared[(((int)threadIdx.x) + 1632)] = kernel[(((((((((int)threadIdx.x) + 1632) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 32) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 2) % 5))];
      kernel_shared[(((int)threadIdx.x) + 1728)] = kernel[(((((((((int)threadIdx.x) + 1728) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 8) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 3) % 5))];
      kernel_shared[(((int)threadIdx.x) + 1824)] = kernel[(((((((((int)threadIdx.x) + 1824) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 24) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 4) % 5))];
      kernel_shared[(((int)threadIdx.x) + 1920)] = kernel[(((((((((int)threadIdx.x) / 40) * 1200) + (rc_outer_outer * 200)) + (((((int)threadIdx.x) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + (((int)threadIdx.x) % 5)) + 57600)];
      kernel_shared[(((int)threadIdx.x) + 2016)] = kernel[(((((((((int)threadIdx.x) + 2016) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 16) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 1) % 5))];
      kernel_shared[(((int)threadIdx.x) + 2112)] = kernel[(((((((((int)threadIdx.x) + 2112) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 32) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 2) % 5))];
      kernel_shared[(((int)threadIdx.x) + 2208)] = kernel[(((((((((int)threadIdx.x) + 2208) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 8) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 3) % 5))];
      kernel_shared[(((int)threadIdx.x) + 2304)] = kernel[(((((((((int)threadIdx.x) + 2304) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 24) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 4) % 5))];
      kernel_shared[(((int)threadIdx.x) + 2400)] = kernel[(((((((((int)threadIdx.x) / 40) * 1200) + (rc_outer_outer * 200)) + (((((int)threadIdx.x) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + (((int)threadIdx.x) % 5)) + 72000)];
      kernel_shared[(((int)threadIdx.x) + 2496)] = kernel[(((((((((int)threadIdx.x) + 2496) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 16) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 1) % 5))];
      kernel_shared[(((int)threadIdx.x) + 2592)] = kernel[(((((((((int)threadIdx.x) + 2592) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 32) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 2) % 5))];
      kernel_shared[(((int)threadIdx.x) + 2688)] = kernel[(((((((((int)threadIdx.x) + 2688) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 8) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 3) % 5))];
      kernel_shared[(((int)threadIdx.x) + 2784)] = kernel[(((((((((int)threadIdx.x) + 2784) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 24) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 4) % 5))];
      kernel_shared[(((int)threadIdx.x) + 2880)] = kernel[(((((((((int)threadIdx.x) / 40) * 1200) + (rc_outer_outer * 200)) + (((((int)threadIdx.x) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + (((int)threadIdx.x) % 5)) + 86400)];
      kernel_shared[(((int)threadIdx.x) + 2976)] = kernel[(((((((((int)threadIdx.x) + 2976) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 16) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 1) % 5))];
      kernel_shared[(((int)threadIdx.x) + 3072)] = kernel[(((((((((int)threadIdx.x) + 3072) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 32) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 2) % 5))];
      kernel_shared[(((int)threadIdx.x) + 3168)] = kernel[(((((((((int)threadIdx.x) + 3168) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 8) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 3) % 5))];
      kernel_shared[(((int)threadIdx.x) + 3264)] = kernel[(((((((((int)threadIdx.x) + 3264) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 24) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 4) % 5))];
      kernel_shared[(((int)threadIdx.x) + 3360)] = kernel[(((((((((int)threadIdx.x) / 40) * 1200) + (rc_outer_outer * 200)) + (((((int)threadIdx.x) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + (((int)threadIdx.x) % 5)) + 100800)];
      kernel_shared[(((int)threadIdx.x) + 3456)] = kernel[(((((((((int)threadIdx.x) + 3456) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 16) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 1) % 5))];
      kernel_shared[(((int)threadIdx.x) + 3552)] = kernel[(((((((((int)threadIdx.x) + 3552) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 32) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 2) % 5))];
      kernel_shared[(((int)threadIdx.x) + 3648)] = kernel[(((((((((int)threadIdx.x) + 3648) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 8) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 3) % 5))];
      kernel_shared[(((int)threadIdx.x) + 3744)] = kernel[(((((((((int)threadIdx.x) + 3744) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 24) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 4) % 5))];
      kernel_shared[(((int)threadIdx.x) + 3840)] = kernel[(((((((((int)threadIdx.x) / 40) * 1200) + (rc_outer_outer * 200)) + (((((int)threadIdx.x) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + (((int)threadIdx.x) % 5)) + 115200)];
      kernel_shared[(((int)threadIdx.x) + 3936)] = kernel[(((((((((int)threadIdx.x) + 3936) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 16) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 1) % 5))];
      kernel_shared[(((int)threadIdx.x) + 4032)] = kernel[(((((((((int)threadIdx.x) + 4032) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 32) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 2) % 5))];
      kernel_shared[(((int)threadIdx.x) + 4128)] = kernel[(((((((((int)threadIdx.x) + 4128) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 8) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 3) % 5))];
      kernel_shared[(((int)threadIdx.x) + 4224)] = kernel[(((((((((int)threadIdx.x) + 4224) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 24) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 4) % 5))];
      kernel_shared[(((int)threadIdx.x) + 4320)] = kernel[(((((((((int)threadIdx.x) / 40) * 1200) + (rc_outer_outer * 200)) + (((((int)threadIdx.x) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + (((int)threadIdx.x) % 5)) + 129600)];
      kernel_shared[(((int)threadIdx.x) + 4416)] = kernel[(((((((((int)threadIdx.x) + 4416) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 16) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 1) % 5))];
      kernel_shared[(((int)threadIdx.x) + 4512)] = kernel[(((((((((int)threadIdx.x) + 4512) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 32) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 2) % 5))];
      kernel_shared[(((int)threadIdx.x) + 4608)] = kernel[(((((((((int)threadIdx.x) + 4608) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 8) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 3) % 5))];
      kernel_shared[(((int)threadIdx.x) + 4704)] = kernel[(((((((((int)threadIdx.x) + 4704) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 24) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 4) % 5))];
      kernel_shared[(((int)threadIdx.x) + 4800)] = kernel[(((((((((int)threadIdx.x) / 40) * 1200) + (rc_outer_outer * 200)) + (((((int)threadIdx.x) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + (((int)threadIdx.x) % 5)) + 144000)];
      kernel_shared[(((int)threadIdx.x) + 4896)] = kernel[(((((((((int)threadIdx.x) + 4896) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 16) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 1) % 5))];
      kernel_shared[(((int)threadIdx.x) + 4992)] = kernel[(((((((((int)threadIdx.x) + 4992) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 32) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 2) % 5))];
      if (((int)threadIdx.x) < 32) {
        kernel_shared[(((int)threadIdx.x) + 5088)] = kernel[(((((((((int)threadIdx.x) + 5088) / 40) * 1200) + (rc_outer_outer * 200)) + ((((((int)threadIdx.x) + 8) % 40) / 5) * 25)) + (ry_outer_outer * 5)) + ((((int)threadIdx.x) + 3) % 5))];
      }
      __syncthreads();
      for (int rc_outer_inner = 0; rc_outer_inner < 2; ++rc_outer_inner) {
        for (int rx_outer_inner = 0; rx_outer_inner < 5; ++rx_outer_inner) {
          for (int ff_outer_inner = 0; ff_outer_inner < 2; ++ff_outer_inner) {
            for (int yy_outer_inner = 0; yy_outer_inner < 3; ++yy_outer_inner) {
              conv2d_nchw[((ff_outer_inner * 18) + (yy_outer_inner * 3))] = (conv2d_nchw[((ff_outer_inner * 18) + (yy_outer_inner * 3))] + (pad_temp_shared[((((rc_outer_inner * 156) + (yy_outer_inner * 13)) + ((((int)threadIdx.x) % 3) * 3)) + rx_outer_inner)] * kernel_shared[(((((((int)threadIdx.x) / 3) * 160) + (ff_outer_inner * 80)) + (rc_outer_inner * 20)) + rx_outer_inner)]));
              conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 1)] = (conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 1)] + (pad_temp_shared[(((((rc_outer_inner * 156) + (yy_outer_inner * 13)) + ((((int)threadIdx.x) % 3) * 3)) + rx_outer_inner) + 1)] * kernel_shared[(((((((int)threadIdx.x) / 3) * 160) + (ff_outer_inner * 80)) + (rc_outer_inner * 20)) + rx_outer_inner)]));
              conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 2)] = (conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 2)] + (pad_temp_shared[(((((rc_outer_inner * 156) + (yy_outer_inner * 13)) + ((((int)threadIdx.x) % 3) * 3)) + rx_outer_inner) + 2)] * kernel_shared[(((((((int)threadIdx.x) / 3) * 160) + (ff_outer_inner * 80)) + (rc_outer_inner * 20)) + rx_outer_inner)]));
              conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 9)] = (conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 9)] + (pad_temp_shared[((((rc_outer_inner * 156) + (yy_outer_inner * 13)) + ((((int)threadIdx.x) % 3) * 3)) + rx_outer_inner)] * kernel_shared[((((((((int)threadIdx.x) / 3) * 160) + (ff_outer_inner * 80)) + (rc_outer_inner * 20)) + rx_outer_inner) + 40)]));
              conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 10)] = (conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 10)] + (pad_temp_shared[(((((rc_outer_inner * 156) + (yy_outer_inner * 13)) + ((((int)threadIdx.x) % 3) * 3)) + rx_outer_inner) + 1)] * kernel_shared[((((((((int)threadIdx.x) / 3) * 160) + (ff_outer_inner * 80)) + (rc_outer_inner * 20)) + rx_outer_inner) + 40)]));
              conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 11)] = (conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 11)] + (pad_temp_shared[(((((rc_outer_inner * 156) + (yy_outer_inner * 13)) + ((((int)threadIdx.x) % 3) * 3)) + rx_outer_inner) + 2)] * kernel_shared[((((((((int)threadIdx.x) / 3) * 160) + (ff_outer_inner * 80)) + (rc_outer_inner * 20)) + rx_outer_inner) + 40)]));
              conv2d_nchw[((ff_outer_inner * 18) + (yy_outer_inner * 3))] = (conv2d_nchw[((ff_outer_inner * 18) + (yy_outer_inner * 3))] + (pad_temp_shared[(((((rc_outer_inner * 156) + (yy_outer_inner * 13)) + ((((int)threadIdx.x) % 3) * 3)) + rx_outer_inner) + 39)] * kernel_shared[((((((((int)threadIdx.x) / 3) * 160) + (ff_outer_inner * 80)) + (rc_outer_inner * 20)) + rx_outer_inner) + 5)]));
              conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 1)] = (conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 1)] + (pad_temp_shared[(((((rc_outer_inner * 156) + (yy_outer_inner * 13)) + ((((int)threadIdx.x) % 3) * 3)) + rx_outer_inner) + 40)] * kernel_shared[((((((((int)threadIdx.x) / 3) * 160) + (ff_outer_inner * 80)) + (rc_outer_inner * 20)) + rx_outer_inner) + 5)]));
              conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 2)] = (conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 2)] + (pad_temp_shared[(((((rc_outer_inner * 156) + (yy_outer_inner * 13)) + ((((int)threadIdx.x) % 3) * 3)) + rx_outer_inner) + 41)] * kernel_shared[((((((((int)threadIdx.x) / 3) * 160) + (ff_outer_inner * 80)) + (rc_outer_inner * 20)) + rx_outer_inner) + 5)]));
              conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 9)] = (conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 9)] + (pad_temp_shared[(((((rc_outer_inner * 156) + (yy_outer_inner * 13)) + ((((int)threadIdx.x) % 3) * 3)) + rx_outer_inner) + 39)] * kernel_shared[((((((((int)threadIdx.x) / 3) * 160) + (ff_outer_inner * 80)) + (rc_outer_inner * 20)) + rx_outer_inner) + 45)]));
              conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 10)] = (conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 10)] + (pad_temp_shared[(((((rc_outer_inner * 156) + (yy_outer_inner * 13)) + ((((int)threadIdx.x) % 3) * 3)) + rx_outer_inner) + 40)] * kernel_shared[((((((((int)threadIdx.x) / 3) * 160) + (ff_outer_inner * 80)) + (rc_outer_inner * 20)) + rx_outer_inner) + 45)]));
              conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 11)] = (conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 11)] + (pad_temp_shared[(((((rc_outer_inner * 156) + (yy_outer_inner * 13)) + ((((int)threadIdx.x) % 3) * 3)) + rx_outer_inner) + 41)] * kernel_shared[((((((((int)threadIdx.x) / 3) * 160) + (ff_outer_inner * 80)) + (rc_outer_inner * 20)) + rx_outer_inner) + 45)]));
              conv2d_nchw[((ff_outer_inner * 18) + (yy_outer_inner * 3))] = (conv2d_nchw[((ff_outer_inner * 18) + (yy_outer_inner * 3))] + (pad_temp_shared[(((((rc_outer_inner * 156) + (yy_outer_inner * 13)) + ((((int)threadIdx.x) % 3) * 3)) + rx_outer_inner) + 78)] * kernel_shared[((((((((int)threadIdx.x) / 3) * 160) + (ff_outer_inner * 80)) + (rc_outer_inner * 20)) + rx_outer_inner) + 10)]));
              conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 1)] = (conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 1)] + (pad_temp_shared[(((((rc_outer_inner * 156) + (yy_outer_inner * 13)) + ((((int)threadIdx.x) % 3) * 3)) + rx_outer_inner) + 79)] * kernel_shared[((((((((int)threadIdx.x) / 3) * 160) + (ff_outer_inner * 80)) + (rc_outer_inner * 20)) + rx_outer_inner) + 10)]));
              conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 2)] = (conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 2)] + (pad_temp_shared[(((((rc_outer_inner * 156) + (yy_outer_inner * 13)) + ((((int)threadIdx.x) % 3) * 3)) + rx_outer_inner) + 80)] * kernel_shared[((((((((int)threadIdx.x) / 3) * 160) + (ff_outer_inner * 80)) + (rc_outer_inner * 20)) + rx_outer_inner) + 10)]));
              conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 9)] = (conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 9)] + (pad_temp_shared[(((((rc_outer_inner * 156) + (yy_outer_inner * 13)) + ((((int)threadIdx.x) % 3) * 3)) + rx_outer_inner) + 78)] * kernel_shared[((((((((int)threadIdx.x) / 3) * 160) + (ff_outer_inner * 80)) + (rc_outer_inner * 20)) + rx_outer_inner) + 50)]));
              conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 10)] = (conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 10)] + (pad_temp_shared[(((((rc_outer_inner * 156) + (yy_outer_inner * 13)) + ((((int)threadIdx.x) % 3) * 3)) + rx_outer_inner) + 79)] * kernel_shared[((((((((int)threadIdx.x) / 3) * 160) + (ff_outer_inner * 80)) + (rc_outer_inner * 20)) + rx_outer_inner) + 50)]));
              conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 11)] = (conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 11)] + (pad_temp_shared[(((((rc_outer_inner * 156) + (yy_outer_inner * 13)) + ((((int)threadIdx.x) % 3) * 3)) + rx_outer_inner) + 80)] * kernel_shared[((((((((int)threadIdx.x) / 3) * 160) + (ff_outer_inner * 80)) + (rc_outer_inner * 20)) + rx_outer_inner) + 50)]));
              conv2d_nchw[((ff_outer_inner * 18) + (yy_outer_inner * 3))] = (conv2d_nchw[((ff_outer_inner * 18) + (yy_outer_inner * 3))] + (pad_temp_shared[(((((rc_outer_inner * 156) + (yy_outer_inner * 13)) + ((((int)threadIdx.x) % 3) * 3)) + rx_outer_inner) + 117)] * kernel_shared[((((((((int)threadIdx.x) / 3) * 160) + (ff_outer_inner * 80)) + (rc_outer_inner * 20)) + rx_outer_inner) + 15)]));
              conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 1)] = (conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 1)] + (pad_temp_shared[(((((rc_outer_inner * 156) + (yy_outer_inner * 13)) + ((((int)threadIdx.x) % 3) * 3)) + rx_outer_inner) + 118)] * kernel_shared[((((((((int)threadIdx.x) / 3) * 160) + (ff_outer_inner * 80)) + (rc_outer_inner * 20)) + rx_outer_inner) + 15)]));
              conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 2)] = (conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 2)] + (pad_temp_shared[(((((rc_outer_inner * 156) + (yy_outer_inner * 13)) + ((((int)threadIdx.x) % 3) * 3)) + rx_outer_inner) + 119)] * kernel_shared[((((((((int)threadIdx.x) / 3) * 160) + (ff_outer_inner * 80)) + (rc_outer_inner * 20)) + rx_outer_inner) + 15)]));
              conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 9)] = (conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 9)] + (pad_temp_shared[(((((rc_outer_inner * 156) + (yy_outer_inner * 13)) + ((((int)threadIdx.x) % 3) * 3)) + rx_outer_inner) + 117)] * kernel_shared[((((((((int)threadIdx.x) / 3) * 160) + (ff_outer_inner * 80)) + (rc_outer_inner * 20)) + rx_outer_inner) + 55)]));
              conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 10)] = (conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 10)] + (pad_temp_shared[(((((rc_outer_inner * 156) + (yy_outer_inner * 13)) + ((((int)threadIdx.x) % 3) * 3)) + rx_outer_inner) + 118)] * kernel_shared[((((((((int)threadIdx.x) / 3) * 160) + (ff_outer_inner * 80)) + (rc_outer_inner * 20)) + rx_outer_inner) + 55)]));
              conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 11)] = (conv2d_nchw[(((ff_outer_inner * 18) + (yy_outer_inner * 3)) + 11)] + (pad_temp_shared[(((((rc_outer_inner * 156) + (yy_outer_inner * 13)) + ((((int)threadIdx.x) % 3) * 3)) + rx_outer_inner) + 119)] * kernel_shared[((((((((int)threadIdx.x) / 3) * 160) + (ff_outer_inner * 80)) + (rc_outer_inner * 20)) + rx_outer_inner) + 55)]));
            }
          }
        }
      }
    }
  }
  for (int i1_inner = 0; i1_inner < 4; ++i1_inner) {
    for (int i2_inner = 0; i2_inner < 3; ++i2_inner) {
      for (int i3_inner = 0; i3_inner < 3; ++i3_inner) {
        compute[((((((((((int)threadIdx.x) / 3) * 2916) + (i1_inner * 729)) + ((((int)blockIdx.x) / 3) * 81)) + (i2_inner * 27)) + ((((int)blockIdx.x) % 3) * 9)) + ((((int)threadIdx.x) % 3) * 3)) + i3_inner)] = max((conv2d_nchw[(((i1_inner * 9) + (i2_inner * 3)) + i3_inner)] + bias[(((((int)threadIdx.x) / 3) * 4) + i1_inner)]), 0.000000e+00f);
      }
    }
  }
}


