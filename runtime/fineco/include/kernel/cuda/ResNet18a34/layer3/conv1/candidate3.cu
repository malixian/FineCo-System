
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
extern "C" __global__ void __launch_bounds__(112) candidate3(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[28];
  __shared__ float pad_temp_shared[648];
  __shared__ float kernel_shared[4608];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[14] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[15] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[16] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[17] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[18] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[19] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  conv2d_nchw_local[20] = 0.000000e+00f;
  conv2d_nchw_local[7] = 0.000000e+00f;
  conv2d_nchw_local[21] = 0.000000e+00f;
  conv2d_nchw_local[8] = 0.000000e+00f;
  conv2d_nchw_local[22] = 0.000000e+00f;
  conv2d_nchw_local[9] = 0.000000e+00f;
  conv2d_nchw_local[23] = 0.000000e+00f;
  conv2d_nchw_local[10] = 0.000000e+00f;
  conv2d_nchw_local[24] = 0.000000e+00f;
  conv2d_nchw_local[11] = 0.000000e+00f;
  conv2d_nchw_local[25] = 0.000000e+00f;
  conv2d_nchw_local[12] = 0.000000e+00f;
  conv2d_nchw_local[26] = 0.000000e+00f;
  conv2d_nchw_local[13] = 0.000000e+00f;
  conv2d_nchw_local[27] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 16; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = (((((1 <= ((((((int)blockIdx.x) & 15) >> 2) * 7) + ((((int)threadIdx.x) % 81) / 9))) && (((((((int)blockIdx.x) & 15) >> 2) * 7) + ((((int)threadIdx.x) % 81) / 9)) < 29)) && (1 <= (((((int)blockIdx.x) & 3) * 7) + (((int)threadIdx.x) % 9)))) && ((((((int)blockIdx.x) & 3) * 7) + (((int)threadIdx.x) % 9)) < 29)) ? data[(((((((rc_outer_outer * 6272) + ((((int)threadIdx.x) / 81) * 784)) + (((((int)blockIdx.x) & 15) >> 2) * 196)) + (((((int)threadIdx.x) % 81) / 9) * 28)) + ((((int)blockIdx.x) & 3) * 7)) + (((int)threadIdx.x) % 9)) - 29)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 112)] = (((((1 <= ((((((int)blockIdx.x) & 15) >> 2) * 7) + (((((int)threadIdx.x) + 31) % 81) / 9))) && (((((((int)blockIdx.x) & 15) >> 2) * 7) + (((((int)threadIdx.x) + 31) % 81) / 9)) < 29)) && (1 <= (((((int)blockIdx.x) & 3) * 7) + ((((int)threadIdx.x) + 4) % 9)))) && ((((((int)blockIdx.x) & 3) * 7) + ((((int)threadIdx.x) + 4) % 9)) < 29)) ? data[(((((((rc_outer_outer * 6272) + (((((int)threadIdx.x) + 112) / 81) * 784)) + (((((int)blockIdx.x) & 15) >> 2) * 196)) + ((((((int)threadIdx.x) + 31) % 81) / 9) * 28)) + ((((int)blockIdx.x) & 3) * 7)) + ((((int)threadIdx.x) + 4) % 9)) - 29)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 224)] = (((((1 <= ((((((int)blockIdx.x) & 15) >> 2) * 7) + (((((int)threadIdx.x) + 62) % 81) / 9))) && (((((((int)blockIdx.x) & 15) >> 2) * 7) + (((((int)threadIdx.x) + 62) % 81) / 9)) < 29)) && (1 <= (((((int)blockIdx.x) & 3) * 7) + ((((int)threadIdx.x) + 8) % 9)))) && ((((((int)blockIdx.x) & 3) * 7) + ((((int)threadIdx.x) + 8) % 9)) < 29)) ? data[(((((((rc_outer_outer * 6272) + (((((int)threadIdx.x) + 224) / 81) * 784)) + (((((int)blockIdx.x) & 15) >> 2) * 196)) + ((((((int)threadIdx.x) + 62) % 81) / 9) * 28)) + ((((int)blockIdx.x) & 3) * 7)) + ((((int)threadIdx.x) + 8) % 9)) - 29)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 336)] = (((((1 <= ((((((int)blockIdx.x) & 15) >> 2) * 7) + (((((int)threadIdx.x) + 12) % 81) / 9))) && (((((((int)blockIdx.x) & 15) >> 2) * 7) + (((((int)threadIdx.x) + 12) % 81) / 9)) < 29)) && (1 <= (((((int)blockIdx.x) & 3) * 7) + ((((int)threadIdx.x) + 3) % 9)))) && ((((((int)blockIdx.x) & 3) * 7) + ((((int)threadIdx.x) + 3) % 9)) < 29)) ? data[(((((((rc_outer_outer * 6272) + (((((int)threadIdx.x) + 336) / 81) * 784)) + (((((int)blockIdx.x) & 15) >> 2) * 196)) + ((((((int)threadIdx.x) + 12) % 81) / 9) * 28)) + ((((int)blockIdx.x) & 3) * 7)) + ((((int)threadIdx.x) + 3) % 9)) - 29)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 448)] = (((((1 <= ((((((int)blockIdx.x) & 15) >> 2) * 7) + (((((int)threadIdx.x) + 43) % 81) / 9))) && (((((((int)blockIdx.x) & 15) >> 2) * 7) + (((((int)threadIdx.x) + 43) % 81) / 9)) < 29)) && (1 <= (((((int)blockIdx.x) & 3) * 7) + ((((int)threadIdx.x) + 7) % 9)))) && ((((((int)blockIdx.x) & 3) * 7) + ((((int)threadIdx.x) + 7) % 9)) < 29)) ? data[(((((((rc_outer_outer * 6272) + (((((int)threadIdx.x) + 448) / 81) * 784)) + (((((int)blockIdx.x) & 15) >> 2) * 196)) + ((((((int)threadIdx.x) + 43) % 81) / 9) * 28)) + ((((int)blockIdx.x) & 3) * 7)) + ((((int)threadIdx.x) + 7) % 9)) - 29)] : 0.000000e+00f);
    if (((int)threadIdx.x) < 88) {
      pad_temp_shared[(((int)threadIdx.x) + 560)] = (((((1 <= ((((((int)blockIdx.x) & 15) >> 2) * 7) + (((((int)threadIdx.x) + 74) % 81) / 9))) && (((((((int)blockIdx.x) & 15) >> 2) * 7) + (((((int)threadIdx.x) + 74) % 81) / 9)) < 29)) && (1 <= (((((int)blockIdx.x) & 3) * 7) + ((((int)threadIdx.x) + 2) % 9)))) && ((((((int)blockIdx.x) & 3) * 7) + ((((int)threadIdx.x) + 2) % 9)) < 29)) ? data[(((((((rc_outer_outer * 6272) + (((((int)threadIdx.x) + 560) / 81) * 784)) + (((((int)blockIdx.x) & 15) >> 2) * 196)) + ((((((int)threadIdx.x) + 74) % 81) / 9) * 28)) + ((((int)blockIdx.x) & 3) * 7)) + ((((int)threadIdx.x) + 2) % 9)) - 29)] : 0.000000e+00f);
    }
    kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) >> 4) * 73728) + ((((int)threadIdx.x) / 72) * 1152)) + (rc_outer_outer * 72)) + (((int)threadIdx.x) % 72))];
    kernel_shared[(((int)threadIdx.x) + 112)] = kernel[(((((((int)blockIdx.x) >> 4) * 73728) + (((((int)threadIdx.x) + 112) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 40) % 72))];
    kernel_shared[(((int)threadIdx.x) + 224)] = kernel[(((((((int)blockIdx.x) >> 4) * 73728) + (((((int)threadIdx.x) + 224) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 8) % 72))];
    kernel_shared[(((int)threadIdx.x) + 336)] = kernel[(((((((int)blockIdx.x) >> 4) * 73728) + (((((int)threadIdx.x) + 336) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 48) % 72))];
    kernel_shared[(((int)threadIdx.x) + 448)] = kernel[(((((((int)blockIdx.x) >> 4) * 73728) + (((((int)threadIdx.x) + 448) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 16) % 72))];
    kernel_shared[(((int)threadIdx.x) + 560)] = kernel[(((((((int)blockIdx.x) >> 4) * 73728) + (((((int)threadIdx.x) + 560) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 56) % 72))];
    kernel_shared[(((int)threadIdx.x) + 672)] = kernel[(((((((int)blockIdx.x) >> 4) * 73728) + (((((int)threadIdx.x) + 672) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 24) % 72))];
    kernel_shared[(((int)threadIdx.x) + 784)] = kernel[(((((((int)blockIdx.x) >> 4) * 73728) + (((((int)threadIdx.x) + 784) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 64) % 72))];
    kernel_shared[(((int)threadIdx.x) + 896)] = kernel[(((((((int)blockIdx.x) >> 4) * 73728) + (((((int)threadIdx.x) + 896) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 32) % 72))];
    kernel_shared[(((int)threadIdx.x) + 1008)] = kernel[((((((((int)blockIdx.x) >> 4) * 73728) + ((((int)threadIdx.x) / 72) * 1152)) + (rc_outer_outer * 72)) + (((int)threadIdx.x) % 72)) + 16128)];
    kernel_shared[(((int)threadIdx.x) + 1120)] = kernel[(((((((int)blockIdx.x) >> 4) * 73728) + (((((int)threadIdx.x) + 1120) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 40) % 72))];
    kernel_shared[(((int)threadIdx.x) + 1232)] = kernel[(((((((int)blockIdx.x) >> 4) * 73728) + (((((int)threadIdx.x) + 1232) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 8) % 72))];
    kernel_shared[(((int)threadIdx.x) + 1344)] = kernel[(((((((int)blockIdx.x) >> 4) * 73728) + (((((int)threadIdx.x) + 1344) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 48) % 72))];
    kernel_shared[(((int)threadIdx.x) + 1456)] = kernel[(((((((int)blockIdx.x) >> 4) * 73728) + (((((int)threadIdx.x) + 1456) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 16) % 72))];
    kernel_shared[(((int)threadIdx.x) + 1568)] = kernel[(((((((int)blockIdx.x) >> 4) * 73728) + (((((int)threadIdx.x) + 1568) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 56) % 72))];
    kernel_shared[(((int)threadIdx.x) + 1680)] = kernel[(((((((int)blockIdx.x) >> 4) * 73728) + (((((int)threadIdx.x) + 1680) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 24) % 72))];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[(((((((int)blockIdx.x) >> 4) * 73728) + (((((int)threadIdx.x) + 1792) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 64) % 72))];
    kernel_shared[(((int)threadIdx.x) + 1904)] = kernel[(((((((int)blockIdx.x) >> 4) * 73728) + (((((int)threadIdx.x) + 1904) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 32) % 72))];
    kernel_shared[(((int)threadIdx.x) + 2016)] = kernel[((((((((int)blockIdx.x) >> 4) * 73728) + ((((int)threadIdx.x) / 72) * 1152)) + (rc_outer_outer * 72)) + (((int)threadIdx.x) % 72)) + 32256)];
    kernel_shared[(((int)threadIdx.x) + 2128)] = kernel[(((((((int)blockIdx.x) >> 4) * 73728) + (((((int)threadIdx.x) + 2128) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 40) % 72))];
    kernel_shared[(((int)threadIdx.x) + 2240)] = kernel[(((((((int)blockIdx.x) >> 4) * 73728) + (((((int)threadIdx.x) + 2240) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 8) % 72))];
    kernel_shared[(((int)threadIdx.x) + 2352)] = kernel[(((((((int)blockIdx.x) >> 4) * 73728) + (((((int)threadIdx.x) + 2352) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 48) % 72))];
    kernel_shared[(((int)threadIdx.x) + 2464)] = kernel[(((((((int)blockIdx.x) >> 4) * 73728) + (((((int)threadIdx.x) + 2464) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 16) % 72))];
    kernel_shared[(((int)threadIdx.x) + 2576)] = kernel[(((((((int)blockIdx.x) >> 4) * 73728) + (((((int)threadIdx.x) + 2576) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 56) % 72))];
    kernel_shared[(((int)threadIdx.x) + 2688)] = kernel[(((((((int)blockIdx.x) >> 4) * 73728) + (((((int)threadIdx.x) + 2688) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 24) % 72))];
    kernel_shared[(((int)threadIdx.x) + 2800)] = kernel[(((((((int)blockIdx.x) >> 4) * 73728) + (((((int)threadIdx.x) + 2800) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 64) % 72))];
    kernel_shared[(((int)threadIdx.x) + 2912)] = kernel[(((((((int)blockIdx.x) >> 4) * 73728) + (((((int)threadIdx.x) + 2912) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 32) % 72))];
    kernel_shared[(((int)threadIdx.x) + 3024)] = kernel[((((((((int)blockIdx.x) >> 4) * 73728) + ((((int)threadIdx.x) / 72) * 1152)) + (rc_outer_outer * 72)) + (((int)threadIdx.x) % 72)) + 48384)];
    kernel_shared[(((int)threadIdx.x) + 3136)] = kernel[(((((((int)blockIdx.x) >> 4) * 73728) + (((((int)threadIdx.x) + 3136) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 40) % 72))];
    kernel_shared[(((int)threadIdx.x) + 3248)] = kernel[(((((((int)blockIdx.x) >> 4) * 73728) + (((((int)threadIdx.x) + 3248) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 8) % 72))];
    kernel_shared[(((int)threadIdx.x) + 3360)] = kernel[(((((((int)blockIdx.x) >> 4) * 73728) + (((((int)threadIdx.x) + 3360) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 48) % 72))];
    kernel_shared[(((int)threadIdx.x) + 3472)] = kernel[(((((((int)blockIdx.x) >> 4) * 73728) + (((((int)threadIdx.x) + 3472) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 16) % 72))];
    kernel_shared[(((int)threadIdx.x) + 3584)] = kernel[(((((((int)blockIdx.x) >> 4) * 73728) + (((((int)threadIdx.x) + 3584) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 56) % 72))];
    kernel_shared[(((int)threadIdx.x) + 3696)] = kernel[(((((((int)blockIdx.x) >> 4) * 73728) + (((((int)threadIdx.x) + 3696) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 24) % 72))];
    kernel_shared[(((int)threadIdx.x) + 3808)] = kernel[(((((((int)blockIdx.x) >> 4) * 73728) + (((((int)threadIdx.x) + 3808) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 64) % 72))];
    kernel_shared[(((int)threadIdx.x) + 3920)] = kernel[(((((((int)blockIdx.x) >> 4) * 73728) + (((((int)threadIdx.x) + 3920) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 32) % 72))];
    kernel_shared[(((int)threadIdx.x) + 4032)] = kernel[((((((((int)blockIdx.x) >> 4) * 73728) + ((((int)threadIdx.x) / 72) * 1152)) + (rc_outer_outer * 72)) + (((int)threadIdx.x) % 72)) + 64512)];
    kernel_shared[(((int)threadIdx.x) + 4144)] = kernel[(((((((int)blockIdx.x) >> 4) * 73728) + (((((int)threadIdx.x) + 4144) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 40) % 72))];
    kernel_shared[(((int)threadIdx.x) + 4256)] = kernel[(((((((int)blockIdx.x) >> 4) * 73728) + (((((int)threadIdx.x) + 4256) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 8) % 72))];
    kernel_shared[(((int)threadIdx.x) + 4368)] = kernel[(((((((int)blockIdx.x) >> 4) * 73728) + (((((int)threadIdx.x) + 4368) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 48) % 72))];
    kernel_shared[(((int)threadIdx.x) + 4480)] = kernel[(((((((int)blockIdx.x) >> 4) * 73728) + (((((int)threadIdx.x) + 4480) / 72) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 16) % 72))];
    if (((int)threadIdx.x) < 16) {
      kernel_shared[(((int)threadIdx.x) + 4592)] = kernel[(((((((int)blockIdx.x) >> 4) * 73728) + (((((int)threadIdx.x) + 4592) / 72) * 1152)) + (rc_outer_outer * 72)) + (((int)threadIdx.x) + 56))];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 8; ++rc_outer_inner) {
      for (int ry_outer_inner = 0; ry_outer_inner < 3; ++ry_outer_inner) {
        for (int rx_inner = 0; rx_inner < 3; ++rx_inner) {
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((rc_outer_inner * 81) + (ry_outer_inner * 9)) + rx_inner) + (((int)threadIdx.x) % 7))] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[((((rc_outer_inner * 81) + (ry_outer_inner * 9)) + rx_inner) + (((int)threadIdx.x) % 7))] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 2304)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((rc_outer_inner * 81) + (ry_outer_inner * 9)) + rx_inner) + (((int)threadIdx.x) % 7)) + 9)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[(((((rc_outer_inner * 81) + (ry_outer_inner * 9)) + rx_inner) + (((int)threadIdx.x) % 7)) + 9)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 2304)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((rc_outer_inner * 81) + (ry_outer_inner * 9)) + rx_inner) + (((int)threadIdx.x) % 7)) + 18)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[(((((rc_outer_inner * 81) + (ry_outer_inner * 9)) + rx_inner) + (((int)threadIdx.x) % 7)) + 18)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 2304)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((rc_outer_inner * 81) + (ry_outer_inner * 9)) + rx_inner) + (((int)threadIdx.x) % 7)) + 27)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[(((((rc_outer_inner * 81) + (ry_outer_inner * 9)) + rx_inner) + (((int)threadIdx.x) % 7)) + 27)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 2304)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((rc_outer_inner * 81) + (ry_outer_inner * 9)) + rx_inner) + (((int)threadIdx.x) % 7)) + 36)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[(((((rc_outer_inner * 81) + (ry_outer_inner * 9)) + rx_inner) + (((int)threadIdx.x) % 7)) + 36)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 2304)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((rc_outer_inner * 81) + (ry_outer_inner * 9)) + rx_inner) + (((int)threadIdx.x) % 7)) + 45)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[(((((rc_outer_inner * 81) + (ry_outer_inner * 9)) + rx_inner) + (((int)threadIdx.x) % 7)) + 45)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 2304)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((rc_outer_inner * 81) + (ry_outer_inner * 9)) + rx_inner) + (((int)threadIdx.x) % 7)) + 54)] * kernel_shared[(((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner)]));
          conv2d_nchw_local[20] = (conv2d_nchw_local[20] + (pad_temp_shared[(((((rc_outer_inner * 81) + (ry_outer_inner * 9)) + rx_inner) + (((int)threadIdx.x) % 7)) + 54)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 2304)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[((((rc_outer_inner * 81) + (ry_outer_inner * 9)) + rx_inner) + (((int)threadIdx.x) % 7))] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 72)]));
          conv2d_nchw_local[21] = (conv2d_nchw_local[21] + (pad_temp_shared[((((rc_outer_inner * 81) + (ry_outer_inner * 9)) + rx_inner) + (((int)threadIdx.x) % 7))] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 2376)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[(((((rc_outer_inner * 81) + (ry_outer_inner * 9)) + rx_inner) + (((int)threadIdx.x) % 7)) + 9)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 72)]));
          conv2d_nchw_local[22] = (conv2d_nchw_local[22] + (pad_temp_shared[(((((rc_outer_inner * 81) + (ry_outer_inner * 9)) + rx_inner) + (((int)threadIdx.x) % 7)) + 9)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 2376)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[(((((rc_outer_inner * 81) + (ry_outer_inner * 9)) + rx_inner) + (((int)threadIdx.x) % 7)) + 18)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 72)]));
          conv2d_nchw_local[23] = (conv2d_nchw_local[23] + (pad_temp_shared[(((((rc_outer_inner * 81) + (ry_outer_inner * 9)) + rx_inner) + (((int)threadIdx.x) % 7)) + 18)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 2376)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[(((((rc_outer_inner * 81) + (ry_outer_inner * 9)) + rx_inner) + (((int)threadIdx.x) % 7)) + 27)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 72)]));
          conv2d_nchw_local[24] = (conv2d_nchw_local[24] + (pad_temp_shared[(((((rc_outer_inner * 81) + (ry_outer_inner * 9)) + rx_inner) + (((int)threadIdx.x) % 7)) + 27)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 2376)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[(((((rc_outer_inner * 81) + (ry_outer_inner * 9)) + rx_inner) + (((int)threadIdx.x) % 7)) + 36)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 72)]));
          conv2d_nchw_local[25] = (conv2d_nchw_local[25] + (pad_temp_shared[(((((rc_outer_inner * 81) + (ry_outer_inner * 9)) + rx_inner) + (((int)threadIdx.x) % 7)) + 36)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 2376)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[(((((rc_outer_inner * 81) + (ry_outer_inner * 9)) + rx_inner) + (((int)threadIdx.x) % 7)) + 45)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 72)]));
          conv2d_nchw_local[26] = (conv2d_nchw_local[26] + (pad_temp_shared[(((((rc_outer_inner * 81) + (ry_outer_inner * 9)) + rx_inner) + (((int)threadIdx.x) % 7)) + 45)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 2376)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[(((((rc_outer_inner * 81) + (ry_outer_inner * 9)) + rx_inner) + (((int)threadIdx.x) % 7)) + 54)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 72)]));
          conv2d_nchw_local[27] = (conv2d_nchw_local[27] + (pad_temp_shared[(((((rc_outer_inner * 81) + (ry_outer_inner * 9)) + rx_inner) + (((int)threadIdx.x) % 7)) + 54)] * kernel_shared[((((((((int)threadIdx.x) / 7) * 144) + (rc_outer_inner * 9)) + (ry_outer_inner * 3)) + rx_inner) + 2376)]));
        }
      }
    }
  }
  for (int ff_inner = 0; ff_inner < 2; ++ff_inner) {
    for (int yy_inner = 0; yy_inner < 7; ++yy_inner) {
      conv2d_nchw[((((((((((int)blockIdx.x) >> 4) * 50176) + ((((int)threadIdx.x) / 7) * 1568)) + (ff_inner * 784)) + (((((int)blockIdx.x) & 15) >> 2) * 196)) + (yy_inner * 28)) + ((((int)blockIdx.x) & 3) * 7)) + (((int)threadIdx.x) % 7))] = conv2d_nchw_local[((ff_inner * 7) + yy_inner)];
      conv2d_nchw[(((((((((((int)blockIdx.x) >> 4) * 50176) + ((((int)threadIdx.x) / 7) * 1568)) + (ff_inner * 784)) + (((((int)blockIdx.x) & 15) >> 2) * 196)) + (yy_inner * 28)) + ((((int)blockIdx.x) & 3) * 7)) + (((int)threadIdx.x) % 7)) + 25088)] = conv2d_nchw_local[(((ff_inner * 7) + yy_inner) + 14)];
    }
  }
}


