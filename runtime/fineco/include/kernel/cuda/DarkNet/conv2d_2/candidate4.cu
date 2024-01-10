
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
extern "C" __global__ void __launch_bounds__(256) candidate4(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ compute, float* __restrict__ bias) {
  float conv2d_nchw[32];
  __shared__ float pad_temp_shared[2880];
  __shared__ float kernel_shared[9216];
  conv2d_nchw[0] = 0.000000e+00f;
  conv2d_nchw[16] = 0.000000e+00f;
  conv2d_nchw[1] = 0.000000e+00f;
  conv2d_nchw[17] = 0.000000e+00f;
  conv2d_nchw[2] = 0.000000e+00f;
  conv2d_nchw[18] = 0.000000e+00f;
  conv2d_nchw[3] = 0.000000e+00f;
  conv2d_nchw[19] = 0.000000e+00f;
  conv2d_nchw[4] = 0.000000e+00f;
  conv2d_nchw[20] = 0.000000e+00f;
  conv2d_nchw[5] = 0.000000e+00f;
  conv2d_nchw[21] = 0.000000e+00f;
  conv2d_nchw[6] = 0.000000e+00f;
  conv2d_nchw[22] = 0.000000e+00f;
  conv2d_nchw[7] = 0.000000e+00f;
  conv2d_nchw[23] = 0.000000e+00f;
  conv2d_nchw[8] = 0.000000e+00f;
  conv2d_nchw[24] = 0.000000e+00f;
  conv2d_nchw[9] = 0.000000e+00f;
  conv2d_nchw[25] = 0.000000e+00f;
  conv2d_nchw[10] = 0.000000e+00f;
  conv2d_nchw[26] = 0.000000e+00f;
  conv2d_nchw[11] = 0.000000e+00f;
  conv2d_nchw[27] = 0.000000e+00f;
  conv2d_nchw[12] = 0.000000e+00f;
  conv2d_nchw[28] = 0.000000e+00f;
  conv2d_nchw[13] = 0.000000e+00f;
  conv2d_nchw[29] = 0.000000e+00f;
  conv2d_nchw[14] = 0.000000e+00f;
  conv2d_nchw[30] = 0.000000e+00f;
  conv2d_nchw[15] = 0.000000e+00f;
  conv2d_nchw[31] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 2; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = (((((1 <= (((((int)blockIdx.x) / 14) * 16) + ((((int)threadIdx.x) % 180) / 10))) && ((((((int)blockIdx.x) / 14) * 16) + ((((int)threadIdx.x) % 180) / 10)) < 113)) && (1 <= (((((int)blockIdx.x) % 14) * 8) + (((int)threadIdx.x) % 10)))) && ((((((int)blockIdx.x) % 14) * 8) + (((int)threadIdx.x) % 10)) < 113)) ? data[(((((((rc_outer_outer * 200704) + ((((int)threadIdx.x) / 180) * 12544)) + ((((int)blockIdx.x) / 14) * 1792)) + (((((int)threadIdx.x) % 180) / 10) * 112)) + ((((int)blockIdx.x) % 14) * 8)) + (((int)threadIdx.x) % 10)) - 113)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 256)] = (((((1 <= (((((int)blockIdx.x) / 14) * 16) + ((((((int)threadIdx.x) >> 1) + 38) % 90) / 5))) && ((((((int)blockIdx.x) / 14) * 16) + ((((((int)threadIdx.x) >> 1) + 38) % 90) / 5)) < 113)) && (1 <= (((((int)blockIdx.x) % 14) * 8) + ((((int)threadIdx.x) + 6) % 10)))) && ((((((int)blockIdx.x) % 14) * 8) + ((((int)threadIdx.x) + 6) % 10)) < 113)) ? data[(((((((rc_outer_outer * 200704) + (((((int)threadIdx.x) + 256) / 180) * 12544)) + ((((int)blockIdx.x) / 14) * 1792)) + (((((((int)threadIdx.x) >> 1) + 38) % 90) / 5) * 112)) + ((((int)blockIdx.x) % 14) * 8)) + ((((int)threadIdx.x) + 6) % 10)) - 113)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 512)] = (((((1 <= (((((int)blockIdx.x) / 14) * 16) + ((((((int)threadIdx.x) >> 1) + 76) % 90) / 5))) && ((((((int)blockIdx.x) / 14) * 16) + ((((((int)threadIdx.x) >> 1) + 76) % 90) / 5)) < 113)) && (1 <= (((((int)blockIdx.x) % 14) * 8) + ((((int)threadIdx.x) + 2) % 10)))) && ((((((int)blockIdx.x) % 14) * 8) + ((((int)threadIdx.x) + 2) % 10)) < 113)) ? data[(((((((rc_outer_outer * 200704) + (((((int)threadIdx.x) + 512) / 180) * 12544)) + ((((int)blockIdx.x) / 14) * 1792)) + (((((((int)threadIdx.x) >> 1) + 76) % 90) / 5) * 112)) + ((((int)blockIdx.x) % 14) * 8)) + ((((int)threadIdx.x) + 2) % 10)) - 113)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 768)] = (((((1 <= (((((int)blockIdx.x) / 14) * 16) + ((((((int)threadIdx.x) >> 1) + 24) % 90) / 5))) && ((((((int)blockIdx.x) / 14) * 16) + ((((((int)threadIdx.x) >> 1) + 24) % 90) / 5)) < 113)) && (1 <= (((((int)blockIdx.x) % 14) * 8) + ((((int)threadIdx.x) + 8) % 10)))) && ((((((int)blockIdx.x) % 14) * 8) + ((((int)threadIdx.x) + 8) % 10)) < 113)) ? data[(((((((rc_outer_outer * 200704) + (((((int)threadIdx.x) + 768) / 180) * 12544)) + ((((int)blockIdx.x) / 14) * 1792)) + (((((((int)threadIdx.x) >> 1) + 24) % 90) / 5) * 112)) + ((((int)blockIdx.x) % 14) * 8)) + ((((int)threadIdx.x) + 8) % 10)) - 113)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1024)] = (((((1 <= (((((int)blockIdx.x) / 14) * 16) + ((((((int)threadIdx.x) >> 1) + 62) % 90) / 5))) && ((((((int)blockIdx.x) / 14) * 16) + ((((((int)threadIdx.x) >> 1) + 62) % 90) / 5)) < 113)) && (1 <= (((((int)blockIdx.x) % 14) * 8) + ((((int)threadIdx.x) + 4) % 10)))) && ((((((int)blockIdx.x) % 14) * 8) + ((((int)threadIdx.x) + 4) % 10)) < 113)) ? data[(((((((rc_outer_outer * 200704) + (((((int)threadIdx.x) + 1024) / 180) * 12544)) + ((((int)blockIdx.x) / 14) * 1792)) + (((((((int)threadIdx.x) >> 1) + 62) % 90) / 5) * 112)) + ((((int)blockIdx.x) % 14) * 8)) + ((((int)threadIdx.x) + 4) % 10)) - 113)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1280)] = (((((1 <= (((((int)blockIdx.x) / 14) * 16) + (((((int)threadIdx.x) / 10) + 2) % 18))) && ((((((int)blockIdx.x) / 14) * 16) + (((((int)threadIdx.x) / 10) + 2) % 18)) < 113)) && (1 <= (((((int)blockIdx.x) % 14) * 8) + (((int)threadIdx.x) % 10)))) && ((((((int)blockIdx.x) % 14) * 8) + (((int)threadIdx.x) % 10)) < 113)) ? data[(((((((rc_outer_outer * 200704) + (((((int)threadIdx.x) + 1280) / 180) * 12544)) + ((((int)blockIdx.x) / 14) * 1792)) + ((((((int)threadIdx.x) / 10) + 2) % 18) * 112)) + ((((int)blockIdx.x) % 14) * 8)) + (((int)threadIdx.x) % 10)) - 113)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1536)] = (((((1 <= (((((int)blockIdx.x) / 14) * 16) + ((((((int)threadIdx.x) >> 1) + 48) % 90) / 5))) && ((((((int)blockIdx.x) / 14) * 16) + ((((((int)threadIdx.x) >> 1) + 48) % 90) / 5)) < 113)) && (1 <= (((((int)blockIdx.x) % 14) * 8) + ((((int)threadIdx.x) + 6) % 10)))) && ((((((int)blockIdx.x) % 14) * 8) + ((((int)threadIdx.x) + 6) % 10)) < 113)) ? data[(((((((rc_outer_outer * 200704) + (((((int)threadIdx.x) + 1536) / 180) * 12544)) + ((((int)blockIdx.x) / 14) * 1792)) + (((((((int)threadIdx.x) >> 1) + 48) % 90) / 5) * 112)) + ((((int)blockIdx.x) % 14) * 8)) + ((((int)threadIdx.x) + 6) % 10)) - 113)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1792)] = (((((1 <= (((((int)blockIdx.x) / 14) * 16) + ((((((int)threadIdx.x) >> 1) + 86) % 90) / 5))) && ((((((int)blockIdx.x) / 14) * 16) + ((((((int)threadIdx.x) >> 1) + 86) % 90) / 5)) < 113)) && (1 <= (((((int)blockIdx.x) % 14) * 8) + ((((int)threadIdx.x) + 2) % 10)))) && ((((((int)blockIdx.x) % 14) * 8) + ((((int)threadIdx.x) + 2) % 10)) < 113)) ? data[(((((((rc_outer_outer * 200704) + (((((int)threadIdx.x) + 1792) / 180) * 12544)) + ((((int)blockIdx.x) / 14) * 1792)) + (((((((int)threadIdx.x) >> 1) + 86) % 90) / 5) * 112)) + ((((int)blockIdx.x) % 14) * 8)) + ((((int)threadIdx.x) + 2) % 10)) - 113)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 2048)] = (((((1 <= (((((int)blockIdx.x) / 14) * 16) + ((((((int)threadIdx.x) >> 1) + 34) % 90) / 5))) && ((((((int)blockIdx.x) / 14) * 16) + ((((((int)threadIdx.x) >> 1) + 34) % 90) / 5)) < 113)) && (1 <= (((((int)blockIdx.x) % 14) * 8) + ((((int)threadIdx.x) + 8) % 10)))) && ((((((int)blockIdx.x) % 14) * 8) + ((((int)threadIdx.x) + 8) % 10)) < 113)) ? data[(((((((rc_outer_outer * 200704) + (((((int)threadIdx.x) + 2048) / 180) * 12544)) + ((((int)blockIdx.x) / 14) * 1792)) + (((((((int)threadIdx.x) >> 1) + 34) % 90) / 5) * 112)) + ((((int)blockIdx.x) % 14) * 8)) + ((((int)threadIdx.x) + 8) % 10)) - 113)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 2304)] = (((((1 <= (((((int)blockIdx.x) / 14) * 16) + ((((((int)threadIdx.x) >> 1) + 72) % 90) / 5))) && ((((((int)blockIdx.x) / 14) * 16) + ((((((int)threadIdx.x) >> 1) + 72) % 90) / 5)) < 113)) && (1 <= (((((int)blockIdx.x) % 14) * 8) + ((((int)threadIdx.x) + 4) % 10)))) && ((((((int)blockIdx.x) % 14) * 8) + ((((int)threadIdx.x) + 4) % 10)) < 113)) ? data[(((((((rc_outer_outer * 200704) + (((((int)threadIdx.x) + 2304) / 180) * 12544)) + ((((int)blockIdx.x) / 14) * 1792)) + (((((((int)threadIdx.x) >> 1) + 72) % 90) / 5) * 112)) + ((((int)blockIdx.x) % 14) * 8)) + ((((int)threadIdx.x) + 4) % 10)) - 113)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 2560)] = (((((1 <= (((((int)blockIdx.x) / 14) * 16) + (((((int)threadIdx.x) / 10) + 4) % 18))) && ((((((int)blockIdx.x) / 14) * 16) + (((((int)threadIdx.x) / 10) + 4) % 18)) < 113)) && (1 <= (((((int)blockIdx.x) % 14) * 8) + (((int)threadIdx.x) % 10)))) && ((((((int)blockIdx.x) % 14) * 8) + (((int)threadIdx.x) % 10)) < 113)) ? data[(((((((rc_outer_outer * 200704) + (((((int)threadIdx.x) + 2560) / 180) * 12544)) + ((((int)blockIdx.x) / 14) * 1792)) + ((((((int)threadIdx.x) / 10) + 4) % 18) * 112)) + ((((int)blockIdx.x) % 14) * 8)) + (((int)threadIdx.x) % 10)) - 113)] : 0.000000e+00f);
    if (((int)threadIdx.x) < 64) {
      pad_temp_shared[(((int)threadIdx.x) + 2816)] = (((((((((int)blockIdx.x) / 14) * 16) + ((((((int)threadIdx.x) >> 1) + 58) % 90) / 5)) < 113) && (1 <= (((((int)blockIdx.x) % 14) * 8) + ((((int)threadIdx.x) + 6) % 10)))) && ((((((int)blockIdx.x) % 14) * 8) + ((((int)threadIdx.x) + 6) % 10)) < 113)) ? data[(((((((rc_outer_outer * 200704) + (((((int)threadIdx.x) + 2816) / 180) * 12544)) + ((((int)blockIdx.x) / 14) * 1792)) + (((((((int)threadIdx.x) >> 1) + 58) % 90) / 5) * 112)) + ((((int)blockIdx.x) % 14) * 8)) + ((((int)threadIdx.x) + 6) % 10)) - 113)] : 0.000000e+00f);
    }
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)threadIdx.x) / 144) * 288) + (rc_outer_outer * 144)) + (((int)threadIdx.x) % 144))];
    kernel_shared[(((int)threadIdx.x) + 256)] = kernel[(((((((int)threadIdx.x) + 256) / 144) * 288) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 112) % 144))];
    kernel_shared[(((int)threadIdx.x) + 512)] = kernel[(((((((int)threadIdx.x) + 512) / 144) * 288) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 80) % 144))];
    kernel_shared[(((int)threadIdx.x) + 768)] = kernel[(((((((int)threadIdx.x) + 768) / 144) * 288) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 48) % 144))];
    kernel_shared[(((int)threadIdx.x) + 1024)] = kernel[(((((((int)threadIdx.x) + 1024) / 144) * 288) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 16) % 144))];
    kernel_shared[(((int)threadIdx.x) + 1280)] = kernel[(((((((int)threadIdx.x) + 1280) / 144) * 288) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 128) % 144))];
    kernel_shared[(((int)threadIdx.x) + 1536)] = kernel[(((((((int)threadIdx.x) + 1536) / 144) * 288) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 96) % 144))];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[(((((((int)threadIdx.x) + 1792) / 144) * 288) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 64) % 144))];
    kernel_shared[(((int)threadIdx.x) + 2048)] = kernel[(((((((int)threadIdx.x) + 2048) / 144) * 288) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 32) % 144))];
    kernel_shared[(((int)threadIdx.x) + 2304)] = kernel[(((((((int)threadIdx.x) / 144) * 288) + (rc_outer_outer * 144)) + (((int)threadIdx.x) % 144)) + 4608)];
    kernel_shared[(((int)threadIdx.x) + 2560)] = kernel[(((((((int)threadIdx.x) + 2560) / 144) * 288) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 112) % 144))];
    kernel_shared[(((int)threadIdx.x) + 2816)] = kernel[(((((((int)threadIdx.x) + 2816) / 144) * 288) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 80) % 144))];
    kernel_shared[(((int)threadIdx.x) + 3072)] = kernel[(((((((int)threadIdx.x) + 3072) / 144) * 288) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 48) % 144))];
    kernel_shared[(((int)threadIdx.x) + 3328)] = kernel[(((((((int)threadIdx.x) + 3328) / 144) * 288) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 16) % 144))];
    kernel_shared[(((int)threadIdx.x) + 3584)] = kernel[(((((((int)threadIdx.x) + 3584) / 144) * 288) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 128) % 144))];
    kernel_shared[(((int)threadIdx.x) + 3840)] = kernel[(((((((int)threadIdx.x) + 3840) / 144) * 288) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 96) % 144))];
    kernel_shared[(((int)threadIdx.x) + 4096)] = kernel[(((((((int)threadIdx.x) + 4096) / 144) * 288) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 64) % 144))];
    kernel_shared[(((int)threadIdx.x) + 4352)] = kernel[(((((((int)threadIdx.x) + 4352) / 144) * 288) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 32) % 144))];
    kernel_shared[(((int)threadIdx.x) + 4608)] = kernel[(((((((int)threadIdx.x) / 144) * 288) + (rc_outer_outer * 144)) + (((int)threadIdx.x) % 144)) + 9216)];
    kernel_shared[(((int)threadIdx.x) + 4864)] = kernel[(((((((int)threadIdx.x) + 4864) / 144) * 288) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 112) % 144))];
    kernel_shared[(((int)threadIdx.x) + 5120)] = kernel[(((((((int)threadIdx.x) + 5120) / 144) * 288) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 80) % 144))];
    kernel_shared[(((int)threadIdx.x) + 5376)] = kernel[(((((((int)threadIdx.x) + 5376) / 144) * 288) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 48) % 144))];
    kernel_shared[(((int)threadIdx.x) + 5632)] = kernel[(((((((int)threadIdx.x) + 5632) / 144) * 288) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 16) % 144))];
    kernel_shared[(((int)threadIdx.x) + 5888)] = kernel[(((((((int)threadIdx.x) + 5888) / 144) * 288) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 128) % 144))];
    kernel_shared[(((int)threadIdx.x) + 6144)] = kernel[(((((((int)threadIdx.x) + 6144) / 144) * 288) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 96) % 144))];
    kernel_shared[(((int)threadIdx.x) + 6400)] = kernel[(((((((int)threadIdx.x) + 6400) / 144) * 288) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 64) % 144))];
    kernel_shared[(((int)threadIdx.x) + 6656)] = kernel[(((((((int)threadIdx.x) + 6656) / 144) * 288) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 32) % 144))];
    kernel_shared[(((int)threadIdx.x) + 6912)] = kernel[(((((((int)threadIdx.x) / 144) * 288) + (rc_outer_outer * 144)) + (((int)threadIdx.x) % 144)) + 13824)];
    kernel_shared[(((int)threadIdx.x) + 7168)] = kernel[(((((((int)threadIdx.x) + 7168) / 144) * 288) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 112) % 144))];
    kernel_shared[(((int)threadIdx.x) + 7424)] = kernel[(((((((int)threadIdx.x) + 7424) / 144) * 288) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 80) % 144))];
    kernel_shared[(((int)threadIdx.x) + 7680)] = kernel[(((((((int)threadIdx.x) + 7680) / 144) * 288) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 48) % 144))];
    kernel_shared[(((int)threadIdx.x) + 7936)] = kernel[(((((((int)threadIdx.x) + 7936) / 144) * 288) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 16) % 144))];
    kernel_shared[(((int)threadIdx.x) + 8192)] = kernel[(((((((int)threadIdx.x) + 8192) / 144) * 288) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 128) % 144))];
    kernel_shared[(((int)threadIdx.x) + 8448)] = kernel[(((((((int)threadIdx.x) + 8448) / 144) * 288) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 96) % 144))];
    kernel_shared[(((int)threadIdx.x) + 8704)] = kernel[(((((((int)threadIdx.x) + 8704) / 144) * 288) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 64) % 144))];
    kernel_shared[(((int)threadIdx.x) + 8960)] = kernel[(((((((int)threadIdx.x) + 8960) / 144) * 288) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 32) % 144))];
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 4; ++rc_outer_inner) {
      for (int rc_inner = 0; rc_inner < 4; ++rc_inner) {
        for (int ry_inner = 0; ry_inner < 3; ++ry_inner) {
          for (int rx_inner = 0; rx_inner < 3; ++rx_inner) {
            conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((((rc_outer_inner * 720) + (rc_inner * 180)) + (((((int)threadIdx.x) & 31) >> 3) * 20)) + (ry_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7))] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 1152) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
            conv2d_nchw[16] = (conv2d_nchw[16] + (pad_temp_shared[(((((((rc_outer_inner * 720) + (rc_inner * 180)) + (((((int)threadIdx.x) & 31) >> 3) * 20)) + (ry_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7)) + 80)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 1152) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
            conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((((((rc_outer_inner * 720) + (rc_inner * 180)) + (((((int)threadIdx.x) & 31) >> 3) * 20)) + (ry_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7)) + 10)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 1152) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
            conv2d_nchw[17] = (conv2d_nchw[17] + (pad_temp_shared[(((((((rc_outer_inner * 720) + (rc_inner * 180)) + (((((int)threadIdx.x) & 31) >> 3) * 20)) + (ry_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7)) + 90)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 1152) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
            conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((((rc_outer_inner * 720) + (rc_inner * 180)) + (((((int)threadIdx.x) & 31) >> 3) * 20)) + (ry_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7))] * kernel_shared[(((((((((int)threadIdx.x) >> 5) * 1152) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 144)]));
            conv2d_nchw[18] = (conv2d_nchw[18] + (pad_temp_shared[(((((((rc_outer_inner * 720) + (rc_inner * 180)) + (((((int)threadIdx.x) & 31) >> 3) * 20)) + (ry_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7)) + 80)] * kernel_shared[(((((((((int)threadIdx.x) >> 5) * 1152) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 144)]));
            conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((((((rc_outer_inner * 720) + (rc_inner * 180)) + (((((int)threadIdx.x) & 31) >> 3) * 20)) + (ry_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7)) + 10)] * kernel_shared[(((((((((int)threadIdx.x) >> 5) * 1152) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 144)]));
            conv2d_nchw[19] = (conv2d_nchw[19] + (pad_temp_shared[(((((((rc_outer_inner * 720) + (rc_inner * 180)) + (((((int)threadIdx.x) & 31) >> 3) * 20)) + (ry_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7)) + 90)] * kernel_shared[(((((((((int)threadIdx.x) >> 5) * 1152) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 144)]));
            conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[((((((rc_outer_inner * 720) + (rc_inner * 180)) + (((((int)threadIdx.x) & 31) >> 3) * 20)) + (ry_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7))] * kernel_shared[(((((((((int)threadIdx.x) >> 5) * 1152) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 288)]));
            conv2d_nchw[20] = (conv2d_nchw[20] + (pad_temp_shared[(((((((rc_outer_inner * 720) + (rc_inner * 180)) + (((((int)threadIdx.x) & 31) >> 3) * 20)) + (ry_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7)) + 80)] * kernel_shared[(((((((((int)threadIdx.x) >> 5) * 1152) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 288)]));
            conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[(((((((rc_outer_inner * 720) + (rc_inner * 180)) + (((((int)threadIdx.x) & 31) >> 3) * 20)) + (ry_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7)) + 10)] * kernel_shared[(((((((((int)threadIdx.x) >> 5) * 1152) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 288)]));
            conv2d_nchw[21] = (conv2d_nchw[21] + (pad_temp_shared[(((((((rc_outer_inner * 720) + (rc_inner * 180)) + (((((int)threadIdx.x) & 31) >> 3) * 20)) + (ry_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7)) + 90)] * kernel_shared[(((((((((int)threadIdx.x) >> 5) * 1152) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 288)]));
            conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[((((((rc_outer_inner * 720) + (rc_inner * 180)) + (((((int)threadIdx.x) & 31) >> 3) * 20)) + (ry_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7))] * kernel_shared[(((((((((int)threadIdx.x) >> 5) * 1152) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 432)]));
            conv2d_nchw[22] = (conv2d_nchw[22] + (pad_temp_shared[(((((((rc_outer_inner * 720) + (rc_inner * 180)) + (((((int)threadIdx.x) & 31) >> 3) * 20)) + (ry_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7)) + 80)] * kernel_shared[(((((((((int)threadIdx.x) >> 5) * 1152) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 432)]));
            conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[(((((((rc_outer_inner * 720) + (rc_inner * 180)) + (((((int)threadIdx.x) & 31) >> 3) * 20)) + (ry_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7)) + 10)] * kernel_shared[(((((((((int)threadIdx.x) >> 5) * 1152) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 432)]));
            conv2d_nchw[23] = (conv2d_nchw[23] + (pad_temp_shared[(((((((rc_outer_inner * 720) + (rc_inner * 180)) + (((((int)threadIdx.x) & 31) >> 3) * 20)) + (ry_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7)) + 90)] * kernel_shared[(((((((((int)threadIdx.x) >> 5) * 1152) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 432)]));
            conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[((((((rc_outer_inner * 720) + (rc_inner * 180)) + (((((int)threadIdx.x) & 31) >> 3) * 20)) + (ry_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7))] * kernel_shared[(((((((((int)threadIdx.x) >> 5) * 1152) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 576)]));
            conv2d_nchw[24] = (conv2d_nchw[24] + (pad_temp_shared[(((((((rc_outer_inner * 720) + (rc_inner * 180)) + (((((int)threadIdx.x) & 31) >> 3) * 20)) + (ry_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7)) + 80)] * kernel_shared[(((((((((int)threadIdx.x) >> 5) * 1152) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 576)]));
            conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[(((((((rc_outer_inner * 720) + (rc_inner * 180)) + (((((int)threadIdx.x) & 31) >> 3) * 20)) + (ry_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7)) + 10)] * kernel_shared[(((((((((int)threadIdx.x) >> 5) * 1152) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 576)]));
            conv2d_nchw[25] = (conv2d_nchw[25] + (pad_temp_shared[(((((((rc_outer_inner * 720) + (rc_inner * 180)) + (((((int)threadIdx.x) & 31) >> 3) * 20)) + (ry_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7)) + 90)] * kernel_shared[(((((((((int)threadIdx.x) >> 5) * 1152) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 576)]));
            conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[((((((rc_outer_inner * 720) + (rc_inner * 180)) + (((((int)threadIdx.x) & 31) >> 3) * 20)) + (ry_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7))] * kernel_shared[(((((((((int)threadIdx.x) >> 5) * 1152) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 720)]));
            conv2d_nchw[26] = (conv2d_nchw[26] + (pad_temp_shared[(((((((rc_outer_inner * 720) + (rc_inner * 180)) + (((((int)threadIdx.x) & 31) >> 3) * 20)) + (ry_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7)) + 80)] * kernel_shared[(((((((((int)threadIdx.x) >> 5) * 1152) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 720)]));
            conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[(((((((rc_outer_inner * 720) + (rc_inner * 180)) + (((((int)threadIdx.x) & 31) >> 3) * 20)) + (ry_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7)) + 10)] * kernel_shared[(((((((((int)threadIdx.x) >> 5) * 1152) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 720)]));
            conv2d_nchw[27] = (conv2d_nchw[27] + (pad_temp_shared[(((((((rc_outer_inner * 720) + (rc_inner * 180)) + (((((int)threadIdx.x) & 31) >> 3) * 20)) + (ry_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7)) + 90)] * kernel_shared[(((((((((int)threadIdx.x) >> 5) * 1152) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 720)]));
            conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[((((((rc_outer_inner * 720) + (rc_inner * 180)) + (((((int)threadIdx.x) & 31) >> 3) * 20)) + (ry_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7))] * kernel_shared[(((((((((int)threadIdx.x) >> 5) * 1152) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 864)]));
            conv2d_nchw[28] = (conv2d_nchw[28] + (pad_temp_shared[(((((((rc_outer_inner * 720) + (rc_inner * 180)) + (((((int)threadIdx.x) & 31) >> 3) * 20)) + (ry_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7)) + 80)] * kernel_shared[(((((((((int)threadIdx.x) >> 5) * 1152) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 864)]));
            conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[(((((((rc_outer_inner * 720) + (rc_inner * 180)) + (((((int)threadIdx.x) & 31) >> 3) * 20)) + (ry_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7)) + 10)] * kernel_shared[(((((((((int)threadIdx.x) >> 5) * 1152) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 864)]));
            conv2d_nchw[29] = (conv2d_nchw[29] + (pad_temp_shared[(((((((rc_outer_inner * 720) + (rc_inner * 180)) + (((((int)threadIdx.x) & 31) >> 3) * 20)) + (ry_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7)) + 90)] * kernel_shared[(((((((((int)threadIdx.x) >> 5) * 1152) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 864)]));
            conv2d_nchw[14] = (conv2d_nchw[14] + (pad_temp_shared[((((((rc_outer_inner * 720) + (rc_inner * 180)) + (((((int)threadIdx.x) & 31) >> 3) * 20)) + (ry_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7))] * kernel_shared[(((((((((int)threadIdx.x) >> 5) * 1152) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 1008)]));
            conv2d_nchw[30] = (conv2d_nchw[30] + (pad_temp_shared[(((((((rc_outer_inner * 720) + (rc_inner * 180)) + (((((int)threadIdx.x) & 31) >> 3) * 20)) + (ry_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7)) + 80)] * kernel_shared[(((((((((int)threadIdx.x) >> 5) * 1152) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 1008)]));
            conv2d_nchw[15] = (conv2d_nchw[15] + (pad_temp_shared[(((((((rc_outer_inner * 720) + (rc_inner * 180)) + (((((int)threadIdx.x) & 31) >> 3) * 20)) + (ry_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7)) + 10)] * kernel_shared[(((((((((int)threadIdx.x) >> 5) * 1152) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 1008)]));
            conv2d_nchw[31] = (conv2d_nchw[31] + (pad_temp_shared[(((((((rc_outer_inner * 720) + (rc_inner * 180)) + (((((int)threadIdx.x) & 31) >> 3) * 20)) + (ry_inner * 10)) + rx_inner) + (((int)threadIdx.x) & 7)) + 90)] * kernel_shared[(((((((((int)threadIdx.x) >> 5) * 1152) + (rc_outer_inner * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 1008)]));
          }
        }
      }
    }
  }
  for (int i1_inner = 0; i1_inner < 8; ++i1_inner) {
    for (int i2_inner = 0; i2_inner < 2; ++i2_inner) {
      compute[((((((((((int)threadIdx.x) >> 5) * 100352) + (i1_inner * 12544)) + ((((int)blockIdx.x) / 14) * 1792)) + (((((int)threadIdx.x) & 31) >> 3) * 224)) + (i2_inner * 112)) + ((((int)blockIdx.x) % 14) * 8)) + (((int)threadIdx.x) & 7))] = max((conv2d_nchw[((i1_inner * 2) + i2_inner)] + bias[(((((int)threadIdx.x) >> 5) * 8) + i1_inner)]), 0.000000e+00f);
      compute[(((((((((((int)threadIdx.x) >> 5) * 100352) + (i1_inner * 12544)) + ((((int)blockIdx.x) / 14) * 1792)) + (((((int)threadIdx.x) & 31) >> 3) * 224)) + (i2_inner * 112)) + ((((int)blockIdx.x) % 14) * 8)) + (((int)threadIdx.x) & 7)) + 896)] = max((conv2d_nchw[(((i1_inner * 2) + i2_inner) + 16)] + bias[(((((int)threadIdx.x) >> 5) * 8) + i1_inner)]), 0.000000e+00f);
    }
  }
}


