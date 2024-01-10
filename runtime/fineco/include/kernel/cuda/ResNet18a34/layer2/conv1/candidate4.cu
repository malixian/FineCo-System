
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
  float conv2d_nchw_local[16];
  __shared__ float pad_temp_shared[1600];
  __shared__ float kernel_shared[9216];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[8] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[9] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[10] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[11] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[12] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[13] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  conv2d_nchw_local[14] = 0.000000e+00f;
  conv2d_nchw_local[7] = 0.000000e+00f;
  conv2d_nchw_local[15] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 4; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = (((((1 <= (((((int)blockIdx.x) / 7) * 8) + ((((int)threadIdx.x) % 100) / 10))) && ((((((int)blockIdx.x) / 7) * 8) + ((((int)threadIdx.x) % 100) / 10)) < 57)) && (1 <= (((((int)blockIdx.x) % 7) * 8) + (((int)threadIdx.x) % 10)))) && ((((((int)blockIdx.x) % 7) * 8) + (((int)threadIdx.x) % 10)) < 57)) ? data[(((((((rc_outer_outer * 50176) + ((((int)threadIdx.x) / 100) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((int)threadIdx.x) % 100) / 10) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) % 10)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 256)] = (((((1 <= (((((int)blockIdx.x) / 7) * 8) + ((((((int)threadIdx.x) >> 1) + 28) % 50) / 5))) && ((((((int)blockIdx.x) / 7) * 8) + ((((((int)threadIdx.x) >> 1) + 28) % 50) / 5)) < 57)) && (1 <= (((((int)blockIdx.x) % 7) * 8) + ((((int)threadIdx.x) + 6) % 10)))) && ((((((int)blockIdx.x) % 7) * 8) + ((((int)threadIdx.x) + 6) % 10)) < 57)) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 256) / 100) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((((int)threadIdx.x) >> 1) + 28) % 50) / 5) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + ((((int)threadIdx.x) + 6) % 10)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 512)] = (((((1 <= (((((int)blockIdx.x) / 7) * 8) + ((((((int)threadIdx.x) >> 1) + 6) % 50) / 5))) && ((((((int)blockIdx.x) / 7) * 8) + ((((((int)threadIdx.x) >> 1) + 6) % 50) / 5)) < 57)) && (1 <= (((((int)blockIdx.x) % 7) * 8) + ((((int)threadIdx.x) + 2) % 10)))) && ((((((int)blockIdx.x) % 7) * 8) + ((((int)threadIdx.x) + 2) % 10)) < 57)) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 512) / 100) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((((int)threadIdx.x) >> 1) + 6) % 50) / 5) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + ((((int)threadIdx.x) + 2) % 10)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 768)] = (((((1 <= (((((int)blockIdx.x) / 7) * 8) + ((((((int)threadIdx.x) >> 1) + 34) % 50) / 5))) && ((((((int)blockIdx.x) / 7) * 8) + ((((((int)threadIdx.x) >> 1) + 34) % 50) / 5)) < 57)) && (1 <= (((((int)blockIdx.x) % 7) * 8) + ((((int)threadIdx.x) + 8) % 10)))) && ((((((int)blockIdx.x) % 7) * 8) + ((((int)threadIdx.x) + 8) % 10)) < 57)) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 768) / 100) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((((int)threadIdx.x) >> 1) + 34) % 50) / 5) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + ((((int)threadIdx.x) + 8) % 10)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1024)] = (((((1 <= (((((int)blockIdx.x) / 7) * 8) + ((((((int)threadIdx.x) >> 1) + 12) % 50) / 5))) && ((((((int)blockIdx.x) / 7) * 8) + ((((((int)threadIdx.x) >> 1) + 12) % 50) / 5)) < 57)) && (1 <= (((((int)blockIdx.x) % 7) * 8) + ((((int)threadIdx.x) + 4) % 10)))) && ((((((int)blockIdx.x) % 7) * 8) + ((((int)threadIdx.x) + 4) % 10)) < 57)) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 1024) / 100) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((((int)threadIdx.x) >> 1) + 12) % 50) / 5) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + ((((int)threadIdx.x) + 4) % 10)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1280)] = (((((1 <= (((((int)blockIdx.x) / 7) * 8) + (((((int)threadIdx.x) / 10) + 8) % 10))) && ((((((int)blockIdx.x) / 7) * 8) + (((((int)threadIdx.x) / 10) + 8) % 10)) < 57)) && (1 <= (((((int)blockIdx.x) % 7) * 8) + (((int)threadIdx.x) % 10)))) && ((((((int)blockIdx.x) % 7) * 8) + (((int)threadIdx.x) % 10)) < 57)) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 1280) / 100) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + ((((((int)threadIdx.x) / 10) + 8) % 10) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) % 10)) - 57)] : 0.000000e+00f);
    if (((int)threadIdx.x) < 64) {
      pad_temp_shared[(((int)threadIdx.x) + 1536)] = (((((((((int)blockIdx.x) / 7) * 8) + ((((((int)threadIdx.x) >> 1) + 18) % 50) / 5)) < 57) && (1 <= (((((int)blockIdx.x) % 7) * 8) + ((((int)threadIdx.x) + 6) % 10)))) && ((((((int)blockIdx.x) % 7) * 8) + ((((int)threadIdx.x) + 6) % 10)) < 57)) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 1536) / 100) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((((int)threadIdx.x) >> 1) + 18) % 50) / 5) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + ((((int)threadIdx.x) + 6) % 10)) - 57)] : 0.000000e+00f);
    }
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)threadIdx.x) / 144) * 576) + (rc_outer_outer * 144)) + (((int)threadIdx.x) % 144))];
    kernel_shared[(((int)threadIdx.x) + 256)] = kernel[(((((((int)threadIdx.x) + 256) / 144) * 576) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 112) % 144))];
    kernel_shared[(((int)threadIdx.x) + 512)] = kernel[(((((((int)threadIdx.x) + 512) / 144) * 576) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 80) % 144))];
    kernel_shared[(((int)threadIdx.x) + 768)] = kernel[(((((((int)threadIdx.x) + 768) / 144) * 576) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 48) % 144))];
    kernel_shared[(((int)threadIdx.x) + 1024)] = kernel[(((((((int)threadIdx.x) + 1024) / 144) * 576) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 16) % 144))];
    kernel_shared[(((int)threadIdx.x) + 1280)] = kernel[(((((((int)threadIdx.x) + 1280) / 144) * 576) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 128) % 144))];
    kernel_shared[(((int)threadIdx.x) + 1536)] = kernel[(((((((int)threadIdx.x) + 1536) / 144) * 576) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 96) % 144))];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[(((((((int)threadIdx.x) + 1792) / 144) * 576) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 64) % 144))];
    kernel_shared[(((int)threadIdx.x) + 2048)] = kernel[(((((((int)threadIdx.x) + 2048) / 144) * 576) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 32) % 144))];
    kernel_shared[(((int)threadIdx.x) + 2304)] = kernel[(((((((int)threadIdx.x) / 144) * 576) + (rc_outer_outer * 144)) + (((int)threadIdx.x) % 144)) + 9216)];
    kernel_shared[(((int)threadIdx.x) + 2560)] = kernel[(((((((int)threadIdx.x) + 2560) / 144) * 576) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 112) % 144))];
    kernel_shared[(((int)threadIdx.x) + 2816)] = kernel[(((((((int)threadIdx.x) + 2816) / 144) * 576) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 80) % 144))];
    kernel_shared[(((int)threadIdx.x) + 3072)] = kernel[(((((((int)threadIdx.x) + 3072) / 144) * 576) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 48) % 144))];
    kernel_shared[(((int)threadIdx.x) + 3328)] = kernel[(((((((int)threadIdx.x) + 3328) / 144) * 576) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 16) % 144))];
    kernel_shared[(((int)threadIdx.x) + 3584)] = kernel[(((((((int)threadIdx.x) + 3584) / 144) * 576) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 128) % 144))];
    kernel_shared[(((int)threadIdx.x) + 3840)] = kernel[(((((((int)threadIdx.x) + 3840) / 144) * 576) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 96) % 144))];
    kernel_shared[(((int)threadIdx.x) + 4096)] = kernel[(((((((int)threadIdx.x) + 4096) / 144) * 576) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 64) % 144))];
    kernel_shared[(((int)threadIdx.x) + 4352)] = kernel[(((((((int)threadIdx.x) + 4352) / 144) * 576) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 32) % 144))];
    kernel_shared[(((int)threadIdx.x) + 4608)] = kernel[(((((((int)threadIdx.x) / 144) * 576) + (rc_outer_outer * 144)) + (((int)threadIdx.x) % 144)) + 18432)];
    kernel_shared[(((int)threadIdx.x) + 4864)] = kernel[(((((((int)threadIdx.x) + 4864) / 144) * 576) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 112) % 144))];
    kernel_shared[(((int)threadIdx.x) + 5120)] = kernel[(((((((int)threadIdx.x) + 5120) / 144) * 576) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 80) % 144))];
    kernel_shared[(((int)threadIdx.x) + 5376)] = kernel[(((((((int)threadIdx.x) + 5376) / 144) * 576) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 48) % 144))];
    kernel_shared[(((int)threadIdx.x) + 5632)] = kernel[(((((((int)threadIdx.x) + 5632) / 144) * 576) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 16) % 144))];
    kernel_shared[(((int)threadIdx.x) + 5888)] = kernel[(((((((int)threadIdx.x) + 5888) / 144) * 576) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 128) % 144))];
    kernel_shared[(((int)threadIdx.x) + 6144)] = kernel[(((((((int)threadIdx.x) + 6144) / 144) * 576) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 96) % 144))];
    kernel_shared[(((int)threadIdx.x) + 6400)] = kernel[(((((((int)threadIdx.x) + 6400) / 144) * 576) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 64) % 144))];
    kernel_shared[(((int)threadIdx.x) + 6656)] = kernel[(((((((int)threadIdx.x) + 6656) / 144) * 576) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 32) % 144))];
    kernel_shared[(((int)threadIdx.x) + 6912)] = kernel[(((((((int)threadIdx.x) / 144) * 576) + (rc_outer_outer * 144)) + (((int)threadIdx.x) % 144)) + 27648)];
    kernel_shared[(((int)threadIdx.x) + 7168)] = kernel[(((((((int)threadIdx.x) + 7168) / 144) * 576) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 112) % 144))];
    kernel_shared[(((int)threadIdx.x) + 7424)] = kernel[(((((((int)threadIdx.x) + 7424) / 144) * 576) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 80) % 144))];
    kernel_shared[(((int)threadIdx.x) + 7680)] = kernel[(((((((int)threadIdx.x) + 7680) / 144) * 576) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 48) % 144))];
    kernel_shared[(((int)threadIdx.x) + 7936)] = kernel[(((((((int)threadIdx.x) + 7936) / 144) * 576) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 16) % 144))];
    kernel_shared[(((int)threadIdx.x) + 8192)] = kernel[(((((((int)threadIdx.x) + 8192) / 144) * 576) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 128) % 144))];
    kernel_shared[(((int)threadIdx.x) + 8448)] = kernel[(((((((int)threadIdx.x) + 8448) / 144) * 576) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 96) % 144))];
    kernel_shared[(((int)threadIdx.x) + 8704)] = kernel[(((((((int)threadIdx.x) + 8704) / 144) * 576) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 64) % 144))];
    kernel_shared[(((int)threadIdx.x) + 8960)] = kernel[(((((((int)threadIdx.x) + 8960) / 144) * 576) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 32) % 144))];
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 16; ++rc_outer_inner) {
      for (int rx_outer_inner = 0; rx_outer_inner < 3; ++rx_outer_inner) {
        conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner)]));
        conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 4608)]));
        conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 1)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner)]));
        conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 1)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 4608)]));
        conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 10)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner)]));
        conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 10)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 4608)]));
        conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 11)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner)]));
        conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 11)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 4608)]));
        conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 20)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner)]));
        conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 20)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 4608)]));
        conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 21)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner)]));
        conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 21)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 4608)]));
        conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 30)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner)]));
        conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 30)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 4608)]));
        conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 31)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner)]));
        conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 31)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 4608)]));
        conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 10)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 3)]));
        conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 10)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 4611)]));
        conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 11)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 3)]));
        conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 11)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 4611)]));
        conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 20)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 3)]));
        conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 20)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 4611)]));
        conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 21)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 3)]));
        conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 21)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 4611)]));
        conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 30)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 3)]));
        conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 30)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 4611)]));
        conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 31)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 3)]));
        conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 31)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 4611)]));
        conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 40)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 3)]));
        conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 40)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 4611)]));
        conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 41)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 3)]));
        conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 41)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 4611)]));
        conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 20)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 6)]));
        conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 20)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 4614)]));
        conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 21)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 6)]));
        conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 21)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 4614)]));
        conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 30)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 6)]));
        conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 30)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 4614)]));
        conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 31)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 6)]));
        conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 31)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 4614)]));
        conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 40)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 6)]));
        conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 40)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 4614)]));
        conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 41)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 6)]));
        conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 41)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 4614)]));
        conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 50)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 6)]));
        conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 50)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 4614)]));
        conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 51)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 6)]));
        conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[(((((rc_outer_inner * 100) + (((((int)threadIdx.x) & 7) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 2)) + rx_outer_inner) + 51)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 144) + (rc_outer_inner * 9)) + rx_outer_inner) + 4614)]));
      }
    }
  }
  for (int yy_inner = 0; yy_inner < 4; ++yy_inner) {
    for (int xx_inner = 0; xx_inner < 2; ++xx_inner) {
      conv2d_nchw[((((((((((int)threadIdx.x) >> 3) * 3136) + ((((int)blockIdx.x) / 7) * 448)) + (((((int)threadIdx.x) & 7) >> 2) * 224)) + (yy_inner * 56)) + ((((int)blockIdx.x) % 7) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + xx_inner)] = conv2d_nchw_local[((yy_inner * 2) + xx_inner)];
      conv2d_nchw[(((((((((((int)threadIdx.x) >> 3) * 3136) + ((((int)blockIdx.x) / 7) * 448)) + (((((int)threadIdx.x) & 7) >> 2) * 224)) + (yy_inner * 56)) + ((((int)blockIdx.x) % 7) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + xx_inner) + 100352)] = conv2d_nchw_local[(((yy_inner * 2) + xx_inner) + 8)];
    }
  }
}


