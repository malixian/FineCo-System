
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
extern "C" __global__ void __launch_bounds__(256) candidate6(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[98];
  __shared__ float pad_temp_shared[6720];
  __shared__ float kernel_shared[768];
  for (int yy_c_outer_inner_init = 0; yy_c_outer_inner_init < 7; ++yy_c_outer_inner_init) {
    conv2d_nchw_local[(yy_c_outer_inner_init * 7)] = 0.000000e+00f;
    conv2d_nchw_local[((yy_c_outer_inner_init * 7) + 49)] = 0.000000e+00f;
    conv2d_nchw_local[((yy_c_outer_inner_init * 7) + 1)] = 0.000000e+00f;
    conv2d_nchw_local[((yy_c_outer_inner_init * 7) + 50)] = 0.000000e+00f;
    conv2d_nchw_local[((yy_c_outer_inner_init * 7) + 2)] = 0.000000e+00f;
    conv2d_nchw_local[((yy_c_outer_inner_init * 7) + 51)] = 0.000000e+00f;
    conv2d_nchw_local[((yy_c_outer_inner_init * 7) + 3)] = 0.000000e+00f;
    conv2d_nchw_local[((yy_c_outer_inner_init * 7) + 52)] = 0.000000e+00f;
    conv2d_nchw_local[((yy_c_outer_inner_init * 7) + 4)] = 0.000000e+00f;
    conv2d_nchw_local[((yy_c_outer_inner_init * 7) + 53)] = 0.000000e+00f;
    conv2d_nchw_local[((yy_c_outer_inner_init * 7) + 5)] = 0.000000e+00f;
    conv2d_nchw_local[((yy_c_outer_inner_init * 7) + 54)] = 0.000000e+00f;
    conv2d_nchw_local[((yy_c_outer_inner_init * 7) + 6)] = 0.000000e+00f;
    conv2d_nchw_local[((yy_c_outer_inner_init * 7) + 55)] = 0.000000e+00f;
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 16; ++rc_outer_outer) {
    for (int rx_outer_outer = 0; rx_outer_outer < 3; ++rx_outer_outer) {
      __syncthreads();
      pad_temp_shared[((int)threadIdx.x)] = ((((1 <= ((((((int)blockIdx.x) & 15) >> 2) * 28) + (((int)threadIdx.x) / 28))) && (1 <= ((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + (((int)threadIdx.x) % 28)))) && (((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + (((int)threadIdx.x) % 28)) < 113)) ? data[(((((((rc_outer_outer * 100352) + (((((int)blockIdx.x) & 15) >> 2) * 3136)) + ((((int)threadIdx.x) / 28) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + rx_outer_outer) + (((int)threadIdx.x) % 28)) - 113)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 256)] = (((1 <= ((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 4) % 28))) && (((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 4) % 28)) < 113)) ? data[(((((((rc_outer_outer * 100352) + (((((int)blockIdx.x) & 15) >> 2) * 3136)) + (((((int)threadIdx.x) + 256) / 28) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + rx_outer_outer) + ((((int)threadIdx.x) + 4) % 28)) - 113)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 512)] = (((1 <= ((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 8) % 28))) && (((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 8) % 28)) < 113)) ? data[(((((((rc_outer_outer * 100352) + (((((int)blockIdx.x) & 15) >> 2) * 3136)) + (((((int)threadIdx.x) + 512) / 28) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + rx_outer_outer) + ((((int)threadIdx.x) + 8) % 28)) - 113)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 768)] = (((((1 <= ((((((int)blockIdx.x) & 15) >> 2) * 28) + ((((((int)threadIdx.x) >> 2) + 192) % 210) / 7))) && (((((((int)blockIdx.x) & 15) >> 2) * 28) + ((((((int)threadIdx.x) >> 2) + 192) % 210) / 7)) < 113)) && (1 <= ((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 12) % 28)))) && (((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 12) % 28)) < 113)) ? data[((((((((rc_outer_outer * 100352) + (((((int)threadIdx.x) + 768) / 840) * 12544)) + (((((int)blockIdx.x) & 15) >> 2) * 3136)) + (((((((int)threadIdx.x) >> 2) + 192) % 210) / 7) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + rx_outer_outer) + ((((int)threadIdx.x) + 12) % 28)) - 113)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 1024)] = (((1 <= ((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 16) % 28))) && (((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 16) % 28)) < 113)) ? data[((((((((rc_outer_outer * 100352) + (((((int)threadIdx.x) + 1024) / 840) * 12544)) + (((((int)blockIdx.x) & 15) >> 2) * 3136)) + (((((((int)threadIdx.x) >> 2) + 46) % 210) / 7) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + rx_outer_outer) + ((((int)threadIdx.x) + 16) % 28)) - 113)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 1280)] = (((1 <= ((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 20) % 28))) && (((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 20) % 28)) < 113)) ? data[((((((((rc_outer_outer * 100352) + (((((int)threadIdx.x) + 1280) / 840) * 12544)) + (((((int)blockIdx.x) & 15) >> 2) * 3136)) + (((((((int)threadIdx.x) >> 2) + 110) % 210) / 7) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + rx_outer_outer) + ((((int)threadIdx.x) + 20) % 28)) - 113)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 1536)] = (((((1 <= ((((((int)blockIdx.x) & 15) >> 2) * 28) + ((((((int)threadIdx.x) >> 2) + 174) % 210) / 7))) && (((((((int)blockIdx.x) & 15) >> 2) * 28) + ((((((int)threadIdx.x) >> 2) + 174) % 210) / 7)) < 113)) && (1 <= ((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 24) % 28)))) && (((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 24) % 28)) < 113)) ? data[((((((((rc_outer_outer * 100352) + (((((int)threadIdx.x) + 1536) / 840) * 12544)) + (((((int)blockIdx.x) & 15) >> 2) * 3136)) + (((((((int)threadIdx.x) >> 2) + 174) % 210) / 7) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + rx_outer_outer) + ((((int)threadIdx.x) + 24) % 28)) - 113)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 1792)] = (((1 <= ((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + (((int)threadIdx.x) % 28))) && (((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + (((int)threadIdx.x) % 28)) < 113)) ? data[((((((((rc_outer_outer * 100352) + (((((int)threadIdx.x) + 1792) / 840) * 12544)) + (((((int)blockIdx.x) & 15) >> 2) * 3136)) + (((((int)threadIdx.x) / 28) + 4) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + rx_outer_outer) + (((int)threadIdx.x) % 28)) - 113)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 2048)] = (((1 <= ((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 4) % 28))) && (((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 4) % 28)) < 113)) ? data[((((((((rc_outer_outer * 100352) + (((((int)threadIdx.x) + 2048) / 840) * 12544)) + (((((int)blockIdx.x) & 15) >> 2) * 3136)) + (((((((int)threadIdx.x) >> 2) + 92) % 210) / 7) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + rx_outer_outer) + ((((int)threadIdx.x) + 4) % 28)) - 113)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 2304)] = (((((1 <= ((((((int)blockIdx.x) & 15) >> 2) * 28) + ((((((int)threadIdx.x) >> 2) + 156) % 210) / 7))) && (((((((int)blockIdx.x) & 15) >> 2) * 28) + ((((((int)threadIdx.x) >> 2) + 156) % 210) / 7)) < 113)) && (1 <= ((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 8) % 28)))) && (((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 8) % 28)) < 113)) ? data[((((((((rc_outer_outer * 100352) + (((((int)threadIdx.x) + 2304) / 840) * 12544)) + (((((int)blockIdx.x) & 15) >> 2) * 3136)) + (((((((int)threadIdx.x) >> 2) + 156) % 210) / 7) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + rx_outer_outer) + ((((int)threadIdx.x) + 8) % 28)) - 113)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 2560)] = (((1 <= ((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 12) % 28))) && (((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 12) % 28)) < 113)) ? data[((((((((rc_outer_outer * 100352) + (((((int)threadIdx.x) + 2560) / 840) * 12544)) + (((((int)blockIdx.x) & 15) >> 2) * 3136)) + (((((((int)threadIdx.x) >> 2) + 10) % 210) / 7) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + rx_outer_outer) + ((((int)threadIdx.x) + 12) % 28)) - 113)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 2816)] = (((1 <= ((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 16) % 28))) && (((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 16) % 28)) < 113)) ? data[((((((((rc_outer_outer * 100352) + (((((int)threadIdx.x) + 2816) / 840) * 12544)) + (((((int)blockIdx.x) & 15) >> 2) * 3136)) + (((((((int)threadIdx.x) >> 2) + 74) % 210) / 7) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + rx_outer_outer) + ((((int)threadIdx.x) + 16) % 28)) - 113)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 3072)] = (((1 <= ((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 20) % 28))) && (((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 20) % 28)) < 113)) ? data[((((((((rc_outer_outer * 100352) + (((((int)threadIdx.x) + 3072) / 840) * 12544)) + (((((int)blockIdx.x) & 15) >> 2) * 3136)) + (((((((int)threadIdx.x) >> 2) + 138) % 210) / 7) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + rx_outer_outer) + ((((int)threadIdx.x) + 20) % 28)) - 113)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 3328)] = (((((1 <= ((((((int)blockIdx.x) & 15) >> 2) * 28) + ((((((int)threadIdx.x) >> 2) + 202) % 210) / 7))) && (((((((int)blockIdx.x) & 15) >> 2) * 28) + ((((((int)threadIdx.x) >> 2) + 202) % 210) / 7)) < 113)) && (1 <= ((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 24) % 28)))) && (((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 24) % 28)) < 113)) ? data[((((((((rc_outer_outer * 100352) + (((((int)threadIdx.x) + 3328) / 840) * 12544)) + (((((int)blockIdx.x) & 15) >> 2) * 3136)) + (((((((int)threadIdx.x) >> 2) + 202) % 210) / 7) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + rx_outer_outer) + ((((int)threadIdx.x) + 24) % 28)) - 113)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 3584)] = (((1 <= ((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + (((int)threadIdx.x) % 28))) && (((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + (((int)threadIdx.x) % 28)) < 113)) ? data[((((((((rc_outer_outer * 100352) + (((((int)threadIdx.x) + 3584) / 840) * 12544)) + (((((int)blockIdx.x) & 15) >> 2) * 3136)) + (((((int)threadIdx.x) / 28) + 8) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + rx_outer_outer) + (((int)threadIdx.x) % 28)) - 113)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 3840)] = (((1 <= ((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 4) % 28))) && (((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 4) % 28)) < 113)) ? data[((((((((rc_outer_outer * 100352) + (((((int)threadIdx.x) + 3840) / 840) * 12544)) + (((((int)blockIdx.x) & 15) >> 2) * 3136)) + (((((((int)threadIdx.x) >> 2) + 120) % 210) / 7) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + rx_outer_outer) + ((((int)threadIdx.x) + 4) % 28)) - 113)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 4096)] = (((((1 <= ((((((int)blockIdx.x) & 15) >> 2) * 28) + ((((((int)threadIdx.x) >> 2) + 184) % 210) / 7))) && (((((((int)blockIdx.x) & 15) >> 2) * 28) + ((((((int)threadIdx.x) >> 2) + 184) % 210) / 7)) < 113)) && (1 <= ((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 8) % 28)))) && (((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 8) % 28)) < 113)) ? data[((((((((rc_outer_outer * 100352) + (((((int)threadIdx.x) + 4096) / 840) * 12544)) + (((((int)blockIdx.x) & 15) >> 2) * 3136)) + (((((((int)threadIdx.x) >> 2) + 184) % 210) / 7) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + rx_outer_outer) + ((((int)threadIdx.x) + 8) % 28)) - 113)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 4352)] = (((1 <= ((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 12) % 28))) && (((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 12) % 28)) < 113)) ? data[((((((((rc_outer_outer * 100352) + (((((int)threadIdx.x) + 4352) / 840) * 12544)) + (((((int)blockIdx.x) & 15) >> 2) * 3136)) + (((((((int)threadIdx.x) >> 2) + 38) % 210) / 7) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + rx_outer_outer) + ((((int)threadIdx.x) + 12) % 28)) - 113)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 4608)] = (((1 <= ((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 16) % 28))) && (((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 16) % 28)) < 113)) ? data[((((((((rc_outer_outer * 100352) + (((((int)threadIdx.x) + 4608) / 840) * 12544)) + (((((int)blockIdx.x) & 15) >> 2) * 3136)) + (((((((int)threadIdx.x) >> 2) + 102) % 210) / 7) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + rx_outer_outer) + ((((int)threadIdx.x) + 16) % 28)) - 113)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 4864)] = (((((1 <= ((((((int)blockIdx.x) & 15) >> 2) * 28) + ((((((int)threadIdx.x) >> 2) + 166) % 210) / 7))) && (((((((int)blockIdx.x) & 15) >> 2) * 28) + ((((((int)threadIdx.x) >> 2) + 166) % 210) / 7)) < 113)) && (1 <= ((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 20) % 28)))) && (((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 20) % 28)) < 113)) ? data[((((((((rc_outer_outer * 100352) + (((((int)threadIdx.x) + 4864) / 840) * 12544)) + (((((int)blockIdx.x) & 15) >> 2) * 3136)) + (((((((int)threadIdx.x) >> 2) + 166) % 210) / 7) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + rx_outer_outer) + ((((int)threadIdx.x) + 20) % 28)) - 113)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 5120)] = (((1 <= ((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 24) % 28))) && (((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 24) % 28)) < 113)) ? data[((((((((rc_outer_outer * 100352) + (((((int)threadIdx.x) + 5120) / 840) * 12544)) + (((((int)blockIdx.x) & 15) >> 2) * 3136)) + (((((((int)threadIdx.x) >> 2) + 20) % 210) / 7) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + rx_outer_outer) + ((((int)threadIdx.x) + 24) % 28)) - 113)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 5376)] = (((1 <= ((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + (((int)threadIdx.x) % 28))) && (((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + (((int)threadIdx.x) % 28)) < 113)) ? data[((((((((rc_outer_outer * 100352) + (((((int)threadIdx.x) + 5376) / 840) * 12544)) + (((((int)blockIdx.x) & 15) >> 2) * 3136)) + (((((int)threadIdx.x) / 28) + 12) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + rx_outer_outer) + (((int)threadIdx.x) % 28)) - 113)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 5632)] = (((((1 <= ((((((int)blockIdx.x) & 15) >> 2) * 28) + ((((((int)threadIdx.x) >> 2) + 148) % 210) / 7))) && (((((((int)blockIdx.x) & 15) >> 2) * 28) + ((((((int)threadIdx.x) >> 2) + 148) % 210) / 7)) < 113)) && (1 <= ((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 4) % 28)))) && (((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 4) % 28)) < 113)) ? data[((((((((rc_outer_outer * 100352) + (((((int)threadIdx.x) + 5632) / 840) * 12544)) + (((((int)blockIdx.x) & 15) >> 2) * 3136)) + (((((((int)threadIdx.x) >> 2) + 148) % 210) / 7) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + rx_outer_outer) + ((((int)threadIdx.x) + 4) % 28)) - 113)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 5888)] = ((((1 <= ((((((int)blockIdx.x) & 15) >> 2) * 28) + ((((((int)threadIdx.x) >> 2) + 2) % 210) / 7))) && (1 <= ((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 8) % 28)))) && (((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 8) % 28)) < 113)) ? data[((((((((rc_outer_outer * 100352) + (((((int)threadIdx.x) + 5888) / 840) * 12544)) + (((((int)blockIdx.x) & 15) >> 2) * 3136)) + (((((((int)threadIdx.x) >> 2) + 2) % 210) / 7) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + rx_outer_outer) + ((((int)threadIdx.x) + 8) % 28)) - 113)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 6144)] = (((1 <= ((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 12) % 28))) && (((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 12) % 28)) < 113)) ? data[((((((((rc_outer_outer * 100352) + (((((int)threadIdx.x) + 6144) / 840) * 12544)) + (((((int)blockIdx.x) & 15) >> 2) * 3136)) + (((((((int)threadIdx.x) >> 2) + 66) % 210) / 7) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + rx_outer_outer) + ((((int)threadIdx.x) + 12) % 28)) - 113)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 6400)] = (((1 <= ((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 16) % 28))) && (((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 16) % 28)) < 113)) ? data[((((((((rc_outer_outer * 100352) + (((((int)threadIdx.x) + 6400) / 840) * 12544)) + (((((int)blockIdx.x) & 15) >> 2) * 3136)) + (((((((int)threadIdx.x) >> 2) + 130) % 210) / 7) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + rx_outer_outer) + ((((int)threadIdx.x) + 16) % 28)) - 113)] : 0.000000e+00f);
      if (((int)threadIdx.x) < 64) {
        pad_temp_shared[(((int)threadIdx.x) + 6656)] = ((((((((((int)blockIdx.x) & 15) >> 2) * 28) + ((((((int)threadIdx.x) >> 2) + 194) % 210) / 7)) < 113) && (1 <= ((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 20) % 28)))) && (((((((int)blockIdx.x) & 3) * 28) + rx_outer_outer) + ((((int)threadIdx.x) + 20) % 28)) < 113)) ? data[((((((((rc_outer_outer * 100352) + (((((int)threadIdx.x) + 6656) / 840) * 12544)) + (((((int)blockIdx.x) & 15) >> 2) * 3136)) + (((((((int)threadIdx.x) >> 2) + 194) % 210) / 7) * 112)) + ((((int)blockIdx.x) & 3) * 28)) + rx_outer_outer) + ((((int)threadIdx.x) + 20) % 28)) - 113)] : 0.000000e+00f);
      }
      kernel_shared[((int)threadIdx.x)] = kernel[((((((((int)blockIdx.x) >> 4) * 36864) + ((((int)threadIdx.x) / 24) * 1152)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) % 24) * 3)) + rx_outer_outer)];
      kernel_shared[(((int)threadIdx.x) + 256)] = kernel[((((((((int)blockIdx.x) >> 4) * 36864) + (((((int)threadIdx.x) + 256) / 24) * 1152)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) + 16) % 24) * 3)) + rx_outer_outer)];
      kernel_shared[(((int)threadIdx.x) + 512)] = kernel[((((((((int)blockIdx.x) >> 4) * 36864) + (((((int)threadIdx.x) + 512) / 24) * 1152)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) + 8) % 24) * 3)) + rx_outer_outer)];
      __syncthreads();
      for (int rc_outer_inner = 0; rc_outer_inner < 8; ++rc_outer_inner) {
        for (int yy_c_outer_inner = 0; yy_c_outer_inner < 7; ++yy_c_outer_inner) {
          conv2d_nchw_local[(yy_c_outer_inner * 7)] = (conv2d_nchw_local[(yy_c_outer_inner * 7)] + (pad_temp_shared[((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7))] * kernel_shared[(((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3))]));
          conv2d_nchw_local[((yy_c_outer_inner * 7) + 49)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + 49)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 14)] * kernel_shared[(((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3))]));
          conv2d_nchw_local[(yy_c_outer_inner * 7)] = (conv2d_nchw_local[(yy_c_outer_inner * 7)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 28)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3)) + 1)]));
          conv2d_nchw_local[((yy_c_outer_inner * 7) + 49)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + 49)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 42)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3)) + 1)]));
          conv2d_nchw_local[(yy_c_outer_inner * 7)] = (conv2d_nchw_local[(yy_c_outer_inner * 7)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 56)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3)) + 2)]));
          conv2d_nchw_local[((yy_c_outer_inner * 7) + 49)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + 49)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 70)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3)) + 2)]));
          conv2d_nchw_local[((yy_c_outer_inner * 7) + 1)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + 1)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 1)] * kernel_shared[(((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3))]));
          conv2d_nchw_local[((yy_c_outer_inner * 7) + 50)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + 50)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 15)] * kernel_shared[(((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3))]));
          conv2d_nchw_local[((yy_c_outer_inner * 7) + 1)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + 1)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 29)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3)) + 1)]));
          conv2d_nchw_local[((yy_c_outer_inner * 7) + 50)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + 50)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 43)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3)) + 1)]));
          conv2d_nchw_local[((yy_c_outer_inner * 7) + 1)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + 1)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 57)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3)) + 2)]));
          conv2d_nchw_local[((yy_c_outer_inner * 7) + 50)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + 50)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 71)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3)) + 2)]));
          conv2d_nchw_local[((yy_c_outer_inner * 7) + 2)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + 2)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 2)] * kernel_shared[(((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3))]));
          conv2d_nchw_local[((yy_c_outer_inner * 7) + 51)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + 51)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 16)] * kernel_shared[(((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3))]));
          conv2d_nchw_local[((yy_c_outer_inner * 7) + 2)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + 2)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 30)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3)) + 1)]));
          conv2d_nchw_local[((yy_c_outer_inner * 7) + 51)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + 51)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 44)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3)) + 1)]));
          conv2d_nchw_local[((yy_c_outer_inner * 7) + 2)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + 2)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 58)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3)) + 2)]));
          conv2d_nchw_local[((yy_c_outer_inner * 7) + 51)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + 51)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 72)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3)) + 2)]));
          conv2d_nchw_local[((yy_c_outer_inner * 7) + 3)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + 3)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 3)] * kernel_shared[(((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3))]));
          conv2d_nchw_local[((yy_c_outer_inner * 7) + 52)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + 52)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 17)] * kernel_shared[(((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3))]));
          conv2d_nchw_local[((yy_c_outer_inner * 7) + 3)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + 3)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 31)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3)) + 1)]));
          conv2d_nchw_local[((yy_c_outer_inner * 7) + 52)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + 52)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 45)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3)) + 1)]));
          conv2d_nchw_local[((yy_c_outer_inner * 7) + 3)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + 3)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 59)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3)) + 2)]));
          conv2d_nchw_local[((yy_c_outer_inner * 7) + 52)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + 52)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 73)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3)) + 2)]));
          conv2d_nchw_local[((yy_c_outer_inner * 7) + 4)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + 4)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 4)] * kernel_shared[(((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3))]));
          conv2d_nchw_local[((yy_c_outer_inner * 7) + 53)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + 53)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 18)] * kernel_shared[(((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3))]));
          conv2d_nchw_local[((yy_c_outer_inner * 7) + 4)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + 4)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 32)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3)) + 1)]));
          conv2d_nchw_local[((yy_c_outer_inner * 7) + 53)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + 53)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 46)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3)) + 1)]));
          conv2d_nchw_local[((yy_c_outer_inner * 7) + 4)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + 4)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 60)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3)) + 2)]));
          conv2d_nchw_local[((yy_c_outer_inner * 7) + 53)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + 53)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 74)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3)) + 2)]));
          conv2d_nchw_local[((yy_c_outer_inner * 7) + 5)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + 5)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 5)] * kernel_shared[(((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3))]));
          conv2d_nchw_local[((yy_c_outer_inner * 7) + 54)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + 54)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 19)] * kernel_shared[(((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3))]));
          conv2d_nchw_local[((yy_c_outer_inner * 7) + 5)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + 5)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 33)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3)) + 1)]));
          conv2d_nchw_local[((yy_c_outer_inner * 7) + 54)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + 54)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 47)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3)) + 1)]));
          conv2d_nchw_local[((yy_c_outer_inner * 7) + 5)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + 5)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 61)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3)) + 2)]));
          conv2d_nchw_local[((yy_c_outer_inner * 7) + 54)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + 54)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 75)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3)) + 2)]));
          conv2d_nchw_local[((yy_c_outer_inner * 7) + 6)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + 6)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 6)] * kernel_shared[(((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3))]));
          conv2d_nchw_local[((yy_c_outer_inner * 7) + 55)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + 55)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 20)] * kernel_shared[(((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3))]));
          conv2d_nchw_local[((yy_c_outer_inner * 7) + 6)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + 6)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 34)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3)) + 1)]));
          conv2d_nchw_local[((yy_c_outer_inner * 7) + 55)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + 55)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 48)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3)) + 1)]));
          conv2d_nchw_local[((yy_c_outer_inner * 7) + 6)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + 6)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 62)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3)) + 2)]));
          conv2d_nchw_local[((yy_c_outer_inner * 7) + 55)] = (conv2d_nchw_local[((yy_c_outer_inner * 7) + 55)] + (pad_temp_shared[(((((rc_outer_inner * 840) + (((((int)threadIdx.x) & 7) >> 1) * 196)) + (yy_c_outer_inner * 28)) + ((((int)threadIdx.x) & 1) * 7)) + 76)] * kernel_shared[((((((int)threadIdx.x) >> 3) * 24) + (rc_outer_inner * 3)) + 2)]));
        }
      }
    }
  }
  for (int yy_inner = 0; yy_inner < 7; ++yy_inner) {
    for (int xx_inner = 0; xx_inner < 7; ++xx_inner) {
      conv2d_nchw[(((((((((((int)blockIdx.x) >> 4) * 401408) + ((((int)threadIdx.x) >> 3) * 12544)) + (((((int)blockIdx.x) & 15) >> 2) * 3136)) + (((((int)threadIdx.x) & 7) >> 1) * 784)) + (yy_inner * 112)) + ((((int)blockIdx.x) & 3) * 28)) + ((((int)threadIdx.x) & 1) * 7)) + xx_inner)] = conv2d_nchw_local[((yy_inner * 7) + xx_inner)];
      conv2d_nchw[((((((((((((int)blockIdx.x) >> 4) * 401408) + ((((int)threadIdx.x) >> 3) * 12544)) + (((((int)blockIdx.x) & 15) >> 2) * 3136)) + (((((int)threadIdx.x) & 7) >> 1) * 784)) + (yy_inner * 112)) + ((((int)blockIdx.x) & 3) * 28)) + ((((int)threadIdx.x) & 1) * 7)) + xx_inner) + 14)] = conv2d_nchw_local[(((yy_inner * 7) + xx_inner) + 49)];
    }
  }
}

