
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
  float conv2d_nchw_local[128];
  __shared__ float pad_temp_shared[648];
  __shared__ float kernel_shared[2304];
  for (int ff_c_outer_inner_init = 0; ff_c_outer_inner_init < 4; ++ff_c_outer_inner_init) {
    conv2d_nchw_local[(ff_c_outer_inner_init * 16)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_outer_inner_init * 16) + 64)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_outer_inner_init * 16) + 8)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_outer_inner_init * 16) + 72)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_outer_inner_init * 16) + 1)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_outer_inner_init * 16) + 65)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_outer_inner_init * 16) + 9)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_outer_inner_init * 16) + 73)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_outer_inner_init * 16) + 2)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_outer_inner_init * 16) + 66)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_outer_inner_init * 16) + 10)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_outer_inner_init * 16) + 74)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_outer_inner_init * 16) + 3)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_outer_inner_init * 16) + 67)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_outer_inner_init * 16) + 11)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_outer_inner_init * 16) + 75)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_outer_inner_init * 16) + 4)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_outer_inner_init * 16) + 68)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_outer_inner_init * 16) + 12)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_outer_inner_init * 16) + 76)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_outer_inner_init * 16) + 5)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_outer_inner_init * 16) + 69)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_outer_inner_init * 16) + 13)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_outer_inner_init * 16) + 77)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_outer_inner_init * 16) + 6)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_outer_inner_init * 16) + 70)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_outer_inner_init * 16) + 14)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_outer_inner_init * 16) + 78)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_outer_inner_init * 16) + 7)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_outer_inner_init * 16) + 71)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_outer_inner_init * 16) + 15)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_outer_inner_init * 16) + 79)] = 0.000000e+00f;
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 32; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = ((((1 <= (((((int)blockIdx.x) / 7) * 16) + (((int)threadIdx.x) / 18))) && (1 <= (((((int)blockIdx.x) % 7) * 16) + (((int)threadIdx.x) % 18)))) && ((((((int)blockIdx.x) % 7) * 16) + (((int)threadIdx.x) % 18)) < 113)) ? data[((((((rc_outer_outer * 25088) + ((((int)blockIdx.x) / 7) * 1792)) + ((((int)threadIdx.x) / 18) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) % 18)) - 113)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 256)] = (((((1 <= (((((int)blockIdx.x) / 7) * 16) + ((((((int)threadIdx.x) >> 1) + 128) % 162) / 9))) && ((((((int)blockIdx.x) / 7) * 16) + ((((((int)threadIdx.x) >> 1) + 128) % 162) / 9)) < 113)) && (1 <= (((((int)blockIdx.x) % 7) * 16) + ((((int)threadIdx.x) + 4) % 18)))) && ((((((int)blockIdx.x) % 7) * 16) + ((((int)threadIdx.x) + 4) % 18)) < 113)) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 256) / 324) * 12544)) + ((((int)blockIdx.x) / 7) * 1792)) + (((((((int)threadIdx.x) >> 1) + 128) % 162) / 9) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + ((((int)threadIdx.x) + 4) % 18)) - 113)] : 0.000000e+00f);
    if (((int)threadIdx.x) < 136) {
      pad_temp_shared[(((int)threadIdx.x) + 512)] = (((((((((int)blockIdx.x) / 7) * 16) + ((((((int)threadIdx.x) >> 1) + 94) % 162) / 9)) < 113) && (1 <= (((((int)blockIdx.x) % 7) * 16) + ((((int)threadIdx.x) + 8) % 18)))) && ((((((int)blockIdx.x) % 7) * 16) + ((((int)threadIdx.x) + 8) % 18)) < 113)) ? data[(((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 512) / 324) * 12544)) + ((((int)blockIdx.x) / 7) * 1792)) + (((((((int)threadIdx.x) >> 1) + 94) % 162) / 9) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + ((((int)threadIdx.x) + 8) % 18)) - 113)] : 0.000000e+00f);
    }
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)threadIdx.x) / 18) * 576) + (rc_outer_outer * 18)) + (((int)threadIdx.x) % 18))];
    kernel_shared[(((int)threadIdx.x) + 256)] = kernel[(((((((int)threadIdx.x) + 256) / 18) * 576) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 4) % 18))];
    kernel_shared[(((int)threadIdx.x) + 512)] = kernel[(((((((int)threadIdx.x) + 512) / 18) * 576) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 8) % 18))];
    kernel_shared[(((int)threadIdx.x) + 768)] = kernel[(((((((int)threadIdx.x) + 768) / 18) * 576) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 12) % 18))];
    kernel_shared[(((int)threadIdx.x) + 1024)] = kernel[(((((((int)threadIdx.x) + 1024) / 18) * 576) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 16) % 18))];
    kernel_shared[(((int)threadIdx.x) + 1280)] = kernel[(((((((int)threadIdx.x) + 1280) / 18) * 576) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 2) % 18))];
    kernel_shared[(((int)threadIdx.x) + 1536)] = kernel[(((((((int)threadIdx.x) + 1536) / 18) * 576) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 6) % 18))];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[(((((((int)threadIdx.x) + 1792) / 18) * 576) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 10) % 18))];
    kernel_shared[(((int)threadIdx.x) + 2048)] = kernel[(((((((int)threadIdx.x) + 2048) / 18) * 576) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 14) % 18))];
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 2; ++rc_outer_inner) {
      for (int rx_outer_inner = 0; rx_outer_inner < 3; ++rx_outer_inner) {
        for (int ff_c_outer_inner = 0; ff_c_outer_inner < 4; ++ff_c_outer_inner) {
          for (int yy_c_outer_inner = 0; yy_c_outer_inner < 4; ++yy_c_outer_inner) {
            conv2d_nchw_local[((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2))] = (conv2d_nchw_local[((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2))] + (pad_temp_shared[(((((rc_outer_inner * 324) + (((((int)threadIdx.x) & 31) >> 3) * 72)) + (yy_c_outer_inner * 18)) + ((((int)threadIdx.x) & 7) * 2)) + rx_outer_inner)] * kernel_shared[(((((((int)threadIdx.x) >> 5) * 144) + (ff_c_outer_inner * 36)) + (rc_outer_inner * 9)) + rx_outer_inner)]));
            conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 64)] = (conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 64)] + (pad_temp_shared[(((((rc_outer_inner * 324) + (((((int)threadIdx.x) & 31) >> 3) * 72)) + (yy_c_outer_inner * 18)) + ((((int)threadIdx.x) & 7) * 2)) + rx_outer_inner)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 144) + (ff_c_outer_inner * 36)) + (rc_outer_inner * 9)) + rx_outer_inner) + 1152)]));
            conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 8)] = (conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 8)] + (pad_temp_shared[(((((rc_outer_inner * 324) + (((((int)threadIdx.x) & 31) >> 3) * 72)) + (yy_c_outer_inner * 18)) + ((((int)threadIdx.x) & 7) * 2)) + rx_outer_inner)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 144) + (ff_c_outer_inner * 36)) + (rc_outer_inner * 9)) + rx_outer_inner) + 18)]));
            conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 72)] = (conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 72)] + (pad_temp_shared[(((((rc_outer_inner * 324) + (((((int)threadIdx.x) & 31) >> 3) * 72)) + (yy_c_outer_inner * 18)) + ((((int)threadIdx.x) & 7) * 2)) + rx_outer_inner)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 144) + (ff_c_outer_inner * 36)) + (rc_outer_inner * 9)) + rx_outer_inner) + 1170)]));
            conv2d_nchw_local[((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2))] = (conv2d_nchw_local[((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2))] + (pad_temp_shared[((((((rc_outer_inner * 324) + (((((int)threadIdx.x) & 31) >> 3) * 72)) + (yy_c_outer_inner * 18)) + ((((int)threadIdx.x) & 7) * 2)) + rx_outer_inner) + 18)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 144) + (ff_c_outer_inner * 36)) + (rc_outer_inner * 9)) + rx_outer_inner) + 3)]));
            conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 64)] = (conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 64)] + (pad_temp_shared[((((((rc_outer_inner * 324) + (((((int)threadIdx.x) & 31) >> 3) * 72)) + (yy_c_outer_inner * 18)) + ((((int)threadIdx.x) & 7) * 2)) + rx_outer_inner) + 18)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 144) + (ff_c_outer_inner * 36)) + (rc_outer_inner * 9)) + rx_outer_inner) + 1155)]));
            conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 8)] = (conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 8)] + (pad_temp_shared[((((((rc_outer_inner * 324) + (((((int)threadIdx.x) & 31) >> 3) * 72)) + (yy_c_outer_inner * 18)) + ((((int)threadIdx.x) & 7) * 2)) + rx_outer_inner) + 18)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 144) + (ff_c_outer_inner * 36)) + (rc_outer_inner * 9)) + rx_outer_inner) + 21)]));
            conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 72)] = (conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 72)] + (pad_temp_shared[((((((rc_outer_inner * 324) + (((((int)threadIdx.x) & 31) >> 3) * 72)) + (yy_c_outer_inner * 18)) + ((((int)threadIdx.x) & 7) * 2)) + rx_outer_inner) + 18)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 144) + (ff_c_outer_inner * 36)) + (rc_outer_inner * 9)) + rx_outer_inner) + 1173)]));
            conv2d_nchw_local[((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2))] = (conv2d_nchw_local[((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2))] + (pad_temp_shared[((((((rc_outer_inner * 324) + (((((int)threadIdx.x) & 31) >> 3) * 72)) + (yy_c_outer_inner * 18)) + ((((int)threadIdx.x) & 7) * 2)) + rx_outer_inner) + 36)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 144) + (ff_c_outer_inner * 36)) + (rc_outer_inner * 9)) + rx_outer_inner) + 6)]));
            conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 64)] = (conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 64)] + (pad_temp_shared[((((((rc_outer_inner * 324) + (((((int)threadIdx.x) & 31) >> 3) * 72)) + (yy_c_outer_inner * 18)) + ((((int)threadIdx.x) & 7) * 2)) + rx_outer_inner) + 36)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 144) + (ff_c_outer_inner * 36)) + (rc_outer_inner * 9)) + rx_outer_inner) + 1158)]));
            conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 8)] = (conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 8)] + (pad_temp_shared[((((((rc_outer_inner * 324) + (((((int)threadIdx.x) & 31) >> 3) * 72)) + (yy_c_outer_inner * 18)) + ((((int)threadIdx.x) & 7) * 2)) + rx_outer_inner) + 36)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 144) + (ff_c_outer_inner * 36)) + (rc_outer_inner * 9)) + rx_outer_inner) + 24)]));
            conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 72)] = (conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 72)] + (pad_temp_shared[((((((rc_outer_inner * 324) + (((((int)threadIdx.x) & 31) >> 3) * 72)) + (yy_c_outer_inner * 18)) + ((((int)threadIdx.x) & 7) * 2)) + rx_outer_inner) + 36)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 144) + (ff_c_outer_inner * 36)) + (rc_outer_inner * 9)) + rx_outer_inner) + 1176)]));
            conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 1)] = (conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 1)] + (pad_temp_shared[((((((rc_outer_inner * 324) + (((((int)threadIdx.x) & 31) >> 3) * 72)) + (yy_c_outer_inner * 18)) + ((((int)threadIdx.x) & 7) * 2)) + rx_outer_inner) + 1)] * kernel_shared[(((((((int)threadIdx.x) >> 5) * 144) + (ff_c_outer_inner * 36)) + (rc_outer_inner * 9)) + rx_outer_inner)]));
            conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 65)] = (conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 65)] + (pad_temp_shared[((((((rc_outer_inner * 324) + (((((int)threadIdx.x) & 31) >> 3) * 72)) + (yy_c_outer_inner * 18)) + ((((int)threadIdx.x) & 7) * 2)) + rx_outer_inner) + 1)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 144) + (ff_c_outer_inner * 36)) + (rc_outer_inner * 9)) + rx_outer_inner) + 1152)]));
            conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 9)] = (conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 9)] + (pad_temp_shared[((((((rc_outer_inner * 324) + (((((int)threadIdx.x) & 31) >> 3) * 72)) + (yy_c_outer_inner * 18)) + ((((int)threadIdx.x) & 7) * 2)) + rx_outer_inner) + 1)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 144) + (ff_c_outer_inner * 36)) + (rc_outer_inner * 9)) + rx_outer_inner) + 18)]));
            conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 73)] = (conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 73)] + (pad_temp_shared[((((((rc_outer_inner * 324) + (((((int)threadIdx.x) & 31) >> 3) * 72)) + (yy_c_outer_inner * 18)) + ((((int)threadIdx.x) & 7) * 2)) + rx_outer_inner) + 1)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 144) + (ff_c_outer_inner * 36)) + (rc_outer_inner * 9)) + rx_outer_inner) + 1170)]));
            conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 1)] = (conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 1)] + (pad_temp_shared[((((((rc_outer_inner * 324) + (((((int)threadIdx.x) & 31) >> 3) * 72)) + (yy_c_outer_inner * 18)) + ((((int)threadIdx.x) & 7) * 2)) + rx_outer_inner) + 19)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 144) + (ff_c_outer_inner * 36)) + (rc_outer_inner * 9)) + rx_outer_inner) + 3)]));
            conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 65)] = (conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 65)] + (pad_temp_shared[((((((rc_outer_inner * 324) + (((((int)threadIdx.x) & 31) >> 3) * 72)) + (yy_c_outer_inner * 18)) + ((((int)threadIdx.x) & 7) * 2)) + rx_outer_inner) + 19)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 144) + (ff_c_outer_inner * 36)) + (rc_outer_inner * 9)) + rx_outer_inner) + 1155)]));
            conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 9)] = (conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 9)] + (pad_temp_shared[((((((rc_outer_inner * 324) + (((((int)threadIdx.x) & 31) >> 3) * 72)) + (yy_c_outer_inner * 18)) + ((((int)threadIdx.x) & 7) * 2)) + rx_outer_inner) + 19)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 144) + (ff_c_outer_inner * 36)) + (rc_outer_inner * 9)) + rx_outer_inner) + 21)]));
            conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 73)] = (conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 73)] + (pad_temp_shared[((((((rc_outer_inner * 324) + (((((int)threadIdx.x) & 31) >> 3) * 72)) + (yy_c_outer_inner * 18)) + ((((int)threadIdx.x) & 7) * 2)) + rx_outer_inner) + 19)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 144) + (ff_c_outer_inner * 36)) + (rc_outer_inner * 9)) + rx_outer_inner) + 1173)]));
            conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 1)] = (conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 1)] + (pad_temp_shared[((((((rc_outer_inner * 324) + (((((int)threadIdx.x) & 31) >> 3) * 72)) + (yy_c_outer_inner * 18)) + ((((int)threadIdx.x) & 7) * 2)) + rx_outer_inner) + 37)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 144) + (ff_c_outer_inner * 36)) + (rc_outer_inner * 9)) + rx_outer_inner) + 6)]));
            conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 65)] = (conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 65)] + (pad_temp_shared[((((((rc_outer_inner * 324) + (((((int)threadIdx.x) & 31) >> 3) * 72)) + (yy_c_outer_inner * 18)) + ((((int)threadIdx.x) & 7) * 2)) + rx_outer_inner) + 37)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 144) + (ff_c_outer_inner * 36)) + (rc_outer_inner * 9)) + rx_outer_inner) + 1158)]));
            conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 9)] = (conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 9)] + (pad_temp_shared[((((((rc_outer_inner * 324) + (((((int)threadIdx.x) & 31) >> 3) * 72)) + (yy_c_outer_inner * 18)) + ((((int)threadIdx.x) & 7) * 2)) + rx_outer_inner) + 37)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 144) + (ff_c_outer_inner * 36)) + (rc_outer_inner * 9)) + rx_outer_inner) + 24)]));
            conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 73)] = (conv2d_nchw_local[(((ff_c_outer_inner * 16) + (yy_c_outer_inner * 2)) + 73)] + (pad_temp_shared[((((((rc_outer_inner * 324) + (((((int)threadIdx.x) & 31) >> 3) * 72)) + (yy_c_outer_inner * 18)) + ((((int)threadIdx.x) & 7) * 2)) + rx_outer_inner) + 37)] * kernel_shared[((((((((int)threadIdx.x) >> 5) * 144) + (ff_c_outer_inner * 36)) + (rc_outer_inner * 9)) + rx_outer_inner) + 1176)]));
          }
        }
      }
    }
  }
  for (int ff_inner = 0; ff_inner < 8; ++ff_inner) {
    for (int yy_inner = 0; yy_inner < 4; ++yy_inner) {
      for (int xx_inner = 0; xx_inner < 2; ++xx_inner) {
        conv2d_nchw[(((((((((((int)threadIdx.x) >> 5) * 100352) + (ff_inner * 12544)) + ((((int)blockIdx.x) / 7) * 1792)) + (((((int)threadIdx.x) & 31) >> 3) * 448)) + (yy_inner * 112)) + ((((int)blockIdx.x) % 7) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + xx_inner)] = conv2d_nchw_local[(((ff_inner * 8) + (yy_inner * 2)) + xx_inner)];
        conv2d_nchw[((((((((((((int)threadIdx.x) >> 5) * 100352) + (ff_inner * 12544)) + ((((int)blockIdx.x) / 7) * 1792)) + (((((int)threadIdx.x) & 31) >> 3) * 448)) + (yy_inner * 112)) + ((((int)blockIdx.x) % 7) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + xx_inner) + 802816)] = conv2d_nchw_local[((((ff_inner * 8) + (yy_inner * 2)) + xx_inner) + 64)];
      }
    }
  }
}


