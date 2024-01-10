
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
extern "C" __global__ void __launch_bounds__(112) candidate0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[8];
  __shared__ float pad_temp_shared[2320];
  __shared__ float kernel_shared[4608];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  conv2d_nchw_local[7] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 8; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[(((int)threadIdx.x) * 6)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + (((((int)threadIdx.x) * 6) % 145) / 29))) && (1 <= ((((int)threadIdx.x) * 6) % 29))) ? data[((((((rc_outer_outer * 12544) + (((((int)threadIdx.x) * 6) / 145) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + ((((((int)threadIdx.x) * 6) % 145) / 29) * 28)) + ((((int)threadIdx.x) * 6) % 29)) - 29)] : 0.000000e+00f);
    pad_temp_shared[((((int)threadIdx.x) * 6) + 1)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((((int)threadIdx.x) * 6) + 1) % 145) / 29))) && (1 <= (((((int)threadIdx.x) * 6) + 1) % 29))) ? data[((((((rc_outer_outer * 12544) + ((((((int)threadIdx.x) * 6) + 1) / 145) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((((((int)threadIdx.x) * 6) + 1) % 145) / 29) * 28)) + (((((int)threadIdx.x) * 6) + 1) % 29)) - 29)] : 0.000000e+00f);
    pad_temp_shared[((((int)threadIdx.x) * 6) + 2)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((((int)threadIdx.x) * 6) + 2) % 145) / 29))) && (1 <= (((((int)threadIdx.x) * 6) + 2) % 29))) ? data[((((((rc_outer_outer * 12544) + ((((((int)threadIdx.x) * 6) + 2) / 145) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((((((int)threadIdx.x) * 6) + 2) % 145) / 29) * 28)) + (((((int)threadIdx.x) * 6) + 2) % 29)) - 29)] : 0.000000e+00f);
    pad_temp_shared[((((int)threadIdx.x) * 6) + 3)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((((int)threadIdx.x) * 6) + 3) % 145) / 29))) && (1 <= (((((int)threadIdx.x) * 6) + 3) % 29))) ? data[((((((rc_outer_outer * 12544) + ((((((int)threadIdx.x) * 6) + 3) / 145) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((((((int)threadIdx.x) * 6) + 3) % 145) / 29) * 28)) + (((((int)threadIdx.x) * 6) + 3) % 29)) - 29)] : 0.000000e+00f);
    pad_temp_shared[((((int)threadIdx.x) * 6) + 4)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((((int)threadIdx.x) * 6) + 4) % 145) / 29))) && (1 <= (((((int)threadIdx.x) * 6) + 4) % 29))) ? data[((((((rc_outer_outer * 12544) + ((((((int)threadIdx.x) * 6) + 4) / 145) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((((((int)threadIdx.x) * 6) + 4) % 145) / 29) * 28)) + (((((int)threadIdx.x) * 6) + 4) % 29)) - 29)] : 0.000000e+00f);
    pad_temp_shared[((((int)threadIdx.x) * 6) + 5)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((((int)threadIdx.x) * 6) + 5) % 145) / 29))) && (1 <= (((((int)threadIdx.x) * 6) + 5) % 29))) ? data[((((((rc_outer_outer * 12544) + ((((((int)threadIdx.x) * 6) + 5) / 145) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((((((int)threadIdx.x) * 6) + 5) % 145) / 29) * 28)) + (((((int)threadIdx.x) * 6) + 5) % 29)) - 29)] : 0.000000e+00f);
    pad_temp_shared[((((int)threadIdx.x) * 6) + 672)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((((int)threadIdx.x) * 6) + 92) % 145) / 29))) && (1 <= (((((int)threadIdx.x) * 6) + 5) % 29))) ? data[((((((rc_outer_outer * 12544) + ((((((int)threadIdx.x) * 6) + 672) / 145) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((((((int)threadIdx.x) * 6) + 92) % 145) / 29) * 28)) + (((((int)threadIdx.x) * 6) + 5) % 29)) - 29)] : 0.000000e+00f);
    pad_temp_shared[((((int)threadIdx.x) * 6) + 673)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((((int)threadIdx.x) * 6) + 93) % 145) / 29))) && (1 <= (((((int)threadIdx.x) * 6) + 6) % 29))) ? data[((((((rc_outer_outer * 12544) + ((((((int)threadIdx.x) * 6) + 673) / 145) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((((((int)threadIdx.x) * 6) + 93) % 145) / 29) * 28)) + (((((int)threadIdx.x) * 6) + 6) % 29)) - 29)] : 0.000000e+00f);
    pad_temp_shared[((((int)threadIdx.x) * 6) + 674)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((((int)threadIdx.x) * 6) + 94) % 145) / 29))) && (1 <= (((((int)threadIdx.x) * 6) + 7) % 29))) ? data[((((((rc_outer_outer * 12544) + ((((((int)threadIdx.x) * 6) + 674) / 145) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((((((int)threadIdx.x) * 6) + 94) % 145) / 29) * 28)) + (((((int)threadIdx.x) * 6) + 7) % 29)) - 29)] : 0.000000e+00f);
    pad_temp_shared[((((int)threadIdx.x) * 6) + 675)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((((int)threadIdx.x) * 6) + 95) % 145) / 29))) && (1 <= (((((int)threadIdx.x) * 6) + 8) % 29))) ? data[((((((rc_outer_outer * 12544) + ((((((int)threadIdx.x) * 6) + 675) / 145) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((((((int)threadIdx.x) * 6) + 95) % 145) / 29) * 28)) + (((((int)threadIdx.x) * 6) + 8) % 29)) - 29)] : 0.000000e+00f);
    pad_temp_shared[((((int)threadIdx.x) * 6) + 676)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((((int)threadIdx.x) * 6) + 96) % 145) / 29))) && (1 <= (((((int)threadIdx.x) * 6) + 9) % 29))) ? data[((((((rc_outer_outer * 12544) + ((((((int)threadIdx.x) * 6) + 676) / 145) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((((((int)threadIdx.x) * 6) + 96) % 145) / 29) * 28)) + (((((int)threadIdx.x) * 6) + 9) % 29)) - 29)] : 0.000000e+00f);
    pad_temp_shared[((((int)threadIdx.x) * 6) + 677)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((((int)threadIdx.x) * 6) + 97) % 145) / 29))) && (1 <= (((((int)threadIdx.x) * 6) + 10) % 29))) ? data[((((((rc_outer_outer * 12544) + ((((((int)threadIdx.x) * 6) + 677) / 145) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((((((int)threadIdx.x) * 6) + 97) % 145) / 29) * 28)) + (((((int)threadIdx.x) * 6) + 10) % 29)) - 29)] : 0.000000e+00f);
    pad_temp_shared[((((int)threadIdx.x) * 6) + 1344)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((((int)threadIdx.x) * 6) + 39) % 145) / 29))) && (1 <= (((((int)threadIdx.x) * 6) + 10) % 29))) ? data[((((((rc_outer_outer * 12544) + ((((((int)threadIdx.x) * 6) + 1344) / 145) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((((((int)threadIdx.x) * 6) + 39) % 145) / 29) * 28)) + (((((int)threadIdx.x) * 6) + 10) % 29)) - 29)] : 0.000000e+00f);
    pad_temp_shared[((((int)threadIdx.x) * 6) + 1345)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((((int)threadIdx.x) * 6) + 40) % 145) / 29))) && (1 <= (((((int)threadIdx.x) * 6) + 11) % 29))) ? data[((((((rc_outer_outer * 12544) + ((((((int)threadIdx.x) * 6) + 1345) / 145) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((((((int)threadIdx.x) * 6) + 40) % 145) / 29) * 28)) + (((((int)threadIdx.x) * 6) + 11) % 29)) - 29)] : 0.000000e+00f);
    pad_temp_shared[((((int)threadIdx.x) * 6) + 1346)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((((int)threadIdx.x) * 6) + 41) % 145) / 29))) && (1 <= (((((int)threadIdx.x) * 6) + 12) % 29))) ? data[((((((rc_outer_outer * 12544) + ((((((int)threadIdx.x) * 6) + 1346) / 145) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((((((int)threadIdx.x) * 6) + 41) % 145) / 29) * 28)) + (((((int)threadIdx.x) * 6) + 12) % 29)) - 29)] : 0.000000e+00f);
    pad_temp_shared[((((int)threadIdx.x) * 6) + 1347)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((((int)threadIdx.x) * 6) + 42) % 145) / 29))) && (1 <= (((((int)threadIdx.x) * 6) + 13) % 29))) ? data[((((((rc_outer_outer * 12544) + ((((((int)threadIdx.x) * 6) + 1347) / 145) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((((((int)threadIdx.x) * 6) + 42) % 145) / 29) * 28)) + (((((int)threadIdx.x) * 6) + 13) % 29)) - 29)] : 0.000000e+00f);
    pad_temp_shared[((((int)threadIdx.x) * 6) + 1348)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((((int)threadIdx.x) * 6) + 43) % 145) / 29))) && (1 <= (((((int)threadIdx.x) * 6) + 14) % 29))) ? data[((((((rc_outer_outer * 12544) + ((((((int)threadIdx.x) * 6) + 1348) / 145) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((((((int)threadIdx.x) * 6) + 43) % 145) / 29) * 28)) + (((((int)threadIdx.x) * 6) + 14) % 29)) - 29)] : 0.000000e+00f);
    pad_temp_shared[((((int)threadIdx.x) * 6) + 1349)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((((int)threadIdx.x) * 6) + 44) % 145) / 29))) && (1 <= (((((int)threadIdx.x) * 6) + 15) % 29))) ? data[((((((rc_outer_outer * 12544) + ((((((int)threadIdx.x) * 6) + 1349) / 145) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((((((int)threadIdx.x) * 6) + 44) % 145) / 29) * 28)) + (((((int)threadIdx.x) * 6) + 15) % 29)) - 29)] : 0.000000e+00f);
    if (((int)threadIdx.x) < 51) {
      pad_temp_shared[((((int)threadIdx.x) * 6) + 2016)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((((int)threadIdx.x) * 6) + 131) % 145) / 29))) && (1 <= (((((int)threadIdx.x) * 6) + 15) % 29))) ? data[((((((rc_outer_outer * 12544) + ((((((int)threadIdx.x) * 6) + 2016) / 145) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((((((int)threadIdx.x) * 6) + 131) % 145) / 29) * 28)) + (((((int)threadIdx.x) * 6) + 15) % 29)) - 29)] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 51) {
      pad_temp_shared[((((int)threadIdx.x) * 6) + 2017)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((((int)threadIdx.x) * 6) + 132) % 145) / 29))) && (1 <= (((((int)threadIdx.x) * 6) + 16) % 29))) ? data[((((((rc_outer_outer * 12544) + ((((((int)threadIdx.x) * 6) + 2017) / 145) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((((((int)threadIdx.x) * 6) + 132) % 145) / 29) * 28)) + (((((int)threadIdx.x) * 6) + 16) % 29)) - 29)] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 51) {
      pad_temp_shared[((((int)threadIdx.x) * 6) + 2018)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((((int)threadIdx.x) * 6) + 133) % 145) / 29))) && (1 <= (((((int)threadIdx.x) * 6) + 17) % 29))) ? data[((((((rc_outer_outer * 12544) + ((((((int)threadIdx.x) * 6) + 2018) / 145) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((((((int)threadIdx.x) * 6) + 133) % 145) / 29) * 28)) + (((((int)threadIdx.x) * 6) + 17) % 29)) - 29)] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 51) {
      pad_temp_shared[((((int)threadIdx.x) * 6) + 2019)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((((int)threadIdx.x) * 6) + 134) % 145) / 29))) && (1 <= (((((int)threadIdx.x) * 6) + 18) % 29))) ? data[((((((rc_outer_outer * 12544) + ((((((int)threadIdx.x) * 6) + 2019) / 145) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((((((int)threadIdx.x) * 6) + 134) % 145) / 29) * 28)) + (((((int)threadIdx.x) * 6) + 18) % 29)) - 29)] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 50) {
      pad_temp_shared[((((int)threadIdx.x) * 6) + 2020)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((((int)threadIdx.x) * 6) + 135) % 145) / 29))) && (1 <= (((((int)threadIdx.x) * 6) + 19) % 29))) ? data[((((((rc_outer_outer * 12544) + ((((((int)threadIdx.x) * 6) + 2020) / 145) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((((((int)threadIdx.x) * 6) + 135) % 145) / 29) * 28)) + (((((int)threadIdx.x) * 6) + 19) % 29)) - 29)] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 50) {
      pad_temp_shared[((((int)threadIdx.x) * 6) + 2021)] = (((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((((int)threadIdx.x) * 6) + 136) % 145) / 29))) && (1 <= (((((int)threadIdx.x) * 6) + 20) % 29))) ? data[((((((rc_outer_outer * 12544) + ((((((int)threadIdx.x) * 6) + 2021) / 145) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((((((int)threadIdx.x) * 6) + 136) % 145) / 29) * 28)) + (((((int)threadIdx.x) * 6) + 20) % 29)) - 29)] : 0.000000e+00f);
    }
    kernel_shared[(((int)threadIdx.x) * 2)] = kernel[(((((((int)blockIdx.x) / 7) * 36864) + ((((int)threadIdx.x) / 72) * 1152)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) % 72) * 2))];
    kernel_shared[((((int)threadIdx.x) * 2) + 1)] = kernel[((((((((int)blockIdx.x) / 7) * 36864) + ((((int)threadIdx.x) / 72) * 1152)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) % 72) * 2)) + 1)];
    kernel_shared[((((int)threadIdx.x) * 2) + 224)] = kernel[(((((((int)blockIdx.x) / 7) * 36864) + (((((int)threadIdx.x) + 112) / 72) * 1152)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) * 2) + 80) % 144))];
    kernel_shared[((((int)threadIdx.x) * 2) + 225)] = kernel[(((((((int)blockIdx.x) / 7) * 36864) + (((((int)threadIdx.x) + 112) / 72) * 1152)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) * 2) + 81) % 144))];
    kernel_shared[((((int)threadIdx.x) * 2) + 448)] = kernel[(((((((int)blockIdx.x) / 7) * 36864) + (((((int)threadIdx.x) + 224) / 72) * 1152)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) * 2) + 16) % 144))];
    kernel_shared[((((int)threadIdx.x) * 2) + 449)] = kernel[(((((((int)blockIdx.x) / 7) * 36864) + (((((int)threadIdx.x) + 224) / 72) * 1152)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) * 2) + 17) % 144))];
    kernel_shared[((((int)threadIdx.x) * 2) + 672)] = kernel[(((((((int)blockIdx.x) / 7) * 36864) + (((((int)threadIdx.x) + 336) / 72) * 1152)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) * 2) + 96) % 144))];
    kernel_shared[((((int)threadIdx.x) * 2) + 673)] = kernel[(((((((int)blockIdx.x) / 7) * 36864) + (((((int)threadIdx.x) + 336) / 72) * 1152)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) * 2) + 97) % 144))];
    kernel_shared[((((int)threadIdx.x) * 2) + 896)] = kernel[(((((((int)blockIdx.x) / 7) * 36864) + (((((int)threadIdx.x) + 448) / 72) * 1152)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) * 2) + 32) % 144))];
    kernel_shared[((((int)threadIdx.x) * 2) + 897)] = kernel[(((((((int)blockIdx.x) / 7) * 36864) + (((((int)threadIdx.x) + 448) / 72) * 1152)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) * 2) + 33) % 144))];
    kernel_shared[((((int)threadIdx.x) * 2) + 1120)] = kernel[(((((((int)blockIdx.x) / 7) * 36864) + (((((int)threadIdx.x) + 560) / 72) * 1152)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) * 2) + 112) % 144))];
    kernel_shared[((((int)threadIdx.x) * 2) + 1121)] = kernel[(((((((int)blockIdx.x) / 7) * 36864) + (((((int)threadIdx.x) + 560) / 72) * 1152)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) * 2) + 113) % 144))];
    kernel_shared[((((int)threadIdx.x) * 2) + 1344)] = kernel[(((((((int)blockIdx.x) / 7) * 36864) + (((((int)threadIdx.x) + 672) / 72) * 1152)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) * 2) + 48) % 144))];
    kernel_shared[((((int)threadIdx.x) * 2) + 1345)] = kernel[(((((((int)blockIdx.x) / 7) * 36864) + (((((int)threadIdx.x) + 672) / 72) * 1152)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) * 2) + 49) % 144))];
    kernel_shared[((((int)threadIdx.x) * 2) + 1568)] = kernel[(((((((int)blockIdx.x) / 7) * 36864) + (((((int)threadIdx.x) + 784) / 72) * 1152)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) * 2) + 128) % 144))];
    kernel_shared[((((int)threadIdx.x) * 2) + 1569)] = kernel[(((((((int)blockIdx.x) / 7) * 36864) + (((((int)threadIdx.x) + 784) / 72) * 1152)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) * 2) + 129) % 144))];
    kernel_shared[((((int)threadIdx.x) * 2) + 1792)] = kernel[(((((((int)blockIdx.x) / 7) * 36864) + (((((int)threadIdx.x) + 896) / 72) * 1152)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) * 2) + 64) % 144))];
    kernel_shared[((((int)threadIdx.x) * 2) + 1793)] = kernel[(((((((int)blockIdx.x) / 7) * 36864) + (((((int)threadIdx.x) + 896) / 72) * 1152)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) * 2) + 65) % 144))];
    kernel_shared[((((int)threadIdx.x) * 2) + 2016)] = kernel[((((((((int)blockIdx.x) / 7) * 36864) + ((((int)threadIdx.x) / 72) * 1152)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) % 72) * 2)) + 16128)];
    kernel_shared[((((int)threadIdx.x) * 2) + 2017)] = kernel[((((((((int)blockIdx.x) / 7) * 36864) + ((((int)threadIdx.x) / 72) * 1152)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) % 72) * 2)) + 16129)];
    kernel_shared[((((int)threadIdx.x) * 2) + 2240)] = kernel[(((((((int)blockIdx.x) / 7) * 36864) + (((((int)threadIdx.x) + 1120) / 72) * 1152)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) * 2) + 80) % 144))];
    kernel_shared[((((int)threadIdx.x) * 2) + 2241)] = kernel[(((((((int)blockIdx.x) / 7) * 36864) + (((((int)threadIdx.x) + 1120) / 72) * 1152)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) * 2) + 81) % 144))];
    kernel_shared[((((int)threadIdx.x) * 2) + 2464)] = kernel[(((((((int)blockIdx.x) / 7) * 36864) + (((((int)threadIdx.x) + 1232) / 72) * 1152)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) * 2) + 16) % 144))];
    kernel_shared[((((int)threadIdx.x) * 2) + 2465)] = kernel[(((((((int)blockIdx.x) / 7) * 36864) + (((((int)threadIdx.x) + 1232) / 72) * 1152)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) * 2) + 17) % 144))];
    kernel_shared[((((int)threadIdx.x) * 2) + 2688)] = kernel[(((((((int)blockIdx.x) / 7) * 36864) + (((((int)threadIdx.x) + 1344) / 72) * 1152)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) * 2) + 96) % 144))];
    kernel_shared[((((int)threadIdx.x) * 2) + 2689)] = kernel[(((((((int)blockIdx.x) / 7) * 36864) + (((((int)threadIdx.x) + 1344) / 72) * 1152)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) * 2) + 97) % 144))];
    kernel_shared[((((int)threadIdx.x) * 2) + 2912)] = kernel[(((((((int)blockIdx.x) / 7) * 36864) + (((((int)threadIdx.x) + 1456) / 72) * 1152)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) * 2) + 32) % 144))];
    kernel_shared[((((int)threadIdx.x) * 2) + 2913)] = kernel[(((((((int)blockIdx.x) / 7) * 36864) + (((((int)threadIdx.x) + 1456) / 72) * 1152)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) * 2) + 33) % 144))];
    kernel_shared[((((int)threadIdx.x) * 2) + 3136)] = kernel[(((((((int)blockIdx.x) / 7) * 36864) + (((((int)threadIdx.x) + 1568) / 72) * 1152)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) * 2) + 112) % 144))];
    kernel_shared[((((int)threadIdx.x) * 2) + 3137)] = kernel[(((((((int)blockIdx.x) / 7) * 36864) + (((((int)threadIdx.x) + 1568) / 72) * 1152)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) * 2) + 113) % 144))];
    kernel_shared[((((int)threadIdx.x) * 2) + 3360)] = kernel[(((((((int)blockIdx.x) / 7) * 36864) + (((((int)threadIdx.x) + 1680) / 72) * 1152)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) * 2) + 48) % 144))];
    kernel_shared[((((int)threadIdx.x) * 2) + 3361)] = kernel[(((((((int)blockIdx.x) / 7) * 36864) + (((((int)threadIdx.x) + 1680) / 72) * 1152)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) * 2) + 49) % 144))];
    kernel_shared[((((int)threadIdx.x) * 2) + 3584)] = kernel[(((((((int)blockIdx.x) / 7) * 36864) + (((((int)threadIdx.x) + 1792) / 72) * 1152)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) * 2) + 128) % 144))];
    kernel_shared[((((int)threadIdx.x) * 2) + 3585)] = kernel[(((((((int)blockIdx.x) / 7) * 36864) + (((((int)threadIdx.x) + 1792) / 72) * 1152)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) * 2) + 129) % 144))];
    kernel_shared[((((int)threadIdx.x) * 2) + 3808)] = kernel[(((((((int)blockIdx.x) / 7) * 36864) + (((((int)threadIdx.x) + 1904) / 72) * 1152)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) * 2) + 64) % 144))];
    kernel_shared[((((int)threadIdx.x) * 2) + 3809)] = kernel[(((((((int)blockIdx.x) / 7) * 36864) + (((((int)threadIdx.x) + 1904) / 72) * 1152)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) * 2) + 65) % 144))];
    kernel_shared[((((int)threadIdx.x) * 2) + 4032)] = kernel[((((((((int)blockIdx.x) / 7) * 36864) + ((((int)threadIdx.x) / 72) * 1152)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) % 72) * 2)) + 32256)];
    kernel_shared[((((int)threadIdx.x) * 2) + 4033)] = kernel[((((((((int)blockIdx.x) / 7) * 36864) + ((((int)threadIdx.x) / 72) * 1152)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) % 72) * 2)) + 32257)];
    kernel_shared[((((int)threadIdx.x) * 2) + 4256)] = kernel[(((((((int)blockIdx.x) / 7) * 36864) + (((((int)threadIdx.x) + 2128) / 72) * 1152)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) * 2) + 80) % 144))];
    kernel_shared[((((int)threadIdx.x) * 2) + 4257)] = kernel[(((((((int)blockIdx.x) / 7) * 36864) + (((((int)threadIdx.x) + 2128) / 72) * 1152)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) * 2) + 81) % 144))];
    if (((int)threadIdx.x) < 64) {
      kernel_shared[((((int)threadIdx.x) * 2) + 4480)] = kernel[(((((((int)blockIdx.x) / 7) * 36864) + (((((int)threadIdx.x) + 2240) / 72) * 1152)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) * 2) + 16))];
    }
    if (((int)threadIdx.x) < 64) {
      kernel_shared[((((int)threadIdx.x) * 2) + 4481)] = kernel[(((((((int)blockIdx.x) / 7) * 36864) + (((((int)threadIdx.x) + 2240) / 72) * 1152)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) * 2) + 17))];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 2; ++rc_outer_inner) {
      for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
        for (int ry_inner = 0; ry_inner < 3; ++ry_inner) {
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((rc_outer_inner * 1160) + (rc_inner * 145)) + (ry_inner * 29)) + ((((int)threadIdx.x) % 14) * 2))] * kernel_shared[(((((((int)threadIdx.x) / 14) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3))]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((rc_outer_inner * 1160) + (rc_inner * 145)) + (ry_inner * 29)) + ((((int)threadIdx.x) % 14) * 2)) + 58)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3))]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((rc_outer_inner * 1160) + (rc_inner * 145)) + (ry_inner * 29)) + ((((int)threadIdx.x) % 14) * 2))] * kernel_shared[((((((((int)threadIdx.x) / 14) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 1152)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((rc_outer_inner * 1160) + (rc_inner * 145)) + (ry_inner * 29)) + ((((int)threadIdx.x) % 14) * 2)) + 58)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 1152)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[((((rc_outer_inner * 1160) + (rc_inner * 145)) + (ry_inner * 29)) + ((((int)threadIdx.x) % 14) * 2))] * kernel_shared[((((((((int)threadIdx.x) / 14) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 2304)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((rc_outer_inner * 1160) + (rc_inner * 145)) + (ry_inner * 29)) + ((((int)threadIdx.x) % 14) * 2)) + 58)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 2304)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[((((rc_outer_inner * 1160) + (rc_inner * 145)) + (ry_inner * 29)) + ((((int)threadIdx.x) % 14) * 2))] * kernel_shared[((((((((int)threadIdx.x) / 14) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 3456)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((rc_outer_inner * 1160) + (rc_inner * 145)) + (ry_inner * 29)) + ((((int)threadIdx.x) % 14) * 2)) + 58)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 3456)]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((rc_outer_inner * 1160) + (rc_inner * 145)) + (ry_inner * 29)) + ((((int)threadIdx.x) % 14) * 2)) + 1)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 1)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((rc_outer_inner * 1160) + (rc_inner * 145)) + (ry_inner * 29)) + ((((int)threadIdx.x) % 14) * 2)) + 59)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 1)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((rc_outer_inner * 1160) + (rc_inner * 145)) + (ry_inner * 29)) + ((((int)threadIdx.x) % 14) * 2)) + 1)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 1153)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((rc_outer_inner * 1160) + (rc_inner * 145)) + (ry_inner * 29)) + ((((int)threadIdx.x) % 14) * 2)) + 59)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 1153)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((rc_outer_inner * 1160) + (rc_inner * 145)) + (ry_inner * 29)) + ((((int)threadIdx.x) % 14) * 2)) + 1)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 2305)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((rc_outer_inner * 1160) + (rc_inner * 145)) + (ry_inner * 29)) + ((((int)threadIdx.x) % 14) * 2)) + 59)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 2305)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((rc_outer_inner * 1160) + (rc_inner * 145)) + (ry_inner * 29)) + ((((int)threadIdx.x) % 14) * 2)) + 1)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 3457)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((rc_outer_inner * 1160) + (rc_inner * 145)) + (ry_inner * 29)) + ((((int)threadIdx.x) % 14) * 2)) + 59)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 3457)]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((rc_outer_inner * 1160) + (rc_inner * 145)) + (ry_inner * 29)) + ((((int)threadIdx.x) % 14) * 2)) + 2)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 2)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((rc_outer_inner * 1160) + (rc_inner * 145)) + (ry_inner * 29)) + ((((int)threadIdx.x) % 14) * 2)) + 60)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 2)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((rc_outer_inner * 1160) + (rc_inner * 145)) + (ry_inner * 29)) + ((((int)threadIdx.x) % 14) * 2)) + 2)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 1154)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((rc_outer_inner * 1160) + (rc_inner * 145)) + (ry_inner * 29)) + ((((int)threadIdx.x) % 14) * 2)) + 60)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 1154)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((rc_outer_inner * 1160) + (rc_inner * 145)) + (ry_inner * 29)) + ((((int)threadIdx.x) % 14) * 2)) + 2)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 2306)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((rc_outer_inner * 1160) + (rc_inner * 145)) + (ry_inner * 29)) + ((((int)threadIdx.x) % 14) * 2)) + 60)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 2306)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((rc_outer_inner * 1160) + (rc_inner * 145)) + (ry_inner * 29)) + ((((int)threadIdx.x) % 14) * 2)) + 2)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 3458)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((rc_outer_inner * 1160) + (rc_inner * 145)) + (ry_inner * 29)) + ((((int)threadIdx.x) % 14) * 2)) + 60)] * kernel_shared[((((((((int)threadIdx.x) / 14) * 144) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 3458)]));
        }
      }
    }
  }
  conv2d_nchw[(((((((int)blockIdx.x) / 7) * 6272) + ((((int)threadIdx.x) / 14) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 14))] = conv2d_nchw_local[0];
  conv2d_nchw[((((((((int)blockIdx.x) / 7) * 6272) + ((((int)threadIdx.x) / 14) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 14)) + 14)] = conv2d_nchw_local[1];
  conv2d_nchw[((((((((int)blockIdx.x) / 7) * 6272) + ((((int)threadIdx.x) / 14) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 14)) + 1568)] = conv2d_nchw_local[2];
  conv2d_nchw[((((((((int)blockIdx.x) / 7) * 6272) + ((((int)threadIdx.x) / 14) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 14)) + 1582)] = conv2d_nchw_local[3];
  conv2d_nchw[((((((((int)blockIdx.x) / 7) * 6272) + ((((int)threadIdx.x) / 14) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 14)) + 3136)] = conv2d_nchw_local[4];
  conv2d_nchw[((((((((int)blockIdx.x) / 7) * 6272) + ((((int)threadIdx.x) / 14) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 14)) + 3150)] = conv2d_nchw_local[5];
  conv2d_nchw[((((((((int)blockIdx.x) / 7) * 6272) + ((((int)threadIdx.x) / 14) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 14)) + 4704)] = conv2d_nchw_local[6];
  conv2d_nchw[((((((((int)blockIdx.x) / 7) * 6272) + ((((int)threadIdx.x) / 14) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 14)) + 4718)] = conv2d_nchw_local[7];
}


