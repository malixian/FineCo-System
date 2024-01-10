
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
extern "C" __global__ void __launch_bounds__(128) candidate0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[32];
  __shared__ float pad_temp_shared[2048];
  __shared__ float kernel_shared[2048];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[16] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[17] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[18] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[19] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[20] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[21] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  conv2d_nchw_local[22] = 0.000000e+00f;
  conv2d_nchw_local[7] = 0.000000e+00f;
  conv2d_nchw_local[23] = 0.000000e+00f;
  conv2d_nchw_local[8] = 0.000000e+00f;
  conv2d_nchw_local[24] = 0.000000e+00f;
  conv2d_nchw_local[9] = 0.000000e+00f;
  conv2d_nchw_local[25] = 0.000000e+00f;
  conv2d_nchw_local[10] = 0.000000e+00f;
  conv2d_nchw_local[26] = 0.000000e+00f;
  conv2d_nchw_local[11] = 0.000000e+00f;
  conv2d_nchw_local[27] = 0.000000e+00f;
  conv2d_nchw_local[12] = 0.000000e+00f;
  conv2d_nchw_local[28] = 0.000000e+00f;
  conv2d_nchw_local[13] = 0.000000e+00f;
  conv2d_nchw_local[29] = 0.000000e+00f;
  conv2d_nchw_local[14] = 0.000000e+00f;
  conv2d_nchw_local[30] = 0.000000e+00f;
  conv2d_nchw_local[15] = 0.000000e+00f;
  conv2d_nchw_local[31] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 8; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = data[((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7))];
    pad_temp_shared[(((int)threadIdx.x) + 128)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 6272)];
    pad_temp_shared[(((int)threadIdx.x) + 256)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 12544)];
    pad_temp_shared[(((int)threadIdx.x) + 384)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 18816)];
    pad_temp_shared[(((int)threadIdx.x) + 512)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 25088)];
    pad_temp_shared[(((int)threadIdx.x) + 640)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 31360)];
    pad_temp_shared[(((int)threadIdx.x) + 768)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 37632)];
    pad_temp_shared[(((int)threadIdx.x) + 896)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 43904)];
    pad_temp_shared[(((int)threadIdx.x) + 1024)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 50176)];
    pad_temp_shared[(((int)threadIdx.x) + 1152)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 56448)];
    pad_temp_shared[(((int)threadIdx.x) + 1280)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 62720)];
    pad_temp_shared[(((int)threadIdx.x) + 1408)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 68992)];
    pad_temp_shared[(((int)threadIdx.x) + 1536)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 75264)];
    pad_temp_shared[(((int)threadIdx.x) + 1664)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 81536)];
    pad_temp_shared[(((int)threadIdx.x) + 1792)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 87808)];
    pad_temp_shared[(((int)threadIdx.x) + 1920)] = data[(((((((rc_outer_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 63) >> 3) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 94080)];
    kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) / 49) * 16384) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31))];
    kernel_shared[(((int)threadIdx.x) + 128)] = kernel[((((((((int)blockIdx.x) / 49) * 16384) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 1024)];
    kernel_shared[(((int)threadIdx.x) + 256)] = kernel[((((((((int)blockIdx.x) / 49) * 16384) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 2048)];
    kernel_shared[(((int)threadIdx.x) + 384)] = kernel[((((((((int)blockIdx.x) / 49) * 16384) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 3072)];
    kernel_shared[(((int)threadIdx.x) + 512)] = kernel[((((((((int)blockIdx.x) / 49) * 16384) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 4096)];
    kernel_shared[(((int)threadIdx.x) + 640)] = kernel[((((((((int)blockIdx.x) / 49) * 16384) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 5120)];
    kernel_shared[(((int)threadIdx.x) + 768)] = kernel[((((((((int)blockIdx.x) / 49) * 16384) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 6144)];
    kernel_shared[(((int)threadIdx.x) + 896)] = kernel[((((((((int)blockIdx.x) / 49) * 16384) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 7168)];
    kernel_shared[(((int)threadIdx.x) + 1024)] = kernel[((((((((int)blockIdx.x) / 49) * 16384) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 8192)];
    kernel_shared[(((int)threadIdx.x) + 1152)] = kernel[((((((((int)blockIdx.x) / 49) * 16384) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 9216)];
    kernel_shared[(((int)threadIdx.x) + 1280)] = kernel[((((((((int)blockIdx.x) / 49) * 16384) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 10240)];
    kernel_shared[(((int)threadIdx.x) + 1408)] = kernel[((((((((int)blockIdx.x) / 49) * 16384) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 11264)];
    kernel_shared[(((int)threadIdx.x) + 1536)] = kernel[((((((((int)blockIdx.x) / 49) * 16384) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 12288)];
    kernel_shared[(((int)threadIdx.x) + 1664)] = kernel[((((((((int)blockIdx.x) / 49) * 16384) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 13312)];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[((((((((int)blockIdx.x) / 49) * 16384) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 14336)];
    kernel_shared[(((int)threadIdx.x) + 1920)] = kernel[((((((((int)blockIdx.x) / 49) * 16384) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 15360)];
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 8; ++rc_outer_inner) {
      for (int ff_c_outer_inner = 0; ff_c_outer_inner < 4; ++ff_c_outer_inner) {
        conv2d_nchw_local[(ff_c_outer_inner * 4)] = (conv2d_nchw_local[(ff_c_outer_inner * 4)] + (pad_temp_shared[(((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7))] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 4))]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 16)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 16)] + (pad_temp_shared[((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 32)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 4))]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 1)] + (pad_temp_shared[((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 8)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 4))]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 17)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 17)] + (pad_temp_shared[((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 40)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 256) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 4))]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 2)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 2)] + (pad_temp_shared[(((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7))] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 256) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 4)) + 32)]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 18)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 18)] + (pad_temp_shared[((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 32)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 256) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 4)) + 32)]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 3)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 3)] + (pad_temp_shared[((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 8)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 256) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 4)) + 32)]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 19)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 19)] + (pad_temp_shared[((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 40)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 256) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 4)) + 32)]));
        conv2d_nchw_local[(ff_c_outer_inner * 4)] = (conv2d_nchw_local[(ff_c_outer_inner * 4)] + (pad_temp_shared[((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 64)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 256) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 4)) + 1)]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 16)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 16)] + (pad_temp_shared[((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 96)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 256) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 4)) + 1)]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 1)] + (pad_temp_shared[((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 72)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 256) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 4)) + 1)]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 17)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 17)] + (pad_temp_shared[((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 104)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 256) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 4)) + 1)]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 2)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 2)] + (pad_temp_shared[((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 64)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 256) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 4)) + 33)]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 18)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 18)] + (pad_temp_shared[((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 96)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 256) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 4)) + 33)]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 3)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 3)] + (pad_temp_shared[((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 72)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 256) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 4)) + 33)]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 19)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 19)] + (pad_temp_shared[((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 104)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 256) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 4)) + 33)]));
        conv2d_nchw_local[(ff_c_outer_inner * 4)] = (conv2d_nchw_local[(ff_c_outer_inner * 4)] + (pad_temp_shared[((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 128)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 256) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 4)) + 2)]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 16)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 16)] + (pad_temp_shared[((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 160)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 256) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 4)) + 2)]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 1)] + (pad_temp_shared[((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 136)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 256) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 4)) + 2)]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 17)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 17)] + (pad_temp_shared[((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 168)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 256) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 4)) + 2)]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 2)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 2)] + (pad_temp_shared[((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 128)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 256) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 4)) + 34)]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 18)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 18)] + (pad_temp_shared[((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 160)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 256) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 4)) + 34)]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 3)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 3)] + (pad_temp_shared[((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 136)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 256) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 4)) + 34)]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 19)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 19)] + (pad_temp_shared[((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 168)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 256) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 4)) + 34)]));
        conv2d_nchw_local[(ff_c_outer_inner * 4)] = (conv2d_nchw_local[(ff_c_outer_inner * 4)] + (pad_temp_shared[((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 192)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 256) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 4)) + 3)]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 16)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 16)] + (pad_temp_shared[((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 224)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 256) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 4)) + 3)]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 1)] + (pad_temp_shared[((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 200)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 256) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 4)) + 3)]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 17)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 17)] + (pad_temp_shared[((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 232)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 256) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 4)) + 3)]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 2)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 2)] + (pad_temp_shared[((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 192)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 256) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 4)) + 35)]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 18)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 18)] + (pad_temp_shared[((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 224)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 256) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 4)) + 35)]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 3)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 3)] + (pad_temp_shared[((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 200)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 256) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 4)) + 35)]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 19)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 19)] + (pad_temp_shared[((((rc_outer_inner * 256) + (((((int)threadIdx.x) & 15) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 232)] * kernel_shared[(((((((int)threadIdx.x) >> 4) * 256) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 4)) + 35)]));
      }
    }
  }
  for (int ff_inner = 0; ff_inner < 8; ++ff_inner) {
    for (int yy_inner = 0; yy_inner < 2; ++yy_inner) {
      conv2d_nchw[(((((((((((int)blockIdx.x) / 49) * 200704) + ((((int)threadIdx.x) >> 4) * 25088)) + (ff_inner * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (yy_inner * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7))] = conv2d_nchw_local[((ff_inner * 2) + yy_inner)];
      conv2d_nchw[((((((((((((int)blockIdx.x) / 49) * 200704) + ((((int)threadIdx.x) >> 4) * 25088)) + (ff_inner * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + (((((int)threadIdx.x) & 15) >> 3) * 112)) + (yy_inner * 56)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 224)] = conv2d_nchw_local[(((ff_inner * 2) + yy_inner) + 16)];
    }
  }
}


