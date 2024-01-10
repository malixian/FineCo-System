
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
extern "C" __global__ void __launch_bounds__(196) candidate6(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[32];
  __shared__ float pad_temp_shared[3136];
  __shared__ float kernel_shared[512];
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
  for (int rc_outer_outer = 0; rc_outer_outer < 16; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = data[(((((rc_outer_outer * 50176) + (((((int)blockIdx.x) & 15) >> 1) * 392)) + ((((int)threadIdx.x) / 28) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + (((int)threadIdx.x) % 28))];
    pad_temp_shared[(((int)threadIdx.x) + 196)] = data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) & 15) >> 1) * 392)) + ((((int)threadIdx.x) / 28) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + (((int)threadIdx.x) % 28)) + 3136)];
    pad_temp_shared[(((int)threadIdx.x) + 392)] = data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) & 15) >> 1) * 392)) + ((((int)threadIdx.x) / 28) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + (((int)threadIdx.x) % 28)) + 6272)];
    pad_temp_shared[(((int)threadIdx.x) + 588)] = data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) & 15) >> 1) * 392)) + ((((int)threadIdx.x) / 28) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + (((int)threadIdx.x) % 28)) + 9408)];
    pad_temp_shared[(((int)threadIdx.x) + 784)] = data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) & 15) >> 1) * 392)) + ((((int)threadIdx.x) / 28) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + (((int)threadIdx.x) % 28)) + 12544)];
    pad_temp_shared[(((int)threadIdx.x) + 980)] = data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) & 15) >> 1) * 392)) + ((((int)threadIdx.x) / 28) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + (((int)threadIdx.x) % 28)) + 15680)];
    pad_temp_shared[(((int)threadIdx.x) + 1176)] = data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) & 15) >> 1) * 392)) + ((((int)threadIdx.x) / 28) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + (((int)threadIdx.x) % 28)) + 18816)];
    pad_temp_shared[(((int)threadIdx.x) + 1372)] = data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) & 15) >> 1) * 392)) + ((((int)threadIdx.x) / 28) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + (((int)threadIdx.x) % 28)) + 21952)];
    pad_temp_shared[(((int)threadIdx.x) + 1568)] = data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) & 15) >> 1) * 392)) + ((((int)threadIdx.x) / 28) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + (((int)threadIdx.x) % 28)) + 25088)];
    pad_temp_shared[(((int)threadIdx.x) + 1764)] = data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) & 15) >> 1) * 392)) + ((((int)threadIdx.x) / 28) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + (((int)threadIdx.x) % 28)) + 28224)];
    pad_temp_shared[(((int)threadIdx.x) + 1960)] = data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) & 15) >> 1) * 392)) + ((((int)threadIdx.x) / 28) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + (((int)threadIdx.x) % 28)) + 31360)];
    pad_temp_shared[(((int)threadIdx.x) + 2156)] = data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) & 15) >> 1) * 392)) + ((((int)threadIdx.x) / 28) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + (((int)threadIdx.x) % 28)) + 34496)];
    pad_temp_shared[(((int)threadIdx.x) + 2352)] = data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) & 15) >> 1) * 392)) + ((((int)threadIdx.x) / 28) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + (((int)threadIdx.x) % 28)) + 37632)];
    pad_temp_shared[(((int)threadIdx.x) + 2548)] = data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) & 15) >> 1) * 392)) + ((((int)threadIdx.x) / 28) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + (((int)threadIdx.x) % 28)) + 40768)];
    pad_temp_shared[(((int)threadIdx.x) + 2744)] = data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) & 15) >> 1) * 392)) + ((((int)threadIdx.x) / 28) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + (((int)threadIdx.x) % 28)) + 43904)];
    pad_temp_shared[(((int)threadIdx.x) + 2940)] = data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) & 15) >> 1) * 392)) + ((((int)threadIdx.x) / 28) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + (((int)threadIdx.x) % 28)) + 47040)];
    kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) >> 4) * 8192) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15))];
    kernel_shared[(((int)threadIdx.x) + 196)] = kernel[(((((((int)blockIdx.x) >> 4) * 8192) + (((((int)threadIdx.x) + 196) >> 4) * 256)) + (rc_outer_outer * 16)) + ((((int)threadIdx.x) + 4) & 15))];
    if (((int)threadIdx.x) < 120) {
      kernel_shared[(((int)threadIdx.x) + 392)] = kernel[(((((((int)blockIdx.x) >> 4) * 8192) + (((((int)threadIdx.x) + 392) >> 4) * 256)) + (rc_outer_outer * 16)) + ((((int)threadIdx.x) + 8) & 15))];
    }
    __syncthreads();
    for (int ff_c_outer_inner = 0; ff_c_outer_inner < 8; ++ff_c_outer_inner) {
      conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14))] * kernel_shared[(((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32))]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 16)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 16)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[(((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32))]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 16)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 17)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 17)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 16)]));
      conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 196)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 1)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 16)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 16)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 210)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 1)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 196)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 17)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 17)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 17)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 210)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 17)]));
      conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 392)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 2)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 16)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 16)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 406)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 2)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 392)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 18)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 17)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 17)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 406)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 18)]));
      conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 588)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 3)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 16)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 16)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 602)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 3)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 588)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 19)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 17)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 17)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 602)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 19)]));
      conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 784)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 4)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 16)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 16)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 798)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 4)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 784)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 20)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 17)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 17)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 798)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 20)]));
      conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 980)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 5)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 16)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 16)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 994)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 5)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 980)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 21)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 17)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 17)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 994)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 21)]));
      conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 1176)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 6)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 16)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 16)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 1190)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 6)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 1176)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 22)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 17)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 17)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 1190)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 22)]));
      conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 1372)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 7)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 16)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 16)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 1386)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 7)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 1372)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 23)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 17)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 17)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 1386)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 23)]));
      conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 1568)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 8)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 16)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 16)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 1582)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 8)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 1568)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 24)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 17)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 17)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 1582)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 24)]));
      conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 1764)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 9)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 16)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 16)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 1778)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 9)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 1764)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 25)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 17)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 17)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 1778)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 25)]));
      conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 1960)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 10)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 16)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 16)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 1974)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 10)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 1960)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 26)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 17)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 17)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 1974)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 26)]));
      conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 2156)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 11)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 16)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 16)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 2170)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 11)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 2156)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 27)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 17)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 17)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 2170)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 27)]));
      conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 2352)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 12)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 16)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 16)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 2366)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 12)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 2352)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 28)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 17)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 17)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 2366)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 28)]));
      conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 2548)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 13)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 16)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 16)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 2562)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 13)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 2548)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 29)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 17)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 17)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 2562)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 29)]));
      conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 2744)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 14)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 16)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 16)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 2758)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 14)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 2744)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 30)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 17)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 17)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 2758)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 30)]));
      conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 2940)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 15)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 16)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 16)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 2954)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 15)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 2940)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 31)]));
      conv2d_nchw_local[((ff_c_outer_inner * 2) + 17)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 17)] + (pad_temp_shared[(((((((int)threadIdx.x) % 98) / 14) * 28) + (((int)threadIdx.x) % 14)) + 2954)] * kernel_shared[((((((int)threadIdx.x) / 98) * 256) + (ff_c_outer_inner * 32)) + 31)]));
    }
  }
  for (int ff_inner = 0; ff_inner < 16; ++ff_inner) {
    conv2d_nchw[((((((((((int)blockIdx.x) >> 4) * 100352) + ((((int)threadIdx.x) / 98) * 50176)) + (ff_inner * 3136)) + (((((int)blockIdx.x) & 15) >> 1) * 392)) + (((((int)threadIdx.x) % 98) / 14) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + (((int)threadIdx.x) % 14))] = conv2d_nchw_local[ff_inner];
    conv2d_nchw[(((((((((((int)blockIdx.x) >> 4) * 100352) + ((((int)threadIdx.x) / 98) * 50176)) + (ff_inner * 3136)) + (((((int)blockIdx.x) & 15) >> 1) * 392)) + (((((int)threadIdx.x) % 98) / 14) * 56)) + ((((int)blockIdx.x) & 1) * 28)) + (((int)threadIdx.x) % 14)) + 14)] = conv2d_nchw_local[(ff_inner + 16)];
  }
}


