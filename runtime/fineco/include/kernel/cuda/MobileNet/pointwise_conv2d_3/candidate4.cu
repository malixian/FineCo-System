
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
extern "C" __global__ void __launch_bounds__(224) candidate4(float* __restrict__ Input, float* __restrict__ kernel, float* __restrict__ compute) {
  float conv2d_nchw[28];
  __shared__ float pad_temp_shared[6272];
  __shared__ float kernel_shared[4096];
  conv2d_nchw[0] = 0.000000e+00f;
  conv2d_nchw[4] = 0.000000e+00f;
  conv2d_nchw[8] = 0.000000e+00f;
  conv2d_nchw[12] = 0.000000e+00f;
  conv2d_nchw[16] = 0.000000e+00f;
  conv2d_nchw[20] = 0.000000e+00f;
  conv2d_nchw[24] = 0.000000e+00f;
  conv2d_nchw[1] = 0.000000e+00f;
  conv2d_nchw[5] = 0.000000e+00f;
  conv2d_nchw[9] = 0.000000e+00f;
  conv2d_nchw[13] = 0.000000e+00f;
  conv2d_nchw[17] = 0.000000e+00f;
  conv2d_nchw[21] = 0.000000e+00f;
  conv2d_nchw[25] = 0.000000e+00f;
  conv2d_nchw[2] = 0.000000e+00f;
  conv2d_nchw[6] = 0.000000e+00f;
  conv2d_nchw[10] = 0.000000e+00f;
  conv2d_nchw[14] = 0.000000e+00f;
  conv2d_nchw[18] = 0.000000e+00f;
  conv2d_nchw[22] = 0.000000e+00f;
  conv2d_nchw[26] = 0.000000e+00f;
  conv2d_nchw[3] = 0.000000e+00f;
  conv2d_nchw[7] = 0.000000e+00f;
  conv2d_nchw[11] = 0.000000e+00f;
  conv2d_nchw[15] = 0.000000e+00f;
  conv2d_nchw[19] = 0.000000e+00f;
  conv2d_nchw[23] = 0.000000e+00f;
  conv2d_nchw[27] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 2; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = Input[((((((rc_outer_outer * 200704) + ((((int)threadIdx.x) / 98) * 3136)) + (((((int)blockIdx.x) & 31) >> 2) * 392)) + (((((int)threadIdx.x) % 98) / 14) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14))];
    pad_temp_shared[(((int)threadIdx.x) + 224)] = Input[((((((rc_outer_outer * 200704) + (((((int)threadIdx.x) + 224) / 98) * 3136)) + (((((int)blockIdx.x) & 31) >> 2) * 392)) + ((((((int)threadIdx.x) / 14) + 2) % 7) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14))];
    pad_temp_shared[(((int)threadIdx.x) + 448)] = Input[((((((rc_outer_outer * 200704) + (((((int)threadIdx.x) + 448) / 98) * 3136)) + (((((int)blockIdx.x) & 31) >> 2) * 392)) + ((((((int)threadIdx.x) / 14) + 4) % 7) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14))];
    pad_temp_shared[(((int)threadIdx.x) + 672)] = Input[((((((rc_outer_outer * 200704) + (((((int)threadIdx.x) + 672) / 98) * 3136)) + (((((int)blockIdx.x) & 31) >> 2) * 392)) + ((((((int)threadIdx.x) / 14) + 6) % 7) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14))];
    pad_temp_shared[(((int)threadIdx.x) + 896)] = Input[((((((rc_outer_outer * 200704) + (((((int)threadIdx.x) + 896) / 98) * 3136)) + (((((int)blockIdx.x) & 31) >> 2) * 392)) + ((((((int)threadIdx.x) / 14) + 1) % 7) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14))];
    pad_temp_shared[(((int)threadIdx.x) + 1120)] = Input[((((((rc_outer_outer * 200704) + (((((int)threadIdx.x) + 1120) / 98) * 3136)) + (((((int)blockIdx.x) & 31) >> 2) * 392)) + ((((((int)threadIdx.x) / 14) + 3) % 7) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14))];
    pad_temp_shared[(((int)threadIdx.x) + 1344)] = Input[((((((rc_outer_outer * 200704) + (((((int)threadIdx.x) + 1344) / 98) * 3136)) + (((((int)blockIdx.x) & 31) >> 2) * 392)) + ((((((int)threadIdx.x) / 14) + 5) % 7) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14))];
    pad_temp_shared[(((int)threadIdx.x) + 1568)] = Input[(((((((rc_outer_outer * 200704) + ((((int)threadIdx.x) / 98) * 3136)) + (((((int)blockIdx.x) & 31) >> 2) * 392)) + (((((int)threadIdx.x) % 98) / 14) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14)) + 50176)];
    pad_temp_shared[(((int)threadIdx.x) + 1792)] = Input[((((((rc_outer_outer * 200704) + (((((int)threadIdx.x) + 1792) / 98) * 3136)) + (((((int)blockIdx.x) & 31) >> 2) * 392)) + ((((((int)threadIdx.x) / 14) + 2) % 7) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14))];
    pad_temp_shared[(((int)threadIdx.x) + 2016)] = Input[((((((rc_outer_outer * 200704) + (((((int)threadIdx.x) + 2016) / 98) * 3136)) + (((((int)blockIdx.x) & 31) >> 2) * 392)) + ((((((int)threadIdx.x) / 14) + 4) % 7) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14))];
    pad_temp_shared[(((int)threadIdx.x) + 2240)] = Input[((((((rc_outer_outer * 200704) + (((((int)threadIdx.x) + 2240) / 98) * 3136)) + (((((int)blockIdx.x) & 31) >> 2) * 392)) + ((((((int)threadIdx.x) / 14) + 6) % 7) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14))];
    pad_temp_shared[(((int)threadIdx.x) + 2464)] = Input[((((((rc_outer_outer * 200704) + (((((int)threadIdx.x) + 2464) / 98) * 3136)) + (((((int)blockIdx.x) & 31) >> 2) * 392)) + ((((((int)threadIdx.x) / 14) + 1) % 7) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14))];
    pad_temp_shared[(((int)threadIdx.x) + 2688)] = Input[((((((rc_outer_outer * 200704) + (((((int)threadIdx.x) + 2688) / 98) * 3136)) + (((((int)blockIdx.x) & 31) >> 2) * 392)) + ((((((int)threadIdx.x) / 14) + 3) % 7) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14))];
    pad_temp_shared[(((int)threadIdx.x) + 2912)] = Input[((((((rc_outer_outer * 200704) + (((((int)threadIdx.x) + 2912) / 98) * 3136)) + (((((int)blockIdx.x) & 31) >> 2) * 392)) + ((((((int)threadIdx.x) / 14) + 5) % 7) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14))];
    pad_temp_shared[(((int)threadIdx.x) + 3136)] = Input[(((((((rc_outer_outer * 200704) + ((((int)threadIdx.x) / 98) * 3136)) + (((((int)blockIdx.x) & 31) >> 2) * 392)) + (((((int)threadIdx.x) % 98) / 14) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14)) + 100352)];
    pad_temp_shared[(((int)threadIdx.x) + 3360)] = Input[((((((rc_outer_outer * 200704) + (((((int)threadIdx.x) + 3360) / 98) * 3136)) + (((((int)blockIdx.x) & 31) >> 2) * 392)) + ((((((int)threadIdx.x) / 14) + 2) % 7) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14))];
    pad_temp_shared[(((int)threadIdx.x) + 3584)] = Input[((((((rc_outer_outer * 200704) + (((((int)threadIdx.x) + 3584) / 98) * 3136)) + (((((int)blockIdx.x) & 31) >> 2) * 392)) + ((((((int)threadIdx.x) / 14) + 4) % 7) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14))];
    pad_temp_shared[(((int)threadIdx.x) + 3808)] = Input[((((((rc_outer_outer * 200704) + (((((int)threadIdx.x) + 3808) / 98) * 3136)) + (((((int)blockIdx.x) & 31) >> 2) * 392)) + ((((((int)threadIdx.x) / 14) + 6) % 7) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14))];
    pad_temp_shared[(((int)threadIdx.x) + 4032)] = Input[((((((rc_outer_outer * 200704) + (((((int)threadIdx.x) + 4032) / 98) * 3136)) + (((((int)blockIdx.x) & 31) >> 2) * 392)) + ((((((int)threadIdx.x) / 14) + 1) % 7) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14))];
    pad_temp_shared[(((int)threadIdx.x) + 4256)] = Input[((((((rc_outer_outer * 200704) + (((((int)threadIdx.x) + 4256) / 98) * 3136)) + (((((int)blockIdx.x) & 31) >> 2) * 392)) + ((((((int)threadIdx.x) / 14) + 3) % 7) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14))];
    pad_temp_shared[(((int)threadIdx.x) + 4480)] = Input[((((((rc_outer_outer * 200704) + (((((int)threadIdx.x) + 4480) / 98) * 3136)) + (((((int)blockIdx.x) & 31) >> 2) * 392)) + ((((((int)threadIdx.x) / 14) + 5) % 7) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14))];
    pad_temp_shared[(((int)threadIdx.x) + 4704)] = Input[(((((((rc_outer_outer * 200704) + ((((int)threadIdx.x) / 98) * 3136)) + (((((int)blockIdx.x) & 31) >> 2) * 392)) + (((((int)threadIdx.x) % 98) / 14) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14)) + 150528)];
    pad_temp_shared[(((int)threadIdx.x) + 4928)] = Input[((((((rc_outer_outer * 200704) + (((((int)threadIdx.x) + 4928) / 98) * 3136)) + (((((int)blockIdx.x) & 31) >> 2) * 392)) + ((((((int)threadIdx.x) / 14) + 2) % 7) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14))];
    pad_temp_shared[(((int)threadIdx.x) + 5152)] = Input[((((((rc_outer_outer * 200704) + (((((int)threadIdx.x) + 5152) / 98) * 3136)) + (((((int)blockIdx.x) & 31) >> 2) * 392)) + ((((((int)threadIdx.x) / 14) + 4) % 7) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14))];
    pad_temp_shared[(((int)threadIdx.x) + 5376)] = Input[((((((rc_outer_outer * 200704) + (((((int)threadIdx.x) + 5376) / 98) * 3136)) + (((((int)blockIdx.x) & 31) >> 2) * 392)) + ((((((int)threadIdx.x) / 14) + 6) % 7) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14))];
    pad_temp_shared[(((int)threadIdx.x) + 5600)] = Input[((((((rc_outer_outer * 200704) + (((((int)threadIdx.x) + 5600) / 98) * 3136)) + (((((int)blockIdx.x) & 31) >> 2) * 392)) + ((((((int)threadIdx.x) / 14) + 1) % 7) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14))];
    pad_temp_shared[(((int)threadIdx.x) + 5824)] = Input[((((((rc_outer_outer * 200704) + (((((int)threadIdx.x) + 5824) / 98) * 3136)) + (((((int)blockIdx.x) & 31) >> 2) * 392)) + ((((((int)threadIdx.x) / 14) + 3) % 7) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14))];
    pad_temp_shared[(((int)threadIdx.x) + 6048)] = Input[((((((rc_outer_outer * 200704) + (((((int)threadIdx.x) + 6048) / 98) * 3136)) + (((((int)blockIdx.x) & 31) >> 2) * 392)) + ((((((int)threadIdx.x) / 14) + 5) % 7) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14))];
    kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) >> 5) * 8192) + ((((int)threadIdx.x) >> 6) * 128)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63))];
    kernel_shared[(((int)threadIdx.x) + 224)] = kernel[(((((((int)blockIdx.x) >> 5) * 8192) + (((((int)threadIdx.x) + 224) >> 6) * 128)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 32) & 63))];
    kernel_shared[(((int)threadIdx.x) + 448)] = kernel[((((((((int)blockIdx.x) >> 5) * 8192) + ((((int)threadIdx.x) >> 6) * 128)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 896)];
    kernel_shared[(((int)threadIdx.x) + 672)] = kernel[(((((((int)blockIdx.x) >> 5) * 8192) + (((((int)threadIdx.x) + 672) >> 6) * 128)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 32) & 63))];
    kernel_shared[(((int)threadIdx.x) + 896)] = kernel[((((((((int)blockIdx.x) >> 5) * 8192) + ((((int)threadIdx.x) >> 6) * 128)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 1792)];
    kernel_shared[(((int)threadIdx.x) + 1120)] = kernel[(((((((int)blockIdx.x) >> 5) * 8192) + (((((int)threadIdx.x) + 1120) >> 6) * 128)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 32) & 63))];
    kernel_shared[(((int)threadIdx.x) + 1344)] = kernel[((((((((int)blockIdx.x) >> 5) * 8192) + ((((int)threadIdx.x) >> 6) * 128)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 2688)];
    kernel_shared[(((int)threadIdx.x) + 1568)] = kernel[(((((((int)blockIdx.x) >> 5) * 8192) + (((((int)threadIdx.x) + 1568) >> 6) * 128)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 32) & 63))];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[((((((((int)blockIdx.x) >> 5) * 8192) + ((((int)threadIdx.x) >> 6) * 128)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 3584)];
    kernel_shared[(((int)threadIdx.x) + 2016)] = kernel[(((((((int)blockIdx.x) >> 5) * 8192) + (((((int)threadIdx.x) + 2016) >> 6) * 128)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 32) & 63))];
    kernel_shared[(((int)threadIdx.x) + 2240)] = kernel[((((((((int)blockIdx.x) >> 5) * 8192) + ((((int)threadIdx.x) >> 6) * 128)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 4480)];
    kernel_shared[(((int)threadIdx.x) + 2464)] = kernel[(((((((int)blockIdx.x) >> 5) * 8192) + (((((int)threadIdx.x) + 2464) >> 6) * 128)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 32) & 63))];
    kernel_shared[(((int)threadIdx.x) + 2688)] = kernel[((((((((int)blockIdx.x) >> 5) * 8192) + ((((int)threadIdx.x) >> 6) * 128)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 5376)];
    kernel_shared[(((int)threadIdx.x) + 2912)] = kernel[(((((((int)blockIdx.x) >> 5) * 8192) + (((((int)threadIdx.x) + 2912) >> 6) * 128)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 32) & 63))];
    kernel_shared[(((int)threadIdx.x) + 3136)] = kernel[((((((((int)blockIdx.x) >> 5) * 8192) + ((((int)threadIdx.x) >> 6) * 128)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 6272)];
    kernel_shared[(((int)threadIdx.x) + 3360)] = kernel[(((((((int)blockIdx.x) >> 5) * 8192) + (((((int)threadIdx.x) + 3360) >> 6) * 128)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 32) & 63))];
    kernel_shared[(((int)threadIdx.x) + 3584)] = kernel[((((((((int)blockIdx.x) >> 5) * 8192) + ((((int)threadIdx.x) >> 6) * 128)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 7168)];
    kernel_shared[(((int)threadIdx.x) + 3808)] = kernel[(((((((int)blockIdx.x) >> 5) * 8192) + (((((int)threadIdx.x) + 3808) >> 6) * 128)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 32) & 63))];
    if (((int)threadIdx.x) < 64) {
      kernel_shared[(((int)threadIdx.x) + 4032)] = kernel[(((((((int)blockIdx.x) >> 5) * 8192) + (rc_outer_outer * 64)) + ((int)threadIdx.x)) + 8064)];
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 64; ++rc_inner) {
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((rc_inner * 98) + (((int)threadIdx.x) % 14))] * kernel_shared[(((((int)threadIdx.x) / 14) * 256) + rc_inner)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[(((rc_inner * 98) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[(((((int)threadIdx.x) / 14) * 256) + rc_inner)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[(((rc_inner * 98) + (((int)threadIdx.x) % 14)) + 28)] * kernel_shared[(((((int)threadIdx.x) / 14) * 256) + rc_inner)]));
      conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[(((rc_inner * 98) + (((int)threadIdx.x) % 14)) + 42)] * kernel_shared[(((((int)threadIdx.x) / 14) * 256) + rc_inner)]));
      conv2d_nchw[16] = (conv2d_nchw[16] + (pad_temp_shared[(((rc_inner * 98) + (((int)threadIdx.x) % 14)) + 56)] * kernel_shared[(((((int)threadIdx.x) / 14) * 256) + rc_inner)]));
      conv2d_nchw[20] = (conv2d_nchw[20] + (pad_temp_shared[(((rc_inner * 98) + (((int)threadIdx.x) % 14)) + 70)] * kernel_shared[(((((int)threadIdx.x) / 14) * 256) + rc_inner)]));
      conv2d_nchw[24] = (conv2d_nchw[24] + (pad_temp_shared[(((rc_inner * 98) + (((int)threadIdx.x) % 14)) + 84)] * kernel_shared[(((((int)threadIdx.x) / 14) * 256) + rc_inner)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((rc_inner * 98) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + rc_inner) + 64)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[(((rc_inner * 98) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + rc_inner) + 64)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[(((rc_inner * 98) + (((int)threadIdx.x) % 14)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + rc_inner) + 64)]));
      conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[(((rc_inner * 98) + (((int)threadIdx.x) % 14)) + 42)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + rc_inner) + 64)]));
      conv2d_nchw[17] = (conv2d_nchw[17] + (pad_temp_shared[(((rc_inner * 98) + (((int)threadIdx.x) % 14)) + 56)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + rc_inner) + 64)]));
      conv2d_nchw[21] = (conv2d_nchw[21] + (pad_temp_shared[(((rc_inner * 98) + (((int)threadIdx.x) % 14)) + 70)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + rc_inner) + 64)]));
      conv2d_nchw[25] = (conv2d_nchw[25] + (pad_temp_shared[(((rc_inner * 98) + (((int)threadIdx.x) % 14)) + 84)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + rc_inner) + 64)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((rc_inner * 98) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + rc_inner) + 128)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[(((rc_inner * 98) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + rc_inner) + 128)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[(((rc_inner * 98) + (((int)threadIdx.x) % 14)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + rc_inner) + 128)]));
      conv2d_nchw[14] = (conv2d_nchw[14] + (pad_temp_shared[(((rc_inner * 98) + (((int)threadIdx.x) % 14)) + 42)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + rc_inner) + 128)]));
      conv2d_nchw[18] = (conv2d_nchw[18] + (pad_temp_shared[(((rc_inner * 98) + (((int)threadIdx.x) % 14)) + 56)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + rc_inner) + 128)]));
      conv2d_nchw[22] = (conv2d_nchw[22] + (pad_temp_shared[(((rc_inner * 98) + (((int)threadIdx.x) % 14)) + 70)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + rc_inner) + 128)]));
      conv2d_nchw[26] = (conv2d_nchw[26] + (pad_temp_shared[(((rc_inner * 98) + (((int)threadIdx.x) % 14)) + 84)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + rc_inner) + 128)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((rc_inner * 98) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + rc_inner) + 192)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[(((rc_inner * 98) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + rc_inner) + 192)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[(((rc_inner * 98) + (((int)threadIdx.x) % 14)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + rc_inner) + 192)]));
      conv2d_nchw[15] = (conv2d_nchw[15] + (pad_temp_shared[(((rc_inner * 98) + (((int)threadIdx.x) % 14)) + 42)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + rc_inner) + 192)]));
      conv2d_nchw[19] = (conv2d_nchw[19] + (pad_temp_shared[(((rc_inner * 98) + (((int)threadIdx.x) % 14)) + 56)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + rc_inner) + 192)]));
      conv2d_nchw[23] = (conv2d_nchw[23] + (pad_temp_shared[(((rc_inner * 98) + (((int)threadIdx.x) % 14)) + 70)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + rc_inner) + 192)]));
      conv2d_nchw[27] = (conv2d_nchw[27] + (pad_temp_shared[(((rc_inner * 98) + (((int)threadIdx.x) % 14)) + 84)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + rc_inner) + 192)]));
    }
  }
  for (int i1_inner = 0; i1_inner < 4; ++i1_inner) {
    compute[(((((((((int)blockIdx.x) >> 5) * 200704) + ((((int)threadIdx.x) / 14) * 12544)) + (i1_inner * 3136)) + (((((int)blockIdx.x) & 31) >> 2) * 392)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14))] = max(conv2d_nchw[i1_inner], 0.000000e+00f);
    compute[((((((((((int)blockIdx.x) >> 5) * 200704) + ((((int)threadIdx.x) / 14) * 12544)) + (i1_inner * 3136)) + (((((int)blockIdx.x) & 31) >> 2) * 392)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14)) + 56)] = max(conv2d_nchw[(i1_inner + 4)], 0.000000e+00f);
    compute[((((((((((int)blockIdx.x) >> 5) * 200704) + ((((int)threadIdx.x) / 14) * 12544)) + (i1_inner * 3136)) + (((((int)blockIdx.x) & 31) >> 2) * 392)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14)) + 112)] = max(conv2d_nchw[(i1_inner + 8)], 0.000000e+00f);
    compute[((((((((((int)blockIdx.x) >> 5) * 200704) + ((((int)threadIdx.x) / 14) * 12544)) + (i1_inner * 3136)) + (((((int)blockIdx.x) & 31) >> 2) * 392)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14)) + 168)] = max(conv2d_nchw[(i1_inner + 12)], 0.000000e+00f);
    compute[((((((((((int)blockIdx.x) >> 5) * 200704) + ((((int)threadIdx.x) / 14) * 12544)) + (i1_inner * 3136)) + (((((int)blockIdx.x) & 31) >> 2) * 392)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14)) + 224)] = max(conv2d_nchw[(i1_inner + 16)], 0.000000e+00f);
    compute[((((((((((int)blockIdx.x) >> 5) * 200704) + ((((int)threadIdx.x) / 14) * 12544)) + (i1_inner * 3136)) + (((((int)blockIdx.x) & 31) >> 2) * 392)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14)) + 280)] = max(conv2d_nchw[(i1_inner + 20)], 0.000000e+00f);
    compute[((((((((((int)blockIdx.x) >> 5) * 200704) + ((((int)threadIdx.x) / 14) * 12544)) + (i1_inner * 3136)) + (((((int)blockIdx.x) & 31) >> 2) * 392)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 14)) + 336)] = max(conv2d_nchw[(i1_inner + 24)], 0.000000e+00f);
  }
}


