
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
extern "C" __global__ void __launch_bounds__(224) candidate3(float* __restrict__ Input, float* __restrict__ kernel, float* __restrict__ compute) {
  float conv2d_nchw[14];
  __shared__ float pad_temp_shared[6272];
  __shared__ float kernel_shared[2048];
  conv2d_nchw[0] = 0.000000e+00f;
  conv2d_nchw[1] = 0.000000e+00f;
  conv2d_nchw[2] = 0.000000e+00f;
  conv2d_nchw[3] = 0.000000e+00f;
  conv2d_nchw[4] = 0.000000e+00f;
  conv2d_nchw[5] = 0.000000e+00f;
  conv2d_nchw[6] = 0.000000e+00f;
  conv2d_nchw[7] = 0.000000e+00f;
  conv2d_nchw[8] = 0.000000e+00f;
  conv2d_nchw[9] = 0.000000e+00f;
  conv2d_nchw[10] = 0.000000e+00f;
  conv2d_nchw[11] = 0.000000e+00f;
  conv2d_nchw[12] = 0.000000e+00f;
  conv2d_nchw[13] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 8; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = Input[((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 7) * 14)) + ((((int)blockIdx.x) & 1) * 7)) + (((int)threadIdx.x) % 7))];
    pad_temp_shared[(((int)threadIdx.x) + 224)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 7) * 14)) + ((((int)blockIdx.x) & 1) * 7)) + (((int)threadIdx.x) % 7)) + 448)];
    pad_temp_shared[(((int)threadIdx.x) + 448)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 7) * 14)) + ((((int)blockIdx.x) & 1) * 7)) + (((int)threadIdx.x) % 7)) + 896)];
    pad_temp_shared[(((int)threadIdx.x) + 672)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 7) * 14)) + ((((int)blockIdx.x) & 1) * 7)) + (((int)threadIdx.x) % 7)) + 1344)];
    pad_temp_shared[(((int)threadIdx.x) + 896)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 7) * 14)) + ((((int)blockIdx.x) & 1) * 7)) + (((int)threadIdx.x) % 7)) + 1792)];
    pad_temp_shared[(((int)threadIdx.x) + 1120)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 7) * 14)) + ((((int)blockIdx.x) & 1) * 7)) + (((int)threadIdx.x) % 7)) + 2240)];
    pad_temp_shared[(((int)threadIdx.x) + 1344)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 7) * 14)) + ((((int)blockIdx.x) & 1) * 7)) + (((int)threadIdx.x) % 7)) + 2688)];
    pad_temp_shared[(((int)threadIdx.x) + 1568)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 7) * 14)) + ((((int)blockIdx.x) & 1) * 7)) + (((int)threadIdx.x) % 7)) + 3136)];
    pad_temp_shared[(((int)threadIdx.x) + 1792)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 7) * 14)) + ((((int)blockIdx.x) & 1) * 7)) + (((int)threadIdx.x) % 7)) + 3584)];
    pad_temp_shared[(((int)threadIdx.x) + 2016)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 7) * 14)) + ((((int)blockIdx.x) & 1) * 7)) + (((int)threadIdx.x) % 7)) + 4032)];
    pad_temp_shared[(((int)threadIdx.x) + 2240)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 7) * 14)) + ((((int)blockIdx.x) & 1) * 7)) + (((int)threadIdx.x) % 7)) + 4480)];
    pad_temp_shared[(((int)threadIdx.x) + 2464)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 7) * 14)) + ((((int)blockIdx.x) & 1) * 7)) + (((int)threadIdx.x) % 7)) + 4928)];
    pad_temp_shared[(((int)threadIdx.x) + 2688)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 7) * 14)) + ((((int)blockIdx.x) & 1) * 7)) + (((int)threadIdx.x) % 7)) + 5376)];
    pad_temp_shared[(((int)threadIdx.x) + 2912)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 7) * 14)) + ((((int)blockIdx.x) & 1) * 7)) + (((int)threadIdx.x) % 7)) + 5824)];
    pad_temp_shared[(((int)threadIdx.x) + 3136)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 7) * 14)) + ((((int)blockIdx.x) & 1) * 7)) + (((int)threadIdx.x) % 7)) + 6272)];
    pad_temp_shared[(((int)threadIdx.x) + 3360)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 7) * 14)) + ((((int)blockIdx.x) & 1) * 7)) + (((int)threadIdx.x) % 7)) + 6720)];
    pad_temp_shared[(((int)threadIdx.x) + 3584)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 7) * 14)) + ((((int)blockIdx.x) & 1) * 7)) + (((int)threadIdx.x) % 7)) + 7168)];
    pad_temp_shared[(((int)threadIdx.x) + 3808)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 7) * 14)) + ((((int)blockIdx.x) & 1) * 7)) + (((int)threadIdx.x) % 7)) + 7616)];
    pad_temp_shared[(((int)threadIdx.x) + 4032)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 7) * 14)) + ((((int)blockIdx.x) & 1) * 7)) + (((int)threadIdx.x) % 7)) + 8064)];
    pad_temp_shared[(((int)threadIdx.x) + 4256)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 7) * 14)) + ((((int)blockIdx.x) & 1) * 7)) + (((int)threadIdx.x) % 7)) + 8512)];
    pad_temp_shared[(((int)threadIdx.x) + 4480)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 7) * 14)) + ((((int)blockIdx.x) & 1) * 7)) + (((int)threadIdx.x) % 7)) + 8960)];
    pad_temp_shared[(((int)threadIdx.x) + 4704)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 7) * 14)) + ((((int)blockIdx.x) & 1) * 7)) + (((int)threadIdx.x) % 7)) + 9408)];
    pad_temp_shared[(((int)threadIdx.x) + 4928)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 7) * 14)) + ((((int)blockIdx.x) & 1) * 7)) + (((int)threadIdx.x) % 7)) + 9856)];
    pad_temp_shared[(((int)threadIdx.x) + 5152)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 7) * 14)) + ((((int)blockIdx.x) & 1) * 7)) + (((int)threadIdx.x) % 7)) + 10304)];
    pad_temp_shared[(((int)threadIdx.x) + 5376)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 7) * 14)) + ((((int)blockIdx.x) & 1) * 7)) + (((int)threadIdx.x) % 7)) + 10752)];
    pad_temp_shared[(((int)threadIdx.x) + 5600)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 7) * 14)) + ((((int)blockIdx.x) & 1) * 7)) + (((int)threadIdx.x) % 7)) + 11200)];
    pad_temp_shared[(((int)threadIdx.x) + 5824)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 7) * 14)) + ((((int)blockIdx.x) & 1) * 7)) + (((int)threadIdx.x) % 7)) + 11648)];
    pad_temp_shared[(((int)threadIdx.x) + 6048)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 7) * 14)) + ((((int)blockIdx.x) & 1) * 7)) + (((int)threadIdx.x) % 7)) + 12096)];
    kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) >> 1) * 16384) + ((((int)threadIdx.x) >> 6) * 512)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63))];
    kernel_shared[(((int)threadIdx.x) + 224)] = kernel[(((((((int)blockIdx.x) >> 1) * 16384) + (((((int)threadIdx.x) + 224) >> 6) * 512)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 32) & 63))];
    kernel_shared[(((int)threadIdx.x) + 448)] = kernel[((((((((int)blockIdx.x) >> 1) * 16384) + ((((int)threadIdx.x) >> 6) * 512)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 3584)];
    kernel_shared[(((int)threadIdx.x) + 672)] = kernel[(((((((int)blockIdx.x) >> 1) * 16384) + (((((int)threadIdx.x) + 672) >> 6) * 512)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 32) & 63))];
    kernel_shared[(((int)threadIdx.x) + 896)] = kernel[((((((((int)blockIdx.x) >> 1) * 16384) + ((((int)threadIdx.x) >> 6) * 512)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 7168)];
    kernel_shared[(((int)threadIdx.x) + 1120)] = kernel[(((((((int)blockIdx.x) >> 1) * 16384) + (((((int)threadIdx.x) + 1120) >> 6) * 512)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 32) & 63))];
    kernel_shared[(((int)threadIdx.x) + 1344)] = kernel[((((((((int)blockIdx.x) >> 1) * 16384) + ((((int)threadIdx.x) >> 6) * 512)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 10752)];
    kernel_shared[(((int)threadIdx.x) + 1568)] = kernel[(((((((int)blockIdx.x) >> 1) * 16384) + (((((int)threadIdx.x) + 1568) >> 6) * 512)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 32) & 63))];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[((((((((int)blockIdx.x) >> 1) * 16384) + ((((int)threadIdx.x) >> 6) * 512)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 14336)];
    if (((int)threadIdx.x) < 32) {
      kernel_shared[(((int)threadIdx.x) + 2016)] = kernel[(((((((int)blockIdx.x) >> 1) * 16384) + (rc_outer_outer * 64)) + (((int)threadIdx.x) + 32)) + 15872)];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 8; ++rc_outer_inner) {
      for (int ff_outer_inner = 0; ff_outer_inner < 2; ++ff_outer_inner) {
        conv2d_nchw[(ff_outer_inner * 7)] = (conv2d_nchw[(ff_outer_inner * 7)] + (pad_temp_shared[(((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7))] * kernel_shared[((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8))]));
        conv2d_nchw[(ff_outer_inner * 7)] = (conv2d_nchw[(ff_outer_inner * 7)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 98)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 1)]));
        conv2d_nchw[(ff_outer_inner * 7)] = (conv2d_nchw[(ff_outer_inner * 7)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 196)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 2)]));
        conv2d_nchw[(ff_outer_inner * 7)] = (conv2d_nchw[(ff_outer_inner * 7)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 294)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 3)]));
        conv2d_nchw[(ff_outer_inner * 7)] = (conv2d_nchw[(ff_outer_inner * 7)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 392)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 4)]));
        conv2d_nchw[(ff_outer_inner * 7)] = (conv2d_nchw[(ff_outer_inner * 7)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 490)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 5)]));
        conv2d_nchw[(ff_outer_inner * 7)] = (conv2d_nchw[(ff_outer_inner * 7)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 588)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 6)]));
        conv2d_nchw[(ff_outer_inner * 7)] = (conv2d_nchw[(ff_outer_inner * 7)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 686)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 7)]));
        conv2d_nchw[((ff_outer_inner * 7) + 1)] = (conv2d_nchw[((ff_outer_inner * 7) + 1)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 7)] * kernel_shared[((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8))]));
        conv2d_nchw[((ff_outer_inner * 7) + 1)] = (conv2d_nchw[((ff_outer_inner * 7) + 1)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 105)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 1)]));
        conv2d_nchw[((ff_outer_inner * 7) + 1)] = (conv2d_nchw[((ff_outer_inner * 7) + 1)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 203)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 2)]));
        conv2d_nchw[((ff_outer_inner * 7) + 1)] = (conv2d_nchw[((ff_outer_inner * 7) + 1)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 301)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 3)]));
        conv2d_nchw[((ff_outer_inner * 7) + 1)] = (conv2d_nchw[((ff_outer_inner * 7) + 1)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 399)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 4)]));
        conv2d_nchw[((ff_outer_inner * 7) + 1)] = (conv2d_nchw[((ff_outer_inner * 7) + 1)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 497)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 5)]));
        conv2d_nchw[((ff_outer_inner * 7) + 1)] = (conv2d_nchw[((ff_outer_inner * 7) + 1)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 595)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 6)]));
        conv2d_nchw[((ff_outer_inner * 7) + 1)] = (conv2d_nchw[((ff_outer_inner * 7) + 1)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 693)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 7)]));
        conv2d_nchw[((ff_outer_inner * 7) + 2)] = (conv2d_nchw[((ff_outer_inner * 7) + 2)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8))]));
        conv2d_nchw[((ff_outer_inner * 7) + 2)] = (conv2d_nchw[((ff_outer_inner * 7) + 2)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 112)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 1)]));
        conv2d_nchw[((ff_outer_inner * 7) + 2)] = (conv2d_nchw[((ff_outer_inner * 7) + 2)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 210)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 2)]));
        conv2d_nchw[((ff_outer_inner * 7) + 2)] = (conv2d_nchw[((ff_outer_inner * 7) + 2)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 308)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 3)]));
        conv2d_nchw[((ff_outer_inner * 7) + 2)] = (conv2d_nchw[((ff_outer_inner * 7) + 2)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 406)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 4)]));
        conv2d_nchw[((ff_outer_inner * 7) + 2)] = (conv2d_nchw[((ff_outer_inner * 7) + 2)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 504)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 5)]));
        conv2d_nchw[((ff_outer_inner * 7) + 2)] = (conv2d_nchw[((ff_outer_inner * 7) + 2)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 602)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 6)]));
        conv2d_nchw[((ff_outer_inner * 7) + 2)] = (conv2d_nchw[((ff_outer_inner * 7) + 2)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 700)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 7)]));
        conv2d_nchw[((ff_outer_inner * 7) + 3)] = (conv2d_nchw[((ff_outer_inner * 7) + 3)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 21)] * kernel_shared[((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8))]));
        conv2d_nchw[((ff_outer_inner * 7) + 3)] = (conv2d_nchw[((ff_outer_inner * 7) + 3)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 119)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 1)]));
        conv2d_nchw[((ff_outer_inner * 7) + 3)] = (conv2d_nchw[((ff_outer_inner * 7) + 3)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 217)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 2)]));
        conv2d_nchw[((ff_outer_inner * 7) + 3)] = (conv2d_nchw[((ff_outer_inner * 7) + 3)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 315)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 3)]));
        conv2d_nchw[((ff_outer_inner * 7) + 3)] = (conv2d_nchw[((ff_outer_inner * 7) + 3)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 413)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 4)]));
        conv2d_nchw[((ff_outer_inner * 7) + 3)] = (conv2d_nchw[((ff_outer_inner * 7) + 3)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 511)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 5)]));
        conv2d_nchw[((ff_outer_inner * 7) + 3)] = (conv2d_nchw[((ff_outer_inner * 7) + 3)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 609)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 6)]));
        conv2d_nchw[((ff_outer_inner * 7) + 3)] = (conv2d_nchw[((ff_outer_inner * 7) + 3)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 707)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 7)]));
        conv2d_nchw[((ff_outer_inner * 7) + 4)] = (conv2d_nchw[((ff_outer_inner * 7) + 4)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8))]));
        conv2d_nchw[((ff_outer_inner * 7) + 4)] = (conv2d_nchw[((ff_outer_inner * 7) + 4)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 126)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 1)]));
        conv2d_nchw[((ff_outer_inner * 7) + 4)] = (conv2d_nchw[((ff_outer_inner * 7) + 4)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 224)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 2)]));
        conv2d_nchw[((ff_outer_inner * 7) + 4)] = (conv2d_nchw[((ff_outer_inner * 7) + 4)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 322)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 3)]));
        conv2d_nchw[((ff_outer_inner * 7) + 4)] = (conv2d_nchw[((ff_outer_inner * 7) + 4)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 420)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 4)]));
        conv2d_nchw[((ff_outer_inner * 7) + 4)] = (conv2d_nchw[((ff_outer_inner * 7) + 4)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 518)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 5)]));
        conv2d_nchw[((ff_outer_inner * 7) + 4)] = (conv2d_nchw[((ff_outer_inner * 7) + 4)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 616)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 6)]));
        conv2d_nchw[((ff_outer_inner * 7) + 4)] = (conv2d_nchw[((ff_outer_inner * 7) + 4)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 714)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 7)]));
        conv2d_nchw[((ff_outer_inner * 7) + 5)] = (conv2d_nchw[((ff_outer_inner * 7) + 5)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 35)] * kernel_shared[((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8))]));
        conv2d_nchw[((ff_outer_inner * 7) + 5)] = (conv2d_nchw[((ff_outer_inner * 7) + 5)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 133)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 1)]));
        conv2d_nchw[((ff_outer_inner * 7) + 5)] = (conv2d_nchw[((ff_outer_inner * 7) + 5)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 231)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 2)]));
        conv2d_nchw[((ff_outer_inner * 7) + 5)] = (conv2d_nchw[((ff_outer_inner * 7) + 5)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 329)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 3)]));
        conv2d_nchw[((ff_outer_inner * 7) + 5)] = (conv2d_nchw[((ff_outer_inner * 7) + 5)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 427)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 4)]));
        conv2d_nchw[((ff_outer_inner * 7) + 5)] = (conv2d_nchw[((ff_outer_inner * 7) + 5)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 525)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 5)]));
        conv2d_nchw[((ff_outer_inner * 7) + 5)] = (conv2d_nchw[((ff_outer_inner * 7) + 5)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 623)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 6)]));
        conv2d_nchw[((ff_outer_inner * 7) + 5)] = (conv2d_nchw[((ff_outer_inner * 7) + 5)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 721)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 7)]));
        conv2d_nchw[((ff_outer_inner * 7) + 6)] = (conv2d_nchw[((ff_outer_inner * 7) + 6)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 42)] * kernel_shared[((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8))]));
        conv2d_nchw[((ff_outer_inner * 7) + 6)] = (conv2d_nchw[((ff_outer_inner * 7) + 6)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 140)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 1)]));
        conv2d_nchw[((ff_outer_inner * 7) + 6)] = (conv2d_nchw[((ff_outer_inner * 7) + 6)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 238)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 2)]));
        conv2d_nchw[((ff_outer_inner * 7) + 6)] = (conv2d_nchw[((ff_outer_inner * 7) + 6)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 336)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 3)]));
        conv2d_nchw[((ff_outer_inner * 7) + 6)] = (conv2d_nchw[((ff_outer_inner * 7) + 6)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 434)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 4)]));
        conv2d_nchw[((ff_outer_inner * 7) + 6)] = (conv2d_nchw[((ff_outer_inner * 7) + 6)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 532)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 5)]));
        conv2d_nchw[((ff_outer_inner * 7) + 6)] = (conv2d_nchw[((ff_outer_inner * 7) + 6)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 630)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 6)]));
        conv2d_nchw[((ff_outer_inner * 7) + 6)] = (conv2d_nchw[((ff_outer_inner * 7) + 6)] + (pad_temp_shared[((((rc_outer_inner * 784) + (((((int)threadIdx.x) % 14) / 7) * 49)) + (((int)threadIdx.x) % 7)) + 728)] * kernel_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 64)) + (rc_outer_inner * 8)) + 7)]));
      }
    }
  }
  for (int i1_inner = 0; i1_inner < 2; ++i1_inner) {
    for (int i2_inner = 0; i2_inner < 7; ++i2_inner) {
      compute[((((((((((int)blockIdx.x) >> 1) * 6272) + ((((int)threadIdx.x) / 14) * 392)) + (i1_inner * 196)) + (((((int)threadIdx.x) % 14) / 7) * 98)) + (i2_inner * 14)) + ((((int)blockIdx.x) & 1) * 7)) + (((int)threadIdx.x) % 7))] = max(conv2d_nchw[((i1_inner * 7) + i2_inner)], 0.000000e+00f);
    }
  }
}


