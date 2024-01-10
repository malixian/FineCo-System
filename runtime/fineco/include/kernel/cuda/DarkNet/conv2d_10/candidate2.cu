
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
extern "C" __global__ void __launch_bounds__(224) candidate2(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ compute, float* __restrict__ bias) {
  float conv2d_nchw[8];
  __shared__ float pad_temp_shared[3584];
  __shared__ float kernel_shared[8192];
  conv2d_nchw[0] = 0.000000e+00f;
  conv2d_nchw[2] = 0.000000e+00f;
  conv2d_nchw[4] = 0.000000e+00f;
  conv2d_nchw[6] = 0.000000e+00f;
  conv2d_nchw[1] = 0.000000e+00f;
  conv2d_nchw[3] = 0.000000e+00f;
  conv2d_nchw[5] = 0.000000e+00f;
  conv2d_nchw[7] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 4; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = data[((((rc_outer_outer * 25088) + ((((int)threadIdx.x) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1))];
    pad_temp_shared[(((int)threadIdx.x) + 224)] = data[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1)) + 1568)];
    pad_temp_shared[(((int)threadIdx.x) + 448)] = data[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1)) + 3136)];
    pad_temp_shared[(((int)threadIdx.x) + 672)] = data[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1)) + 4704)];
    pad_temp_shared[(((int)threadIdx.x) + 896)] = data[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1)) + 6272)];
    pad_temp_shared[(((int)threadIdx.x) + 1120)] = data[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1)) + 7840)];
    pad_temp_shared[(((int)threadIdx.x) + 1344)] = data[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1)) + 9408)];
    pad_temp_shared[(((int)threadIdx.x) + 1568)] = data[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1)) + 10976)];
    pad_temp_shared[(((int)threadIdx.x) + 1792)] = data[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1)) + 12544)];
    pad_temp_shared[(((int)threadIdx.x) + 2016)] = data[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1)) + 14112)];
    pad_temp_shared[(((int)threadIdx.x) + 2240)] = data[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1)) + 15680)];
    pad_temp_shared[(((int)threadIdx.x) + 2464)] = data[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1)) + 17248)];
    pad_temp_shared[(((int)threadIdx.x) + 2688)] = data[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1)) + 18816)];
    pad_temp_shared[(((int)threadIdx.x) + 2912)] = data[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1)) + 20384)];
    pad_temp_shared[(((int)threadIdx.x) + 3136)] = data[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1)) + 21952)];
    pad_temp_shared[(((int)threadIdx.x) + 3360)] = data[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1)) + 23520)];
    kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) / 7) * 32768) + ((((int)threadIdx.x) >> 7) * 512)) + (rc_outer_outer * 128)) + (((int)threadIdx.x) & 127))];
    kernel_shared[(((int)threadIdx.x) + 224)] = kernel[(((((((int)blockIdx.x) / 7) * 32768) + (((((int)threadIdx.x) + 224) >> 7) * 512)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 96) & 127))];
    kernel_shared[(((int)threadIdx.x) + 448)] = kernel[(((((((int)blockIdx.x) / 7) * 32768) + (((((int)threadIdx.x) + 448) >> 7) * 512)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 64) & 127))];
    kernel_shared[(((int)threadIdx.x) + 672)] = kernel[(((((((int)blockIdx.x) / 7) * 32768) + (((((int)threadIdx.x) + 672) >> 7) * 512)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 32) & 127))];
    kernel_shared[(((int)threadIdx.x) + 896)] = kernel[((((((((int)blockIdx.x) / 7) * 32768) + ((((int)threadIdx.x) >> 7) * 512)) + (rc_outer_outer * 128)) + (((int)threadIdx.x) & 127)) + 3584)];
    kernel_shared[(((int)threadIdx.x) + 1120)] = kernel[(((((((int)blockIdx.x) / 7) * 32768) + (((((int)threadIdx.x) + 1120) >> 7) * 512)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 96) & 127))];
    kernel_shared[(((int)threadIdx.x) + 1344)] = kernel[(((((((int)blockIdx.x) / 7) * 32768) + (((((int)threadIdx.x) + 1344) >> 7) * 512)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 64) & 127))];
    kernel_shared[(((int)threadIdx.x) + 1568)] = kernel[(((((((int)blockIdx.x) / 7) * 32768) + (((((int)threadIdx.x) + 1568) >> 7) * 512)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 32) & 127))];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[((((((((int)blockIdx.x) / 7) * 32768) + ((((int)threadIdx.x) >> 7) * 512)) + (rc_outer_outer * 128)) + (((int)threadIdx.x) & 127)) + 7168)];
    kernel_shared[(((int)threadIdx.x) + 2016)] = kernel[(((((((int)blockIdx.x) / 7) * 32768) + (((((int)threadIdx.x) + 2016) >> 7) * 512)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 96) & 127))];
    kernel_shared[(((int)threadIdx.x) + 2240)] = kernel[(((((((int)blockIdx.x) / 7) * 32768) + (((((int)threadIdx.x) + 2240) >> 7) * 512)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 64) & 127))];
    kernel_shared[(((int)threadIdx.x) + 2464)] = kernel[(((((((int)blockIdx.x) / 7) * 32768) + (((((int)threadIdx.x) + 2464) >> 7) * 512)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 32) & 127))];
    kernel_shared[(((int)threadIdx.x) + 2688)] = kernel[((((((((int)blockIdx.x) / 7) * 32768) + ((((int)threadIdx.x) >> 7) * 512)) + (rc_outer_outer * 128)) + (((int)threadIdx.x) & 127)) + 10752)];
    kernel_shared[(((int)threadIdx.x) + 2912)] = kernel[(((((((int)blockIdx.x) / 7) * 32768) + (((((int)threadIdx.x) + 2912) >> 7) * 512)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 96) & 127))];
    kernel_shared[(((int)threadIdx.x) + 3136)] = kernel[(((((((int)blockIdx.x) / 7) * 32768) + (((((int)threadIdx.x) + 3136) >> 7) * 512)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 64) & 127))];
    kernel_shared[(((int)threadIdx.x) + 3360)] = kernel[(((((((int)blockIdx.x) / 7) * 32768) + (((((int)threadIdx.x) + 3360) >> 7) * 512)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 32) & 127))];
    kernel_shared[(((int)threadIdx.x) + 3584)] = kernel[((((((((int)blockIdx.x) / 7) * 32768) + ((((int)threadIdx.x) >> 7) * 512)) + (rc_outer_outer * 128)) + (((int)threadIdx.x) & 127)) + 14336)];
    kernel_shared[(((int)threadIdx.x) + 3808)] = kernel[(((((((int)blockIdx.x) / 7) * 32768) + (((((int)threadIdx.x) + 3808) >> 7) * 512)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 96) & 127))];
    kernel_shared[(((int)threadIdx.x) + 4032)] = kernel[(((((((int)blockIdx.x) / 7) * 32768) + (((((int)threadIdx.x) + 4032) >> 7) * 512)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 64) & 127))];
    kernel_shared[(((int)threadIdx.x) + 4256)] = kernel[(((((((int)blockIdx.x) / 7) * 32768) + (((((int)threadIdx.x) + 4256) >> 7) * 512)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 32) & 127))];
    kernel_shared[(((int)threadIdx.x) + 4480)] = kernel[((((((((int)blockIdx.x) / 7) * 32768) + ((((int)threadIdx.x) >> 7) * 512)) + (rc_outer_outer * 128)) + (((int)threadIdx.x) & 127)) + 17920)];
    kernel_shared[(((int)threadIdx.x) + 4704)] = kernel[(((((((int)blockIdx.x) / 7) * 32768) + (((((int)threadIdx.x) + 4704) >> 7) * 512)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 96) & 127))];
    kernel_shared[(((int)threadIdx.x) + 4928)] = kernel[(((((((int)blockIdx.x) / 7) * 32768) + (((((int)threadIdx.x) + 4928) >> 7) * 512)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 64) & 127))];
    kernel_shared[(((int)threadIdx.x) + 5152)] = kernel[(((((((int)blockIdx.x) / 7) * 32768) + (((((int)threadIdx.x) + 5152) >> 7) * 512)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 32) & 127))];
    kernel_shared[(((int)threadIdx.x) + 5376)] = kernel[((((((((int)blockIdx.x) / 7) * 32768) + ((((int)threadIdx.x) >> 7) * 512)) + (rc_outer_outer * 128)) + (((int)threadIdx.x) & 127)) + 21504)];
    kernel_shared[(((int)threadIdx.x) + 5600)] = kernel[(((((((int)blockIdx.x) / 7) * 32768) + (((((int)threadIdx.x) + 5600) >> 7) * 512)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 96) & 127))];
    kernel_shared[(((int)threadIdx.x) + 5824)] = kernel[(((((((int)blockIdx.x) / 7) * 32768) + (((((int)threadIdx.x) + 5824) >> 7) * 512)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 64) & 127))];
    kernel_shared[(((int)threadIdx.x) + 6048)] = kernel[(((((((int)blockIdx.x) / 7) * 32768) + (((((int)threadIdx.x) + 6048) >> 7) * 512)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 32) & 127))];
    kernel_shared[(((int)threadIdx.x) + 6272)] = kernel[((((((((int)blockIdx.x) / 7) * 32768) + ((((int)threadIdx.x) >> 7) * 512)) + (rc_outer_outer * 128)) + (((int)threadIdx.x) & 127)) + 25088)];
    kernel_shared[(((int)threadIdx.x) + 6496)] = kernel[(((((((int)blockIdx.x) / 7) * 32768) + (((((int)threadIdx.x) + 6496) >> 7) * 512)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 96) & 127))];
    kernel_shared[(((int)threadIdx.x) + 6720)] = kernel[(((((((int)blockIdx.x) / 7) * 32768) + (((((int)threadIdx.x) + 6720) >> 7) * 512)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 64) & 127))];
    kernel_shared[(((int)threadIdx.x) + 6944)] = kernel[(((((((int)blockIdx.x) / 7) * 32768) + (((((int)threadIdx.x) + 6944) >> 7) * 512)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 32) & 127))];
    kernel_shared[(((int)threadIdx.x) + 7168)] = kernel[((((((((int)blockIdx.x) / 7) * 32768) + ((((int)threadIdx.x) >> 7) * 512)) + (rc_outer_outer * 128)) + (((int)threadIdx.x) & 127)) + 28672)];
    kernel_shared[(((int)threadIdx.x) + 7392)] = kernel[(((((((int)blockIdx.x) / 7) * 32768) + (((((int)threadIdx.x) + 7392) >> 7) * 512)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 96) & 127))];
    kernel_shared[(((int)threadIdx.x) + 7616)] = kernel[(((((((int)blockIdx.x) / 7) * 32768) + (((((int)threadIdx.x) + 7616) >> 7) * 512)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 64) & 127))];
    kernel_shared[(((int)threadIdx.x) + 7840)] = kernel[(((((((int)blockIdx.x) / 7) * 32768) + (((((int)threadIdx.x) + 7840) >> 7) * 512)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 32) & 127))];
    if (((int)threadIdx.x) < 128) {
      kernel_shared[(((int)threadIdx.x) + 8064)] = kernel[(((((((int)blockIdx.x) / 7) * 32768) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 32256)];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 16; ++rc_outer_inner) {
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((rc_outer_inner * 224) + (((int)threadIdx.x) % 14))] * kernel_shared[(((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8))]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[(((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8))]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[((rc_outer_inner * 224) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 4096)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 4096)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 1)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 42)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 1)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 4097)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 42)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 4097)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 56)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 2)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 70)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 2)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 56)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 4098)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 70)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 4098)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 84)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 3)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 98)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 3)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 84)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 4099)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 98)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 4099)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 112)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 4)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 126)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 4)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 112)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 4100)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 126)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 4100)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 140)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 5)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 154)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 5)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 140)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 4101)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 154)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 4101)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 168)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 6)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 182)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 6)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 168)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 4102)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 182)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 4102)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 196)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 7)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 210)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 7)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 196)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 4103)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 210)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 4103)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((rc_outer_inner * 224) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 128)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 128)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[((rc_outer_inner * 224) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 4224)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 4224)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 129)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 42)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 129)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 4225)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 42)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 4225)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 56)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 130)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 70)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 130)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 56)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 4226)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 70)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 4226)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 84)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 131)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 98)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 131)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 84)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 4227)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 98)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 4227)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 112)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 132)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 126)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 132)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 112)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 4228)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 126)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 4228)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 140)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 133)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 154)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 133)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 140)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 4229)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 154)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 4229)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 168)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 134)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 182)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 134)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 168)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 4230)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 182)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 4230)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 196)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 135)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 210)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 135)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 196)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 4231)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[(((rc_outer_inner * 224) + (((int)threadIdx.x) % 14)) + 210)] * kernel_shared[((((((int)threadIdx.x) / 14) * 256) + (rc_outer_inner * 8)) + 4231)]));
    }
  }
  for (int i1_inner = 0; i1_inner < 2; ++i1_inner) {
    compute[(((((((((int)blockIdx.x) / 7) * 12544) + ((((int)threadIdx.x) / 14) * 392)) + (i1_inner * 196)) + (((((int)threadIdx.x) % 14) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1))] = max((conv2d_nchw[i1_inner] + bias[((((((int)blockIdx.x) / 7) * 64) + ((((int)threadIdx.x) / 14) * 2)) + i1_inner)]), 0.000000e+00f);
    compute[((((((((((int)blockIdx.x) / 7) * 12544) + ((((int)threadIdx.x) / 14) * 392)) + (i1_inner * 196)) + (((((int)threadIdx.x) % 14) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1)) + 98)] = max((conv2d_nchw[(i1_inner + 2)] + bias[((((((int)blockIdx.x) / 7) * 64) + ((((int)threadIdx.x) / 14) * 2)) + i1_inner)]), 0.000000e+00f);
    compute[((((((((((int)blockIdx.x) / 7) * 12544) + ((((int)threadIdx.x) / 14) * 392)) + (i1_inner * 196)) + (((((int)threadIdx.x) % 14) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1)) + 6272)] = max((conv2d_nchw[(i1_inner + 4)] + bias[(((((((int)blockIdx.x) / 7) * 64) + ((((int)threadIdx.x) / 14) * 2)) + i1_inner) + 32)]), 0.000000e+00f);
    compute[((((((((((int)blockIdx.x) / 7) * 12544) + ((((int)threadIdx.x) / 14) * 392)) + (i1_inner * 196)) + (((((int)threadIdx.x) % 14) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1)) + 6370)] = max((conv2d_nchw[(i1_inner + 6)] + bias[(((((((int)blockIdx.x) / 7) * 64) + ((((int)threadIdx.x) / 14) * 2)) + i1_inner) + 32)]), 0.000000e+00f);
  }
}


